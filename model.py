"""
model.py

Deep Document Unwarping — geometric reconstruction model.

Key property:
- The network predicts a dense backward sampling grid (UV / flow), NOT the flat image.
- The rectified image is produced via torch.nn.functional.grid_sample(input, grid).

Non-square + full-res friendly:
- Works with arbitrary HxW (e.g., 1224x1584).
- Can predict the warp at LOWER resolution (pred_downscale) and upsample UV to full-res,
  then warp the full-res image. This is the main performance trick.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError("This project requires timm: pip install timm") from e


# -------------------------
# Blocks
# -------------------------

def _norm(num_channels: int, kind: str = "gn") -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(num_channels)
    if kind == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    g = 32
    g = min(g, num_channels)
    while num_channels % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "gn"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n1 = _norm(out_ch, norm)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n2 = _norm(out_ch, norm)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.n1(self.conv1(x)))
        x = self.act(self.n2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "gn"):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


# -------------------------
# Grid helpers
# -------------------------

def make_identity_uv01(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]
    return uv


def uv01_to_grid(uv01: torch.Tensor) -> torch.Tensor:
    u = uv01[:, 0]
    v = uv01[:, 1]
    x = u * 2.0 - 1.0
    y = v * 2.0 - 1.0
    return torch.stack([x, y], dim=-1)  # [B,H,W,2]


def _safe_pred_size(h: int, w: int, down: int) -> Tuple[int, int]:
    """
    Choose a smaller prediction size for warp estimation.
    Keep it at least 64 px in each dimension to avoid tiny feature maps.
    """
    if down <= 1:
        return h, w
    hp = max(64, int((h + down - 1) // down))
    wp = max(64, int((w + down - 1) // down))
    return hp, wp


# -------------------------
# Model
# -------------------------

@dataclass
class ModelConfig:
    backbone: str = "convnext_tiny"
    pretrained: bool = True
    normalize_backbone: bool = True  # ImageNet mean/std for timm backbones

    norm: str = "gn"

    uv_mode: str = "residual"        # "residual" or "absolute"
    max_disp: float = 0.35           # residual displacement in UV01 units

    align_corners: bool = True
    padding_mode: str = "border"     # grid_sample padding
    warp_mode: str = "bilinear"      # "bilinear" or "bicubic"

    freeze_backbone: bool = False

    # Performance knobs
    pred_downscale: int = 1          # predict UV on downscaled input, then upsample to full-res
    use_checkpoint: bool = False     # activation checkpointing (recompute in backward)

    # Two-stage warp head (coarse base + fine residual)
    two_stage_warp: bool = True
    coarse_stride: int = 64          # roughly one control point per this many pixels at predicted scale
    coarse_disp_scale: float = 1.5   # coarse max_disp multiplier (residual mode)
    fine_disp_scale: float = 0.5     # fine residual max_disp multiplier (residual mode)


class UnwarpNet(nn.Module):
    """
    Predicts a dense backward UV grid (uv01) and rectifies via grid_sample.

    Returns dict:
      - uv01: [B,2,H_full,W_full] in [0,1]
      - grid: [B,H_full,W_full,2] in [-1,1]
      - rectified: [B,3,H_full,W_full]
      - uv01_pred: [B,2,H_pred,W_pred] (optional, for diagnostics)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Backbone normalization (timm ImageNet defaults). Input expected in [0,1].
        self.register_buffer('bb_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('bb_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        feat_ch = self.encoder.feature_info.channels()

        # Decoder width is intentionally moderate. Warps are smooth; huge channels aren't needed.
        self.dec4 = ConvBlock(feat_ch[3], 512, norm=cfg.norm)
        self.up3 = UpBlock(512, feat_ch[2], 256, norm=cfg.norm)
        self.up2 = UpBlock(256, feat_ch[1], 128, norm=cfg.norm)
        self.up1 = UpBlock(128, feat_ch[0], 64, norm=cfg.norm)

        self.out_fine = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            _norm(64, cfg.norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

        # Coarse head: applied on a pooled feature map to predict a low-frequency base warp
        self.out_coarse = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            _norm(64, cfg.norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

        # Start near identity (critical for stability at high-res)
        nn.init.zeros_(self.out_fine[-1].weight)
        nn.init.zeros_(self.out_fine[-1].bias)
        nn.init.zeros_(self.out_coarse[-1].weight)
        nn.init.zeros_(self.out_coarse[-1].bias)

        if cfg.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _encode(self, x: torch.Tensor):
        if not self.cfg.use_checkpoint:
            return tuple(self.encoder(x))
        from torch.utils.checkpoint import checkpoint

        def _f(inp):
            return tuple(self.encoder(inp))
        return checkpoint(_f, x, use_reentrant=False)

    def _maybe_ckpt(self, fn, *args):
        if not self.cfg.use_checkpoint:
            return fn(*args)
        from torch.utils.checkpoint import checkpoint
        return checkpoint(fn, *args, use_reentrant=False)

    def forward(
        self,
        x: torch.Tensor,
        pred_downscale: Optional[int] = None,
        x_full: Optional[torch.Tensor] = None,
        return_uv_pred: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        x: tensor used to estimate the warp (if x_full is None, also the one being warped).
        x_full: optional full-res tensor to warp. If provided, warp is estimated from x (or downscaled x_full),
                but rectification is applied to x_full.
        pred_downscale: overrides cfg.pred_downscale for this forward call.
        """
        if x_full is None:
            x_full = x

        b, c, h_full, w_full = x_full.shape
        pd = int(pred_downscale) if pred_downscale is not None else int(self.cfg.pred_downscale)
        pd = max(1, pd)

        # Warp estimation input (downscaled for speed)
        h_pred, w_pred = _safe_pred_size(h_full, w_full, pd)
        if (h_pred, w_pred) != (h_full, w_full):
            x_pred = F.interpolate(x_full, size=(h_pred, w_pred), mode="bilinear", align_corners=False)
        else:
            x_pred = x_full

        x_enc = x_pred

        if self.cfg.normalize_backbone and (x_enc.ndim == 4 and x_enc.shape[1] == 3):
            mean = self.bb_mean.to(dtype=x_enc.dtype, device=x_enc.device)
            std = self.bb_std.to(dtype=x_enc.dtype, device=x_enc.device)
            x_enc = (x_enc - mean) / std

        f1, f2, f3, f4 = self._encode(x_enc)
        y4 = self._maybe_ckpt(self.dec4, f4)
        y3 = self._maybe_ckpt(lambda a, s: self.up3(a, s), y4, f3)
        y2 = self._maybe_ckpt(lambda a, s: self.up2(a, s), y3, f2)
        y1 = self._maybe_ckpt(lambda a, s: self.up1(a, s), y2, f1)

        y = F.interpolate(y1, size=(h_pred, w_pred), mode="bilinear", align_corners=False)

        uv01_coarse = None
        uv01_coarse_up = None
        uv01_fine_delta = None

        if self.cfg.two_stage_warp:
            # Coarse control grid resolution (roughly one point per coarse_stride pixels)
            coarse_h = max(4, int(round(float(h_pred) / float(self.cfg.coarse_stride))))
            coarse_w = max(4, int(round(float(w_pred) / float(self.cfg.coarse_stride))))
            y_coarse = F.adaptive_avg_pool2d(y, (coarse_h, coarse_w))
            uv_raw_c = self.out_coarse(y_coarse)

            if self.cfg.uv_mode == "absolute":
                uv01_coarse = torch.sigmoid(uv_raw_c)
            else:
                id_uv_c = make_identity_uv01(coarse_h, coarse_w, device=x_full.device, dtype=uv_raw_c.dtype)
                delta_c = torch.tanh(uv_raw_c) * (float(self.cfg.max_disp) * float(self.cfg.coarse_disp_scale))
                uv01_coarse = (id_uv_c + delta_c).clamp(0.0, 1.0)

            uv01_coarse_up = F.interpolate(uv01_coarse, size=(h_pred, w_pred), mode="bilinear", align_corners=False)
            uv01_coarse_up = uv01_coarse_up.clamp(0.0, 1.0)

            uv_raw_f = self.out_fine(y)
            uv01_fine_delta = torch.tanh(uv_raw_f) * (float(self.cfg.max_disp) * float(self.cfg.fine_disp_scale))
            uv01_pred = (uv01_coarse_up + uv01_fine_delta).clamp(0.0, 1.0)
        else:
            uv_raw = self.out_fine(y)
            if self.cfg.uv_mode == "absolute":
                uv01_pred = torch.sigmoid(uv_raw)
            else:
                id_uv = make_identity_uv01(h_pred, w_pred, device=x_full.device, dtype=uv_raw.dtype)
                delta = torch.tanh(uv_raw) * float(self.cfg.max_disp)
                uv01_pred = (id_uv + delta).clamp(0.0, 1.0)

        # Upsample UV to full-res (uv01 is normalized, so upsampling is "scale-free")
        if (h_pred, w_pred) != (h_full, w_full):
            uv01 = F.interpolate(uv01_pred, size=(h_full, w_full), mode="bilinear", align_corners=False)
            uv01 = uv01.clamp(0.0, 1.0)
        else:
            uv01 = uv01_pred

        grid = uv01_to_grid(uv01)

        rect = F.grid_sample(
            x_full,
            grid,
            mode=self.cfg.warp_mode,
            padding_mode=self.cfg.padding_mode,
            align_corners=self.cfg.align_corners,
        )

        out = {"uv01": uv01, "grid": grid, "rectified": rect}
        if return_uv_pred:
            out["uv01_pred"] = uv01_pred
            if self.cfg.two_stage_warp and uv01_coarse is not None:
                out["uv01_coarse_pred"] = uv01_coarse
            if self.cfg.two_stage_warp and uv01_coarse_up is not None:
                out["uv01_coarse_up_pred"] = uv01_coarse_up
            if self.cfg.two_stage_warp and uv01_fine_delta is not None:
                out["uv01_fine_delta_pred"] = uv01_fine_delta
        return out


def build_model(
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    uv_mode: str = "residual",
    max_disp: float = 0.35,
    norm: str = "gn",
    align_corners: bool = True,
    padding_mode: str = "border",
    warp_mode: str = "bilinear",
    freeze_backbone: bool = False,
    pred_downscale: int = 1,
    use_checkpoint: bool = False,
    # two-stage warp head
    two_stage_warp: bool = True,
    coarse_stride: int = 64,
    coarse_disp_scale: float = 1.5,
    fine_disp_scale: float = 0.5,
) -> UnwarpNet:
    cfg = ModelConfig(
        backbone=backbone,
        pretrained=pretrained,
        uv_mode=uv_mode,
        max_disp=max_disp,
        norm=norm,
        align_corners=align_corners,
        padding_mode=padding_mode,
        warp_mode=warp_mode,
        freeze_backbone=freeze_backbone,
        pred_downscale=pred_downscale,
        use_checkpoint=use_checkpoint,
        two_stage_warp=two_stage_warp,
        coarse_stride=coarse_stride,
        coarse_disp_scale=coarse_disp_scale,
        fine_disp_scale=fine_disp_scale,
    )
    return UnwarpNet(cfg)


# -------------------------
# CLI sanity
# -------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UnwarpNet sanity forward.")
    p.add_argument("--backbone", type=str, default="convnext_tiny")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--uv-mode", type=str, default="residual", choices=["residual", "absolute"])
    p.add_argument("--max-disp", type=float, default=0.35)
    p.add_argument("--warp-mode", type=str, default="bilinear", choices=["bilinear", "bicubic"])
    p.add_argument("--pred-downscale", type=int, default=2, help="Predict warp at 1/N resolution then upsample to full-res.")
    p.add_argument("--checkpoint", action="store_true", help="Enable activation checkpointing.")
    p.add_argument("--img-size", type=int, default=512, help="Square sanity input size (if img-h/img-w not set).")
    p.add_argument("--img-h", type=int, default=None)
    p.add_argument("--img-w", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--channels-last", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    h = args.img_h if args.img_h is not None else args.img_size
    w = args.img_w if args.img_w is not None else args.img_size

    model = build_model(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        uv_mode=args.uv_mode,
        max_disp=args.max_disp,
        warp_mode=args.warp_mode,
        pred_downscale=args.pred_downscale,
        use_checkpoint=args.checkpoint,
    ).to(device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    model.eval()
    with torch.no_grad():
        x = torch.rand(1, 3, h, w, device=device)
        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)
        out = model(x, return_uv_pred=True)
        uv_pred = out.get("uv01_pred", None)
        print("rectified:", tuple(out["rectified"].shape), out["rectified"].dtype)
        print("uv01:", tuple(out["uv01"].shape), float(out["uv01"].min()), float(out["uv01"].max()))
        if uv_pred is not None:
            print("uv01_pred:", tuple(uv_pred.shape))
        print("grid:", tuple(out["grid"].shape), float(out["grid"].min()), float(out["grid"].max()))


if __name__ == "__main__":
    main()
