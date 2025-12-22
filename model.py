"""
model.py - Document Unwarping Model with Attention, Bias, and Refinement
Predicts UV grid for geometric document rectification
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# NORMALIZATION

def make_norm(channels: int, kind: str = "gn") -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    g = min(32, channels)
    while channels % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, channels)


# BUILDING BLOCKS
class FeatureConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "gn"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            make_norm(out_ch, norm_type),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            make_norm(out_ch, norm_type),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm_type: str = "gn"):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.conv_block = FeatureConv(out_ch + skip_ch, out_ch, norm_type)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


# UV grid helpers
def uv_identity(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create identity UV grid normalized to [0,1]."""
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]

def uv_to_grid(uv: torch.Tensor) -> torch.Tensor:
    """Convert UV [0,1] to PyTorch sampling grid [-1,1]."""
    u, v = uv[:, 0], uv[:, 1]
    return torch.stack([u * 2 - 1, v * 2 - 1], dim=-1)  # [B,H,W,2]

def pred_size_safe(h: int, w: int, down: int) -> Tuple[int, int]:
    if down <= 1:
        return h, w
    return max(64, (h + down - 1) // down), max(64, (w + down - 1) // down)


# MODEL CONFIG
@dataclass
class DocumentUnwarpConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    normalize_backbone: bool = True
    norm: str = "gn"
    uv_mode: str = "residual"
    max_disp: float = 0.35
    align_corners: bool = True
    padding_mode: str = "border"
    warp_mode: str = "bilinear" #bicubic
    freeze_backbone: bool = False
    pred_downscale: int = 1
    two_stage_warp: bool = True
    coarse_stride: int = 64
    coarse_disp_scale: float = 1.5
    fine_disp_scale: float = 0.5
    use_attention: bool = True
    use_uv_bias: bool = True
    use_post_refine: bool = False


# MAIN MODEL
class DocumentUnwarpNet(nn.Module):
    """
    Document unwarping with attention, bias, and refinement improvements
    """

    def __init__(self, cfg: DocumentUnwarpConfig):
        super().__init__()
        self.cfg = cfg

        # Backbone normalization
        self.register_buffer("bb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("bb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        feat_ch = self.encoder.feature_info.channels()

        # Decoder
        self.dec4 = FeatureConv(feat_ch[3], 512, cfg.norm)
        self.up_blocks = nn.ModuleList([
            DecoderBlock(512, feat_ch[2], 256, cfg.norm),
            DecoderBlock(256, feat_ch[1], 128, cfg.norm),
            DecoderBlock(128, feat_ch[0], 64, cfg.norm)
        ])

        # UV prediction heads
        self.head_fine = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            make_norm(64, cfg.norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

        self.head_coarse = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            make_norm(64, cfg.norm),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

        # UV ATTENTION # TSTING STILL
        if cfg.use_attention:
            self.uv_attention = nn.Sequential(
                nn.Conv2d(64, 32, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 2, 1),
                nn.Sigmoid()
            )
        
        # LEARNABLE UV BIAS # TSTING STILL
        if cfg.use_uv_bias:
            self.uv_bias = nn.Parameter(torch.zeros(1, 2, 1, 1))
        
        # POST WARP REFINEMENT # TSTING STILL
        if cfg.use_post_refine:
            self.post_refine = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 3, 3, padding=1)
            )
            nn.init.zeros_(self.post_refine[-1].weight)
            nn.init.zeros_(self.post_refine[-1].bias)
        # Init
        for m in [self.head_fine[-1], self.head_coarse[-1]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        if cfg.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode(self, x: torch.Tensor):
        """Extract features"""
        return tuple(self.encoder(x))

    def forward(
        self,
        x: torch.Tensor,
        pred_downscale: Optional[int] = None,
        input_image: Optional[torch.Tensor] = None,
        return_uv_pred: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if input_image is None:
            input_image = x

        b, c, h_full, w_full = input_image.shape
        pd = max(1, pred_downscale or self.cfg.pred_downscale)

        # Compute prediction resolution
        h_pred, w_pred = pred_size_safe(h_full, w_full, pd)
        warp_input = input_image if (h_pred, w_pred) == (h_full, w_full) else F.interpolate(
            input_image, (h_pred, w_pred), mode="bilinear", align_corners=False
        )

        # Apply normalization for pretrained encoder
        if self.cfg.normalize_backbone and warp_input.shape[1] == 3:
            warp_input = (warp_input - self.bb_mean.to(dtype=warp_input.dtype, device=warp_input.device)) / \
                         self.bb_std.to(dtype=warp_input.dtype, device=warp_input.device)

        # Encode
        feats = self.encode(warp_input)

        # Decode
        y = self.dec4(feats[3])
        for i, up in enumerate(self.up_blocks):
            y = up(y, feats[2 - i])

        # Final feature map
        y = F.interpolate(y, (h_pred, w_pred), mode="bilinear", align_corners=False)

        # UV prediction
        uv_coarse = uv_coarse_up = uv_fine_delta = None

        if self.cfg.two_stage_warp:
            # Coarse scale
            coarse_h = max(4, round(h_pred / self.cfg.coarse_stride))
            coarse_w = max(4, round(w_pred / self.cfg.coarse_stride))
            y_coarse = F.adaptive_avg_pool2d(y, (coarse_h, coarse_w))
            uv_raw_c = self.head_coarse(y_coarse)

            # APPLY ATTENTION TO COARSE UV
            if hasattr(self, 'uv_attention'):
                attention_weights = self.uv_attention(y_coarse)
                uv_raw_c = uv_raw_c * attention_weights

            if self.cfg.uv_mode == "absolute":
                uv_coarse = torch.sigmoid(uv_raw_c)
            else:
                delta_c = torch.tanh(uv_raw_c) * self.cfg.max_disp * self.cfg.coarse_disp_scale
                
                # APPLY UV BIAS
                if hasattr(self, 'uv_bias'):
                    delta_c = delta_c + self.uv_bias
                
                uv_coarse = (
                    uv_identity(coarse_h, coarse_w, device=x.device, dtype=uv_raw_c.dtype) + delta_c
                ).clamp(0, 1)

            uv_coarse_up = F.interpolate(
                uv_coarse, (h_pred, w_pred), mode="bilinear", align_corners=False
            ).clamp(0, 1)

            # Fine scale
            uv_fine_delta = torch.tanh(self.head_fine(y)) * self.cfg.max_disp * self.cfg.fine_disp_scale
            uv_pred = (uv_coarse_up + uv_fine_delta).clamp(0, 1)
        else:
            # Single stage
            uv_raw = self.head_fine(y)
            if self.cfg.uv_mode == "absolute":
                uv_pred = torch.sigmoid(uv_raw)
            else:
                delta = torch.tanh(uv_raw) * self.cfg.max_disp
                
                # BIAS
                if hasattr(self, 'uv_bias'):
                    delta = delta + self.uv_bias
                
                uv_pred = (
                    uv_identity(h_pred, w_pred, device=x.device, dtype=uv_raw.dtype) + delta
                ).clamp(0, 1)

        # Upsample UV to full res
        uv_coords = F.interpolate(uv_pred, (h_full, w_full), mode="bilinear", align_corners=False) \
            if (h_pred, w_pred) != (h_full, w_full) else uv_pred
        uv_coords = uv_coords.clamp(0, 1)

        # Apply warping to full image
        grid = uv_to_grid(uv_coords)
        rectified = F.grid_sample(
            input_image,
            grid,
            mode=self.cfg.warp_mode,
            padding_mode=self.cfg.padding_mode,
            align_corners=self.cfg.align_corners,
        )

        # post warp
        if hasattr(self, 'post_refine'):
            refinement = self.post_refine(rectified)
            rectified = (rectified + refinement).clamp(0.0, 1.0)

        # Build output
        out = {"uv": uv_coords, "grid": grid, "rectified": rectified}

        if return_uv_pred:
            out["uv_pred"] = uv_pred
            if self.cfg.two_stage_warp:
                out.update({
                    "uv_coarse_pred": uv_coarse,
                    "uv_coarse_up_pred": uv_coarse_up,
                    "uv_fine_delta_pred": uv_fine_delta
                })

        return out


# MODEL BUILDER
def make_documentunwarp(
    backbone: str = "resnet50",
    pretrained: bool = True,
    uv_mode: str = "residual",
    max_disp: float = 0.35,
    norm: str = "gn",
    align_corners: bool = True,
    padding_mode: str = "border",
    warp_mode: str = "bilinear",
    freeze_backbone: bool = False,
    pred_downscale: int = 1,
    two_stage_warp: bool = True,
    coarse_stride: int = 64,
    coarse_disp_scale: float = 1.5,
    fine_disp_scale: float = 0.5,
    # new to test
    use_attention: bool = True,
    use_uv_bias: bool = True,
    use_post_refine: bool = False,
) -> DocumentUnwarpNet:
    """
    Build DocumentUnwarpNet with attention, bias, and refinement improvements
    """
    cfg = DocumentUnwarpConfig(
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
        two_stage_warp=two_stage_warp,
        coarse_stride=coarse_stride,
        coarse_disp_scale=coarse_disp_scale,
        fine_disp_scale=fine_disp_scale,
        use_attention=use_attention,
        use_uv_bias=use_uv_bias,
        use_post_refine=use_post_refine,
    )
    return DocumentUnwarpNet(cfg)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DocumentUnwarpNet with improvements")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model = make_documentunwarp(backbone=args.backbone).to(device)
    model.eval()
    
    with torch.no_grad():
        x = torch.rand(1, 3, args.img_size, args.img_size, device=device)
        out = model(x, return_uv_pred=True)
        
        print("Forward pass successful")
        print(f"  Rectified: {tuple(out['rectified'].shape)}")
        print(f"  UV: {tuple(out['uv'].shape)} [{out['uv'].min():.3f}, {out['uv'].max():.3f}]")
        print(f"  Grid: {tuple(out['grid'].shape)}")
        
        # Count new feature parameters
        attn_params = sum(p.numel() for p in model.uv_attention.parameters()) if hasattr(model, 'uv_attention') else 0
        bias_params = model.uv_bias.numel() if hasattr(model, 'uv_bias') else 0
        refine_params = sum(p.numel() for p in model.post_refine.parameters()) if hasattr(model, 'post_refine') else 0
        
        print(f"\nNew features added:")
        print(f"  Attention: {attn_params:,} params")
        print(f"  UV bias: {bias_params:,} params")
        print(f"  Refinement: {refine_params:,} params")
        print(f"  Total new: {attn_params + bias_params + refine_params:,} params")