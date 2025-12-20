import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from dataclasses import dataclass

@dataclass
class ModelConfig:
    backbone: str = "convnext_tiny"
    pretrained: bool = True
    uv_mode: str = "residual"  
    max_disp: float = 0.35     
    norm: str = "gn"           # GroupNorm is more stable for small batches
    two_stage_warp: bool = True
    coarse_stride: int = 64
    pred_downscale: int = 2    # Predict geometry at 1/2 res for A100 speed

def _norm(num_channels: int, kind: str = "gn") -> nn.Module:
    if kind == "bn": return nn.BatchNorm2d(num_channels)
    g = min(32, num_channels)
    return nn.GroupNorm(g, num_channels)

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "gn"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n1 = _norm(out_ch, norm)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n2 = _norm(out_ch, norm)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.n1(self.conv1(x)))
        x = self.act(self.n2(self.conv2(x)))
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "gn"):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, norm=norm)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class UnwarpNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        feat_ch = self.encoder.feature_info.channels()

        self.dec4 = ConvBlock(feat_ch[3], 512, norm=cfg.norm)
        self.up3 = UpBlock(512, feat_ch[2], 256, norm=cfg.norm)
        self.up2 = UpBlock(256, feat_ch[1], 128, norm=cfg.norm)
        self.up1 = UpBlock(128, feat_ch[0], 64, norm=cfg.norm)

        # Fine Head
        self.out_fine = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), _norm(64, cfg.norm), nn.SiLU(), nn.Conv2d(64, 2, 1))
        
        # Coarse Head (Stabilizer)
        if cfg.two_stage_warp:
            self.out_coarse = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), _norm(64, cfg.norm), nn.SiLU(), nn.Conv2d(64, 2, 1))
            nn.init.zeros_(self.out_coarse[-1].weight)
            nn.init.zeros_(self.out_coarse[-1].bias)

        nn.init.zeros_(self.out_fine[-1].weight)
        nn.init.zeros_(self.out_fine[-1].bias)

    def forward(self, x_full):
        b, c, h_full, w_full = x_full.shape
        pd = self.cfg.pred_downscale
        
        # Geometry estimation at lower res for A100 speed
        h_pred, w_pred = h_full // pd, w_full // pd
        x_pred = F.interpolate(x_full, size=(h_pred, w_pred), mode="bilinear", align_corners=False)

        # Encoder-Decoder
        f1, f2, f3, f4 = self.encoder(x_pred)
        y = self.up1(self.up2(self.up3(self.dec4(f4), f3), f2), f1)
        y = F.interpolate(y, size=(h_pred, w_pred), mode="bilinear", align_corners=False)

        # Hierarchical Warp Prediction
        # Stage 1: Coarse (The "Sketch")
        ch, cw = max(4, h_pred//self.cfg.coarse_stride), max(4, w_pred//self.cfg.coarse_stride)
        y_c = F.adaptive_avg_pool2d(y, (ch, cw))
        uv_c = self.out_coarse(y_c)
        
        # Create identity and add coarse delta
        ys = torch.linspace(0, 1, ch, device=x_full.device)
        xs = torch.linspace(0, 1, cw, device=x_full.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        id_uv_c = torch.stack([xx, yy], dim=0).unsqueeze(0)
        uv01_c = (id_uv_c + torch.tanh(uv_c) * (self.cfg.max_disp * 1.5)).clamp(0,1)
        
        # Stage 2: Fine (The "Detail")
        uv01_c_up = F.interpolate(uv01_c, size=(h_pred, w_pred), mode="bilinear", align_corners=False)
        uv_f = self.out_fine(y)
        uv01_pred = (uv01_c_up + torch.tanh(uv_f) * (self.cfg.max_disp * 0.5)).clamp(0, 1)

        # Upsample UV to full-res and sample
        uv01_full = F.interpolate(uv01_pred, size=(h_full, w_full), mode="bilinear", align_corners=False)
        grid = uv01_full.permute(0, 2, 3, 1) * 2.0 - 1.0 # Convert [0,1] to [-1,1]

        rectified = F.grid_sample(x_full, grid, mode='bicubic', padding_mode='border', align_corners=True)
        
        return {"rectified": rectified, "uv01": uv01_full, "grid": grid}
