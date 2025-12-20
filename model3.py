# -*- coding: utf-8 -*-
"""
HIGH-CLARITY Document Unwarping Model
=====================================
Optimized for A100 training: Uses Bicubic sampling and 
Enhanced Flow Heads to preserve text legibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DocumentUnwarpModel(nn.Module):
    def __init__(self, pretrained=True, flow_scale=1.0):
        """
        Args:
            pretrained: Use pretrained ResNet50 encoder
            flow_scale: 1.0 (Full range displacement for steep warps)
        """
        super().__init__()
        self.flow_scale = flow_scale
        
        # ============================================================
        # ENCODER: Pretrained ResNet50
        # ============================================================
        self.encoder = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        
        # ============================================================
        # DECODER: U-Net Skip Connection Architecture
        # ============================================================
        self.decoder4 = self._make_decoder_block(2048, 1024)
        self.decoder3 = self._make_decoder_block(1024, 512)
        self.decoder2 = self._make_decoder_block(512, 256)
        self.decoder1 = self._make_decoder_block(256, 128)
        
        # Enhanced Final Upsample (More channels = more detail)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # ============================================================
        # ENHANCED FLOW HEAD (Sharper displacement prediction)
        # ============================================================
        self.flow_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        
        # Initialize to zero so the model starts with a "Flat" identity grid
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def create_sampling_grid(self, flow):
        B, _, H, W = flow.shape
        y = torch.linspace(-1, 1, H, device=flow.device)
        x = torch.linspace(-1, 1, W, device=flow.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        identity_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        flow_permuted = flow.permute(0, 2, 3, 1)
        return identity_grid + flow_permuted * self.flow_scale
    
    def forward(self, x):
        # Encoder
        e1, e2, e3, e4 = self.encoder(x)
        
        # Decoder with Residual Connections
        d4 = self._match_size(self.decoder4(e4), e3) + e3
        d3 = self._match_size(self.decoder3(d4), e2) + e2
        d2 = self._match_size(self.decoder2(d3), e1) + e1
        d1 = self.decoder1(d2)
        
        # Final Processing
        features = self._match_size(self.final_upsample(d1), x)
        flow = torch.tanh(self.flow_head(features))
        
        sampling_grid = self.create_sampling_grid(flow)
        
        # CRITICAL FIX: Bicubic mode for text clarity
        rectified = F.grid_sample(
            x,
            sampling_grid,
            mode='bicubic', # <--- Sharpness upgrade
            padding_mode='border',
            align_corners=True
        )
        
        return rectified, flow, sampling_grid

    def _match_size(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
        return x
