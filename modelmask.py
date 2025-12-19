# -*- coding: utf-8 -*-
"""
MASK-GUIDED Document Unwarping Model
======================================
Uses the border mask as an input channel to explicitly guide
the model about where the document is located.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MaskGuidedDocumentUnwarpModel(nn.Module):
    def __init__(self, pretrained=True, flow_scale=1.0):
        """
        Mask-guided unwarping model that takes RGB + Mask as input.
        
        Args:
            pretrained: Use pretrained ResNet50 encoder
            flow_scale: 1.0 (Full range displacement for steep warps)
        """
        super().__init__()
        self.flow_scale = flow_scale
        
        # ============================================================
        # INPUT PROJECTION: RGB (3) + Mask (1) -> 3 channels
        # ============================================================
        # This projects 4-channel input to 3 channels so we can use
        # pretrained ResNet50 weights
        self.input_projection = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # ============================================================
        # ENCODER: Pretrained ResNet50 (starting from layer1)
        # ============================================================
        resnet = timm.create_model('resnet50', pretrained=pretrained)
        
        # Skip the initial conv since we handle 4 channels ourselves
        self.encoder_pool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1  # 256 channels
        self.encoder_layer2 = resnet.layer2  # 512 channels
        self.encoder_layer3 = resnet.layer3  # 1024 channels
        self.encoder_layer4 = resnet.layer4  # 2048 channels
        
        # ============================================================
        # DECODER: U-Net Skip Connection Architecture
        # ============================================================
        self.decoder4 = self._make_decoder_block(2048, 1024)
        self.decoder3 = self._make_decoder_block(1024, 512)
        self.decoder2 = self._make_decoder_block(512, 256)
        self.decoder1 = self._make_decoder_block(256, 128)
        
        # Enhanced Final Upsample
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
        # MASK-AWARE FLOW HEAD
        # ============================================================
        # Concatenate mask again at the end to reinforce boundary awareness
        self.flow_head = nn.Sequential(
            nn.Conv2d(65, 64, 3, padding=1),  # 64 features + 1 mask
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        
        # Initialize to zero (identity warp initially)
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
    
    def forward(self, rgb, mask):
        """
        Forward pass with explicit mask input.
        
        Args:
            rgb: [B, 3, H, W] - Input warped document
            mask: [B, 1, H, W] - Border mask (1 = document, 0 = background)
        
        Returns:
            rectified: [B, 3, H, W] - Unwarped document
            flow: [B, 2, H, W] - Predicted flow field
            sampling_grid: [B, H, W, 2] - Sampling grid used
        """
        # Concatenate RGB and Mask
        x_with_mask = torch.cat([rgb, mask], dim=1)  # [B, 4, H, W]
        
        # Input projection (4 -> 64 channels)
        x = self.input_projection(x_with_mask)  # [B, 64, H/2, W/2]
        
        # Encoder
        x = self.encoder_pool(x)
        e1 = self.encoder_layer1(x)   # [B, 256, H/4, W/4]
        e2 = self.encoder_layer2(e1)  # [B, 512, H/8, W/8]
        e3 = self.encoder_layer3(e2)  # [B, 1024, H/16, W/16]
        e4 = self.encoder_layer4(e3)  # [B, 2048, H/32, W/32]
        
        # Decoder with Skip Connections
        d4 = self._match_size(self.decoder4(e4), e3) + e3
        d3 = self._match_size(self.decoder3(d4), e2) + e2
        d2 = self._match_size(self.decoder2(d3), e1) + e1
        d1 = self.decoder1(d2)
        
        # Final Processing
        features = self._match_size(self.final_upsample(d1), rgb)  # [B, 64, H, W]
        
        # Concatenate mask again for boundary-aware flow prediction
        mask_resized = F.interpolate(mask, size=features.shape[2:], 
                                     mode='bilinear', align_corners=True)
        features_with_mask = torch.cat([features, mask_resized], dim=1)  # [B, 65, H, W]
        
        # Predict flow
        flow = torch.tanh(self.flow_head(features_with_mask))  # [B, 2, H, W]
        
        # CRITICAL: Mask the flow field - zero flow outside document
        flow = flow * mask
        
        # Create sampling grid
        sampling_grid = self.create_sampling_grid(flow)
        
        # Apply warping with bicubic for text clarity
        rectified = F.grid_sample(
            rgb,
            sampling_grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=True
        )
        
        return rectified, flow, sampling_grid
    
    def _match_size(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], 
                            mode='bilinear', align_corners=True)
        return x


class BackwardCompatibleWrapper(nn.Module):
    """
    Wrapper to make the mask-guided model compatible with existing code
    that calls model(rgb) without mask.
    """
    def __init__(self, flow_scale=1.0):
        super().__init__()
        self.model = MaskGuidedDocumentUnwarpModel(flow_scale=flow_scale)
        
    def forward(self, rgb, mask=None):
        """
        Args:
            rgb: [B, 3, H, W] - Input image
            mask: [B, 1, H, W] - Border mask (optional for compatibility)
        """
        if mask is None:
            # Fallback: create a dummy mask (all ones) if none provided
            mask = torch.ones(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3], 
                            device=rgb.device, dtype=rgb.dtype)
        
        return self.model(rgb, mask)


# For backward compatibility, export the original class name
DocumentUnwarpModel = BackwardCompatibleWrapper
