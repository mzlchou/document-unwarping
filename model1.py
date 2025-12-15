"""
Document Unwarping Model using Flow-based U-Net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from dataset_loader import create_base_grid


class UnwarpUNet(nn.Module):
    """
    U-Net architecture with ResNet50 encoder for predicting flow fields for unwarping.
    
    Architecture:
        1. Encoder: Pretrained ResNet50 extracts hierarchical features
        2. Decoder: U-Net upsampling with skip connections
        3. Flow Head: Predicts 2D displacement field
        4. Warping: Uses grid_sample to dewarp the input
    """
    
    def __init__(self, flow_scale=1.0):
        super().__init__() #setup nn.module
        self.flow_scale = flow_scale  # Scale factor for flow magnitude
        
        # -----------------------
        # 1. Pretrained Encoder (ResNet50)
        # -----------------------
        self.encoder = timm.create_model(
            "resnet50",
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)  # Get features at different scales
        )
        # ResNet50 feature channels: [256, 512, 1024, 2048]
        encoder_channels = [256, 512, 1024, 2048]
        
        # -----------------------
        # 2. Decoder (U-Net style with skip connections)
        # -----------------------
        # Each decoder block: upsample + conv + combine with skip
        self.up4 = self._up_block(encoder_channels[3], encoder_channels[2])  # 2048 -> 1024
        self.up3 = self._up_block(encoder_channels[2], encoder_channels[1])  # 1024 -> 512
        self.up2 = self._up_block(encoder_channels[1], encoder_channels[0])  # 512 -> 256
        self.up1 = self._up_block(encoder_channels[0], 128)                   # 256 -> 128
        
        # Final upsampling to match input resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # -----------------------
        # 3. Flow Prediction Head
        # -----------------------
        # Predicts 2 channels: (flow_x, flow_y) displacement for each pixel
        self.flow_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        
        # Initialize flow head with small weights (start with identity transform)
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)
    
    def _up_block(self, in_ch, out_ch):
        """
        Upsampling block for decoder.
        
        Args:
            in_ch: Input channels
            out_ch: Output channels
        
        Returns:
            Sequential module performing: upsample -> conv -> relu -> conv -> relu
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _match_shapes(self, x, target):
        """
        Resize x to match target's spatial dimensions.
        Handles potential shape mismatches from skip connections.
        """
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(
                x, 
                size=target.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        return x
    
    def forward(self, x):
        """
        Forward pass: predict flow field and warp input image.
        
        Args:
            x: Input warped document image [B, 3, H, W]
        
        Returns:
            rectified: Dewarped image [B, 3, H, W]
            flow: Predicted flow field [B, 2, H, W]
            sampling_grid: Grid used for warping [B, H, W, 2]
        """
        B, C, H, W = x.shape
        
        # -----------------------
        # STEP 1: Encoder - Extract hierarchical features
        # -----------------------
        feats = self.encoder(x)
        f1, f2, f3, f4 = feats  # Increasing depth: 256, 512, 1024, 2048 channels
        
        # -----------------------
        # STEP 2: Decoder - Upsample with skip connections
        # -----------------------
        # Level 4: Upsample deepest features
        d4 = self.up4(f4)
        d4 = self._match_shapes(d4, f3)
        d4 = d4 + f3  # Skip connection
        
        # Level 3
        d3 = self.up3(d4)
        d3 = self._match_shapes(d3, f2)
        d3 = d3 + f2  # Skip connection
        
        # Level 2
        d2 = self.up2(d3)
        d2 = self._match_shapes(d2, f1)
        d2 = d2 + f1  # Skip connection
        
        # Level 1
        d1 = self.up1(d2)
        
        # Final upsampling to input resolution
        d0 = self.final_up(d1)
        d0 = self._match_shapes(d0, x)  # Ensure exact size match
        
        # -----------------------
        # STEP 3: Predict Flow Field
        # -----------------------
        # Output: [B, 2, H, W] representing (Δx, Δy) for each pixel
        flow = self.flow_head(d0)
        
        # Scale flow to appropriate range
        # tanh bounds to [-1, 1], then scale by flow_scale
        flow = torch.tanh(flow) * self.flow_scale
        
        # -----------------------
        # STEP 4: Create Sampling Grid
        # -----------------------
        # Base grid: regular mesh in [-1, 1] coordinate space
        base_grid = create_base_grid(B, H, W, x.device)  # [B, H, W, 2]
        
        # Add predicted flow to base grid
        # Need to permute flow from [B, 2, H, W] to [B, H, W, 2]
        sampling_grid = base_grid + flow.permute(0, 2, 3, 1)
        
        # -----------------------
        # STEP 5: Warp Input Image
        # -----------------------
        # grid_sample: For each output pixel, look up corresponding input pixel
        rectified = F.grid_sample(
            x,
            sampling_grid,
            mode="bilinear",          # Smooth interpolation
            padding_mode="border",    # Clamp out-of-bounds coordinates
            align_corners=True
        )
        
        return rectified, flow, sampling_grid


class UnwarpUNetWithRefinement(nn.Module):
    """
    Enhanced version with coarse-to-fine refinement.
    
    This model first warps the image using predicted flow,
    then applies a refinement network to fix remaining artifacts.
    """
    
    def __init__(self, flow_scale=1.0):
        super().__init__()
        
        # Base flow prediction network
        self.flow_net = UnwarpUNet(flow_scale=flow_scale)
        
        # Refinement network: takes warped image and predicts residuals
        self.refinement = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()  # Bound residuals to [-1, 1]
        )
    
    def forward(self, x):
        """
        Two-stage forward: coarse warping + fine refinement.
        """
        # Stage 1: Geometric warping
        coarse, flow, grid = self.flow_net(x)
        
        # Stage 2: Appearance refinement
        residual = self.refinement(coarse) * 0.1  # Small residuals
        refined = torch.clamp(coarse + residual, 0, 1)
        
        return refined, flow, grid, coarse


# -----------------------
# Helper function for visualization
# -----------------------
def flow_to_color(flow):
    """
    Convert flow field to RGB color visualization.
    
    Args:
        flow: [2, H, W] or [B, 2, H, W] tensor
    
    Returns:
        RGB image where hue = direction, saturation = magnitude
    """
    import numpy as np
    
    if flow.dim() == 4:
        flow = flow[0]  # Take first batch
    
    flow = flow.detach().cpu().numpy()
    u, v = flow[0], flow[1]
    
    # Compute magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    
    # Convert to HSV
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 180  # Hue
    hsv[..., 1] = np.clip(mag * 255 / mag.max(), 0, 255)  # Saturation
    hsv[..., 2] = 255  # Value
    
    import cv2
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


# -----------------------
# Model factory function
# -----------------------
def get_model(model_type='base', flow_scale=1.0):
    """
    Factory function to create models.
    
    Args:
        model_type: 'base' or 'refinement'
        flow_scale: Scaling factor for flow magnitude
    
    Returns:
        Model instance
    """
    if model_type == 'base':
        return UnwarpUNet(flow_scale=flow_scale)
    elif model_type == 'refinement':
        return UnwarpUNetWithRefinement(flow_scale=flow_scale)
    else:
        raise ValueError(f"Unknown model type: {model_type}")