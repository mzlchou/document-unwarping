"""
Improved Document Unwarping Model
==================================
Clean, well-documented implementation with flow-based geometric correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DocumentUnwarpModel(nn.Module):
    """
    Flow-based document unwarping using U-Net architecture.
    
    Key Concepts:
    1. Encoder: Extract features from warped document (uses pretrained ResNet50)
    2. Decoder: Upsample features back to full resolution with skip connections
    3. Flow Head: Predict displacement field [B, 2, H, W] (dx, dy for each pixel)
    4. Warping: Use grid_sample to warp input according to predicted flow
    
    The model learns: "For each pixel in the warped image, where should it move
    to in the flat document?"
    """
    
    def __init__(self, pretrained=True, flow_scale=0.5):
        """
        Args:
            pretrained: Use pretrained ResNet50 encoder
            flow_scale: Maximum flow displacement (0.5 = half image size)
        """
        super().__init__()
        self.flow_scale = flow_scale
        
        # ============================================================
        # ENCODER: Pretrained ResNet50 (extracts features at 4 scales)
        # ============================================================
        self.encoder = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)  # Returns features at 4 different scales
        )
        # Feature channels at each scale: [256, 512, 1024, 2048]
        # Feature resolutions: [H/4, H/8, H/16, H/32] (progressively smaller)
        
        # ============================================================
        # DECODER: Upsample features back to full resolution
        # ============================================================
        # Each decoder block: upsample 2x + convolutions
        self.decoder4 = self._make_decoder_block(2048, 1024)  # Deepest level
        self.decoder3 = self._make_decoder_block(1024, 512)
        self.decoder2 = self._make_decoder_block(512, 256)
        self.decoder1 = self._make_decoder_block(256, 128)
        
        # Final upsampling to match input resolution
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # ============================================================
        # FLOW HEAD: Predict 2-channel displacement field
        # ============================================================
        self.flow_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1)  # Output: [B, 2, H, W]
        )
        
        # Initialize flow head to predict small displacements initially
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)
    
    def _make_decoder_block(self, in_channels, out_channels):
        """
        Create a decoder block: upsample + 2 conv layers with batch norm.
        
        This is the standard U-Net decoder pattern.
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def create_sampling_grid(self, flow):
        """
        Convert flow field to sampling grid for grid_sample.
        
        Args:
            flow: [B, 2, H, W] displacement field
        
        Returns:
            grid: [B, H, W, 2] sampling coordinates in range [-1, 1]
        
        Explanation:
        - grid_sample expects coordinates in [-1, 1] where:
          * (-1, -1) = top-left corner
          * (1, 1) = bottom-right corner
        - We start with identity grid (regular mesh)
        - Add predicted flow to get final sampling locations
        """
        B, _, H, W = flow.shape
        
        # Create identity grid in [-1, 1] coordinates
        y = torch.linspace(-1, 1, H, device=flow.device)
        x = torch.linspace(-1, 1, W, device=flow.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        identity_grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        identity_grid = identity_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        
        # Add flow (need to permute from [B, 2, H, W] to [B, H, W, 2])
        flow_permuted = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # Scale flow to appropriate range and add to identity
        sampling_grid = identity_grid + flow_permuted * self.flow_scale
        
        return sampling_grid
    
    def forward(self, x):
        """
        Forward pass: predict flow and warp input.
        
        Args:
            x: [B, 3, H, W] input warped document
        
        Returns:
            rectified: [B, 3, H, W] dewarped document
            flow: [B, 2, H, W] predicted displacement field
            grid: [B, H, W, 2] sampling grid (for visualization)
        """
        B, C, H, W = x.shape
        
        # ============================================================
        # STEP 1: Extract multi-scale features with encoder
        # ============================================================
        encoder_features = self.encoder(x)
        e1, e2, e3, e4 = encoder_features  # 4 feature maps at different scales
        
        # ============================================================
        # STEP 2: Decode features with skip connections (U-Net style)
        # ============================================================
        # Decoder level 4 (deepest)
        d4 = self.decoder4(e4)
        d4 = self._match_size(d4, e3)  # Resize if needed
        d4 = d4 + e3  # Skip connection (residual)
        
        # Decoder level 3
        d3 = self.decoder3(d4)
        d3 = self._match_size(d3, e2)
        d3 = d3 + e2  # Skip connection
        
        # Decoder level 2
        d2 = self.decoder2(d3)
        d2 = self._match_size(d2, e1)
        d2 = d2 + e1  # Skip connection
        
        # Decoder level 1
        d1 = self.decoder1(d2)
        
        # Final upsampling to input resolution
        features = self.final_upsample(d1)
        features = self._match_size(features, x)  # Ensure exact size match
        
        # ============================================================
        # STEP 3: Predict flow field
        # ============================================================
        flow = self.flow_head(features)  # [B, 2, H, W]
        flow = torch.tanh(flow)  # Bound to [-1, 1]
        
        # ============================================================
        # STEP 4: Create sampling grid and warp input
        # ============================================================
        sampling_grid = self.create_sampling_grid(flow)
        
        # grid_sample: for each output pixel, sample from input at grid location
        rectified = F.grid_sample(
            x,
            sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return rectified, flow, sampling_grid
    
    def _match_size(self, x, target):
        """Resize x to match target's spatial dimensions."""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], 
                            mode='bilinear', align_corners=True)
        return x


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def visualize_flow(flow, max_magnitude=None):
    """
    Convert flow field to color image for visualization.
    
    Args:
        flow: [2, H, W] or [B, 2, H, W] flow field
        max_magnitude: Maximum flow magnitude for normalization
    
    Returns:
        RGB image where color indicates flow direction and brightness indicates magnitude
    """
    import numpy as np
    
    if flow.dim() == 4:
        flow = flow[0]  # Take first batch item
    
    flow_np = flow.detach().cpu().numpy()
    u, v = flow_np[0], flow_np[1]
    
    # Compute magnitude and angle
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    
    if max_magnitude is None:
        max_magnitude = magnitude.max()
    
    # Create HSV color wheel
    hsv = np.zeros((flow_np.shape[1], flow_np.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # Hue = direction
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = (np.clip(magnitude / (max_magnitude + 1e-8), 0, 1) * 255).astype(np.uint8)  # Value = magnitude
    
    # Convert to RGB
    import cv2
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def visualize_results(input_img, prediction, ground_truth, flow=None):
    """
    Create comprehensive visualization of results.
    
    Args:
        input_img: [3, H, W] input image
        prediction: [3, H, W] predicted dewarped image
        ground_truth: [3, H, W] ground truth flat image
        flow: [2, H, W] predicted flow field (optional)
    """
    import matplotlib.pyplot as plt
    
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def denorm(img):
        img = img.cpu() * std + mean
        return img.permute(1, 2, 0).clamp(0, 1).numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Row 1: Input and prediction
    axes[0, 0].imshow(denorm(input_img))
    axes[0, 0].set_title('Input (Warped)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(denorm(prediction))
    axes[0, 1].set_title('Prediction (Dewarped)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Row 2: Ground truth and flow
    axes[1, 0].imshow(denorm(ground_truth))
    axes[1, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    if flow is not None:
        flow_vis = visualize_flow(flow)
        axes[1, 1].imshow(flow_vis)
        axes[1, 1].set_title('Flow Field\n(Color=Direction, Brightness=Magnitude)', 
                            fontsize=14, fontweight='bold')
    else:
        axes[1, 1].axis('off')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == '__main__':
    # Test model creation
    model = DocumentUnwarpModel(pretrained=False)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    rectified, flow, grid = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {rectified.shape}")
    print(f"Flow shape: {flow.shape}")
    print(f"Grid shape: {grid.shape}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✓ Model test passed!")