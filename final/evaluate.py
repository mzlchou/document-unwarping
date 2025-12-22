"""
evaluate.py
Evaluation and Inference Script for Document Unwarping
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from model import make_documentunwarp
from dataset_loader import get_dataloaders

# Metrics
try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: skimage not available. SSIM/PSNR metrics disabled.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Edge detection disabled.")


# FUNCTIONS

def denormalize(img):
    """Denormalize from ImageNet normalization to [0,1] range"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    return (img * std + mean).clamp(0, 1)


def extract_document_region(batch, channel_idx=0):
    """EXACT copy from train.py - creates paper mask from border channel"""
    border = batch["border"]
    if border.ndim == 3: 
        border = border.unsqueeze(1)
    
    # Get center region (1/8 of image size)
    b, _, h, w = border.shape
    ch = max(1, h // 8)
    cw = max(1, w // 8)
    center = border[:, :, h//2 - ch//2 : h//2 + ch//2, 
                          w//2 - cw//2 : w//2 + cw//2]
    center_mean = center.mean().item()
    
    # Use lower threshold (0.25 instead of 0.5)
    thresh = 0.25
    
    # Determine polarity from mid
    if center_mean < 0.5:
        output_mask = (border < thresh).float()
    else:
        output_mask = (border > thresh).float()
    
    # Dilate mask
    output_mask = F.max_pool2d(output_mask, kernel_size=11, stride=1, padding=5)
    
    if output_mask.shape[1] > 1:
        output_mask = output_mask[:, channel_idx:channel_idx+1]
    
    return output_mask


def warp_mask(mask_in, grid, align_corners=True):
    if mask_in.ndim != 4 or mask_in.shape[1] != 1:
        raise ValueError(f"mask_in must be [B,1,H,W], got {tuple(mask_in.shape)}")
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(f"grid must be [B,H,W,2], got {tuple(grid.shape)}")
    m = F.grid_sample(mask_in, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners)
    return m.clamp(0.0, 1.0)


def compute_ssim_map(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8)
    return ssim_map


def masked_ssim_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        mask = mask[:, 0:1]

    mask_sum = mask.sum() + 1e-8
    mask_norm = mask / mask_sum

    ssim_map = compute_ssim_map(pred, target, data_range)
    ssim_map = ssim_map.mean(dim=1, keepdim=True)

    masked_ssim = (ssim_map * mask_norm).sum()
    return 1.0 - masked_ssim


def masked_ssim_value(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return 1.0 - masked_ssim_loss(pred, target, mask, data_range)


# ADDITIONAL METRICS (using skimage for comparison)
def compute_metrics(pred, target):
    """
    Compute SSIM and PSNR using skimage
    
    Args:
        pred: [3, H, W] tensor (denormalized)
        target: [3, H, W] tensor (denormalized)
    
    Returns:
        dict with 'ssim' and 'psnr' scores
    """
    if not METRICS_AVAILABLE:
        return {'ssim': 0.0, 'psnr': 0.0}
    
    # Convert to numpy [H, W, C] in range [0, 1]
    pred_np = pred.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    target_np = target.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    
    # SSIM
    ssim_score = structural_similarity(
        pred_np, target_np,
        channel_axis=2,
        data_range=1.0
    )
    
    # PSNR
    psnr_score = peak_signal_noise_ratio(
        target_np, pred_np,
        data_range=1.0
    )
    
    return {
        'ssim': float(ssim_score),
        'psnr': float(psnr_score)
    }


# FLOW VISUALIZATION
def flow_to_color(flow):
    """Convert optical flow to color visualization"""
    if torch.is_tensor(flow):
        flow = flow.cpu().numpy()
    
    u = flow[0]
    v = flow[1]
    
    # Compute magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)
    
    # Normalize magnitude
    mag_max = mag.max() + 1e-8
    mag_norm = mag / mag_max
    
    # Convert to HSV then RGB
    h = (ang + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v_val = mag_norm
    
    # Manual HSV to RGB
    h6 = h * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v_val * (1 - s)
    q = v_val * (1 - f * s)
    t = v_val * (1 - (1 - f) * s)
    
    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    
    mask = i == 0
    rgb[mask] = np.stack([v_val[mask], t[mask], p[mask]], axis=-1)
    mask = i == 1
    rgb[mask] = np.stack([q[mask], v_val[mask], p[mask]], axis=-1)
    mask = i == 2
    rgb[mask] = np.stack([p[mask], v_val[mask], t[mask]], axis=-1)
    mask = i == 3
    rgb[mask] = np.stack([p[mask], q[mask], v_val[mask]], axis=-1)
    mask = i == 4
    rgb[mask] = np.stack([t[mask], p[mask], v_val[mask]], axis=-1)
    mask = i == 5
    rgb[mask] = np.stack([v_val[mask], p[mask], q[mask]], axis=-1)
    
    return (rgb * 255).astype(np.uint8)


# VISUALIZATION
def visualize_result_comprehensive(input_img, pred_img, gt_img, flow, mask, save_path, metrics=None):
    """Create 8-panel comprehensive visualization"""
    fig = plt.figure(figsize=(20, 10))
    
    # Convert to [H, W, C]
    inp_np = input_img.cpu().permute(1, 2, 0).numpy()
    pred_np = pred_img.cpu().permute(1, 2, 0).numpy()
    gt_np = gt_img.cpu().permute(1, 2, 0).numpy()
    mask_np = mask.cpu().squeeze().numpy() if mask is not None else None
    
    # Row 1: Images
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(np.clip(inp_np, 0, 1))
    ax1.set_title('Input (Warped)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(np.clip(pred_np, 0, 1))
    ax2.set_title('Prediction (Dewarped)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(np.clip(gt_np, 0, 1))
    ax3.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 4, 4)
    diff = np.abs(pred_np - gt_np).mean(axis=2)
    im4 = ax4.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    ax4.set_title('Absolute Difference', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Row 2: Analysis
    ax5 = plt.subplot(2, 4, 5)
    if flow is not None:
        flow_np = flow.cpu().numpy() if torch.is_tensor(flow) else flow
        flow_vis = flow_to_color(flow_np)
        ax5.imshow(flow_vis)
        ax5.set_title('Flow Field (UV)', fontsize=12, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'Flow N/A', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Flow Field', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = plt.subplot(2, 4, 6)
    if mask_np is not None:
        im6 = ax6.imshow(mask_np, cmap='gray')
        ax6.set_title('Paper Mask', fontsize=12, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Mask N/A', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Paper Mask', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # Edge comparison
    ax7 = plt.subplot(2, 4, 7)
    if CV2_AVAILABLE:
        pred_gray = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
        if pred_gray.ndim == 3:
            pred_gray = cv2.cvtColor(pred_gray, cv2.COLOR_RGB2GRAY)
        pred_edges = cv2.Canny(pred_gray, 100, 200)
        ax7.imshow(pred_edges, cmap='gray')
    else:
        ax7.text(0.5, 0.5, 'cv2 N/A', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Prediction Edges', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    ax8 = plt.subplot(2, 4, 8)
    if CV2_AVAILABLE:
        gt_gray = (np.clip(gt_np, 0, 1) * 255).astype(np.uint8)
        if gt_gray.ndim == 3:
            gt_gray = cv2.cvtColor(gt_gray, cv2.COLOR_RGB2GRAY)
        gt_edges = cv2.Canny(gt_gray, 100, 200)
        ax8.imshow(gt_edges, cmap='gray')
    else:
        ax8.text(0.5, 0.5, 'cv2 N/A', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('Ground Truth Edges', fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    # Add metrics
    if metrics:
        title_parts = []
        if 'psnr' in metrics:
            title_parts.append(f"PSNR: {metrics['psnr']:.2f} dB")
        if 'ssim' in metrics:
            title_parts.append(f"SSIM: {metrics['ssim']:.4f}")
        if 'masked_ssim' in metrics:
            title_parts.append(f"Masked SSIM: {metrics['masked_ssim']:.4f}")
        if title_parts:
            fig.suptitle(" | ".join(title_parts), fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# VIS FOR SSIM RANGE
def plot_ssim_histogram(all_ssim, save_path, num_samples=None):
    if num_samples is None:
        num_samples = len(all_ssim)
    
    ssim_mean = np.mean(all_ssim)
    ssim_std = np.std(all_ssim)
    ssim_median = np.median(all_ssim)
    ssim_min = min(all_ssim)
    ssim_max = max(all_ssim)
    ssim_25 = np.percentile(all_ssim, 25)
    ssim_75 = np.percentile(all_ssim, 75)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    counts, bins, patches = ax.hist(all_ssim, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    
    ax.axvline(ssim_mean, color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {ssim_mean:.4f}', zorder=10)
    
    ax.axvline(ssim_median, color='green', linestyle='--', linewidth=2.5,
               label=f'Median: {ssim_median:.4f}', zorder=10)
    
    ax.axvline(ssim_25, color='orange', linestyle=':', linewidth=2,
               label=f'25th %ile: {ssim_25:.4f}', alpha=0.7)
    ax.axvline(ssim_75, color='orange', linestyle=':', linewidth=2,
               label=f'75th %ile: {ssim_75:.4f}', alpha=0.7)
    
    ax.set_xlabel('SSIM Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=14, fontweight='bold')
    ax.set_title(f'SSIM Distribution Across {num_samples} Images', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')

    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {ssim_mean:.4f}\n'
    stats_text += f'Std Dev: {ssim_std:.4f}\n'
    stats_text += f'Median: {ssim_median:.4f}\n'
    stats_text += f'Min: {ssim_min:.4f}\n'
    stats_text += f'Max: {ssim_max:.4f}\n'
    stats_text += f'Range: {ssim_max - ssim_min:.4f}'
    
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"SSIM histogram saved to {save_path}")

# MAIN EVALUATION FUNCTION
def evaluate_model(model, dataloader, device, save_dir=None, pred_downscale=2, num_vis=10, 
                   checkpoint_path=None, data_dir=None):
    """
    Evaluate model on entire dataset using EXACT training metrics
    """
    model.eval()
    
    all_ssim = []
    all_psnr = []
    all_masked_ssim = []
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    print("Evaluating model...")
    
    sample_idx = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            rgb = batch['rgb'].to(device)
            gt = batch['ground_truth'].to(device)
            
            # Create mask
            mask = extract_document_region(batch).to(device)
            
            # Forward pass - model returns dict
            pred = model(rgb, pred_downscale=pred_downscale, return_uv_pred=True)
            rectified = pred['rectified']
            uv = pred['uv']
            grid = pred['grid']
            
            # Denormalize
            rectified_denorm = denormalize(rectified)
            gt_denorm = denormalize(gt)
            
            # Warp mask to output space
            mask_src_soft = warp_mask(mask.to(dtype=torch.float32), grid, align_corners=True)
            mask_out = (mask_src_soft > 0.5).to(dtype=rectified_denorm.dtype)
            
            # Compute metrics for each sample
            for i in range(rgb.shape[0]):
                # Full image metrics (for reference)
                metrics = compute_metrics(rectified_denorm[i], gt_denorm[i])
                all_ssim.append(metrics['ssim'])
                all_psnr.append(metrics['psnr'])
                
                # Masked SSIM
                masked_ssim_val = float(masked_ssim_value(
                    rectified_denorm[i:i+1],
                    gt_denorm[i:i+1],
                    mask_out[i:i+1],
                    data_range=1.0
                ).item())
                all_masked_ssim.append(masked_ssim_val)
                
                if save_dir:
                    # Save rectified image
                    pred_img = (rectified_denorm[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    if CV2_AVAILABLE:
                        cv2.imwrite(
                            str(save_dir / f'rectified_{sample_idx:04d}.png'),
                            cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                        )
                    else:
                        Image.fromarray(pred_img).save(save_dir / f'rectified_{sample_idx:04d}.png')
                    
                    # Save comprehensive visualization for first num_vis samples
                    if sample_idx < num_vis:
                        sample_metrics = {
                            'psnr': metrics['psnr'],
                            'ssim': metrics['ssim'],
                            'masked_ssim': masked_ssim_val
                        }
                        
                        visualize_result_comprehensive(
                            denormalize(rgb[i:i+1])[0],
                            rectified_denorm[i],
                            gt_denorm[i],
                            uv[i],
                            mask_out[i:i+1],
                            save_dir / f'result_{sample_idx:04d}.png',
                            sample_metrics
                        )
                
                sample_idx += 1
    
    # Compute average metrics
    results = {
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'psnr_mean': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
        'masked_ssim_mean': np.mean(all_masked_ssim),
        'masked_ssim_std': np.std(all_masked_ssim),
        'num_samples': len(all_ssim),
        'all_psnr': all_psnr,
        'all_ssim': all_ssim,
        'all_masked_ssim': all_masked_ssim
    }
    
    if save_dir:
        # Save results
        results_file = save_dir / 'results.txt'
        with open(results_file, 'w') as f:
            f.write("DOCUMENT UNWARPING EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            if checkpoint_path:
                f.write(f"Checkpoint: {checkpoint_path}\n")
            if data_dir:
                f.write(f"Dataset: {data_dir}\n")
            f.write(f"Number of samples: {results['num_samples']}\n\n")
            f.write(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB\n")
            f.write(f"SSIM (full image): {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}\n")
            f.write(f"Masked SSIM: {results['masked_ssim_mean']:.4f} ± {results['masked_ssim_std']:.4f} (TRAINING METRIC)\n")
            f.write("=" * 60 + "\n\n")
            f.write("Per-sample metrics:\n")
            for idx, (p, s, ms) in enumerate(zip(all_psnr, all_ssim, all_masked_ssim)):
                f.write(f"Sample {idx:04d}: PSNR={p:.2f} dB, SSIM={s:.4f}, Masked SSIM={ms:.4f}\n")
        
        print(f"\nResults saved to {results_file}")
        
        # Save distribution plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        
        axes[0].hist(all_psnr, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(results['psnr_mean'], color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {results["psnr_mean"]:.2f}')
        axes[0].set_xlabel('PSNR (dB)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('PSNR Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].hist(all_ssim, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1].axvline(results['ssim_mean'], color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {results["ssim_mean"]:.4f}')
        axes[1].set_xlabel('SSIM', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('SSIM Distribution (Full Image)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        axes[2].hist(all_masked_ssim, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[2].axvline(results['masked_ssim_mean'], color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {results["masked_ssim_mean"]:.4f}')
        axes[2].set_xlabel('Masked SSIM', fontsize=12)
        axes[2].set_ylabel('Count', fontsize=12)
        axes[2].set_title('Masked SSIM (Training Metric)', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_ssim_histogram(all_ssim, save_dir / 'ssim_histogram.png', results['num_samples'])

        print(f"Metrics distribution saved")
        print(f"{min(num_vis, results['num_samples'])} comprehensive visualizations saved")
        print(f"{results['num_samples']} individual rectified images saved")
    
    return results


# MODEL LOADING
def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Build model with config from checkpoint
    model = make_documentunwarp(
        backbone=config.get('backbone', 'resnet50'),
        pretrained=False,
        uv_mode='residual',
        max_disp=config.get('max_disp', 0.25),
        warp_mode=config.get('warp_mode', 'bicubic'),
        pred_downscale=config.get('pred_downscale', 2),
        two_stage_warp=config.get('two_stage_warp', True),
        coarse_stride=config.get('coarse_stride', 64),
        coarse_disp_scale=config.get('coarse_disp_scale', 1.5),
        fine_disp_scale=config.get('fine_disp_scale', 0.5),
        use_attention=False,
        use_uv_bias=False,
        use_post_refine=False,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_ssim' in checkpoint:
        print(f"  Val SSIM: {checkpoint['val_ssim']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model, config


# MAIN
def main_evaluate():
    """Evaluate model on validation set"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Document Unwarping Model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='renders/synthetic_data_pitch_sweep',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=512,
                       help='Image size for evaluation')
    parser.add_argument('--num_vis', type=int, default=10,
                       help='Number of comprehensive visualizations')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("Document Unwarping Model Evaluation")
    print("="*60)
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    pred_downscale = config.get('pred_downscale', 2)
    
    # Load validation data
    print(f"\nLoading validation data from: {args.data_dir}")
    _, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        use_border=True,
        use_uv=False,
        use_depth=False,
        num_workers=4
    )
    print(f"Validation batches: {len(val_loader)}")
    
    # Evaluate
    results = evaluate_model(
        model, val_loader, device,
        save_dir=args.output_dir,
        pred_downscale=pred_downscale,
        num_vis=args.num_vis,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Samples:            {results['num_samples']}")
    print(f"PSNR:               {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"SSIM (full image):  {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"Masked SSIM:        {results['masked_ssim_mean']:.4f} ± {results['masked_ssim_std']:.4f} TRAINING METRIC")
    print("="*60)
    
    # Quality assessment using masked SSIM (training metric)
    print("\nQuality Assessment (based on Masked SSIM):")
    if results['masked_ssim_mean'] > 0.85:
        print("EXCELLENT")
    elif results['masked_ssim_mean'] > 0.75:
        print("GOOD")
    elif results['masked_ssim_mean'] > 0.65:
        print("FAIR")
    else:
        print("NEEDS IMPROVEMENT")
    
    print(f"\n✅ Evaluation complete! All results saved to: {args.output_dir}")


if __name__ == '__main__':
    main_evaluate()