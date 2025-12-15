"""
Complete Evaluation Script for Document Unwarping Model
Fixed version with shape mismatch handling
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import argparse

from dataset_loader import get_dataloaders
from model import get_model, flow_to_color

# SSIM metric
try:
    from pytorch_msssim import ssim
    SSIM_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-msssim not available")
    SSIM_AVAILABLE = False


def compute_metrics(pred, target):
    """
    Compute PSNR and SSIM metrics.
    
    Args:
        pred: Predicted image [B, 3, H, W] or [3, H, W]
        target: Ground truth image [B, 3, H, W] or [3, H, W]
    
    Returns:
        dict with 'psnr' and 'ssim' scores
    """
    metrics = {}
    
    # Ensure batch dimension
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # FIX: Resize prediction to match target if shapes don't match
    if pred.shape[2:] != target.shape[2:]:
        print(f"  Resizing prediction from {pred.shape[2:]} to {target.shape[2:]}")
        pred = F.interpolate(
            pred, 
            size=target.shape[2:],
            mode='bilinear', 
            align_corners=True
        )
    
    # PSNR
    mse = F.mse_loss(pred, target)
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    metrics['psnr'] = psnr.item()
    
    # SSIM
    if SSIM_AVAILABLE:
        ssim_score = ssim(pred, target, data_range=1.0)
        metrics['ssim'] = ssim_score.item()
    else:
        metrics['ssim'] = 0.0
    
    return metrics


def visualize_result(input_img, pred_img, gt_img, flow, save_path, metrics=None):
    """
    Create comprehensive visualization of a single result.
    
    Args:
        input_img: Input warped image [3, H, W]
        pred_img: Predicted flat image [3, H, W]
        gt_img: Ground truth flat image [3, H, W]
        flow: Flow field [2, H, W]
        save_path: Where to save the visualization
        metrics: Optional dict with metric scores
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Convert to numpy [H, W, C]
    inp_np = input_img.cpu().permute(1, 2, 0).numpy()
    pred_np = pred_img.cpu().permute(1, 2, 0).numpy()
    gt_np = gt_img.cpu().permute(1, 2, 0).numpy()
    flow_np = flow.cpu().numpy()
    
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
    flow_vis = flow_to_color(torch.tensor(flow_np))
    ax5.imshow(flow_vis)
    ax5.set_title('Flow Field (Color-coded)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = plt.subplot(2, 4, 6)
    flow_mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
    im6 = ax6.imshow(flow_mag, cmap='jet')
    ax6.set_title('Flow Magnitude', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Edge comparison
    ax7 = plt.subplot(2, 4, 7)
    pred_gray = (pred_np * 255).astype(np.uint8)
    if pred_gray.ndim == 3:
        pred_gray = cv2.cvtColor(pred_gray, cv2.COLOR_RGB2GRAY)
    pred_edges = cv2.Canny(pred_gray, 100, 200)
    ax7.imshow(pred_edges, cmap='gray')
    ax7.set_title('Prediction Edges', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    ax8 = plt.subplot(2, 4, 8)
    gt_gray = (gt_np * 255).astype(np.uint8)
    if gt_gray.ndim == 3:
        gt_gray = cv2.cvtColor(gt_gray, cv2.COLOR_RGB2GRAY)
    gt_edges = cv2.Canny(gt_gray, 100, 200)
    ax8.imshow(gt_edges, cmap='gray')
    ax8.set_title('Ground Truth Edges', fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    # Add metrics text if provided
    if metrics:
        fig.suptitle(
            f"PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}",
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(checkpoint_path, data_dir, output_dir, num_vis=10):
    """
    Evaluate trained model on validation set.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Path to dataset
        output_dir: Where to save results
        num_vis: Number of visualizations to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'base')
    flow_scale = config.get('flow_scale', 1.0)
    img_size = config.get('img_size', 512)
    
    # Handle both tuple and int img_size
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    
    print(f"\nCheckpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'unknown'):.4f}" if 'val_loss' in checkpoint else "  Val loss: unknown")
    print(f"  Model type: {model_type}")
    print(f"  Image size: {img_size}")
    
    # Load model
    print("\nInitializing model...")
    model = get_model(model_type=model_type, flow_scale=flow_scale)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Load data
    print("\nLoading validation data...")
    _, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=1,  # Evaluate one at a time
        train_split=0.8,
        img_size=img_size,
        use_border=True
    )
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Evaluation
    print("\nEvaluating...")
    all_metrics = {'psnr': [], 'ssim': []}
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            rgb = batch['rgb'].to(device)
            gt = batch['ground_truth'].to(device)
            
            # Forward pass
            rectified, flow, _ = model(rgb)
            
            # Compute metrics
            metrics = compute_metrics(rectified, gt)
            all_metrics['psnr'].append(metrics['psnr'])
            all_metrics['ssim'].append(metrics['ssim'])
            
            # Save visualizations for first N samples
            if idx < num_vis:
                vis_path = output_dir / f'result_{idx:04d}.png'
                visualize_result(
                    rgb[0], rectified[0], gt[0], flow[0], 
                    vis_path, metrics
                )
            
            # Save individual predictions
            pred_path = output_dir / f'rectified_{idx:04d}.png'
            pred_img = (rectified[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(pred_path), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
    
    # Compute statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    psnr_mean = np.mean(all_metrics['psnr'])
    psnr_std = np.std(all_metrics['psnr'])
    ssim_mean = np.mean(all_metrics['ssim'])
    ssim_std = np.std(all_metrics['ssim'])
    
    print(f"\nMetrics:")
    print(f"  PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB")
    print(f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    print("="*60)
    
    # Save results to file
    results_file = output_dir / 'results.txt'
    with open(results_file, 'w') as f:
        f.write("DOCUMENT UNWARPING EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Number of samples: {len(val_loader.dataset)}\n\n")
        f.write(f"PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB\n")
        f.write(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}\n")
        f.write("="*60 + "\n\n")
        f.write("Per-sample metrics:\n")
        for i, (p, s) in enumerate(zip(all_metrics['psnr'], all_metrics['ssim'])):
            f.write(f"Sample {i:04d}: PSNR={p:.2f} dB, SSIM={s:.4f}\n")
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Create summary plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # PSNR distribution
    axes[0].hist(all_metrics['psnr'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(psnr_mean, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {psnr_mean:.2f}')
    axes[0].set_xlabel('PSNR (dB)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('PSNR Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # SSIM distribution
    axes[1].hist(all_metrics['ssim'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(ssim_mean, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {ssim_mean:.4f}')
    axes[1].set_xlabel('SSIM', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('SSIM Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics distribution plot saved")
    print(f"✓ {num_vis} visualizations saved")
    print(f"\n✓ Evaluation complete! All results in {output_dir}")
    
    return all_metrics


def inference_on_folder(checkpoint_path, input_folder, output_folder):
    """
    Run inference on a folder of images.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        input_folder: Folder containing input images
        output_folder: Where to save results
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = get_model(
        model_type=config.get('model_type', 'base'),
        flow_scale=config.get('flow_scale', 1.0)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model loaded")
    
    # Find all images
    image_files = list(input_folder.glob('*.jpg')) + list(input_folder.glob('*.png'))
    print(f"\nFound {len(image_files)} images")
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    print("\nProcessing images...")
    with torch.no_grad():
        for img_path in tqdm(image_files):
            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Inference
            rectified, flow, _ = model(img_tensor)
            
            # Save rectified image
            output_path = output_folder / f"rectified_{img_path.stem}.png"
            pred_img = (rectified[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(output_path), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
            
            # Save flow visualization
            flow_path = output_folder / f"flow_{img_path.stem}.png"
            flow_vis = flow_to_color(flow[0])
            cv2.imwrite(str(flow_path), cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR))
    
    print(f"\n✓ Results saved to {output_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate document unwarping model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, default='renders/synthetic_data_pitch_sweep',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Where to save evaluation results')
    parser.add_argument('--num_vis', type=int, default=10, 
                        help='Number of visualizations to save')
    parser.add_argument('--inference_only', action='store_true', 
                        help='Run inference on folder instead of evaluation')
    parser.add_argument('--input_folder', type=str, 
                        help='Input folder for inference mode')
    
    args = parser.parse_args()
    
    if args.inference_only:
        if args.input_folder is None:
            print("Error: --input_folder required for inference mode")
            print("Usage: python evaluate.py --checkpoint model.pth --inference_only --input_folder ./images")
        else:
            inference_on_folder(args.checkpoint, args.input_folder, args.output_dir)
    else:
        evaluate_model(args.checkpoint, args.data_dir, args.output_dir, args.num_vis)