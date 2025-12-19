"""
Evaluation and Inference Script for Document Unwarping
=======================================================
Evaluate trained model and run inference on new images.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Import your modules
from dataset_loader import get_dataloaders
from final.model2 import DocumentUnwarpModel, visualize_results

# Metrics
try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("⚠️  skimage not available. Install with: pip install scikit-image")


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_metrics(pred, target):
    """
    Compute SSIM and PSNR between prediction and target.
    
    Args:
        pred: [3, H, W] tensor
        target: [3, H, W] tensor
    
    Returns:
        dict with 'ssim' and 'psnr' scores
    """
    if not METRICS_AVAILABLE:
        return {'ssim': 0.0, 'psnr': 0.0}
    
    # Convert to numpy [H, W, C] in range [0, 1]
    pred_np = pred.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    target_np = target.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    
    # Compute SSIM (structural similarity)
    ssim_score = structural_similarity(
        pred_np, target_np,
        channel_axis=2,
        data_range=1.0
    )
    
    # Compute PSNR (peak signal-to-noise ratio)
    psnr_score = peak_signal_noise_ratio(
        target_np, pred_np,
        data_range=1.0
    )
    
    return {
        'ssim': float(ssim_score),
        'psnr': float(psnr_score)
    }


def evaluate_model(model, dataloader, device, save_dir=None):
    """
    Evaluate model on entire dataset.
    
    Args:
        model: DocumentUnwarpModel
        dataloader: validation dataloader
        device: torch device
        save_dir: optional directory to save visualizations
    
    Returns:
        dict with average metrics
    """
    model.eval()
    
    all_ssim = []
    all_psnr = []
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            rgb = batch['rgb'].to(device)
            gt = batch['ground_truth'].to(device)
            
            # Forward pass
            rectified, flow, grid = model(rgb)
            
            # Denormalize for metrics
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            
            rectified_denorm = (rectified * std + mean).clamp(0, 1)
            gt_denorm = (gt * std + mean).clamp(0, 1)
            
            # Compute metrics for each sample in batch
            for i in range(rgb.shape[0]):
                metrics = compute_metrics(rectified_denorm[i], gt_denorm[i])
                all_ssim.append(metrics['ssim'])
                all_psnr.append(metrics['psnr'])
            
            # Save visualizations for first few batches
            if save_dir and batch_idx < 5:
                for i in range(min(2, rgb.shape[0])):
                    fig = visualize_results(
                        rgb[i],
                        rectified[i],
                        gt[i],
                        flow[i]
                    )
                    plt.savefig(
                        save_dir / f'eval_batch{batch_idx}_sample{i}.png',
                        dpi=150, bbox_inches='tight'
                    )
                    plt.close(fig)
    
    # Compute average metrics
    results = {
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'psnr_mean': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
        'num_samples': len(all_ssim)
    }
    
    return results


# ============================================================
# INFERENCE ON FOLDER
# ============================================================

def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: path to .pth file
        device: torch device
    
    Returns:
        model: loaded DocumentUnwarpModel
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same config
    config = checkpoint.get('config', {})
    model = DocumentUnwarpModel(
        pretrained=False,  # Weights already trained
        flow_scale=config.get('flow_scale', 0.5)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model


def preprocess_image(image_path, img_size=512):
    """
    Load and preprocess image for inference.
    
    Args:
        image_path: path to image file
        img_size: target image size
    
    Returns:
        tensor: [1, 3, H, W] preprocessed image
        original: PIL Image (for saving later)
    """
    from torchvision import transforms
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return tensor, img


def postprocess_image(tensor):
    """
    Convert model output back to PIL Image.
    
    Args:
        tensor: [1, 3, H, W] or [3, H, W] output tensor
    
    Returns:
        PIL Image
    """
    from torchvision import transforms
    
    if tensor.dim() == 4:
        tensor = tensor[0]  # Remove batch dimension
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = (tensor.cpu() * std + mean).clamp(0, 1)
    
    # Convert to PIL
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    
    return img


def process_folder(model, input_folder, output_folder, device='cuda', img_size=512):
    """
    Process all images in a folder.
    
    Args:
        model: trained DocumentUnwarpModel
        input_folder: folder containing warped images
        output_folder: folder to save dewarped images
        device: torch device
        img_size: image size for processing
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if len(image_files) == 0:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Processing...")
    
    model.eval()
    
    with torch.no_grad():
        for img_path in tqdm(image_files):
            try:
                # Load and preprocess
                tensor, original = preprocess_image(img_path, img_size)
                tensor = tensor.to(device)
                
                # Inference
                rectified, flow, grid = model(tensor)
                
                # Postprocess
                output_img = postprocess_image(rectified)
                
                # Resize back to original size if needed
                output_img = output_img.resize(original.size, Image.LANCZOS)
                
                # Save
                output_name = img_path.stem + '_rectified' + img_path.suffix
                output_img.save(output_path / output_name, quality=95)
                
                # Also save flow visualization
                from final.model2 import visualize_flow
                flow_vis = visualize_flow(flow)
                flow_pil = Image.fromarray(flow_vis)
                flow_pil = flow_pil.resize(original.size, Image.NEAREST)
                flow_name = img_path.stem + '_flow.png'
                flow_pil.save(output_path / flow_name)
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
    
    print(f"\n✓ Done! Results saved to {output_folder}")


# ============================================================
# MAIN FUNCTIONS
# ============================================================

def main_evaluate():
    """Evaluate model on validation set."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Load model
    model = load_model('checkpoints/best_model.pth', device)
    
    # Load validation data
    print("\nLoading validation data...")
    _, val_loader = get_dataloaders(
        data_dir='renders/synthetic_data_pitch_sweep',
        batch_size=8,
        img_size=(512, 512),
        use_border=True
    )
    
    # Evaluate
    results = evaluate_model(
        model, val_loader, device,
        save_dir='evaluation_results'
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Samples: {results['num_samples']}")
    print(f"SSIM:    {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"PSNR:    {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print("="*60)


def main_inference():
    """Run inference on a folder of images."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Document Unwarping Inference')
    parser.add_argument('--input', type=str, required=True,
                       help='Input folder with warped images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output folder for dewarped images')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--size', type=int, default=512,
                       help='Image size for processing')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("Document Unwarping Inference")
    print("="*60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model}")
    print("="*60 + "\n")
    
    # Load model
    model = load_model(args.model, device)
    
    # Process folder
    process_folder(
        model,
        args.input,
        args.output,
        device,
        img_size=args.size
    )


if __name__ == '__main__':
    # Run evaluation (uncomment one)
    main_evaluate()
    
    # Or run inference (uncomment and modify)
    # python evaluate.py --input path/to/warped/images --output path/to/output