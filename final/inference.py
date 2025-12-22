"""
inference.py
Automatic Document Unwarping Inference Script
Process a folder of warped document images and output rectified results
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse

from model import make_documentunwarp


def denormalize(img):
    """Denormalize from ImageNet normalization to [0,1] range"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    return (img * std + mean).clamp(0, 1)


def flow_to_color(flow):
    """Convert UV flow to color visualization"""
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
    
    # Manual HSV to RGB conversion
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


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint
    """
    print(f"Loading model from: {checkpoint_path}")
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
    
    print(f"Model loaded successfully")
    print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_ssim' in checkpoint:
        print(f"   Val SSIM: {checkpoint['val_ssim']:.4f}")
    
    return model, config


def preprocess_image(img_path, img_size=512):
    """
    Load and preprocess a single image
    
    Args:
        img_path: Path to input image
        img_size: Target size for processing
    
    Returns:
        img_tensor: Preprocessed tensor [1, 3, H, W]
        original_img: Original PIL image for reference
        original_size: (width, height) of original image
    """
    # Load image
    img = Image.open(img_path).convert('RGB')
    original_size = img.size  # (width, height)
    
    # Resize to target size
    img_resized = img.resize((img_size, img_size), Image.BILINEAR)
    
    # Convert to tensor and normalize (ImageNet normalization)
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor, img, original_size


def inference_folder(model, input_folder, output_folder, device='cuda', 
                     pred_downscale=2, save_uv=True, img_size=512):
    """
    Process a folder of images automatically
    
    Args:
        model: Loaded model
        input_folder: Path to folder with input images
        output_folder: Path to save rectified images
        device: Device to run inference on
        pred_downscale: Downscaling factor for UV prediction
        save_uv: Whether to save UV flow visualizations
        img_size: Size to process images at
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories for outputs
    rectified_dir = output_folder / 'rectified'
    rectified_dir.mkdir(exist_ok=True)
    
    if save_uv:
        uv_dir = output_folder / 'uv_visualizations'
        uv_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_folder.glob(ext)))
    
    if len(image_files) == 0:
        print(f"No images found in {input_folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {len(image_files)} images from: {input_folder}")
    print(f"Output directory: {output_folder}")
    print(f"{'='*60}\n")
    
    model.eval()
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Load and preprocess
                img_tensor, original_img, original_size = preprocess_image(img_path, img_size)
                img_tensor = img_tensor.to(device)
                
                # Run model
                pred = model(img_tensor, pred_downscale=pred_downscale, return_uv_pred=True)
                rectified = pred['rectified']
                uv = pred['uv']
                
                # Denormalize
                rectified_denorm = denormalize(rectified)
                
                # Convert to numpy image
                rectified_img = rectified_denorm[0].cpu().permute(1, 2, 0).numpy()
                rectified_img = (rectified_img * 255).clip(0, 255).astype(np.uint8)
                
                # Resize back to original size if needed
                rectified_pil = Image.fromarray(rectified_img)
                if original_size != (img_size, img_size):
                    rectified_pil = rectified_pil.resize(original_size, Image.BILINEAR)
                
                # Save rectified image
                output_path = rectified_dir / f"rectified_{img_path.name}"
                rectified_pil.save(output_path)
                
                # Save UV visualization if requested
                if save_uv:
                    uv_vis = flow_to_color(uv[0].cpu())
                    uv_path = uv_dir / f"uv_{img_path.stem}.png"
                    Image.fromarray(uv_vis).save(uv_path)
                
            except Exception as e:
                print(f"\nError processing {img_path.name}: {str(e)}")
                continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"   Rectified images saved to: {rectified_dir}")
    if save_uv:
        print(f"   UV visualizations saved to: {uv_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Automatic Document Unwarping Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python inference.py --input input_images/ --output results/
  
  # Specify checkpoint and don't save UV visualizations
  python inference.py --input my_docs/ --output unwarp_results/ --checkpoint best_model.pth --no-uv
  
  # Process at higher resolution
  python inference.py --input scans/ --output clean/ --img_size 768
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input folder containing warped document images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output folder for rectified images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--img_size', type=int, default=512,
                       help='Image size for processing (default: 512)')
    parser.add_argument('--pred_downscale', type=int, default=2,
                       help='UV prediction downscaling factor (default: 2)')
    parser.add_argument('--no-uv', action='store_true',
                       help='Skip saving UV flow visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model, config = load_model(args.checkpoint, args.device)
    
    # Get pred_downscale from config if available
    pred_downscale = config.get('pred_downscale', args.pred_downscale)
    
    # Run inference
    inference_folder(
        model=model,
        input_folder=args.input,
        output_folder=args.output,
        device=args.device,
        pred_downscale=pred_downscale,
        save_uv=not args.no_uv,
        img_size=args.img_size
    )


if __name__ == '__main__':
    main()