"""
train.ipynb 

RUN IN COLAB FOR TRAINING WITH A100
"""

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time
import math
from kornia.color import rgb_to_grayscale
import kornia.filters as kf
import kornia.metrics as km

try:
    get_ipython().run_line_magic('pip', 'install -q timm kornia tensorboard')
except:
    pass

# for colab, importing drive to load in dataset
try:
    from google.colab import drive
    drive.mount('/content/drive')
except:
    pass


# Cloning my github for my model and dataset loader files
GITHUB_USERNAME = "mzlchou"
GITHUB_REPO = "document-unwarping"
repo_url = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
print("🔄 Cloning repository...")
if os.path.exists(GITHUB_REPO):
    !rm -rf {GITHUB_REPO}
!git clone {repo_url}
os.chdir(GITHUB_REPO)
print(f"Repository cloned to: {os.getcwd()}")


# Config for easy adjustments
CONFIG = {
    # trying to optimize for A100
    'batch_size': 8,
    'img_size': 512,
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'weight_decay': 1e-2,
    'num_workers': 8,
    'save_every': 5,
    'early_stopping_patience': 50,
    'eff_batch': 16,

    # Model settings
    'backbone': 'resnet50',
    'warp_mode': 'bicubic',
    'pred_downscale': 2,
    'two_stage_warp': True,
    'max_disp': 0.25,
    'coarse_stride': 64,
    'coarse_disp_scale': 1.5,
    'fine_disp_scale': 0.5,

    # Loss Weights
    'tv_weight': 0.03,
    'tv_start_epoch': 3,
    'fold_weight': 0.5,
    'fold_start_epoch': 6,
    'fold_det_eps': 0.2,
    'fold_downscale': 2,
    'fold_clip': 10.0,
    'hp_weight': 0.15,
    'hp_start_epoch': 6,
    # 'src_paper_weight': 0.2,
    # 'src_paper_ramp_epochs': 5,
    'uv_weight': 0.1,
    'uv_start_epoch': 3,
    'uv_ramp_epochs': 3,
    'grad_clip': 1.0,

    # NEW 3D flattening losses for progressive ramping for continuous smoothing
    # the goal here was to try and smooth out the "lumps" --> it kinda worked not really though
    'area_weight': 0.02, #weak
    'area_start_epoch': 3,# needs to start early to smooth asap
    'area_ramp_epochs': 27,
    'area_max_multiplier': 10.0, # By 30: 0.02 × 10 = 0.2

    'laplacian_weight': 0.01,
    'laplacian_start_epoch': 3, # same starting early
    'laplacian_ramp_epochs': 27,
    'laplacian_max_multiplier': 10.0, # By 30: 0.01 × 10 = 0.1

    'use_curvature_fold': False,

    # strong coverage for no cropping
    'src_paper_weight': 0.2, #.2
    'src_paper_ramp_epochs': 10,

    'zip_path': '/content/drive/MyDrive/renders.zip',
    'data_dir': '/content/dataset'
}

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# Local copy for A100 optimization
# I kept burning through credits and had to buy more so i thought this would be better time efficient
LOCAL_ZIP = '/content/renders_temp.zip'
if os.path.exists(CONFIG.get('zip_path', '')):
    print("\nCopying to local...")
    shutil.copy(CONFIG['zip_path'], LOCAL_ZIP)
    print("Extracting zip...")
    os.system(f'unzip -q {LOCAL_ZIP} -d {CONFIG["data_dir"]}')
    os.remove(LOCAL_ZIP)
    print("Dataset ready")
    EXTRACTED_ROOT = f"{CONFIG['data_dir']}/renders/synthetic_data_pitch_sweep"
else:
    print("Dataset already good")
    EXTRACTED_ROOT = CONFIG.get('data_dir', './renders/synthetic_data_pitch_sweep')



# IMPORTS
from model import make_documentunwarp  
from final.dataset_loader import get_dataloaders

# not using anymore
#from pytorch_msssim import ms_ssim, ssim

print("Imports done")


# DENORMALIZE FUNCTION
def denormalize(img):
    #Denormalize from ImageNet normalization to [0,1] range
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    return (img * std + mean).clamp(0, 1)



# IMPORTANT LOSS FUNCTIONS

# def compute_ssim_map(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> torch.Tensor:
#     return km.ssim(pred, target, window_size=window_size, max_val=data_range)
def compute_ssim_map(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute SSIM map (fallback when pytorch-msssim unavailable)."""
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
    # Masked SSIM loss --> weights SSIM by mask pred/target should be denormalized [0,1] images
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        mask = mask[:, 0:1]

    mask_sum = mask.sum() + 1e-8
    mask_norm = mask / mask_sum

    # Compute SSIM map
    ssim_map = compute_ssim_map(pred, target, data_range=data_range)
    ssim_map = ssim_map.mean(dim=1, keepdim=True)  # Average across channels

    # Weighted SSIM
    masked_ssim = (ssim_map * mask_norm).sum()

    return 1.0 - masked_ssim


def masked_ssim_value(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    # Return masked SSIM value in [0,1]
    return 1.0 - masked_ssim_loss(pred, target, mask, data_range)

def expand_mask_region(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius is None or radius <= 0:
        return mask
    k = int(radius) * 2 + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=int(radius))

def extract_document_region(batch, channel_idx=0):
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
    
    # Determine polarity from CENTER (not global mean)
    if center_mean < 0.5:
        output_mask = (border < thresh).float()
    else:
        output_mask = (border > thresh).float()
    output_mask = F.max_pool2d(output_mask, kernel_size=11, stride=1, padding=5)
    return output_mask


def warp_mask(mask_in: torch.Tensor, grid: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    if mask_in.ndim != 4:
        mask_in = mask_in.unsqueeze(1) if mask_in.ndim == 3 else mask_in

    # We use zeros padding to ensure regions outside the grid don't contribute to loss
    warped_mask = F.grid_sample(
        mask_in.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners
    )

    if warped_mask.shape[-1] > 256:
        warped_mask = F.avg_pool2d(warped_mask, kernel_size=3, stride=1, padding=1)

    return torch.clamp(warped_mask, 0.0, 1.0)


def warp_det(uv: torch.Tensor) -> torch.Tensor:
    if uv.ndim != 4 or uv.shape[1] < 2:
        raise ValueError(f"uv must be [B,>=2,H,W], got {tuple(uv.shape)}")
    b, _, h, w = uv.shape
    u = uv[:, 0]
    v = uv[:, 1]

    # Finite differences
    du_dx = (u[:, :, 1:] - u[:, :, :-1]) * float(max(w - 1, 1))
    dv_dx = (v[:, :, 1:] - v[:, :, :-1]) * float(max(w - 1, 1))
    du_dy = (u[:, 1:, :] - u[:, :-1, :]) * float(max(h - 1, 1))
    dv_dy = (v[:, 1:, :] - v[:, :-1, :]) * float(max(h - 1, 1))

    # Crop to matching dimensions
    du_dx_i = du_dx[:, 1:, :]
    dv_dx_i = dv_dx[:, 1:, :]
    du_dy_i = du_dy[:, :, 1:]
    dv_dy_i = dv_dy[:, :, 1:]

    return du_dx_i * dv_dy_i - du_dy_i * dv_dx_i


def fold_loss_with_curvature(uv: torch.Tensor, det_eps: float = 0.2, downscale: int = 2, clip: float = 10.0):
    """
    Fold loss with curvature weighting - FIXED VERSION
    Penalizes folds (negative Jacobian determinant) more heavily in high-curvature regions
    """
    # Downsample UV
    uv = F.avg_pool2d(uv, kernel_size=downscale, stride=downscale) if downscale > 1 else uv

    # Jacobian determinant
    det = warp_det(uv)

    # Laplacian magnitude (FIXED - no F.laplacian())
    u, v = uv[:, 0], uv[:, 1]
    u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')
    v_pad = F.pad(v, (1, 1, 1, 1), mode='replicate')

    laplacian_u = (u_pad[:, 1:-1, :-2] + u_pad[:, 1:-1, 2:] +
                   u_pad[:, :-2, 1:-1] + u_pad[:, 2:, 1:-1] -
                   4 * u_pad[:, 1:-1, 1:-1])
    laplacian_v = (v_pad[:, 1:-1, :-2] + v_pad[:, 1:-1, 2:] +
                   v_pad[:, :-2, 1:-1] + v_pad[:, 2:, 1:-1] -
                   4 * v_pad[:, 1:-1, 1:-1])

    # Match dimensions with determinant tensor
    h_det, w_det = det.shape[1:]
    curvature = torch.abs(laplacian_u[:, :h_det, :w_det]) + torch.abs(laplacian_v[:, :h_det, :w_det])

    # Hinge loss
    eps = float(max(det_eps, 1e-3))
    hinge = F.relu(eps - det) / eps

    # Weight fold penalty by curvature (1.0 + curvature)
    weighted_loss = torch.clamp((hinge * (1.0 + curvature)), 0.0, clip).mean()

    fold_frac = (det <= 0.0).float().mean()

    return weighted_loss, fold_frac


def tv_smoothness(uv: torch.Tensor) -> torch.Tensor:
    diff_h = torch.diff(uv, dim=-1)
    diff_v = torch.diff(uv, dim=-2)

    return (diff_h.pow(2).mean() + diff_v.pow(2).mean()).sqrt()


def gaussian_kernal1(device: torch.device, sigma: float = 1.5) -> torch.Tensor:
    ksize = int(2 * round(3 * sigma) + 1)
    x = torch.arange(ksize, device=device).float() - (ksize - 1) / 2
    kernel = torch.exp(-x.pow(2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(y: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    # sigma is the standard deviation, k_size is the window
    return kf.gaussian_blur2d(y, kernel_size=(5, 5), sigma=(sigma, sigma))


def highpass_gradient_loss(rect: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    y_r, y_g = rgb_to_grayscale(rect), rgb_to_grayscale(gt)

    # Define Sobel kernels
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=rect.device).float().view(1,1,3,3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=rect.device).float().view(1,1,3,3)

    # 3. Get gradients (edges)
    grad_rx = F.conv2d(y_r, kx, padding=1)
    grad_ry = F.conv2d(y_r, ky, padding=1)
    grad_gx = F.conv2d(y_g, kx, padding=1)
    grad_gy = F.conv2d(y_g, ky, padding=1)

    # 4. Charbonnier penalty on the difference of edge magnitudes
    diff_x = grad_rx - grad_gx
    diff_y = grad_ry - grad_gy
    loss = torch.sqrt(diff_x**2 + diff_y**2 + eps**2)

    return (loss * mask).mean()


# NEW 3D FLATTENING LOSSES

def compute_local_area(uv: torch.Tensor) -> torch.Tensor:
    """
    Compute local area scaling from UV field.
    Area = |det(Jacobian)| tells us how much each region is stretched/compressed.
    """
    det = warp_det(uv)
    return det.abs()


def area_preservation_loss(uv: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Penalize non-uniform area scaling.
    This prevents "lumps" by forcing the warp to stretch uniformly.
    """
    area = compute_local_area(uv)  # |det(J)| -> [B, H-2, W-2]

    if mask is not None:
        # Get the actual spatial dimensions of area (after Jacobian computation)
        area_h, area_w = area.shape[1:]

        # Ensure mask is [B, 1, H, W]
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # Downsample mask to match area tensor dimensions
        if mask.shape[2:] != (area_h, area_w):
            mask_down = F.interpolate(mask.float(), size=(area_h, area_w), mode='area')
            mask_down = mask_down[:, 0]  # [B, H, W]
        else:
            mask_down = mask[:, 0] if mask.ndim == 4 else mask

        # Compute mean area on paper region
        area_masked = area * mask_down
        mean_area = area_masked.sum() / (mask_down.sum() + 1e-8)

        # Penalize deviation from mean
        area_var = ((area - mean_area).abs() * mask_down).mean()
    else:
        # No mask: just minimize variance
        mean_area = area.mean()
        area_var = (area - mean_area).abs().mean()

    return area_var


def laplacian_smoothness_loss(uv: torch.Tensor) -> torch.Tensor:
    """
    Laplacian smoothness: penalize local curvature in the UV field.
    This smooths out "lumpy" warps by encouraging smooth transitions.

    Laplacian = ∇²u = (u[i-1] + u[i+1] + u[j-1] + u[j+1] - 4*u[i,j])
    """
    u, v = uv[:, 0], uv[:, 1]

    # Pad for boundary handling
    u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')
    v_pad = F.pad(v, (1, 1, 1, 1), mode='replicate')

    # Laplacian (discrete approximation)
    laplacian_u = (
        u_pad[:, 1:-1, :-2] +    # left
        u_pad[:, 1:-1, 2:] +     # right
        u_pad[:, :-2, 1:-1] +    # top
        u_pad[:, 2:, 1:-1] -     # bottom
        4 * u_pad[:, 1:-1, 1:-1] # center
    )

    laplacian_v = (
        v_pad[:, 1:-1, :-2] +
        v_pad[:, 1:-1, 2:] +
        v_pad[:, :-2, 1:-1] +
        v_pad[:, 2:, 1:-1] -
        4 * v_pad[:, 1:-1, 1:-1]
    )

    # L2 norm of Laplacian (smoothness measure)
    laplacian_loss = (laplacian_u ** 2 + laplacian_v ** 2).mean()

    return laplacian_loss


def curvature_weighted_fold_loss(
    uv: torch.Tensor,
    det_eps: float = 0.2,
    downscale: int = 2,
    clip: float = 10.0,
) -> tuple:
    """
    Fold loss penalizes folds in high-curvature regions

    Idea is all of the lumps create high curvature so preventing folds
    in these regions, the model is forced to stretch rather than project them flat
    """
    # Downsample UV
    uv = F.avg_pool2d(uv, kernel_size=downscale, stride=downscale) if downscale > 1 else uv

    # Jacobian determinant
    det = warp_det(uv)
    fold_frac = (det <= 0.0).float().mean()

    # Laplacian magnitude
    u, v = uv[:, 0], uv[:, 1]
    u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')
    v_pad = F.pad(v, (1, 1, 1, 1), mode='replicate')

    laplacian_u = (u_pad[:, 1:-1, :-2] + u_pad[:, 1:-1, 2:] +
                   u_pad[:, :-2, 1:-1] + u_pad[:, 2:, 1:-1] -
                   4 * u_pad[:, 1:-1, 1:-1])
    laplacian_v = (v_pad[:, 1:-1, :-2] + v_pad[:, 1:-1, 2:] +
                   v_pad[:, :-2, 1:-1] + v_pad[:, 2:, 1:-1] -
                   4 * v_pad[:, 1:-1, 1:-1])

    # Curvature magnitude
    h_det, w_det = det.shape[1:]
    curvature = torch.sqrt(laplacian_u[:, :h_det, :w_det] ** 2 +
                          laplacian_v[:, :h_det, :w_det] ** 2)

    # Normalize to 0,1
    curvature_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-8)

    # Weight fold penalty by curvature (1.0 + 2.0*curvature)
    # High curvature regions get 3x stronger penalty
    weights = 1.0 + 2.0 * curvature_norm

    # Weighted loss
    eps = float(max(det_eps, 1e-3))
    hinge = F.relu(eps - det) / eps
    hinge = hinge * hinge  # ^2

    weighted_hinge = hinge * weights
    loss = torch.clamp(weighted_hinge, 0.0, clip).mean()

    return loss, fold_frac


print("Loss functions defined (including new 3D flattening)")



# MODEL + DATA SETUP
device = torch.device('cuda')
print(f"\n GPU: {torch.cuda.get_device_name(0)}")

torch.backends.cudnn.benchmark = True
try:
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
except:
    pass

model = make_documentunwarp(
    backbone=CONFIG['backbone'],
    pretrained=True,
    uv_mode="residual",  #absolute
    max_disp=CONFIG['max_disp'],
    warp_mode=CONFIG['warp_mode'],
    pred_downscale=CONFIG['pred_downscale'],
    two_stage_warp=CONFIG['two_stage_warp'],
    coarse_stride=CONFIG['coarse_stride'],
    coarse_disp_scale=CONFIG['coarse_disp_scale'],
    fine_disp_scale=CONFIG['fine_disp_scale'],
    use_attention=False,
    use_uv_bias=False,
    use_post_refine=False,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {num_params:,} parameters")

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
scaler = GradScaler()

accum_steps = max(1, int(math.ceil(float(CONFIG['eff_batch']) / float(CONFIG['batch_size']))))
print(f"Gradient accumulation steps: {accum_steps}")

print("\nLoading dataset")
train_loader, val_loader = get_dataloaders(
    data_dir=EXTRACTED_ROOT,
    batch_size=CONFIG['batch_size'],
    img_size=(CONFIG['img_size'], CONFIG['img_size']),
    use_border=True,
    use_uv=True,
    use_depth=False,
    num_workers=CONFIG['num_workers']
)
print(f"Train: {len(train_loader)} batches")
print(f"Val:   {len(val_loader)} batches")

# METRIC VALIDATION TEST (suggested by chat lol)
print("\nTesting metric calculation")
test_batch = next(iter(val_loader))
test_gt = test_batch['ground_truth'][:2].to(device)
test_mask = extract_document_region(test_batch)[:2].to(device)

with torch.no_grad():
    # Denormalize GT
    test_gt_denorm = denormalize(test_gt)

    # Self-comparison (should be ~1.0)
    self_ssim = masked_ssim_value(test_gt_denorm, test_gt_denorm, test_mask, data_range=1.0)
    print(f"  Masked SSIM (GT vs GT): {self_ssim.item():.4f} (should be ~1.0)")

    if self_ssim.item() < 0.95:
        print("WARNING: Self-SSIM is low, check denormalization!")
    else:
        print("Metric calculation verified")

checkpoint_dir = Path('/content/checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)


# VISUALIZATION HELPER for live training monitoring
def visualize_results(epoch):
    model.eval()
    batch = next(iter(val_loader))

    with torch.no_grad():
        rgb = batch['rgb'][:2].to(device)
        gt = batch['ground_truth'][:2].to(device)
        mask = batch['border'][:2].to(device)

        with autocast():
            pred = model(rgb, pred_downscale=CONFIG['pred_downscale'], return_uv_pred=True)
            rectified = pred['rectified']

    rgb_vis = denormalize(rgb).cpu()
    pred_vis = denormalize(rectified).cpu()
    gt_vis = denormalize(gt).cpu()
    mask_cpu = mask.cpu()
    pred_masked_vis = pred_vis * mask_cpu

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(2):
        axes[i, 0].imshow(rgb_vis[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Input (Warped)', fontweight='bold')
        axes[i, 1].imshow(pred_vis[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title('Raw Prediction', fontweight='bold')
        axes[i, 2].imshow(pred_masked_vis[i].permute(1, 2, 0).numpy())
        axes[i, 2].set_title('Cleaned (Masked)', fontweight='bold')
        axes[i, 3].imshow(gt_vis[i].permute(1, 2, 0).numpy())
        axes[i, 3].set_title('Target (GT)', fontweight='bold')
        for j in range(4): axes[i, j].axis('off')

    plt.suptitle(f'Epoch {epoch} - Results (with 3D Flattening)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# TRAINING LOOP
print("\n" + "="*70)
print("STARTING TRAINING (Geometry + Text + 3D Flattening)")
print("="*70)

train_losses, val_losses, train_ssims, val_ssims = [], [], [], []
best_val_loss = float('inf')
best_val_ssim = 0.0
patience_counter = 0
start_time = time.time()

for epoch in range(1, CONFIG['num_epochs'] + 1):
    #TRAINING
    model.train()
    epoch_loss, epoch_ssim = 0.0, 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']}", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for bi, batch in enumerate(pbar):
        rgb = batch['rgb'].to(device, non_blocking=True)
        gt = batch['ground_truth'].to(device, non_blocking=True)

        mask = extract_document_region(batch)
        mask = mask.to(device)

        with autocast():
            pred = model(rgb, pred_downscale=CONFIG['pred_downscale'], return_uv_pred=True)
            rectified = pred['rectified']
            uv = pred['uv']
            grid = pred['grid']
            uv_pred = pred.get('uv_pred', None)

            # Denormalize before computing losses
            rect_denorm = denormalize(rectified)
            gt_denorm = denormalize(gt)

            # Warp mask to output space
            mask_src_soft = warp_mask(mask.to(dtype=torch.float32), grid, align_corners=True)
            mask_out = (mask_src_soft > 0.5).to(dtype=rect_denorm.dtype)

            # RECONSTRUCTION LOSS (masked SSIM on denormalized images)
            loss_recon = masked_ssim_loss(rect_denorm, gt_denorm, mask_out, data_range=1.0)

            # ****PROGRESSIVE LOSS SCHEDULE****
            # TV smoothness (starts epoch 3)
            tv_w = 0.0
            loss_tv = torch.tensor(0.0, device=device)
            if epoch >= CONFIG['tv_start_epoch']:
                tv_ramp = min(1.0, float(epoch - CONFIG['tv_start_epoch'] + 1) / 3.0)
                tv_w = CONFIG['tv_weight'] * tv_ramp
                uv_for_tv = uv_pred if (uv_pred is not None and CONFIG['pred_downscale'] > 1) else uv
                loss_tv = tv_smoothness(uv_for_tv)

            # Fold loss (starts epoch 6)
            loss_fold = torch.tensor(0.0, device=device)
            if epoch >= CONFIG['fold_start_epoch']:
                # Standard fold loss (worked before)
                loss_fold, fold_frac = fold_loss_with_curvature(
                    uv,
                    det_eps=CONFIG['fold_det_eps'],
                    downscale=CONFIG['fold_downscale'],
                    clip=CONFIG['fold_clip'],
                )

            # High pass loss (starts epoch 6 as befoire)
            loss_hp = torch.tensor(0.0, device=device)
            if epoch >= CONFIG['hp_start_epoch']:
                loss_hp = highpass_gradient_loss(rect_denorm, gt_denorm, mask_out)

            # Source paper coverage
            src_ramp = min(1.0, float(epoch) / CONFIG['src_paper_ramp_epochs'])
            loss_src = ((1.0 - mask_src_soft) ** 2).mean() * (CONFIG['src_paper_weight'] * src_ramp)

            # UV supervision (starts epoch 3)
            loss_uv = torch.tensor(0.0, device=device)
            if 'uv' in batch and epoch >= CONFIG['uv_start_epoch']:
                gt_uv = batch['uv'].to(device)
                gt_uv_denorm = denormalize(gt_uv)
                gt_uv_coords = gt_uv_denorm[:, :2]

                if gt_uv_coords.shape[2:] != uv.shape[2:]:
                    gt_uv_coords = F.interpolate(gt_uv_coords, size=uv.shape[2:], mode='bilinear', align_corners=True)

                uv_mask = mask if mask.shape[1] == 1 else mask[:, 0:1]
                if uv_mask.shape[2:] != gt_uv_coords.shape[2:]:
                    uv_mask = F.interpolate(uv_mask.float(), size=gt_uv_coords.shape[2:], mode='bilinear', align_corners=True)
                uv_mask = uv_mask.expand_as(gt_uv_coords)

                uv_ramp = min(1.0, float(epoch - CONFIG['uv_start_epoch'] + 1) / CONFIG['uv_ramp_epochs'])
                uv_diff = (uv - gt_uv_coords).abs()
                uv_mask_sum = uv_mask.sum() + 1e-8
                loss_uv = ((uv_diff * uv_mask).sum() / uv_mask_sum) * uv_ramp * CONFIG['uv_weight']

            # AREA PRESERVATION (starts epoch 3, ramp up)
            loss_area = torch.tensor(0.0, device=device)
            if epoch >= CONFIG['area_start_epoch']:
                #starts at 1x, reaches max_multipkler by end
                epochs_since_start = epoch - CONFIG['area_start_epoch'] + 1
                ramp_progress = min(1.0, epochs_since_start / CONFIG['area_ramp_epochs'])
                area_multiplier = 1.0 + (CONFIG['area_max_multiplier'] - 1.0) * ramp_progress

                loss_area_raw = area_preservation_loss(uv, mask_out)
                loss_area = loss_area_raw * area_multiplier

            # LAPLACIAN SMOOTHNESS (starts epoch 3, same ramp up)
            loss_laplacian = torch.tensor(0.0, device=device)
            if epoch >= CONFIG['laplacian_start_epoch']:
                # starts at 1x, reaches max_multipkler by end
                epochs_since_start = epoch - CONFIG['laplacian_start_epoch'] + 1
                ramp_progress = min(1.0, epochs_since_start / CONFIG['laplacian_ramp_epochs'])
                lap_multiplier = 1.0 + (CONFIG['laplacian_max_multiplier'] - 1.0) * ramp_progress

                uv_for_smooth = uv_pred if (uv_pred is not None and CONFIG['pred_downscale'] > 1) else uv
                loss_laplacian_raw = laplacian_smoothness_loss(uv_for_smooth)
                loss_laplacian = loss_laplacian_raw * lap_multiplier

            # Total loss metric
            raw_loss = (
                loss_recon
                + tv_w * loss_tv
                + (CONFIG['fold_weight'] * loss_fold if epoch >= CONFIG['fold_start_epoch'] else 0.0)
                + (CONFIG['hp_weight'] * loss_hp if epoch >= CONFIG['hp_start_epoch'] else 0.0)
                + loss_src
                + loss_uv
                + (CONFIG['area_weight'] * loss_area if epoch >= CONFIG['area_start_epoch'] else 0.0)  # 🆕
                + (CONFIG['laplacian_weight'] * loss_laplacian if epoch >= CONFIG['laplacian_start_epoch'] else 0.0)  # 🆕
            )

            loss = raw_loss / float(accum_steps)

        scaler.scale(loss).backward()

        # METRICS (on denormalized images)
        with torch.no_grad():
            ssim_val = float(masked_ssim_value(rect_denorm.float(), gt_denorm.float(), mask_out.float(), data_range=1.0).item())
            epoch_ssim += ssim_val

        do_step = ((bi + 1) % accum_steps == 0) or ((bi + 1) == len(train_loader))
        if do_step:
            if CONFIG['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += raw_loss.item()

        # NaN DETECTION with EARLY STOPPING cus sometimes it would blow up and id waste credits
        if not torch.isfinite(raw_loss):
            print(f"\nNaN detected at epoch {epoch}, batch {bi}!")
            print(f"  Best checkpoint saved at epoch {epoch-1}")
            print(f"\n  Recommendation: Restart training with more conservative weights")
            raise RuntimeError("NaN loss detected - training aborted")

        # Progress bar
        info = {'loss': f'{raw_loss.item():.4f}', 'ssim': f'{ssim_val:.3f}'}
        if tv_w > 0: info['tv'] = f'{tv_w:.3f}'
        if epoch >= CONFIG['fold_start_epoch']: info['fold'] = f'{loss_fold.item():.3f}'
        if epoch >= CONFIG['hp_start_epoch']: info['hp'] = f'{loss_hp.item():.3f}'
        if loss_uv.item() > 0: info['uv'] = f'{loss_uv.item():.4f}'
        if epoch >= CONFIG['area_start_epoch']:
            # Show ramping multiplier
            eps_area = epoch - CONFIG['area_start_epoch'] + 1
            area_mult = 1.0 + (CONFIG['area_max_multiplier'] - 1.0) * min(1.0, eps_area / CONFIG['area_ramp_epochs'])
            info['area'] = f'{loss_area.item():.3f}(×{area_mult:.1f})'
        if epoch >= CONFIG['laplacian_start_epoch']:
            eps_lap = epoch - CONFIG['laplacian_start_epoch'] + 1
            lap_mult = 1.0 + (CONFIG['laplacian_max_multiplier'] - 1.0) * min(1.0, eps_lap / CONFIG['laplacian_ramp_epochs'])
            info['lap'] = f'{loss_laplacian.item():.3f}(×{lap_mult:.1f})'
        pbar.set_postfix(info)

    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_ssim = epoch_ssim / len(train_loader)
    train_losses.append(avg_train_loss)
    train_ssims.append(avg_train_ssim)

    #VALIDATION
    model.eval()
    val_loss, val_ssim = 0.0, 0.0

    with torch.no_grad(), autocast():
        for batch in val_loader:
            rgb = batch['rgb'].to(device)
            gt = batch['ground_truth'].to(device)

            mask = extract_document_region(batch)
            mask = mask.to(device)

            pred = model(rgb, pred_downscale=CONFIG['pred_downscale'], return_uv_pred=True)
            rectified = pred['rectified']
            grid = pred['grid']

            # Denormalize
            rect_denorm = denormalize(rectified)
            gt_denorm = denormalize(gt)

            mask_src_soft = warp_mask(mask.to(dtype=torch.float32), grid, align_corners=True)
            mask_out = (mask_src_soft > 0.5).to(dtype=rect_denorm.dtype)

            loss_recon = masked_ssim_loss(rect_denorm, gt_denorm, mask_out, data_range=1.0)
            val_loss += loss_recon.item()
            val_ssim += float(masked_ssim_value(rect_denorm.float(), gt_denorm.float(), mask_out.float(), data_range=1.0).item())

    avg_val_loss = val_loss / len(val_loader)
    avg_val_ssim = val_ssim / len(val_loader)
    val_losses.append(avg_val_loss)
    val_ssims.append(avg_val_ssim)

    scheduler.step()

    #LOGGING
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{CONFIG['num_epochs']}")
    print(f"{'='*70}")
    print(f"Train Loss: {avg_train_loss:.4f} | Train SSIM: {avg_train_ssim:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f} | Val SSIM:   {avg_val_ssim:.4f}")
    print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

    #one thing about this is that it likes to cheat so this isnt super helpful
    if avg_val_ssim > 0.85:
        print(f"Quality: EXCELLENT")
    elif avg_val_ssim > 0.75:
        print(f"Quality: GOOD")
    elif avg_val_ssim > 0.65:
        print(f"Quality: FAIR")
    else:
        print(f"Quality: NEEDS IMPROVEMENT")

    # CHECKPOINTING
    is_best_loss = avg_val_loss < best_val_loss
    is_best_ssim = avg_val_ssim > best_val_ssim

    if is_best_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': avg_val_loss,
            'val_ssim': avg_val_ssim,
            'config': CONFIG
        }, checkpoint_dir / 'best_model.pth')
        print(f"****NEW BEST MODEL (loss)")
    else:
        patience_counter += 1

    if is_best_ssim:
        best_val_ssim = avg_val_ssim
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_ssim': avg_val_ssim,
            'config': CONFIG
        }, checkpoint_dir / 'best_ssim_model.pth')
        print(f"****NEW BEST MODEL (SSIM)!")

    if epoch % CONFIG['save_every'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': CONFIG
        }, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')

    # VIZ
    if epoch % 5 == 0 or epoch == 1:
        print("Generating vis")
        visualize_results(epoch)

    #early stop
    if patience_counter >= CONFIG['early_stopping_patience']:
        print(f"\nEarly stopping at epoch {epoch}")
        break


#FINAL MODEL & PLOTS
print("\n" + "="*70)
print("SAVING FINAL MODEL")
print("="*70)

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_ssims': train_ssims,
    'val_ssims': val_ssims,
    'best_val_loss': best_val_loss,
    'best_val_ssim': best_val_ssim,
    'config': CONFIG
}, checkpoint_dir / 'final_model.pth')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(train_losses, 'o-', label='Train', linewidth=2, markersize=3)
axes[0].plot(val_losses, 's-', label='Val', linewidth=2, markersize=3)
axes[0].axhline(best_val_loss, color='r', linestyle='--', alpha=0.5, label=f'Best ({best_val_loss:.3f})')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(train_ssims, 'o-', label='Train', linewidth=2, markersize=3)
axes[1].plot(val_ssims, 's-', label='Val', linewidth=2, markersize=3)
axes[1].axhline(best_val_ssim, color='g', linestyle='--', alpha=0.5, label=f'Best ({best_val_ssim:.3f})')
axes[1].axhline(0.85, color='green', linestyle=':', alpha=0.3)
axes[1].axhline(0.75, color='orange', linestyle=':', alpha=0.3)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('SSIM')
axes[1].set_title('SSIM (Higher = Better)', fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(checkpoint_dir / 'training_curves.png', dpi=150)
plt.show()

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Best Val SSIM: {best_val_ssim:.4f}")
print(f"Total Time: {(time.time()-start_time)/60:.1f} minutes")
print("\n3D Flattening Losses:")
print(f"  Area Preservation: Activated at epoch {CONFIG['area_start_epoch']}")
print(f"  Laplacian Smoothness: Activated at epoch {CONFIG['laplacian_start_epoch']}")
print(f"  Curvature-Weighted Fold: {'Enabled' if CONFIG['use_curvature_fold'] else 'Disabled'}")

try:
    from google.colab import files
    files.download(str(checkpoint_dir / 'best_model.pth'))
    files.download(str(checkpoint_dir / 'best_ssim_model.pth'))
    files.download(str(checkpoint_dir / 'training_curves.png'))
except:
    print("\nFiles saved to checkpoint directory")

print("\nTraining complete!")