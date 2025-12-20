# -*- coding: utf-8 -*-
"""
FINAL A100 TRAINING SCRIPT: OPTIMIZED FOR TEXT CLARITY
Combines A100 speed + Improved architecture + Advanced losses
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


# ============================================================
# STEP 1: INSTALL & SETUP
# ============================================================
try:
    get_ipython().run_line_magic('pip', 'install -q timm pytorch-msssim tensorboard')
except:
    pass

try:
    from google.colab import drive
    drive.mount('/content/drive')
except:
    pass


# ============================================================
# STEP 2: CLONE GITHUB REPO (if needed)
# ============================================================
# Uncomment if needed:
GITHUB_USERNAME = "mzlchou"
GITHUB_REPO = "document-unwarping"
repo_url = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
print("🔄 Cloning repository...")
if os.path.exists(GITHUB_REPO):
    !rm -rf {GITHUB_REPO}
!git clone {repo_url}
os.chdir(GITHUB_REPO)
print(f"✓ Repository cloned to: {os.getcwd()}")


# ============================================================
# STEP 3: CONFIGURATION (A100 OPTIMIZED + TEXT CLARITY)
# ============================================================
CONFIG = {
    # A100 Settings
    'batch_size': 8,
    'img_size': 512,
    'num_epochs': 50,
    'learning_rate': 8e-5,         # Better learning rate from exampletrain.py
    'weight_decay': 1e-2,
    'num_workers': 8,
    'save_every': 5,
    'early_stopping_patience': 12,
    'eff_batch': 16,               # Effective batch size for gradient accumulation

    # Model settings
    'backbone': 'convnext_tiny',
    'warp_mode': 'bicubic',         # Better for text clarity
    'pred_downscale': 2,           # Predict at lower res, upsample UV
    'two_stage_warp': True,         # Use coarse + fine warp head
    'max_disp': 0.35,
    'coarse_stride': 64,
    'coarse_disp_scale': 1.5,
    'fine_disp_scale': 0.5,

    # Loss Weights (from exampletrain.py)
    'tv_weight': 0.03,
    'fold_weight': 0.5,
    'fold_det_eps': 0.2,
    'fold_downscale': 2,
    'fold_clip': 10.0,
    'hp_weight': 0.15,              # High-pass Charbonnier for text clarity
    'src_paper_weight': 0.2,        # Source paper coverage
    'src_paper_ramp_epochs': 5,
    'grad_clip': 1.0,

    # Paths
    'zip_path': '/content/drive/MyDrive/renders.zip',
    'data_dir': '/content/dataset'
}


print("\n⚙️  Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# ============================================================
# STEP 4: FAST LOCAL COPY (if needed)
# ============================================================
LOCAL_ZIP = '/content/renders_temp.zip'
if os.path.exists(CONFIG.get('zip_path', '')):
    print("\n🚀 Copying to local NVMe...")
    shutil.copy(CONFIG['zip_path'], LOCAL_ZIP)
    print("📂 Extracting...")
    os.system(f'unzip -q {LOCAL_ZIP} -d {CONFIG["data_dir"]}')
    os.remove(LOCAL_ZIP)
    print("✅ Dataset ready")
    EXTRACTED_ROOT = f"{CONFIG['data_dir']}/renders/synthetic_data_pitch_sweep"
else:
    # Use existing dataset path
    EXTRACTED_ROOT = CONFIG.get('data_dir', './renders/synthetic_data_pitch_sweep')


# ============================================================
# STEP 5: IMPORTS
# ============================================================
from model import build_model
from dataset_loader import get_dataloaders

# MS-SSIM
try:
    from pytorch_msssim import ms_ssim
    _MS_SSIM = ms_ssim
except Exception:
    _MS_SSIM = None
    print("⚠️  pytorch-msssim not available, using fallback SSIM")

print("✓ Imports successful")


# ============================================================
# STEP 6: LOSS FUNCTIONS (from exampletrain.py)
# ============================================================

def ms_ssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Returns 1 - MS-SSIM (lower is better). pred/target expected in [0,1]."""
    if _MS_SSIM is not None:
        val = _MS_SSIM(pred, target, data_range=data_range, size_average=True)
        return 1.0 - val
    
    # fallback (not true MS-SSIM)
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))
    return 1.0 - ssim_map.mean()


def ms_ssim_value(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Return MS-SSIM value in [0,1]. pred/target expected in [0,1]."""
    if _MS_SSIM is not None:
        return _MS_SSIM(pred, target, data_range=data_range, size_average=True)
    return 1.0 - ms_ssim_loss(pred, target, data_range=data_range)


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary/soft dilation via max-pool. radius in pixels."""
    if radius is None or radius <= 0:
        return mask
    k = int(radius) * 2 + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=int(radius))


def make_paper_mask(
    batch: dict,
    mode: str = "border",
    polarity: str = "auto",
    thresh: float = 0.25,
    binarize: bool = True,
    dilate: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    if mode == "none" or ("border" not in batch):
        b, _, h, w = batch["rgb"].shape
        return torch.ones((b, 1, h, w), device=batch["rgb"].device, dtype=batch["rgb"].dtype)
    
    border = batch["border"]
    if border.ndim == 3:
        border = border.unsqueeze(1)
    if border.shape[1] > 1:
        border = border.mean(dim=1, keepdim=True)
    
    if polarity == "auto":
        b, _, h, w = border.shape
        ch = max(1, h // 8)
        cw = max(1, w // 8)
        center = border[:, :, h // 2 - ch // 2 : h // 2 + ch // 2, w // 2 - cw // 2 : w // 2 + cw // 2]
        cm = center.mean().item()
        polarity_eff = "low" if cm < 0.5 else "high"
    else:
        polarity_eff = polarity
    
    if polarity_eff == "low":
        mask = (border < thresh).to(border.dtype) if binarize else (1.0 - (border / (thresh + eps)).clamp(0.0, 1.0))
    else:
        mask = (border > thresh).to(border.dtype) if binarize else ((border - thresh) / (1.0 - thresh + eps)).clamp(0.0, 1.0)
    
    if dilate and dilate > 0:
        mask = dilate_mask(mask, int(dilate))
    return mask


def apply_mask_blend_white(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return img * mask + (1.0 - mask) * 1.0


def warp_mask_to_output(mask_in: torch.Tensor, grid: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    """Warp a source-space paper mask into output space using the predicted backward grid."""
    if mask_in.ndim != 4 or mask_in.shape[1] != 1:
        raise ValueError(f"mask_in must be [B,1,H,W], got {tuple(mask_in.shape)}")
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(f"grid must be [B,H,W,2], got {tuple(grid.shape)}")
    m = F.grid_sample(mask_in, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners)
    return m.clamp(0.0, 1.0)


def make_identity_uv01(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]
    return uv


def det_jacobian_uv01(uv01: torch.Tensor) -> torch.Tensor:
    """Resolution-normalized Jacobian det for uv01; identity det ~ 1.0."""
    if uv01.ndim != 4 or uv01.shape[1] < 2:
        raise ValueError(f"uv01 must be [B,>=2,H,W], got {tuple(uv01.shape)}")
    b, _, h, w = uv01.shape
    u = uv01[:, 0]
    v = uv01[:, 1]
    
    du_dx = (u[:, :, 1:] - u[:, :, :-1]) * float(max(w - 1, 1))
    dv_dx = (v[:, :, 1:] - v[:, :, :-1]) * float(max(w - 1, 1))
    du_dy = (u[:, 1:, :] - u[:, :-1, :]) * float(max(h - 1, 1))
    dv_dy = (v[:, 1:, :] - v[:, :-1, :]) * float(max(h - 1, 1))
    
    du_dx_i = du_dx[:, 1:, :]
    dv_dx_i = dv_dx[:, 1:, :]
    du_dy_i = du_dy[:, :, 1:]
    dv_dy_i = dv_dy[:, :, 1:]
    
    return du_dx_i * dv_dy_i - du_dy_i * dv_dx_i


def fold_loss_from_uv01(
    uv01: torch.Tensor,
    det_eps: float = 0.2,
    downscale: int = 2,
    clip: float = 10.0,
    hinge_pow: int = 2,
) -> tuple:
    """Anti-fold loss using det_jacobian_uv01. Returns (loss, fold_frac)."""
    uv = F.avg_pool2d(uv01, kernel_size=downscale, stride=downscale) if downscale > 1 else uv01
    det = det_jacobian_uv01(uv)
    fold_frac = (det <= 0.0).float().mean()
    
    eps = float(max(det_eps, 1e-3))
    hinge = F.relu(eps - det) / eps
    if hinge_pow == 2:
        hinge = hinge * hinge
    loss = torch.clamp(hinge, 0.0, clip).mean()
    return loss, fold_frac


def tv_smoothness(uv01: torch.Tensor) -> torch.Tensor:
    du = (uv01[:, :, :, 1:] - uv01[:, :, :, :-1]).abs().mean()
    dv = (uv01[:, :, 1:, :] - uv01[:, :, :-1, :]).abs().mean()
    return du + dv


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _gaussian_kernel_1d(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    k = k / k.sum()
    return k


def gaussian_blur_1ch(y: torch.Tensor) -> torch.Tensor:
    device = y.device
    k = _gaussian_kernel_1d(device, torch.float32)
    y32 = y.float()
    w_h = k.view(1, 1, 1, -1)
    y32 = F.pad(y32, (2, 2, 0, 0), mode="reflect")
    y32 = F.conv2d(y32, w_h)
    w_v = k.view(1, 1, -1, 1)
    y32 = F.pad(y32, (0, 0, 2, 2), mode="reflect")
    y32 = F.conv2d(y32, w_v)
    return y32.to(dtype=y.dtype)


def highpass_charbonnier_loss(rect: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    y_r = rgb_to_luma(rect)
    y_g = rgb_to_luma(gt)
    hp_r = y_r - gaussian_blur_1ch(y_r)
    hp_g = y_g - gaussian_blur_1ch(y_g)
    diff = hp_r - hp_g
    loss = torch.sqrt(diff * diff + (eps * eps))
    return (loss * mask).mean()


print("✓ Loss functions defined")


# ============================================================
# STEP 7: MODEL & DATA SETUP
# ============================================================
device = torch.device('cuda')
print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")

# A100 optimizations
torch.backends.cudnn.benchmark = True
try:
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
except Exception:
    pass
try:
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"
except Exception:
    pass

model = build_model(
    backbone=CONFIG['backbone'],
    pretrained=True,
    uv_mode="residual",
    max_disp=CONFIG['max_disp'],
    warp_mode=CONFIG['warp_mode'],
    pred_downscale=CONFIG['pred_downscale'],
    two_stage_warp=CONFIG['two_stage_warp'],
    coarse_stride=CONFIG['coarse_stride'],
    coarse_disp_scale=CONFIG['coarse_disp_scale'],
    fine_disp_scale=CONFIG['fine_disp_scale'],
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model: {num_params:,} parameters")

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
scaler = GradScaler()

# Gradient accumulation steps
accum_steps = max(1, int(math.ceil(float(CONFIG['eff_batch']) / float(CONFIG['batch_size']))))
print(f"✓ Gradient accumulation steps: {accum_steps}")

print("\n📊 Loading dataset...")
train_loader, val_loader = get_dataloaders(
    data_dir=EXTRACTED_ROOT,
    batch_size=CONFIG['batch_size'],
    img_size=(CONFIG['img_size'], CONFIG['img_size']),
    use_border=True,
    num_workers=CONFIG['num_workers']
)
print(f"✓ Train: {len(train_loader)} batches")
print(f"✓ Val:   {len(val_loader)} batches")

# Checkpoint Directories
checkpoint_dir = Path('/content/checkpoints')
drive_backup_dir = Path('/content/drive/MyDrive/unwarp_checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)
try:
    drive_backup_dir.mkdir(exist_ok=True, parents=True)
except:
    pass


# ============================================================
# STEP 8: VISUALIZATION HELPER
# ============================================================
def denormalize(img):
    """Denormalize for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    return (img * std + mean).clamp(0, 1)


def visualize_results(epoch):
    """Visualizes predictions with improved clarity."""
    model.eval()
    batch = next(iter(val_loader))
    
    with torch.no_grad():
        rgb = batch['rgb'][:2].to(device)
        gt = batch['ground_truth'][:2].to(device)
        mask = batch['border'][:2].to(device)
        
        with autocast():
            pred = model(rgb, pred_downscale=CONFIG['pred_downscale'])
            rectified = pred['rectified']
    
    # Denormalize
    rgb_vis = denormalize(rgb).cpu()
    pred_vis = denormalize(rectified).cpu()
    gt_vis = denormalize(gt).cpu()
    
    # Apply mask
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
    
    plt.suptitle(f'Epoch {epoch} - Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 9: TRAINING LOOP
# ============================================================
print("\n" + "="*70)
print("🔥 STARTING A100 TRAINING (Improved Architecture + Advanced Losses)")
print("="*70)

train_losses, val_losses, train_mssims, val_mssims = [], [], [], []
best_val_loss = float('inf')
patience_counter = 0
start_time = time.time()

for epoch in range(1, CONFIG['num_epochs'] + 1):
    # ========== TRAINING ==========
    model.train()
    epoch_loss, epoch_mssim = 0.0, 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    
    for bi, batch in enumerate(pbar):
        rgb = batch['rgb'].to(device, non_blocking=True)
        gt = batch['ground_truth'].to(device, non_blocking=True)
        
        # Create mask
        mask = make_paper_mask(
            batch,
            mode="border",
            polarity="auto",
            thresh=0.25,
            binarize=True,
            dilate=5,
        )
        
        with autocast():
            pred = model(rgb, pred_downscale=CONFIG['pred_downscale'], return_uv_pred=True)
            rectified = pred['rectified']
            uv01 = pred['uv01']
            grid = pred['grid']
            uv01_pred = pred.get('uv01_pred', None)
            
            rect01 = rectified.clamp(0.0, 1.0)
            gt01 = gt.clamp(0.0, 1.0)
            
            # Warp mask to output space
            mask_src_soft = warp_mask_to_output(mask.to(dtype=torch.float32), grid, align_corners=True)
            mask_out_full = (mask_src_soft > 0.5).to(dtype=rect01.dtype)
            
            # Reconstruction loss (masked MS-SSIM)
            rect_m = apply_mask_blend_white(rect01, mask_out_full)
            gt_m = apply_mask_blend_white(gt01, mask_out_full)
            loss_recon = ms_ssim_loss(rect_m, gt_m, data_range=1.0)
            
            # TV smoothness
            uv_for_tv = uv01_pred if (uv01_pred is not None and CONFIG['pred_downscale'] > 1) else uv01
            loss_tv = tv_smoothness(uv_for_tv)
            
            # Fold loss
            use_fullres_fold = True
            uv_for_fold = uv01 if (use_fullres_fold or CONFIG['pred_downscale'] <= 1 or uv01_pred is None) else uv01_pred
            loss_fold, fold_frac = fold_loss_from_uv01(
                uv_for_fold,
                det_eps=CONFIG['fold_det_eps'],
                downscale=CONFIG['fold_downscale'],
                clip=CONFIG['fold_clip'],
                hinge_pow=2,
            )
            
            # High-pass Charbonnier loss
            loss_hp = highpass_charbonnier_loss(rect01, gt01, mask_out_full)
            
            # Source paper coverage loss
            ramp = min(1.0, float(epoch) / float(max(1, CONFIG['src_paper_ramp_epochs'])))
            loss_src_paper = ((1.0 - mask_src_soft) ** 2).mean() * (CONFIG['src_paper_weight'] * ramp)
            
            # Total loss
            raw_loss = (
                loss_recon
                + CONFIG['tv_weight'] * loss_tv
                + CONFIG['fold_weight'] * loss_fold
                + CONFIG['hp_weight'] * loss_hp
                + loss_src_paper
            )
            
            loss = raw_loss / float(accum_steps)
        
        scaler.scale(loss).backward()
        
        # Metrics
        with torch.no_grad():
            msm_val = float(ms_ssim_value(rect_m.float(), gt_m.float(), data_range=1.0).item())
            epoch_mssim += msm_val
        
        do_step = ((bi + 1) % accum_steps == 0) or ((bi + 1) == len(train_loader))
        if do_step:
            if CONFIG['grad_clip'] and CONFIG['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        epoch_loss += raw_loss.item()
        pbar.set_postfix({'loss': f'{raw_loss.item():.4f}', 'ms-ssim': f'{msm_val:.3f}'})
    
    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_mssim = epoch_mssim / len(train_loader)
    train_losses.append(avg_train_loss)
    train_mssims.append(avg_train_mssim)
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss, val_mssim = 0.0, 0.0
    
    with torch.no_grad(), autocast():
        for batch in val_loader:
            rgb = batch['rgb'].to(device)
            gt = batch['ground_truth'].to(device)
            
            mask = make_paper_mask(
                batch,
                mode="border",
                polarity="auto",
                thresh=0.25,
                binarize=True,
                dilate=5,
            )
            
            pred = model(rgb, pred_downscale=CONFIG['pred_downscale'], return_uv_pred=True)
            rectified = pred['rectified']
            uv01 = pred['uv01']
            grid = pred['grid']
            
            rect01 = rectified.clamp(0.0, 1.0)
            gt01 = gt.clamp(0.0, 1.0)
            
            mask_src_soft = warp_mask_to_output(mask.to(dtype=torch.float32), grid, align_corners=True)
            mask_out_full = (mask_src_soft > 0.5).to(dtype=rect01.dtype)
            
            rect_m = apply_mask_blend_white(rect01, mask_out_full)
            gt_m = apply_mask_blend_white(gt01, mask_out_full)
            
            loss_recon = ms_ssim_loss(rect_m, gt_m, data_range=1.0)
            val_loss += loss_recon.item()
            val_mssim += float(ms_ssim_value(rect_m.float(), gt_m.float(), data_range=1.0).item())
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_mssim = val_mssim / len(val_loader)
    val_losses.append(avg_val_loss)
    val_mssims.append(avg_val_mssim)
    
    scheduler.step()
    
    # ========== LOGGING ==========
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{CONFIG['num_epochs']}")
    print(f"{'='*70}")
    print(f"Train Loss: {avg_train_loss:.4f} | Train MS-SSIM: {avg_train_mssim:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f} | Val MS-SSIM:   {avg_val_mssim:.4f}")
    print(f"LR:         {scheduler.get_last_lr()[0]:.6f}")
    
    # Quality assessment
    if avg_val_mssim > 0.85:
        print(f"🎯 Quality: EXCELLENT - Text should be readable")
    elif avg_val_mssim > 0.75:
        print(f"✅ Quality: GOOD - Text mostly readable")
    elif avg_val_mssim > 0.65:
        print(f"⚠️  Quality: FAIR - Text clarity needs improvement")
    else:
        print(f"❌ Quality: NEEDS IMPROVEMENT")
    
    # ========== CHECKPOINTING ==========
    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        patience_counter = 0
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_mssim': avg_train_mssim,
            'val_mssim': avg_val_mssim,
            'config': CONFIG
        }
        
        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        print(f"⭐ NEW BEST MODEL!")
    else:
        patience_counter += 1
    
    # Periodic checkpoints
    if epoch % CONFIG['save_every'] == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG
        }
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')
    
    # ========== VISUALIZATION ==========
    if epoch % 5 == 0 or epoch == 1:
        print("📊 Generating visualization...")
        visualize_results(epoch)
    
    # ========== EARLY STOPPING ==========
    if patience_counter >= CONFIG['early_stopping_patience']:
        print(f"\n⏹  Early stopping at epoch {epoch}")
        break


# ============================================================
# STEP 10: FINAL MODEL & PLOTS
# ============================================================
print("\n" + "="*70)
print("💾 SAVING FINAL MODEL")
print("="*70)

final_checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_mssims': train_mssims,
    'val_mssims': val_mssims,
    'best_val_loss': best_val_loss,
    'config': CONFIG
}
torch.save(final_checkpoint, checkpoint_dir / 'final_model.pth')

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(train_losses, 'o-', label='Train', linewidth=2, markersize=3)
axes[0].plot(val_losses, 's-', label='Val', linewidth=2, markersize=3)
axes[0].axhline(best_val_loss, color='r', linestyle='--', alpha=0.5, label=f'Best ({best_val_loss:.3f})')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss (Lower = Better)', fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(train_mssims, 'o-', label='Train', linewidth=2, markersize=3)
axes[1].plot(val_mssims, 's-', label='Val', linewidth=2, markersize=3)
axes[1].axhline(0.85, color='g', linestyle='--', alpha=0.5, label='Excellent (0.85)')
axes[1].axhline(0.75, color='orange', linestyle='--', alpha=0.5, label='Good (0.75)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MS-SSIM')
axes[1].set_title('MS-SSIM (Higher = Better Text Quality)', fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(checkpoint_dir / 'training_curves.png', dpi=150)
plt.show()

# ============================================================
# STEP 11: SUMMARY
# ============================================================
print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Best Val MS-SSIM: {max(val_mssims):.4f}")
print(f"Total Time:    {(time.time()-start_time)/60:.1f} minutes")
print(f"\n📝 Text Readability Target: MS-SSIM > 0.75")
print(f"   Your Result: {'✅ ACHIEVED' if max(val_mssims) > 0.75 else '⚠️  NEEDS MORE TRAINING'}")

try:
    from google.colab import files
    files.download(str(checkpoint_dir / 'best_model.pth'))
    files.download(str(checkpoint_dir / 'training_curves.png'))
    print("\n✓ Files downloaded")
except:
    print("\n✓ Files saved to checkpoint directory")

print("\n⏰ Training complete!")
