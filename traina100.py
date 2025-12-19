# -*- coding: utf-8 -*-
"""
CORRECTED A100 TRAINING SCRIPT: GEOMETRY OPTIMIZED
Combines A100 speed with proper geometric loss + complete training pipeline
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

# ============================================================
# STEP 1: INSTALL & SETUP
# ============================================================
!pip install -q timm pytorch-msssim tensorboard
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# STEP 2: CLONE GITHUB REPO
# ============================================================
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
# STEP 3: CONFIGURATION
# ============================================================
CONFIG = {
    'batch_size': 32,              
    'img_size': 512,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'num_workers': 8,              
    'flow_scale': 1.0,             # 1.0 for better geometry
    'save_every': 5,
    'early_stopping_patience': 12,
    'zip_path': '/content/drive/MyDrive/renders.zip',
    'data_dir': '/content/dataset'
}

print("\n⚙️  Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# ============================================================
# STEP 4: FAST LOCAL COPY
# ============================================================
LOCAL_ZIP = '/content/renders_temp.zip'
if os.path.exists(CONFIG['zip_path']):
    print("\n🚀 Copying to local NVMe...")
    shutil.copy(CONFIG['zip_path'], LOCAL_ZIP)
    print("📂 Extracting...")
    !unzip -q {LOCAL_ZIP} -d {CONFIG['data_dir']}
    os.remove(LOCAL_ZIP)
    print("✅ Dataset ready")
else:
    raise FileNotFoundError(f"ZIP not found: {CONFIG['zip_path']}")

EXTRACTED_ROOT = f"{CONFIG['data_dir']}/renders/synthetic_data_pitch_sweep"

# ============================================================
# STEP 5: IMPORTS
# ============================================================
from model import DocumentUnwarpModel
from dataset_loader import get_dataloaders
from pytorch_msssim import ssim
print("✓ Imports successful")

# ============================================================
# STEP 6: CORRECTED GEOMETRIC LOSS
# ============================================================
class GeometricCombinedLoss(nn.Module):
    """
    Geometry-focused loss with proper handling of normalized images.
    30% Masked L1 + 70% SSIM
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred, target, mask):
        # 1. Masked L1 Loss (on normalized images - this is fine)
        l1_raw = self.l1(pred, target)
        mask_expanded = mask.expand_as(l1_raw)
        l1_loss = (l1_raw * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        
        # 2. SSIM Loss on DENORMALIZED images (CRITICAL FIX!)
        # Images are normalized, so we need to denormalize for SSIM
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_denorm = (pred * std + mean).clamp(0, 1)
        target_denorm = (target * std + mean).clamp(0, 1)
        
        ssim_val = ssim(pred_denorm, target_denorm, data_range=1.0)
        ssim_loss = 1 - ssim_val
        
        # 3. Combine (30% L1, 70% SSIM for geometry focus)
        return (0.3 * l1_loss) + (0.7 * ssim_loss)

print("✓ Loss function defined")

# ============================================================
# STEP 7: MODEL & DATA SETUP
# ============================================================
device = torch.device('cuda')
print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")

model = DocumentUnwarpModel(flow_scale=CONFIG['flow_scale']).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model: {num_params:,} parameters")

criterion = GeometricCombinedLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
scaler = GradScaler()

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
drive_backup_dir.mkdir(exist_ok=True, parents=True)

# ============================================================
# STEP 8: VISUALIZATION HELPER (WITH MASKING!)
# ============================================================
def denormalize(img):
    """Denormalize for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    return (img * std + mean).clamp(0, 1)

def visualize_results(epoch):
    """Visualize predictions with background masking."""
    model.eval()
    batch = next(iter(val_loader))
    
    with torch.no_grad():
        rgb = batch['rgb'][:2].to(device)
        gt = batch['ground_truth'][:2].to(device)
        mask = batch['border'][:2].to(device)
        pred, _, _ = model(rgb)
        
        # CRITICAL: Apply mask to remove background
        pred_masked = pred * mask
    
    # Denormalize
    rgb_vis = denormalize(rgb).cpu()
    pred_vis = denormalize(pred_masked).cpu()
    gt_vis = denormalize(gt).cpu()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(2):
        axes[i, 0].imshow(rgb_vis[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Input (Warped)', fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred_vis[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title('Prediction', fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(gt_vis[i].permute(1, 2, 0).numpy())
        axes[i, 2].set_title('Ground Truth', fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save to both local and Drive
    save_path = checkpoint_dir / f'epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    shutil.copy(save_path, drive_backup_dir / f'epoch_{epoch:03d}.png')
    
    plt.show()
    plt.close()

print("✓ Visualization ready")

# ============================================================
# STEP 9: TRAINING LOOP
# ============================================================
print("\n" + "="*70)
print("🔥 STARTING TRAINING")
print("="*70)

train_losses, val_losses, train_ssims, val_ssims = [], [], [], []
best_val_loss = float('inf')
patience_counter = 0
start_time = time.time()

for epoch in range(1, CONFIG['num_epochs'] + 1):
    # ========== TRAINING ==========
    model.train()
    epoch_loss, epoch_ssim = 0.0, 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']}", leave=False)
    for batch in pbar:
        rgb = batch['rgb'].to(device, non_blocking=True)
        gt = batch['ground_truth'].to(device, non_blocking=True)
        mask = batch['border'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            rectified, _, _ = model(rgb)
            loss = criterion(rectified, gt, mask)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
        # Compute SSIM for tracking (denormalized)
        with torch.no_grad():
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(rgb.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(rgb.device)
            pred_denorm = (rectified * std + mean).clamp(0, 1)
            gt_denorm = (gt * std + mean).clamp(0, 1)
            batch_ssim = ssim(pred_denorm, gt_denorm, data_range=1.0).item()
            epoch_ssim += batch_ssim
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'ssim': f'{batch_ssim:.3f}'})
    
    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_ssim = epoch_ssim / len(train_loader)
    train_losses.append(avg_train_loss)
    train_ssims.append(avg_train_ssim)
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss, val_ssim = 0.0, 0.0
    
    with torch.no_grad(), autocast():
        for batch in val_loader:
            rgb = batch['rgb'].to(device)
            gt = batch['ground_truth'].to(device)
            mask = batch['border'].to(device)
            
            rectified, _, _ = model(rgb)
            val_loss += criterion(rectified, gt, mask).item()
            
            # SSIM on denormalized
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(rgb.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(rgb.device)
            pred_denorm = (rectified * std + mean).clamp(0, 1)
            gt_denorm = (gt * std + mean).clamp(0, 1)
            val_ssim += ssim(pred_denorm, gt_denorm, data_range=1.0).item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_ssim = val_ssim / len(val_loader)
    val_losses.append(avg_val_loss)
    val_ssims.append(avg_val_ssim)
    
    scheduler.step()
    
    # ========== LOGGING ==========
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{CONFIG['num_epochs']}")
    print(f"{'='*70}")
    print(f"Train Loss: {avg_train_loss:.4f} | Train SSIM: {avg_train_ssim:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f} | Val SSIM:   {avg_val_ssim:.4f}")
    print(f"LR:         {scheduler.get_last_lr()[0]:.6f}")
    
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
            'train_ssim': avg_train_ssim,
            'val_ssim': avg_val_ssim,
            'config': CONFIG
        }
        
        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        shutil.copy(checkpoint_dir / 'best_model.pth', 
                   drive_backup_dir / 'best_model.pth')
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
        shutil.copy(checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth',
                   drive_backup_dir / f'checkpoint_epoch_{epoch:03d}.pth')
    
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
    'train_ssims': train_ssims,
    'val_ssims': val_ssims,
    'best_val_loss': best_val_loss,
    'config': CONFIG
}
torch.save(final_checkpoint, checkpoint_dir / 'final_model.pth')
shutil.copy(checkpoint_dir / 'final_model.pth', drive_backup_dir / 'final_model.pth')

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(train_losses, 'o-', label='Train', linewidth=2)
axes[0].plot(val_losses, 's-', label='Val', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(train_ssims, 'o-', label='Train', linewidth=2)
axes[1].plot(val_ssims, 's-', label='Val', linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('SSIM')
axes[1].set_title('SSIM'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(checkpoint_dir / 'training_curves.png', dpi=150)
shutil.copy(checkpoint_dir / 'training_curves.png', 
           drive_backup_dir / 'training_curves.png')
plt.show()

# ============================================================
# STEP 11: SUMMARY & DOWNLOAD
# ============================================================
print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Best Val SSIM: {max(val_ssims):.4f}")
print(f"Total Time:    {(time.time()-start_time)/60:.1f} minutes")

from google.colab import files
try:
    files.download(str(checkpoint_dir / 'best_model.pth'))
    files.download(str(checkpoint_dir / 'training_curves.png'))
except:
    print("Files saved to Drive backup")

print("\n⏰ Terminating in 60s...")
time.sleep(60)
from google.colab import runtime
runtime.unassign()