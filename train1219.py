# -*- coding: utf-8 -*-
"""
FINAL A100 TRAINING SCRIPT: OPTIMIZED FOR TEXT CLARITY
Combines A100 speed + Old Script stability + Text readability
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
# STEP 3: CONFIGURATION (A100 OPTIMIZED + TEXT CLARITY)
# ============================================================
CONFIG = {
    # A100 Settings
    'batch_size': 32,
    'img_size': 512,
    'num_epochs': 50,
    'learning_rate': 1e-4,         # Same as old script
    'num_workers': 8,
    'flow_scale': 1.0,             # ✅ CRITICAL: 1.0 like old script
    'save_every': 5,
    'early_stopping_patience': 12,


    # Loss Weights (Rebalanced for text clarity)
    'l1_weight': 0.7,              # 70% - Heavy lifting for geometry
    'ssim_weight': 0.2,            # 20% - Structural validation
    'perceptual_weight': 0.1,      # 10% - Text clarity boost
    'use_perceptual': True,        # Enable perceptual loss


    # Paths
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
import torchvision.models as models
print("✓ Imports successful")


# ============================================================
# STEP 6: FIXED PERCEPTUAL LOSS (FOR TEXT CLARITY)
# ============================================================
class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss with proper denormalization.
    This helps preserve high-frequency details like text edges.
    """
    def __init__(self):
        super().__init__()
        # Use VGG16 features (captures edges and textures)
        vgg = models.vgg16(pretrained=True).features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False


    def forward(self, pred, target):
        # ✅ CRITICAL FIX: Denormalize before VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)


        pred_denorm = (pred * std + mean).clamp(0, 1)
        target_denorm = (target * std + mean).clamp(0, 1)


        pred_features = self.vgg(pred_denorm)
        target_features = self.vgg(target_denorm)
        return F.mse_loss(pred_features, target_features)


# ============================================================
# STEP 7: ENHANCED GEOMETRIC LOSS (WITH TEXT CLARITY)
# ============================================================
class EnhancedGeometricLoss(nn.Module):
    """
    Balanced loss for geometry + text clarity:
    - L1 (70%): Heavy lifting for geometric correction
    - SSIM (20%): Structural validation
    - Perceptual (10%): Text sharpness and detail preservation
    """
    def __init__(self, l1_weight=0.7, ssim_weight=0.2, perceptual_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.l1 = nn.L1Loss(reduction='none')


    def forward(self, pred, target, mask, perceptual_fn=None):
        total_loss = 0.0


        # 1. Masked L1 Loss (Heavy lifter - 70%)
        if self.l1_weight > 0:
            l1_raw = self.l1(pred, target)
            mask_expanded = mask.expand_as(l1_raw)
            l1_loss = (l1_raw * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
            total_loss += self.l1_weight * l1_loss


        # 2. SSIM Loss on DENORMALIZED images (Structural - 20%)
        if self.ssim_weight > 0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)


            pred_denorm = (pred * std + mean).clamp(0, 1)
            target_denorm = (target * std + mean).clamp(0, 1)


            ssim_val = ssim(pred_denorm, target_denorm, data_range=1.0)
            ssim_loss = 1 - ssim_val
            total_loss += self.ssim_weight * ssim_loss


        # 3. Perceptual Loss (Text clarity - 10%)
        if self.perceptual_weight > 0 and perceptual_fn is not None:
            perc_loss = perceptual_fn(pred, target)
            total_loss += self.perceptual_weight * perc_loss


        return total_loss


print("✓ Loss function defined")


# ============================================================
# STEP 8: MODEL & DATA SETUP
# ============================================================
device = torch.device('cuda')
print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")


model = DocumentUnwarpModel(flow_scale=CONFIG['flow_scale']).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model: {num_params:,} parameters")


# Initialize losses
criterion = EnhancedGeometricLoss(
    l1_weight=CONFIG['l1_weight'],
    ssim_weight=CONFIG['ssim_weight'],
    perceptual_weight=CONFIG['perceptual_weight']
)


# Perceptual loss (for text clarity)
perceptual_loss = None
if CONFIG['use_perceptual']:
    try:
        perceptual_loss = PerceptualLoss().to(device)
        print("✓ Perceptual loss initialized (for text clarity)")
    except Exception as e:
        print(f"⚠️  Perceptual loss failed: {e}")
        print("   Continuing with L1+SSIM only")
        CONFIG['perceptual_weight'] = 0.0


optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
scaler = GradScaler()


print(f"✓ Loss: L1({CONFIG['l1_weight']:.1f}) + SSIM({CONFIG['ssim_weight']:.1f}) + Perceptual({CONFIG['perceptual_weight']:.1f})")


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
# STEP 9: VISUALIZATION HELPER (WITH MASKING!)
# ============================================================
def denormalize(img):
    """Denormalize for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    return (img * std + mean).clamp(0, 1)


# def visualize_results(epoch):
#     """Visualize predictions with background masking."""
#     model.eval()
#     batch = next(iter(val_loader))


#     with torch.no_grad():
#         rgb = batch['rgb'][:2].to(device)
#         gt = batch['ground_truth'][:2].to(device)
#         mask = batch['border'][:2].to(device)
#         pred, _, _ = model(rgb)


#         # ✅ CRITICAL: Apply mask to remove background (grass)
#         pred_masked = pred * mask


#     # Denormalize
#     rgb_vis = denormalize(rgb).cpu()
#     pred_vis = denormalize(pred_masked).cpu()
#     gt_vis = denormalize(gt).cpu()


#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     for i in range(2):
#         axes[i, 0].imshow(rgb_vis[i].permute(1, 2, 0).numpy())
#         axes[i, 0].set_title('Input (Warped)', fontweight='bold', fontsize=12)
#         axes[i, 0].axis('off')


#         axes[i, 1].imshow(pred_vis[i].permute(1, 2, 0).numpy())
#         axes[i, 1].set_title('Prediction (Clean)', fontweight='bold', fontsize=12)
#         axes[i, 1].axis('off')


#         axes[i, 2].imshow(gt_vis[i].permute(1, 2, 0).numpy())
#         axes[i, 2].set_title('Ground Truth', fontweight='bold', fontsize=12)
#         axes[i, 2].axis('off')


#     plt.suptitle(f'Epoch {epoch} - Text Clarity Enhanced', fontsize=16, fontweight='bold')
#     plt.tight_layout()


#     # Save to both local and Drive
#     save_path = checkpoint_dir / f'epoch_{epoch:03d}.png'
#     plt.savefig(save_path, dpi=100, bbox_inches='tight')
#     shutil.copy(save_path, drive_backup_dir / f'epoch_{epoch:03d}.png')


#     plt.show()
#     plt.close()


# print("✓ Visualization ready")
def visualize_results(epoch):
    """Visualizes predictions with improved clarity and no gray fog."""
    model.eval()
    batch = next(iter(val_loader))

    with torch.no_grad():
        rgb = batch['rgb'][:2].to(device)
        gt = batch['ground_truth'][:2].to(device)
        mask = batch['border'][:2].to(device)
        pred, _, _ = model(rgb)

    # 1. Denormalize EVERYTHING first to get real colors
    rgb_vis = denormalize(rgb).cpu()
    pred_raw_vis = denormalize(pred).cpu() # Raw model output
    gt_vis = denormalize(gt).cpu()

    # 2. Apply mask only for the "Cleaned" view
    # This keeps the background black instead of gray
    mask_cpu = mask.cpu()
    pred_masked_vis = pred_raw_vis * mask_cpu

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(2):
        # Input
        axes[i, 0].imshow(rgb_vis[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Input (Warped)', fontweight='bold')

        # Raw Prediction (No Mask - See everything!)
        axes[i, 1].imshow(pred_raw_vis[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title('Raw Prediction (Unmasked)', fontweight='bold')

        # Cleaned Prediction (Masked)
        axes[i, 2].imshow(pred_masked_vis[i].permute(1, 2, 0).numpy())
        axes[i, 2].set_title('Cleaned (Masked)', fontweight='bold')

        # Ground Truth
        axes[i, 3].imshow(gt_vis[i].permute(1, 2, 0).numpy())
        axes[i, 3].set_title('Target (GT)', fontweight='bold')

        for j in range(4): axes[i, j].axis('off')

    plt.suptitle(f'Epoch {epoch} - Visibility Enhanced', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 10: TRAINING LOOP
# ============================================================
print("\n" + "="*70)
print("🔥 STARTING A100 TRAINING (Text Clarity Enhanced)")
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
            loss = criterion(rectified, gt, mask, perceptual_fn=perceptual_loss)


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
            val_loss += criterion(rectified, gt, mask, perceptual_fn=perceptual_loss).item()


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


    # Quality assessment
    if avg_val_ssim > 0.85:
        print(f"🎯 Quality: EXCELLENT - Text should be readable")
    elif avg_val_ssim > 0.75:
        print(f"✅ Quality: GOOD - Text mostly readable")
    elif avg_val_ssim > 0.65:
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
            'train_ssim': avg_train_ssim,
            'val_ssim': avg_val_ssim,
            'config': CONFIG
        }


        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        # shutil.copy(checkpoint_dir / 'best_model.pth',
                  #  drive_backup_dir / 'best_model.pth')
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
        # shutil.copy(checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth',
                  #  drive_backup_dir / f'checkpoint_epoch_{epoch:03d}.pth')


    # ========== VISUALIZATION ==========
    if epoch % 5 == 0 or epoch == 1:
        print("📊 Generating visualization...")
        visualize_results(epoch)


    # ========== EARLY STOPPING ==========
    if patience_counter >= CONFIG['early_stopping_patience']:
        print(f"\n⏹  Early stopping at epoch {epoch}")
        break


# ============================================================
# STEP 11: FINAL MODEL & PLOTS
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
# shutil.copy(checkpoint_dir / 'final_model.pth', drive_backup_dir / 'final_model.pth')


# Training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))


axes[0].plot(train_losses, 'o-', label='Train', linewidth=2, markersize=3)
axes[0].plot(val_losses, 's-', label='Val', linewidth=2, markersize=3)
axes[0].axhline(best_val_loss, color='r', linestyle='--', alpha=0.5, label=f'Best ({best_val_loss:.3f})')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss (Lower = Better)', fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3)


axes[1].plot(train_ssims, 'o-', label='Train', linewidth=2, markersize=3)
axes[1].plot(val_ssims, 's-', label='Val', linewidth=2, markersize=3)
axes[1].axhline(0.85, color='g', linestyle='--', alpha=0.5, label='Excellent (0.85)')
axes[1].axhline(0.75, color='orange', linestyle='--', alpha=0.5, label='Good (0.75)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('SSIM')
axes[1].set_title('SSIM (Higher = Better Text Quality)', fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)


plt.tight_layout()
plt.savefig(checkpoint_dir / 'training_curves.png', dpi=150)
# shutil.copy(checkpoint_dir / 'training_curves.png',
          #  drive_backup_dir / 'training_curves.png')
plt.show()


# ============================================================
# STEP 12: SUMMARY & DOWNLOAD
# ============================================================
print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Best Val SSIM: {max(val_ssims):.4f}")
print(f"Total Time:    {(time.time()-start_time)/60:.1f} minutes")
print(f"\n📝 Text Readability Target: SSIM > 0.75")
print(f"   Your Result: {'✅ ACHIEVED' if max(val_ssims) > 0.75 else '⚠️  NEEDS MORE TRAINING'}")


from google.colab import files
try:
    files.download(str(checkpoint_dir / 'best_model.pth'))
    files.download(str(checkpoint_dir / 'training_curves.png'))
    print("\n✓ Files downloaded")
except:
    print("\n✓ Files saved to Drive backup")


print("\n⏰ Terminating in 60s to save A100 credits...")
time.sleep(60)
from google.colab import runtime
runtime.unassign()
