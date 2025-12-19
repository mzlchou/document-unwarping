# -*- coding: utf-8 -*-
"""
IMPROVED A100 TRAINING SCRIPT
Optimized for high-speed training and credit conservation
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

# 1. SETUP
!pip install -q timm pytorch-msssim tensorboard
from google.colab import drive
drive.mount('/content/drive')

# Configuration (A100 optimized)
CONFIG = {
    'batch_size': 32,          
    'img_size': 512,
    'num_epochs': 60,
    'learning_rate': 1e-4,
    'num_workers': 8,          
    'flow_scale': 2.0,         
    'save_every': 5,
    'early_stopping_patience': 15,  
    'zip_path': '/content/drive/MyDrive/renders.zip',
    'data_dir': '/content/dataset'
}

# 2. FAST LOCAL COPY
LOCAL_ZIP = '/content/renders_temp.zip'
if os.path.exists(CONFIG['zip_path']):
    print("🚀 Copying to local NVMe...")
    shutil.copy(CONFIG['zip_path'], LOCAL_ZIP)
    print("📂 Extracting...")
    !unzip -q {LOCAL_ZIP} -d {CONFIG['data_dir']}
    os.remove(LOCAL_ZIP)
    print("✅ Dataset ready on fast storage")

EXTRACTED_ROOT = f"{CONFIG['data_dir']}/renders/synthetic_data_pitch_sweep"

# 3. IMPORTS
from model import DocumentUnwarpModel
from dataset_loader import get_dataloaders
from pytorch_msssim import ssim
import torchvision.models as models

# 4. LOSSES
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16]
        self.vgg = vgg.eval().cuda()
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        return F.mse_loss(self.vgg(pred), self.vgg(target))

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.perceptual = PerceptualLoss()
    
    def forward(self, pred, target, mask):
        l1_raw = self.l1(pred, target)
        mask_expanded = mask.expand_as(l1_raw)
        l1_loss = (l1_raw * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        p_loss = self.perceptual(pred, target)
        return (0.2 * l1_loss) + (0.5 * ssim_loss) + (0.3 * p_loss)

# 5. MODEL & OPTIMIZER
device = torch.device('cuda')
model = DocumentUnwarpModel(flow_scale=CONFIG['flow_scale']).to(device)
criterion = CombinedLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
scaler = GradScaler()  # Mixed precision (FP16/FP32)

# DATA LOADING
train_loader, val_loader = get_dataloaders(
    data_dir=EXTRACTED_ROOT,
    batch_size=CONFIG['batch_size'],
    img_size=(CONFIG['img_size'], CONFIG['img_size']),
    use_border=True,
    num_workers=CONFIG['num_workers']
)

# 6. VISUALIZATION HELPER
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    return (img * std + mean).clamp(0, 1)

def visualize_results(epoch):
    model.eval()
    batch = next(iter(val_loader))
    with torch.no_grad(), autocast():
        rgb = batch['rgb'][:2].to(device)
        gt = batch['ground_truth'][:2].to(device)
        pred, _, _ = model(rgb)
    
    rgb_vis = denormalize(rgb).cpu()
    pred_vis = denormalize(pred).cpu()
    gt_vis = denormalize(gt).cpu()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(2):
        axes[i, 0].imshow(rgb_vis[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Input', fontweight='bold')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(pred_vis[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title('Prediction', fontweight='bold')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(gt_vis[i].permute(1, 2, 0).numpy())
        axes[i, 2].set_title('Ground Truth', fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(checkpoint_dir / f'epoch_{epoch:03d}.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

# 7. TRAINING LOOP
checkpoint_dir = Path('/content/drive/MyDrive/unwarp_checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

print("\n" + "="*60 + "\n🔥 A100 TRAINING START\n" + "="*60)

train_losses, val_losses, train_ssims, val_ssims = [], [], [], []
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1, CONFIG['num_epochs'] + 1):
    model.train()
    epoch_loss, epoch_ssim = 0.0, 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
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
        with torch.no_grad():
            epoch_ssim += ssim(rectified, gt, data_range=1.0).item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Stats calculation
    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_ssim = epoch_ssim / len(train_loader)
    
    # Validation
    model.eval()
    val_loss, val_ssim_score = 0.0, 0.0
    with torch.no_grad(), autocast():
        for batch in val_loader:
            rgb = batch['rgb'].to(device)
            gt = batch['ground_truth'].to(device)
            mask = batch['border'].to(device)
            rectified, _, _ = model(rgb)
            val_loss += criterion(rectified, gt, mask).item()
            val_ssim_score += ssim(rectified, gt, data_range=1.0).item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_ssim = val_ssim_score / len(val_loader)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_ssims.append(avg_train_ssim)
    val_ssims.append(avg_val_ssim)
    
    scheduler.step()
    
    # Best Model Save
    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({'model_state_dict': model.state_dict()}, checkpoint_dir / 'best_model.pth')
        print(f" ✨ Epoch {epoch}: NEW BEST! Val SSIM: {avg_val_ssim:.4f}")

    if epoch % 5 == 0 or epoch == 1:
        visualize_results(epoch)

    # Early Stopping
    if not is_best:
        patience_counter += 1
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"⏹ Early stopping at epoch {epoch}")
            break

# ============================================================
# FINAL WRAP-UP (RUNS ONCE)
# ============================================================
print("\n" + "="*60 + "\n🎉 TRAINING COMPLETE\n" + "="*60)

# Plotting Curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(train_losses, label='Train'); axes[0].plot(val_losses, label='Val')
axes[0].set_title('Loss'); axes[0].legend()
axes[1].plot(train_ssims, label='Train'); axes[1].plot(val_ssims, label='Val')
axes[1].set_title('SSIM'); axes[1].legend()
plt.savefig(checkpoint_dir / 'training_curves.png')
plt.show()

# Trigger Downloads
from google.colab import files
try:
    print("📥 Downloading artifacts...")
    files.download(str(checkpoint_dir / 'best_model.pth'))
    files.download(str(checkpoint_dir / 'training_curves.png'))
except Exception as e:
    print(f"⚠️ Download error: {e}. Check Drive.")

# Auto-Kill to save money
print("\n⏰ Terminating runtime in 60s to save A100 credits...")
time.sleep(60)
from google.colab import runtime
runtime.unassign()