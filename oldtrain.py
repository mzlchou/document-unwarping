!nvidia-smi
import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [code]
!pip install -q timm pytorch-msssim tensorboard

# %% [code]
from google.colab import drive
drive.mount('/content/drive')

print("Google Drive mounted")

import os

GITHUB_USERNAME = "mzlchou"
GITHUB_REPO = "document-unwarping"

# Clone the repo
repo_url = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
print(f"Cloning from: {repo_url}")

if os.path.exists(GITHUB_REPO):
    !rm -rf {GITHUB_REPO}

!git clone {repo_url}

# Navigate into the repo
os.chdir(GITHUB_REPO)
print(f"Repository cloned to: {os.getcwd()}")

# Show files
print("\nFiles in repo:")
!ls -la

# %% [code]
from model import DocumentUnwarpModel
from dataset_loader import get_dataloaders, visualize_batch

print("Successfully imported model and dataset_loader from your GitHub repo!")

# %% [code]
import os

# Path to your ZIP file in Google Drive
ZIP_PATH = '/content/drive/MyDrive/renders.zip'

# Check if ZIP exists in Drive
if os.path.exists(ZIP_PATH):
    print(f"✓ Found renders.zip in Google Drive")

    # Extract it to current directory
    print("Extracting dataset...")
    !unzip -q {ZIP_PATH}

    # Set the data directory
    DATA_DIR = 'renders/synthetic_data_pitch_sweep'

    # Verify extraction worked
    if os.path.exists(DATA_DIR):
        print(f"✓ Dataset extracted successfully!")
        print(f"Dataset location: {DATA_DIR}")

        # Show what's inside
        print("\nDataset contents:")
        !ls {DATA_DIR}

        # Count files in each folder
        print("\nFile counts:")
        for folder in ['rgb', 'ground_truth', 'border', 'uv', 'depth']:
            folder_path = f"{DATA_DIR}/{folder}"
            if os.path.exists(folder_path):
                count = len(os.listdir(folder_path))
                print(f"  {folder}/: {count} files")
    else:
        print("✗ Extraction failed - check ZIP structure")
        print("Expected structure inside ZIP:")
        print("  renders/")
        print("    └── synthetic_data_pitch_sweep/")
        print("        ├── rgb/")
        print("        ├── ground_truth/")
        print("        └── ...")
else:
    print(f"✗ ZIP not found at: {ZIP_PATH}")
    print("\nPlease check:")
    print("1. Is the file named exactly 'renders.zip'?")
    print("2. Is it in the root of MyDrive (not in a subfolder)?")
    print("\nCurrent Drive contents:")
    !ls /content/drive/MyDrive/

# %% [code]
print("Testing data loading...")

train_loader, val_loader = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=2,
    img_size=(256, 256),  # Small for quick test
    use_border=True
)

print(f"✓ Train batches: {len(train_loader)}")
print(f"✓ Val batches: {len(val_loader)}")

# Visualize one batch
import matplotlib.pyplot as plt

batch = next(iter(train_loader))
visualize_batch(batch)

# %% [code]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = DocumentUnwarpModel(flow_scale=1.0).to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Model created with {num_params:,} parameters")

# Test forward pass
x = torch.randn(1, 3, 256, 256).to(device)
with torch.no_grad():
    rectified, flow, grid = model(x)

print(f"\n✓ Model test successful!")
print(f"  Input:  {x.shape}")
print(f"  Output: {rectified.shape}")
print(f"  Flow:   {flow.shape}")

# %% [code]
CONFIG = {
    'batch_size': 4,        # Adjust based on GPU (T4: 4-8, A100: 16-32)
    'img_size': 512,        # Can use 256 for faster training
    'num_epochs': 30,
    'learning_rate': 1e-4,
    'save_every': 5,        # Save checkpoint every N epochs
}

print("Training Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# %% [code]
!pip install -q pytorch-msssim

print("✓ Dependencies installed")

# Additional imports for training
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

# Check for SSIM
try:
    from pytorch_msssim import ssim
    SSIM_AVAILABLE = True
    print("✓ SSIM available - using combined L1 + SSIM loss")
except:
    SSIM_AVAILABLE = False
    print("⚠️  SSIM not available - using L1 only")

# %% [code]
# ============================================================
# COMBINED LOSS FUNCTION
# ============================================================

class CombinedLoss(nn.Module):
    """
    Combined L1 + SSIM loss with document masking.

    Why this works:
    - L1: Helps with pixel-level accuracy (30%)
    - SSIM: Focuses on structure/geometry (70%) - robust to lighting!
    - Masking: Ignores background, focuses on document
    """
    def __init__(self, l1_weight=0.3, ssim_weight=0.7):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight if SSIM_AVAILABLE else 0.0
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred, target, mask=None):
        total_loss = 0.0

        # L1 Loss with masking
        if self.l1_weight > 0:
            l1_loss = self.l1(pred, target)

            if mask is not None:
                # Apply mask to focus on document pixels only
                mask_expanded = mask.expand_as(l1_loss)  # [B, 1, H, W] → [B, 3, H, W]
                l1_loss = (l1_loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
            else:
                l1_loss = l1_loss.mean()

            total_loss += self.l1_weight * l1_loss

        # SSIM Loss (structural similarity - better for geometry!)
        if SSIM_AVAILABLE and self.ssim_weight > 0:
            ssim_val = ssim(pred, target, data_range=1.0)
            ssim_loss = 1 - ssim_val  # Convert to loss
            total_loss += self.ssim_weight * ssim_loss

        return total_loss

print("✓ Loss function defined")

# %% [code]
# ============================================================
# TRAINING SETUP
# ============================================================

# Create checkpoint directory
checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

print("="*60)
print("Loading Full Dataset")
print("="*60)

# Load FULL dataset for training (not the test batch)
train_loader, val_loader = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=CONFIG['batch_size'],
    img_size=(CONFIG['img_size'], CONFIG['img_size']),
    use_border=True,  # Enable masking!
    num_workers=2
)

print(f"✓ Train batches: {len(train_loader)}")
print(f"✓ Val batches: {len(val_loader)}")
print(f"✓ Train samples: {len(train_loader.dataset)}")
print(f"✓ Val samples: {len(val_loader.dataset)}")

print("\n" + "="*60)
print("Creating Model & Optimizer")
print("="*60)

# Create model (fresh one for training)
model = DocumentUnwarpModel(flow_scale=1.0).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model parameters: {num_params:,}")

# Loss and optimizer
criterion = CombinedLoss(l1_weight=0.3, ssim_weight=0.7)
optimizer = optim.Adam(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=1e-5
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)

print(f"✓ Loss: L1 (0.3) + SSIM (0.7)" if SSIM_AVAILABLE else "✓ Loss: L1 only")
print(f"✓ Optimizer: Adam (lr={CONFIG['learning_rate']})")
print(f"✓ Scheduler: ReduceLROnPlateau")
print("\n✅ Ready to train!")

# %% [code]
# ============================================================
# VISUALIZATION HELPER
# ============================================================

def denormalize(img):
    """Denormalize image for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    return (img * std + mean).clamp(0, 1)


def visualize_results(epoch):
    """Show model predictions vs ground truth."""
    model.eval()
    batch = next(iter(val_loader))

    with torch.no_grad():
        rgb = batch['rgb'][:2].to(device)
        gt = batch['ground_truth'][:2].to(device)
        pred, flow, _ = model(rgb)

    # Denormalize for display
    rgb_vis = denormalize(rgb).cpu()
    pred_vis = denormalize(pred).cpu()
    gt_vis = denormalize(gt).cpu()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i in range(2):
        axes[i, 0].imshow(rgb_vis[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Input (Warped)', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pred_vis[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(gt_vis[i].permute(1, 2, 0).numpy())
        axes[i, 2].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')

    plt.suptitle(f'Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(checkpoint_dir / f'epoch_{epoch:03d}.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

print("✓ Visualization helper ready")

# %% [code]
# ============================================================
# TRAINING LOOP WITH EARLY STOPPING
# ============================================================

import torch
from tqdm import tqdm

print("="*60)
print(f"🚀 Starting Training - {CONFIG['num_epochs']} Epochs")
print("="*60)

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0  # Early stopping counter

EARLY_STOPPING_PATIENCE = 10  # Stop if val_loss doesn't improve for 10 epochs

for epoch in range(1, CONFIG['num_epochs'] + 1):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{CONFIG['num_epochs']}")
    print(f"{'='*60}")

    # ===================== TRAIN =====================
    model.train()
    epoch_loss = 0.0
    train_pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(train_pbar):
        rgb = batch['rgb'].to(device)
        gt = batch['ground_truth'].to(device)
        mask = batch['border'].to(device)

        optimizer.zero_grad()
        rectified, flow, _ = model(rgb)
        loss = criterion(rectified, gt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg': f'{epoch_loss/(batch_idx+1):.4f}'
        })

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # =================== VALIDATION ===================
    model.eval()
    val_loss = 0.0
    val_pbar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in val_pbar:
            rgb = batch['rgb'].to(device)
            gt = batch['ground_truth'].to(device)
            mask = batch['border'].to(device)

            rectified, flow, _ = model(rgb)
            loss = criterion(rectified, gt, mask)
            val_loss += loss.item()
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # =================== LR SCHEDULER ===================
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # =================== CHECKPOINT ===================
    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'config': CONFIG,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    # Save checkpoints
    torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        print(f"  ✨ NEW BEST! Saved best model")
    if epoch % CONFIG['save_every'] == 0:
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')
        print(f"  ✓ Saved periodic checkpoint")

    # =================== VISUALIZATION ===================
    if epoch % 5 == 0 or epoch == 1:
        print(f"\n📸 Generating visualization...")
        visualize_results(epoch)

    # =================== EARLY STOPPING ===================
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n⏹ Early stopping triggered after {epoch} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
        break

    # =================== LOG RESULTS ===================
    print(f"\n📊 Epoch {epoch} Results:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  LR:         {current_lr:.6f}")
    print("-"*60)

# =================== TRAINING COMPLETE ===================
print("\n" + "="*60)
print("🎉 Training Complete!")
print("="*60)
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Final LR: {optimizer.param_groups[0]['lr']:.6f}")


# %% [code]
# ============================================================
# PLOT TRAINING CURVES
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss curves
axes[0].plot(train_losses, label='Train', linewidth=2, marker='o', markersize=4)
axes[0].plot(val_losses, label='Val', linewidth=2, marker='s', markersize=4)
axes[0].axhline(best_val_loss, color='r', linestyle='--', alpha=0.5,
                label=f'Best ({best_val_loss:.4f})')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Validation improvement
improvements = [val_losses[0]]
for i in range(1, len(val_losses)):
    if val_losses[i] < min(val_losses[:i]):
        improvements.append(val_losses[i])
    else:
        improvements.append(improvements[-1])

axes[1].plot(val_losses, label='Val Loss', linewidth=2, alpha=0.6)
axes[1].plot(improvements, label='Best So Far', linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Validation Loss', fontsize=12)
axes[1].set_title('Validation Progress', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(checkpoint_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Curves saved to: {checkpoint_dir / 'training_curves.png'}")


# %% [code]
# Final comparison
print("Generating final visualization...")
visualize_results(CONFIG['num_epochs'])

print("\n✅ Training complete! Next steps:")
print("  1. Download 'best_model.pth' from checkpoints/")
print("  2. Run evaluation locally")
print("  3. Create your report")


# %% [code]
from google.colab import files

# Download the best model
files.download('checkpoints/best_model.pth')
print("downloaded model")