"""
MedDINO-VISTA3D: Complete Implementation (Single File)
======================================================

Hybrid architecture combining:
- MedDINOv3: DINOv2 ViT encoder + multi-scale token aggregation
- VISTA3D: Class embeddings + per-class binary heads

No LLM, no point-prompts, fully automatic segmentation.

Usage:
    # Test model architecture
    python meddino_vista3d_complete.py --mode test
    
    # Quick test training (3+2 epochs, 20 cases)
    python meddino_vista3d_complete.py --mode quick_test
    
    # Full training (100+50 epochs, all cases)
    python meddino_vista3d_complete.py --mode full

Author: Based on MedDINOv3 and VISTA3D papers
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Disable xFormers (uses more memory than standard PyTorch attention)
os.environ["XFORMERS_DISABLED"] = "1"
# Disable xFormers for RTX 5090 compatibility (capability 12.0 not fully supported)
os.environ["XFORMERS_DISABLED"] = "1"


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Paths
    "kits23_dir": "./kits23/dataset",
    "output_dir": "./output/meddino_vista3d",
    
    # Model
    "num_classes": 4,  # background, kidney, tumor, cyst
    "backbone": "dinov2_vitb14",
    
    # Training (Memory-optimized for RTX 5090 32GB)
    "batch_size": 1,  # Physical batch size (use gradient accumulation for larger effective batch)
    "accumulation_steps": 4,  # Effective batch size = 1 * 4 = 4
    "num_epochs_stage1": 100,  # Frozen encoder
    "num_epochs_stage2": 50,   # Fine-tuned encoder
    "lr_stage1": 1e-4,
    "lr_stage2": 1e-5,
    "weight_decay": 1e-5,
    
    # Data (Aggressively reduced for memory safety)
    "patch_size": (70, 140, 140),  # (D, H, W) - multiples of 14, minimal size for memory
    "num_samples_per_volume": 2,  # Samples per volume
    "val_split": 0.2,
    
    # Model
    "deep_supervision": False,  # Disable to save memory (uses 3x less activation memory)
    
    # Early stopping
    "early_stopping_patience": 15,  # Stop if no improvement for N epochs
    
    # Quick test mode
    "quick_epochs_stage1": 3,
    "quick_epochs_stage2": 2,
    "quick_cases": 20,
    
    # Labels
    "labels": {
        "background": 0,
        "kidney": 1,
        "tumor": 2,
        "cyst": 3
    }
}


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MedDINOEncoder(nn.Module):
    """
    MedDINOv3-style encoder using DINOv2 backbone with multi-scale aggregation.
    Processes 3D volumes slice-by-slice (2.5D approach).
    """
    def __init__(
        self, 
        backbone: str = "dinov2_vitb14", 
        feature_dim: int = 256, 
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # Load pretrained DINOv2 backbone
        print(f"Loading {backbone} backbone...")
        self.vit = torch.hub.load(
            'facebookresearch/dinov2', 
            backbone,
            pretrained=True,
            verbose=False
        )
        self.hidden_dim = 768  # ViT-B hidden dimension
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("ViT backbone frozen")
                
        # Multi-scale projection heads (0-indexed: layers 2, 5, 8, 11 out of 12 total)
        self.scale_layers = [2, 5, 8, 11]
        self.proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU()
            ) for _ in self.scale_layers
        ])
        
        # Feature pyramid fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 3D aggregation across depth dimension
        self.depth_aggregator = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) - 3D CT volume
            
        Returns:
            features: (B, C, D, H/14, W/14) - 3D feature volume
        """
        B, C, D, H, W = x.shape
        assert C == 1, f"Expected 1 channel, got {C}"
        
        # Prepare slices for ViT (needs 3 channels)
        x = x.squeeze(1)  # (B, D, H, W)
        x = x.view(B * D, 1, H, W)
        x = x.repeat(1, 3, 1, 1)  # (B*D, 3, H, W) - ViT expects RGB
        
        # Extract multi-scale features from ViT
        with torch.set_grad_enabled(not self._is_frozen()):
            features = self.vit.get_intermediate_layers(
                x, 
                n=self.scale_layers, 
                reshape=True
            )
        
        # Project and fuse multi-scale features
        h, w = features[0].shape[2], features[0].shape[3]
        multi_scale = []
        
        for i, feat in enumerate(features):
            # Ensure same spatial size
            if feat.shape[2:] != (h, w):
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            
            # Project: (B*D, 768, h, w) -> (B*D, feature_dim, h, w)
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B * D * h * w, self.hidden_dim)
            proj = self.proj_heads[i](feat_flat)
            proj = proj.reshape(B * D, h, w, -1).permute(0, 3, 1, 2)
            multi_scale.append(proj)
        
        # Concatenate and fuse: (B*D, C*4, h, w) -> (B*D, C, h, w)
        fused = torch.cat(multi_scale, dim=1)
        fused = self.fusion(fused)
        
        # Reshape to 3D volume: (B*D, C, h, w) -> (B, C, D, h, w)
        _, C_out, h, w = fused.shape
        fused_3d = fused.reshape(B, D, C_out, h, w).permute(0, 2, 1, 3, 4)
        
        # 3D aggregation across depth
        features_3d = self.depth_aggregator(fused_3d)
        
        return features_3d
    
    def _is_frozen(self) -> bool:
        """Check if ViT backbone is frozen"""
        return not next(self.vit.parameters()).requires_grad


class VISTA3DDecoder(nn.Module):
    """
    VISTA3D-style automatic segmentation decoder.
    Uses class embeddings and per-class binary heads.
    """
    def __init__(
        self, 
        in_channels: int = 256, 
        num_classes: int = 4,
        deep_supervision: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        # Learnable class embeddings (key VISTA3D innovation)
        self.class_embeddings = nn.Embedding(num_classes, in_channels)
        
        # Upsampling blocks (feature_dim -> feature_dim/2 -> feature_dim/4 -> feature_dim/8)
        self.up1 = self._make_up_block(in_channels, in_channels // 2)       # 2x
        self.up2 = self._make_up_block(in_channels // 2, in_channels // 4)  # 4x
        self.up3 = self._make_up_block(in_channels // 4, in_channels // 8)  # 8x
        
        # Per-class binary segmentation heads
        self.class_heads = nn.ModuleList([
            nn.Conv3d(in_channels // 8, 1, kernel_size=1)
            for _ in range(num_classes)
        ])
        
        # Deep supervision heads (optional)
        if deep_supervision:
            self.ds_head1 = nn.ModuleList([
                nn.Conv3d(in_channels // 2, 1, kernel_size=1)
                for _ in range(num_classes)
            ])
            self.ds_head2 = nn.ModuleList([
                nn.Conv3d(in_channels // 4, 1, kernel_size=1)
                for _ in range(num_classes)
            ])
        
    def _make_up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create upsampling block"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, D, H, W) from encoder
            
        Returns:
            logits: (B, num_classes, D, H', W') or tuple if deep_supervision
        """
        # Upsample progressively
        x1 = self.up1(features)  # (B, C/2, D*2, H*2, W*2)
        x2 = self.up2(x1)        # (B, C/4, D*4, H*4, W*4)
        x3 = self.up3(x2)        # (B, C/8, D*8, H*8, W*8)
        
        # Get per-class outputs from final resolution
        outputs = []
        for head in self.class_heads:
            outputs.append(head(x3))
        logits = torch.cat(outputs, dim=1)  # (B, num_classes, D*8, H*8, W*8)
        
        # Deep supervision outputs (if training)
        if self.deep_supervision and self.training:
            ds1_outputs = [head(x1) for head in self.ds_head1]
            ds1 = torch.cat(ds1_outputs, dim=1)
            
            ds2_outputs = [head(x2) for head in self.ds_head2]
            ds2 = torch.cat(ds2_outputs, dim=1)
            
            return logits, ds1, ds2
        
        return logits


class MedDINOVISTA3D(nn.Module):
    """
    Complete MedDINO-VISTA3D hybrid model.
    
    Architecture:
        Input (B, 1, D, H, W) 
        -> MedDINO Encoder (2.5D ViT + 3D aggregation)
        -> VISTA3D Decoder (class embeddings + binary heads)
        -> Output (B, num_classes, D, H, W)
    """
    def __init__(
        self, 
        num_classes: int = 4,
        freeze_encoder: bool = True,
        backbone: str = "dinov2_vitb14",
        deep_supervision: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        # Encoder: MedDINOv3-style
        self.encoder = MedDINOEncoder(
            backbone=backbone,
            feature_dim=256,
            freeze_backbone=freeze_encoder
        )
        
        # Decoder: VISTA3D-style
        self.decoder = VISTA3DDecoder(
            in_channels=256,
            num_classes=num_classes,
            deep_supervision=deep_supervision
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) - 3D CT volume
            
        Returns:
            logits: (B, num_classes, D, H, W)
                or tuple (logits, ds1, ds2) if deep_supervision and training
        """
        # Encode
        features = self.encoder(x)
        
        # Decode
        outputs = self.decoder(features)
        
        # Resize to match input spatial dimensions
        if isinstance(outputs, tuple):
            # Deep supervision case
            logits, ds1, ds2 = outputs
            logits = F.interpolate(logits, size=x.shape[2:], mode='trilinear', align_corners=False)
            ds1 = F.interpolate(ds1, size=x.shape[2:], mode='trilinear', align_corners=False)
            ds2 = F.interpolate(ds2, size=x.shape[2:], mode='trilinear', align_corners=False)
            return logits, ds1, ds2
        else:
            logits = outputs
            if logits.shape[2:] != x.shape[2:]:
                logits = F.interpolate(logits, size=x.shape[2:], mode='trilinear', align_corners=False)
            return logits
    
    def unfreeze_encoder(self, num_layers: int = 4):
        """Unfreeze last N layers of ViT encoder for fine-tuning"""
        # Unfreeze ViT parameters
        params = list(self.encoder.vit.parameters())
        for param in params[-num_layers:]:
            param.requires_grad = True
        print(f"Unfroze last {num_layers} layers of encoder")


# ============================================================================
# DATASET
# ============================================================================

class KiTS23Dataset(Dataset):
    """KiTS23 dataset with patch sampling"""
    
    def __init__(
        self, 
        data_dicts: List[Dict], 
        patch_size: Tuple[int, int, int],
        num_samples: int = 2,
        is_train: bool = True
    ):
        self.data_dicts = data_dicts
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data_dicts) * self.num_samples
    
    def __getitem__(self, idx):
        # Get volume index
        vol_idx = idx // self.num_samples
        data_dict = self.data_dicts[vol_idx]
        
        # Load volume
        image = nib.load(data_dict["image"]).get_fdata()
        label = nib.load(data_dict["label"]).get_fdata()
        
        # Normalize image
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Sample patch
        patch_img, patch_lbl = self._sample_patch(image, label)
        
        # Convert to tensor
        patch_img = torch.from_numpy(patch_img).float().unsqueeze(0)  # (1, D, H, W)
        patch_lbl = torch.from_numpy(patch_lbl).long().unsqueeze(0)   # (1, D, H, W)
        
        return {"image": patch_img, "label": patch_lbl}
    
    def _sample_patch(self, image, label):
        """Sample random patch from volume"""
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # Handle case where volume is smaller than patch
        if d < pd or h < ph or w < pw:
            # Pad volume
            pad_d = max(0, pd - d)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            d, h, w = image.shape
        
        # Random crop
        if self.is_train:
            d_start = np.random.randint(0, max(1, d - pd + 1))
            h_start = np.random.randint(0, max(1, h - ph + 1))
            w_start = np.random.randint(0, max(1, w - pw + 1))
        else:
            # Center crop for validation
            d_start = (d - pd) // 2
            h_start = (h - ph) // 2
            w_start = (w - pw) // 2
        
        patch_img = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        patch_lbl = label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        return patch_img, patch_lbl


def get_dataloaders(config, max_cases=None):
    """Create train and validation dataloaders"""
    data_dir = Path(config["kits23_dir"])
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    
    if max_cases:
        cases = cases[:max_cases]
    
    # Create data dicts
    data_dicts = []
    for case in cases:
        img_path = case / "imaging.nii.gz"
        seg_path = case / "segmentation.nii.gz"
        if img_path.exists() and seg_path.exists():
            data_dicts.append({"image": str(img_path), "label": str(seg_path)})
    
    # Split
    val_size = int(len(data_dicts) * config["val_split"])
    train_files = data_dicts[val_size:]
    val_files = data_dicts[:val_size]
    
    print(f"Training cases: {len(train_files)}")
    print(f"Validation cases: {len(val_files)}")
    
    # Create datasets
    train_dataset = KiTS23Dataset(
        train_files, 
        config["patch_size"],
        num_samples=config["num_samples_per_volume"],
        is_train=True
    )
    
    val_dataset = KiTS23Dataset(
        val_files,
        config["patch_size"],
        num_samples=1,
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class HybridLoss(nn.Module):
    """Hybrid loss: Dice + Cross Entropy + Deep Supervision"""
    
    def __init__(self, ds_weights=[1.0, 0.5, 0.25]):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ds_weights = ds_weights
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for multi-class"""
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.squeeze(1).long(), num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_onehot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, outputs, target):
        """
        outputs: logits or (logits, ds1, ds2)
        target: (B, 1, D, H, W)
        """
        if isinstance(outputs, tuple):
            # Deep supervision
            logits, ds1, ds2 = outputs
            predictions = [logits, ds1, ds2]
        else:
            predictions = [outputs]
        
        total_loss = 0
        for i, pred in enumerate(predictions):
            # Resize target to match prediction
            if pred.shape[2:] != target.shape[2:]:
                target_resized = F.interpolate(
                    target.float(), 
                    size=pred.shape[2:], 
                    mode='nearest'
                ).long()
            else:
                target_resized = target
            
            # Cross Entropy
            ce_loss = self.ce(pred, target_resized.squeeze(1))
            
            # Dice Loss
            dice_loss = self.dice_loss(pred, target_resized)
            
            # Combined
            loss = ce_loss + dice_loss
            
            # Weighted by deep supervision
            weight = self.ds_weights[i] if i < len(self.ds_weights) else 1.0
            total_loss += weight * loss
        
        return total_loss


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_dice(pred, target):
    """Compute per-class and mean Dice scores"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    dice_per_class = {}
    class_names = ['kidney', 'tumor', 'cyst']
    
    for i, c in enumerate(range(1, 4)):  # Skip background
        pred_c = (pred == c)
        target_c = (target == c)
        
        if target_c.sum() == 0 and pred_c.sum() == 0:
            dice_per_class[class_names[i]] = 1.0
        elif target_c.sum() == 0:
            dice_per_class[class_names[i]] = 0.0
        else:
            intersection = (pred_c & target_c).sum()
            dice = 2 * intersection / (pred_c.sum() + target_c.sum())
            dice_per_class[class_names[i]] = dice
    
    dice_per_class['mean'] = np.mean(list(dice_per_class.values()))
    return dice_per_class


def train_epoch(model, loader, optimizer, loss_fn, scaler, device, accumulation_steps=1):
    """Train for one epoch with gradient accumulation"""
    model.train()
    epoch_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad()
    
    for i, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        with autocast('cuda'):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss = loss / accumulation_steps  # Scale loss for accumulation
        
        scaler.scale(loss).backward()
        
        # Only step optimizer every accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps  # Unscale for logging
        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
    
    return epoch_loss / len(loader)


def validate(model, loader, loss_fn, device):
    """Validate model with per-class metrics"""
    model.eval()
    val_loss = 0
    dice_scores = {'kidney': [], 'tumor': [], 'cyst': [], 'mean': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Use autocast for validation too (float16 for xFormers compatibility)
            with autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            
            # Compute Dice
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            pred = torch.argmax(outputs, dim=1, keepdim=True)
            dice = compute_dice(pred, labels)
            
            # Accumulate per-class scores
            for key in dice_scores.keys():
                dice_scores[key].append(dice[key])
    
    # Average all scores
    avg_dice = {k: np.mean(v) for k, v in dice_scores.items()}
    return val_loss / len(loader), avg_dice


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(config, stage1_epochs, stage2_epochs, max_cases=None):
    """Main training loop"""
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("=" * 60)
    print("MedDINO-VISTA3D Training")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Data
    train_loader, val_loader = get_dataloaders(config, max_cases)
    
    # Model
    model = MedDINOVISTA3D(
        num_classes=config["num_classes"],
        freeze_encoder=True,
        backbone=config["backbone"],
        deep_supervision=config.get("deep_supervision", False)  # Use config setting
    ).to(device)
    
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable (stage 1): {trainable:,}")
    
    # Loss and optimizer
    loss_fn = HybridLoss()
    scaler = GradScaler('cuda')
    
    best_dice = 0
    patience_counter = 0
    patience = config.get("early_stopping_patience", 15)
    
    # ========================================================================
    # STAGE 1: Frozen Encoder
    # ========================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Training Decoder Only (Encoder Frozen)")
    print("=" * 60)
    print(f"Early stopping patience: {patience} epochs")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["lr_stage1"],
        weight_decay=config["weight_decay"]
    )
    
    for epoch in range(stage1_epochs):
        print(f"\nEpoch {epoch+1}/{stage1_epochs}")
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            accumulation_steps=config.get("accumulation_steps", 1)
        )
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Dice - Kidney: {val_dice['kidney']:.4f} | Tumor: {val_dice['tumor']:.4f} | Cyst: {val_dice['cyst']:.4f} | Mean: {val_dice['mean']:.4f}")
        
        # Save best (based on mean Dice)
        if val_dice['mean'] > best_dice:
            best_dice = val_dice['mean']
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model_stage1.pth")
            print(f"✅ Best model saved (Mean Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping triggered - no improvement for {patience} epochs")
                break
    
    # ========================================================================
    # STAGE 2: Fine-tune Encoder
    # ========================================================================
    if stage2_epochs > 0:
        print("\n" + "=" * 60)
        print("STAGE 2: Fine-tuning Encoder")
        print("=" * 60)
        print(f"Early stopping patience: {patience} epochs")
        
        model.unfreeze_encoder(num_layers=4)
        
        total, trainable = count_parameters(model)
        print(f"Trainable (stage 2): {trainable:,}")
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr_stage2"],
            weight_decay=config["weight_decay"]
        )
        
        # Reset patience counter for stage 2
        patience_counter = 0
        
        for epoch in range(stage2_epochs):
            print(f"\nEpoch {epoch+1}/{stage2_epochs}")
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, scaler, device,
                accumulation_steps=config.get("accumulation_steps", 1)
            )
            val_loss, val_dice = validate(model, val_loader, loss_fn, device)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Dice - Kidney: {val_dice['kidney']:.4f} | Tumor: {val_dice['tumor']:.4f} | Cyst: {val_dice['cyst']:.4f} | Mean: {val_dice['mean']:.4f}")
            
            if val_dice['mean'] > best_dice:
                best_dice = val_dice['mean']
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best_model_stage2.pth")
                print(f"✅ Best model saved (Mean Dice: {best_dice:.4f})")
            else:
                patience_counter += 1
                print(f"⏳ No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\n⚠️ Early stopping triggered - no improvement for {patience} epochs")
                    break
    
    # Final save
    torch.save(model.state_dict(), output_dir / "final_model.pth")
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete! Best Dice: {best_dice:.4f}")
    print("=" * 60)


# ============================================================================
# MODEL TEST
# ============================================================================

def test_model():
    """Test model with dummy data"""
    print("=" * 60)
    print("Testing MedDINO-VISTA3D Architecture")
    print("=" * 60)
    
    # Create model
    model = MedDINOVISTA3D(
        num_classes=4,
        freeze_encoder=True,
        deep_supervision=True
    )
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Dummy input (small size for testing)
    x = torch.randn(1, 1, 32, 128, 128).to(device)
    print(f"Input shape: {x.shape}")
    
    model.train()
    with torch.no_grad():
        outputs = model(x)
    
    if isinstance(outputs, tuple):
        logits, ds1, ds2 = outputs
        print(f"Output shape: {logits.shape}")
        print(f"DS1 shape: {ds1.shape}")
        print(f"DS2 shape: {ds2.shape}")
    else:
        print(f"Output shape: {outputs.shape}")
    
    # Test eval mode
    model.eval()
    with torch.no_grad():
        outputs_eval = model(x)
    
    if isinstance(outputs_eval, tuple):
        print("\n⚠️ Warning: Deep supervision should be disabled in eval mode")
    else:
        print(f"\nEval output shape: {outputs_eval.shape}")
    
    print("\n✅ Model test passed!")
    print("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MedDINO-VISTA3D: Complete Training Pipeline")
    parser.add_argument("--mode", type=str, default="test",
                       choices=["test", "quick_test", "full"],
                       help="Mode: test (architecture only), quick_test (3+2 epochs), full (100+50 epochs)")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        print("Running architecture test...")
        test_model()
    elif args.mode == "quick_test":
        print("Running QUICK TEST mode (3+2 epochs, 20 cases)")
        train(
            CONFIG,
            stage1_epochs=CONFIG["quick_epochs_stage1"],
            stage2_epochs=CONFIG["quick_epochs_stage2"],
            max_cases=CONFIG["quick_cases"]
        )
    else:
        print("Running FULL training (100+50 epochs, all cases)")
        train(
            CONFIG,
            stage1_epochs=CONFIG["num_epochs_stage1"],
            stage2_epochs=CONFIG["num_epochs_stage2"],
            max_cases=None
        )


if __name__ == "__main__":
    main()
