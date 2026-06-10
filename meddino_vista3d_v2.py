"""
MedDINO-VISTA3D v2: Improved Class Imbalance Handling
=====================================================

FIXES from v1:
- Added Focal Loss for tumor/cyst (was: 0.0186 tumor Dice)
- Improved positive sampling (70% pos, 30% neg for tumor/cyst)
- Class-weighted loss components
- Higher deep supervision weights for small structures

Target: Tumor Dice 0.60+, Cyst Dice 0.50+

Usage:
    python meddino_vista3d_v2.py --mode quick_test
    python meddino_vista3d_v2.py --mode full
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Paths
    "kits23_dir": "./kits23/dataset",
    "output_dir": "./output/meddino_vista3d_v2",
    
    # Model
    "num_classes": 4,
    "backbone": "dinov2_vitb14",
    
    # Training (Optimized for RTX 5090 32GB)
    "batch_size": 2,
    "num_epochs_stage1": 100,
    "num_epochs_stage2": 50,
    "lr_stage1": 1e-4,
    "lr_stage2": 1e-5,
    "weight_decay": 1e-5,
    
    # Data - IMPROVED SAMPLING for tumor/cyst (Reduced for memory)
    "patch_size": (70, 140, 140),  # Reduced from (140,224,224) to fit in VRAM
    "num_samples_per_volume": 2,  # Reduced from 4
    "positive_sample_ratio": 0.7,  # 70% patches contain tumor/cyst
    "val_split": 0.2,
    
    # Quick test
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
# MODEL ARCHITECTURE (same as v1)
# ============================================================================

class MedDINOEncoder(nn.Module):
    """MedDINOv3-style encoder"""
    def __init__(self, backbone: str = "dinov2_vitb14", feature_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        
        print(f"Loading {backbone} backbone...")
        self.vit = torch.hub.load('facebookresearch/dinov2', backbone, pretrained=True, verbose=False)
        self.hidden_dim = 768
        
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("ViT backbone frozen")
                
        self.scale_layers = [2, 5, 8, 11]
        self.proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU()
            ) for _ in self.scale_layers
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.depth_aggregator = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x = x.squeeze(1).view(B * D, 1, H, W).repeat(1, 3, 1, 1)
        
        with torch.set_grad_enabled(not self._is_frozen()):
            features = self.vit.get_intermediate_layers(x, n=self.scale_layers, reshape=True)
        
        h, w = features[0].shape[2], features[0].shape[3]
        multi_scale = []
        
        for i, feat in enumerate(features):
            if feat.shape[2:] != (h, w):
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B * D * h * w, self.hidden_dim)
            proj = self.proj_heads[i](feat_flat)
            proj = proj.reshape(B * D, h, w, -1).permute(0, 3, 1, 2)
            multi_scale.append(proj)
        
        fused = torch.cat(multi_scale, dim=1)
        fused = self.fusion(fused)
        
        _, C_out, h, w = fused.shape
        fused_3d = fused.reshape(B, D, C_out, h, w).permute(0, 2, 1, 3, 4)
        features_3d = self.depth_aggregator(fused_3d)
        
        return features_3d
    
    def _is_frozen(self) -> bool:
        return not next(self.vit.parameters()).requires_grad


class VISTA3DDecoder(nn.Module):
    """VISTA3D-style decoder"""
    def __init__(self, in_channels: int = 256, num_classes: int = 4, deep_supervision: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        self.class_embeddings = nn.Embedding(num_classes, in_channels)
        
        self.up1 = self._make_up_block(in_channels, in_channels // 2)
        self.up2 = self._make_up_block(in_channels // 2, in_channels // 4)
        self.up3 = self._make_up_block(in_channels // 4, in_channels // 8)
        
        self.class_heads = nn.ModuleList([
            nn.Conv3d(in_channels // 8, 1, kernel_size=1) for _ in range(num_classes)
        ])
        
        if deep_supervision:
            self.ds_head1 = nn.ModuleList([nn.Conv3d(in_channels // 2, 1, kernel_size=1) for _ in range(num_classes)])
            self.ds_head2 = nn.ModuleList([nn.Conv3d(in_channels // 4, 1, kernel_size=1) for _ in range(num_classes)])
        
    def _make_up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x1 = self.up1(features)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        
        outputs = [head(x3) for head in self.class_heads]
        logits = torch.cat(outputs, dim=1)
        
        if self.deep_supervision and self.training:
            ds1 = torch.cat([head(x1) for head in self.ds_head1], dim=1)
            ds2 = torch.cat([head(x2) for head in self.ds_head2], dim=1)
            return logits, ds1, ds2
        
        return logits


class MedDINOVISTA3D(nn.Module):
    """Complete model"""
    def __init__(self, num_classes: int = 4, freeze_encoder: bool = True, backbone: str = "dinov2_vitb14", deep_supervision: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        self.encoder = MedDINOEncoder(backbone=backbone, feature_dim=256, freeze_backbone=freeze_encoder)
        self.decoder = VISTA3DDecoder(in_channels=256, num_classes=num_classes, deep_supervision=deep_supervision)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        outputs = self.decoder(features)
        
        if isinstance(outputs, tuple):
            logits, ds1, ds2 = outputs
            logits = F.interpolate(logits, size=x.shape[2:], mode='trilinear', align_corners=False)
            ds1 = F.interpolate(ds1, size=x.shape[2:], mode='trilinear', align_corners=False)
            ds2 = F.interpolate(ds2, size=x.shape[2:], mode='trilinear', align_corners=False)
            return logits, ds1, ds2
        else:
            if outputs.shape[2:] != x.shape[2:]:
                outputs = F.interpolate(outputs, size=x.shape[2:], mode='trilinear', align_corners=False)
            return outputs
    
    def unfreeze_encoder(self, num_layers: int = 4):
        params = list(self.encoder.vit.parameters())
        for param in params[-num_layers:]:
            param.requires_grad = True
        print(f"Unfroze last {num_layers} layers of encoder")


# ============================================================================
# DATASET - IMPROVED SAMPLING
# ============================================================================

class KiTS23Dataset(Dataset):
    """KiTS23 with IMPROVED positive sampling for tumor/cyst"""
    
    def __init__(self, data_dicts: List[Dict], patch_size: Tuple[int, int, int], num_samples: int = 2, 
                 positive_ratio: float = 0.7, is_train: bool = True):
        self.data_dicts = data_dicts
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.positive_ratio = positive_ratio
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data_dicts) * self.num_samples
    
    def __getitem__(self, idx):
        vol_idx = idx // self.num_samples
        sample_idx = idx % self.num_samples
        data_dict = self.data_dicts[vol_idx]
        
        image = nib.load(data_dict["image"]).get_fdata()
        label = nib.load(data_dict["label"]).get_fdata()
        
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        # IMPROVED: Sample tumor/cyst patches more often
        force_positive = self.is_train and (np.random.rand() < self.positive_ratio)
        patch_img, patch_lbl = self._sample_patch(image, label, force_positive)
        
        patch_img = torch.from_numpy(patch_img).float().unsqueeze(0)
        patch_lbl = torch.from_numpy(patch_lbl).long().unsqueeze(0)
        
        return {"image": patch_img, "label": patch_lbl}
    
    def _sample_patch(self, image, label, force_positive=False):
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # Pad if needed
        if d < pd or h < ph or w < pw:
            pad_d, pad_h, pad_w = max(0, pd - d), max(0, ph - h), max(0, pw - w)
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            d, h, w = image.shape
        
        if self.is_train and force_positive:
            # Sample patch containing tumor (2) or cyst (3)
            tumor_cyst_mask = (label == 2) | (label == 3)
            if tumor_cyst_mask.sum() > 0:
                coords = np.argwhere(tumor_cyst_mask)
                center = coords[np.random.randint(len(coords))]
                
                d_start = np.clip(center[0] - pd//2, 0, d - pd)
                h_start = np.clip(center[1] - ph//2, 0, h - ph)
                w_start = np.clip(center[2] - pw//2, 0, w - pw)
            else:
                # Fallback: random crop
                d_start = np.random.randint(0, max(1, d - pd + 1))
                h_start = np.random.randint(0, max(1, h - ph + 1))
                w_start = np.random.randint(0, max(1, w - pw + 1))
        else:
            # Random crop
            d_start = np.random.randint(0, max(1, d - pd + 1))
            h_start = np.random.randint(0, max(1, h - ph + 1))
            w_start = np.random.randint(0, max(1, w - pw + 1))
        
        patch_img = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        patch_lbl = label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        return patch_img, patch_lbl


def get_dataloaders(config, max_cases=None):
    """Create dataloaders"""
    data_dir = Path(config["kits23_dir"])
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    
    if max_cases:
        cases = cases[:max_cases]
    
    data_dicts = []
    for case in cases:
        img_path, seg_path = case / "imaging.nii.gz", case / "segmentation.nii.gz"
        if img_path.exists() and seg_path.exists():
            data_dicts.append({"image": str(img_path), "label": str(seg_path)})
    
    val_size = int(len(data_dicts) * config["val_split"])
    train_files, val_files = data_dicts[val_size:], data_dicts[:val_size]
    
    print(f"Training: {len(train_files)} | Validation: {len(val_files)}")
    
    train_dataset = KiTS23Dataset(
        train_files, config["patch_size"], 
        num_samples=config["num_samples_per_volume"],
        positive_ratio=config["positive_sample_ratio"],
        is_train=True
    )
    
    val_dataset = KiTS23Dataset(
        val_files, config["patch_size"], 
        num_samples=1, positive_ratio=0.0, is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


# ============================================================================
# IMPROVED LOSS - FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target.squeeze(1), reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class ImprovedHybridLoss(nn.Module):
    """Improved loss with Focal Loss for tumor/cyst"""
    def __init__(self, ds_weights=[1.0, 0.7, 0.5]):  # Higher DS weights
        super().__init__()
        
        # Class weights: heavily weight tumor (2) and cyst (3)
        class_weights = torch.tensor([0.5, 1.0, 5.0, 5.0])  # BG, Kidney, Tumor, Cyst
        
        self.focal = FocalLoss(alpha=class_weights, gamma=2.0)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.ds_weights = ds_weights
        
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.squeeze(1).long(), num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_onehot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
        
        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Weight tumor/cyst more
        class_weights = torch.tensor([0.5, 1.0, 3.0, 3.0], device=dice.device)
        weighted_dice = (dice * class_weights.unsqueeze(0)).sum(dim=1) / class_weights.sum()
        
        return 1 - weighted_dice.mean()
    
    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            logits, ds1, ds2 = outputs
            predictions = [logits, ds1, ds2]
        else:
            predictions = [outputs]
        
        total_loss = 0
        for i, pred in enumerate(predictions):
            if pred.shape[2:] != target.shape[2:]:
                target_resized = F.interpolate(target.float(), size=pred.shape[2:], mode='nearest').long()
            else:
                target_resized = target
            
            focal_loss = self.focal(pred, target_resized)
            dice_loss = self.dice_loss(pred, target_resized)
            loss = focal_loss + dice_loss
            
            weight = self.ds_weights[i] if i < len(self.ds_weights) else 1.0
            total_loss += weight * loss
        
        return total_loss


# ============================================================================
# TRAINING UTILITIES (same as v1 but with new loss)
# ============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_dice(pred, target):
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    dice_per_class = {}
    class_names = ['kidney', 'tumor', 'cyst']
    
    for i, c in enumerate(range(1, 4)):
        pred_c, target_c = (pred == c), (target == c)
        
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


def train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0
    dice_scores = {'kidney': [], 'tumor': [], 'cyst': [], 'mean': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            pred = torch.argmax(outputs, dim=1, keepdim=True)
            dice = compute_dice(pred, labels)
            
            for key in dice_scores.keys():
                dice_scores[key].append(dice[key])
    
    avg_dice = {k: np.mean(v) for k, v in dice_scores.items()}
    return val_loss / len(loader), avg_dice


def train(config, stage1_epochs, stage2_epochs, max_cases=None):
    print("="*60)
    print("MedDINO-VISTA3D v2 Training (Improved Class Balance)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    
    train_loader, val_loader = get_dataloaders(config, max_cases)
    
    model = MedDINOVISTA3D(
        num_classes=config["num_classes"],
        freeze_encoder=True,
        backbone=config["backbone"],
        deep_supervision=True
    ).to(device)
    
    total, trainable = count_parameters(model)
    print(f"\nTotal: {total:,} | Trainable (stage 1): {trainable:,}")
    
    loss_fn = ImprovedHybridLoss()
    scaler = GradScaler('cuda')
    best_dice = 0
    
    # STAGE 1
    print("\n" + "="*60)
    print("STAGE 1: Frozen Encoder + Focal Loss")
    print("="*60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_stage1"], weight_decay=config["weight_decay"])
    
    for epoch in range(stage1_epochs):
        print(f"\nEpoch {epoch+1}/{stage1_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Dice - Kidney: {val_dice['kidney']:.4f} | Tumor: {val_dice['tumor']:.4f} | Cyst: {val_dice['cyst']:.4f} | Mean: {val_dice['mean']:.4f}")
        
        if val_dice['mean'] > best_dice:
            best_dice = val_dice['mean']
            torch.save(model.state_dict(), output_dir / "best_model_stage1.pth")
            print(f"✅ Best model saved (Mean Dice: {best_dice:.4f})")
    
    # STAGE 2
    if stage2_epochs > 0:
        print("\n" + "="*60)
        print("STAGE 2: Fine-tuned Encoder")
        print("="*60)
        
        model.unfreeze_encoder(num_layers=4)
        total, trainable = count_parameters(model)
        print(f"Trainable (stage 2): {trainable:,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_stage2"], weight_decay=config["weight_decay"])
        
        for epoch in range(stage2_epochs):
            print(f"\nEpoch {epoch+1}/{stage2_epochs}")
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
            val_loss, val_dice = validate(model, val_loader, loss_fn, device)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Dice - Kidney: {val_dice['kidney']:.4f} | Tumor: {val_dice['tumor']:.4f} | Cyst: {val_dice['cyst']:.4f} | Mean: {val_dice['mean']:.4f}")
            
            if val_dice['mean'] > best_dice:
                best_dice = val_dice['mean']
                torch.save(model.state_dict(), output_dir / "best_model_stage2.pth")
                print(f"✅ Best model saved (Mean Dice: {best_dice:.4f})")
    
    torch.save(model.state_dict(), output_dir / "final_model.pth")
    print("\n" + "="*60)
    print(f"✅ Training complete! Best Dice: {best_dice:.4f}")
    print("="*60)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MedDINO-VISTA3D v2: Improved Class Balance")
    parser.add_argument("--mode", type=str, default="quick_test", choices=["quick_test", "full"])
    args = parser.parse_args()
    
    if args.mode == "quick_test":
        print("QUICK TEST (3+2 epochs, 20 cases)")
        train(CONFIG, stage1_epochs=CONFIG["quick_epochs_stage1"], stage2_epochs=CONFIG["quick_epochs_stage2"], max_cases=CONFIG["quick_cases"])
    else:
        print("FULL TRAINING (100+50 epochs, all cases)")
        train(CONFIG, stage1_epochs=CONFIG["num_epochs_stage1"], stage2_epochs=CONFIG["num_epochs_stage2"], max_cases=None)


if __name__ == "__main__":
    main()
