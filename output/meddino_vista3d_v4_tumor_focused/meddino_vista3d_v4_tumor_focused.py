"""
MedDINO-VISTA3D V4 Tumor-Focused Edition
=========================================

OPTIMIZED FOR TUMOR SEGMENTATION - Target: 75% Dice

KEY CHANGES FROM V3:
- Tumor-focused sampling: 50% Tumor, 25% Cyst, 15% Kidney, 10% Random
- Rebalanced loss weights: Tumor=15.0, Cyst=10.0 (vs V3: Tumor=12.0, Cyst=20.0)
- Same proven architecture from V3
- Dedicated output directory: ./output/meddino_vista3d_v4_tumor_focused

ARCHITECTURE (unchanged from V3):
- Multi-scale skip connections from DINOv2 layers
- Progressive 2-stage upsampling decoder (memory-optimized)
- Spatial attention at decoder
- Tversky (α=0.15, β=0.85) + Focal (γ=2.5) + Dice hybrid loss

Usage:
    python meddino_vista3d_v4_tumor_focused.py --mode quick_test
    python meddino_vista3d_v4_tumor_focused.py --mode full
"""

import os
import json
import time
from datetime import datetime
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm
from einops import rearrange

# Settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["XFORMERS_DISABLED"] = "1"

# ============================================================================
# CONFIGURATION
# ============================================================================

def get_config(mode):
    """Returns config based on mode (quick_test or full)"""
    base = {
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/meddino_vista3d_v4_tumor_focused",  # V4 dedicated folder
        "dinov2_backbone": "dinov2_vitb14",
        "hf_token": None,
        "num_classes": 4,
        "batch_size": 2,  # Increased for RTX 5090
        "val_split": 0.2,
        "num_samples_per_volume": 2,  # Increased for efficiency
        "patch_size": (140, 224, 224),  # Restored to full size, divisible by 14
        "lr_stage1": 1e-4,
        "lr_stage2": 1e-5,
        "weight_decay": 1e-5,
    }

    if mode == "quick_test":
        base.update({
            "num_epochs_stage1": 15,
            "num_epochs_stage2": 10,
            "quick_cases": 100,
            "use_warmup": False,
            "patience_stage1": 3,
            "patience_stage2": 3,
        })
    else:
        base.update({
            "num_epochs_stage1": 60,
            "num_epochs_stage2": 50,
            "quick_cases": None,
            "use_warmup": True,
            "warmup_epochs": 5,  
            "patience_stage1": 10,
            "patience_stage2": 10,
        })
    return base


# ============================================================================
# ENHANCED ENCODER - Multi-Scale DINOv2 with Skip Connections
# ============================================================================

class EnhancedMedDINOEncoder(nn.Module):
    """
    Enhanced DINOv2 encoder that extracts multi-scale features
    for U-Net style skip connections.
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        feature_dim: int = 256,
        freeze_backbone: bool = True,
        token: str = None,
    ):
        super().__init__()
        print(f"Loading Enhanced DINOv2 Encoder: {model_name}...")
        
        # Load DINOv2
        self.vit = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True, verbose=False
        )
        self.hidden_dim = 768  # ViT-B/14
        
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Critical layers for skip connections (reduced to 2 for memory)
        self.scale_layers = [8, 11]  # Most important layers
        self.fusion_layers = [2, 5, 8, 11]  # All layers for bottleneck fusion
        
        # Projection heads for each scale
        self.proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
            )
            for _ in range(4)
        ])
        
        # Fusion for bottleneck
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 3D depth aggregator for bottleneck
        self.depth_aggregator = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        print(f"✓ Enhanced encoder initialized with {len(self.scale_layers)} skip connection levels")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, D, H, W = x.shape
        
        # Prepare for ViT (flatten depth into batch)
        x_2d = x.squeeze(1).view(B * D, 1, H, W).repeat(1, 3, 1, 1)
        
        # Extract features for skip connections (2 layers)
        with torch.set_grad_enabled(not next(self.vit.parameters()).requires_grad):
            skip_features_2d = [
                self.vit.get_intermediate_layers(x_2d, n=[i])[0] for i in self.scale_layers
            ]
            
            # Extract all features for bottleneck fusion
            fusion_features_2d = [
                self.vit.get_intermediate_layers(x_2d, n=[i])[0] for i in self.fusion_layers
            ]
        
        # Process features
        h, w = H // 14, W // 14  # ViT-B/14 patch size
        
        # Process skip connections (2 layers)
        skip_features_3d = []
        for i, feat in enumerate(skip_features_2d):
            num_patches = feat.shape[1]
            patch_h = patch_w = int(num_patches ** 0.5)
            feat_2d = feat.permute(0, 2, 1).reshape(feat.shape[0], self.hidden_dim, patch_h, patch_w)
            feat_2d = F.interpolate(feat_2d, size=(h, w), mode="bilinear", align_corners=False)
            feat_proj = self.proj_heads[i](feat_2d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            feat_3d = rearrange(feat_proj, '(b d) c h w -> b c d h w', b=B, d=D)
            skip_features_3d.append(feat_3d)
        
        # Process fusion features (4 layers)
        multi_scale_2d = []
        for i, feat in enumerate(fusion_features_2d):
            num_patches = feat.shape[1]
            patch_h = patch_w = int(num_patches ** 0.5)
            feat_2d = feat.permute(0, 2, 1).reshape(feat.shape[0], self.hidden_dim, patch_h, patch_w)
            feat_2d = F.interpolate(feat_2d, size=(h, w), mode="bilinear", align_corners=False)
            feat_proj = self.proj_heads[i](feat_2d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            multi_scale_2d.append(feat_proj)
        
        # Create bottleneck (fused multi-scale)
        fused_2d = self.fusion(torch.cat(multi_scale_2d, dim=1))
        fused_3d = rearrange(fused_2d, '(b d) c h w -> b c d h w', b=B, d=D)
        bottleneck = self.depth_aggregator(fused_3d)
        
        return {
            'bottleneck': bottleneck,  # (B, 256, D, H/14, W/14)
            'skip1': skip_features_3d[0],  # Layer 8 (B, 256, D, h, w)
            'skip2': skip_features_3d[1],  # Layer 11
        }


# ============================================================================
# SPATIAL ATTENTION MODULE
# ============================================================================

class SpatialAttention3D(nn.Module):
    """3D Spatial Attention for decoder refinement"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


# ============================================================================
# U-NET STYLE DECODER WITH SKIP CONNECTIONS
# ============================================================================

class UpBlock(nn.Module):
    """Upsampling block with skip connection and attention"""
    
    def __init__(self, in_channels, out_channels, skip_channels, use_attention=True):
        super().__init__()
        
        # Transposed conv for upsampling
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        
        # Skip connection adapter
        self.skip_conv = nn.Conv3d(skip_channels, out_channels, 1)
        
        # Refinement conv
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Optional spatial attention
        self.attention = SpatialAttention3D() if use_attention else nn.Identity()
    
    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        
        # Adapt skip connection
        skip = self.skip_conv(skip)
        
        # Match sizes if needed
        if x.shape != skip.shape:
            skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        # Concatenate
        x = torch.cat([x, skip], dim=1)
        
        # Refine
        x = self.conv(x)
        
        # Apply attention
        x = self.attention(x)
        
        return x


class EnhancedVISTA3DDecoder(nn.Module):
    """
    Enhanced VISTA3D-style decoder with U-Net skip connections.
    Maintains compatibility with VISTA3D for future LLM expert integration.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 4,
        decoder_channels: List[int] = [128, 64],  # Simplified to 2 stages
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # 2-stage upsampling with 2 skip connections (memory optimized)
        self.up1 = UpBlock(in_channels, decoder_channels[0], skip_channels=256, use_attention=True)
        self.up2 = UpBlock(decoder_channels[0], decoder_channels[1], skip_channels=256, use_attention=True)
        
        # Final upsampling and projection
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(decoder_channels[1], 32, 2, stride=2),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv3d(32, num_classes, 1)
        
        print(f"✓ Memory-optimized decoder initialized: {decoder_channels} (2 skip connections)")
    
    def forward(self, encoder_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Start from bottleneck
        x = encoder_outputs['bottleneck']  # (B, 256, D, H/14, W/14)
        
        # 2-stage upsampling with skip connections
        x = self.up1(x, encoder_outputs['skip2'])  # + Layer 11 (deepest)
        x = self.up2(x, encoder_outputs['skip1'])  # + Layer 8
        
        # Final upsampling to full resolution
        x = self.final_up(x)
        
        # Final segmentation
        logits = self.final(x)
        
        return logits


# ============================================================================
# COMPLETE MODEL
# ============================================================================

class MedDINOVISTA3DEnhanced(nn.Module):
    """Enhanced MedDINO-VISTA3D with U-Net decoder"""
    
    def __init__(
        self,
        num_classes=4,
        freeze_encoder=True,
        dropout=0.1,
        dinov2_backbone="dinov2_vitb14",
        token=None,
    ):
        super().__init__()
        self.encoder = EnhancedMedDINOEncoder(
            model_name=dinov2_backbone,
            freeze_backbone=freeze_encoder,
            token=token
        )
        self.decoder = EnhancedVISTA3DDecoder(
            num_classes=num_classes,
            dropout=dropout
        )
    
    def forward(self, x):
        # Multi-scale encoding with skip connections
        encoder_out = self.encoder(x)
        
        # U-Net decoding
        logits = self.decoder(encoder_out)
        
        # Resize to exact input size if needed
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        
        return logits
    
    def unfreeze_encoder(self, num_layers=4):
        params = list(self.encoder.vit.parameters())
        for param in params[-num_layers:]:
            param.requires_grad = True
        print(f"Unfroze last {num_layers} layers of encoder")


# ============================================================================
# LOSS FUNCTIONS - TUMOR-FOCUSED
# ============================================================================

class TverskyLoss(nn.Module):
    """Tversky Loss - heavily penalizes false negatives (missing tumors/cysts)"""
    
    def __init__(self, alpha=0.15, beta=0.85, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.squeeze(1).long(), num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
        
        TP = (pred * target_onehot).sum(dim=(2, 3, 4))
        FP = ((1 - target_onehot) * pred).sum(dim=(2, 3, 4))
        FN = (target_onehot * (1 - pred)).sum(dim=(2, 3, 4))
        
        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        return 1 - tversky.mean()


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if isinstance(alpha, (float, int, list)):
            alpha = torch.tensor(alpha)
        
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = alpha
        
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets.squeeze(1), weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TumorFocusedHybridLoss(nn.Module):
    """V4 TUMOR-FOCUSED Loss: Tversky + Focal + Dice"""
    
    def __init__(self):
        super().__init__()
        # TUMOR-FOCUSED class weights: BG, Kidney, Tumor, Cyst
        w = torch.tensor([0.1, 1.0, 15.0, 10.0])  # Tumor priority!
        self.register_buffer("class_weights", w)
        
        # Tversky with aggressive FN penalty
        self.tversky = TverskyLoss(alpha=0.15, beta=0.85)
        
        # Focal loss
        self.focal = FocalLoss(alpha=w, gamma=2.5)  # Higher gamma for harder focus
        
        # Standard Dice
        self.dice_weight = 0.3
    
    def forward(self, outputs, target):
        # Tversky (main - focus on FN)
        tv = self.tversky(outputs, target)
        
        # Focal (class imbalance)
        fc = self.focal(outputs, target)
        
        # Dice (standard segmentation)
        pred_soft = F.softmax(outputs, dim=1)
        target_oh = F.one_hot(target.squeeze(1).long(), num_classes=outputs.shape[1])
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()
        
        dice_per_class = 2 * (pred_soft * target_oh).sum(dim=(2,3,4)) / \
                         ((pred_soft + target_oh).sum(dim=(2,3,4)) + 1e-6)
        dice_loss = 1 - dice_per_class.mean()
        
        # Combine: 50% Tversky, 30% Focal, 20% Dice
        return 0.5 * tv + 0.3 * fc + 0.2 * dice_loss


# ============================================================================
# DATASET - TUMOR-FOCUSED Sampling
# ============================================================================

class KiTS23Dataset(Dataset):
    def __init__(self, data_dicts, patch_size, num_samples=2, is_train=True):
        self.data_dicts = data_dicts
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.is_train = is_train
    
    def __len__(self):
        return len(self.data_dicts) * self.num_samples
    
    def __getitem__(self, idx):
        vol_idx = idx // self.num_samples
        img = nib.load(self.data_dicts[vol_idx]["image"]).get_fdata()
        lbl = nib.load(self.data_dicts[vol_idx]["label"]).get_fdata()
        
        # Normalize
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        p_img, p_lbl = self._sample_patch(img, lbl)
        return {
            "image": torch.from_numpy(p_img).float().unsqueeze(0),
            "label": torch.from_numpy(p_lbl).long().unsqueeze(0),
        }
    
    def _sample_patch(self, img, lbl):
        d, h, w = img.shape
        pd, ph, pw = self.patch_size
        
        if d < pd or h < ph or w < pw:
            pad = ((0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w)))
            img, lbl = np.pad(img, pad, mode="constant"), np.pad(lbl, pad, mode="constant")
            d, h, w = img.shape
        
        if self.is_train:
            # V4 TUMOR-FOCUSED: 50% Tumor, 25% Cyst, 15% Kidney, 10% Random
            rand_val = np.random.random()
            
            target_class = None
            if rand_val < 0.5:
                target_class = 2  # Tumor (50%!)
            elif rand_val < 0.75:
                target_class = 3  # Cyst (25%)
            elif rand_val < 0.9:
                target_class = 1  # Kidney (15%)
            
            # Try to find target class
            if target_class is not None:
                indices = np.argwhere(lbl == target_class)
                if len(indices) > 0:
                    c = indices[np.random.randint(len(indices))]
                    ds, hs, ws = (
                        np.clip(c[0] - pd // 2, 0, d - pd),
                        np.clip(c[1] - ph // 2, 0, h - ph),
                        np.clip(c[2] - pw // 2, 0, w - pw),
                    )
                else:
                    # Fallback
                    fallback_indices = np.argwhere((lbl == 2) | (lbl == 3))
                    if len(fallback_indices) > 0:
                        c = fallback_indices[np.random.randint(len(fallback_indices))]
                        ds, hs, ws = (
                            np.clip(c[0] - pd // 2, 0, d - pd),
                            np.clip(c[1] - ph // 2, 0, h - ph),
                            np.clip(c[2] - pw // 2, 0, w - pw),
                        )
                    else:
                        kid_idx = np.argwhere(lbl == 1)
                        if len(kid_idx) > 0:
                            c = kid_idx[np.random.randint(len(kid_idx))]
                            ds, hs, ws = (
                                np.clip(c[0] - pd // 2, 0, d - pd),
                                np.clip(c[1] - ph // 2, 0, h - ph),
                                np.clip(c[2] - pw // 2, 0, w - pw),
                            )
                        else:
                            ds, hs, ws = (
                                np.random.randint(0, max(1, d - pd + 1)),
                                np.random.randint(0, max(1, h - ph + 1)),
                                np.random.randint(0, max(1, w - pw + 1)),
                            )
            else:
                # 10% Random
                ds, hs, ws = (
                    np.random.randint(0, max(1, d - pd + 1)),
                    np.random.randint(0, max(1, h - ph + 1)),
                    np.random.randint(0, max(1, w - pw + 1)),
                )
        else:
            ds, hs, ws = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2
        
        return (
            img[ds:ds + pd, hs:hs + ph, ws:ws + pw],
            lbl[ds:ds + pd, hs:hs + ph, ws:ws + pw],
        )


def get_dataloaders(config, max_cases=None):
    data_dir = Path(config["kits23_dir"])
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    if max_cases:
        cases = cases[:max_cases]
    
    data_dicts = [
        {"image": str(c / "imaging.nii.gz"), "label": str(c / "segmentation.nii.gz")}
        for c in cases if (c / "imaging.nii.gz").exists()
    ]
    
    split = int(len(data_dicts) * config["val_split"])
    train_ds = KiTS23Dataset(data_dicts[split:], config["patch_size"], config["num_samples_per_volume"], True)
    val_ds = KiTS23Dataset(data_dicts[:split], config["patch_size"], 1, False)
    
    return (
        DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True),
    )


# ============================================================================
# TRAINING UTILITIES (Same as V3)
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class MetricTracker:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
    
    def update(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        mask = (targets >= 0) & (targets < self.num_classes)
        self.confusion_matrix += torch.bincount(
            self.num_classes * targets[mask].long() + preds[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
    
    def compute(self):
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(0) - tp
        fn = self.confusion_matrix.sum(1) - tp
        
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        return {
            "dice": dice.numpy(),
            "iou": iou.numpy(),
            "precision": precision.numpy(),
            "recall": recall.numpy(),
        }
    
    def format_results(self):
        metrics = self.compute()
        classes = ["BG", "Kidney", "Tumor", "Cyst"]
        
        mean_dice = np.mean(metrics["dice"][1:])
        mean_iou = np.mean(metrics["iou"][1:])
        
        s = f"\n  Mean Dice (Fg): {mean_dice:.4f} | Mean IoU (Fg): {mean_iou:.4f}\n"
        s += f"  {'Class':<10} {'Dice':<8} {'IoU':<8} {'Prec':<8} {'Recall':<8}\n"
        s += "-" * 46 + "\n"
        for i, c in enumerate(classes):
            s += f"  {c:<10} {metrics['dice'][i]:.4f}   {metrics['iou'][i]:.4f}   {metrics['precision'][i]:.4f}   {metrics['recall'][i]:.4f}\n"
        return s, mean_dice


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        try:
            img, lbl = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=True):
                out = model(img)
                loss = loss_fn(out, lbl)
            
            if not torch.isfinite(loss):
                print(f"\n⚠ Warning: Non-finite loss at batch {batch_idx}, skipping...")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                del img, lbl, out, loss
                torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"\n⚠ Error at batch {batch_idx}: {e}")
            torch.cuda.empty_cache()
            continue
    
    return epoch_loss / max(batch_count, 1)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0
    tracker = MetricTracker(num_classes=4)
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
            try:
                img, lbl = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
                
                with torch.amp.autocast("cuda", enabled=True):
                    out = model(img)
                    loss = loss_fn(out, lbl)
                    
                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        batch_count += 1
                
                preds = torch.argmax(out, dim=1)
                tracker.update(preds, lbl.squeeze(1))
                
                if batch_idx % 5 == 0:
                    del img, lbl, out, loss, preds
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"\n⚠ Validation error at batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                continue
    
    report, mean_dice = tracker.format_results()
    metrics_dict = tracker.compute()  # Also return metrics dict for JSON logging
    return val_loss / max(batch_count, 1), mean_dice, report, metrics_dict


def save_epoch_results(output_dir, stage, epoch, train_loss, val_loss, metrics, epoch_time, is_best=False):
    """Save epoch results to JSON with timing"""
    epoch_data = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "epoch": epoch,
        "epoch_time_seconds": round(epoch_time, 2),
        "epoch_time_minutes": round(epoch_time / 60, 2),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "metrics": {
            "mean_dice_fg": float(metrics["dice"][1:].mean()),
            "mean_iou_fg": float(metrics["iou"][1:].mean()),
            "classes": {
                "background": {
                    "dice": float(metrics["dice"][0]),
                    "iou": float(metrics["iou"][0]),
                    "precision": float(metrics["precision"][0]),
                    "recall": float(metrics["recall"][0])
                },
                "kidney": {
                    "dice": float(metrics["dice"][1]),
                    "iou": float(metrics["iou"][1]),
                    "precision": float(metrics["precision"][1]),
                    "recall": float(metrics["recall"][1])
                },
                "tumor": {
                    "dice": float(metrics["dice"][2]),
                    "iou": float(metrics["iou"][2]),
                    "precision": float(metrics["precision"][2]),
                    "recall": float(metrics["recall"][2])
                },
                "cyst": {
                    "dice": float(metrics["dice"][3]),
                    "iou": float(metrics["iou"][3]),
                    "precision": float(metrics["precision"][3]),
                    "recall": float(metrics["recall"][3])
                }
            }
        },
        "is_best": bool(is_best)  # Convert numpy bool to Python bool
    }
    
    history_file = output_dir / "training_history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = {"epochs": []}
    
    history["epochs"].append(epoch_data)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def save_checkpoint(output_dir, epoch, stage, model, optimizer, scheduler, best_dice, scaler):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'stage': stage,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'best_dice': best_dice,
    }
    checkpoint_path = output_dir / f"checkpoint_{stage}_epoch{epoch+1}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"  → Saved checkpoint: {checkpoint_path}")


def load_checkpoint(output_dir, stage, model, optimizer, scheduler, scaler, device):
    """Load latest checkpoint for a stage"""
    checkpoints = list(output_dir.glob(f"checkpoint_{stage}_*.pth"))
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint


def save_final_evaluation(output_dir, best_dice, total_params, config, total_time):
    """Save final evaluation summary"""
    eval_data = {
        "timestamp": datetime.now().isoformat(),
        "version": "V4_TUMOR_FOCUSED",
        "best_dice": float(best_dice),
        "total_parameters": int(total_params),
        "total_training_time_hours": round(total_time / 3600, 2),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
    }
    
    with open(output_dir / "final_evaluation.json", 'w') as f:
        json.dump(eval_data, f, indent=2)


def get_scheduler(optimizer, num_epochs, is_stage2=False):
    """Cosine annealing scheduler"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(config, resume=False):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"MedDINO-VISTA3D V4 TUMOR-FOCUSED Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Sampling: 50% Tumor, 25% Cyst, 15% Kidney, 10% Random")
    print(f"Loss Weights: Tumor=15.0, Cyst=10.0")
    print(f"{'='*60}\n")
    
    # Data
    train_loader, val_loader = get_dataloaders(config, max_cases=config.get("quick_cases"))
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # Model
    model = MedDINOVISTA3DEnhanced(
        num_classes=config["num_classes"],
        freeze_encoder=True,
        dinov2_backbone=config["dinov2_backbone"],
        token=config["hf_token"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,} | Trainable: {trainable_params:,}")
    
    # Loss & Optimizer
    loss_fn = TumorFocusedHybridLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_stage1"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    
    # Scheduler
    scheduler = get_scheduler(optimizer, config["num_epochs_stage1"])
    if config.get("use_warmup"):
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=config["warmup_epochs"]
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[config["warmup_epochs"]]
        )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["patience_stage1"], 
        verbose=True, 
        path=str(output_dir / "best_model_stage1.pth")
    )
    
    # Checkpoint frequency
    checkpoint_freq = config.get('checkpoint_freq', 5)
    
    # STAGE 1
    print("\n--- STAGE 1: Training Decoder (Frozen Encoder) ---")
    best_dice = 0.0
    total_training_start = time.time()
    
    # Try to resume from Stage 1 checkpoint
    start_epoch = 0
    if resume:
        checkpoint = load_checkpoint(output_dir, "stage1", model, optimizer, scheduler, scaler, device)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint['best_dice']
    
    for epoch in range(start_epoch, config["num_epochs_stage1"]):
        epoch_start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        val_loss, mean_dice, report, metrics_dict = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        
        print(f"Ep {epoch+1}/{config['num_epochs_stage1']} | Loss: {train_loss:.4f}/{val_loss:.4f} | Dice: {mean_dice:.4f} | Time: {epoch_time/60:.1f}min")
        print(report)
        
        # Save epoch results to JSON with timing
        save_epoch_results(
            output_dir, "stage1", epoch+1, train_loss, val_loss,
            metrics_dict, epoch_time, is_best=(mean_dice > best_dice)
        )
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered in Stage 1")
            break
        
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), output_dir / "best_model_stage1_dice.pth")
            print(f"  -> Saved Best Dice Model (Dice: {best_dice:.4f})")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(output_dir, epoch, "stage1", model, optimizer, scheduler, best_dice, scaler)
    
    # STAGE 2
    if config["num_epochs_stage2"] > 0:
        print("\n--- STAGE 2: Fine-tuning Encoder ---")
        if (output_dir / "best_model_stage1_dice.pth").exists():
            model.load_state_dict(torch.load(output_dir / "best_model_stage1_dice.pth"))
        elif (output_dir / "best_model_stage1.pth").exists():
            model.load_state_dict(torch.load(output_dir / "best_model_stage1.pth"))
        
        model.unfreeze_encoder(num_layers=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_stage2"], weight_decay=config["weight_decay"])
        scheduler = get_scheduler(optimizer, config["num_epochs_stage2"], is_stage2=True)
        early_stopping = EarlyStopping(patience=config["patience_stage2"], verbose=True, path=str(output_dir / "best_model_final.pth"))
        
        # Try to resume from Stage 2 checkpoint
        start_epoch_stage2 = 0
        if resume:
            checkpoint = load_checkpoint(output_dir, "stage2", model, optimizer, scheduler, scaler, device)
            if checkpoint:
                start_epoch_stage2 = checkpoint['epoch'] + 1
                best_dice = checkpoint['best_dice']
        
        for epoch in range(start_epoch_stage2, config["num_epochs_stage2"]):
            epoch_start_time = time.time()
            
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
            val_loss, mean_dice, report, metrics_dict = validate(model, val_loader, loss_fn, device)
            scheduler.step()
            epoch_time = time.time() - epoch_start_time
            
            print(f"Ep {epoch+1}/{config['num_epochs_stage2']} | Loss: {train_loss:.4f}/{val_loss:.4f} | Dice: {mean_dice:.4f} | Time: {epoch_time/60:.1f}min")
            print(report)
            
            # Save epoch results to JSON with timing
            save_epoch_results(
                output_dir, "stage2", epoch+1, train_loss, val_loss,
                metrics_dict, epoch_time, is_best=(mean_dice > best_dice)
            )
            
            early_stopping(val_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping triggered in Stage 2")
                break
            
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), output_dir / "best_model_final_dice.pth")
                print(f"  -> Saved Best Final Dice Model (Dice: {best_dice:.4f})")
            
            # Save checkpoint every N epochs
            if (epoch + 1) % checkpoint_freq == 0:
                save_checkpoint(output_dir, epoch, "stage2", model, optimizer, scheduler, best_dice, scaler)
    
    total_training_time = time.time() - total_training_start
    print(f"\nTraining Complete. Best Overall Dice: {best_dice:.4f}")
    print(f"Total Training Time: {total_training_time/3600:.2f} hours")
    
    # Save final evaluation
    total_params = sum(p.numel() for p in model.parameters())
    save_final_evaluation(output_dir, best_dice, total_params, config, total_training_time)
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full", choices=["quick_test", "full"])
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()
    
    config = get_config(args.mode)
    config['checkpoint_freq'] = 5  # Save checkpoint every 5 epochs
    train(config, resume=args.resume)
