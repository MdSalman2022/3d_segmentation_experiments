"""
KiTS23 Hybrid SOTA Segmentation
================================
Full hybrid approach combining:
1. Two-stage cascade (low-res kidney → high-res tumor/cyst)
2. Spatial-Channel Attention U-Net
3. Deep Supervision
4. Cyclic LR for Uncertainty Ensemble
5. Advanced loss functions (Tversky + Dice + FocalCE)

Target: Tumor Dice 75-85%, Cyst Dice 65-75%

Usage:
    python kits23_hybrid_sota_compile.py --mode train --quick     # Quick test (5 epochs)
    python kits23_hybrid_sota_compile.py --mode train             # Full training
    python kits23_hybrid_sota_compile.py --mode inference
    python kits23_hybrid_sota_compile.py --mode evaluate
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
import argparse
import warnings
from pathlib import Path
from typing import Sequence, Tuple, Union, List, Dict, Optional
from datetime import datetime, timedelta
import time

import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import binary_dilation, generate_binary_structure

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Speed optimizations (free 10-20% speedup)
torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for faster matmul
torch.backends.cudnn.allow_tf32 = True  # TF32 for faster convolutions

from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandRotated, RandZoomd, RandFlipd, RandShiftIntensityd, RandScaleIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd, EnsureTyped,
    AsDiscreted, Rand3DElasticd,
)
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, PersistentDataset, DataLoader, Dataset, decollate_batch


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "kits23_dir": "./kits23/dataset",
    "output_dir": "./output/hybrid_sota",
    "cache_dir": "./output/hybrid_sota/persistent_cache",  # Disk cache for fast loading
    "seed": 42,
    
    # Stage 1: Low-res kidney localization (memory-optimized)
    "stage1": {
        "spacing": (2.0, 2.0, 2.0),
        "patch_size": (128, 128, 64),  # Fits all resampled images
        "channels": (32, 64, 128, 256),
        "strides": (2, 2, 2),
        "out_channels": 2,  # bg + kidney (binary)
        "epochs": 50,
        "quick_epochs": 1,
        "val_interval": 1,
        "early_stopping_patience": 15,  # Stop if no improvement for 30 validations
        "batch_size": 2,
        "num_samples": 6,
        "learning_rate": 1e-4,
        "cache_rate": 0.1,
        "num_workers": 8,
        "roi_threshold": 0.1,
        "dilation_size": 11,
    },
    
    # Stage 2: High-res tumor/cyst segmentation (optimized for 34GB VRAM)
    "stage2": {
        "spacing": (0.75, 0.75, 1.0),  # Finer resolution for small tumors (from analysis)
        "patch_size": (128, 128, 96),
        "channels": (64, 128, 256, 512, 512),
        "strides": (2, 2, 2, 2),
        "out_channels": 4,  # bg, kidney, tumor, cyst
        "epochs": 50,
        "quick_epochs": 1,
        "val_interval": 1,
        "early_stopping_patience": 20,  # Stop if no improvement for 50 validations
        "batch_size": 1,
        "num_samples": 4,
        "learning_rate": 1e-4,
        "deep_supervision": True,
        "class_weights": [0.1, 0.3, 15.0, 12.0],
        "num_workers": 8,
    },
    
    # Cyclic LR for uncertainty ensemble
    "cyclic_lr": {
        "T_0": 100,      # First cycle length
        "T_mult": 2,     # Multiply cycle length
        "eta_min": 1e-6, # Minimum LR
    },
    
    # Ensemble checkpoints to save (epochs)
    "ensemble_checkpoints": [100, 300, 500],
    
    # Inference (optimized for speed)
    "inference": {
        "sw_batch_size": 4,   # Increased for faster inference
        "overlap": 0.25,      # Reduced from 0.5 for faster validation
    },
    
    # Data
    "data": {
        "intensity_min": -79,
        "intensity_max": 304,
        "train_val_split": 0.85,
        "val_samples": 20,       # Limit validation to 20 samples for speed (set to -1 for all)
        "cascade_samples": 20,   # Separate setting for cascade inference (set to -1 for all)
    },
}


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class TverskyLoss(nn.Module):
    """Tversky Loss - penalizes FN more than FP for small structures."""
    
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, include_background=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target.squeeze(1).long(), num_classes)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
        
        start_ch = 0 if self.include_background else 1
        pred = pred[:, start_ch:]
        target_onehot = target_onehot[:, start_ch:]
        
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target_onehot.view(target_onehot.shape[0], target_onehot.shape[1], -1)
        
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))
        else:
            self.weight = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(pred, target.squeeze(1).long(), 
                            weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class HybridLoss(nn.Module):
    """Combined loss for Stage 2: Tversky + Dice + FocalCE."""
    
    def __init__(self, class_weights=None):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.2, beta=0.8, include_background=False)
        self.dice = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.focal = FocalLoss(gamma=2.0, weight=class_weights)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.4 * self.tversky(pred, target) + \
               0.3 * self.dice(pred, target) + \
               0.3 * self.focal(pred, target)


class DeepSupervisionLoss(nn.Module):
    """Wrapper for deep supervision with weighted multi-scale losses."""
    
    def __init__(self, loss_fn, weights=(1.0, 0.5, 0.25, 0.125)):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights
    
    def forward(self, preds: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for i, (pred, weight) in enumerate(zip(preds, self.weights)):
            if pred.shape[2:] != target.shape[2:]:
                target_ds = F.interpolate(target.float(), size=pred.shape[2:], mode='nearest')
            else:
                target_ds = target
            total_loss += weight * self.loss_fn(pred, target_ds)
        return total_loss


# ============================================================================
# ATTENTION MODULES
# ============================================================================
class ChannelAttention(nn.Module):
    """Channel Attention (SE-style)."""
    
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, max(channels // ratio, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(channels // ratio, 1), channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class CBAM(nn.Module):
    """CBAM: Channel + Spatial Attention."""
    
    def __init__(self, channels: int, ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, ratio)
        self.spatial = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ============================================================================
# SC-UNET MODEL WITH DEEP SUPERVISION
# ============================================================================
class ConvBlock(nn.Module):
    """Basic 3D conv block with InstanceNorm and ReLU."""
    
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResConvBlock(nn.Module):
    """Residual conv block for better gradient flow."""
    
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + residual)


class SCUNetEncoder(nn.Module):
    """SC-UNet Encoder with attention."""
    
    def __init__(self, in_ch: int, channels: Tuple[int, ...], 
                 strides: Tuple[int, ...], use_attention: bool = True):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.attentions = nn.ModuleList() if use_attention else None
        
        prev_ch = in_ch
        for i, ch in enumerate(channels):
            self.encoders.append(ResConvBlock(prev_ch, ch, dropout=0.1))
            if i < len(strides):
                self.pools.append(nn.MaxPool3d(strides[i]))
            if use_attention:
                self.attentions.append(CBAM(ch))
            prev_ch = ch
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if self.attentions:
                x = self.attentions[i](x)
            if i < len(self.pools):
                skips.append(x)
                x = self.pools[i](x)
        return x, skips


class SCUNetDecoder(nn.Module):
    """SC-UNet Decoder with optional deep supervision."""
    
    def __init__(self, channels: Tuple[int, ...], strides: Tuple[int, ...],
                 out_channels: int, deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        reversed_ch = list(reversed(channels))
        reversed_strides = list(reversed(strides))
        
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
        
        for i in range(len(reversed_strides)):
            in_ch = reversed_ch[i]
            skip_ch = reversed_ch[i + 1] if i + 1 < len(reversed_ch) else reversed_ch[-1]
            out_ch = skip_ch
            
            self.upconvs.append(
                nn.ConvTranspose3d(in_ch, in_ch, reversed_strides[i], stride=reversed_strides[i])
            )
            self.decoders.append(ResConvBlock(in_ch + skip_ch, out_ch))
            self.attentions.append(CBAM(out_ch))
            
            if deep_supervision and i < len(reversed_strides) - 1:
                self.ds_heads.append(nn.Conv3d(out_ch, out_channels, 1))
        
        self.final = nn.Conv3d(reversed_ch[-1], out_channels, 1)
    
    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        ds_outputs = []
        reversed_skips = list(reversed(skips))
        
        for i, (up, dec, att) in enumerate(zip(self.upconvs, self.decoders, self.attentions)):
            x = up(x)
            skip = reversed_skips[i]
            
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            x = att(x)
            
            if self.deep_supervision and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))
        
        final_out = self.final(x)
        
        if self.deep_supervision:
            return [final_out] + ds_outputs
        return final_out


class SCUNet(nn.Module):
    """Full SC-UNet with optional deep supervision."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 channels: Tuple[int, ...] = (64, 128, 256, 512, 512),
                 strides: Tuple[int, ...] = (2, 2, 2, 2),
                 deep_supervision: bool = False):
        super().__init__()
        self.encoder = SCUNetEncoder(in_channels, channels, strides)
        self.decoder = SCUNetDecoder(channels, strides, out_channels, deep_supervision)
        self.deep_supervision = deep_supervision
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        bottleneck, skips = self.encoder(x)
        return self.decoder(bottleneck, skips)


# ============================================================================
# DATA LOADING
# ============================================================================
def get_data_dicts(data_dir: str) -> List[Dict]:
    """Get data dictionaries for KiTS23."""
    data_path = Path(data_dir)
    data_dicts = []
    
    for case_dir in sorted(data_path.iterdir()):
        if case_dir.is_dir() and case_dir.name.startswith('case_'):
            img = case_dir / "imaging.nii.gz"
            seg = case_dir / "segmentation.nii.gz"
            if img.exists() and seg.exists():
                data_dicts.append({
                    "image": str(img), "label": str(seg), "case_id": case_dir.name
                })
    return data_dicts


def get_stage1_transforms(config: dict, training: bool = True) -> Compose:
    """Transforms for Stage 1 (kidney localization)."""
    cfg = config["stage1"]
    data_cfg = config["data"]
    
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=data_cfg["intensity_min"],
                            a_max=data_cfg["intensity_max"], b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=cfg["patch_size"]),
    ]
    
    if training:
        base.extend([
            RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                                   spatial_size=cfg["patch_size"], pos=1, neg=1,
                                   num_samples=cfg["num_samples"]),
            Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 8),
                          magnitude_range=(50, 150), prob=0.3, mode=("bilinear", "nearest")),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.1),
        ])
    
    base.append(EnsureTyped(keys=["image", "label"]))
    return Compose(base)


def get_stage2_transforms(config: dict, training: bool = True) -> Compose:
    """Transforms for Stage 2 (tumor/cyst segmentation)."""
    cfg = config["stage2"]
    data_cfg = config["data"]
    
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=data_cfg["intensity_min"],
                            a_max=data_cfg["intensity_max"], b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=cfg["patch_size"]),
    ]
    
    if training:
        base.extend([
            RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                                   spatial_size=cfg["patch_size"], pos=4, neg=1,
                                   num_samples=cfg["num_samples"]),
            RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3,
                       prob=0.5, mode=("bilinear", "nearest")),
            RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.3,
                     mode=("trilinear", "nearest")),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.1),
            RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
        ])
    
    base.append(EnsureTyped(keys=["image", "label"]))
    return Compose(base)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_stage1(config: dict, num_epochs: int = None, resume: bool = False):
    """Train Stage 1: Kidney localization."""
    print("\n" + "=" * 70)
    print("  STAGE 1: Low-Resolution Kidney Localization")
    print("=" * 70)
    
    cfg = config["stage1"]
    epochs = num_epochs or cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_dir = Path(config["output_dir"]) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    data_dicts = get_data_dicts(config["kits23_dir"])
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    train_files, val_files = data_dicts[:split_idx], data_dicts[split_idx:]
    
    # Limit validation samples for speed
    val_samples = config["data"].get("val_samples", -1)
    if val_samples > 0 and len(val_files) > val_samples:
        val_files = val_files[:val_samples]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Use PersistentDataset for disk caching - first run preprocesses, subsequent runs are instant
    cache_dir = Path(config.get("cache_dir", "./cache")) / "stage1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_ds = PersistentDataset(train_files, get_stage1_transforms(config, True), cache_dir=str(cache_dir / "train"))
    val_ds = PersistentDataset(val_files, get_stage1_transforms(config, False), cache_dir=str(cache_dir / "val"))
    
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, 
                             num_workers=cfg.get("num_workers", 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=cfg.get("num_workers", 4) // 2)
    
    # Model - lighter for stage 1
    model = SCUNet(
        in_channels=1, out_channels=cfg["out_channels"],
        channels=cfg["channels"], strides=cfg["strides"],
        deep_supervision=False
    ).to(device)
    
    print(f"Stage 1 params: {sum(p.numel() for p in model.parameters()):,}")
    
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    scaler = torch.amp.GradScaler('cuda')
    
    best_metric = -1
    patience_counter = 0
    patience = cfg.get("early_stopping_patience", 30)
    epoch_times = []
    start_epoch = 0
    
    # Resume from checkpoint if requested
    resume_ckpt = checkpoint_dir / "stage1_last.pth"
    if resume and resume_ckpt.exists():
        print(f"\n📂 Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_metric" in ckpt:
            best_metric = ckpt["best_metric"]
        if "patience_counter" in ckpt:
            patience_counter = ckpt["patience_counter"]
        print(f"   Resuming from epoch {start_epoch}, best_metric: {best_metric:.4f}")
    training_start = time.time()
    
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m {int(secs)}s"
        else:
            hours, remainder = divmod(seconds, 3600)
            mins, secs = divmod(remainder, 60)
            return f"{int(hours)}h {int(mins)}m"
    
    print(f"\n🚀 Starting Stage 1 training at {datetime.now().strftime('%H:%M:%S')}")
    if start_epoch > 0:
        print(f"   (Resuming from epoch {start_epoch + 1})")
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs} | Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            inputs = batch["image"].to(device)
            labels = (batch["label"] > 0).float().to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Timing
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        elapsed = time.time() - training_start
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch * (epochs - epoch - 1)
        eta_finish = datetime.now() + timedelta(seconds=eta)
        
        print(f"\n📊 Loss: {avg_loss:.4f}")
        print(f"⏱️  Epoch: {format_time(epoch_duration)} | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)} (~{eta_finish.strftime('%H:%M:%S')})")
        
        # Validation
        val_interval = cfg.get("val_interval", 10)
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validation"):
                    val_in = val_batch["image"].to(device)
                    val_lbl = (val_batch["label"] > 0).float().to(device)
                    val_out = sliding_window_inference(val_in, cfg["patch_size"], 4, model)
                    val_out = torch.argmax(val_out, dim=1, keepdim=True)
                    dice_metric(val_out, val_lbl)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                print(f"\n🎯 Validation Kidney Dice: {metric:.4f}")
                
                if metric > best_metric:
                    best_metric = metric
                    patience_counter = 0  # Reset patience
                    torch.save({"model": model.state_dict(), "epoch": epoch, "dice": metric},
                              checkpoint_dir / "stage1_best.pth")
                    print(f"✅ New best model saved!")
                else:
                    patience_counter += 1
                    print(f"⏳ No improvement ({patience_counter}/{patience} validations)")
                
                print(f"🏆 Best so far: {best_metric:.4f}")
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping triggered! No improvement for {patience} validations.")
                    break
        
        # Save checkpoint after every epoch (for resume)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "patience_counter": patience_counter
        }, checkpoint_dir / "stage1_last.pth")
    
    total_time = time.time() - training_start
    print(f"\n{'='*60}")
    print(f"  STAGE 1 COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best Kidney Dice: {best_metric:.4f}")
    print(f"{'='*60}")
    
    return model


def train_stage2(config: dict, num_epochs: int = None, resume: bool = False):
    """Train Stage 2: Tumor/Cyst segmentation with deep supervision and cyclic LR."""
    print("\n" + "=" * 70)
    print("  STAGE 2: High-Resolution Tumor/Cyst Segmentation")
    print("=" * 70)
    
    cfg = config["stage2"]
    epochs = num_epochs or cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_dir = Path(config["output_dir"]) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    data_dicts = get_data_dicts(config["kits23_dir"])
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    train_files, val_files = data_dicts[:split_idx], data_dicts[split_idx:]
    
    # Limit validation samples for speed
    val_samples = config["data"].get("val_samples", -1)
    if val_samples > 0 and len(val_files) > val_samples:
        val_files = val_files[:val_samples]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Use PersistentDataset for disk caching - first run preprocesses, subsequent runs are instant
    cache_dir = Path(config.get("cache_dir", "./cache")) / "stage2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_ds = PersistentDataset(train_files, get_stage2_transforms(config, True), cache_dir=str(cache_dir / "train"))
    val_ds = PersistentDataset(val_files, get_stage2_transforms(config, False), cache_dir=str(cache_dir / "val"))
    
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)
    
    # Model with deep supervision
    model = SCUNet(
        in_channels=1, out_channels=cfg["out_channels"],
        channels=cfg["channels"], strides=cfg["strides"],
        deep_supervision=cfg["deep_supervision"]
    ).to(device)
    
    print(f"Stage 2 params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with deep supervision wrapper
    base_loss = HybridLoss(class_weights=cfg["class_weights"]).to(device)
    if cfg["deep_supervision"]:
        loss_fn = DeepSupervisionLoss(base_loss)
    else:
        loss_fn = base_loss
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    
    # Cyclic LR for uncertainty ensemble
    clr = config["cyclic_lr"]
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=clr["T_0"], T_mult=clr["T_mult"], eta_min=clr["eta_min"])
    
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    scaler = torch.amp.GradScaler('cuda')
    
    best_metric = -1
    best_metrics = {"kidney": 0, "tumor": 0, "cyst": 0}
    patience_counter = 0
    patience = cfg.get("early_stopping_patience", 50)
    history = {"kidney": [], "tumor": [], "cyst": [], "mean": [], "loss": [], "lr": []}
    ensemble_ckpts = config["ensemble_checkpoints"]
    epoch_times = []
    start_epoch = 0
    
    # Resume from checkpoint if requested
    resume_ckpt = checkpoint_dir / "stage2_last.pth"
    if resume and resume_ckpt.exists():
        print(f"\n📂 Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_metric" in ckpt:
            best_metric = ckpt["best_metric"]
        if "best_metrics" in ckpt:
            best_metrics = ckpt["best_metrics"]
        if "patience_counter" in ckpt:
            patience_counter = ckpt["patience_counter"]
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"   Resuming from epoch {start_epoch}, best_metric: {best_metric:.4f}")
    
    training_start = time.time()
    
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m {int(secs)}s"
        else:
            hours, remainder = divmod(seconds, 3600)
            mins, secs = divmod(remainder, 60)
            return f"{int(hours)}h {int(mins)}m"
    
    post_pred = Compose([AsDiscreted(keys="pred", argmax=True, to_onehot=4)])
    post_label = Compose([AsDiscreted(keys="label", to_onehot=4)])
    
    print(f"\n🚀 Starting Stage 2 training at {datetime.now().strftime('%H:%M:%S')}")
    if start_epoch > 0:
        print(f"   (Resuming from epoch {start_epoch + 1})")
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs} | LR: {lr:.2e} | Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")
        
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        history["loss"].append(avg_loss)
        history["lr"].append(lr)
        
        # Timing
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        elapsed = time.time() - training_start
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch * (epochs - epoch - 1)
        eta_finish = datetime.now() + timedelta(seconds=eta)
        
        print(f"\n📊 Loss: {avg_loss:.4f}")
        print(f"⏱️  Epoch: {format_time(epoch_duration)} | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)} (~{eta_finish.strftime('%H:%M:%S')})")
        
        # Validation
        val_interval = cfg.get("val_interval", 10)
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            model.eval()
            # Temporarily disable deep supervision for validation
            model.deep_supervision = False
            model.decoder.deep_supervision = False
            
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validation"):
                    val_in = val_batch["image"].to(device)
                    val_lbl = val_batch["label"].to(device)
                    
                    val_out = sliding_window_inference(val_in, cfg["patch_size"], 
                                                       config["inference"]["sw_batch_size"], model)
                    # Handle any remaining list outputs
                    if isinstance(val_out, list):
                        val_out = val_out[0]
                    
                    val_out = [post_pred({"pred": i})["pred"] for i in decollate_batch(val_out)]
                    val_lbl = [post_label({"label": i})["label"] for i in decollate_batch(val_lbl)]
                    dice_metric(val_out, val_lbl)
                
                # Re-enable deep supervision for training
                model.deep_supervision = cfg["deep_supervision"]
                model.decoder.deep_supervision = cfg["deep_supervision"]
                
                per_class = dice_metric.aggregate()
                dice_metric.reset()
                
                kidney, tumor, cyst = per_class[0].item(), per_class[1].item(), per_class[2].item()
                mean_dice = per_class.mean().item()
                
                history["kidney"].append(kidney)
                history["tumor"].append(tumor)
                history["cyst"].append(cyst)
                history["mean"].append(mean_dice)
                
                # Detailed results table
                print(f"\n🎯 Validation Results:")
                print(f"   {'-'*40}")
                print(f"   {'Class':<12} {'Dice Score':>12}")
                print(f"   {'-'*40}")
                print(f"   {'Kidney':<12} {kidney:>12.4f}")
                print(f"   {'Tumor':<12} {tumor:>12.4f}")
                print(f"   {'Cyst':<12} {cyst:>12.4f}")
                print(f"   {'-'*40}")
                print(f"   {'Mean':<12} {mean_dice:>12.4f}")
                print(f"   {'-'*40}")
                
                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_metrics = {"kidney": kidney, "tumor": tumor, "cyst": cyst}
                    patience_counter = 0  # Reset patience
                    torch.save({
                        "model": model.state_dict(), "epoch": epoch,
                        "kidney": kidney, "tumor": tumor, "cyst": cyst
                    }, checkpoint_dir / "stage2_best.pth")
                    print(f"   ✅ New best model saved!")
                else:
                    patience_counter += 1
                    print(f"   ⏳ No improvement ({patience_counter}/{patience} validations)")
                
                print(f"\n   🏆 Best Mean Dice so far: {best_metric:.4f}")
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping triggered! No improvement for {patience} validations.")
                    break
        
        # Save ensemble checkpoint
        if (epoch + 1) in ensemble_ckpts:
            torch.save({"model": model.state_dict(), "epoch": epoch},
                      checkpoint_dir / f"stage2_cycle_{epoch+1}.pth")
            print(f"\n💾 Saved ensemble checkpoint at epoch {epoch+1}")
        
        # Save checkpoint after every epoch (for resume)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "best_metrics": best_metrics,
            "patience_counter": patience_counter,
            "history": history
        }, checkpoint_dir / "stage2_last.pth")
    
    # Save history
    with open(Path(config["output_dir"]) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Final summary
    total_time = time.time() - training_start
    print(f"\n{'='*70}")
    print(f"  STAGE 2 COMPLETE")
    print(f"{'='*70}")
    print(f"\n⏱️  Total training time: {format_time(total_time)}")
    print(f"\n🏆 Best Model Results:")
    print(f"   {'-'*40}")
    print(f"   {'Class':<12} {'Dice Score':>12}")
    print(f"   {'-'*40}")
    print(f"   {'Kidney':<12} {best_metrics['kidney']:>12.4f}")
    print(f"   {'Tumor':<12} {best_metrics['tumor']:>12.4f}")
    print(f"   {'Cyst':<12} {best_metrics['cyst']:>12.4f}")
    print(f"   {'-'*40}")
    print(f"   {'Mean':<12} {best_metric:>12.4f}")
    print(f"   {'-'*40}")
    print(f"\n📁 Checkpoints saved to: {checkpoint_dir}")
    print(f"📊 History saved to: {Path(config['output_dir']) / 'training_history.json'}")
    
    return model


# ============================================================================
# INFERENCE WITH CASCADE (Stage 1 ROI → Stage 2 Segmentation)
# ============================================================================
def run_inference(config: dict, use_ensemble: bool = True):
    """Run cascade inference: Stage 1 finds kidney ROI, Stage 2 segments within it."""
    print("\n" + "=" * 70)
    print("  CASCADE INFERENCE" + (" (Ensemble)" if use_ensemble else ""))
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg1 = config["stage1"]
    cfg2 = config["stage2"]
    checkpoint_dir = Path(config["output_dir"]) / "checkpoints"
    output_dir = Path(config["output_dir"]) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================
    # Load Stage 1 Model
    # ========================
    print("\n📦 Loading Stage 1 model (kidney localization)...")
    stage1_model = SCUNet(
        in_channels=1, out_channels=cfg1["out_channels"],
        channels=cfg1["channels"], strides=cfg1["strides"],
        deep_supervision=False
    ).to(device)
    
    stage1_ckpt = checkpoint_dir / "stage1_best.pth"
    if not stage1_ckpt.exists():
        print("❌ Stage 1 model not found! Run training first.")
        return
    stage1_model.load_state_dict(torch.load(stage1_ckpt, map_location=device)["model"])
    stage1_model.eval()
    print(f"✅ Stage 1 loaded: {stage1_ckpt}")
    
    # ========================
    # Load Stage 2 Model(s)
    # ========================
    print("\n📦 Loading Stage 2 model(s) (tumor/cyst segmentation)...")
    # Create with deep_supervision=True to match training, then disable for inference
    stage2_model = SCUNet(
        in_channels=1, out_channels=cfg2["out_channels"],
        channels=cfg2["channels"], strides=cfg2["strides"],
        deep_supervision=cfg2.get("deep_supervision", True)  # Match training config
    ).to(device)
    
    checkpoints = []
    if use_ensemble:
        for ep in config["ensemble_checkpoints"]:
            ckpt_path = checkpoint_dir / f"stage2_cycle_{ep}.pth"
            if ckpt_path.exists():
                checkpoints.append(ckpt_path)
        if (checkpoint_dir / "stage2_best.pth").exists():
            checkpoints.append(checkpoint_dir / "stage2_best.pth")
    
    if not checkpoints:
        checkpoints = [checkpoint_dir / "stage2_best.pth"]
    
    if not checkpoints[0].exists():
        print("❌ Stage 2 model not found! Run training first.")
        return
    
    print(f"✅ Using {len(checkpoints)} Stage 2 checkpoint(s)")
    
    # ========================
    # Inference Transforms (NO CropForeground - keeps spatial alignment!)
    # ========================
    from monai.transforms import Invertd, SaveImaged
    
    cfg2 = config["stage2"]
    data_cfg = config["data"]
    
    # Simple inference transforms without cropping
    inference_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=cfg2["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=data_cfg["intensity_min"],
                            a_max=data_cfg["intensity_max"], b_min=0, b_max=1, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # ========================
    # Run Inference (on validation set only)
    # ========================
    data_dicts = get_data_dicts(config["kits23_dir"])
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    val_dicts = data_dicts[split_idx:]
    
    # Limit samples using cascade_samples
    cascade_samples = config["data"].get("cascade_samples", -1)
    if cascade_samples > 0 and len(val_dicts) > cascade_samples:
        val_dicts = val_dicts[:cascade_samples]
    
    print(f"\n🔄 Processing {len(val_dicts)} validation cases...")
    
    total_start = time.time()
    case_times = []
    
    for i, data_dict in enumerate(val_dicts):
        case_start = time.time()
        case_id = data_dict["case_id"]
        print(f"\n  [{i+1}/{len(val_dicts)}] {case_id}")
        
        # Load and transform data
        print(f"    🎯 Segmenting...", end=" ", flush=True)
        data = inference_transforms({"image": data_dict["image"], "label": data_dict["label"]})
        inputs = data["image"].unsqueeze(0).to(device)
        
        # Ensemble inference
        all_probs = []
        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location=device)
            stage2_model.load_state_dict(ckpt["model"])
            stage2_model.deep_supervision = False
            stage2_model.decoder.deep_supervision = False
            stage2_model.eval()
            
            with torch.no_grad():
                outputs = sliding_window_inference(
                    inputs, cfg2["patch_size"],
                    config["inference"]["sw_batch_size"], stage2_model,
                    overlap=config["inference"]["overlap"]
                )
                if isinstance(outputs, list):
                    outputs = outputs[0]
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)
        
        # Average ensemble
        avg_probs = torch.stack(all_probs).mean(dim=0)
        pred = torch.argmax(avg_probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        print("✓")
        
        # Post-process
        pred = postprocess(pred)
        
        # Resize to original resolution using scipy zoom
        orig = nib.load(data_dict["image"])
        orig_shape = orig.get_fdata().shape
        
        if pred.shape != orig_shape:
            from scipy.ndimage import zoom
            factors = np.array(orig_shape) / np.array(pred.shape)
            pred = zoom(pred, factors, order=0).astype(np.uint8)
        
        # Save with original affine to maintain spatial alignment
        nib.save(nib.Nifti1Image(pred, orig.affine), output_dir / f"{case_id}.nii.gz")
        
        # Timing
        case_time = time.time() - case_start
        case_times.append(case_time)
        avg_time = sum(case_times) / len(case_times)
        eta = avg_time * (len(val_dicts) - i - 1)
        print(f"    💾 Saved: {case_id}.nii.gz ({pred.shape}) | ⏱️ {case_time:.1f}s (avg: {avg_time:.1f}s, ETA: {eta:.0f}s)")
    
    total_time = time.time() - total_start
    print(f"\n✅ Predictions saved to {output_dir}")
    print(f"⏱️  Total: {total_time:.1f}s | Avg per case: {total_time/len(val_dicts):.1f}s")


def postprocess(pred: np.ndarray, min_kidney: int = 1000, min_tumor: int = 50) -> np.ndarray:
    """Post-process predictions with connected component analysis."""
    result = np.zeros_like(pred)
    
    for label, min_size in [(1, min_kidney), (2, min_tumor), (3, min_tumor)]:
        mask = (pred == label).astype(np.uint8)
        if mask.sum() == 0:
            continue
        
        labeled, num = ndimage.label(mask)
        for i in range(1, num + 1):
            comp = (labeled == i)
            if comp.sum() >= min_size:
                result[comp] = label
    
    return result


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate(config: dict):
    """Evaluate predictions on validation set."""
    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)
    
    pred_dir = Path(config["output_dir"]) / "predictions"
    data_dicts = get_data_dicts(config["kits23_dir"])
    
    # Only evaluate validation samples
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    val_dicts = data_dicts[split_idx:]
    
    # Limit samples (same as inference)
    val_samples = config["data"].get("val_samples", -1)
    if val_samples > 0 and len(val_dicts) > val_samples:
        val_dicts = val_dicts[:val_samples]
    
    results = {"kidney": [], "tumor": [], "cyst": []}
    
    for data_dict in tqdm(val_dicts, desc="Evaluating"):
        case_id = data_dict["case_id"]
        pred_path = pred_dir / f"{case_id}.nii.gz"
        
        if not pred_path.exists():
            continue
        
        pred = nib.load(pred_path).get_fdata().astype(np.int32)
        gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
        
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            factors = np.array(gt.shape) / np.array(pred.shape)
            pred = zoom(pred, factors, order=0).astype(np.int32)
        
        for label, name in [(1, "kidney"), (2, "tumor"), (3, "cyst")]:
            p = (pred == label).astype(float)
            g = (gt == label).astype(float)
            inter = (p * g).sum()
            union = p.sum() + g.sum()
            dice = 2 * inter / union if union > 0 else 1.0
            results[name].append(dice)
    
    print("\n" + "-" * 50)
    print(f"{'Class':<12} {'Mean':>10} {'Std':>10}")
    print("-" * 50)
    for name in ["kidney", "tumor", "cyst"]:
        print(f"{name.capitalize():<12} {np.mean(results[name]):>10.4f} {np.std(results[name]):>10.4f}")
    print("-" * 50)
    overall = np.mean([np.mean(results[c]) for c in results])
    print(f"{'Overall':<12} {overall:>10.4f}")
    
    return results


# ============================================================================
# VISUALIZATION (Paper-Ready Figures)
# ============================================================================
def generate_visualizations(config: dict, num_cases: int = 5):
    """Generate comprehensive paper-ready visualizations. Modular and safe."""
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as mpatches
    except ImportError:
        print("❌ matplotlib not installed. Skipping visualizations.")
        return
    
    output_dir = Path(config["output_dir"])
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Custom colormap for segmentation
    colors = ['black', 'green', 'red', 'blue']  # bg, kidney, tumor, cyst
    cmap = ListedColormap(colors)
    legend_patches = [
        mpatches.Patch(color='green', label='Kidney'),
        mpatches.Patch(color='red', label='Tumor'),
        mpatches.Patch(color='blue', label='Cyst'),
    ]
    
    # ========================================================================
    # 1. Training Curves (Both Stages)
    # ========================================================================
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        has_data = False
        
        # Try to load stage-specific or combined history
        history_files = [
            ("stage1_history.json", 0),
            ("stage2_history.json", 1),
        ]
        
        for filename, stage_num in history_files:
            history_path = output_dir / filename
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
                has_data = True
                
                # Plot loss (try different key names)
                loss_keys = ["train_loss", "loss", "epoch_loss"]
                for key in loss_keys:
                    if key in history and history[key]:
                        axes[stage_num, 0].plot(history[key], 'b-', linewidth=2, label='Train')
                        break
                val_loss_keys = ["val_loss", "validation_loss"]
                for key in val_loss_keys:
                    if key in history and history[key]:
                        axes[stage_num, 0].plot(history[key], 'r-', linewidth=2, label='Val')
                        break
                axes[stage_num, 0].set_title(f"Stage {stage_num+1}: Loss", fontsize=12, fontweight='bold')
                axes[stage_num, 0].set_xlabel("Epoch")
                axes[stage_num, 0].set_ylabel("Loss")
                axes[stage_num, 0].legend()
                axes[stage_num, 0].grid(True, alpha=0.3)
                
                # Plot Dice (try different key names)
                dice_keys = ["val_dice", "dice", "mean_dice", "val_mean_dice"]
                for key in dice_keys:
                    if key in history and history[key]:
                        axes[stage_num, 1].plot(history[key], 'g-', linewidth=2)
                        break
                axes[stage_num, 1].set_title(f"Stage {stage_num+1}: Validation Dice", fontsize=12, fontweight='bold')
                axes[stage_num, 1].set_xlabel("Epoch")
                axes[stage_num, 1].set_ylabel("Dice")
                axes[stage_num, 1].set_ylim(0, 1)
                axes[stage_num, 1].grid(True, alpha=0.3)
                
                # Plot LR
                if "lr" in history and history["lr"]:
                    axes[stage_num, 2].plot(history["lr"], 'm-', linewidth=2)
                    axes[stage_num, 2].set_title(f"Stage {stage_num+1}: Learning Rate", fontsize=12, fontweight='bold')
                    axes[stage_num, 2].set_xlabel("Epoch")
                    axes[stage_num, 2].set_ylabel("LR")
                    axes[stage_num, 2].set_yscale('log')
                    axes[stage_num, 2].grid(True, alpha=0.3)
        
        # Also try loading combined training_history.json
        combined_path = output_dir / "training_history.json"
        if combined_path.exists():
            with open(combined_path) as f:
                history = json.load(f)
            print(f"📊 Found training_history.json with keys: {list(history.keys())}")
            has_data = True
            
            # Try to plot whatever is available
            row = 0  # Use first row for combined history
            dice_keys = ["kidney", "tumor", "cyst", "mean"]  # Class names for Dice
            colors = {'kidney': 'green', 'tumor': 'red', 'cyst': 'blue', 'mean': 'purple'}
            
            for key in history:
                if history[key] and isinstance(history[key], list) and len(history[key]) > 0:
                    if "loss" in key.lower():
                        axes[row, 0].plot(history[key], 'b-', linewidth=2, marker='o', label=key)
                    elif key.lower() in dice_keys or "dice" in key.lower():
                        color = colors.get(key.lower(), 'gray')
                        axes[row, 1].plot(history[key], color=color, linewidth=2, marker='o', label=key.capitalize())
                    elif "lr" in key.lower():
                        axes[row, 2].plot(history[key], 'm-', linewidth=2, marker='o', label=key)
            
            # Set titles and formatting
            axes[row, 0].set_title("Training Loss", fontsize=12, fontweight='bold')
            axes[row, 0].set_xlabel("Epoch")
            axes[row, 0].set_ylabel("Loss")
            
            axes[row, 1].set_title("Validation Dice by Class", fontsize=12, fontweight='bold')
            axes[row, 1].set_xlabel("Epoch")
            axes[row, 1].set_ylabel("Dice")
            axes[row, 1].set_ylim(0, 1)
            
            axes[row, 2].set_title("Learning Rate", fontsize=12, fontweight='bold')
            axes[row, 2].set_xlabel("Epoch")
            axes[row, 2].set_ylabel("LR")
            
            for col in range(3):
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        if has_data:
            plt.suptitle("Training Progress", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(fig_dir / "training_curves.png", dpi=150, bbox_inches='tight')
            print(f"✅ Saved: training_curves.png")
        else:
            print(f"⚠️ No training history files found")
        plt.close()
    except Exception as e:
        print(f"⚠️ Training curves failed: {e}")
    
    # ========================================================================
    # 2. Multi-View Slice Comparisons (Axial, Coronal, Sagittal)
    # ========================================================================
    try:
        pred_dir = output_dir / "predictions"
        data_dicts = get_data_dicts(config["kits23_dir"])
        split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
        val_dicts = data_dicts[split_idx:][:num_cases]
        
        for data_dict in val_dicts:
            case_id = data_dict["case_id"]
            pred_path = pred_dir / f"{case_id}.nii.gz"
            
            if not pred_path.exists():
                continue
            
            # Load data
            img = nib.load(data_dict["image"]).get_fdata()
            gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
            pred = nib.load(pred_path).get_fdata().astype(np.int32)
            
            # Find best slices for each view
            tumor_cyst = (gt >= 2).astype(int)
            axial_slice = tumor_cyst.sum(axis=(1, 2)).argmax()
            coronal_slice = tumor_cyst.sum(axis=(0, 2)).argmax()
            sagittal_slice = tumor_cyst.sum(axis=(0, 1)).argmax()
            
            # Create multi-view figure
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            views = [
                ("Axial", axial_slice, lambda x: x[axial_slice]),
                ("Coronal", coronal_slice, lambda x: x[:, coronal_slice, :]),
                ("Sagittal", sagittal_slice, lambda x: x[:, :, sagittal_slice]),
            ]
            
            for row, (view_name, slice_idx, get_slice) in enumerate(views):
                img_slice = get_slice(img)
                gt_slice = get_slice(gt)
                pred_slice = get_slice(pred)
                
                # CT image
                axes[row, 0].imshow(img_slice.T if row > 0 else img_slice, cmap='gray', vmin=-100, vmax=300, origin='lower')
                axes[row, 0].set_title(f"{view_name} - CT", fontsize=10, fontweight='bold')
                axes[row, 0].axis('off')
                
                # Ground truth
                axes[row, 1].imshow(img_slice.T if row > 0 else img_slice, cmap='gray', vmin=-100, vmax=300, origin='lower')
                gt_overlay = np.ma.masked_where(gt_slice == 0, gt_slice)
                axes[row, 1].imshow((gt_overlay.T if row > 0 else gt_overlay), cmap=cmap, alpha=0.6, vmin=0, vmax=3, origin='lower')
                axes[row, 1].set_title(f"{view_name} - GT", fontsize=10, fontweight='bold')
                axes[row, 1].axis('off')
                
                # Prediction
                axes[row, 2].imshow(img_slice.T if row > 0 else img_slice, cmap='gray', vmin=-100, vmax=300, origin='lower')
                pred_overlay = np.ma.masked_where(pred_slice == 0, pred_slice)
                axes[row, 2].imshow((pred_overlay.T if row > 0 else pred_overlay), cmap=cmap, alpha=0.6, vmin=0, vmax=3, origin='lower')
                axes[row, 2].set_title(f"{view_name} - Pred", fontsize=10, fontweight='bold')
                axes[row, 2].axis('off')
                
                # Error map
                diff = (gt_slice != pred_slice).astype(int)
                axes[row, 3].imshow(img_slice.T if row > 0 else img_slice, cmap='gray', vmin=-100, vmax=300, origin='lower')
                diff_overlay = np.ma.masked_where(diff == 0, diff)
                axes[row, 3].imshow((diff_overlay.T if row > 0 else diff_overlay), cmap='Reds', alpha=0.7, origin='lower')
                axes[row, 3].set_title(f"{view_name} - Errors", fontsize=10, fontweight='bold')
                axes[row, 3].axis('off')
            
            # Add legend
            fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)
            plt.suptitle(f"{case_id} - Multi-View Comparison", fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(fig_dir / f"{case_id}_multiview.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {case_id}_multiview.png")
            
    except Exception as e:
        print(f"⚠️ Multi-view comparisons failed: {e}")
    
    # ========================================================================
    # 3. 3D Maximum Intensity Projection (MIP) - Shows full kidney/tumor
    # ========================================================================
    try:
        pred_dir = output_dir / "predictions"
        data_dicts = get_data_dicts(config["kits23_dir"])
        split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
        val_dicts = data_dicts[split_idx:][:num_cases]
        
        for data_dict in val_dicts:
            case_id = data_dict["case_id"]
            pred_path = pred_dir / f"{case_id}.nii.gz"
            
            if not pred_path.exists():
                continue
            
            gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
            pred = nib.load(pred_path).get_fdata().astype(np.int32)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            for row, (data, title) in enumerate([(gt, "Ground Truth"), (pred, "Prediction")]):
                # Axial MIP
                axes[row, 0].imshow(data.max(axis=0).T, cmap=cmap, vmin=0, vmax=3, origin='lower')
                axes[row, 0].set_title(f"{title} - Axial MIP", fontsize=12, fontweight='bold')
                axes[row, 0].axis('off')
                
                # Coronal MIP
                axes[row, 1].imshow(data.max(axis=1).T, cmap=cmap, vmin=0, vmax=3, origin='lower')
                axes[row, 1].set_title(f"{title} - Coronal MIP", fontsize=12, fontweight='bold')
                axes[row, 1].axis('off')
                
                # Sagittal MIP
                axes[row, 2].imshow(data.max(axis=2).T, cmap=cmap, vmin=0, vmax=3, origin='lower')
                axes[row, 2].set_title(f"{title} - Sagittal MIP", fontsize=12, fontweight='bold')
                axes[row, 2].axis('off')
            
            fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)
            plt.suptitle(f"{case_id} - 3D Maximum Intensity Projection", fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(fig_dir / f"{case_id}_mip.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {case_id}_mip.png")
            
    except Exception as e:
        print(f"⚠️ MIP visualization failed: {e}")
    
    # ========================================================================
    # 4. Per-Case Dice Bar Chart
    # ========================================================================
    try:
        pred_dir = output_dir / "predictions"
        data_dicts = get_data_dicts(config["kits23_dir"])
        split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
        val_dicts = data_dicts[split_idx:]
        
        case_ids = []
        kidney_dice = []
        tumor_dice = []
        cyst_dice = []
        
        for data_dict in val_dicts:
            case_id = data_dict["case_id"]
            pred_path = pred_dir / f"{case_id}.nii.gz"
            if not pred_path.exists():
                continue
            
            pred = nib.load(pred_path).get_fdata().astype(np.int32)
            gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
            
            case_ids.append(case_id.replace("case_", ""))
            
            for label, lst in [(1, kidney_dice), (2, tumor_dice), (3, cyst_dice)]:
                p = (pred == label).astype(float)
                g = (gt == label).astype(float)
                inter = (p * g).sum()
                union = p.sum() + g.sum()
                dice = 2 * inter / union if union > 0 else 1.0
                lst.append(dice)
        
        if case_ids:
            x = np.arange(len(case_ids))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(max(12, len(case_ids) * 0.8), 6))
            ax.bar(x - width, kidney_dice, width, label='Kidney', color='green', alpha=0.7)
            ax.bar(x, tumor_dice, width, label='Tumor', color='red', alpha=0.7)
            ax.bar(x + width, cyst_dice, width, label='Cyst', color='blue', alpha=0.7)
            
            ax.set_xlabel('Case ID', fontsize=12)
            ax.set_ylabel('Dice Score', fontsize=12)
            ax.set_title('Dice Score per Case', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(case_ids, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add mean lines
            ax.axhline(y=np.mean(kidney_dice), color='green', linestyle='--', alpha=0.5, label=f'Kidney Mean: {np.mean(kidney_dice):.3f}')
            ax.axhline(y=np.mean(tumor_dice), color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=np.mean(cyst_dice), color='blue', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(fig_dir / "dice_per_case.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: dice_per_case.png")
    except Exception as e:
        print(f"⚠️ Per-case bar chart failed: {e}")
    
    # ========================================================================
    # 5. Dice Score Boxplot
    # ========================================================================
    try:
        pred_dir = output_dir / "predictions"
        data_dicts = get_data_dicts(config["kits23_dir"])
        split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
        val_dicts = data_dicts[split_idx:]
        
        results = {"Kidney": [], "Tumor": [], "Cyst": []}
        
        for data_dict in val_dicts:
            case_id = data_dict["case_id"]
            pred_path = pred_dir / f"{case_id}.nii.gz"
            if not pred_path.exists():
                continue
            
            pred = nib.load(pred_path).get_fdata().astype(np.int32)
            gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
            
            for label, name in [(1, "Kidney"), (2, "Tumor"), (3, "Cyst")]:
                p = (pred == label).astype(float)
                g = (gt == label).astype(float)
                inter = (p * g).sum()
                union = p.sum() + g.sum()
                dice = 2 * inter / union if union > 0 else 1.0
                results[name].append(dice)
        
        if any(results.values()):
            fig, ax = plt.subplots(figsize=(8, 6))
            data = [results["Kidney"], results["Tumor"], results["Cyst"]]
            bp = ax.boxplot(data, labels=["Kidney", "Tumor", "Cyst"], patch_artist=True)
            
            colors_bp = ['green', 'red', 'blue']
            for patch, color in zip(bp['boxes'], colors_bp):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_ylabel("Dice Score", fontsize=12)
            ax.set_title("Dice Score Distribution by Class", fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add mean values
            means = [np.mean(d) for d in data]
            for i, mean in enumerate(means):
                ax.text(i+1, mean + 0.02, f'{mean:.3f}', ha='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(fig_dir / "dice_boxplot.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: dice_boxplot.png")
    except Exception as e:
        print(f"⚠️ Dice boxplot failed: {e}")
    
    # ========================================================================
    # 6. Volume Comparison (GT vs Pred)
    # ========================================================================
    try:
        pred_dir = output_dir / "predictions"
        data_dicts = get_data_dicts(config["kits23_dir"])
        split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
        val_dicts = data_dicts[split_idx:]
        
        gt_volumes = {"Kidney": [], "Tumor": [], "Cyst": []}
        pred_volumes = {"Kidney": [], "Tumor": [], "Cyst": []}
        
        for data_dict in val_dicts:
            case_id = data_dict["case_id"]
            pred_path = pred_dir / f"{case_id}.nii.gz"
            if not pred_path.exists():
                continue
            
            pred = nib.load(pred_path).get_fdata().astype(np.int32)
            gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
            
            for label, name in [(1, "Kidney"), (2, "Tumor"), (3, "Cyst")]:
                gt_volumes[name].append((gt == label).sum())
                pred_volumes[name].append((pred == label).sum())
        
        if any(gt_volumes.values()):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, (name, color) in enumerate([("Kidney", 'green'), ("Tumor", 'red'), ("Cyst", 'blue')]):
                gt_v = np.array(gt_volumes[name]) / 1000  # Convert to K voxels
                pred_v = np.array(pred_volumes[name]) / 1000
                
                max_val = max(gt_v.max(), pred_v.max()) * 1.1
                axes[i].scatter(gt_v, pred_v, c=color, alpha=0.6, s=50)
                axes[i].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect')
                axes[i].set_xlabel("Ground Truth (K voxels)", fontsize=10)
                axes[i].set_ylabel("Prediction (K voxels)", fontsize=10)
                axes[i].set_title(f"{name} Volume Correlation", fontsize=12, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlim(0, max_val)
                axes[i].set_ylim(0, max_val)
                
                # Add correlation coefficient
                if len(gt_v) > 1:
                    corr = np.corrcoef(gt_v, pred_v)[0, 1]
                    axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes, 
                                fontsize=10, verticalalignment='top', fontweight='bold')
            
            plt.suptitle("Volume Correlation: Ground Truth vs Prediction", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(fig_dir / "volume_correlation.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: volume_correlation.png")
    except Exception as e:
        print(f"⚠️ Volume correlation failed: {e}")
    
    # ========================================================================
    # 7. Summary Statistics Table (as image)
    # ========================================================================
    try:
        pred_dir = output_dir / "predictions"
        data_dicts = get_data_dicts(config["kits23_dir"])
        split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
        val_dicts = data_dicts[split_idx:]
        
        results = {"Kidney": [], "Tumor": [], "Cyst": []}
        
        for data_dict in val_dicts:
            case_id = data_dict["case_id"]
            pred_path = pred_dir / f"{case_id}.nii.gz"
            if not pred_path.exists():
                continue
            
            pred = nib.load(pred_path).get_fdata().astype(np.int32)
            gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
            
            for label, name in [(1, "Kidney"), (2, "Tumor"), (3, "Cyst")]:
                p = (pred == label).astype(float)
                g = (gt == label).astype(float)
                inter = (p * g).sum()
                union = p.sum() + g.sum()
                dice = 2 * inter / union if union > 0 else 1.0
                results[name].append(dice)
        
        if any(results.values()):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis('off')
            
            # Create table data
            table_data = [
                ["Class", "Mean Dice", "Std", "Min", "Max", "N"],
                ["Kidney", f"{np.mean(results['Kidney']):.4f}", f"{np.std(results['Kidney']):.4f}",
                 f"{np.min(results['Kidney']):.4f}", f"{np.max(results['Kidney']):.4f}", str(len(results['Kidney']))],
                ["Tumor", f"{np.mean(results['Tumor']):.4f}", f"{np.std(results['Tumor']):.4f}",
                 f"{np.min(results['Tumor']):.4f}", f"{np.max(results['Tumor']):.4f}", str(len(results['Tumor']))],
                ["Cyst", f"{np.mean(results['Cyst']):.4f}", f"{np.std(results['Cyst']):.4f}",
                 f"{np.min(results['Cyst']):.4f}", f"{np.max(results['Cyst']):.4f}", str(len(results['Cyst']))],
                ["Overall", f"{np.mean([np.mean(results[c]) for c in results]):.4f}", "-", "-", "-", "-"],
            ]
            
            table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                           colWidths=[0.15, 0.15, 0.12, 0.12, 0.12, 0.08])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.8)
            
            # Style header
            for j in range(len(table_data[0])):
                table[(0, j)].set_facecolor('#4472C4')
                table[(0, j)].set_text_props(color='white', fontweight='bold')
            
            # Style class names
            colors_table = ['white', '#C6EFCE', '#FFC7CE', '#BDD7EE', '#F2F2F2']
            for i in range(1, len(table_data)):
                table[(i, 0)].set_facecolor(colors_table[i])
                table[(i, 0)].set_text_props(fontweight='bold')
            
            plt.title("Segmentation Results Summary", fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(fig_dir / "summary_table.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: summary_table.png")
    except Exception as e:
        print(f"⚠️ Summary table failed: {e}")
    
    print(f"\n📁 Figures saved to: {fig_dir}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="KiTS23 Hybrid SOTA Segmentation")
    parser.add_argument("--mode", choices=["train", "train_stage1", "train_stage2", 
                                           "inference", "evaluate", "visualize"], default="train")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 epochs per stage)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--no_ensemble", action="store_true", help="Disable ensemble inference")
    args = parser.parse_args()
    
    config = CONFIG.copy()
    set_determinism(config["seed"])
    
    # Quick mode
    if args.quick:
        stage1_epochs = config["stage1"]["quick_epochs"]
        stage2_epochs = config["stage2"]["quick_epochs"]
    else:
        stage1_epochs = args.epochs or config["stage1"]["epochs"]
        stage2_epochs = args.epochs or config["stage2"]["epochs"]
    
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  KiTS23 HYBRID SOTA SEGMENTATION")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Quick: {args.quick}")
    print(f"Resume: {args.resume}")
    print(f"Stage 1 epochs: {stage1_epochs}")
    print(f"Stage 2 epochs: {stage2_epochs}")
    
    # ========================
    # FULL PIPELINE (train mode does everything)
    # ========================
    if args.mode == "train":
        # Step 1: Train Stage 1 (Kidney Localization)
        train_stage1(config, stage1_epochs, resume=args.resume)
        
        # Step 2: Train Stage 2 (Tumor/Cyst Segmentation)
        train_stage2(config, stage2_epochs, resume=args.resume)
        
        # Step 3: Run Cascade Inference
        print("\n" + "=" * 70)
        print("  RUNNING CASCADE INFERENCE")
        print("=" * 70)
        # Use fewer samples for quick tests
        if args.quick:
            config["data"]["cascade_samples"] = 3  # Only 3 samples for quick test
        run_inference(config, use_ensemble=not args.no_ensemble)
        
        # Step 4: Evaluate Results
        print("\n" + "=" * 70)
        print("  EVALUATING RESULTS")
        print("=" * 70)
        evaluate(config)
        
        # Step 5: Generate Visualizations (safe - won't crash if fails)
        try:
            generate_visualizations(config)
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
    
    # Individual stages (for debugging/resume)
    elif args.mode == "train_stage1":
        train_stage1(config, stage1_epochs, resume=args.resume)
    
    elif args.mode == "train_stage2":
        train_stage2(config, stage2_epochs, resume=args.resume)
    
    elif args.mode == "inference":
        run_inference(config, use_ensemble=not args.no_ensemble)
    
    elif args.mode == "evaluate":
        evaluate(config)
    
    elif args.mode == "visualize":
        generate_visualizations(config)
    
    print("\n" + "=" * 70)
    print("  ✅ ALL DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
