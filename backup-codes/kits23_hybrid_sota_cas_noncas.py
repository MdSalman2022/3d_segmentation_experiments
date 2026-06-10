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
    python kits23_hybrid_sota_cas_noncas.py --mode train --quick     # Quick test (5 epochs)
    python kits23_hybrid_sota_cas_noncas.py --mode train             # Full training
    python kits23_hybrid_sota_cas_noncas.py --mode inference
    python kits23_hybrid_sota_cas_noncas.py --mode evaluate
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
        "epochs": 300,
        "quick_epochs": 1,
        "val_interval": 1,
        "early_stopping_patience": 20,  # Stop if no improvement for 30 validations
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
        "epochs": 500,
        "quick_epochs": 1,
        "val_interval": 1,
        "early_stopping_patience": 30,  # Stop if no improvement for 50 validations
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
        "val_samples": 20,    # Limit validation to 20 samples for speed (set to -1 for all)
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
def run_inference(config: dict, use_ensemble: bool = True, use_cascade: bool = True):
    """Run inference. If use_cascade=True, Stage 1 finds kidney ROI first."""
    mode_str = "CASCADE " if use_cascade else ""
    ensemble_str = " (Ensemble)" if use_ensemble else ""
    print("\n" + "=" * 70)
    print(f"  {mode_str}INFERENCE{ensemble_str}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg1 = config["stage1"]
    cfg2 = config["stage2"]
    checkpoint_dir = Path(config["output_dir"]) / "checkpoints"
    # Save to different directories for comparison
    pred_subdir = "predictions_cascade" if use_cascade else "predictions_nocascade"
    output_dir = Path(config["output_dir"]) / pred_subdir
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
    # Transforms
    # ========================
    stage1_transforms = get_stage1_transforms(config, training=False)
    stage2_transforms = get_stage2_transforms(config, training=False)
    
    # ========================
    # Run Cascade Inference (on validation set only)
    # ========================
    data_dicts = get_data_dicts(config["kits23_dir"])
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    val_dicts = data_dicts[split_idx:]
    
    # Limit samples for speed
    val_samples = config["data"].get("val_samples", -1)
    if val_samples > 0 and len(val_dicts) > val_samples:
        val_dicts = val_dicts[:val_samples]
    
    print(f"\n🔄 Processing {len(val_dicts)} validation cases...")
    
    for i, data_dict in enumerate(val_dicts):
        case_id = data_dict["case_id"]
        print(f"\n  [{i+1}/{len(val_dicts)}] {case_id}")
        
        bbox = None  # Default: no cropping
        
        # --- Stage 1: Find kidney ROI (only if using cascade) ---
        if use_cascade:
            print(f"    🔍 Stage 1: Finding kidney ROI...", end=" ", flush=True)
            data1 = stage1_transforms({"image": data_dict["image"], "label": data_dict["label"]})
            inputs1 = data1["image"].unsqueeze(0).to(device)
            
            with torch.no_grad():
                kidney_out = sliding_window_inference(
                    inputs1, cfg1["patch_size"], 4, stage1_model,
                    overlap=0.25
                )
                kidney_pred = torch.argmax(kidney_out, dim=1).squeeze().cpu().numpy()
            print("✓")
            
            # Find kidney bounding box with dilation
            kidney_mask = kidney_pred > 0
            if kidney_mask.sum() == 0:
                print(f"⚠️ No kidney detected in {case_id}, using full image")
            else:
                # Dilate kidney mask for margin
                struct = generate_binary_structure(3, 2)
                dilated = binary_dilation(kidney_mask, struct, iterations=cfg1.get("dilation_size", 11))
                
                # Get bounding box
                coords = np.where(dilated)
                bbox = {
                    "z": (max(0, coords[0].min() - 10), min(kidney_mask.shape[0], coords[0].max() + 10)),
                    "y": (max(0, coords[1].min() - 10), min(kidney_mask.shape[1], coords[1].max() + 10)),
                    "x": (max(0, coords[2].min() - 10), min(kidney_mask.shape[2], coords[2].max() + 10)),
                }
        
        # --- Stage 2: Segment tumor/cyst ---
        print(f"    🎯 Stage 2: Segmenting tumor/cyst...", end=" ", flush=True)
        data2 = stage2_transforms({"image": data_dict["image"], "label": data_dict["label"]})
        inputs2 = data2["image"].unsqueeze(0).to(device)
        orig_shape = inputs2.shape[2:]
        
        # Crop to ROI if available
        if bbox is not None:
            # Scale bbox to Stage 2 resolution
            scale_z = orig_shape[0] / kidney_mask.shape[0]
            scale_y = orig_shape[1] / kidney_mask.shape[1]
            scale_x = orig_shape[2] / kidney_mask.shape[2]
            
            z1, z2 = int(bbox["z"][0] * scale_z), int(bbox["z"][1] * scale_z)
            y1, y2 = int(bbox["y"][0] * scale_y), int(bbox["y"][1] * scale_y)
            x1, x2 = int(bbox["x"][0] * scale_x), int(bbox["x"][1] * scale_x)
            
            cropped = inputs2[:, :, z1:z2, y1:y2, x1:x2]
        else:
            cropped = inputs2
            z1, y1, x1 = 0, 0, 0
        
        # Ensemble inference
        all_probs = []
        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location=device)
            stage2_model.load_state_dict(ckpt["model"])
            # Disable deep supervision for inference (outputs single tensor)
            stage2_model.deep_supervision = False
            stage2_model.decoder.deep_supervision = False
            stage2_model.eval()
            
            with torch.no_grad():
                outputs = sliding_window_inference(
                    cropped, cfg2["patch_size"],
                    config["inference"]["sw_batch_size"], stage2_model,
                    overlap=config["inference"]["overlap"]
                )
                if isinstance(outputs, list):
                    outputs = outputs[0]
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)
        
        # Average ensemble
        avg_probs = torch.stack(all_probs).mean(dim=0)
        pred_cropped = torch.argmax(avg_probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        print("✓")
        
        # Place back in full image
        if bbox is not None:
            pred_full = np.zeros(orig_shape, dtype=np.uint8)
            pred_full[z1:z2, y1:y2, x1:x2] = pred_cropped
        else:
            pred_full = pred_cropped
        
        # Post-process
        pred_full = postprocess(pred_full)
        
        # Resize to original resolution before saving
        orig = nib.load(data_dict["image"])
        orig_data_shape = orig.get_fdata().shape
        if pred_full.shape != orig_data_shape:
            from scipy.ndimage import zoom
            factors = np.array(orig_data_shape) / np.array(pred_full.shape)
            pred_full = zoom(pred_full, factors, order=0).astype(np.uint8)
        
        # Save at original resolution
        nib.save(nib.Nifti1Image(pred_full, orig.affine), output_dir / f"{case_id}.nii.gz")
        print(f"    💾 Saved: {case_id}.nii.gz ({pred_full.shape})")
    
    print(f"\n✅ Cascade predictions saved to {output_dir}")


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
def evaluate(config: dict, pred_dir: str = None):
    """Evaluate predictions on validation set."""
    if pred_dir is None:
        pred_dir = Path(config["output_dir"]) / "predictions_cascade"
    else:
        pred_dir = Path(pred_dir)
    
    print("\n" + "=" * 70)
    print(f"  EVALUATION: {pred_dir.name}")
    print("=" * 70)
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
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="KiTS23 Hybrid SOTA Segmentation")
    parser.add_argument("--mode", choices=["train", "train_stage1", "train_stage2", 
                                           "inference", "evaluate"], default="train")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 epochs per stage)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--no_ensemble", action="store_true", help="Disable ensemble inference")
    parser.add_argument("--no_cascade", action="store_true", help="Disable cascade (use Stage 2 on full image)")
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
        
        # Use fewer samples for quick tests
        if args.quick:
            config["data"]["val_samples"] = 3  # Only 3 samples for quick test
        
        # Step 3a: Run NON-CASCADE Inference (Stage 2 only, full image)
        print("\n" + "=" * 70)
        print("  RUNNING NON-CASCADE INFERENCE (Stage 2 on full image)")
        print("=" * 70)
        run_inference(config, use_ensemble=not args.no_ensemble, use_cascade=False)
        
        # Step 3b: Evaluate Non-Cascade Results
        nocascade_dir = Path(config["output_dir"]) / "predictions_nocascade"
        evaluate(config, pred_dir=str(nocascade_dir))
        
        # Step 4a: Run CASCADE Inference (Stage 1 → Stage 2)
        print("\n" + "=" * 70)
        print("  RUNNING CASCADE INFERENCE (Stage 1 → Stage 2)")
        print("=" * 70)
        run_inference(config, use_ensemble=not args.no_ensemble, use_cascade=True)
        
        # Step 4b: Evaluate Cascade Results
        cascade_dir = Path(config["output_dir"]) / "predictions_cascade"
        evaluate(config, pred_dir=str(cascade_dir))
    
    # Individual stages (for debugging/resume)
    elif args.mode == "train_stage1":
        train_stage1(config, stage1_epochs, resume=args.resume)
    
    elif args.mode == "train_stage2":
        train_stage2(config, stage2_epochs, resume=args.resume)
    
    elif args.mode == "inference":
        run_inference(config, use_ensemble=not args.no_ensemble, use_cascade=not args.no_cascade)
    
    elif args.mode == "evaluate":
        evaluate(config)
    
    print("\n" + "=" * 70)
    print("  ✅ ALL DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
