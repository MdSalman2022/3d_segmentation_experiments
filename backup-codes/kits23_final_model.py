"""
KiTS23 Final Model - High-Performance Kidney/Tumor/Cyst Segmentation
======================================================================
Based on MICCAI 2023 KiTS23 Challenge 2nd place solution approach.

Target Performance:
- Kidney Dice >= 0.70 (2nd place: 0.948 kidney+masses)
- Tumor Dice >= 0.50 (2nd place: 0.738)
- Cyst Dice >= 0.50

Key Strategies:
1. Multi-resolution ensemble (3 model configurations)
2. ResidualEncoderUNet architecture (better gradient flow)
3. Advanced post-processing (kidney + tumor refinement)
4. Majority voting aggregation

Hardware: Optimized for RTX 5090 (32GB VRAM, 8 cores/16 threads)

Usage:
    python kits23_final_model.py --mode train --quick     # Quick test (~30 min)
    python kits23_final_model.py --mode train             # Standard training
    python kits23_final_model.py --mode train --full      # Full training (best results)
    python kits23_final_model.py --mode inference
    python kits23_final_model.py --mode evaluate

Author: Based on khuhm/KiTS23-2nd-place, deepdrivepl/kits23, anhtuduong/kits23-nnunet
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
import argparse
import warnings
from pathlib import Path
from typing import Sequence, Tuple, Union, List, Dict, Optional, Callable
from datetime import datetime, timedelta
from collections import OrderedDict
import time
import copy

import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, label as scipy_label
from scipy.stats import mode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

# Speed optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandRotate90d, RandFlipd, RandShiftIntensityd, RandScaleIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd, EnsureTyped,
    AsDiscreted, Rand3DElasticd, RandRotated, RandZoomd, NormalizeIntensityd,
    ToTensord,
)
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, PersistentDataset, DataLoader, Dataset, decollate_batch

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION - Optimized for RTX 5090 (32GB VRAM)
# ============================================================================
CONFIG = {
    "kits23_dir": "./kits23/dataset",
    "output_dir": "./output/kits23_final",
    "cache_dir": "./output/kits23_final/cache",
    "seed": 42,
    
    # ========================================================================
    # Model Configurations (following 2nd place approach)
    # ========================================================================
    "models": {
        # Model 1: Low-resolution PlainConvUNet
        "lowres_plain": {
            "name": "Lowres PlainConv",
            "spacing": (3.0, 3.0, 3.0),  # Coarse resolution for context
            "patch_size": (128, 128, 128),
            "channels": (32, 64, 128, 256, 320, 320),
            "strides": (1, 2, 2, 2, 2, 2),  # First stride=1, rest downsample
            "kernel_sizes": (3, 3, 3, 3, 3, 3),
            "unet_class": "PlainConvUNet",
            "batch_size": 2,
            "num_samples": 4,
        },
        
        # Model 2: Low-resolution ResidualEncoderUNet
        "lowres_residual": {
            "name": "Lowres Residual",
            "spacing": (3.0, 3.0, 3.0),
            "patch_size": (128, 128, 128),
            "channels": (32, 64, 128, 256, 320, 320),
            "strides": (1, 2, 2, 2, 2, 2),
            "kernel_sizes": (3, 3, 3, 3, 3, 3),
            "unet_class": "ResidualEncoderUNet",
            "batch_size": 2,
            "num_samples": 4,
        },
        
        # Model 3: Full-resolution PlainConvUNet (memory-optimized)
        "fullres_batch4": {
            "name": "Fullres Batch2",
            "spacing": (1.5, 1.5, 1.5),  # Fine resolution (reduced from 1.0 to prevent OOM)
            "patch_size": (96, 96, 96),   # Smaller patches to fit in VRAM
            "channels": (32, 64, 128, 256, 320, 320),
            "strides": (1, 2, 2, 2, 2, 2),
            "kernel_sizes": (3, 3, 3, 3, 3, 3),
            "unet_class": "PlainConvUNet",
            "batch_size": 2,  # Reduced from 4 to prevent OOM
            "num_samples": 2,  # Reduced samples per volume
        },
    },
    
    # ========================================================================
    # Training Settings
    # ========================================================================
    "training": {
        "epochs_quick": 30,  # Minimum for tumor/cyst detection
        "epochs_standard": 200,
        "epochs_full": 500,
        "learning_rate": 1e-2,  # nnU-Net default
        "weight_decay": 3e-5,
        "deep_supervision": True,
        "deep_supervision_weights": [1.0, 0.5, 0.25, 0.125, 0.0625],
        "val_interval": 5,
        "early_stopping_patience": 50,
        "num_workers": 8,  # 16 threads available
    },
    
    # ========================================================================
    # Data Settings
    # ========================================================================
    "data": {
        "intensity_min": -79,  # CT kidney window
        "intensity_max": 304,
        "train_val_split": 0.85,
        "num_classes": 4,  # bg, kidney, tumor, cyst
    },
    
    # ========================================================================
    # Inference Settings
    # ========================================================================
    "inference": {
        "sw_batch_size": 4,
        "overlap": 0.5,
        "use_tta": False,  # Test-time augmentation (slower but better)
    },
    
    # ========================================================================
    # Post-Processing Settings (critical for performance)
    # ========================================================================
    "postprocess": {
        "kidney_min_size": 1000,  # Minimum kidney component size (voxels)
        "tumor_min_size": 50,     # Minimum tumor component size
        "cyst_min_size": 50,      # Minimum cyst component size
        "dilation_iterations": 2, # Dilation for tumor-kidney connection check
    },
}


# ============================================================================
# LEARNING RATE SCHEDULER (nnU-Net Polynomial Decay)
# ============================================================================
class PolynomialLR(_LRScheduler):
    """Polynomial learning rate decay (nnU-Net default)."""
    
    def __init__(self, optimizer, total_iters, power=0.9, last_epoch=-1):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [0.0 for _ in self.base_lrs]
        return [
            base_lr * (1 - self.last_epoch / self.total_iters) ** self.power
            for base_lr in self.base_lrs
        ]

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class DiceCELossWithDS(nn.Module):
    """Dice + Cross-Entropy Loss with Deep Supervision and Class Weighting.
    
    Class weights prioritize tumor and cyst (smaller structures).
    """
    
    def __init__(self, weights: List[float] = None, include_background: bool = False,
                 class_weights: List[float] = None):
        super().__init__()
        self.weights = weights or [1.0]
        
        # Class weights: [bg, kidney, tumor, cyst]
        # Higher weight = more importance during training
        if class_weights is None:
            # Default: prioritize tumor (10x) and cyst (8x) over kidney (1x)
            class_weights = [0.1, 1.0, 10.0, 8.0]
        
        self.dice = DiceLoss(include_background=include_background, 
                            to_onehot_y=True, softmax=True)
        
        # Weighted cross-entropy for class imbalance
        self.register_buffer('ce_weights', torch.tensor(class_weights, dtype=torch.float32))
        self.ce = None  # Will be created on first forward pass
    
    def forward(self, preds: Union[torch.Tensor, List[torch.Tensor]], 
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Either single tensor or list of tensors (deep supervision)
            target: (B, 1, D, H, W) ground truth
        """
        if not isinstance(preds, list):
            preds = [preds]
        
        total_loss = 0.0
        for i, pred in enumerate(preds):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            if weight == 0:
                continue
                
            # Downsample target if needed
            if pred.shape[2:] != target.shape[2:]:
                target_ds = F.interpolate(target.float(), size=pred.shape[2:], 
                                         mode='nearest').long()
            else:
                target_ds = target
            
            # Dice loss (already handles class imbalance somewhat)
            dice_loss = self.dice(pred, target_ds)
            
            # Weighted cross-entropy
            ce_loss = F.cross_entropy(
                pred, target_ds.squeeze(1).long(),
                weight=self.ce_weights.to(pred.device)
            )
            
            # Combine: more weight on CE to push tumor/cyst learning
            total_loss += weight * (0.5 * dice_loss + 0.5 * ce_loss)
        
        return total_loss


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================
class ConvBlock(nn.Module):
    """Basic 3D convolution block with InstanceNorm and LeakyReLU."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual convolution block (better gradient flow for deep networks)."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        
        # Skip connection
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )
        else:
            self.skip = nn.Identity()
        
        self.relu = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + identity)


class PlainConvUNet(nn.Module):
    """Plain Convolution U-Net (nnU-Net style)."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 channels: Tuple[int, ...] = (32, 64, 128, 256, 320, 320),
                 strides: Tuple[int, ...] = (1, 2, 2, 2, 2, 2),
                 kernel_sizes: Tuple[int, ...] = (3, 3, 3, 3, 3, 3),
                 deep_supervision: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.n_stages = len(channels)
        
        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for i, (ch, stride, ks) in enumerate(zip(channels, strides, kernel_sizes)):
            self.encoders.append(ConvBlock(prev_ch, ch, ks, stride, dropout))
            prev_ch = ch
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        reversed_ch = list(reversed(channels))
        reversed_strides = list(reversed(strides))
        
        # Skip the bottleneck layer for decoder
        for i in range(len(reversed_ch) - 1):
            in_ch = reversed_ch[i]
            skip_ch = reversed_ch[i + 1]
            out_ch = skip_ch
            
            if reversed_strides[i] > 1:
                self.upconvs.append(
                    nn.ConvTranspose3d(in_ch, in_ch, reversed_strides[i], 
                                      stride=reversed_strides[i])
                )
            else:
                self.upconvs.append(nn.Identity())
            
            self.decoders.append(ConvBlock(in_ch + skip_ch, out_ch))
        
        # Output heads
        self.out_conv = nn.Conv3d(channels[0], out_channels, 1)
        
        # Deep supervision heads - use decoder output channels (reversed order)
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            # After each decoder block i, output channels = reversed_ch[i+1]
            for i in range(len(reversed_ch) - 2):  # All but the last decoder output
                decoder_out_ch = reversed_ch[i + 1]
                self.ds_heads.append(nn.Conv3d(decoder_out_ch, out_channels, 1))
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Encoder path with skip connections
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
        
        # Decoder path
        ds_outputs = []
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = skips[-(i + 1)]
            
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            
            # Deep supervision outputs
            if self.deep_supervision and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))
        
        main_out = self.out_conv(x)
        
        if self.deep_supervision and self.training:
            return [main_out] + ds_outputs
        return main_out


class ResidualEncoderUNet(nn.Module):
    """Residual Encoder U-Net (better for deeper networks)."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 channels: Tuple[int, ...] = (32, 64, 128, 256, 320, 320),
                 strides: Tuple[int, ...] = (1, 2, 2, 2, 2, 2),
                 kernel_sizes: Tuple[int, ...] = (3, 3, 3, 3, 3, 3),
                 deep_supervision: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.n_stages = len(channels)
        
        # Encoder with residual blocks
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for i, (ch, stride, ks) in enumerate(zip(channels, strides, kernel_sizes)):
            self.encoders.append(ResidualBlock(prev_ch, ch, ks, stride, dropout))
            prev_ch = ch
        
        # Decoder (same as PlainConvUNet)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        reversed_ch = list(reversed(channels))
        reversed_strides = list(reversed(strides))
        
        for i in range(len(reversed_ch) - 1):
            in_ch = reversed_ch[i]
            skip_ch = reversed_ch[i + 1]
            out_ch = skip_ch
            
            if reversed_strides[i] > 1:
                self.upconvs.append(
                    nn.ConvTranspose3d(in_ch, in_ch, reversed_strides[i], 
                                      stride=reversed_strides[i])
                )
            else:
                self.upconvs.append(nn.Identity())
            
            self.decoders.append(ConvBlock(in_ch + skip_ch, out_ch))
        
        # Output heads
        self.out_conv = nn.Conv3d(channels[0], out_channels, 1)
        
        # Deep supervision heads - use decoder output channels (reversed order)
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            # After each decoder block i, output channels = reversed_ch[i+1]
            for i in range(len(reversed_ch) - 2):  # All but the last decoder output
                decoder_out_ch = reversed_ch[i + 1]
                self.ds_heads.append(nn.Conv3d(decoder_out_ch, out_channels, 1))
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Encoder path
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
        
        # Decoder path
        ds_outputs = []
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = skips[-(i + 1)]
            
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            
            if self.deep_supervision and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))
        
        main_out = self.out_conv(x)
        
        if self.deep_supervision and self.training:
            return [main_out] + ds_outputs
        return main_out


def get_model(model_key: str, config: dict) -> nn.Module:
    """Factory function to create model based on configuration."""
    model_cfg = config["models"][model_key]
    unet_class = model_cfg["unet_class"]
    
    common_params = {
        "in_channels": 1,
        "out_channels": config["data"]["num_classes"],
        "channels": model_cfg["channels"],
        "strides": model_cfg["strides"],
        "kernel_sizes": model_cfg["kernel_sizes"],
        "deep_supervision": config["training"]["deep_supervision"],
    }
    
    if unet_class == "PlainConvUNet":
        return PlainConvUNet(**common_params)
    elif unet_class == "ResidualEncoderUNet":
        return ResidualEncoderUNet(**common_params)
    else:
        raise ValueError(f"Unknown UNet class: {unet_class}")


# ============================================================================
# DATA LOADING
# ============================================================================
def get_data_dicts(data_dir: str) -> List[Dict]:
    """Get data dictionaries for KiTS23 dataset."""
    data_path = Path(data_dir)
    data_dicts = []
    
    for case_dir in sorted(data_path.iterdir()):
        if case_dir.is_dir() and case_dir.name.startswith('case_'):
            img = case_dir / "imaging.nii.gz"
            seg = case_dir / "segmentation.nii.gz"
            if img.exists() and seg.exists():
                data_dicts.append({
                    "image": str(img), 
                    "label": str(seg), 
                    "case_id": case_dir.name
                })
    
    return data_dicts


def get_transforms(model_key: str, config: dict, training: bool = True) -> Compose:
    """Get transforms for a specific model configuration."""
    model_cfg = config["models"][model_key]
    data_cfg = config["data"]
    
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=model_cfg["spacing"], 
                mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], 
                            a_min=data_cfg["intensity_min"],
                            a_max=data_cfg["intensity_max"],
                            b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=model_cfg["patch_size"]),
    ]
    
    if training:
        # nnU-Net-style augmentation
        augmentations = [
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label",
                spatial_size=model_cfg["patch_size"],
                pos=4, neg=1,  # 4:1 ratio foreground:background (more tumor/cyst samples)
                num_samples=model_cfg["num_samples"],
            ),
            # Spatial augmentations
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # Intensity augmentations
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.05, prob=0.15),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.0), 
                               sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0), prob=0.15),
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.15),
        ]
        base_transforms.extend(augmentations)
    
    # Ensure data is converted to tensors (critical for collation)
    base_transforms.append(EnsureTyped(keys=["image", "label"], data_type="tensor", track_meta=False))
    base_transforms.append(ToTensord(keys=["image", "label"]))
    
    return Compose(base_transforms)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_model(model_key: str, config: dict, num_epochs: int, resume: bool = False):
    """Train a single model configuration."""
    model_cfg = config["models"][model_key]
    train_cfg = config["training"]
    
    print("\n" + "=" * 70)
    print(f"  TRAINING: {model_cfg['name']}")
    print("=" * 70)
    print(f"  Spacing: {model_cfg['spacing']}")
    print(f"  Patch Size: {model_cfg['patch_size']}")
    print(f"  Batch Size: {model_cfg['batch_size']}")
    print(f"  UNet Class: {model_cfg['unet_class']}")
    print(f"  Epochs: {num_epochs}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Directories
    output_dir = Path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints" / model_key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path(config["cache_dir"]) / model_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    data_dicts = get_data_dicts(config["kits23_dir"])
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    train_files, val_files = data_dicts[:split_idx], data_dicts[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Datasets
    train_ds = PersistentDataset(
        train_files, 
        get_transforms(model_key, config, training=True),
        cache_dir=str(cache_dir / "train")
    )
    val_ds = PersistentDataset(
        val_files,
        get_transforms(model_key, config, training=False),
        cache_dir=str(cache_dir / "val")
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=model_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=train_cfg["num_workers"] // 2
    )
    
    # Model
    model = get_model(model_key, config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Loss
    ds_weights = train_cfg["deep_supervision_weights"]
    loss_fn = DiceCELossWithDS(weights=ds_weights, include_background=False)
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        momentum=0.99,
        weight_decay=train_cfg["weight_decay"],
        nesterov=True
    )
    
    # Scheduler
    total_iters = num_epochs * len(train_loader)
    scheduler = PolynomialLR(optimizer, total_iters, power=0.9)
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    scaler = torch.amp.GradScaler('cuda')
    
    # Training state
    best_metric = -1
    patience_counter = 0
    start_epoch = 0
    history = {"train_loss": [], "val_dice": [], "lr": []}
    
    # Resume from checkpoint
    last_ckpt = checkpoint_dir / "last.pth"
    if resume and last_ckpt.exists():
        print(f"\n📂 Resuming from: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -1)
        history = ckpt.get("history", history)
        print(f"   Resuming from epoch {start_epoch}, best: {best_metric:.4f}")
    
    # Training loop
    training_start = time.time()
    
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            return f"{int(hours)}h {int(mins)}m"
    
    print(f"\n🚀 Starting training at {datetime.now().strftime('%H:%M:%S')}")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(avg_loss)
        history["lr"].append(current_lr)
        
        # Timing
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - training_start
        eta = epoch_time * (num_epochs - epoch - 1)
        
        print(f"\n📊 Epoch {epoch+1}: Loss={avg_loss:.4f}, LR={current_lr:.2e}")
        print(f"⏱️  Time: {format_time(epoch_time)} | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}")
        
        # Validation
        if (epoch + 1) % train_cfg["val_interval"] == 0 or epoch == num_epochs - 1:
            model.eval()
            val_dice_scores = []
            
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validation"):
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    
                    with torch.amp.autocast('cuda'):
                        val_outputs = sliding_window_inference(
                            val_inputs, model_cfg["patch_size"],
                            config["inference"]["sw_batch_size"], model,
                            overlap=config["inference"]["overlap"]
                        )
                    
                    val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                    dice_metric(val_outputs, val_labels)
                
                # Get per-class Dice
                metric_values = dice_metric.aggregate()
                dice_metric.reset()
                
                if len(metric_values.shape) > 0:
                    kidney_dice = metric_values[0].item()
                    tumor_dice = metric_values[1].item() if len(metric_values) > 1 else 0
                    cyst_dice = metric_values[2].item() if len(metric_values) > 2 else 0
                else:
                    kidney_dice = metric_values.item()
                    tumor_dice = cyst_dice = 0
                
                mean_dice = (kidney_dice + tumor_dice + cyst_dice) / 3
                history["val_dice"].append(mean_dice)
                
                print(f"\n🎯 Validation Dice:")
                print(f"   Kidney: {kidney_dice:.4f}")
                print(f"   Tumor:  {tumor_dice:.4f}")
                print(f"   Cyst:   {cyst_dice:.4f}")
                print(f"   Mean:   {mean_dice:.4f}")
                
                # Save best model
                if mean_dice > best_metric:
                    best_metric = mean_dice
                    patience_counter = 0
                    torch.save({
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "dice": mean_dice,
                        "kidney_dice": kidney_dice,
                        "tumor_dice": tumor_dice,
                        "cyst_dice": cyst_dice,
                    }, checkpoint_dir / "best.pth")
                    print(f"✅ New best model saved!")
                else:
                    patience_counter += 1
                    print(f"⏳ No improvement ({patience_counter}/{train_cfg['early_stopping_patience']})")
                
                print(f"🏆 Best: {best_metric:.4f}")
                
                # Early stopping
                if patience_counter >= train_cfg["early_stopping_patience"]:
                    print(f"\n🛑 Early stopping triggered!")
                    break
        
        # Save checkpoint
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "history": history,
        }, last_ckpt)
    
    # Training complete
    total_time = time.time() - training_start
    print(f"\n{'=' * 70}")
    print(f"  {model_cfg['name']} TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best Dice: {best_metric:.4f}")
    print(f"{'=' * 70}")
    
    # Save training history
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return model


def train_all_models(config: dict, num_epochs: int, resume: bool = False):
    """Train all three model configurations."""
    print("\n" + "=" * 70)
    print("  TRAINING ALL MODELS (2nd Place Approach)")
    print("=" * 70)
    
    for model_key in config["models"].keys():
        train_model(model_key, config, num_epochs, resume)
    
    print("\n✅ All models trained!")


# ============================================================================
# POST-PROCESSING (Critical for high performance)
# ============================================================================
def kidney_postprocess(lowres_pred: np.ndarray, fullres_pred: np.ndarray,
                       config: dict) -> np.ndarray:
    """
    Kidney post-processing (from 2nd place solution):
    1. Take intersection of lowres and fullres kidney masks
    2. Keep only the 2 largest connected components (left/right kidney)
    3. Remove small isolated regions
    """
    pp_cfg = config["postprocess"]
    
    # Get kidney masks (kidney + tumor + cyst = any foreground)
    kidney_lr = (lowres_pred >= 1)
    kidney_fr = (fullres_pred >= 1)
    
    # Intersection: agree on kidney region
    kidney_merged = kidney_lr & kidney_fr
    
    # Connected components
    labeled, n_components = scipy_label(kidney_merged)
    
    if n_components == 0:
        return fullres_pred  # No kidney found, return original
    
    # Get component sizes
    component_sizes = []
    for i in range(1, n_components + 1):
        size = (labeled == i).sum()
        component_sizes.append((i, size))
    
    # Sort by size, keep top 2 (left and right kidney)
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    top_components = component_sizes[:2]
    
    # Create refined kidney mask
    kidney_refined = np.zeros_like(kidney_merged)
    for comp_idx, size in top_components:
        if size >= pp_cfg["kidney_min_size"]:
            kidney_refined |= (labeled == comp_idx)
    
    # Apply refined mask to predictions
    # Keep predictions only within refined kidney region
    result = fullres_pred.copy()
    result[~kidney_refined] = 0
    
    return result


def tumor_postprocess(pred: np.ndarray, config: dict) -> np.ndarray:
    """
    Tumor post-processing:
    1. Remove tumors not connected to kidney region
    2. Remove small tumor components
    """
    pp_cfg = config["postprocess"]
    
    # Get masks
    kidney_mask = (pred == 1)
    tumor_mask = (pred == 2)
    cyst_mask = (pred == 3)
    
    # Dilate kidney slightly to catch tumors on boundary
    struct = generate_binary_structure(3, 1)
    kidney_dilated = binary_dilation(kidney_mask, struct, 
                                     iterations=pp_cfg["dilation_iterations"])
    
    # Process tumors
    tumor_labeled, n_tumors = scipy_label(tumor_mask)
    tumor_refined = np.zeros_like(tumor_mask)
    
    for i in range(1, n_tumors + 1):
        component = (tumor_labeled == i)
        size = component.sum()
        
        # Keep if connected to kidney and large enough
        if size >= pp_cfg["tumor_min_size"]:
            if np.any(component & kidney_dilated):
                tumor_refined |= component
    
    # Process cysts similarly
    cyst_labeled, n_cysts = scipy_label(cyst_mask)
    cyst_refined = np.zeros_like(cyst_mask)
    
    for i in range(1, n_cysts + 1):
        component = (cyst_labeled == i)
        size = component.sum()
        
        if size >= pp_cfg["cyst_min_size"]:
            if np.any(component & kidney_dilated):
                cyst_refined |= component
    
    # Reconstruct prediction
    result = np.zeros_like(pred)
    result[kidney_mask] = 1
    result[tumor_refined] = 2
    result[cyst_refined] = 3
    
    return result


def majority_voting(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Combine predictions from multiple models via majority voting.
    
    Args:
        predictions: List of (D, H, W) arrays with label values 0-3
    
    Returns:
        Final prediction array
    """
    if len(predictions) == 1:
        return predictions[0]
    
    stacked = np.stack(predictions, axis=0)  # (N, D, H, W)
    
    # For each voxel, take the most common label
    result, _ = mode(stacked, axis=0, keepdims=False)
    
    return result.squeeze().astype(np.int32)


# ============================================================================
# INFERENCE
# ============================================================================
def run_inference(config: dict, models_to_use: List[str] = None):
    """Run inference with all trained models and apply post-processing."""
    if models_to_use is None:
        models_to_use = list(config["models"].keys())
    
    print("\n" + "=" * 70)
    print("  INFERENCE PIPELINE")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output_dir"])
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Load validation cases
    data_dicts = get_data_dicts(config["kits23_dir"])
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    val_dicts = data_dicts[split_idx:]
    
    print(f"Running inference on {len(val_dicts)} validation cases")
    
    # Load all models
    models = {}
    for model_key in models_to_use:
        ckpt_path = output_dir / "checkpoints" / model_key / "best.pth"
        if not ckpt_path.exists():
            print(f"⚠️  Checkpoint not found for {model_key}, skipping")
            continue
        
        model = get_model(model_key, config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[model_key] = model
        print(f"✓ Loaded {model_key}")
    
    if not models:
        print("❌ No models found!")
        return
    
    # Process each case
    for data_dict in tqdm(val_dicts, desc="Inference"):
        case_id = data_dict["case_id"]
        
        # Load and preprocess image for each model
        predictions = {}
        
        for model_key, model in models.items():
            model_cfg = config["models"][model_key]
            
            # Load image
            img_nii = nib.load(data_dict["image"])
            img_data = img_nii.get_fdata().astype(np.float32)
            original_shape = img_data.shape
            original_affine = img_nii.affine
            
            # Preprocess: intensity scaling
            data_cfg = config["data"]
            img_data = np.clip(img_data, data_cfg["intensity_min"], data_cfg["intensity_max"])
            img_data = (img_data - data_cfg["intensity_min"]) / \
                      (data_cfg["intensity_max"] - data_cfg["intensity_min"])
            
            # Add batch and channel dimensions
            img_tensor = torch.from_numpy(img_data[None, None]).float().to(device)
            
            # Run inference
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    output = sliding_window_inference(
                        img_tensor, model_cfg["patch_size"],
                        config["inference"]["sw_batch_size"], model,
                        overlap=config["inference"]["overlap"]
                    )
            
            # Get prediction
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            predictions[model_key] = pred.astype(np.int32)
        
        # Apply post-processing
        # Step 1: Kidney post-processing using lowres + fullres
        if "lowres_plain" in predictions and "fullres_batch4" in predictions:
            predictions["fullres_batch4"] = kidney_postprocess(
                predictions["lowres_plain"],
                predictions["fullres_batch4"],
                config
            )
        
        if "lowres_residual" in predictions and "fullres_batch4" in predictions:
            # Use lowres_residual with fullres as well
            predictions["lowres_residual"] = kidney_postprocess(
                predictions["lowres_residual"],
                predictions["fullres_batch4"],
                config
            )
        
        # Step 2: Tumor post-processing on each prediction
        for model_key in predictions:
            predictions[model_key] = tumor_postprocess(predictions[model_key], config)
        
        # Step 3: Majority voting
        pred_list = list(predictions.values())
        final_pred = majority_voting(pred_list)
        
        # Save prediction
        pred_nii = nib.Nifti1Image(final_pred.astype(np.int16), original_affine)
        nib.save(pred_nii, predictions_dir / f"{case_id}.nii.gz")
    
    print(f"\n✅ Predictions saved to: {predictions_dir}")


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate(config: dict):
    """Evaluate predictions against ground truth."""
    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)
    
    output_dir = Path(config["output_dir"])
    predictions_dir = output_dir / "predictions"
    
    # Load validation cases
    data_dicts = get_data_dicts(config["kits23_dir"])
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    val_dicts = data_dicts[split_idx:]
    
    # Compute metrics
    results = {"kidney": [], "tumor": [], "cyst": [], "masses": []}
    
    for data_dict in tqdm(val_dicts, desc="Evaluating"):
        case_id = data_dict["case_id"]
        pred_path = predictions_dir / f"{case_id}.nii.gz"
        
        if not pred_path.exists():
            continue
        
        # Load prediction and ground truth
        pred = nib.load(pred_path).get_fdata().astype(np.int32)
        gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
        
        # Compute Dice for each class
        for label, name in [(1, "kidney"), (2, "tumor"), (3, "cyst")]:
            pred_mask = (pred == label).astype(float)
            gt_mask = (gt == label).astype(float)
            
            intersection = (pred_mask * gt_mask).sum()
            union = pred_mask.sum() + gt_mask.sum()
            
            if union > 0:
                dice = 2 * intersection / union
            else:
                dice = 1.0 if pred_mask.sum() == 0 else 0.0
            
            results[name].append(dice)
        
        # Masses (tumor + cyst)
        pred_masses = ((pred == 2) | (pred == 3)).astype(float)
        gt_masses = ((gt == 2) | (gt == 3)).astype(float)
        
        intersection = (pred_masses * gt_masses).sum()
        union = pred_masses.sum() + gt_masses.sum()
        
        if union > 0:
            masses_dice = 2 * intersection / union
        else:
            masses_dice = 1.0 if pred_masses.sum() == 0 else 0.0
        
        results["masses"].append(masses_dice)
    
    # Print results
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    
    for name in ["kidney", "tumor", "cyst", "masses"]:
        if results[name]:
            mean_dice = np.mean(results[name])
            std_dice = np.std(results[name])
            print(f"  {name.capitalize():8}: {mean_dice:.4f} ± {std_dice:.4f}")
    
    # Overall
    mean_overall = np.mean([np.mean(results[n]) for n in ["kidney", "tumor", "cyst"] if results[n]])
    print(f"  {'Overall':8}: {mean_overall:.4f}")
    
    # Check against targets
    print("\n" + "-" * 70)
    print("  TARGET CHECK")
    print("-" * 70)
    
    kidney_dice = np.mean(results["kidney"]) if results["kidney"] else 0
    tumor_dice = np.mean(results["tumor"]) if results["tumor"] else 0
    cyst_dice = np.mean(results["cyst"]) if results["cyst"] else 0
    
    targets = [
        ("Kidney Dice >= 0.70", kidney_dice >= 0.70, kidney_dice),
        ("Tumor Dice >= 0.50", tumor_dice >= 0.50, tumor_dice),
        ("Cyst Dice >= 0.50", cyst_dice >= 0.50, cyst_dice),
    ]
    
    for name, achieved, value in targets:
        status = "✅ PASS" if achieved else "❌ FAIL"
        print(f"  {status} {name} (got {value:.4f})")
    
    print("=" * 70)
    
    # Save detailed results
    detailed_results = {
        "metrics": {
            "kidney_dice_mean": float(np.mean(results["kidney"])) if results["kidney"] else None,
            "kidney_dice_std": float(np.std(results["kidney"])) if results["kidney"] else None,
            "tumor_dice_mean": float(np.mean(results["tumor"])) if results["tumor"] else None,
            "tumor_dice_std": float(np.std(results["tumor"])) if results["tumor"] else None,
            "cyst_dice_mean": float(np.mean(results["cyst"])) if results["cyst"] else None,
            "cyst_dice_std": float(np.std(results["cyst"])) if results["cyst"] else None,
            "masses_dice_mean": float(np.mean(results["masses"])) if results["masses"] else None,
            "overall_dice_mean": float(mean_overall),
        },
        "per_case": {
            "kidney": results["kidney"],
            "tumor": results["tumor"],
            "cyst": results["cyst"],
            "masses": results["masses"],
        }
    }
    
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir / 'evaluation_results.json'}")
    
    return detailed_results


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="KiTS23 Final Model")
    parser.add_argument("--mode", choices=["train", "train_single", "inference", "evaluate", "full_pipeline"],
                       default="train", help="Execution mode")
    parser.add_argument("--model", choices=["lowres_plain", "lowres_residual", "fullres_batch4"],
                       help="Specific model to train (for train_single mode)")
    parser.add_argument("--quick", action="store_true", help="Quick test (50 epochs)")
    parser.add_argument("--full", action="store_true", help="Full training (500 epochs)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    args = parser.parse_args()
    
    config = CONFIG.copy()
    set_determinism(config["seed"])
    
    # Determine epochs
    if args.epochs:
        num_epochs = args.epochs
    elif args.quick:
        num_epochs = config["training"]["epochs_quick"]
    elif args.full:
        num_epochs = config["training"]["epochs_full"]
    else:
        num_epochs = config["training"]["epochs_standard"]
    
    # Create output dir
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  KiTS23 FINAL MODEL")
    print("  Based on MICCAI 2023 2nd Place Solution")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {num_epochs}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
    print("=" * 70)
    
    # Execute mode
    if args.mode == "train":
        train_all_models(config, num_epochs, resume=args.resume)
    
    elif args.mode == "train_single":
        if not args.model:
            print("Error: --model required for train_single mode")
            return
        train_model(args.model, config, num_epochs, resume=args.resume)
    
    elif args.mode == "inference":
        run_inference(config)
    
    elif args.mode == "evaluate":
        evaluate(config)
    
    elif args.mode == "full_pipeline":
        # Train all models
        train_all_models(config, num_epochs, resume=args.resume)
        
        # Run inference
        run_inference(config)
        
        # Evaluate
        evaluate(config)
    
    print("\n" + "=" * 70)
    print("  ✅ COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
