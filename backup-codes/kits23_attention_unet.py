"""
KiTS23 Spatial and Channel Attention U-Net
============================================
Implementation based on: srg9000/kits21_spatial_channel_attention
Paper: "Kidney and Kidney Tumor Segmentation using Spatial and Channel attention enhanced U-Net"
       https://link.springer.com/chapter/10.1007/978-3-030-98385-7_20

This is a complete, self-contained implementation adapted for the KiTS23 dataset.

Key Features:
1. 3D U-Net with Spatial and Channel Attention mechanisms (CBAM-style)
2. Channel Attention: Squeeze-and-Excitation with avg+max pooling
3. Spatial Attention: Conv-based attention maps for spatial focus
4. MONAI-based data loading and augmentation pipeline
5. DiceCE Loss for multi-class segmentation
6. Sliding window inference for full-volume prediction

Labels:
- 0: Background
- 1: Kidney
- 2: Tumor  
- 3: Cyst

Usage:
    python kits23_attention_unet.py --mode train           # Train the model
    python kits23_attention_unet.py --mode train --quick   # Quick test (5 epochs)
    python kits23_attention_unet.py --mode inference       # Run inference
    python kits23_attention_unet.py --mode evaluate        # Evaluate predictions
    python kits23_attention_unet.py --mode all             # Full pipeline
"""

import os
import sys
import json
import shutil
import argparse
import warnings
from pathlib import Path
from typing import Sequence, Tuple, Union, List, Dict, Optional
from datetime import datetime, timedelta
import time

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# MONAI imports
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    Rand3DElasticd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    AsDiscreted,
    Invertd,
    SaveImaged,
)
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Dataset paths (relative to script location)
    "kits23_dir": "./kits23/dataset",
    
    # Output paths
    "output_dir": "./output/kits23_attention_unet",
    "checkpoint_dir": "./output/kits23_attention_unet/checkpoints",
    "predictions_dir": "./output/kits23_attention_unet/predictions",
    "visualizations_dir": "./output/kits23_attention_unet/visualizations",
    
    # Model architecture
    "model": {
        "dimensions": 3,
        "in_channels": 1,
        "out_channels": 4,  # background, kidney, tumor, cyst
        "channels": (64, 128, 256, 512, 512),  # Encoder channel progression
        "strides": (2, 2, 2, 2),  # Downsampling strides
        "num_res_units": 0,  # 0 = plain conv, >0 = residual units
        "attention_ratio": 16,  # Channel attention reduction ratio
        "spatial_kernel": 7,  # Spatial attention kernel size
    },
    
    # Training settings
    "training": {
        "num_epochs": 100,
        "quick_epochs": 1,
        "batch_size": 2,  # Per GPU
        "num_samples": 4,  # Patches per volume
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "val_interval": 1,
        "cache_rate": 0.1,  # SmartCache rate
        "num_workers": 8,
    },
    
    # Data settings
    "data": {
        "patch_size": (160, 160, 64),  # Training patch size
        "spacing": (2.0, 1.62, 1.62),  # Target voxel spacing (mm)
        "intensity_min": -80,  # CT window min
        "intensity_max": 305,  # CT window max
        "train_val_split": 0.85,  # 85% train, 15% val
    },
    
    # Inference settings
    "inference": {
        "roi_size": (160, 160, 64),
        "sw_batch_size": 4,
        "overlap": 0.5,
    },
    
    # Label definitions
    "labels": {
        "background": 0,
        "kidney": 1,
        "tumor": 2,
        "cyst": 3,
    },
    
    # Random seed
    "seed": 42,
}


# ============================================================================
# ATTENTION MODULES
# ============================================================================
class ChannelAttention(nn.Module):
    """
    Channel Attention Module (SE-style with avg + max pooling).
    
    Applies squeeze-and-excitation attention along channel dimension.
    Combines both average and max pooling for richer attention.
    
    Reference: CBAM (https://arxiv.org/abs/1807.06521)
    """
    
    def __init__(self, submodule: nn.Module, in_planes: int, out_planes: int, 
                 ratio: int = 16):
        super().__init__()
        self.submodule = submodule
        self.in_planes = in_planes
        
        # Adaptive pooling for global context
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP for channel transformation
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, max(in_planes // ratio, 1), 1, bias=False),
            nn.GELU(),
            nn.Conv3d(max(in_planes // ratio, 1), out_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply submodule first
        y = self.submodule(x)
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        
        return y * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Applies attention across spatial dimensions using channel-wise 
    statistics (mean and max).
    
    Reference: CBAM (https://arxiv.org/abs/1807.06521)
    """
    
    def __init__(self, submodule: nn.Module, in_channels: int, 
                 kernel_size: int = 7, out_channels: Optional[int] = None,
                 add_conv_1x1: bool = False):
        super().__init__()
        self.submodule = submodule
        self.add_conv_1x1 = add_conv_1x1
        
        # Flatten channels to 2 (avg + max)
        self.conv_flat = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        
        # Scale adjustment layers
        self.pool_layer = nn.MaxPool3d(2)
        self.upscale_layer = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.sigmoid = nn.Sigmoid()
        
        # Optional 1x1 conv for channel adjustment
        if add_conv_1x1:
            if out_channels is None:
                raise ValueError("out_channels required when add_conv_1x1=True")
            self.conv_1x1 = nn.Conv3d(in_channels, out_channels, 1, bias=False)
            self.act = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply submodule
        y = self.submodule(x)
        
        # Spatial attention via channel statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.conv1(x_cat)
        
        # Handle size mismatch
        if attention.shape[-1] > y.shape[-1]:
            attention = self.pool_layer(attention)
        elif attention.shape[-1] < y.shape[-1]:
            attention = self.upscale_layer(attention)
        
        out = y * self.sigmoid(attention)
        
        # Optional channel reduction
        if self.add_conv_1x1:
            out = self.act(self.conv_1x1(out))
        
        return out


# ============================================================================
# ATTENTION U-NET MODEL
# ============================================================================
class AttentionUNet(nn.Module):
    """
    3D U-Net with Spatial and Channel Attention.
    
    Architecture:
    - Encoder: Conv blocks with progressive downsampling
    - Decoder: ConvTranspose blocks with skip connections
    - Attention: Channel attention at bottleneck, spatial attention in decoder
    
    Based on: srg9000/kits21_spatial_channel_attention
    """
    
    def __init__(
        self,
        dimensions: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: Sequence[int] = (64, 128, 256, 512, 512),
        strides: Sequence[int] = (2, 2, 2, 2),
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        attention_ratio: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        
        # Validate inputs
        if len(channels) < 2:
            raise ValueError("channels must have at least 2 elements")
        if len(strides) != len(channels) - 1:
            warnings.warn(f"strides length ({len(strides)}) != channels - 1 ({len(channels) - 1})")
        
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.attention_ratio = attention_ratio
        self.spatial_kernel = spatial_kernel
        
        # Build the model recursively
        self.model = self._create_block(
            in_channels, out_channels, channels, strides, is_top=True
        )
    
    def _create_block(
        self,
        inc: int,
        outc: int,
        channels: Sequence[int],
        strides: Sequence[int],
        is_top: bool
    ) -> nn.Sequential:
        """Recursively build U-Net blocks with attention."""
        
        c = channels[0]
        s = strides[0] if strides else 1
        
        if len(channels) > 2:
            # Continue recursion
            subblock = self._create_block(c, c, channels[1:], strides[1:], False)
            upc = c * 2
            
            # Add spatial attention at upper levels
            add_spatial = len(channels) > len(self.channels) - 1
            add_channel = False
        else:
            # Bottom layer
            subblock = self._get_bottom_layer(c, channels[1])
            # Apply channel attention at bottleneck
            subblock = ChannelAttention(
                subblock, in_planes=c, out_planes=channels[1],
                ratio=self.attention_ratio
            )
            upc = c + channels[1]
            add_spatial = False
            add_channel = False
        
        # Create downsampling layer
        down = self._get_down_layer(inc, c, s, is_top)
        if add_spatial:
            down = SpatialAttention(down, in_channels=inc, kernel_size=self.spatial_kernel)
        
        # Create upsampling layer
        if len(channels) == len(self.channels) and add_spatial:
            up = self._get_up_layer(upc, upc, s, is_top)
        else:
            up = self._get_up_layer(upc, outc, s, is_top)
        
        if add_spatial:
            add_conv_1x1 = len(channels) == len(self.channels)
            up = SpatialAttention(
                up, in_channels=upc, out_channels=outc,
                add_conv_1x1=add_conv_1x1, kernel_size=self.spatial_kernel
            )
        
        return nn.Sequential(down, SkipConnection(subblock), up)
    
    def _get_down_layer(self, in_channels: int, out_channels: int, 
                        strides: int, is_top: bool) -> nn.Module:
        """Create encoder (downsampling) layer."""
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions, in_channels, out_channels,
                strides=strides, kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout
            )
        return Convolution(
            self.dimensions, in_channels, out_channels,
            strides=strides, kernel_size=self.kernel_size,
            act=self.act, norm=self.norm, dropout=self.dropout
        )
    
    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create bottleneck layer."""
        return self._get_down_layer(in_channels, out_channels, 1, False)
    
    def _get_up_layer(self, in_channels: int, out_channels: int, 
                      strides: int, is_top: bool) -> nn.Module:
        """Create decoder (upsampling) layer."""
        conv = Convolution(
            self.dimensions, in_channels, out_channels,
            strides=strides, kernel_size=self.up_kernel_size,
            act=self.act, norm=self.norm, dropout=self.dropout,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True
        )
        
        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions, out_channels, out_channels,
                strides=1, kernel_size=self.kernel_size, subunits=1,
                act=self.act, norm=self.norm, dropout=self.dropout,
                last_conv_only=is_top
            )
            conv = nn.Sequential(conv, ru)
        
        return conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================================
# DATA LOADING
# ============================================================================
def get_kits23_data_dicts(data_dir: str, require_segmentation: bool = True) -> List[Dict]:
    """
    Get data dictionaries for all KiTS23 cases.
    
    Args:
        data_dir: Path to KiTS23 dataset directory
        require_segmentation: Only include cases with segmentation masks
    
    Returns:
        List of {"image": path, "label": path} dictionaries
    """
    data_path = Path(data_dir)
    data_dicts = []
    
    cases = sorted([d for d in data_path.iterdir() 
                   if d.is_dir() and d.name.startswith('case_')])
    
    for case_dir in cases:
        img_path = case_dir / "imaging.nii.gz"
        seg_path = case_dir / "segmentation.nii.gz"
        
        if img_path.exists():
            if require_segmentation:
                if seg_path.exists():
                    data_dicts.append({
                        "image": str(img_path),
                        "label": str(seg_path),
                        "case_id": case_dir.name
                    })
            else:
                data_dicts.append({
                    "image": str(img_path),
                    "case_id": case_dir.name
                })
    
    return data_dicts


def get_train_transforms(config: dict) -> Compose:
    """Get training data transforms with augmentation."""
    data_cfg = config["data"]
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=data_cfg["spacing"],
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=data_cfg["intensity_min"],
            a_max=data_cfg["intensity_max"],
            b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Pad to ensure minimum size before random crop
        SpatialPadd(keys=["image", "label"], spatial_size=data_cfg["patch_size"], mode="constant"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=data_cfg["patch_size"],
            pos=1, neg=1,
            num_samples=config["training"]["num_samples"],
            image_key="image",
            image_threshold=0,
            allow_smaller=True  # Allow smaller patches if image is still smaller after padding
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.5,
            sigma_range=(5, 8),
            magnitude_range=(50, 150),
            spatial_size=data_cfg["patch_size"],
            translate_range=(10, 10, 5),
            rotate_range=(np.pi / 36, np.pi / 36, np.pi),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="zeros"
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.25),
        RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.1),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms(config: dict) -> Compose:
    """Get validation data transforms (no augmentation)."""
    data_cfg = config["data"]
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=data_cfg["spacing"],
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=data_cfg["intensity_min"],
            a_max=data_cfg["intensity_max"],
            b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Pad to ensure minimum size for sliding window inference
        SpatialPadd(keys=["image", "label"], spatial_size=data_cfg["patch_size"], mode="constant"),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_inference_transforms(config: dict) -> Compose:
    """Get inference transforms (image only)."""
    data_cfg = config["data"]
    
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(
            keys=["image"],
            pixdim=data_cfg["spacing"],
            mode="bilinear"
        ),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=data_cfg["intensity_min"],
            a_max=data_cfg["intensity_max"],
            b_min=0.0, b_max=1.0, clip=True
        ),
        # Pad to ensure minimum size for sliding window inference
        SpatialPadd(keys=["image"], spatial_size=data_cfg["patch_size"], mode="constant"),
        EnsureTyped(keys=["image"]),
    ])


# ============================================================================
# TRAINING
# ============================================================================
def create_model(config: dict, device: torch.device) -> AttentionUNet:
    """Create and initialize the Attention U-Net model."""
    model_cfg = config["model"]
    
    model = AttentionUNet(
        dimensions=model_cfg["dimensions"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=model_cfg["channels"],
        strides=model_cfg["strides"],
        num_res_units=model_cfg["num_res_units"],
        attention_ratio=model_cfg["attention_ratio"],
        spatial_kernel=model_cfg["spatial_kernel"],
    )
    
    return model.to(device)


def train_model(config: dict, num_epochs: int = None, resume_from: str = None):
    """
    Train the Attention U-Net model.
    
    Args:
        config: Configuration dictionary
        num_epochs: Override number of epochs
        resume_from: Path to checkpoint to resume from
    """
    print("\n" + "=" * 70)
    print("  TRAINING: Spatial-Channel Attention U-Net for KiTS23")
    print("=" * 70)
    
    # Setup
    set_determinism(seed=config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directories
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data
    print("\n📊 Loading dataset...")
    data_dicts = get_kits23_data_dicts(config["kits23_dir"])
    print(f"Found {len(data_dicts)} cases with segmentation")
    
    # Split data
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    train_files = data_dicts[:split_idx]
    val_files = data_dicts[split_idx:]
    print(f"Train: {len(train_files)} cases")
    print(f"Val: {len(val_files)} cases")
    
    # Create datasets
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    
    train_cfg = config["training"]
    
    try:
        from monai.data import SmartCacheDataset
        train_ds = SmartCacheDataset(
            data=train_files, 
            transform=train_transforms,
            cache_rate=train_cfg["cache_rate"],
            replace_rate=0.5,
            num_init_workers=train_cfg["num_workers"]
        )
    except ImportError:
        print("SmartCacheDataset not available, using Dataset")
        train_ds = Dataset(data=train_files, transform=train_transforms)
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg["batch_size"],
        shuffle=True, 
        num_workers=train_cfg["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        num_workers=train_cfg["num_workers"]
    )
    
    # Create model
    print("\n🔧 Creating model...")
    model = create_model(config, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = -1
    if resume_from and Path(resume_from).exists():
        print(f"\n📥 Loading checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_metric" in checkpoint:
            best_metric = checkpoint["best_metric"]
        print(f"Resuming from epoch {start_epoch}, best metric: {best_metric:.4f}")
    
    # Loss and optimizer
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    # Metrics - use mean_batch to get per-class Dice scores
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    
    # Training settings
    epochs = num_epochs if num_epochs else train_cfg["num_epochs"]
    val_interval = train_cfg["val_interval"]
    
    # Post-processing transforms
    post_pred = Compose([
        EnsureTyped(keys="pred"),
        AsDiscreted(keys="pred", argmax=True, to_onehot=4)
    ])
    post_label = Compose([
        EnsureTyped(keys="label"),
        AsDiscreted(keys="label", to_onehot=4)
    ])
    
    # Training history
    epoch_loss_values = []
    metric_values = []  # Mean Dice per epoch
    metric_values_per_class = {"kidney": [], "tumor": [], "cyst": []}  # Per-class Dice
    best_metric_epoch = -1
    best_metrics_per_class = {"kidney": 0.0, "tumor": 0.0, "cyst": 0.0}
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Validation every {val_interval} epoch(s)")
    
    # Time tracking
    training_start_time = time.time()
    epoch_times = []  # Store duration of each epoch
    
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        current_time = datetime.now().strftime('%H:%M:%S')
        
        print("-" * 70)
        print(f"Epoch {epoch + 1}/{epochs} | 🕐 Started: {current_time}")
        
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in tqdm(train_loader, desc="Training"):
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        # Calculate epoch time and ETA
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        total_elapsed = epoch_end_time - training_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_finish = datetime.now() + timedelta(seconds=eta_seconds)
        
        # Format times
        def format_duration(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                mins, secs = divmod(seconds, 60)
                return f"{int(mins)}m {int(secs)}s"
            else:
                hours, remainder = divmod(seconds, 3600)
                mins, secs = divmod(remainder, 60)
                return f"{int(hours)}h {int(mins)}m {int(secs)}s"
        
        print(f"Average loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"⏱️  Epoch time: {format_duration(epoch_duration)} | "
              f"Elapsed: {format_duration(total_elapsed)} | "
              f"ETA: {format_duration(eta_seconds)} (finish ~{eta_finish.strftime('%H:%M:%S')})")
        
        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in tqdm(val_loader, desc="Validation"):
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    
                    # Sliding window inference
                    val_outputs = sliding_window_inference(
                        val_inputs,
                        config["inference"]["roi_size"],
                        config["inference"]["sw_batch_size"],
                        model
                    )
                    
                    # Post-process
                    val_outputs = [post_pred({"pred": i})["pred"] for i in decollate_batch(val_outputs)]
                    val_labels = [post_label({"label": i})["label"] for i in decollate_batch(val_labels)]
                    
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                # Get per-class Dice scores
                per_class_dice = dice_metric.aggregate()
                dice_metric.reset()
                
                # Extract individual class scores
                kidney_dice = per_class_dice[0].item()
                tumor_dice = per_class_dice[1].item()
                cyst_dice = per_class_dice[2].item()
                mean_dice = per_class_dice.mean().item()
                
                # Store metrics
                metric_values.append(mean_dice)
                metric_values_per_class["kidney"].append(kidney_dice)
                metric_values_per_class["tumor"].append(tumor_dice)
                metric_values_per_class["cyst"].append(cyst_dice)
                
                # Print detailed validation results
                print(f"\n📊 Validation Results:")
                print(f"   Kidney Dice: {kidney_dice:.4f}")
                print(f"   Tumor Dice:  {tumor_dice:.4f}")
                print(f"   Cyst Dice:   {cyst_dice:.4f}")
                print(f"   Mean Dice:   {mean_dice:.4f}")
                
                # Save best model (based on mean Dice)
                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_metric_epoch = epoch + 1
                    best_metrics_per_class = {
                        "kidney": kidney_dice,
                        "tumor": tumor_dice,
                        "cyst": cyst_dice
                    }
                    
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric,
                        "best_metrics_per_class": best_metrics_per_class,
                    }, checkpoint_dir / "best_model.pth")
                    
                    print(f"\n✅ New best model saved! Mean Dice: {best_metric:.4f}")
                
                print(f"\n🏆 Best Mean Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth")
    
    # Save final model
    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }, checkpoint_dir / "last_model.pth")
    
    # Save training history with detailed per-class metrics
    history = {
        "epoch_loss": epoch_loss_values,
        "val_dice_mean": metric_values,
        "val_dice_kidney": metric_values_per_class["kidney"],
        "val_dice_tumor": metric_values_per_class["tumor"],
        "val_dice_cyst": metric_values_per_class["cyst"],
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "best_metrics_per_class": best_metrics_per_class,
        "config": {
            "epochs": epochs,
            "batch_size": train_cfg["batch_size"],
            "learning_rate": train_cfg["learning_rate"],
            "patch_size": list(config["data"]["patch_size"]),
        }
    }
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Print detailed training summary
    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
    # Format times helper (define again for summary section)
    def format_duration_summary(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m {int(secs)}s"
        else:
            hours, remainder = divmod(seconds, 3600)
            mins, secs = divmod(remainder, 60)
            return f"{int(hours)}h {int(mins)}m {int(secs)}s"
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n⏱️  Timing Summary:")
    print(f"   Started:          {datetime.fromtimestamp(training_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Finished:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Total Time:       {format_duration_summary(total_training_time)}")
    print(f"   Avg Epoch Time:   {format_duration_summary(avg_epoch_time)}")
    print(f"\n📈 Training Summary:")
    print(f"   Total Epochs: {epochs}")
    print(f"   Final Loss: {epoch_loss_values[-1]:.4f}")
    print(f"\n🏆 Best Model (Epoch {best_metric_epoch}):")
    print("-" * 40)
    print(f"   {'Class':<12} {'Dice Score':>12}")
    print("-" * 40)
    print(f"   {'Kidney':<12} {best_metrics_per_class['kidney']:>12.4f}")
    print(f"   {'Tumor':<12} {best_metrics_per_class['tumor']:>12.4f}")
    print(f"   {'Cyst':<12} {best_metrics_per_class['cyst']:>12.4f}")
    print("-" * 40)
    print(f"   {'Mean':<12} {best_metric:>12.4f}")
    print("-" * 40)
    print(f"\n📁 Outputs saved to:")
    print(f"   Checkpoints: {checkpoint_dir}")
    print(f"   History: {checkpoint_dir / 'training_history.json'}")
    print(f"\n💡 To run inference: python {Path(__file__).name} --mode inference")
    print(f"   To evaluate: python {Path(__file__).name} --mode evaluate")
    
    return model


# ============================================================================
# INFERENCE
# ============================================================================
def run_inference(config: dict, checkpoint_path: str = None, output_dir: str = None):
    """
    Run inference on all cases.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for predictions
    """
    print("\n" + "=" * 70)
    print("  INFERENCE: Spatial-Channel Attention U-Net")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = create_model(config, device)
    
    if checkpoint_path is None:
        checkpoint_path = Path(config["checkpoint_dir"]) / "best_model.pth"
    
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    
    # Setup output
    if output_dir is None:
        output_dir = config["predictions_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all cases (including those without segmentation)
    data_dicts = get_kits23_data_dicts(config["kits23_dir"], require_segmentation=False)
    print(f"Running inference on {len(data_dicts)} cases")
    
    # Transforms
    transforms = get_inference_transforms(config)
    
    with torch.no_grad():
        for data_dict in tqdm(data_dicts, desc="Inference"):
            case_id = data_dict["case_id"]
            
            # Load and transform
            data = transforms({"image": data_dict["image"]})
            inputs = data["image"].unsqueeze(0).to(device)
            
            # Sliding window inference
            outputs = sliding_window_inference(
                inputs,
                config["inference"]["roi_size"],
                config["inference"]["sw_batch_size"],
                model,
                overlap=config["inference"]["overlap"]
            )
            
            # Post-process
            pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # Load original to get affine
            orig_nii = nib.load(data_dict["image"])
            
            # Save prediction
            pred_nii = nib.Nifti1Image(pred, orig_nii.affine)
            nib.save(pred_nii, Path(output_dir) / f"{case_id}.nii.gz")
    
    print(f"\n✅ Predictions saved to: {output_dir}")


# ============================================================================
# EVALUATION
# ============================================================================
def compute_dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Compute Dice score for a specific label."""
    pred_binary = (pred == label).astype(np.float32)
    gt_binary = (gt == label).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2 * intersection / union


def evaluate_predictions(config: dict, predictions_dir: str = None):
    """
    Evaluate predictions against ground truth.
    
    Args:
        config: Configuration dictionary
        predictions_dir: Directory containing predictions
    """
    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)
    
    if predictions_dir is None:
        predictions_dir = config["predictions_dir"]
    
    pred_path = Path(predictions_dir)
    data_dicts = get_kits23_data_dicts(config["kits23_dir"], require_segmentation=True)
    
    results = {
        "kidney": [],
        "tumor": [],
        "cyst": [],
        "overall": [],
        "case_ids": []
    }
    
    evaluated_count = 0
    skipped_count = 0
    
    for data_dict in tqdm(data_dicts, desc="Evaluating"):
        case_id = data_dict["case_id"]
        pred_file = pred_path / f"{case_id}.nii.gz"
        
        if not pred_file.exists():
            skipped_count += 1
            continue
        
        evaluated_count += 1
        
        # Load prediction and ground truth
        pred = nib.load(pred_file).get_fdata().astype(np.int32)
        gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
        
        # Handle shape mismatch
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            zoom_factors = np.array(gt.shape) / np.array(pred.shape)
            pred = zoom(pred, zoom_factors, order=0).astype(np.int32)
        
        # Compute per-class Dice
        kidney_dice = compute_dice_score(pred, gt, 1)
        tumor_dice = compute_dice_score(pred, gt, 2)
        cyst_dice = compute_dice_score(pred, gt, 3)
        
        results["kidney"].append(kidney_dice)
        results["tumor"].append(tumor_dice)
        results["cyst"].append(cyst_dice)
        results["overall"].append(np.mean([kidney_dice, tumor_dice, cyst_dice]))
        results["case_ids"].append(case_id)
    
    # Print detailed evaluation results
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n📊 Cases Evaluated: {evaluated_count}")
    if skipped_count > 0:
        print(f"⚠️  Cases Skipped (no prediction): {skipped_count}")
    
    # Print per-class metrics table
    print("\n" + "-" * 65)
    print(f"{'Class':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 65)
    
    for cls in ["kidney", "tumor", "cyst"]:
        scores = results[cls]
        print(f"{cls.capitalize():<12} {np.mean(scores):>10.4f} {np.std(scores):>10.4f} {np.min(scores):>10.4f} {np.max(scores):>10.4f}")
    
    print("-" * 65)
    overall_scores = results["overall"]
    print(f"{'Overall':<12} {np.mean(overall_scores):>10.4f} {np.std(overall_scores):>10.4f} {np.min(overall_scores):>10.4f} {np.max(overall_scores):>10.4f}")
    print("-" * 65)
    
    # Find best and worst cases
    best_idx = np.argmax(results["overall"])
    worst_idx = np.argmin(results["overall"])
    
    print(f"\n🏆 Best Case:  {results['case_ids'][best_idx]} (Overall Dice: {results['overall'][best_idx]:.4f})")
    print(f"📉 Worst Case: {results['case_ids'][worst_idx]} (Overall Dice: {results['overall'][worst_idx]:.4f})")
    
    # Save detailed results
    results_file = Path(config["output_dir"]) / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "per_case": {
                "case_ids": results["case_ids"],
                "kidney": results["kidney"],
                "tumor": results["tumor"],
                "cyst": results["cyst"],
                "overall": results["overall"],
            },
            "summary": {
                "num_cases_evaluated": evaluated_count,
                "num_cases_skipped": skipped_count,
                "kidney_mean": float(np.mean(results["kidney"])),
                "kidney_std": float(np.std(results["kidney"])),
                "kidney_min": float(np.min(results["kidney"])),
                "kidney_max": float(np.max(results["kidney"])),
                "tumor_mean": float(np.mean(results["tumor"])),
                "tumor_std": float(np.std(results["tumor"])),
                "tumor_min": float(np.min(results["tumor"])),
                "tumor_max": float(np.max(results["tumor"])),
                "cyst_mean": float(np.mean(results["cyst"])),
                "cyst_std": float(np.std(results["cyst"])),
                "cyst_min": float(np.min(results["cyst"])),
                "cyst_max": float(np.max(results["cyst"])),
                "overall_mean": float(np.mean(results["overall"])),
                "overall_std": float(np.std(results["overall"])),
                "overall_min": float(np.min(results["overall"])),
                "overall_max": float(np.max(results["overall"])),
                "best_case": results["case_ids"][best_idx],
                "worst_case": results["case_ids"][worst_idx],
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================
def setup_directories(config: dict):
    """Create all required directories."""
    dirs = [
        config["output_dir"],
        config["checkpoint_dir"],
        config["predictions_dir"],
        config["visualizations_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="KiTS23 Spatial-Channel Attention U-Net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python kits23_attention_unet.py --mode train
    python kits23_attention_unet.py --mode train --quick
    python kits23_attention_unet.py --mode inference
    python kits23_attention_unet.py --mode evaluate
    python kits23_attention_unet.py --mode all
        """
    )
    
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "inference", "evaluate", "all"],
        help="Mode to run"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode with 5 epochs"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint for inference or resume training"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Override KiTS23 data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory"
    )
    
    args = parser.parse_args()
    
    # Update config with CLI arguments
    config = CONFIG.copy()
    
    if args.data_dir:
        config["kits23_dir"] = args.data_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
        config["checkpoint_dir"] = f"{args.output_dir}/checkpoints"
        config["predictions_dir"] = f"{args.output_dir}/predictions"
        config["visualizations_dir"] = f"{args.output_dir}/visualizations"
    
    # Determine epochs
    if args.quick:
        num_epochs = config["training"]["quick_epochs"]
    elif args.epochs:
        num_epochs = args.epochs
    else:
        num_epochs = None  # Use default
    
    print("\n" + "=" * 70)
    print("  KiTS23 Spatial-Channel Attention U-Net")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Data directory: {config['kits23_dir']}")
    print(f"Output directory: {config['output_dir']}")
    if num_epochs:
        print(f"Epochs: {num_epochs}")
    
    # Setup directories
    setup_directories(config)
    
    # Run selected mode
    if args.mode in ["train", "all"]:
        train_model(config, num_epochs=num_epochs, resume_from=args.checkpoint)
    
    if args.mode in ["inference", "all"]:
        run_inference(config, checkpoint_path=args.checkpoint)
    
    if args.mode in ["evaluate", "all"]:
        evaluate_predictions(config)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
