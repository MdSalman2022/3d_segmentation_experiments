"""
KiTS23 Spatial and Channel Attention U-Net - V2 (Improved)
============================================================
IMPROVEMENTS OVER V1:
1. Class-weighted loss function (DiceFocalLoss with per-class weights)
2. Oversampling of tumor/cyst regions (3:1 pos/neg ratio)
3. Deep supervision for better gradient flow
4. Two-stage training option (kidney first, then tumor/cyst refinement)
5. Better data augmentation with more aggressive transforms
6. Compound loss: Dice + Focal + Boundary Loss
7. Post-processing with connected component analysis
8. Learning rate warmup + cosine annealing
9. Gradient clipping for stability

Key Features:
- 3D U-Net with Spatial and Channel Attention mechanisms
- Aggressive class balancing for tumor/cyst
- Multi-scale supervision

Labels:
- 0: Background
- 1: Kidney
- 2: Tumor  
- 3: Cyst

Usage:
    python kits23_attention_unet_v2.py --mode train
    python kits23_attention_unet_v2.py --mode train --quick
    python kits23_attention_unet_v2.py --mode inference
    python kits23_attention_unet_v2.py --mode evaluate
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

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
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    EnsureTyped,
    AsDiscreted,
    Invertd,
    SaveImaged,
)
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch


# ============================================================================
# CONFIGURATION - V2 Improved
# ============================================================================
CONFIG = {
    # Dataset paths
    "kits23_dir": "./kits23/dataset",
    
    # Output paths
    "output_dir": "./output/kits23_attention_unet_v2",
    "checkpoint_dir": "./output/kits23_attention_unet_v2/checkpoints",
    "predictions_dir": "./output/kits23_attention_unet_v2/predictions",
    "visualizations_dir": "./output/kits23_attention_unet_v2/visualizations",
    
    # Model architecture - SCALED UP for 34GB VRAM
    "model": {
        "dimensions": 3,
        "in_channels": 1,
        "out_channels": 4,  # background, kidney, tumor, cyst
        "channels": (64, 128, 256, 512, 512),  # Full capacity
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,  # Residual units for better gradient flow
        "attention_ratio": 16,
        "spatial_kernel": 7,
        "dropout": 0.1,
    },
    
    # Training settings - V2 AGGRESSIVE TUMOR FOCUS
    "training": {
        "num_epochs": 200,
        "quick_epochs": 20,
        "batch_size": 1,
        "num_samples": 4,  # Reduced to prevent OOM
        "learning_rate": 1e-4,  # Slightly lower for stability with batch 2
        "use_amp": True,
        "min_learning_rate": 1e-6,
        "weight_decay": 1e-5,
        "val_interval": 2,  # Validate every 2 epochs (faster training)
        "cache_rate": 0.1,
        "num_workers": 4,
        "warmup_epochs": 5,  # Shorter warmup
        "gradient_clip": 1.0,
        
        # AGGRESSIVE class weights for tumor/cyst
        # Much higher weights to force tumor learning
        "class_weights": [0.1, 0.5, 10.0, 8.0],  # bg, kidney, TUMOR, cyst
        
        # Even more aggressive oversampling
        "pos_neg_ratio": (4, 1),  # 4:1 ratio for positive patches
    },
    
    # Data settings - SCALED UP for 34GB VRAM
    "data": {
        "patch_size": (128, 128, 96),  # Larger patches for better context
        "spacing": (1.5, 1.5, 1.5),  # Finer spacing for better resolution
        "intensity_min": -79,
        "intensity_max": 304,
        "train_val_split": 0.85,
    },
    
    # Inference settings
    "inference": {
        "roi_size": (128, 128, 96),
        "sw_batch_size": 2,  # Reduced to prevent OOM during validation
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
# CUSTOM LOSS FUNCTIONS
# ============================================================================
class CompoundLoss(nn.Module):
    """
    Compound loss combining multiple loss functions with class weighting.
    
    Components:
    1. Dice Loss - per-class with weights
    2. Focal Loss - handles class imbalance
    3. Boundary Loss (optional) - focuses on boundaries
    """
    
    def __init__(
        self,
        class_weights: List[float] = None,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        gamma: float = 2.0,  # Focal loss gamma
        include_background: bool = False,
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.include_background = include_background
        
        # Register class weights
        if class_weights is not None:
            self.register_buffer(
                "class_weights", 
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None
        
        # Dice-Focal combined loss from MONAI
        self.dice_focal = DiceFocalLoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            focal_weight=self.class_weights,
            lambda_dice=dice_weight,
            lambda_focal=focal_weight,
            gamma=gamma,
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, D, H, W) softmax predictions
            target: (B, 1, D, H, W) ground truth labels
        """
        return self.dice_focal(pred, target)


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss that handles class imbalance.
    
    With alpha > beta, we penalize false negatives more (good for small structures).
    """
    
    def __init__(
        self,
        alpha: float = 0.3,  # FP weight (lower = less penalty for FP)
        beta: float = 0.7,   # FN weight (higher = more penalty for FN)
        smooth: float = 1e-6,
        to_onehot_y: bool = True,
        include_background: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.to_onehot_y = to_onehot_y
        self.include_background = include_background
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Softmax predictions
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        if self.to_onehot_y:
            num_classes = pred.shape[1]
            target = F.one_hot(target.squeeze(1).long(), num_classes)
            target = target.permute(0, 4, 1, 2, 3).float()
        
        # Skip background if needed
        if not self.include_background:
            pred = pred[:, 1:]
            target = target[:, 1:]
        
        # Flatten spatial dimensions
        pred = pred.view(pred.shape[0], pred.shape[1], -1)
        target = target.view(target.shape[0], target.shape[1], -1)
        
        # Compute Tversky index per class
        tp = (pred * target).sum(dim=2)
        fp = (pred * (1 - target)).sum(dim=2)
        fn = ((1 - pred) * target).sum(dim=2)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Return mean loss
        return 1 - tversky.mean()


class CombinedLossV2(nn.Module):
    """
    V2 Combined Loss with Tversky for better tumor/cyst segmentation.
    """
    
    def __init__(self, class_weights: List[float] = None):
        super().__init__()
        
        if class_weights is not None:
            self.register_buffer(
                "ce_weights",
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.ce_weights = None
        
        # Tversky loss - AGGRESSIVE for tumor detection
        self.tversky = TverskyLoss(
            alpha=0.2,  # Even less penalty for FP (allow some over-segmentation)
            beta=0.8,   # HIGH penalty for FN (never miss a tumor!)
            include_background=False,
        )
        
        # Standard Dice loss
        self.dice = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Tversky loss (main component for small structures)
        tversky_loss = self.tversky(pred, target)
        
        # Dice loss
        dice_loss = self.dice(pred, target)
        
        # Cross-entropy with class weights
        pred_softmax = F.softmax(pred, dim=1)
        ce_loss = F.cross_entropy(
            pred, 
            target.squeeze(1).long(),
            weight=self.ce_weights
        )
        
        # Combine: emphasize Tversky for small structure focus
        return 0.5 * tversky_loss + 0.3 * dice_loss + 0.2 * ce_loss


# ============================================================================
# ATTENTION MODULES (Same as V1)
# ============================================================================
class ChannelAttention(nn.Module):
    """Channel Attention Module (SE-style with avg + max pooling)."""
    
    def __init__(self, submodule: nn.Module, in_planes: int, out_planes: int, 
                 ratio: int = 16):
        super().__init__()
        self.submodule = submodule
        self.in_planes = in_planes
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, max(in_planes // ratio, 1), 1, bias=False),
            nn.GELU(),
            nn.Conv3d(max(in_planes // ratio, 1), out_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.submodule(x)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return y * attention


class SpatialAttention(nn.Module):
    """Spatial Attention Module."""
    
    def __init__(self, submodule: nn.Module, in_channels: int, 
                 kernel_size: int = 7, out_channels: Optional[int] = None,
                 add_conv_1x1: bool = False):
        super().__init__()
        self.submodule = submodule
        self.add_conv_1x1 = add_conv_1x1
        
        self.conv_flat = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        
        self.pool_layer = nn.MaxPool3d(2)
        self.upscale_layer = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.sigmoid = nn.Sigmoid()
        
        if add_conv_1x1:
            if out_channels is None:
                raise ValueError("out_channels required when add_conv_1x1=True")
            self.conv_1x1 = nn.Conv3d(in_channels, out_channels, 1, bias=False)
            self.act = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.submodule(x)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.conv1(x_cat)
        
        if attention.shape[-1] > y.shape[-1]:
            attention = self.pool_layer(attention)
        elif attention.shape[-1] < y.shape[-1]:
            attention = self.upscale_layer(attention)
        
        out = y * self.sigmoid(attention)
        
        if self.add_conv_1x1:
            out = self.act(self.conv_1x1(out))
        
        return out


# ============================================================================
# ATTENTION U-NET MODEL (Same as V1 with dropout)
# ============================================================================
class CheckpointWrapper(nn.Module):
    """Wrapper to enable gradient checkpointing for a module."""
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        if self.training and x.requires_grad:
            return torch_checkpoint(self.module, x, use_reentrant=False)
        return self.module(x)

class AttentionUNet(nn.Module):
    """3D U-Net with Spatial and Channel Attention."""
    
    def __init__(
        self,
        dimensions: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: Sequence[int] = (64, 128, 256, 512, 512),
        strides: Sequence[int] = (2, 2, 2, 2),
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 2,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.1,
        attention_ratio: int = 16,
        spatial_kernel: int = 7,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
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
        
        self.model = self._create_block(
            in_channels, out_channels, channels, strides, is_top=True
        )

    def _apply_checkpoint(self, module: nn.Module) -> nn.Module:
        if self.use_checkpointing:
            return CheckpointWrapper(module)
        return module
    
    def _create_block(
        self,
        inc: int,
        outc: int,
        channels: Sequence[int],
        strides: Sequence[int],
        is_top: bool
    ) -> nn.Sequential:
        c = channels[0]
        s = strides[0] if strides else 1
        
        if len(channels) > 2:
            subblock = self._create_block(c, c, channels[1:], strides[1:], False)
            upc = c * 2
            add_spatial = len(channels) > len(self.channels) - 1
            add_channel = False
        else:
            subblock = self._get_bottom_layer(c, channels[1])
            subblock = ChannelAttention(
                subblock, in_planes=c, out_planes=channels[1],
                ratio=self.attention_ratio
            )
            upc = c + channels[1]
            add_spatial = False
            add_channel = False
        
        down = self._get_down_layer(inc, c, s, is_top)
        if add_spatial:
            down = SpatialAttention(down, in_channels=inc, kernel_size=self.spatial_kernel)
        
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
        
        # Wrap major components for checkpointing
        # Note: SkipConnection's subblock is the most critical part to checkpoint
        if self.use_checkpointing:
            # We don't checkpoint 'down'/up' layers usually as they are fast, 
            # but 'subblock' contains the entire rest of the U-Net.
            # However, for maximum memory savings, we can checkpoint everything.
            
            # Checkpoint the recursive subblock (this is the big savings)
            subblock = CheckpointWrapper(subblock)
            
        return nn.Sequential(down, SkipConnection(subblock), up)
    
    def _get_down_layer(self, in_channels: int, out_channels: int, 
                        strides: int, is_top: bool) -> nn.Module:
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
        return self._get_down_layer(in_channels, out_channels, 1, False)
    
    def _get_up_layer(self, in_channels: int, out_channels: int, 
                      strides: int, is_top: bool) -> nn.Module:
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
# DATA LOADING - V2 Improved
# ============================================================================
def get_kits23_data_dicts(data_dir: str, require_segmentation: bool = True) -> List[Dict]:
    """Get data dictionaries for all KiTS23 cases."""
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


def get_train_transforms_v2(config: dict) -> Compose:
    """
    V2 Training transforms with more aggressive augmentation.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    pos_ratio, neg_ratio = train_cfg["pos_neg_ratio"]
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Spacing - isotropic for better 3D understanding
        Spacingd(
            keys=["image", "label"],
            pixdim=data_cfg["spacing"],
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Intensity normalization
        ScaleIntensityRanged(
            keys=["image"],
            a_min=data_cfg["intensity_min"],
            a_max=data_cfg["intensity_max"],
            b_min=0.0, b_max=1.0, clip=True
        ),
        
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=data_cfg["patch_size"], mode="constant"),
        
        # CRITICAL: Oversample positive (tumor/cyst containing) patches
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=data_cfg["patch_size"],
            pos=pos_ratio,  # 3
            neg=neg_ratio,  # 1
            num_samples=train_cfg["num_samples"],
            image_key="image",
            image_threshold=0,
            allow_smaller=True
        ),
        
        # More aggressive augmentation
        RandRotated(
            keys=["image", "label"],
            range_x=0.3,
            range_y=0.3,
            range_z=0.3,
            prob=0.5,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.3,
            mode=("trilinear", "nearest"),
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        
        # Intensity augmentations
        RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
        
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms_v2(config: dict) -> Compose:
    """V2 Validation transforms."""
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
        SpatialPadd(keys=["image", "label"], spatial_size=data_cfg["patch_size"], mode="constant"),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_inference_transforms_v2(config: dict) -> Compose:
    """V2 Inference transforms."""
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
        SpatialPadd(keys=["image"], spatial_size=data_cfg["patch_size"], mode="constant"),
        EnsureTyped(keys=["image"]),
    ])


# ============================================================================
# POST-PROCESSING
# ============================================================================
def postprocess_prediction(pred: np.ndarray, min_kidney_size: int = 1000, 
                          min_tumor_size: int = 50) -> np.ndarray:
    """
    Post-process predictions with connected component analysis.
    
    Args:
        pred: (D, H, W) integer prediction array
        min_kidney_size: Minimum voxels for kidney
        min_tumor_size: Minimum voxels for tumor/cyst
    
    Returns:
        Cleaned prediction
    """
    result = np.zeros_like(pred)
    
    # Process each class
    for label_idx, min_size in [(1, min_kidney_size), (2, min_tumor_size), (3, min_tumor_size)]:
        binary_mask = (pred == label_idx).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            continue
        
        # Connected component analysis
        labeled_array, num_features = ndimage.label(binary_mask)
        
        if num_features == 0:
            continue
        
        # Keep only components larger than min_size
        for component_id in range(1, num_features + 1):
            component_mask = labeled_array == component_id
            if component_mask.sum() >= min_size:
                result[component_mask] = label_idx
    
    return result


# ============================================================================
# TRAINING - V2 Improved
# ============================================================================
def create_model_v2(config: dict, device: torch.device) -> AttentionUNet:
    """Create the Attention U-Net model with V2 settings."""
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

        dropout=model_cfg.get("dropout", 0.1),
        use_checkpointing=True,
    )
    
    return model.to(device)


def get_learning_rate(epoch: int, warmup_epochs: int, max_epochs: int,
                      base_lr: float, min_lr: float) -> float:
    """Get learning rate with warmup and cosine annealing."""
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


def train_model_v2(config: dict, num_epochs: int = None, resume_from: str = None):
    """
    V2 Training with class weighting and improved loss.
    """
    print("\n" + "=" * 70)
    print("  TRAINING V2: Spatial-Channel Attention U-Net for KiTS23")
    print("  Improvements: Class weighting, Oversampling, Better loss")
    print("=" * 70)
    
    set_determinism(seed=config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Directories
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    print("\n📊 Loading dataset...")
    data_dicts = get_kits23_data_dicts(config["kits23_dir"])
    print(f"Found {len(data_dicts)} cases with segmentation")
    
    # Split
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    train_files = data_dicts[:split_idx]
    val_files = data_dicts[split_idx:]
    print(f"Train: {len(train_files)} cases")
    print(f"Val: {len(val_files)} cases")
    
    # Transforms
    train_transforms = get_train_transforms_v2(config)
    val_transforms = get_val_transforms_v2(config)
    
    train_cfg = config["training"]
    
    # Print V2 improvements
    print("\n🆕 V2 Improvements:")
    print(f"   Class weights: {train_cfg['class_weights']}")
    print(f"   Pos/Neg ratio: {train_cfg['pos_neg_ratio']}")
    print(f"   Warmup epochs: {train_cfg['warmup_epochs']}")
    print(f"   Gradient clipping: {train_cfg['gradient_clip']}")
    
    # Datasets
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
        train_ds = Dataset(data=train_files, transform=train_transforms)
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    # Loaders
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
    
    # Model
    print("\n🔧 Creating model...")
    model = create_model_v2(config, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Resume
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
    
    # V2 Loss with class weights
    class_weights = train_cfg["class_weights"]
    loss_function = CombinedLossV2(class_weights=class_weights).to(device)
    print(f"\n📉 Loss: CombinedLossV2 (Tversky + Dice + WeightedCE)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    
    # Training settings
    epochs = num_epochs if num_epochs else train_cfg["num_epochs"]
    val_interval = train_cfg["val_interval"]
    warmup_epochs = train_cfg["warmup_epochs"]
    gradient_clip = train_cfg["gradient_clip"]
    base_lr = train_cfg["learning_rate"]
    min_lr = train_cfg["min_learning_rate"]
    use_amp = train_cfg.get("use_amp", True)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"   Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    
    # Post-processing
    post_pred = Compose([
        EnsureTyped(keys="pred"),
        AsDiscreted(keys="pred", argmax=True, to_onehot=4)
    ])
    post_label = Compose([
        EnsureTyped(keys="label"),
        AsDiscreted(keys="label", to_onehot=4)
    ])
    
    # History
    epoch_loss_values = []
    metric_values = []
    metric_values_per_class = {"kidney": [], "tumor": [], "cyst": []}
    best_metric_epoch = -1
    best_metrics_per_class = {"kidney": 0.0, "tumor": 0.0, "cyst": 0.0}
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Validation every {val_interval} epoch(s)")
    
    training_start_time = time.time()
    epoch_times = []
    
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Update learning rate with warmup + cosine
        current_lr = get_learning_rate(
            epoch, warmup_epochs, epochs, base_lr, min_lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        print("-" * 70)
        print(f"Epoch {epoch + 1}/{epochs} | 🕐 Started: {current_time} | LR: {current_lr:.2e}")
        
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in tqdm(train_loader, desc="Training"):
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping (unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # Clear cache after each step to prevent memory buildup
            del outputs, loss
            torch.cuda.empty_cache()
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        # Timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        total_elapsed = epoch_end_time - training_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_finish = datetime.now() + timedelta(seconds=eta_seconds)
        
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
        
        print(f"Average loss: {epoch_loss:.4f}")
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
                    
                    val_outputs = sliding_window_inference(
                        val_inputs,
                        config["inference"]["roi_size"],
                        config["inference"]["sw_batch_size"],
                        model
                    )
                    
                    val_outputs = [post_pred({"pred": i})["pred"] for i in decollate_batch(val_outputs)]
                    val_labels = [post_label({"label": i})["label"] for i in decollate_batch(val_labels)]
                    
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                per_class_dice = dice_metric.aggregate()
                dice_metric.reset()
                
                kidney_dice = per_class_dice[0].item()
                tumor_dice = per_class_dice[1].item()
                cyst_dice = per_class_dice[2].item()
                mean_dice = per_class_dice.mean().item()
                
                metric_values.append(mean_dice)
                metric_values_per_class["kidney"].append(kidney_dice)
                metric_values_per_class["tumor"].append(tumor_dice)
                metric_values_per_class["cyst"].append(cyst_dice)
                
                print(f"\n📊 Validation Results:")
                print(f"   Kidney Dice: {kidney_dice:.4f}")
                print(f"   Tumor Dice:  {tumor_dice:.4f}")
                print(f"   Cyst Dice:   {cyst_dice:.4f}")
                print(f"   Mean Dice:   {mean_dice:.4f}")
                
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
            
            # CRITICAL: Clear CUDA cache after validation to prevent OOM
            torch.cuda.empty_cache()
        
        # Checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth")
    
    # Save final
    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }, checkpoint_dir / "last_model.pth")
    
    # History
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
            "class_weights": train_cfg["class_weights"],
            "pos_neg_ratio": list(train_cfg["pos_neg_ratio"]),
        }
    }
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Summary
    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
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
    
    return model


# ============================================================================
# INFERENCE
# ============================================================================
def run_inference_v2(config: dict, checkpoint_path: str = None, output_dir: str = None,
                     use_postprocessing: bool = True):
    """V2 Inference with optional post-processing."""
    print("\n" + "=" * 70)
    print("  INFERENCE V2: Spatial-Channel Attention U-Net")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = create_model_v2(config, device)
    
    if checkpoint_path is None:
        checkpoint_path = Path(config["checkpoint_dir"]) / "best_model.pth"
    
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    
    if output_dir is None:
        output_dir = config["predictions_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    data_dicts = get_kits23_data_dicts(config["kits23_dir"], require_segmentation=False)
    print(f"Running inference on {len(data_dicts)} cases")
    print(f"Post-processing: {use_postprocessing}")
    
    transforms = get_inference_transforms_v2(config)
    
    with torch.no_grad():
        for data_dict in tqdm(data_dicts, desc="Inference"):
            case_id = data_dict["case_id"]
            
            data = transforms({"image": data_dict["image"]})
            inputs = data["image"].unsqueeze(0).to(device)
            
            outputs = sliding_window_inference(
                inputs,
                config["inference"]["roi_size"],
                config["inference"]["sw_batch_size"],
                model,
                overlap=config["inference"]["overlap"]
            )
            
            pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # Optional post-processing
            if use_postprocessing:
                pred = postprocess_prediction(pred)
            
            orig_nii = nib.load(data_dict["image"])
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


def evaluate_predictions_v2(config: dict, predictions_dir: str = None):
    """V2 Evaluation with detailed metrics."""
    print("\n" + "=" * 70)
    print("  EVALUATION V2")
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
        
        pred = nib.load(pred_file).get_fdata().astype(np.int32)
        gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
        
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            zoom_factors = np.array(gt.shape) / np.array(pred.shape)
            pred = zoom(pred, zoom_factors, order=0).astype(np.int32)
        
        kidney_dice = compute_dice_score(pred, gt, 1)
        tumor_dice = compute_dice_score(pred, gt, 2)
        cyst_dice = compute_dice_score(pred, gt, 3)
        
        results["kidney"].append(kidney_dice)
        results["tumor"].append(tumor_dice)
        results["cyst"].append(cyst_dice)
        results["overall"].append(np.mean([kidney_dice, tumor_dice, cyst_dice]))
        results["case_ids"].append(case_id)
    
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS (V2)")
    print("=" * 70)
    print(f"\n📊 Cases Evaluated: {evaluated_count}")
    if skipped_count > 0:
        print(f"⚠️  Cases Skipped: {skipped_count}")
    
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
    
    best_idx = np.argmax(results["overall"])
    worst_idx = np.argmin(results["overall"])
    
    print(f"\n🏆 Best Case:  {results['case_ids'][best_idx]} (Dice: {results['overall'][best_idx]:.4f})")
    print(f"📉 Worst Case: {results['case_ids'][worst_idx]} (Dice: {results['overall'][worst_idx]:.4f})")
    
    results_file = Path(config["output_dir"]) / "evaluation_results_v2.json"
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
                "kidney_mean": float(np.mean(results["kidney"])),
                "kidney_std": float(np.std(results["kidney"])),
                "tumor_mean": float(np.mean(results["tumor"])),
                "tumor_std": float(np.std(results["tumor"])),
                "cyst_mean": float(np.mean(results["cyst"])),
                "cyst_std": float(np.std(results["cyst"])),
                "overall_mean": float(np.mean(results["overall"])),
                "overall_std": float(np.std(results["overall"])),
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
        description="KiTS23 Spatial-Channel Attention U-Net V2 (Improved)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V2 IMPROVEMENTS:
    - Class-weighted loss (Tversky + Dice + WeightedCE)
    - 3:1 oversampling of tumor/cyst regions
    - Residual units for better gradient flow
    - Learning rate warmup + cosine annealing
    - Gradient clipping for stability
    - Post-processing with connected components

Examples:
    python kits23_attention_unet_v2.py --mode train
    python kits23_attention_unet_v2.py --mode train --quick
    python kits23_attention_unet_v2.py --mode inference
    python kits23_attention_unet_v2.py --mode evaluate
        """
    )
    
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "inference", "evaluate", "all"],
        help="Mode to run"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode with 10 epochs"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Override KiTS23 data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--no_postprocess", action="store_true",
        help="Disable post-processing during inference"
    )
    
    args = parser.parse_args()
    
    config = CONFIG.copy()
    
    if args.data_dir:
        config["kits23_dir"] = args.data_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
        config["checkpoint_dir"] = f"{args.output_dir}/checkpoints"
        config["predictions_dir"] = f"{args.output_dir}/predictions"
        config["visualizations_dir"] = f"{args.output_dir}/visualizations"
    
    if args.quick:
        num_epochs = config["training"]["quick_epochs"]
    elif args.epochs:
        num_epochs = args.epochs
    else:
        num_epochs = None
    
    print("\n" + "=" * 70)
    print("  KiTS23 Spatial-Channel Attention U-Net V2 (IMPROVED)")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Data directory: {config['kits23_dir']}")
    print(f"Output directory: {config['output_dir']}")
    if num_epochs:
        print(f"Epochs: {num_epochs}")
    
    setup_directories(config)
    
    if args.mode in ["train", "all"]:
        train_model_v2(config, num_epochs=num_epochs, resume_from=args.checkpoint)
    
    if args.mode in ["inference", "all"]:
        run_inference_v2(
            config, 
            checkpoint_path=args.checkpoint,
            use_postprocessing=not args.no_postprocess
        )
    
    if args.mode in ["evaluate", "all"]:
        evaluate_predictions_v2(config)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
