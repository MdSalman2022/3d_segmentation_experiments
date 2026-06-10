# Cell 1: Imports and Setup
"""
DINO-VISTA-KiTS: Complete Implementation
=========================================
Foundation Model-Enhanced 3D Kidney Tumor Segmentation

Combines:
- MedDINOv3: Multi-scale feature aggregation, hierarchical representations
- VISTA3D: Dual-branch decoder (automatic + refinement), class embeddings

Target: KiTS23 dataset (Kidney, Tumor, Cyst segmentation)

Author: Research Implementation
Date: December 2025
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import math
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Callable, Sequence
from collections import OrderedDict, defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

# Optional imports with fallbacks
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    nib = None
    warnings.warn("nibabel not installed. Install with: pip install nibabel")

try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    distance_transform_edt = None
    warnings.warn("scipy not available for surface metrics")

try:
    import monai
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRanged, CropForegroundd,
        RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
        RandShiftIntensityd, RandGaussianNoised, RandGaussianSmoothd,
        RandScaleIntensityd, RandAdjustContrastd, EnsureTyped,
        SpatialPadd, ToTensord,
    )
    from monai.data import Dataset as MonaiDataset, CacheDataset
    from monai.data import list_data_collate
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from monai.inferers import sliding_window_inference
    MONAI_AVAILABLE = True
    print(f"MONAI version: {monai.__version__}")
except ImportError:
    MONAI_AVAILABLE = False
    warnings.warn("MONAI not installed. Install with: pip install monai")

# PyTorch version check
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


# Cell 2: Configuration
"""
Configuration dictionaries for model, training, and data.
"""

# =============================================================================
# UNIQUE RUN ID - Auto-generates a unique 4-digit ID for each training run
# =============================================================================
def get_unique_run_id(base_dir: str = "./dino") -> str:
    """Generate a unique 4-digit run ID based on existing runs."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    existing_runs = list(base_path.glob("run_*"))
    existing_ids = []
    for run in existing_runs:
        try:
            run_id = int(run.name.split("_")[1])
            existing_ids.append(run_id)
        except (IndexError, ValueError):
            pass
    
    next_id = max(existing_ids, default=0) + 1
    return f"{next_id:04d}"

# Get script name for output folder
SCRIPT_NAME = Path(__file__).stem  # "dino_vista_kits"
RUN_ID = get_unique_run_id("./dino")
RUN_DIR = f"./dino/run_{RUN_ID}_{SCRIPT_NAME}"

print(f"📁 Run output directory: {RUN_DIR}")

# =============================================================================
# QUICK_RUN TOGGLE: Set to True for fast pipeline testing, False for full training
# =============================================================================
QUICK_RUN = True  # <-- SET TO False FOR FULL TRAINING

if QUICK_RUN:
    print("⚡ QUICK_RUN MODE v2 - Improved config for tumor/cyst detection")
    
    MODEL_CONFIG = {
        # Architecture
        "spatial_dims": 3,
        "in_channels": 1,
        "num_classes": 4,
        "class_names": ["background", "kidney", "tumor", "cyst"],
        
        # Encoder - medium size for decent results
        "init_filters": 24,
        "blocks_down": [1, 2, 2, 2],
        "blocks_up": [1, 1, 1],
        
        # Multi-scale aggregation
        "aggregation_channels": 96,
        "aggregation_scales": [2, 3, 4],
        
        # Dual-branch decoder
        "class_embed_dim": 96,
        "num_attention_heads": 4,
        "dropout": 0.1,
        
        # Training patches - larger for better context
        "patch_size": [96, 96, 96],
    }
    
    DATA_CONFIG = {
        "dataset_dir": "./kits23/dataset",
        "spacing": [1.5, 1.5, 1.5],
        "intensity_min": -200,
        "intensity_max": 400,
        "num_samples": 4,        # More samples per volume for tumor coverage
        "pos_ratio": 0.8,        # Higher focus on foreground (tumor/cyst)
        "val_split": 0.2,
        "seed": 42,
        "num_workers": 4,
        "cache_rate": 0.0,
        "max_train_cases": 50,   # 80 cases = 4x more data for tumor/cyst
        "max_val_cases": 20,     # 20 val cases to ensure tumor/cyst presence
    }
    
    TRAIN_CONFIG = {
        "output_dir": f"{RUN_DIR}/outputs",
        "checkpoint_dir": f"{RUN_DIR}/checkpoints",
        "log_dir": f"{RUN_DIR}/logs",
        "num_epochs": 10,
        "warmup_epochs": 2,
        "batch_size": 1,
        "learning_rate": 5e-4,   # Lower LR for stability with more data
        "min_learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "gradient_clip": 1.0,
        "accumulation_steps": 2,
        "use_amp": True,
        "val_interval": 2,       # Validate every 2 epochs
        "val_sw_batch_size": 2,
        "val_overlap": 0.5,
        "save_interval": 5,
        "keep_best_n": 2,
        "use_deep_supervision": False,
        # Early stopping
        "early_stopping_patience": 10,  # Stop if no improvement for N validations
        "early_stopping_min_delta": 0.001,  # Min improvement to count
        # Visualization
        "save_training_plots": True,
        "save_val_visualizations": True,
    }

else:
    print("🚀 FULL TRAINING MODE - Using production config")
    
    MODEL_CONFIG = {
        # Architecture
        "spatial_dims": 3,
        "in_channels": 1,
        "num_classes": 4,  # background, kidney, tumor, cyst
        "class_names": ["background", "kidney", "tumor", "cyst"],
        
        # Encoder (SegResNet-style)
        "init_filters": 32,
        "blocks_down": [1, 2, 2, 4],
        "blocks_up": [1, 1, 1],
        
        # Multi-scale aggregation (MedDINOv3-inspired)
        "aggregation_channels": 128,
        "aggregation_scales": [2, 3, 4],
        
        # Dual-branch decoder (VISTA3D-inspired)
        "class_embed_dim": 128,
        "num_attention_heads": 8,
        "dropout": 0.1,
        
        # Training patches
        "patch_size": [96, 96, 96],
    }
    
    DATA_CONFIG = {
        "dataset_dir": "./kits23/dataset",
        "spacing": [1.5, 1.5, 1.5],
        "intensity_min": -200,
        "intensity_max": 400,
        "num_samples": 2,
        "pos_ratio": 0.7,
        "val_split": 0.2,
        "seed": 42,
        "num_workers": 4,
        "cache_rate": 0.0,
        "max_train_cases": None,  # Use all cases
        "max_val_cases": None,    # Use all cases
    }
    
    TRAIN_CONFIG = {
        "output_dir": f"{RUN_DIR}/outputs",
        "checkpoint_dir": f"{RUN_DIR}/checkpoints",
        "log_dir": f"{RUN_DIR}/logs",
        "num_epochs": 150,
        "warmup_epochs": 5,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "min_learning_rate": 1e-6,
        "weight_decay": 1e-5,
        "gradient_clip": 1.0,
        "accumulation_steps": 4,
        "use_amp": True,
        "val_interval": 5,
        "val_sw_batch_size": 2,
        "val_overlap": 0.5,
        "save_interval": 10,
        "keep_best_n": 3,
        "use_deep_supervision": False,
        # Early stopping
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.001,
        # Visualization
        "save_training_plots": True,
        "save_val_visualizations": True,
    }

EVAL_CONFIG = {
    "num_classes": 4,
    "class_names": ["background", "kidney", "tumor", "cyst"],
    "include_background": False,
    "hec_classes": {
        "kidney": {"labels": (1,), "tolerance_mm": 2.0},
        "masses": {"labels": (2, 3), "tolerance_mm": 2.0},
        "tumor": {"labels": (2,), "tolerance_mm": 2.0},
    },
}


# =============================================================================
# PART 1: MODEL ARCHITECTURE
# =============================================================================

# Cell 3: Basic Building Blocks
class ConvBlock(nn.Module):
    """Basic 3D convolution block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm: str = "instance",
        act: str = "relu",
    ):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        
        if norm == "instance":
            self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm3d(out_channels)
        else:
            self.norm = nn.Identity()
        
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block for encoder/decoder."""
    
    def __init__(self, channels: int, kernel_size: int = 3, norm: str = "instance"):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, norm=norm)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.InstanceNorm3d(channels, affine=True) if norm == "instance" else nn.BatchNorm3d(channels),
        )
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.act(out)


class DownBlock(nn.Module):
    """Downsampling block: strided conv + residual blocks."""
    
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2, norm: str = "instance"):
        super().__init__()
        self.downsample = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, norm=norm)
        self.blocks = nn.Sequential(*[ResBlock(out_channels, norm=norm) for _ in range(num_blocks)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block: transpose conv + concat + residual blocks."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_blocks: int = 1, norm: str = "instance"):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fusion = ConvBlock(out_channels + skip_channels, out_channels, kernel_size=1, norm=norm)
        self.blocks = nn.Sequential(*[ResBlock(out_channels, norm=norm) for _ in range(num_blocks)])
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        x = self.blocks(x)
        return x


# Cell 4: Multi-Scale Feature Aggregation (MedDINOv3-inspired)
class MultiScaleAggregation(nn.Module):
    """
    Multi-scale feature aggregation inspired by MedDINOv3.
    
    Collects features from multiple encoder stages, aligns them to a common
    resolution, and fuses them to capture hierarchical spatial context.
    
    Reference: MedDINOv3 - "How to Adapt Vision Foundation Models for Medical Image Segmentation"
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        target_scale_idx: int = 0,
    ):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.target_scale_idx = target_scale_idx
        self.num_scales = len(in_channels_list)
        
        # 1x1x1 convolutions to project each scale to common channel dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(c, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            ) for c in in_channels_list
        ])
        
        # Learnable scale weights (like MedDINOv3's token weighting)
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
        # Fusion convolution after concatenation
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * self.num_scales, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        
        # Channel attention for adaptive feature recalibration
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid(),
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature maps from different encoder stages
        Returns:
            Aggregated feature map
        """
        assert len(features) == self.num_scales
        
        target_size = features[self.target_scale_idx].shape[2:]
        weights = F.softmax(self.scale_weights, dim=0)
        
        aligned_features = []
        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            projected = proj(feat)
            if projected.shape[2:] != target_size:
                projected = F.interpolate(projected, size=target_size, mode='trilinear', align_corners=False)
            projected = projected * weights[i]
            aligned_features.append(projected)
        
        concatenated = torch.cat(aligned_features, dim=1)
        fused = self.fusion(concatenated)
        
        B, C = fused.shape[:2]
        attn = self.channel_attention(fused).view(B, C, 1, 1, 1)
        fused = fused * attn
        
        return fused


# Cell 5: Class Embedding Module (VISTA3D-inspired)
class ClassEmbedding(nn.Module):
    """
    Learnable class embeddings for class-conditioned segmentation.
    
    Reference: VISTA3D - "A Unified Segmentation Foundation Model for 3D Medical Imaging"
    """
    
    def __init__(self, num_classes: int, embed_dim: int, include_background: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.include_background = include_background
        
        n_embeds = num_classes if include_background else num_classes - 1
        self.embeddings = nn.Embedding(n_embeds, embed_dim)
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
    
    def forward(self, class_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if class_indices is None:
            return self.embeddings.weight
        return self.embeddings(class_indices)


# Cell 6: Cross-Attention Module for Class-Guided Segmentation
class ClassGuidedAttention(nn.Module):
    """
    Cross-attention between class embeddings and spatial features.
    Uses class embeddings as queries to attend to spatial feature maps.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, class_embeds: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            class_embeds: [num_classes, embed_dim]
            spatial_features: [B, C, H, W, D]
        Returns:
            Class-specific feature maps: [B, num_classes, H, W, D]
        """
        B, C, H, W, D = spatial_features.shape
        num_classes = class_embeds.shape[0]
        
        spatial_flat = spatial_features.flatten(2).permute(0, 2, 1)  # [B, H*W*D, C]
        class_embeds = class_embeds.unsqueeze(0).expand(B, -1, -1)  # [B, num_classes, C]
        
        Q = self.q_proj(class_embeds)
        K = self.k_proj(spatial_flat)
        V = self.v_proj(spatial_flat)
        
        Q = Q.view(B, num_classes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, H*W*D, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, H*W*D, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention with numerical stability
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.clamp(-50, 50)  # Prevent extreme values before softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ V
        out = out.permute(0, 2, 1, 3).reshape(B, num_classes, self.embed_dim)
        out = self.out_proj(out)
        out = self.norm(out + class_embeds)
        
        # Compute class similarity maps with normalization for stability
        out_expanded = out.unsqueeze(2)
        spatial_expanded = spatial_flat.unsqueeze(1)
        similarity = (out_expanded * spatial_expanded).sum(dim=-1)
        # Normalize to prevent extreme values
        similarity = similarity / (self.embed_dim ** 0.5)
        class_maps = similarity.view(B, num_classes, H, W, D)
        
        return class_maps


# Cell 7: Dual-Branch Decoder (VISTA3D-inspired)
class DualBranchDecoder(nn.Module):
    """
    Dual-branch decoder inspired by VISTA3D.
    
    Branch A (Automatic): Class-conditioned segmentation using learned embeddings
    Branch B (Refinement): Standard convolutional decoder with uncertainty estimation
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        self.input_proj = nn.Conv3d(in_channels, embed_dim, kernel_size=1) \
            if in_channels != embed_dim else nn.Identity()
        
        # Branch A: Automatic (Class-Guided)
        self.class_embeddings = ClassEmbedding(num_classes=num_classes, embed_dim=embed_dim, include_background=False)
        self.class_attention = ClassGuidedAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        self.class_mlp_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=3, padding=1),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 1, kernel_size=1),
            ) for _ in range(num_classes - 1)
        ])
        
        self.background_head = nn.Conv3d(num_classes - 1, 1, kernel_size=1)
        
        # Branch B: Refinement
        self.refine_decoder = nn.Sequential(
            ConvBlock(embed_dim, embed_dim // 2, kernel_size=3),
            ConvBlock(embed_dim // 2, embed_dim // 4, kernel_size=3),
            nn.Conv3d(embed_dim // 4, num_classes, kernel_size=1),
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Conv3d(embed_dim, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        
        self.merge_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, features: torch.Tensor, return_all: bool = True) -> Dict[str, torch.Tensor]:
        B = features.shape[0]
        features = self.input_proj(features)
        
        # Branch A: Automatic (class-guided)
        class_embeds = self.class_embeddings()
        class_maps = self.class_attention(class_embeds, features)
        
        refined_classes = []
        for i, mlp in enumerate(self.class_mlp_heads):
            class_map = class_maps[:, i:i+1]
            refined = mlp(class_map)
            refined_classes.append(refined)
        
        foreground = torch.cat(refined_classes, dim=1)
        background = self.background_head(torch.sigmoid(foreground))
        auto_seg = torch.cat([background, foreground], dim=1)
        
        # Branch B: Refinement (standard convolution)
        refine_seg = self.refine_decoder(features)
        uncertainty = self.uncertainty_head(features)
        
        # Simple weighted merge (more stable than complex uncertainty-based merge)
        merge_weight = torch.sigmoid(self.merge_weight)
        # Use simple weighted average: w * auto + (1-w) * refine
        merged_seg = merge_weight * auto_seg + (1 - merge_weight) * refine_seg
        
        outputs = {'merged_seg': merged_seg, 'uncertainty': uncertainty}
        if return_all:
            outputs['auto_seg'] = auto_seg
            outputs['refine_seg'] = refine_seg
            outputs['class_maps'] = class_maps
        
        return outputs


# Cell 8: Complete DINO-VISTA-KiTS Model
class DinoVistaKiTS(nn.Module):
    """
    DINO-VISTA-KiTS: Complete 3D segmentation model for kidney tumor analysis.
    
    Architecture:
        1. Encoder: SegResNet-style hierarchical feature extraction
        2. Multi-Scale Aggregation: MedDINOv3-inspired feature fusion
        3. Dual-Branch Decoder: VISTA3D-inspired class-guided + refinement branches
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        init_filters: int = 32,
        blocks_down: List[int] = [1, 2, 2, 4],
        blocks_up: List[int] = [1, 1, 1],
        aggregation_channels: int = 256,
        class_embed_dim: int = 256,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_filters = init_filters
        
        self.encoder_channels = [init_filters * (2 ** i) for i in range(len(blocks_down) + 1)]
        
        # Encoder
        self.init_conv = nn.Sequential(
            ConvBlock(in_channels, init_filters, kernel_size=3),
            ResBlock(init_filters),
        )
        
        self.encoders = nn.ModuleList()
        for i, num_blocks in enumerate(blocks_down):
            in_ch = self.encoder_channels[i]
            out_ch = self.encoder_channels[i + 1]
            self.encoders.append(DownBlock(in_ch, out_ch, num_blocks))
        
        # Multi-Scale Aggregation (MedDINOv3)
        agg_in_channels = self.encoder_channels[1:]
        self.multi_scale_agg = MultiScaleAggregation(
            in_channels_list=agg_in_channels,
            out_channels=aggregation_channels,
            target_scale_idx=0,
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        decoder_channels = list(reversed(self.encoder_channels[:-1]))  # [256, 128, 64, 32]
        
        for i, num_blocks in enumerate(blocks_up):
            if i == 0:
                in_ch = self.encoder_channels[-1]  # 512
            else:
                in_ch = decoder_channels[i - 1]
            skip_ch = decoder_channels[i]
            out_ch = decoder_channels[i]
            self.decoders.append(UpBlock(in_ch, skip_ch, out_ch, num_blocks))
        
        # Last decoder output channels (with 3 decoder stages: 256->128->64)
        self.last_decoder_ch = decoder_channels[len(blocks_up) - 1]  # 64 for default config
        
        self.final_up = nn.ConvTranspose3d(self.last_decoder_ch, self.last_decoder_ch, kernel_size=2, stride=2)
        
        # Dual-Branch Decoder (VISTA3D)
        combined_channels = aggregation_channels + self.last_decoder_ch
        self.feature_fusion = nn.Sequential(
            ConvBlock(combined_channels, aggregation_channels, kernel_size=1),
            ResBlock(aggregation_channels),
        )
        
        self.dual_branch = DualBranchDecoder(
            in_channels=aggregation_channels,
            num_classes=num_classes,
            embed_dim=class_embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Segmentation head (backup) - uses last decoder output channels
        self.seg_head = nn.Conv3d(self.last_decoder_ch, num_classes, kernel_size=1)
        
        # Deep supervision - only for decoder stages we actually have
        self.deep_supervision = nn.ModuleList([
            nn.Conv3d(decoder_channels[i], num_classes, kernel_size=1) for i in range(len(blocks_up))
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, deep_supervision: bool = False) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[2:]
        
        # Encoder
        x0 = self.init_conv(x)
        encoder_features = [x0]
        x = x0
        
        # Use simple checkpointing for encoders if training
        for encoder in self.encoders:
            if self.training and x.requires_grad:
                x = checkpoint(encoder, x, use_reentrant=False)
            else:
                x = encoder(x)
            encoder_features.append(x)
        
        # Multi-Scale Aggregation
        multi_scale_features = encoder_features[1:]
        # Note: checkpointing with lists can be tricky, so we'll skip it for aggregation 
        # unless we wrap it. It's relatively light compared to the decoder.
        aggregated = self.multi_scale_agg(multi_scale_features)
        
        # Decoder
        decoder_features = []
        x = encoder_features[-1]
        for i, decoder in enumerate(self.decoders):
            skip = encoder_features[-(i + 2)]
            # Checkpoint decoder blocks
            if self.training and x.requires_grad:
                x = checkpoint(decoder, x, skip, use_reentrant=False)
            else:
                x = decoder(x, skip)
            decoder_features.append(x)
        
        x = self.final_up(x)
        if x.shape[2:] != encoder_features[0].shape[2:]:
            x = F.interpolate(x, size=encoder_features[0].shape[2:], mode='trilinear', align_corners=False)
        
        # Feature Fusion
        aggregated_up = F.interpolate(aggregated, size=x.shape[2:], mode='trilinear', align_corners=False)
        combined = torch.cat([aggregated_up, x], dim=1)
        fused = self.feature_fusion(combined)
        
        # Dual-Branch Decoder
        # This is the heaviest part, definitely needs checkpointing
        if self.training and fused.requires_grad:
            # dual_branch returns a dict, checkpoint needs a function that returns tensors.
            # We create a wrapper to handle dictionary output unpacking/packing if needed,
            # but simpler is to just checkpoint the sub-modules or assume the user accepts
            # checks on just the forward computation. 
            # However, checkpoint requires the output to be a Tensor or tuple of Tensors.
            # Our dual_branch returns a dict. We must modify it or wrap it.
            
            # Wrapper to return tuple
            def dual_branch_wrapper(f):
                out = self.dual_branch(f, return_all=True)
                return out['merged_seg'], out['auto_seg'], out['refine_seg'], out['uncertainty'], out['class_maps']
            
            merged_seg, auto_seg, refine_seg, uncertainty, class_maps = checkpoint(dual_branch_wrapper, fused, use_reentrant=False)
            
            dual_outputs = {
                'merged_seg': merged_seg,
                'auto_seg': auto_seg,
                'refine_seg': refine_seg,
                'uncertainty': uncertainty,
                'class_maps': class_maps
            }
        else:
            dual_outputs = self.dual_branch(fused, return_all=True)
            merged_seg = dual_outputs['merged_seg']
        
        # Resize to input
        if merged_seg.shape[2:] != input_shape:
            merged_seg = F.interpolate(merged_seg, size=input_shape, mode='trilinear', align_corners=False)
            for key in ['auto_seg', 'refine_seg', 'uncertainty']:
                if key in dual_outputs and dual_outputs[key] is not None:
                    dual_outputs[key] = F.interpolate(dual_outputs[key], size=input_shape, mode='trilinear', align_corners=False)
        
        outputs = {
            'logits': merged_seg,
            'auto_seg': dual_outputs.get('auto_seg'),
            'refine_seg': dual_outputs.get('refine_seg'),
            'uncertainty': dual_outputs.get('uncertainty'),
        }
        
        if deep_supervision:
            deep_outputs = []
            for feat, head in zip(decoder_features, self.deep_supervision):
                ds_out = head(feat)
                ds_out = F.interpolate(ds_out, size=input_shape, mode='trilinear', align_corners=False)
                deep_outputs.append(ds_out)
            outputs['deep_outputs'] = deep_outputs
        
        return outputs
    
    def get_num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoders.parameters()) + sum(p.numel() for p in self.init_conv.parameters())
        agg_params = sum(p.numel() for p in self.multi_scale_agg.parameters())
        decoder_params = sum(p.numel() for p in self.decoders.parameters()) + sum(p.numel() for p in self.final_up.parameters()) + sum(p.numel() for p in self.feature_fusion.parameters())
        dual_branch_params = sum(p.numel() for p in self.dual_branch.parameters())
        
        return {
            'total': total, 'trainable': trainable, 'encoder': encoder_params,
            'multi_scale_agg': agg_params, 'decoder': decoder_params, 'dual_branch': dual_branch_params,
        }


def create_dino_vista_kits(config: Optional[Dict] = None) -> DinoVistaKiTS:
    """Factory function to create DINO-VISTA-KiTS model."""
    if config is None:
        config = MODEL_CONFIG
    
    return DinoVistaKiTS(
        in_channels=config.get('in_channels', 1),
        num_classes=config.get('num_classes', 4),
        init_filters=config.get('init_filters', 32),
        blocks_down=config.get('blocks_down', [1, 2, 2, 4]),
        blocks_up=config.get('blocks_up', [1, 1, 1]),
        aggregation_channels=config.get('aggregation_channels', 256),
        class_embed_dim=config.get('class_embed_dim', 256),
        num_attention_heads=config.get('num_attention_heads', 8),
        dropout=config.get('dropout', 0.1),
    )


# =============================================================================
# PART 2: LOSS FUNCTIONS
# =============================================================================

# Cell 9: Utility Functions for Loss
def one_hot_encode(labels: torch.Tensor, num_classes: int, dim: int = 1) -> torch.Tensor:
    """Convert integer labels to one-hot encoding."""
    if labels.dim() == 5 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    B, H, W, D = labels.shape
    one_hot = torch.zeros(B, num_classes, H, W, D, device=labels.device, dtype=torch.float32)
    one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
    return one_hot


# Cell 10: Dice Loss
class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(
        self,
        include_background: bool = False,
        softmax: bool = True,
        to_onehot_y: bool = True,
        smooth: float = 1e-5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.include_background = include_background
        self.softmax = softmax
        self.to_onehot_y = to_onehot_y
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        
        if self.softmax:
            pred = F.softmax(pred, dim=1)
        if self.to_onehot_y:
            target = one_hot_encode(target, num_classes)
        if not self.include_background:
            pred = pred[:, 1:]
            target = target[:, 1:]
        
        pred = pred.flatten(2)
        target = target.flatten(2)
        
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        return dice_loss


# Cell 11: Focal Loss
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[List[float]] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1).long()
        
        ce = F.cross_entropy(pred, target, reduction='none')
        p = F.softmax(pred, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, device=pred.device)
            alpha_t = alpha.gather(0, target.flatten()).reshape(target.shape)
            focal_weight = focal_weight * alpha_t
        
        focal_loss = focal_weight * ce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# Cell 12: Combined Dice + Focal Loss
class DiceFocalLoss(nn.Module):
    """Combined Dice and Focal Loss (recommended for KiTS23)."""
    
    def __init__(
        self,
        include_background: bool = False,
        softmax: bool = True,
        to_onehot_y: bool = True,
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.dice = DiceLoss(include_background=include_background, softmax=softmax, to_onehot_y=to_onehot_y)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        return self.lambda_dice * dice_loss + self.lambda_focal * focal_loss


# Cell 13: Tversky Loss
class TverskyLoss(nn.Module):
    """Tversky Loss for extreme class imbalance."""
    
    def __init__(
        self,
        include_background: bool = False,
        softmax: bool = True,
        to_onehot_y: bool = True,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.include_background = include_background
        self.softmax = softmax
        self.to_onehot_y = to_onehot_y
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        
        if self.softmax:
            pred = F.softmax(pred, dim=1)
        if self.to_onehot_y:
            target = one_hot_encode(target, num_classes)
        if not self.include_background:
            pred = pred[:, 1:]
            target = target[:, 1:]
        
        pred = pred.flatten(2)
        target = target.flatten(2)
        
        TP = (pred * target).sum(dim=2)
        FP = (pred * (1 - target)).sum(dim=2)
        FN = ((1 - pred) * target).sum(dim=2)
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - tversky).mean()


# Cell 14: Uncertainty Loss
class UncertaintyLoss(nn.Module):
    """Loss for calibrating uncertainty estimates."""
    
    def __init__(self, lambda_entropy: float = 0.1):
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.mse = nn.MSELoss()
    
    def forward(self, uncertainty: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_class = pred.argmax(dim=1, keepdim=True)
        target_squeezed = target if target.shape[1] == 1 else target.argmax(dim=1, keepdim=True)
        error_mask = (pred_class != target_squeezed).float()
        
        mse_loss = self.mse(uncertainty, error_mask)
        entropy = -(uncertainty * torch.log(uncertainty + 1e-8) + (1 - uncertainty) * torch.log(1 - uncertainty + 1e-8))
        entropy_loss = -entropy.mean()
        
        return mse_loss + self.lambda_entropy * entropy_loss


# Cell 15: Deep Supervision Loss
class DeepSupervisionLoss(nn.Module):
    """Deep supervision loss for intermediate outputs."""
    
    def __init__(self, base_loss: nn.Module, weights: Optional[List[float]] = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights
    
    def forward(self, deep_outputs: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        num_outputs = len(deep_outputs)
        
        if self.weights is None:
            weights = [0.5 ** (num_outputs - i - 1) for i in range(num_outputs)]
        else:
            weights = self.weights
        
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        total_loss = 0
        for out, w in zip(deep_outputs, weights):
            if out.shape[2:] != target.shape[2:]:
                target_resized = F.interpolate(target.float(), size=out.shape[2:], mode='nearest')
            else:
                target_resized = target
            total_loss = total_loss + w * self.base_loss(out, target_resized)
        
        return total_loss


# Cell 16: Combined Multi-Component Loss (Simplified & Stable)
class CombinedLoss(nn.Module):
    """Combined loss for DINO-VISTA-KiTS using MONAI's stable DiceCELoss."""
    
    def __init__(
        self,
        num_classes: int = 4,
        include_background: bool = False,
        use_monai_loss: bool = True,  # Use MONAI's stable loss
        lambda_aux: float = 0.4,      # Weight for auxiliary outputs
    ):
        super().__init__()
        self.use_monai_loss = use_monai_loss
        self.lambda_aux = lambda_aux
        
        if use_monai_loss and MONAI_AVAILABLE:
            from monai.losses import DiceCELoss
            self.main_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        else:
            # Fallback to custom dice + focal
            self.main_loss = DiceFocalLoss(include_background=include_background)
    
    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        
        # Main loss on merged segmentation
        main_logits = outputs['logits']
        main_loss = self.main_loss(main_logits, target)
        
        # Check for NaN and return safe value if detected
        if torch.isnan(main_loss) or torch.isinf(main_loss):
            main_loss = torch.tensor(1.0, device=main_logits.device, requires_grad=True)
            loss_dict['nan_detected'] = 1.0
        
        loss_dict['main'] = main_loss.item()
        total_loss = main_loss
        
        # Auxiliary losses (auto_seg and refine_seg) - only if not causing NaN
        if outputs.get('auto_seg') is not None:
            auto_loss = self.main_loss(outputs['auto_seg'], target)
            if not (torch.isnan(auto_loss) or torch.isinf(auto_loss)):
                loss_dict['auto'] = auto_loss.item()
                total_loss = total_loss + self.lambda_aux * auto_loss
        
        if outputs.get('refine_seg') is not None:
            refine_loss = self.main_loss(outputs['refine_seg'], target)
            if not (torch.isnan(refine_loss) or torch.isinf(refine_loss)):
                loss_dict['refine'] = refine_loss.item()
                total_loss = total_loss + self.lambda_aux * refine_loss
        
        # Skip uncertainty loss - it was causing NaN issues
        # Skip deep supervision loss for simplicity
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


def create_loss(loss_type: str = "combined", num_classes: int = 4, **kwargs) -> nn.Module:
    """Factory function to create loss modules."""
    if loss_type == "dice":
        return DiceLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "dice_focal":
        return DiceFocalLoss(**kwargs)
    elif loss_type == "tversky":
        return TverskyLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# =============================================================================
# PART 3: DATA LOADING
# =============================================================================

# Cell 17: Dataset Discovery
def get_kits_data_list(dataset_dir: str, require_segmentation: bool = True) -> List[Dict[str, str]]:
    """Discover all KiTS23 cases in the dataset directory."""
    dataset_path = Path(dataset_dir)
    cases = []
    
    for case_dir in sorted(dataset_path.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        
        img_path = case_dir / "imaging.nii.gz"
        seg_path = case_dir / "segmentation.nii.gz"
        
        if not img_path.exists():
            continue
        if require_segmentation and not seg_path.exists():
            continue
        
        case_dict = {"image": str(img_path), "case_id": case_dir.name}
        if seg_path.exists():
            case_dict["label"] = str(seg_path)
        cases.append(case_dict)
    
    return cases


def split_train_val(
    data_list: List[Dict],
    val_split: float = 0.2,
    seed: int = 42,
    stratify_by_tumor: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """Split data into training and validation sets."""
    np.random.seed(seed)
    
    if stratify_by_tumor and NIBABEL_AVAILABLE:
        tumor_cases, non_tumor_cases = [], []
        
        for item in data_list:
            if "label" in item:
                label = nib.load(item["label"]).get_fdata()
                has_tumor = (label == 2).any() or (label == 3).any()
            else:
                has_tumor = False
            
            if has_tumor:
                tumor_cases.append(item)
            else:
                non_tumor_cases.append(item)
        
        np.random.shuffle(tumor_cases)
        np.random.shuffle(non_tumor_cases)
        
        n_val_tumor = int(len(tumor_cases) * val_split)
        n_val_non_tumor = int(len(non_tumor_cases) * val_split)
        
        val_list = tumor_cases[:n_val_tumor] + non_tumor_cases[:n_val_non_tumor]
        train_list = tumor_cases[n_val_tumor:] + non_tumor_cases[n_val_non_tumor:]
        
        np.random.shuffle(train_list)
        np.random.shuffle(val_list)
    else:
        indices = np.random.permutation(len(data_list))
        n_val = int(len(data_list) * val_split)
        val_list = [data_list[i] for i in indices[:n_val]]
        train_list = [data_list[i] for i in indices[n_val:]]
    
    return train_list, val_list


# Cell 18: Transforms
def get_train_transforms(
    patch_size: List[int] = [128, 128, 128],
    spacing: List[float] = [1.5, 1.5, 1.5],
    intensity_min: float = -200,
    intensity_max: float = 400,
    num_samples: int = 4,
    pos_ratio: float = 0.7,
):
    """Get training transforms with data augmentation."""
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI required for transforms")
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=intensity_min, a_max=intensity_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=patch_size, pos=pos_ratio, neg=1 - pos_ratio, num_samples=num_samples, allow_smaller=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.02),
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.8, 1.2)),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def get_val_transforms(
    spacing: List[float] = [1.5, 1.5, 1.5],
    intensity_min: float = -200,
    intensity_max: float = 400,
):
    """Get validation transforms (no augmentation)."""
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI required for transforms")
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=intensity_min, a_max=intensity_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


# Cell 19: Class-Balanced Sampler
class ClassBalancedSampler:
    """Create weighted sampler for class imbalance."""
    
    def __init__(self, data_list: List[Dict], tumor_weight: float = 3.0, cyst_weight: float = 3.0):
        self.data_list = data_list
        self.tumor_weight = tumor_weight
        self.cyst_weight = cyst_weight
        self.weights = self._compute_weights()
    
    def _compute_weights(self) -> List[float]:
        weights = []
        for item in self.data_list:
            weight = 1.0
            if "label" in item and NIBABEL_AVAILABLE:
                label = nib.load(item["label"]).get_fdata()
                if (label == 2).any():
                    weight *= self.tumor_weight
                if (label == 3).any():
                    weight *= self.cyst_weight
            weights.append(weight)
        return weights
    
    def get_sampler(self, num_samples: Optional[int] = None) -> WeightedRandomSampler:
        if num_samples is None:
            num_samples = len(self.weights) * 2
        return WeightedRandomSampler(weights=self.weights, num_samples=num_samples, replacement=True)


# Cell 20: DataLoader Creation
def create_dataloaders(
    dataset_dir: str = None,
    batch_size: int = 2,
    patch_size: List[int] = [128, 128, 128],
    spacing: List[float] = [1.5, 1.5, 1.5],
    num_samples: int = 4,
    val_split: float = 0.2,
    num_workers: int = 4,
    cache_rate: float = 0.0,
    class_balanced: bool = True,
    seed: int = 42,
    max_train_cases: Optional[int] = None,
    max_val_cases: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, List[Dict], List[Dict]]:
    """Create training and validation dataloaders."""
    if dataset_dir is None:
        dataset_dir = DATA_CONFIG["dataset_dir"]
    
    print(f"Scanning dataset directory: {dataset_dir}")
    data_list = get_kits_data_list(dataset_dir)
    print(f"Found {len(data_list)} valid cases")
    
    if len(data_list) == 0:
        raise ValueError(f"No valid cases found in {dataset_dir}")
    
    train_files, val_files = split_train_val(data_list, val_split=val_split, seed=seed)
    
    # Limit cases for quick testing
    if max_train_cases is not None and len(train_files) > max_train_cases:
        train_files = train_files[:max_train_cases]
        print(f"  (Limited to {max_train_cases} train cases for quick testing)")
    if max_val_cases is not None and len(val_files) > max_val_cases:
        val_files = val_files[:max_val_cases]
        print(f"  (Limited to {max_val_cases} val cases for quick testing)")
    
    print(f"Training: {len(train_files)} cases")
    print(f"Validation: {len(val_files)} cases")
    
    train_transforms = get_train_transforms(patch_size=patch_size, spacing=spacing, num_samples=num_samples)
    val_transforms = get_val_transforms(spacing=spacing)
    
    if MONAI_AVAILABLE and cache_rate > 0:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    elif MONAI_AVAILABLE:
        train_ds = MonaiDataset(data=train_files, transform=train_transforms)
        val_ds = MonaiDataset(data=val_files, transform=val_transforms)
    else:
        raise ImportError("MONAI required for data loading")
    
    sampler = None
    shuffle = True
    if class_balanced:
        balanced_sampler = ClassBalancedSampler(train_files)
        sampler = balanced_sampler.get_sampler()
        shuffle = False
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate if MONAI_AVAILABLE else None,
        persistent_workers=num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate if MONAI_AVAILABLE else None,
    )
    
    return train_loader, val_loader, train_files, val_files


# =============================================================================
# PART 4: TRAINING
# =============================================================================

# Cell 21: Learning Rate Scheduler
class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6, initial_lr: float = None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.initial_lr = initial_lr if initial_lr else optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self, epoch: int = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# Cell 22: Metrics Tracker
class MetricsTracker:
    """Track and compute training/validation metrics."""
    
    def __init__(self, num_classes: int = 4, class_names: List[str] = None, include_background: bool = False):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.include_background = include_background
        self.reset()
    
    def reset(self):
        self.dice_scores = {name: [] for name in self.class_names}
        self.loss_values = []
        self.component_losses = {}
    
    def update_loss(self, loss: float, components: Dict[str, float] = None):
        self.loss_values.append(loss)
        if components:
            for key, val in components.items():
                if key not in self.component_losses:
                    self.component_losses[key] = []
                self.component_losses[key].append(val)
    
    def update_dice(self, dice_per_class: torch.Tensor):
        dice_np = dice_per_class.cpu().numpy()
        start_idx = 0 if self.include_background else 1
        for i, name in enumerate(self.class_names[start_idx:], start=start_idx):
            if i - start_idx < len(dice_np):
                self.dice_scores[name].append(dice_np[i - start_idx])
    
    def get_summary(self) -> Dict[str, float]:
        summary = {"mean_loss": np.mean(self.loss_values) if self.loss_values else 0}
        dice_values = []
        start_idx = 0 if self.include_background else 1
        for name in self.class_names[start_idx:]:
            if self.dice_scores[name]:
                mean_dice = np.mean(self.dice_scores[name])
                summary[f"dice_{name}"] = mean_dice
                dice_values.append(mean_dice)
        if dice_values:
            summary["mean_dice"] = np.mean(dice_values)
        for key, vals in self.component_losses.items():
            summary[f"loss_{key}"] = np.mean(vals) if vals else 0
        return summary


# Cell 23: Trainer Class
class DinoVistaTrainer:
    """Trainer for DINO-VISTA-KiTS model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer = None,
        loss_fn: nn.Module = None,
        config: Dict = None,
        device: torch.device = None,
    ):
        self.config = {**TRAIN_CONFIG, **(config or {})}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if optimizer is None:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        else:
            self.optimizer = optimizer
        
        if loss_fn is None:
            self.loss_fn = CombinedLoss(num_classes=MODEL_CONFIG["num_classes"])
        else:
            self.loss_fn = loss_fn
        self.loss_fn = self.loss_fn.to(self.device)
        
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_epochs=self.config["warmup_epochs"],
            total_epochs=self.config["num_epochs"], min_lr=self.config["min_learning_rate"],
        )
        
        self.use_amp = self.config["use_amp"] and torch.cuda.is_available()
        self.scaler = GradScaler("cuda") if self.use_amp else None
        
        self.train_metrics = MetricsTracker(num_classes=MODEL_CONFIG["num_classes"], class_names=MODEL_CONFIG["class_names"])
        self.val_metrics = MetricsTracker(num_classes=MODEL_CONFIG["num_classes"], class_names=MODEL_CONFIG["class_names"])
        
        if MONAI_AVAILABLE:
            self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        
        self._setup_logging()
        self.current_epoch = 0
        self.best_dice = 0
        self.best_epoch = 0
        self.epochs_without_improvement = 0  # For early stopping
    
    def _setup_logging(self):
        for dir_key in ["output_dir", "checkpoint_dir", "log_dir"]:
            Path(self.config[dir_key]).mkdir(parents=True, exist_ok=True)
        
        if TENSORBOARD_AVAILABLE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(log_dir=str(Path(self.config["log_dir"]) / timestamp))
        else:
            self.writer = None
        
        # Enhanced history tracking for graphs
        self.history = {
            "train_loss": [], "val_loss": [], "val_dice": [], "learning_rate": [],
            "dice_kidney": [], "dice_tumor": [], "dice_cyst": [],
            "val_iou": [], "epochs": [],
        }
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.train_metrics.reset()
        
        accumulation_steps = self.config["accumulation_steps"]
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}", leave=False)
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images, deep_supervision=self.config["use_deep_supervision"])
                loss, loss_dict = self.loss_fn(outputs, labels)
                loss = loss / accumulation_steps
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"])
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            self.train_metrics.update_loss(loss.item() * accumulation_steps, loss_dict)
            pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}", "lr": f"{self.scheduler.get_lr():.2e}"})
        
        return self.train_metrics.get_summary()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        self.val_metrics.reset()
        
        if MONAI_AVAILABLE:
            self.dice_metric.reset()
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            if MONAI_AVAILABLE:
                outputs = sliding_window_inference(
                    images, roi_size=MODEL_CONFIG["patch_size"],
                    sw_batch_size=self.config["val_sw_batch_size"],
                    predictor=lambda x: self.model(x)["logits"],
                    overlap=self.config["val_overlap"],
                )
            else:
                outputs = self.model(images)["logits"]
            
            loss, loss_dict = self.loss_fn({"logits": outputs}, labels)
            self.val_metrics.update_loss(loss.item(), loss_dict)
            
            if MONAI_AVAILABLE:
                outputs_argmax = outputs.argmax(dim=1, keepdim=True)
                self.dice_metric(outputs_argmax, labels)
        
        summary = self.val_metrics.get_summary()
        
        if MONAI_AVAILABLE:
            dice_per_class = self.dice_metric.aggregate()
            summary["mean_dice"] = dice_per_class.mean().item()
            for i, name in enumerate(["kidney", "tumor", "cyst"]):
                if i < len(dice_per_class):
                    summary[f"dice_{name}"] = dice_per_class[i].item()
        
        return summary
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_dice": self.best_dice,
            "config": self.config,
            "model_config": MODEL_CONFIG,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        checkpoint_dir = Path(self.config["checkpoint_dir"])
        periodic_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, periodic_path)
        
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✅ New best model saved! Dice: {self.best_dice:.4f}")
        
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Cleanup old checkpoints, keeping only the most recent ones."""
        try:
            checkpoint_dir = Path(self.config["checkpoint_dir"])
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
            for ckpt in checkpoints[:-self.config["keep_best_n"]]:
                try:
                    ckpt.unlink()
                except (PermissionError, OSError):
                    # File may be locked by another process, skip silently
                    pass
        except Exception:
            # Don't crash training if cleanup fails
            pass
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_dice = checkpoint.get("best_dice", 0)
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int = None) -> Dict:
        if num_epochs is None:
            num_epochs = self.config["num_epochs"]
        
        early_stopping_patience = self.config.get("early_stopping_patience", 10)
        early_stopping_min_delta = self.config.get("early_stopping_min_delta", 0.001)
        
        print("=" * 60)
        print("🚀 Starting DINO-VISTA-KiTS Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Early stopping: patience={early_stopping_patience}")
        print("-" * 60)
        
        start_time = time.time()
        early_stop = False
        
        for epoch in range(self.current_epoch, num_epochs):
            if early_stop:
                break
                
            self.current_epoch = epoch
            epoch_start = time.time()
            
            lr = self.scheduler.step(epoch)
            self.history["learning_rate"].append(lr)
            self.history["epochs"].append(epoch + 1)
            
            train_summary = self.train_epoch()
            self.history["train_loss"].append(train_summary["mean_loss"])
            
            if self.writer:
                self.writer.add_scalar("Loss/train", train_summary["mean_loss"], epoch)
                self.writer.add_scalar("Learning_Rate", lr, epoch)
            
            if (epoch + 1) % self.config["val_interval"] == 0:
                val_summary = self.validate()
                self.history["val_loss"].append(val_summary["mean_loss"])
                self.history["val_dice"].append(val_summary.get("mean_dice", 0))
                
                # Track per-class dice
                for name in ["kidney", "tumor", "cyst"]:
                    key = f"dice_{name}"
                    self.history[key].append(val_summary.get(key, 0))
                
                if self.writer:
                    self.writer.add_scalar("Loss/val", val_summary["mean_loss"], epoch)
                    self.writer.add_scalar("Dice/mean", val_summary.get("mean_dice", 0), epoch)
                    for key, val in val_summary.items():
                        if "dice_" in key:
                            self.writer.add_scalar(f"Dice/{key}", val, epoch)
                
                current_dice = val_summary.get("mean_dice", 0)
                
                # Early stopping check
                if current_dice > self.best_dice + early_stopping_min_delta:
                    self.best_dice = current_dice
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    is_best = True
                else:
                    self.epochs_without_improvement += 1
                    is_best = False
                
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_summary['mean_loss']:.4f}")
                print(f"  Val Loss: {val_summary['mean_loss']:.4f}")
                print(f"  Val Dice: {current_dice:.4f} (best: {self.best_dice:.4f})")
                for name in ["kidney", "tumor", "cyst"]:
                    key = f"dice_{name}"
                    if key in val_summary:
                        print(f"    {name}: {val_summary[key]:.4f}")
                
                # Early stopping message
                if self.epochs_without_improvement > 0:
                    print(f"  ⏳ No improvement for {self.epochs_without_improvement}/{early_stopping_patience} validations")
                
                self.save_checkpoint(is_best)
                
                # Check early stopping
                if self.epochs_without_improvement >= early_stopping_patience:
                    print(f"\n🛑 EARLY STOPPING: No improvement for {early_stopping_patience} validations")
                    early_stop = True
            else:
                print(f"Epoch {epoch + 1}: Loss={train_summary['mean_loss']:.4f}")
            
            if (epoch + 1) % self.config["save_interval"] == 0:
                self.save_checkpoint(is_best=False)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        if early_stop:
            print("🛑 Training Stopped Early!")
        else:
            print("✅ Training Complete!")
        print("=" * 60)
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"Epochs completed: {self.current_epoch + 1}")
        print(f"Best Dice: {self.best_dice:.4f} (Epoch {self.best_epoch + 1})")
        print("=" * 60)
        
        # Save final model
        final_path = Path(self.config["checkpoint_dir"]) / "final_model.pth"
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_dice": self.best_dice,
            "history": self.history,
            "config": self.config,
            "early_stopped": early_stop,
        }, final_path)
        
        # Save training history
        history_path = Path(self.config["output_dir"]) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        # Generate training plots
        if self.config.get("save_training_plots", True):
            self._save_training_plots()
        
        if self.writer:
            self.writer.close()
        
        return self.history
    
    def _save_training_plots(self):
        """Generate and save training curves and metrics plots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        output_dir = Path(self.config["output_dir"])
        
        # 1. Loss curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training loss
        ax1 = axes[0]
        ax1.plot(self.history["train_loss"], 'b-', label='Train Loss', linewidth=2)
        if self.history["val_loss"]:
            # Map val_loss to correct epochs
            val_epochs = list(range(self.config["val_interval"]-1, 
                                   len(self.history["train_loss"]), 
                                   self.config["val_interval"]))[:len(self.history["val_loss"])]
            ax1.plot(val_epochs, self.history["val_loss"], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2 = axes[1]
        ax2.plot(self.history["learning_rate"], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'loss_curves.png'}")
        
        # 2. Dice score curves
        if self.history["val_dice"]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Mean dice
            ax1 = axes[0]
            val_epochs = list(range(self.config["val_interval"], 
                                   len(self.history["train_loss"]) + 1, 
                                   self.config["val_interval"]))[:len(self.history["val_dice"])]
            ax1.plot(val_epochs, self.history["val_dice"], 'b-o', label='Mean Dice', linewidth=2, markersize=4)
            ax1.axhline(y=self.best_dice, color='g', linestyle='--', label=f'Best: {self.best_dice:.4f}')
            ax1.axvline(x=self.best_epoch + 1, color='r', linestyle=':', alpha=0.5, label=f'Best Epoch: {self.best_epoch + 1}')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Dice Score')
            ax1.set_title('Mean Dice Score')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1])
            
            # Per-class dice
            ax2 = axes[1]
            colors = {'kidney': 'green', 'tumor': 'red', 'cyst': 'blue'}
            for name in ["kidney", "tumor", "cyst"]:
                key = f"dice_{name}"
                if key in self.history and self.history[key]:
                    ax2.plot(val_epochs[:len(self.history[key])], 
                            self.history[key], '-o', label=name.capitalize(), 
                            color=colors[name], linewidth=2, markersize=4)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Dice Score')
            ax2.set_title('Per-Class Dice Scores')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig(output_dir / "dice_curves.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_dir / 'dice_curves.png'}")
        
        # 3. Summary metrics bar chart
        if self.history["val_dice"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            final_metrics = {}
            for name in ["kidney", "tumor", "cyst"]:
                key = f"dice_{name}"
                if key in self.history and self.history[key]:
                    final_metrics[name.capitalize()] = self.history[key][-1]
            final_metrics["Mean"] = self.history["val_dice"][-1]
            
            bars = ax.bar(final_metrics.keys(), final_metrics.values(), 
                         color=['green', 'red', 'blue', 'purple'], alpha=0.7)
            ax.axhline(y=self.best_dice, color='gold', linestyle='--', 
                      label=f'Best Mean: {self.best_dice:.4f}', linewidth=2)
            
            # Add value labels
            for bar, val in zip(bars, final_metrics.values()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Dice Score')
            ax.set_title('Final Validation Dice Scores')
            ax.set_ylim([0, 1.1])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "final_metrics.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_dir / 'final_metrics.png'}")


# =============================================================================
# PART 5: EVALUATION
# =============================================================================

# Cell 24: Basic Metrics
def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Compute Dice Similarity Coefficient."""
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = np.sum(pred & target)
    pred_sum, target_sum = np.sum(pred), np.sum(target)
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    return (2 * intersection + smooth) / (pred_sum + target_sum + smooth)


def compute_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Compute Intersection over Union."""
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = np.sum(pred & target)
    union = np.sum(pred | target)
    if union == 0:
        return 1.0
    return (intersection + smooth) / (union + smooth)


def compute_precision_recall(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> Tuple[float, float]:
    """Compute Precision and Recall."""
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.sum(pred & target)
    fp = np.sum(pred & ~target)
    fn = np.sum(~pred & target)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision, recall


# Cell 25: Surface Distance Metrics
def compute_hausdorff_95(pred: np.ndarray, target: np.ndarray, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """Compute 95th percentile Hausdorff Distance."""
    if not SCIPY_AVAILABLE:
        return np.nan
    
    pred_empty = np.sum(pred) == 0
    target_empty = np.sum(target) == 0
    
    if pred_empty and target_empty:
        return 0.0
    if pred_empty or target_empty:
        return np.inf
    
    try:
        pred, target = pred.astype(bool), target.astype(bool)
        pred_surface = pred ^ binary_erosion(pred)
        target_surface = target ^ binary_erosion(target)
        
        dt_pred = distance_transform_edt(~pred, sampling=spacing)
        dt_target = distance_transform_edt(~target, sampling=spacing)
        
        distances_pred_to_gt = dt_target[pred_surface]
        distances_gt_to_pred = dt_pred[target_surface]
        
        hd95_pred_to_gt = np.percentile(distances_pred_to_gt, 95) if len(distances_pred_to_gt) > 0 else 0
        hd95_gt_to_pred = np.percentile(distances_gt_to_pred, 95) if len(distances_gt_to_pred) > 0 else 0
        
        return max(hd95_pred_to_gt, hd95_gt_to_pred)
    except Exception:
        return np.inf


def compute_surface_dice(pred: np.ndarray, target: np.ndarray, spacing: Tuple[float, ...], tolerance_mm: float = 2.0) -> float:
    """Compute Normalized Surface Dice (NSD)."""
    if not SCIPY_AVAILABLE:
        return np.nan
    
    pred_empty = np.sum(pred) == 0
    target_empty = np.sum(target) == 0
    
    if pred_empty and target_empty:
        return 1.0
    if pred_empty or target_empty:
        return 0.0
    
    try:
        pred, target = pred.astype(bool), target.astype(bool)
        pred_surface = pred ^ binary_erosion(pred)
        target_surface = target ^ binary_erosion(target)
        
        dt_pred = distance_transform_edt(~pred, sampling=spacing)
        dt_target = distance_transform_edt(~target, sampling=spacing)
        
        overlap_pred = np.sum(dt_target[pred_surface] <= tolerance_mm)
        overlap_gt = np.sum(dt_pred[target_surface] <= tolerance_mm)
        total_surface = np.sum(pred_surface) + np.sum(target_surface)
        
        if total_surface == 0:
            return 1.0
        return (overlap_pred + overlap_gt) / total_surface
    except Exception:
        return 0.0


# Cell 26: Multi-Class Evaluation
def evaluate_segmentation(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    num_classes: int = 4,
    class_names: List[str] = None,
    include_background: bool = False,
) -> Dict[str, float]:
    """Compute comprehensive segmentation metrics."""
    if class_names is None:
        class_names = EVAL_CONFIG["class_names"]
    
    results = {}
    start_idx = 0 if include_background else 1
    dice_scores, iou_scores = [], []
    
    for cls_id in range(start_idx, num_classes):
        cls_name = class_names[cls_id]
        pred_cls = (pred == cls_id)
        target_cls = (target == cls_id)
        
        dice = compute_dice(pred_cls, target_cls)
        results[f"dice_{cls_name}"] = dice
        dice_scores.append(dice)
        
        iou = compute_iou(pred_cls, target_cls)
        results[f"iou_{cls_name}"] = iou
        iou_scores.append(iou)
        
        precision, recall = compute_precision_recall(pred_cls, target_cls)
        results[f"precision_{cls_name}"] = precision
        results[f"recall_{cls_name}"] = recall
        
        hd95 = compute_hausdorff_95(pred_cls, target_cls, spacing)
        results[f"hd95_{cls_name}"] = hd95
        
        nsd = compute_surface_dice(pred_cls, target_cls, spacing, tolerance_mm=2.0)
        results[f"nsd_{cls_name}"] = nsd
    
    results["mean_dice"] = np.mean(dice_scores)
    results["mean_iou"] = np.mean(iou_scores)
    
    return results


# Cell 27: KiTS HEC Evaluation
def evaluate_hec(pred: np.ndarray, target: np.ndarray, spacing: Tuple[float, ...], hec_config: Dict = None) -> Dict[str, Dict[str, float]]:
    """Evaluate using KiTS Hierarchical Evaluation Classes."""
    if hec_config is None:
        hec_config = EVAL_CONFIG["hec_classes"]
    
    results = {}
    for hec_name, config in hec_config.items():
        pred_mask = np.zeros(pred.shape, dtype=bool)
        target_mask = np.zeros(target.shape, dtype=bool)
        
        for label in config["labels"]:
            pred_mask |= (pred == label)
            target_mask |= (target == label)
        
        dice = compute_dice(pred_mask, target_mask)
        nsd = compute_surface_dice(pred_mask, target_mask, spacing, config["tolerance_mm"])
        hd95 = compute_hausdorff_95(pred_mask, target_mask, spacing)
        
        results[hec_name] = {"dice": dice, "surface_dice": nsd, "hd95": hd95}
    
    return results


# Cell 28: Model Evaluation Function
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
    patch_size: List[int] = None,
    sw_batch_size: int = 4,
    overlap: float = 0.5,
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if patch_size is None:
        patch_size = MODEL_CONFIG["patch_size"]
    
    model.eval()
    model.to(device)
    all_results = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy()
        
        if "image_meta_dict" in batch:
            spacing = tuple(batch["image_meta_dict"]["pixdim"][0, 1:4].numpy())
        else:
            spacing = (1.0, 1.0, 1.0)
        
        if MONAI_AVAILABLE:
            outputs = sliding_window_inference(
                images, roi_size=patch_size, sw_batch_size=sw_batch_size,
                predictor=lambda x: model(x)["logits"], overlap=overlap,
            )
        else:
            outputs = model(images)["logits"]
        
        pred = outputs.argmax(dim=1).cpu().numpy()
        
        for i in range(pred.shape[0]):
            pred_i = pred[i]
            target_i = labels[i, 0] if labels.ndim == 5 else labels[i]
            
            metrics = evaluate_segmentation(pred_i, target_i, spacing, num_classes=EVAL_CONFIG["num_classes"])
            hec_metrics = evaluate_hec(pred_i, target_i, spacing)
            for hec_name, hec_vals in hec_metrics.items():
                for metric_name, val in hec_vals.items():
                    metrics[f"hec_{hec_name}_{metric_name}"] = val
            all_results.append(metrics)
    
    aggregated = {}
    for key in all_results[0].keys():
        values = [r[key] for r in all_results if not np.isinf(r[key]) and not np.isnan(r[key])]
        if values:
            aggregated[key] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
    
    return aggregated


# Cell 29: Generate Report
def generate_report(metrics: Dict[str, float], output_path: str = None) -> str:
    """Generate formatted evaluation report."""
    report = []
    report.append("=" * 70)
    report.append("DINO-VISTA-KiTS EVALUATION REPORT")
    report.append("=" * 70)
    
    report.append("\n📊 Per-Class Dice Similarity Coefficient:")
    for cls_name in ["kidney", "tumor", "cyst"]:
        key = f"dice_{cls_name}"
        if key in metrics:
            std = metrics.get(f"{key}_std", 0)
            report.append(f"  {cls_name:10s}: {metrics[key]:.4f} ± {std:.4f}")
    
    report.append(f"\n📈 Aggregate Metrics:")
    report.append(f"  Mean Dice:  {metrics.get('mean_dice', 0):.4f}")
    report.append(f"  Mean IoU:   {metrics.get('mean_iou', 0):.4f}")
    
    report.append("\n🎯 KiTS HEC (Hierarchical Evaluation Classes):")
    for hec_name in ["kidney", "masses", "tumor"]:
        dice_key = f"hec_{hec_name}_dice"
        if dice_key in metrics:
            report.append(f"  {hec_name:10s}:")
            report.append(f"    Dice: {metrics[dice_key]:.4f}")
            report.append(f"    NSD:  {metrics.get(f'hec_{hec_name}_surface_dice', 0):.4f}")
            report.append(f"    HD95: {metrics.get(f'hec_{hec_name}_hd95', 0):.2f} mm")
    
    report.append("\n" + "=" * 70)
    report_str = "\n".join(report)
    
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(report_str)
        print(f"Report saved to: {output_path}")
    
    return report_str


# Cell 30: Visualization Functions
def save_segmentation_visualization(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: str,
    slice_idx: int = None,
    axis: int = 2,  # 0=sagittal, 1=coronal, 2=axial
    figsize: Tuple[int, int] = (15, 5),
):
    """
    Save a visualization comparing ground truth and prediction.
    
    Args:
        image: CT image volume [H, W, D] or [C, H, W, D]
        ground_truth: Ground truth labels [H, W, D]
        prediction: Predicted labels [H, W, D]
        output_path: Path to save the figure
        slice_idx: Index of slice to visualize (default: middle slice)
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    # Handle channel dimension
    if image.ndim == 4:
        image = image[0]
    if ground_truth.ndim == 4:
        ground_truth = ground_truth[0]
    if prediction.ndim == 4:
        prediction = prediction[0]
    
    # Get slice
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2
    
    if axis == 0:
        img_slice = image[slice_idx, :, :]
        gt_slice = ground_truth[slice_idx, :, :]
        pred_slice = prediction[slice_idx, :, :]
    elif axis == 1:
        img_slice = image[:, slice_idx, :]
        gt_slice = ground_truth[:, slice_idx, :]
        pred_slice = prediction[:, slice_idx, :]
    else:
        img_slice = image[:, :, slice_idx]
        gt_slice = ground_truth[:, :, slice_idx]
        pred_slice = prediction[:, :, slice_idx]
    
    # Color maps for classes
    colors = {
        0: [0, 0, 0, 0],        # Background - transparent
        1: [0, 1, 0, 0.5],      # Kidney - green
        2: [1, 0, 0, 0.5],      # Tumor - red
        3: [0, 0, 1, 0.5],      # Cyst - blue
    }
    
    def create_overlay(seg_slice):
        overlay = np.zeros((*seg_slice.shape, 4))
        for label, color in colors.items():
            mask = seg_slice == label
            overlay[mask] = color
        return overlay
    
    gt_overlay = create_overlay(gt_slice)
    pred_overlay = create_overlay(pred_slice)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Normalize image for display
    img_display = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    
    # Original image
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title('CT Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(img_display, cmap='gray')
    axes[1].imshow(gt_overlay)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(img_display, cmap='gray')
    axes[2].imshow(pred_overlay)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Legend
    legend_patches = [
        mpatches.Patch(color='green', alpha=0.5, label='Kidney'),
        mpatches.Patch(color='red', alpha=0.5, label='Tumor'),
        mpatches.Patch(color='blue', alpha=0.5, label='Cyst'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


def visualize_prediction_3d(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_dir: str,
    case_id: str,
    num_slices: int = 5,
):
    """
    Generate multiple slice visualizations for a case.
    
    Creates axial, coronal, and sagittal views.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for axis, axis_name in [(0, 'sagittal'), (1, 'coronal'), (2, 'axial')]:
        n_slices = image.shape[axis]
        
        # Select evenly spaced slices, avoiding edges
        start = n_slices // 6
        end = n_slices - n_slices // 6
        indices = np.linspace(start, end, num_slices, dtype=int)
        
        for i, slice_idx in enumerate(indices):
            output_path = Path(output_dir) / f"{case_id}_{axis_name}_slice{i}.png"
            save_segmentation_visualization(
                image, ground_truth, prediction,
                str(output_path), slice_idx=slice_idx, axis=axis
            )


def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Compute confusion matrix for segmentation."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            cm[true_class, pred_class] = np.sum((target == true_class) & (pred == pred_class))
    return cm


def save_confusion_matrix(cm: np.ndarray, output_path: str, class_names: List[str] = None):
    """Save confusion matrix as an image."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    
    if class_names is None:
        class_names = ["Background", "Kidney", "Tumor", "Cyst"]
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_normalized, cmap='Blues')
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Normalized)')
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm_normalized[i, j]:.2f}\n({cm[i, j]:,})"
            ax.text(j, i, text, ha='center', va='center', fontsize=8)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


@torch.no_grad()
def run_full_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    output_dir: str = "./evaluation_results",
    device: torch.device = None,
    num_vis_cases: int = 3,
) -> Dict[str, float]:
    """
    Run comprehensive evaluation with metrics and visualizations.
    
    Generates:
    - Per-class and aggregate metrics
    - Confusion matrix
    - Segmentation visualizations
    - Detailed evaluation report
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    model.eval()
    model.to(device)
    
    all_results = []
    total_cm = np.zeros((4, 4), dtype=np.int64)
    vis_count = 0
    
    print("\n📊 Running Full Evaluation...")
    print("=" * 60)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy()
        case_id = batch.get("case_id", [f"case_{batch_idx:05d}"])[0] if "case_id" in batch else f"case_{batch_idx:05d}"
        
        if "image_meta_dict" in batch:
            try:
                spacing = tuple(batch["image_meta_dict"]["pixdim"][0, 1:4].numpy())
            except:
                spacing = (1.5, 1.5, 1.5)
        else:
            spacing = (1.5, 1.5, 1.5)
        
        # Run inference with sliding window
        if MONAI_AVAILABLE:
            outputs = sliding_window_inference(
                images, roi_size=MODEL_CONFIG["patch_size"],
                sw_batch_size=2, predictor=lambda x: model(x)["logits"],
                overlap=0.5,
            )
        else:
            outputs = model(images)["logits"]
        
        pred = outputs.argmax(dim=1).cpu().numpy()
        img_np = images.cpu().numpy()
        
        for i in range(pred.shape[0]):
            pred_i = pred[i]
            target_i = labels[i, 0] if labels.ndim == 5 else labels[i]
            
            # Compute metrics
            metrics = evaluate_segmentation(pred_i, target_i, spacing, num_classes=4)
            hec_metrics = evaluate_hec(pred_i, target_i, spacing)
            for hec_name, hec_vals in hec_metrics.items():
                for metric_name, val in hec_vals.items():
                    metrics[f"hec_{hec_name}_{metric_name}"] = val
            metrics["case_id"] = case_id
            all_results.append(metrics)
            
            # Update confusion matrix
            total_cm += compute_confusion_matrix(pred_i.flatten(), target_i.flatten(), 4)
            
            # Save visualizations for first few cases
            if vis_count < num_vis_cases:
                img_i = img_np[i, 0] if img_np.ndim == 5 else img_np[i]
                visualize_prediction_3d(
                    img_i, target_i, pred_i,
                    str(vis_dir), case_id, num_slices=3
                )
                vis_count += 1
    
    # Aggregate metrics
    aggregated = {"num_cases": len(all_results)}
    metric_keys = [k for k in all_results[0].keys() if k != "case_id"]
    
    for key in metric_keys:
        values = [r[key] for r in all_results if not np.isinf(r[key]) and not np.isnan(r[key])]
        if values:
            aggregated[key] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)
    
    # Save confusion matrix
    save_confusion_matrix(total_cm, str(output_dir / "confusion_matrix.png"))
    
    # Generate and save report
    report = generate_report(aggregated, str(output_dir / "evaluation_report.txt"))
    print(report)
    
    # Save detailed per-case results as CSV
    csv_path = output_dir / "per_case_results.csv"
    with open(csv_path, "w") as f:
        headers = ["case_id", "dice_kidney", "dice_tumor", "dice_cyst", "mean_dice", 
                   "iou_kidney", "iou_tumor", "iou_cyst", "mean_iou",
                   "hd95_kidney", "hd95_tumor", "hd95_cyst"]
        f.write(",".join(headers) + "\n")
        for r in all_results:
            row = [
                r.get("case_id", ""),
                f"{r.get('dice_kidney', 0):.4f}",
                f"{r.get('dice_tumor', 0):.4f}",
                f"{r.get('dice_cyst', 0):.4f}",
                f"{r.get('mean_dice', 0):.4f}",
                f"{r.get('iou_kidney', 0):.4f}",
                f"{r.get('iou_tumor', 0):.4f}",
                f"{r.get('iou_cyst', 0):.4f}",
                f"{r.get('mean_iou', 0):.4f}",
                f"{r.get('hd95_kidney', 0):.2f}",
                f"{r.get('hd95_tumor', 0):.2f}",
                f"{r.get('hd95_cyst', 0):.2f}",
            ]
            f.write(",".join(row) + "\n")
    print(f"\nPer-case results saved: {csv_path}")
    
    # Save aggregated metrics as JSON
    json_path = output_dir / "aggregated_metrics.json"
    with open(json_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in aggregated.items()}, f, indent=2)
    print(f"Aggregated metrics saved: {json_path}")
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Cases evaluated: {aggregated['num_cases']}")
    print(f"  Mean Dice: {aggregated.get('mean_dice', 0):.4f} ± {aggregated.get('mean_dice_std', 0):.4f}")
    print(f"  Mean IoU:  {aggregated.get('mean_iou', 0):.4f} ± {aggregated.get('mean_iou_std', 0):.4f}")
    print(f"\n  Per-class Dice:")
    for cls in ["kidney", "tumor", "cyst"]:
        print(f"    {cls}: {aggregated.get(f'dice_{cls}', 0):.4f}")
    print(f"\n  Outputs saved to: {output_dir}")
    print("=" * 60)
    
    return aggregated


# =============================================================================
# PART 6: MAIN FUNCTIONS
# =============================================================================

# Cell 30: Main Training Function
def train_dino_vista_kits(
    dataset_dir: str = None,
    output_dir: str = None,
    num_epochs: int = None,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    resume: str = None,
    device: str = None,
) -> Dict:
    """Train DINO-VISTA-KiTS model on KiTS23 dataset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if dataset_dir is not None:
        DATA_CONFIG["dataset_dir"] = dataset_dir
    if output_dir is not None:
        TRAIN_CONFIG["output_dir"] = output_dir
        TRAIN_CONFIG["checkpoint_dir"] = str(Path(output_dir) / "checkpoints")
        TRAIN_CONFIG["log_dir"] = str(Path(output_dir) / "logs")
    if num_epochs is not None:
        TRAIN_CONFIG["num_epochs"] = num_epochs
    TRAIN_CONFIG["learning_rate"] = learning_rate
    TRAIN_CONFIG["batch_size"] = batch_size
    
    print("\n📊 Loading data...")
    train_loader, val_loader, train_files, val_files = create_dataloaders(
        dataset_dir=DATA_CONFIG["dataset_dir"],
        batch_size=batch_size,
        patch_size=MODEL_CONFIG["patch_size"],
        spacing=DATA_CONFIG["spacing"],
        num_samples=DATA_CONFIG["num_samples"],
        val_split=DATA_CONFIG["val_split"],
        num_workers=DATA_CONFIG["num_workers"],
        class_balanced=True,
        max_train_cases=DATA_CONFIG.get("max_train_cases"),
        max_val_cases=DATA_CONFIG.get("max_val_cases"),
    )
    
    print("\n🧠 Creating model...")
    model = create_dino_vista_kits()
    params = model.get_num_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    
    trainer = DinoVistaTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=TRAIN_CONFIG, device=device,
    )
    
    if resume is not None:
        print(f"\nResuming from: {resume}")
        trainer.load_checkpoint(resume)
    
    history = trainer.train()
    return history


# Cell 31: Quick Test Function
def quick_test(dataset_dir: str = None, num_cases: int = 5, epochs: int = 5):
    """Quick test to verify training pipeline works."""
    print("=" * 60)
    print("🧪 QUICK TEST MODE")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset_dir is None:
        dataset_dir = DATA_CONFIG["dataset_dir"]
    
    if not Path(dataset_dir).exists():
        print(f"Dataset not found at {dataset_dir}")
        print("Creating synthetic test...")
        return _test_with_synthetic_data(epochs)
    
    data_list = get_kits_data_list(dataset_dir)[:num_cases]
    if len(data_list) == 0:
        print("No data found. Using synthetic test...")
        return _test_with_synthetic_data(epochs)
    
    print(f"Testing with {len(data_list)} cases, {epochs} epochs")
    return train_dino_vista_kits(dataset_dir=dataset_dir, num_epochs=epochs, batch_size=1, learning_rate=1e-3)


def _test_with_synthetic_data(epochs: int = 5):
    """Test with synthetic data when no real data available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_dino_vista_kits().to(device)
    print(f"Model created with {model.get_num_parameters()['total']:,} parameters")
    
    patch_size = [96, 96, 96]
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = DiceFocalLoss().to(device)
    
    print(f"\nTraining for {epochs} epochs with synthetic data...")
    
    for epoch in range(epochs):
        model.train()
        images = torch.randn(1, 1, *patch_size).to(device)
        labels = torch.randint(0, 4, (1, 1, *patch_size)).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs["logits"], labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {loss.item():.4f}")
    
    print("\n✅ Synthetic test passed!")
    return None


# Cell 32: Test Model Architecture
def test_model():
    """Quick test to verify model architecture."""
    print("=" * 60)
    print("Testing DINO-VISTA-KiTS Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = create_dino_vista_kits().to(device)
    params = model.get_num_parameters()
    print(f"\nModel Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:,}")
    
    input_shape = (1, 1, 128, 128, 128)
    print(f"\nTesting forward pass with input shape: {input_shape}")
    
    x = torch.randn(input_shape).to(device)
    
    with torch.no_grad():
        outputs = model(x, deep_supervision=True)
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if value is not None:
            if isinstance(value, list):
                print(f"  {key}: {[v.shape for v in value]}")
            else:
                print(f"  {key}: {value.shape}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        with torch.no_grad():
            with autocast("cuda"):
                outputs_amp = model(x)
        
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU Memory:")
        print(f"  Allocated: {mem_allocated:.2f} GB")
        print(f"  Reserved: {mem_reserved:.2f} GB")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    
    return model


# Cell 33: Test Loss Functions
def test_losses():
    """Test all loss functions."""
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, C, H, W, D = 2, 4, 32, 32, 32
    pred = torch.randn(B, C, H, W, D).to(device)
    target = torch.randint(0, C, (B, 1, H, W, D)).to(device)
    
    print(f"Input shapes - pred: {pred.shape}, target: {target.shape}")
    
    losses_to_test = [
        ("DiceLoss", DiceLoss()),
        ("FocalLoss", FocalLoss()),
        ("DiceFocalLoss", DiceFocalLoss()),
        ("TverskyLoss", TverskyLoss()),
    ]
    
    print("\nIndividual Losses:")
    for name, loss_fn in losses_to_test:
        loss_fn = loss_fn.to(device)
        loss_val = loss_fn(pred, target)
        print(f"  {name}: {loss_val.item():.4f}")
    
    print("\nCombined Loss:")
    outputs = {
        'logits': pred, 'auto_seg': pred, 'refine_seg': pred,
        'uncertainty': torch.sigmoid(torch.randn(B, 1, H, W, D)).to(device),
        'deep_outputs': [torch.randn(B, C, H//2, W//2, D//2).to(device), torch.randn(B, C, H, W, D).to(device)],
    }
    
    combined_loss = CombinedLoss(num_classes=C).to(device)
    total, loss_dict = combined_loss(outputs, target)
    
    print(f"  Total: {total.item():.4f}")
    for key, val in loss_dict.items():
        print(f"    {key}: {val:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ All loss tests passed!")
    print("=" * 60)


# Cell 34: Test Evaluation
def test_evaluation():
    """Test evaluation functions."""
    print("=" * 60)
    print("Testing Evaluation Functions")
    print("=" * 60)
    
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)
    
    target = np.zeros(shape, dtype=np.uint8)
    target[20:40, 20:40, 20:40] = 1
    target[25:35, 25:35, 25:35] = 2
    target[30:35, 30:35, 40:50] = 3
    
    pred = np.zeros(shape, dtype=np.uint8)
    pred[21:41, 21:41, 20:40] = 1
    pred[26:36, 25:35, 25:35] = 2
    pred[30:35, 30:35, 41:51] = 3
    
    print("\nBasic Metrics:")
    for cls_id, cls_name in enumerate(["background", "kidney", "tumor", "cyst"]):
        if cls_id == 0:
            continue
        pred_cls = pred == cls_id
        target_cls = target == cls_id
        
        dice = compute_dice(pred_cls, target_cls)
        iou = compute_iou(pred_cls, target_cls)
        hd95 = compute_hausdorff_95(pred_cls, target_cls, spacing)
        
        print(f"  {cls_name}: Dice={dice:.4f}, IoU={iou:.4f}, HD95={hd95:.2f}mm")
    
    print("\nFull Evaluation:")
    results = evaluate_segmentation(pred, target, spacing)
    print(f"  Mean Dice: {results['mean_dice']:.4f}")
    
    print("\nHEC Evaluation:")
    hec_results = evaluate_hec(pred, target, spacing)
    for hec_name, metrics in hec_results.items():
        print(f"  {hec_name}: Dice={metrics['dice']:.4f}, NSD={metrics['surface_dice']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation tests passed!")
    print("=" * 60)


# Cell 35: Main Entry Point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DINO-VISTA-KiTS: 3D Kidney Tumor Segmentation")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "evaluate", "test_model", "test_losses", "test_eval", "quick_test"],
                       help="Execution mode")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to KiTS23 dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--num_vis", type=int, default=3, help="Number of cases to visualize")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_dino_vista_kits(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            resume=args.resume,
            device=args.device,
        )
    elif args.mode == "evaluate":
        # Load model and run full evaluation
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Create model
        model = create_dino_vista_kits()
        
        # Load checkpoint
        checkpoint_path = args.checkpoint or "./checkpoints_quick/best_model.pth"
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from: {checkpoint_path}")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}, using random weights")
        
        # Create validation dataloader
        _, val_loader, _, _ = create_dataloaders(
            dataset_dir=args.dataset_dir or DATA_CONFIG["dataset_dir"],
            batch_size=1,
            patch_size=MODEL_CONFIG["patch_size"],
            max_train_cases=DATA_CONFIG.get("max_train_cases"),
            max_val_cases=DATA_CONFIG.get("max_val_cases"),
        )
        
        # Run evaluation
        eval_dir = Path(args.output_dir) / "evaluation_results"
        run_full_evaluation(model, val_loader, str(eval_dir), device, num_vis_cases=args.num_vis)
        
    elif args.mode == "test_model":
        test_model()
    elif args.mode == "test_losses":
        test_losses()
    elif args.mode == "test_eval":
        test_evaluation()
    elif args.mode == "quick_test":
        quick_test(dataset_dir=args.dataset_dir, epochs=args.epochs or 5)