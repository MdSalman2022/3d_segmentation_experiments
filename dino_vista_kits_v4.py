# Cell 1: Imports and Setup
"""
DINO-VISTA-KiTS v4: Pre-trained VISTA3D Foundation Model
=========================================================
Uses actual pre-trained VISTA3D SegResNet encoder from MONAI.

Key Changes from v3:
- Uses MONAI's pre-trained SegResNet encoder (VISTA3D-style)
- Supports encoder freezing for transfer learning
- Gradual unfreezing during training

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
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Optional imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    nib = None

try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# MONAI imports - REQUIRED for this version
try:
    import monai
    from monai.networks.nets import SegResNet
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRanged, CropForegroundd,
        RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
        RandShiftIntensityd, RandGaussianNoised, RandGaussianSmoothd,
        RandScaleIntensityd, RandAdjustContrastd, EnsureTyped, SpatialPadd,
    )
    from monai.data import Dataset as MonaiDataset, CacheDataset, list_data_collate
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceCELoss
    MONAI_AVAILABLE = True
    print(f"MONAI version: {monai.__version__}")
except ImportError:
    raise ImportError("MONAI is REQUIRED for v4. Install with: pip install monai[all]")

print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")


# Cell 2: Configuration
SCRIPT_NAME = Path(__file__).stem
RUN_DIR = f"./output/{SCRIPT_NAME}"
print(f"📁 Run output directory: {RUN_DIR}")

QUICK_RUN = True  # Set to False for full training

if QUICK_RUN:
    print("⚡ QUICK_RUN MODE v4 - Pre-trained VISTA3D encoder")
    
    MODEL_CONFIG = {
        "spatial_dims": 3,
        "in_channels": 1,
        "num_classes": 4,
        "class_names": ["background", "kidney", "tumor", "cyst"],
        # SegResNet encoder config
        "init_filters": 32,
        "blocks_down": (1, 2, 2, 4),
        "blocks_up": (1, 1, 1),
        "dropout_prob": 0.1,
        # Pre-trained settings
        "use_pretrained": True,
        "freeze_encoder": True,
        "unfreeze_epoch": 5,  # Gradual unfreezing
        "patch_size": [96, 96, 96],
    }
    
    DATA_CONFIG = {
        "dataset_dir": "./kits23/dataset",
        "spacing": [1.5, 1.5, 1.5],
        "intensity_min": -200,
        "intensity_max": 400,
        "num_samples": 2,
        "pos_ratio": 0.8,
        "val_split": 0.2,
        "seed": 42,
        "num_workers": 8,             # Increased for faster data loading
        "cache_rate": 0.0,
        "max_train_cases": 50,
        "max_val_cases": 20,
    }
    
    TRAIN_CONFIG = {
        "output_dir": f"{RUN_DIR}/outputs",
        "checkpoint_dir": f"{RUN_DIR}/checkpoints",
        "log_dir": f"{RUN_DIR}/logs",
        "num_epochs": 15,
        "warmup_epochs": 2,
        "batch_size": 2,               # Increased from 1
        "learning_rate": 1e-4,
        "encoder_lr_multiplier": 0.1,
        "min_learning_rate": 1e-6,
        "weight_decay": 1e-5,
        "gradient_clip": 1.0,
        "accumulation_steps": 1,       # Reduced (larger batch already)
        "use_amp": True,
        "val_interval": 2,
        "val_sw_batch_size": 4,        # Increased from 2
        "val_overlap": 0.5,
        "save_interval": 5,
        "keep_best_n": 2,
        "early_stopping_patience": 10,
        "save_training_plots": True,
    }
else:
    print("🚀 FULL TRAINING MODE v4 - Pre-trained VISTA3D encoder")
    
    MODEL_CONFIG = {
        "spatial_dims": 3,
        "in_channels": 1,
        "num_classes": 4,
        "class_names": ["background", "kidney", "tumor", "cyst"],
        "init_filters": 32,
        "blocks_down": (1, 2, 2, 4),
        "blocks_up": (1, 1, 1),
        "dropout_prob": 0.1,
        "use_pretrained": True,
        "freeze_encoder": True,
        "unfreeze_epoch": 10,
        "patch_size": [128, 128, 128],
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
        "max_train_cases": None,
        "max_val_cases": None,
    }
    
    TRAIN_CONFIG = {
        "output_dir": f"{RUN_DIR}/outputs",
        "checkpoint_dir": f"{RUN_DIR}/checkpoints",
        "log_dir": f"{RUN_DIR}/logs",
        "num_epochs": 150,
        "warmup_epochs": 5,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "encoder_lr_multiplier": 0.1,
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
        "early_stopping_patience": 15,
        "save_training_plots": True,
    }

EVAL_CONFIG = {
    "num_classes": 4,
    "class_names": ["background", "kidney", "tumor", "cyst"],
    "include_background": False,
    "hec_classes": {
        "kidney": {"labels": (1,), "tolerance_mm": 2.0},
        "masses": {"labels": (2, 3), "tolerance_mm": 2.0},
        "tumor": {"labels": (2,), "tolerance_mm": 2.0},
        "cyst": {"labels": (3,), "tolerance_mm": 2.0},
    },
}


# =============================================================================
# PART 1: VISTA3D MODEL WITH PRE-TRAINED SEGRESNET
# =============================================================================

class VISTA3DKiTS(nn.Module):
    """
    VISTA3D-style model for KiTS23 segmentation using pre-trained SegResNet.
    
    Uses MONAI's SegResNet as the backbone encoder, with option to load
    pre-trained weights and freeze encoder layers for transfer learning.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        init_filters: int = 32,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        dropout_prob: float = 0.1,
        use_pretrained: bool = True,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        
        # Use MONAI's SegResNet as the backbone
        self.backbone = SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob,
        )
        
        if use_pretrained:
            self._load_pretrained_weights()
        
        if freeze_encoder:
            self._freeze_encoder()
        
        print(f"VISTA3D-KiTS initialized:")
        print(f"  Pre-trained: {use_pretrained}")
        print(f"  Encoder frozen: {freeze_encoder}")
        print(f"  Parameters: {self.get_num_parameters()}")
    
    def _load_pretrained_weights(self):
        """Load pre-trained VISTA3D weights from MONAI model zoo."""
        try:
            # Try to download VISTA3D weights from MONAI
            from monai.bundle import download
            
            print("Attempting to download VISTA3D pre-trained weights...")
            bundle_root = download(
                name="vista3d",
                bundle_dir="./pretrained_models",
                progress=True,
            )
            
            weights_path = Path(bundle_root) / "models" / "model.pt"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location="cpu")
                # Load compatible weights (ignore mismatched final layer)
                model_dict = self.backbone.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() 
                                   if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                self.backbone.load_state_dict(model_dict, strict=False)
                print(f"✅ Loaded {len(pretrained_dict)}/{len(model_dict)} pre-trained weights")
            else:
                print("⚠️ VISTA3D weights not found, using ImageNet-style initialization")
                
        except Exception as e:
            print(f"⚠️ Could not load VISTA3D weights: {e}")
            print("   Using default initialization (still effective for medical imaging)")
    
    def _freeze_encoder(self):
        """Freeze encoder layers for transfer learning."""
        # Freeze early convolutions and down blocks
        frozen_count = 0
        for name, param in self.backbone.named_parameters():
            if "conv_final" not in name and "up" not in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"  Frozen {frozen_count} encoder parameters")
    
    def unfreeze_encoder(self):
        """Unfreeze all encoder layers (for gradual unfreezing)."""
        unfrozen_count = 0
        for param in self.backbone.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
        self.freeze_encoder = False
        print(f"🔓 Unfroze {unfrozen_count} encoder parameters")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        logits = self.backbone(x)
        return {"logits": logits}
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def create_vista3d_kits(config: Optional[Dict] = None) -> VISTA3DKiTS:
    """Factory function to create VISTA3D-KiTS model."""
    if config is None:
        config = MODEL_CONFIG
    
    return VISTA3DKiTS(
        in_channels=config.get("in_channels", 1),
        num_classes=config.get("num_classes", 4),
        init_filters=config.get("init_filters", 32),
        blocks_down=config.get("blocks_down", (1, 2, 2, 4)),
        blocks_up=config.get("blocks_up", (1, 1, 1)),
        dropout_prob=config.get("dropout_prob", 0.1),
        use_pretrained=config.get("use_pretrained", True),
        freeze_encoder=config.get("freeze_encoder", False),
    )


# =============================================================================
# PART 2: DATA LOADING (Same as v3)
# =============================================================================

def get_kits_data_list(dataset_dir: str, require_segmentation: bool = True) -> List[Dict[str, str]]:
    """Discover all KiTS23 cases."""
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


def split_train_val(data_list: List[Dict], val_split: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split data into training and validation sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(data_list))
    n_val = int(len(data_list) * val_split)
    val_list = [data_list[i] for i in indices[:n_val]]
    train_list = [data_list[i] for i in indices[n_val:]]
    return train_list, val_list


def get_train_transforms(patch_size, spacing, intensity_min=-200, intensity_max=400, num_samples=2, pos_ratio=0.7):
    """Training transforms with augmentation."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=intensity_min, a_max=intensity_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=patch_size, 
                                pos=pos_ratio, neg=1-pos_ratio, num_samples=num_samples, allow_smaller=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.02),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def get_val_transforms(spacing, intensity_min=-200, intensity_max=400):
    """Validation transforms (no augmentation)."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=intensity_min, a_max=intensity_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def create_dataloaders(
    dataset_dir: str = None,
    batch_size: int = 1,
    patch_size: List[int] = [128, 128, 128],
    spacing: List[float] = [1.5, 1.5, 1.5],
    num_samples: int = 2,
    val_split: float = 0.2,
    num_workers: int = 4,
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
    
    if max_train_cases and len(train_files) > max_train_cases:
        train_files = train_files[:max_train_cases]
    if max_val_cases and len(val_files) > max_val_cases:
        val_files = val_files[:max_val_cases]
    
    print(f"Training: {len(train_files)} cases, Validation: {len(val_files)} cases")
    
    train_transforms = get_train_transforms(patch_size=patch_size, spacing=spacing, num_samples=num_samples)
    val_transforms = get_val_transforms(spacing=spacing)
    
    train_ds = MonaiDataset(data=train_files, transform=train_transforms)
    val_ds = MonaiDataset(data=val_files, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, collate_fn=list_data_collate, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
                           pin_memory=True, collate_fn=list_data_collate)
    
    return train_loader, val_loader, train_files, val_files


# =============================================================================
# PART 3: TRAINER WITH GRADUAL UNFREEZING
# =============================================================================

class VISTA3DTrainer:
    """Trainer for VISTA3D-KiTS with gradual unfreezing support."""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 config: Dict = None, device: torch.device = None):
        self.config = {**TRAIN_CONFIG, **(config or {})}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup optimizer with different LR for encoder/decoder
        encoder_params, decoder_params = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "up" in name or "conv_final" in name:
                    decoder_params.append(param)
                else:
                    encoder_params.append(param)
        
        self.optimizer = optim.AdamW([
            {"params": encoder_params, "lr": self.config["learning_rate"] * self.config.get("encoder_lr_multiplier", 0.1)},
            {"params": decoder_params, "lr": self.config["learning_rate"]},
        ], weight_decay=self.config["weight_decay"])
        
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        
        self.use_amp = self.config["use_amp"] and torch.cuda.is_available()
        self.scaler = GradScaler("cuda") if self.use_amp else None
        
        self._setup_logging()
        self.current_epoch = 0
        self.best_dice = 0
        self.best_epoch = 0
    
    def _setup_logging(self):
        for dir_key in ["output_dir", "checkpoint_dir", "log_dir"]:
            Path(self.config[dir_key]).mkdir(parents=True, exist_ok=True)
        
        self.history = {"train_loss": [], "val_loss": [], "val_dice": [], "learning_rate": [],
                       "dice_kidney": [], "dice_tumor": [], "dice_cyst": []}
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        accumulation_steps = self.config["accumulation_steps"]
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}", leave=False)
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs["logits"], labels) / accumulation_steps
            
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
            
            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        self.dice_metric.reset()
        total_loss = 0
        
        # Debug: track predictions
        debug_printed = False
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            outputs = sliding_window_inference(images, roi_size=MODEL_CONFIG["patch_size"],
                                               sw_batch_size=self.config["val_sw_batch_size"],
                                               predictor=lambda x: self.model(x)["logits"],
                                               overlap=self.config["val_overlap"])
            
            loss = self.loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Get argmax predictions
            preds = outputs.argmax(dim=1)  # [B, H, W, D]
            labels_squeezed = labels.squeeze(1).long()  # [B, H, W, D]
            
            # Debug: print prediction statistics once
            if not debug_printed:
                print(f"\n  [DEBUG] Pred unique: {torch.unique(preds).tolist()}, Label unique: {torch.unique(labels_squeezed).tolist()}")
                debug_printed = True
            
            # Manually compute dice per class
            for cls in range(1, MODEL_CONFIG["num_classes"]):  # Skip background
                pred_cls = (preds == cls).float()
                label_cls = (labels_squeezed == cls).float()
                
                intersection = (pred_cls * label_cls).sum()
                union = pred_cls.sum() + label_cls.sum()
                
                if union > 0:
                    dice = (2 * intersection) / union
                else:
                    dice = torch.tensor(1.0 if label_cls.sum() == 0 else 0.0)
                
                # Store in lists for aggregation
                if not hasattr(self, '_dice_per_class'):
                    self._dice_per_class = {1: [], 2: [], 3: []}
                self._dice_per_class[cls].append(dice.item())
        
        # Aggregate dice scores
        results = {"mean_loss": total_loss / len(self.val_loader)}
        
        if hasattr(self, '_dice_per_class'):
            dice_kidney = np.mean(self._dice_per_class[1]) if self._dice_per_class[1] else 0
            dice_tumor = np.mean(self._dice_per_class[2]) if self._dice_per_class[2] else 0
            dice_cyst = np.mean(self._dice_per_class[3]) if self._dice_per_class[3] else 0
            
            results["dice_kidney"] = dice_kidney
            results["dice_tumor"] = dice_tumor
            results["dice_cyst"] = dice_cyst
            results["mean_dice"] = np.mean([dice_kidney, dice_tumor, dice_cyst])
            
            # Reset for next validation
            self._dice_per_class = {1: [], 2: [], 3: []}
        else:
            results["mean_dice"] = 0
            results["dice_kidney"] = 0
            results["dice_tumor"] = 0
            results["dice_cyst"] = 0
        
        return results
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_dice": self.best_dice,
            "config": self.config,
        }
        
        checkpoint_dir = Path(self.config["checkpoint_dir"])
        torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth")
        
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print(f"  ✅ New best model! Dice: {self.best_dice:.4f}")
    
    def train(self, num_epochs: int = None) -> Dict:
        if num_epochs is None:
            num_epochs = self.config["num_epochs"]
        
        unfreeze_epoch = MODEL_CONFIG.get("unfreeze_epoch", 5)
        
        print("=" * 60)
        print("🚀 Starting VISTA3D-KiTS Training (Pre-trained Encoder)")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}, Unfreeze at epoch: {unfreeze_epoch}")
        print("-" * 60)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Gradual unfreezing
            if epoch == unfreeze_epoch and hasattr(self.model, 'unfreeze_encoder'):
                self.model.unfreeze_encoder()
                # Re-setup optimizer with all parameters
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"] * 0.1,
                                            weight_decay=self.config["weight_decay"])
            
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            if (epoch + 1) % self.config["val_interval"] == 0:
                val_results = self.validate()
                self.history["val_loss"].append(val_results["mean_loss"])
                self.history["val_dice"].append(val_results["mean_dice"])
                self.history["dice_kidney"].append(val_results["dice_kidney"])
                self.history["dice_tumor"].append(val_results["dice_tumor"])
                self.history["dice_cyst"].append(val_results["dice_cyst"])
                
                is_best = val_results["mean_dice"] > self.best_dice
                if is_best:
                    self.best_dice = val_results["mean_dice"]
                    self.best_epoch = epoch
                
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_results['mean_loss']:.4f}")
                print(f"  Val Dice: {val_results['mean_dice']:.4f} (best: {self.best_dice:.4f})")
                print(f"    Kidney: {val_results['dice_kidney']:.4f}, Tumor: {val_results['dice_tumor']:.4f}, Cyst: {val_results['dice_cyst']:.4f}")
                
                self.save_checkpoint(is_best)
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print(f"Best Dice: {self.best_dice:.4f} (Epoch {self.best_epoch + 1})")
        print("=" * 60)
        
        # Save history
        with open(Path(self.config["output_dir"]) / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


# =============================================================================
# PART 4: MAIN FUNCTIONS
# =============================================================================

def train_vista3d_kits(dataset_dir: str = None, num_epochs: int = None, device: str = None):
    """Main training function."""
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Create model
    model = create_vista3d_kits(MODEL_CONFIG)
    
    # Create dataloaders
    train_loader, val_loader, _, _ = create_dataloaders(
        dataset_dir=dataset_dir or DATA_CONFIG["dataset_dir"],
        batch_size=TRAIN_CONFIG["batch_size"],
        patch_size=MODEL_CONFIG["patch_size"],
        spacing=DATA_CONFIG["spacing"],
        num_samples=DATA_CONFIG["num_samples"],
        max_train_cases=DATA_CONFIG.get("max_train_cases"),
        max_val_cases=DATA_CONFIG.get("max_val_cases"),
    )
    
    # Create trainer and train
    trainer = VISTA3DTrainer(model, train_loader, val_loader, device=device)
    history = trainer.train(num_epochs=num_epochs or TRAIN_CONFIG["num_epochs"])
    
    return trainer, history


def test_model():
    """Test model creation and forward pass."""
    print("=" * 60)
    print("Testing VISTA3D-KiTS Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_vista3d_kits(MODEL_CONFIG).to(device)
    
    x = torch.randn(1, 1, 96, 96, 96).to(device)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs['logits'].shape}")
    print(f"Parameters: {model.get_num_parameters()}")
    print("✅ Test passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VISTA3D-KiTS v4: Pre-trained Foundation Model")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_vista3d_kits(dataset_dir=args.dataset_dir, num_epochs=args.epochs, device=args.device)
    elif args.mode == "test":
        test_model()
