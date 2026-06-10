"""
KiTS23 SC-UNet (Spatial-Channel Attention U-Net)
=================================================
Based on: Gohil & Lad - "Kidney Tumor Segmentation using Spatial-Channel Attention"

This implementation follows the paper's configuration which achieved:
- Kidney Dice: 0.952
- Tumor Dice:  0.665
- Cyst Dice:   0.656

Key differences from typical U-Net:
1. Smaller, shallower network (32, 64, 128) - only 3 encoder stages
2. CBAM-style attention (Spatial + Channel)
3. Standard DiceCE loss (no aggressive weighting needed)
4. Anisotropic spacing: 2.0 × 1.62 × 1.62 mm
5. Trained for 300-500 epochs

Labels:
- 0: Background
- 1: Kidney
- 2: Tumor  
- 3: Cyst

Usage:
    python sc_unet_kits23.py --mode train
    python sc_unet_kits23.py --mode train --quick
    python sc_unet_kits23.py --mode inference
    python sc_unet_kits23.py --mode evaluate
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
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
    EnsureTyped,
    AsDiscreted,
)
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import SmartCacheDataset, DataLoader, Dataset, decollate_batch


# ============================================================================
# CONFIGURATION - Based on SC-UNet Paper
# ============================================================================
CONFIG = {
    # Dataset paths
    "kits23_dir": "./kits23/dataset",
    
    # Output paths
    "output_dir": "./output/sc_unet",
    "checkpoint_dir": "./output/sc_unet/checkpoints",
    "predictions_dir": "./output/sc_unet/predictions",
    
    # Model architecture - Paper: (32, 64, 128) encoder
    "model": {
        "in_channels": 1,
        "out_channels": 4,  # background, kidney, tumor, cyst
        "encoder_channels": (32, 64, 128),  # 3-stage encoder (paper config)
        "attention_ratio": 8,  # Channel attention reduction ratio
    },
    
    # Training settings - Paper: 300-500 epochs
    "training": {
        "num_epochs": 500,
        "quick_epochs": 5,  # Quick test
        "batch_size": 2,
        "num_samples": 4,  # Patches per volume
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "val_interval": 5,  # Validate every 5 epochs
        "cache_rate": 0.4,  # Paper: 0.4
        "replace_rate": 0.5,
        "num_workers": 4,
        "use_amp": True,
    },
    
    # Data settings - Paper config
    "data": {
        "patch_size": (64, 128, 128),  # Paper: 64×128×128
        "spacing": (2.0, 1.62, 1.62),  # Paper: anisotropic spacing
        "intensity_min": -80,  # Paper HU range
        "intensity_max": 305,
        "train_val_split": 0.9,  # Paper: 90/10
    },
    
    # Inference settings - Paper: 0.8 overlap
    "inference": {
        "roi_size": (64, 128, 128),
        "sw_batch_size": 4,
        "overlap": 0.8,  # Paper: 0.8 (high overlap)
    },
    
    # Random seed
    "seed": 42,
}


# ============================================================================
# SC-UNET MODEL - Based on Paper Architecture
# ============================================================================
class ChannelAttention(nn.Module):
    """
    Channel Attention Module (SE-style).
    Learns channel-wise importance using global pooling.
    """
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Learns spatial importance using channel pooling.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines Channel and Spatial attention sequentially.
    """
    def __init__(self, channels: int, ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvBlock(nn.Module):
    """Basic 3D convolution block with BatchNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with convolution and optional attention."""
    def __init__(self, in_channels: int, out_channels: int, 
                 use_attention: bool = False, attention_ratio: int = 8):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2)
        self.attention = CBAM(out_channels, attention_ratio) if use_attention else None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        if self.attention:
            x = self.attention(x)
        return self.pool(x), x  # pooled, skip


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 use_attention: bool = False, attention_ratio: int = 8):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        self.attention = CBAM(out_channels, attention_ratio) if use_attention else None
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2,
                         diff_d // 2, diff_d - diff_d // 2])
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        if self.attention:
            x = self.attention(x)
        return x


class SCUNet(nn.Module):
    """
    SC-UNet: Spatial-Channel Attention U-Net for 3D Medical Image Segmentation.
    
    Architecture (from paper):
    - 3-stage encoder: (32, 64, 128)
    - Bottleneck: 128 channels with Channel Attention
    - 3-stage decoder with skip connections
    - Spatial attention on first encoder and last decoder
    
    Args:
        in_channels: Input channels (1 for CT)
        out_channels: Output classes (4 for KiTS23)
        encoder_channels: Tuple of encoder channel sizes
        attention_ratio: Reduction ratio for channel attention
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        encoder_channels: Tuple[int, ...] = (32, 64, 128),
        attention_ratio: int = 8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder - 3 stages
        # Stage 1: Spatial attention (paper config)
        self.enc1 = EncoderBlock(in_channels, encoder_channels[0], 
                                  use_attention=True, attention_ratio=attention_ratio)
        self.enc2 = EncoderBlock(encoder_channels[0], encoder_channels[1], 
                                  use_attention=False)
        self.enc3 = EncoderBlock(encoder_channels[1], encoder_channels[2], 
                                  use_attention=False)
        
        # Bottleneck - Channel attention (paper config)
        self.bottleneck = nn.Sequential(
            ConvBlock(encoder_channels[2], encoder_channels[2]),
            ChannelAttention(encoder_channels[2], attention_ratio)
        )
        
        # Decoder - 3 stages
        # After concat: bottleneck + skip3 = 128 + 128 = 256
        self.dec3 = DecoderBlock(encoder_channels[2], encoder_channels[2], 
                                  encoder_channels[1], use_attention=False)
        self.dec2 = DecoderBlock(encoder_channels[1], encoder_channels[1], 
                                  encoder_channels[0], use_attention=False)
        # Last decoder: Spatial attention (paper config)
        self.dec1 = DecoderBlock(encoder_channels[0], encoder_channels[0], 
                                  encoder_channels[0], use_attention=True, 
                                  attention_ratio=attention_ratio)
        
        # Output
        self.out_conv = nn.Conv3d(encoder_channels[0], out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x, skip1 = self.enc1(x)  # 32 channels
        x, skip2 = self.enc2(x)  # 64 channels
        x, skip3 = self.enc3(x)  # 128 channels
        
        # Bottleneck
        x = self.bottleneck(x)  # 128 channels with attention
        
        # Decoder
        x = self.dec3(x, skip3)  # 64 channels
        x = self.dec2(x, skip2)  # 32 channels
        x = self.dec1(x, skip1)  # 32 channels with attention
        
        # Output
        return self.out_conv(x)


# ============================================================================
# DATA LOADING
# ============================================================================
def get_kits23_data_dicts(data_dir: str) -> List[Dict]:
    """Get data dictionaries for KiTS23 cases with segmentation."""
    data_path = Path(data_dir)
    data_dicts = []
    
    cases = sorted([d for d in data_path.iterdir() 
                   if d.is_dir() and d.name.startswith('case_')])
    
    for case_dir in cases:
        img_path = case_dir / "imaging.nii.gz"
        seg_path = case_dir / "segmentation.nii.gz"
        
        if img_path.exists() and seg_path.exists():
            data_dicts.append({
                "image": str(img_path),
                "label": str(seg_path),
                "case_id": case_dir.name
            })
    
    return data_dicts


def get_train_transforms(config: dict) -> Compose:
    """Training transforms based on paper configuration."""
    data_cfg = config["data"]
    train_cfg = config["training"]
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Paper spacing: 2.0 × 1.62 × 1.62
        Spacingd(
            keys=["image", "label"],
            pixdim=data_cfg["spacing"],
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Paper HU clipping: (-80, 305)
        ScaleIntensityRanged(
            keys=["image"],
            a_min=data_cfg["intensity_min"],
            a_max=data_cfg["intensity_max"],
            b_min=0.0, b_max=1.0, clip=True
        ),
        
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=data_cfg["patch_size"]),
        
        # Random cropping
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=data_cfg["patch_size"],
            pos=1, neg=1,  # Paper didn't use aggressive oversampling
            num_samples=train_cfg["num_samples"],
            image_key="image",
            image_threshold=0,
        ),
        
        # Paper augmentations
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 8),
            magnitude_range=(50, 150),
            prob=0.5,
            mode=("bilinear", "nearest"),
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.25),
        RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.1),
        
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms(config: dict) -> Compose:
    """Validation transforms."""
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
        EnsureTyped(keys=["image", "label"]),
    ])


# ============================================================================
# TRAINING
# ============================================================================
def train_model(config: dict, num_epochs: int = None, resume_from: str = None):
    """Train SC-UNet model."""
    print("\n" + "=" * 70)
    print("  SC-UNET: Spatial-Channel Attention U-Net for KiTS23")
    print("  Based on: Gohil & Lad Paper Configuration")
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
    
    # Split (90/10 per paper)
    split_idx = int(len(data_dicts) * config["data"]["train_val_split"])
    train_files = data_dicts[:split_idx]
    val_files = data_dicts[split_idx:]
    print(f"Train: {len(train_files)} cases")
    print(f"Val: {len(val_files)} cases")
    
    # Transforms
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    
    train_cfg = config["training"]
    
    # Datasets with SmartCache (paper: 0.4 cache rate)
    print(f"\n📦 Creating datasets (cache_rate={train_cfg['cache_rate']})...")
    try:
        train_ds = SmartCacheDataset(
            data=train_files, 
            transform=train_transforms,
            cache_rate=train_cfg["cache_rate"],
            replace_rate=train_cfg["replace_rate"],
            num_init_workers=train_cfg["num_workers"]
        )
    except:
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
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=train_cfg["num_workers"])
    
    # Model
    print("\n🔧 Creating SC-UNet model...")
    model_cfg = config["model"]
    model = SCUNet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        encoder_channels=model_cfg["encoder_channels"],
        attention_ratio=model_cfg["attention_ratio"],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Encoder: {model_cfg['encoder_channels']}")
    
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
        print(f"Resuming from epoch {start_epoch}")
    
    # Loss - Paper: standard DiceCE (no aggressive weighting)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    print(f"\n📉 Loss: DiceCE (standard, no class weights)")
    
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
    use_amp = train_cfg.get("use_amp", True)
    
    # AMP
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"   Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    
    # Post-processing
    post_pred = Compose([AsDiscreted(keys="pred", argmax=True, to_onehot=4)])
    post_label = Compose([AsDiscreted(keys="label", to_onehot=4)])
    
    # History
    epoch_loss_values = []
    best_metric_epoch = -1
    best_metrics_per_class = {"kidney": 0.0, "tumor": 0.0, "cyst": 0.0}
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Validation every {val_interval} epochs")
    
    training_start = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        print("-" * 70)
        print(f"Epoch {epoch + 1}/{epochs}")
        
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in tqdm(train_loader, desc="Training"):
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            del outputs, loss
            torch.cuda.empty_cache()
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        epoch_time = time.time() - epoch_start
        print(f"Loss: {epoch_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in tqdm(val_loader, desc="Validation"):
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    
                    # High overlap (0.8) as per paper
                    val_outputs = sliding_window_inference(
                        val_inputs,
                        config["inference"]["roi_size"],
                        config["inference"]["sw_batch_size"],
                        model,
                        overlap=config["inference"]["overlap"]
                    )
                    
                    val_outputs = [post_pred({"pred": i})["pred"] for i in decollate_batch(val_outputs)]
                    val_labels = [post_label({"label": i})["label"] for i in decollate_batch(val_labels)]
                    
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                per_class = dice_metric.aggregate()
                dice_metric.reset()
                
                kidney = per_class[0].item()
                tumor = per_class[1].item()
                cyst = per_class[2].item()
                mean_dice = per_class.mean().item()
                
                print(f"\n📊 Validation:")
                print(f"   Kidney: {kidney:.4f} | Tumor: {tumor:.4f} | Cyst: {cyst:.4f}")
                print(f"   Mean: {mean_dice:.4f}")
                
                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_metric_epoch = epoch + 1
                    best_metrics_per_class = {"kidney": kidney, "tumor": tumor, "cyst": cyst}
                    
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric,
                        "best_metrics_per_class": best_metrics_per_class,
                    }, checkpoint_dir / "best_model.pth")
                    print(f"✅ New best! Mean Dice: {best_metric:.4f}")
                
                print(f"🏆 Best: {best_metric:.4f} (epoch {best_metric_epoch})")
            
            torch.cuda.empty_cache()
        
        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
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
        "best_metric": best_metric,
    }, checkpoint_dir / "last_model.pth")
    
    # Summary
    total_time = time.time() - training_start
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"\n🏆 Best Model (Epoch {best_metric_epoch}):")
    print(f"   Kidney: {best_metrics_per_class['kidney']:.4f}")
    print(f"   Tumor:  {best_metrics_per_class['tumor']:.4f}")
    print(f"   Cyst:   {best_metrics_per_class['cyst']:.4f}")
    print(f"   Mean:   {best_metric:.4f}")
    
    # Save history
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump({
            "epoch_loss": epoch_loss_values,
            "best_metric": best_metric,
            "best_metric_epoch": best_metric_epoch,
            "best_metrics_per_class": best_metrics_per_class,
        }, f, indent=2)
    
    return model


# ============================================================================
# INFERENCE
# ============================================================================
def run_inference(config: dict, checkpoint_path: str = None):
    """Run inference with trained SC-UNet."""
    print("\n" + "=" * 70)
    print("  SC-UNET INFERENCE")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_cfg = config["model"]
    model = SCUNet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        encoder_channels=model_cfg["encoder_channels"],
        attention_ratio=model_cfg["attention_ratio"],
    ).to(device)
    
    if checkpoint_path is None:
        checkpoint_path = Path(config["checkpoint_dir"]) / "best_model.pth"
    
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    output_dir = Path(config["predictions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dicts = get_kits23_data_dicts(config["kits23_dir"])
    transforms = get_val_transforms(config)
    
    print(f"Running inference on {len(data_dicts)} cases (overlap={config['inference']['overlap']})")
    
    with torch.no_grad():
        for data_dict in tqdm(data_dicts, desc="Inference"):
            case_id = data_dict["case_id"]
            
            data = transforms({"image": data_dict["image"], "label": data_dict["label"]})
            inputs = data["image"].unsqueeze(0).to(device)
            
            outputs = sliding_window_inference(
                inputs,
                config["inference"]["roi_size"],
                config["inference"]["sw_batch_size"],
                model,
                overlap=config["inference"]["overlap"]
            )
            
            pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            orig_nii = nib.load(data_dict["image"])
            pred_nii = nib.Nifti1Image(pred, orig_nii.affine)
            nib.save(pred_nii, output_dir / f"{case_id}.nii.gz")
    
    print(f"\n✅ Predictions saved to: {output_dir}")


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate(config: dict):
    """Evaluate SC-UNet predictions."""
    print("\n" + "=" * 70)
    print("  SC-UNET EVALUATION")
    print("=" * 70)
    
    pred_path = Path(config["predictions_dir"])
    data_dicts = get_kits23_data_dicts(config["kits23_dir"])
    
    results = {"kidney": [], "tumor": [], "cyst": []}
    
    for data_dict in tqdm(data_dicts, desc="Evaluating"):
        case_id = data_dict["case_id"]
        pred_file = pred_path / f"{case_id}.nii.gz"
        
        if not pred_file.exists():
            continue
        
        pred = nib.load(pred_file).get_fdata().astype(np.int32)
        gt = nib.load(data_dict["label"]).get_fdata().astype(np.int32)
        
        # Resize if needed
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            factors = np.array(gt.shape) / np.array(pred.shape)
            pred = zoom(pred, factors, order=0).astype(np.int32)
        
        for label, name in [(1, "kidney"), (2, "tumor"), (3, "cyst")]:
            pred_bin = (pred == label).astype(float)
            gt_bin = (gt == label).astype(float)
            
            intersection = np.sum(pred_bin * gt_bin)
            union = np.sum(pred_bin) + np.sum(gt_bin)
            
            dice = 2 * intersection / union if union > 0 else 1.0
            results[name].append(dice)
    
    print("\n" + "-" * 50)
    print(f"{'Class':<12} {'Mean':>10} {'Std':>10}")
    print("-" * 50)
    for cls in ["kidney", "tumor", "cyst"]:
        print(f"{cls.capitalize():<12} {np.mean(results[cls]):>10.4f} {np.std(results[cls]):>10.4f}")
    print("-" * 50)
    overall = np.mean([np.mean(results[c]) for c in results])
    print(f"{'Overall':<12} {overall:>10.4f}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="SC-UNet for KiTS23")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "inference", "evaluate", "all"])
    parser.add_argument("--quick", action="store_true", help="Quick 50-epoch test")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    
    args = parser.parse_args()
    config = CONFIG.copy()
    
    if args.data_dir:
        config["kits23_dir"] = args.data_dir
    
    if args.quick:
        num_epochs = config["training"]["quick_epochs"]
    elif args.epochs:
        num_epochs = args.epochs
    else:
        num_epochs = None
    
    print("\n" + "=" * 70)
    print("  SC-UNET: KiTS23 Kidney/Tumor/Cyst Segmentation")
    print("  Based on: Gohil & Lad Paper")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Data: {config['kits23_dir']}")
    if num_epochs:
        print(f"Epochs: {num_epochs}")
    
    # Create directories
    for d in [config["output_dir"], config["checkpoint_dir"], config["predictions_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    if args.mode in ["train", "all"]:
        train_model(config, num_epochs=num_epochs, resume_from=args.checkpoint)
    
    if args.mode in ["inference", "all"]:
        run_inference(config, checkpoint_path=args.checkpoint)
    
    if args.mode in ["evaluate", "all"]:
        evaluate(config)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
