"""
MedDINO-VISTA3D V5 — Architectural Overhaul
=============================================

12 proven fixes from V4 analysis + ~3x speed improvement.

SPEED IMPROVEMENTS (37h → ~12-15h estimated):
  - Fix #9:  Single ViT forward pass (was 6 redundant passes → 6x encoder speedup)
  - Spatial-only decoder upsampling (depth stays constant, no wasted 140→1120→140)
  - Fewer epochs needed (40+25 vs 60+50) due to better training dynamics

ARCHITECTURAL FIXES:
  - Fix #1:  CT HU windowing (clip [-79,304] + z-score) instead of per-volume z-score
  - Fix #2:  Data augmentation (flips, rotations, intensity, noise)
  - Fix #3:  Multi-resolution skip connections (upsampled to different target sizes)
  - Fix #4:  3D depth convolutions on all skip features (inter-slice context)
  - Fix #5:  High-resolution bypass CNN (preserves small-structure detail)
  - Fix #6:  Deep supervision at 3 decoder stages
  - Fix #7:  Loss excludes background from Tversky/Dice mean
  - Fix #8:  Lower cyst weight (10→4) to reduce false positives
  - Fix #10: Foreground-centered validation crops (not naive center)
  - Fix #11: Dropout3d in decoder (was accepted but never applied)
  - Fix #12: EarlyStopping on dice (not on misleading hybrid loss)

Usage:
    python meddino_vista3d_v5.py --mode quick_test
    python meddino_vista3d_v5.py --mode full
    python meddino_vista3d_v5.py --mode full --resume
"""

import os
import json
import time
from datetime import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm
from einops import rearrange

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["XFORMERS_DISABLED"] = "1"


# ============================================================================
# CT NORMALIZATION CONSTANTS (Fix #1)
# Computed from KiTS23 foreground statistics (matches nnU-Net preprocessing)
# ============================================================================

HU_CLIP_MIN = -79.0  # 0.5th percentile of foreground HU
HU_CLIP_MAX = 304.0  # 99.5th percentile of foreground HU
HU_MEAN = 104.0  # Foreground mean HU
HU_STD = 76.0  # Foreground std HU


def normalize_ct(img: np.ndarray) -> np.ndarray:
    """Fix #1: Proper CT normalization using global HU statistics.

    Per-volume z-score destroys absolute HU information that distinguishes
    tissue types (kidney ~30-50 HU, cyst ~0-20 HU, tumor ~20-80 HU).
    """
    img = np.clip(img, HU_CLIP_MIN, HU_CLIP_MAX)
    img = (img - HU_MEAN) / HU_STD
    return img.astype(np.float32)


# ============================================================================
# CONFIGURATION
# ============================================================================


def get_config(mode: str) -> dict:
    base = {
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/meddino_vista3d_v5",
        "dinov2_backbone": "dinov2_vitb14",
        "hf_token": None,
        "num_classes": 4,
        "batch_size": 4,
        "val_split": 0.2,
        "num_samples_per_volume": 2,
        "patch_size": (140, 224, 224),
        "lr_stage1": 1e-4,
        "lr_stage2": 1e-5,
        "weight_decay": 1e-5,
        "dropout": 0.1,
    }

    if mode == "quick_test":
        base.update(
            {
                "num_epochs_stage1": 2,
                "num_epochs_stage2": 2,
                "quick_cases": 20,
                "use_warmup": False,
                "patience_stage1": 4,
                "patience_stage2": 4,
            }
        )
    else:
        # Fewer epochs needed — better loss/augmentation → faster convergence
        base.update(
            {
                "num_epochs_stage1": 40,  # Was 60
                "num_epochs_stage2": 25,  # Was 50
                "quick_cases": None,
                "use_warmup": True,
                "warmup_epochs": 3,
                "patience_stage1": 4,
                "patience_stage2": 4,
            }
        )
    return base


# ============================================================================
# HIGH-RESOLUTION BYPASS CNN (Fix #5)
#
# DINOv2's 14x downsampling destroys small structures. A tumor of 20³ voxels
# maps to ~1.4³ voxels at the bottleneck — unrecoverable. This lightweight
# 3D CNN operates at 1/4 resolution, preserving detail the ViT loses.
# ============================================================================


class HighResBypass(nn.Module):
    """Lightweight 3D CNN that preserves fine spatial detail."""

    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 1, D, H, W) → (B, 16, D, H/2, W/2)
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            # (B, 16, D, H/2, W/2) → (B, 32, D, H/4, W/4)
            nn.Conv3d(16, out_channels, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# SPATIAL ATTENTION
# ============================================================================


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


# ============================================================================
# DECODER UP-BLOCK (Fix #3: spatial-only upsampling)
#
# V4 used stride=2 on ALL dims including depth, blowing D from 140→1120
# then interpolating back. V5 upsamples only H,W — depth stays at D=140.
# ============================================================================


class UpBlockV5(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, skip_ch: int, use_attention: bool = True
    ):
        super().__init__()
        # Spatial-only upsampling: (D,H,W) → (D, 2H, 2W)
        self.upsample = nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.skip_conv = nn.Conv3d(skip_ch, out_ch, 1)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch * 2, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attention = SpatialAttention3D() if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        skip = self.skip_conv(skip)
        # Match sizes — skip comes from ViT at fixed H/14, W/14
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(
                skip, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x


# ============================================================================
# ENCODER V5 (Fix #9: single ViT pass, Fix #4: 3D depth convolutions)
# ============================================================================


class EncoderV5(nn.Module):
    """
    DINOv2 encoder with:
      - Single forward pass extracting layers [2, 5, 8, 11] simultaneously
        (V4 did 6 separate passes — 6x wasted compute)
      - 3D depth convolutions on every skip feature (inter-slice context)
      - Learnable fusion weights for bottleneck
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        feature_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        print(f"Loading DINOv2 Encoder: {model_name} ...")
        self.vit = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True, verbose=False
        )
        self.hidden_dim = 768  # ViT-B/14

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        # Layers extracted in a SINGLE pass (Fix #9)
        self.extract_layers = [2, 5, 8, 11]

        # Per-layer projection: 768 → feature_dim
        self.proj_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )

        # Fix #4: 3D depth convolutions for inter-slice context on every skip
        self.depth_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        feature_dim,
                        feature_dim,
                        kernel_size=(3, 1, 1),
                        padding=(1, 0, 0),
                    ),
                    nn.InstanceNorm3d(feature_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )

        # Learnable fusion weights for bottleneck (cheap alternative to 1024→256 conv)
        self.fusion_weights = nn.Parameter(torch.ones(4) / 4.0)

        # Bottleneck depth aggregator
        self.depth_agg = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
        )

        print(
            f"  ✓ Single-pass encoder: layers {self.extract_layers}, dim={feature_dim}"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, D, H, W = x.shape

        # Flatten depth into batch, expand 1-ch → 3-ch for ViT
        x_2d = x.squeeze(1).reshape(B * D, 1, H, W).expand(-1, 3, -1, -1)

        # ====== SINGLE ViT forward pass (Fix #9) ======
        # V4 did 6 separate calls here — each running all 12 ViT blocks.
        # This single call runs the 12 blocks once, extracting 4 intermediates.
        with torch.set_grad_enabled(not next(self.vit.parameters()).requires_grad):
            all_features = self.vit.get_intermediate_layers(x_2d, n=self.extract_layers)

        h, w = H // 14, W // 14  # ViT-B/14 patch grid

        # Project, reshape to 3D, apply depth convolutions
        processed = []
        for i, feat in enumerate(all_features):
            num_patches = feat.shape[1]
            ph = pw = int(num_patches**0.5)
            # (B*D, num_patches, 768) → (B*D, 768, ph, pw)
            feat_2d = feat.permute(0, 2, 1).reshape(B * D, self.hidden_dim, ph, pw)
            feat_2d = F.interpolate(
                feat_2d, size=(h, w), mode="bilinear", align_corners=False
            )
            # Project 768 → 256
            feat_proj = self.proj_heads[i](feat_2d.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )
            # Reshape to 3D: (B*D, 256, h, w) → (B, 256, D, h, w)
            feat_3d = rearrange(feat_proj, "(b d) c h w -> b c d h w", b=B, d=D)
            # Fix #4: inter-slice 3D convolution
            feat_3d = self.depth_convs[i](feat_3d)
            processed.append(feat_3d)

        # Bottleneck: learnable weighted sum of all 4 layers
        weights = F.softmax(self.fusion_weights, dim=0)
        bottleneck = sum(w * p for w, p in zip(weights, processed))
        bottleneck = self.depth_agg(bottleneck)

        return {
            "bottleneck": bottleneck,  # (B, 256, D, h, w)  — fused from all layers
            "skip_early": processed[0],  # Layer 2  — most spatial detail
            "skip_mid": processed[1],  # Layer 5  — intermediate features
            "skip_late": processed[2],  # Layer 8  — deeper semantics
        }


# ============================================================================
# DECODER V5 (Fix #6: deep supervision, Fix #11: dropout)
#
# 3 decoder stages with spatial-only upsampling:
#   16×16 → 32×32 → 64×64 → 128×128 → interpolate to 224×224
# Deep supervision heads at each intermediate resolution.
# High-res bypass fused before the final output.
# ============================================================================


class DecoderV5(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 4,
        bypass_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 3 upsampling stages: 16→32→64→128 (spatial only, depth stays at D)
        self.up1 = UpBlockV5(in_channels, 128, skip_ch=256, use_attention=True)
        self.up2 = UpBlockV5(128, 64, skip_ch=256, use_attention=True)
        self.up3 = UpBlockV5(64, 32, skip_ch=256, use_attention=True)

        # Fix #5: High-res bypass fusion (residual addition before final)
        self.bypass_proj = nn.Sequential(
            nn.Conv3d(bypass_channels, 32, 1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Final classification
        self.final = nn.Conv3d(32, num_classes, 1)

        # Fix #6: Deep supervision heads (at 32×32, 64×64, 128×128)
        self.ds_head1 = nn.Conv3d(128, num_classes, 1)  # After up1 (32×32)
        self.ds_head2 = nn.Conv3d(64, num_classes, 1)  # After up2 (64×64)
        self.ds_head3 = nn.Conv3d(32, num_classes, 1)  # After up3 (128×128)

        # Fix #11: Dropout (V4 accepted but never used it)
        self.dropout = nn.Dropout3d(dropout)

        print(
            f"  ✓ Decoder: 3 stages, deep supervision ×3, dropout={dropout}, bypass fusion"
        )

    def forward(
        self,
        encoder_out: Dict[str, torch.Tensor],
        bypass_feat: torch.Tensor,
        target_size: Tuple[int, int, int],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

        x = encoder_out["bottleneck"]

        # Stage 1: 16→32
        x = self.up1(x, encoder_out["skip_late"])
        x = self.dropout(x)
        ds1 = self.ds_head1(x)

        # Stage 2: 32→64
        x = self.up2(x, encoder_out["skip_mid"])
        x = self.dropout(x)
        ds2 = self.ds_head2(x)

        # Stage 3: 64→128
        x = self.up3(x, encoder_out["skip_early"])
        x = self.dropout(x)
        ds3 = self.ds_head3(x)

        # Fix #5: Fuse high-res bypass (residual addition)
        bypass_resized = F.interpolate(
            bypass_feat, size=x.shape[2:], mode="trilinear", align_corners=False
        )
        x = x + self.bypass_proj(bypass_resized)

        # Final output: interpolate from 128×128 → target (224×224)
        logits = self.final(x)
        if logits.shape[2:] != target_size:
            logits = F.interpolate(
                logits, size=target_size, mode="trilinear", align_corners=False
            )

        if self.training:
            return logits, [ds1, ds2, ds3]
        return logits


# ============================================================================
# COMPLETE MODEL
# ============================================================================


class MedDINOVISTA3DV5(nn.Module):
    """V5: DINOv2 encoder + hi-res bypass + 3-stage decoder with deep supervision."""

    def __init__(
        self,
        num_classes: int = 4,
        freeze_encoder: bool = True,
        dropout: float = 0.1,
        dinov2_backbone: str = "dinov2_vitb14",
    ):
        super().__init__()
        self.encoder = EncoderV5(
            model_name=dinov2_backbone,
            freeze_backbone=freeze_encoder,
        )
        self.bypass = HighResBypass(in_channels=1, out_channels=32)
        self.decoder = DecoderV5(
            num_classes=num_classes,
            bypass_channels=32,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        target_size = x.shape[2:]  # (D, H, W)
        bypass_feat = self.bypass(x)  # (B, 32, D, H/4, W/4)
        encoder_out = self.encoder(x)  # dict of 3D feature maps
        return self.decoder(encoder_out, bypass_feat, target_size)

    def unfreeze_encoder(self, num_blocks: int = 4):
        """Unfreeze the last N transformer blocks of DINOv2."""
        blocks = list(self.encoder.vit.blocks)
        for block in blocks[-num_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        # Also unfreeze the final norm
        for p in self.encoder.vit.norm.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"  Unfroze last {num_blocks} ViT blocks. Trainable params: {trainable:,}"
        )


# ============================================================================
# LOSS FUNCTIONS (Fix #7: exclude BG, Fix #8: lower cyst weight)
# ============================================================================


class V5HybridLoss(nn.Module):
    """
    Tversky + Focal + Dice hybrid.

    Fixes from V4:
      - Tversky/Dice mean computed over FOREGROUND classes only (was all 4 incl. BG)
      - Cyst weight lowered from 10.0 → 4.0 to reduce false positives
      - Tversky α/β moderated (0.3/0.7 vs 0.15/0.85) to reduce over-aggression
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # Fix #8: BG=0.1, Kidney=1.0, Tumor=12.0, Cyst=4.0
        w = torch.tensor([0.1, 1.0, 12.0, 4.0])
        self.register_buffer("class_weights", w)

        self.alpha = 0.3  # FP penalty (V4 was 0.15 — too soft on FP)
        self.beta = 0.7  # FN penalty (V4 was 0.85 — too aggressive, caused cyst FP)
        self.gamma = 2.0  # Focal gamma (V4 was 2.5)
        self.smooth = 1e-6

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_soft = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target.squeeze(1).long(), self.num_classes)
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()

        TP = (pred_soft * target_oh).sum(dim=(2, 3, 4))
        FP = ((1 - target_oh) * pred_soft).sum(dim=(2, 3, 4))
        FN = (target_oh * (1 - pred_soft)).sum(dim=(2, 3, 4))

        # ---- Tversky (Fix #7: foreground only) ----
        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        tversky_loss = 1 - tversky[:, 1:].mean()  # Exclude class 0 (BG)

        # ---- Dice (Fix #7: foreground only) ----
        dice = 2 * TP / (2 * TP + FP + FN + self.smooth)
        dice_loss = 1 - dice[:, 1:].mean()  # Exclude class 0 (BG)

        # ---- Focal CE ----
        ce = F.cross_entropy(
            pred, target.squeeze(1), weight=self.class_weights, reduction="none"
        )
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()

        return 0.4 * tversky_loss + 0.3 * focal_loss + 0.3 * dice_loss


class DeepSupervisionWrapper(nn.Module):
    """Fix #6: Wraps any base loss to add deep supervision at intermediate resolutions."""

    def __init__(
        self, base_loss: nn.Module, ds_weights: List[float] = [0.125, 0.25, 0.5]
    ):
        super().__init__()
        self.base_loss = base_loss
        self.ds_weights = ds_weights  # coarse → fine

    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, tuple):
            main_logits, ds_logits = outputs
        else:
            return self.base_loss(outputs, target)

        # Main loss
        total = self.base_loss(main_logits, target)

        # Deep supervision losses
        for logits, w in zip(ds_logits, self.ds_weights):
            target_ds = F.interpolate(
                target.float(), size=logits.shape[2:], mode="nearest"
            ).long()
            total = total + w * self.base_loss(logits, target_ds)

        return total


# ============================================================================
# DATASET (Fix #1: HU norm, Fix #2: augmentation, Fix #10: fg-centered val)
# ============================================================================


class KiTS23DatasetV5(Dataset):
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

        # Fix #1: HU windowing with global statistics
        img = normalize_ct(img)

        # Extract patch
        p_img, p_lbl = self._sample_patch(img, lbl)

        # Fix #2: Data augmentation (training only)
        if self.is_train:
            p_img, p_lbl = self._augment(p_img, p_lbl)

        return {
            "image": torch.from_numpy(p_img.copy()).float().unsqueeze(0),
            "label": torch.from_numpy(p_lbl.copy()).long().unsqueeze(0),
        }

    def _augment(self, img: np.ndarray, lbl: np.ndarray):
        """Fix #2: Standard 3D medical image augmentation (applied jointly to img+lbl)."""
        # Random flipping (each axis independently, 50% chance)
        if np.random.random() > 0.5:
            img = img[::-1, :, :]
            lbl = lbl[::-1, :, :]
        if np.random.random() > 0.5:
            img = img[:, ::-1, :]
            lbl = lbl[:, ::-1, :]
        if np.random.random() > 0.5:
            img = img[:, :, ::-1]
            lbl = lbl[:, :, ::-1]

        # Random 90° rotation in axial plane
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k, axes=(1, 2)).copy()

        # ---- Intensity augmentation (image only) ----
        # Multiplicative + additive brightness
        if np.random.random() > 0.5:
            img = img * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)

        # Gamma correction
        if np.random.random() > 0.5:
            img_min = img.min()
            img_range = img.max() - img_min + 1e-8
            gamma = np.random.uniform(0.7, 1.5)
            img = ((img - img_min) / img_range) ** gamma * img_range + img_min

        # Gaussian noise
        if np.random.random() > 0.7:
            noise_std = np.random.uniform(0.01, 0.05)
            img = img + np.random.normal(0, noise_std, img.shape).astype(np.float32)

        return img, lbl

    def _sample_patch(self, img, lbl):
        d, h, w = img.shape
        pd, ph, pw = self.patch_size

        # Pad if volume is smaller than patch
        if d < pd or h < ph or w < pw:
            pad = ((0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w)))
            img = np.pad(img, pad, mode="constant")
            lbl = np.pad(lbl, pad, mode="constant")
            d, h, w = img.shape

        if self.is_train:
            # Tumor-focused sampling: 50% Tumor, 25% Cyst, 15% Kidney, 10% Random
            rand_val = np.random.random()
            target_class = None
            if rand_val < 0.5:
                target_class = 2  # Tumor
            elif rand_val < 0.75:
                target_class = 3  # Cyst
            elif rand_val < 0.9:
                target_class = 1  # Kidney

            if target_class is not None:
                ds, hs, ws = self._find_class_center(
                    lbl, target_class, d, h, w, pd, ph, pw
                )
            else:
                ds = np.random.randint(0, max(1, d - pd + 1))
                hs = np.random.randint(0, max(1, h - ph + 1))
                ws = np.random.randint(0, max(1, w - pw + 1))
        else:
            # Fix #10: Center on foreground, not just volume center
            fg_indices = np.argwhere(lbl > 0)
            if len(fg_indices) > 0:
                center = fg_indices.mean(axis=0).astype(int)
                ds = int(np.clip(center[0] - pd // 2, 0, max(0, d - pd)))
                hs = int(np.clip(center[1] - ph // 2, 0, max(0, h - ph)))
                ws = int(np.clip(center[2] - pw // 2, 0, max(0, w - pw)))
            else:
                ds, hs, ws = (
                    max(0, (d - pd) // 2),
                    max(0, (h - ph) // 2),
                    max(0, (w - pw) // 2),
                )

        return (
            img[ds : ds + pd, hs : hs + ph, ws : ws + pw],
            lbl[ds : ds + pd, hs : hs + ph, ws : ws + pw],
        )

    def _find_class_center(self, lbl, target_class, d, h, w, pd, ph, pw):
        """Find a patch centered on a voxel of the target class."""
        indices = np.argwhere(lbl == target_class)
        if len(indices) == 0:
            # Fallback: try tumor or cyst
            indices = np.argwhere((lbl == 2) | (lbl == 3))
        if len(indices) == 0:
            # Fallback: try kidney
            indices = np.argwhere(lbl == 1)
        if len(indices) == 0:
            # Random
            return (
                np.random.randint(0, max(1, d - pd + 1)),
                np.random.randint(0, max(1, h - ph + 1)),
                np.random.randint(0, max(1, w - pw + 1)),
            )

        c = indices[np.random.randint(len(indices))]
        return (
            int(np.clip(c[0] - pd // 2, 0, max(0, d - pd))),
            int(np.clip(c[1] - ph // 2, 0, max(0, h - ph))),
            int(np.clip(c[2] - pw // 2, 0, max(0, w - pw))),
        )


def get_dataloaders(config, max_cases=None):
    data_dir = Path(config["kits23_dir"])
    cases = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")]
    )
    if max_cases:
        cases = cases[:max_cases]

    data_dicts = [
        {"image": str(c / "imaging.nii.gz"), "label": str(c / "segmentation.nii.gz")}
        for c in cases
        if (c / "imaging.nii.gz").exists()
    ]

    split = int(len(data_dicts) * config["val_split"])
    train_ds = KiTS23DatasetV5(
        data_dicts[split:], config["patch_size"], config["num_samples_per_volume"], True
    )
    val_ds = KiTS23DatasetV5(data_dicts[:split], config["patch_size"], 1, False)

    return (
        DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        ),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True),
    )


# ============================================================================
# TRAINING UTILITIES
# ============================================================================


class EarlyStoppingOnDice:
    """Fix #12: Stop based on dice (the actual metric) not on the hybrid loss."""

    def __init__(self, patience: int = 8, delta: float = 0.001, path: str = "best.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_dice = -1.0
        self.early_stop = False

    def __call__(self, dice: float, model: nn.Module):
        if dice > self.best_dice + self.delta:
            self.best_dice = dice
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"    ✓ Best dice: {dice:.4f} — saved to {Path(self.path).name}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            print(
                f"    EarlyStopping {self.counter}/{self.patience} (best={self.best_dice:.4f})"
            )


class MetricTracker:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, preds, targets):
        preds, targets = preds.cpu(), targets.cpu()
        mask = (targets >= 0) & (targets < self.num_classes)
        self.confusion_matrix += torch.bincount(
            self.num_classes * targets[mask].long() + preds[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(0) - tp
        fn = self.confusion_matrix.sum(1) - tp
        return {
            "dice": (2 * tp / (2 * tp + fp + fn + 1e-8)).numpy(),
            "iou": (tp / (tp + fp + fn + 1e-8)).numpy(),
            "precision": (tp / (tp + fp + 1e-8)).numpy(),
            "recall": (tp / (tp + fn + 1e-8)).numpy(),
        }

    def format_results(self):
        m = self.compute()
        classes = ["BG", "Kidney", "Tumor", "Cyst"]
        mean_dice = float(np.mean(m["dice"][1:]))
        mean_iou = float(np.mean(m["iou"][1:]))
        s = f"\n  Mean Dice (Fg): {mean_dice:.4f} | Mean IoU (Fg): {mean_iou:.4f}\n"
        s += f"  {'Class':<10} {'Dice':<8} {'IoU':<8} {'Prec':<8} {'Recall':<8}\n"
        s += "  " + "-" * 44 + "\n"
        for i, c in enumerate(classes):
            s += f"  {c:<10} {m['dice'][i]:.4f}   {m['iou'][i]:.4f}   {m['precision'][i]:.4f}   {m['recall'][i]:.4f}\n"
        return s, mean_dice


# ============================================================================
# TRAIN / VALIDATE
# ============================================================================


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    epoch_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        try:
            img = batch["image"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                out = model(img)
                loss = loss_fn(out, lbl)

            if not torch.isfinite(loss):
                print(f"\n  ⚠ Non-finite loss at batch {batch_idx}, skipping")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            count += 1

            if batch_idx % 10 == 0:
                del img, lbl, out, loss
                torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"\n  ⚠ Error batch {batch_idx}: {e}")
            torch.cuda.empty_cache()
            continue

    return epoch_loss / max(count, 1)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    count = 0
    tracker = MetricTracker(num_classes=4)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="  Val  ", leave=False)):
            try:
                img = batch["image"].to(device, non_blocking=True)
                lbl = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=True):
                    out = model(img)
                    loss = loss_fn(out, lbl)
                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        count += 1

                preds = torch.argmax(out, dim=1)
                tracker.update(preds, lbl.squeeze(1))

                if batch_idx % 5 == 0:
                    del img, lbl, out, loss, preds
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"\n  ⚠ Val error batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                continue

    report, mean_dice = tracker.format_results()
    metrics_dict = tracker.compute()
    return val_loss / max(count, 1), mean_dice, report, metrics_dict


# ============================================================================
# LOGGING HELPERS
# ============================================================================


def save_epoch_results(
    output_dir, stage, epoch, train_loss, val_loss, metrics, epoch_time, is_best
):
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
                name: {
                    "dice": float(metrics["dice"][i]),
                    "iou": float(metrics["iou"][i]),
                    "precision": float(metrics["precision"][i]),
                    "recall": float(metrics["recall"][i]),
                }
                for i, name in enumerate(["background", "kidney", "tumor", "cyst"])
            },
        },
        "is_best": bool(is_best),
    }

    history_file = output_dir / "training_history.json"
    history = {"version": "V5", "epochs": []}
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
    history["epochs"].append(epoch_data)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def save_checkpoint(
    output_dir, epoch, stage, model, optimizer, scheduler, best_dice, scaler
):
    path = output_dir / f"checkpoint_{stage}_epoch{epoch + 1}.pth"
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict(),
            "best_dice": best_dice,
        },
        path,
    )
    print(f"    → Checkpoint: {path.name}")


def load_checkpoint(output_dir, stage, model, optimizer, scheduler, scaler, device):
    checkpoints = sorted(output_dir.glob(f"checkpoint_{stage}_*.pth"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"  Resuming from {latest.name}")
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt["scheduler_state_dict"]:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================


def train(config, resume=False):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 64}")
    print(f"  MedDINO-VISTA3D  V5  —  Architectural Overhaul")
    print(f"{'=' * 64}")
    print(f"  Device  : {device}")
    print(f"  Output  : {output_dir}")
    print(
        f"  Fixes   : 12 (HU-norm, augment, multi-scale, deep-sup, hi-res bypass, ...)"
    )
    print(f"  Speed   : Single ViT pass + spatial-only decoder → ~3x faster")
    print(f"{'=' * 64}\n")

    # ---- Data ----
    train_loader, val_loader = get_dataloaders(
        config, max_cases=config.get("quick_cases")
    )
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Model ----
    model = MedDINOVISTA3DV5(
        num_classes=config["num_classes"],
        freeze_encoder=True,
        dropout=config["dropout"],
        dinov2_backbone=config["dinov2_backbone"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params:,} total | {trainable_params:,} trainable\n")

    # ---- Loss ----
    base_loss = V5HybridLoss(num_classes=config["num_classes"]).to(device)
    loss_fn = DeepSupervisionWrapper(base_loss, ds_weights=[0.125, 0.25, 0.5]).to(
        device
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr_stage1"], weight_decay=config["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda")

    # ---- Scheduler ----
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs_stage1"]
    )
    if config.get("use_warmup"):
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=config["warmup_epochs"]
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, scheduler], milestones=[config["warmup_epochs"]]
        )

    # ---- Early stopping (Fix #12: on dice, not loss) ----
    early_stopping = EarlyStoppingOnDice(
        patience=config["patience_stage1"],
        path=str(output_dir / "best_stage1.pth"),
    )

    checkpoint_freq = config.get("checkpoint_freq", 5)
    best_dice = 0.0
    total_start = time.time()

    # ======================= STAGE 1 =======================
    print("─── STAGE 1: Train decoder + bypass (frozen ViT encoder) ───")
    start_epoch = 0
    if resume:
        ckpt = load_checkpoint(
            output_dir, "stage1", model, optimizer, scheduler, scaler, device
        )
        if ckpt:
            start_epoch = ckpt["epoch"] + 1
            best_dice = ckpt["best_dice"]

    for epoch in range(start_epoch, config["num_epochs_stage1"]):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device
        )
        val_loss, mean_dice, report, metrics_dict = validate(
            model, val_loader, loss_fn, device
        )
        scheduler.step()
        dt = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Ep {epoch + 1:>2}/{config['num_epochs_stage1']} | "
            f"Loss {train_loss:.4f}/{val_loss:.4f} | Dice {mean_dice:.4f} | "
            f"lr {lr:.2e} | {dt / 60:.1f}min"
        )
        print(report)

        is_best = mean_dice > best_dice
        if is_best:
            best_dice = mean_dice

        save_epoch_results(
            output_dir,
            "stage1",
            epoch + 1,
            train_loss,
            val_loss,
            metrics_dict,
            dt,
            is_best,
        )
        early_stopping(mean_dice, model)
        if early_stopping.early_stop:
            print("  Early stopping triggered in Stage 1")
            break
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(
                output_dir,
                epoch,
                "stage1",
                model,
                optimizer,
                scheduler,
                best_dice,
                scaler,
            )

    # ======================= STAGE 2 =======================
    if config["num_epochs_stage2"] > 0:
        print("\n─── STAGE 2: Fine-tune ViT encoder (last 4 blocks) ───")

        # Load best stage 1 weights
        best_path = output_dir / "best_stage1.pth"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))

        model.unfreeze_encoder(num_blocks=4)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "encoder.vit" in n and p.requires_grad
                    ],
                    "lr": config["lr_stage2"],
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "encoder.vit" not in n and p.requires_grad
                    ],
                    "lr": config["lr_stage2"] * 5,
                },  # Decoder/bypass get higher lr
            ],
            weight_decay=config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs_stage2"]
        )
        early_stopping = EarlyStoppingOnDice(
            patience=config["patience_stage2"],
            path=str(output_dir / "best_final.pth"),
        )

        start_epoch_s2 = 0
        if resume:
            ckpt = load_checkpoint(
                output_dir, "stage2", model, optimizer, scheduler, scaler, device
            )
            if ckpt:
                start_epoch_s2 = ckpt["epoch"] + 1
                best_dice = ckpt["best_dice"]

        for epoch in range(start_epoch_s2, config["num_epochs_stage2"]):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, scaler, device
            )
            val_loss, mean_dice, report, metrics_dict = validate(
                model, val_loader, loss_fn, device
            )
            scheduler.step()
            dt = time.time() - t0

            lr_vit = optimizer.param_groups[0]["lr"]
            lr_dec = optimizer.param_groups[1]["lr"]
            print(
                f"  Ep {epoch + 1:>2}/{config['num_epochs_stage2']} | "
                f"Loss {train_loss:.4f}/{val_loss:.4f} | Dice {mean_dice:.4f} | "
                f"lr_vit {lr_vit:.2e} lr_dec {lr_dec:.2e} | {dt / 60:.1f}min"
            )
            print(report)

            is_best = mean_dice > best_dice
            if is_best:
                best_dice = mean_dice

            save_epoch_results(
                output_dir,
                "stage2",
                epoch + 1,
                train_loss,
                val_loss,
                metrics_dict,
                dt,
                is_best,
            )
            early_stopping(mean_dice, model)
            if early_stopping.early_stop:
                print("  Early stopping triggered in Stage 2")
                break
            if (epoch + 1) % checkpoint_freq == 0:
                save_checkpoint(
                    output_dir,
                    epoch,
                    "stage2",
                    model,
                    optimizer,
                    scheduler,
                    best_dice,
                    scaler,
                )

    # ---- Summary ----
    total_time = time.time() - total_start
    print(f"\n{'=' * 64}")
    print(f"  Training complete.  Best Dice: {best_dice:.4f}")
    print(f"  Total time: {total_time / 3600:.2f} hours")
    print(f"{'=' * 64}\n")

    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(
            {
                "version": "V5",
                "best_dice": float(best_dice),
                "total_params": total_params,
                "total_time_hours": round(total_time / 3600, 2),
                "config": {
                    k: str(v) if isinstance(v, Path) else v for k, v in config.items()
                },
            },
            f,
            indent=2,
        )

    return model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedDINO-VISTA3D V5")
    parser.add_argument(
        "--mode", type=str, default="full", choices=["quick_test", "full"]
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    args = parser.parse_args()

    config = get_config(args.mode)
    config["checkpoint_freq"] = 5
    train(config, resume=args.resume)
