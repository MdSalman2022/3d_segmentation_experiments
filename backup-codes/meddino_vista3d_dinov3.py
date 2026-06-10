"""
MedDINO-VISTA3D with DINOv3 Encoder Upgrade
============================================

Architecture:
- Encoder: DINOv3 (ViT-B/16) - Upgraded from DINOv2
- Decoder: VISTA3D Transformer (Cross-Attention + Class Queries)
- Loss: Tversky + CrossEntropy (Focus on small tumors)

DINOv3 Improvements:
- Gram Anchoring for better dense features
- Trained on 1.7B images (vs 142M in DINOv2)
- Out-of-box segmentation quality
- Text alignment capability (dino.txt)

Usage:
    # Quick Test (3 epochs)
    python meddino_vista3d_dinov3.py --mode quick_test

    # Full Training (100+50 epochs)
    python meddino_vista3d_dinov3.py --mode full
"""

import os
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

# Settings
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["XFORMERS_DISABLED"] = "1"

# ============================================================================
# CONFIGURATION
# ============================================================================


def get_config(mode):
    """Returns config based on mode (quick_test or full)"""
    base = {
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/meddino_vista3d_dinov3",
        "dinov2_backbone": "dinov2_vitb14",  # Using proven torch.hub approach
        "hf_token": None,
        "num_classes": 4,
        "batch_size": 2,
        "val_split": 0.2,
        "num_samples_per_volume": 2,
        "patch_size": (140, 224, 224),  # Same as bak3 (working version)
        "lr_stage1": 1e-4,
        "lr_stage2": 1e-5,
        "weight_decay": 1e-5,
    }

    if mode == "quick_test":
        base.update(
            {
                "num_epochs_stage1": 10,
                "num_epochs_stage2": 10,
                "quick_cases": 100,
                "use_warmup": False,
                "patience_stage1": 3,
                "patience_stage2": 3,
            }
        )
    else:
        base.update(
            {
                "num_epochs_stage1": 100,
                "num_epochs_stage2": 50,
                "quick_cases": None,
                "use_warmup": True,
                "warmup_epochs": 10,
                "patience_stage1": 15,
                "patience_stage2": 10,
            }
        )
    return base


# ============================================================================
# MODEL ARCHITECTURE - DINOv3 ENCODER
# ============================================================================


class MedDINOv3Encoder(nn.Module):
    """
    MedDINO Encoder with DINOv2 backbone via torch.hub.
    This is the PROVEN approach from bak3.py that works reliably.
    Uses same architecture as the working version.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        feature_dim: int = 256,
        freeze_backbone: bool = True,
        token: str = None,
    ):
        super().__init__()
        print(f"Loading DINOv2 via torch.hub: {model_name}...")

        # Load pretrained model via torch.hub (same as working bak3.py)
        self.vit = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True, verbose=False
        )
        self.hidden_dim = 768  # ViT-B/14

        print(f"✓ Loaded {model_name} successfully with pretrained weights")

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Multi-scale projections
        self.scale_layers = [
            2,
            5,
            8,
            11,
        ]  # 0-indexed in vit_base (which handles range logic)

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

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # 3D Aggregation
        self.depth_aggregator = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        # Prepare for ViT
        x = x.squeeze(1).view(B * D, 1, H, W).repeat(1, 3, 1, 1)

        # Extract features (same as bak3.py)
        with torch.set_grad_enabled(not next(self.vit.parameters()).requires_grad):
            features = [
                self.vit.get_intermediate_layers(x, n=[i])[0] for i in self.scale_layers
            ]

        # Project and Fuse
        h, w = H // 14, W // 14  # ViT-B/14 patch size
        multi_scale = []
        for i, feat in enumerate(features):
            # feat: (B*D, N, C)
            num_patches = feat.shape[1]
            patch_h = patch_w = int(num_patches**0.5)
            feat_2d = feat.permute(0, 2, 1).reshape(
                feat.shape[0], self.hidden_dim, patch_h, patch_w
            )
            feat_2d = F.interpolate(
                feat_2d, size=(h, w), mode="bilinear", align_corners=False
            )
            feat_proj = self.proj_heads[i](feat_2d.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )
            multi_scale.append(feat_proj)

        fused = self.fusion(torch.cat(multi_scale, dim=1))
        fused_3d = fused.reshape(B, D, -1, h, w).permute(0, 2, 1, 3, 4)
        return self.depth_aggregator(fused_3d)


# ============================================================================
# VISTA3D DECODER (Unchanged)
# ============================================================================


class VISTA3DTransformerDecoder(nn.Module):
    """
    VISTA3D-style Decoder with Cross-Attention and Class Queries.
    Includes Dropout for regularization in full training.
    """

    def __init__(
        self, in_channels: int = 256, num_classes: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout

        # 1. Upsample features to reduce memory cost before Attention
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )  # Output: (B, C, D, H/4, W/4)

        # 2. Learnable Class Queries
        self.class_queries = nn.Embedding(num_classes - 1, in_channels)

        # 3. Transformer Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=8,
            batch_first=True,
            dropout=dropout,  # Regularization
        )

        # 4. Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # Regularization
            nn.Linear(in_channels * 4, in_channels),
        )

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        # 5. Final Projection
        self.map_head = nn.Linear(in_channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = features.shape

        # 1. Upsample features
        x = self.upsample(features)
        _, _, D_u, H_u, W_u = x.shape

        # 2. Prepare features for Attention (Keys/Values)
        feat_seq = x.view(B, C, -1).permute(0, 2, 1)  # (B, Seq_Len, C)

        # 3. Prepare Queries
        queries = self.class_queries.weight.unsqueeze(0).repeat(B, 1, 1)

        # 4. Cross Attention
        attn_out, _ = self.cross_attn(query=queries, key=feat_seq, value=feat_seq)

        # 5. FFN & Residual
        queries = queries + self.norm1(attn_out)
        queries = queries + self.norm2(self.ffn(queries))

        # 6. Generate Masks
        masks = torch.bmm(queries, feat_seq.permute(0, 2, 1))
        masks = masks.view(B, self.num_classes - 1, D_u, H_u, W_u)

        # 7. Upsample to input size and combine with background
        masks = F.interpolate(
            masks, size=features.shape[2:], mode="trilinear", align_corners=False
        )
        bg_map = torch.zeros(B, 1, *masks.shape[2:]).to(masks.device)
        logits = torch.cat([bg_map, masks], dim=1)

        return logits


class MedDINOv3VISTA3D(nn.Module):
    """MedDINO-VISTA3D with DINOv2 Encoder (torch.hub - Proven Approach)"""

    def __init__(
        self,
        num_classes=4,
        freeze_encoder=True,
        dropout=0.1,
        dinov2_backbone="dinov2_vitb14",
        token=None,
    ):
        super().__init__()
        self.encoder = MedDINOv3Encoder(
            model_name=dinov2_backbone, freeze_backbone=freeze_encoder, token=token
        )
        self.decoder = VISTA3DTransformerDecoder(
            num_classes=num_classes, dropout=dropout
        )

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.decoder(feats)

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
# DATASET (Unchanged from original)
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
            img, lbl = np.pad(img, pad, mode="constant"), np.pad(
                lbl, pad, mode="constant"
            )
            d, h, w = img.shape

        if self.is_train:
            # Balanced Sampling: 40% Tumor, 40% Cyst, 10% Kidney, 10% Random
            rand_val = np.random.random()
            
            target_class = None
            if rand_val < 0.4:
                target_class = 3  # Cyst
            elif rand_val < 0.8:
                target_class = 2  # Tumor
            elif rand_val < 0.9:
                target_class = 1  # Kidney
            
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
                    # Fallback if specific class not found -> try tumor then kidney
                    fallback_indices = np.argwhere((lbl == 2) | (lbl == 3))
                    if len(fallback_indices) > 0:
                        c = fallback_indices[np.random.randint(len(fallback_indices))]
                        ds, hs, ws = (
                            np.clip(c[0] - pd // 2, 0, d - pd),
                            np.clip(c[1] - ph // 2, 0, h - ph),
                            np.clip(c[2] - pw // 2, 0, w - pw),
                        )
                    else:
                        # Fallback to kidney
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
                                np.random.randint(0, d - pd + 1),
                                np.random.randint(0, h - ph + 1),
                                np.random.randint(0, w - pw + 1),
                            )
            else:
                # 10% Random
                ds, hs, ws = (
                    np.random.randint(0, d - pd + 1),
                    np.random.randint(0, h - ph + 1),
                    np.random.randint(0, w - pw + 1),
                )
        else:
            ds, hs, ws = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2

        return (
            img[ds : ds + pd, hs : hs + ph, ws : ws + pw],
            lbl[ds : ds + pd, hs : hs + ph, ws : ws + pw],
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
    train_ds = KiTS23Dataset(
        data_dicts[split:], config["patch_size"], config["num_samples_per_volume"], True
    )
    val_ds = KiTS23Dataset(data_dicts[:split], config["patch_size"], 1, False)

    return (
        DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=2,  # Reduced to prevent worker deadlock
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Reduce memory pressure
        ),
        DataLoader(
            val_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=1,  # Reduced for stability
            pin_memory=True
        ),
    )


# ============================================================================
# LOSS FUNCTION (Unchanged)
# ============================================================================


class TverskyLoss(nn.Module):
    """
    Tversky Loss: Generalization of Dice.
    Penalizes False Negatives (missing tumors) more than False Positives.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
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
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if isinstance(alpha, (float, int, list)):
            alpha = torch.tensor(alpha)
        
        # Register as buffer so it moves to GPU automatically
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = alpha
            
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, C, D, H, W) logits
        # targets: (B, 1, D, H, W) labels
        ce_loss = F.cross_entropy(inputs, targets.squeeze(1), weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Weights: BG, Kidney, Tumor, Cyst
        w = torch.tensor([0.1, 1.0, 10.0, 15.0]) # Increased Cyst weight
        self.register_buffer("ce_weights", w)
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.focal = FocalLoss(alpha=w, gamma=2.0)

    def forward(self, outputs, target):
        # ce = F.cross_entropy(outputs, target.squeeze(1), weight=self.ce_weights)
        fc = self.focal(outputs, target)
        tv = self.tversky(outputs, target)
        return fc + tv


# ============================================================================
# TRAINING UTILS (Unchanged)
# ============================================================================


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
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
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class MetricTracker:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        mask = (targets >= 0) & (targets < self.num_classes)
        self.confusion_matrix += torch.bincount(
            self.num_classes * targets[mask].long() + preds[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        # Calculate TP, FP, FN, TN for each class
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(0) - tp
        fn = self.confusion_matrix.sum(1) - tp

        # Dice: 2TP / (2TP + FP + FN)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        # IoU: TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-8)

        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp + 1e-8)

        # Recall: TP / (TP + FN)
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

        # Mean (excluding background)
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
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            with torch.amp.autocast("cuda", enabled=True):
                out = model(img)
                loss = loss_fn(out, lbl)

            # Check for NaN/Inf
            if not torch.isfinite(loss):
                print(f"\n⚠ Warning: Non-finite loss at batch {batch_idx}, skipping...")
                continue

            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Explicit memory cleanup every 10 batches
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
                
                # Memory cleanup
                if batch_idx % 5 == 0:
                    del img, lbl, out, loss, preds
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                print(f"\n⚠ Validation error at batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                continue

    report, mean_dice = tracker.format_results()
    return val_loss / max(batch_count, 1), mean_dice, report


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def extract_features(model, image_slice):
    """Extract DINOv2 features from a 2D slice for visualization."""
    model.eval()
    with torch.no_grad():
        img_2d = torch.from_numpy(image_slice).float().unsqueeze(0).unsqueeze(0)

        # Resize to be divisible by 14
        h, w = img_2d.shape[2:]
        new_h = ((h + 13) // 14) * 14
        new_w = ((w + 13) // 14) * 14
        img_2d = F.interpolate(
            img_2d, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        img_rgb = img_2d.repeat(1, 3, 1, 1)

        features = model.encoder.vit.get_intermediate_layers(img_rgb, n=[11])[0]
        n_patches = features.shape[1]
        grid_size = int(np.sqrt(n_patches))
        features = features.reshape(grid_size, grid_size, -1)

        return features.cpu().numpy()


def compute_cosine_similarity(features, reference_patch_idx):
    """Compute cosine similarity between reference patch and all others."""
    h, w, c = features.shape
    features_flat = features.reshape(-1, c)
    features_norm = features_flat / (
        np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8
    )

    ref_idx = reference_patch_idx[0] * w + reference_patch_idx[1]
    ref_feature = features_norm[ref_idx : ref_idx + 1]

    similarities = np.dot(features_norm, ref_feature.T).squeeze()
    return similarities.reshape(h, w)


def visualize_training_results(config, checkpoints, case_idx=0, slice_idx=None):
    """Generate attention evolution and decoder attention visualizations."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not available, skipping visualization")
        return

    output_dir = Path(config["output_dir"])
    case_dir = Path(config["kits23_dir"]) / f"case_{case_idx:05d}"

    if not case_dir.exists():
        print(f"⚠ Case directory not found: {case_dir}")
        return

    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    # Load image
    img_path = case_dir / "imaging.nii.gz"
    if not img_path.exists():
        print(f"⚠ Image not found: {img_path}")
        return

    img = nib.load(img_path).get_fdata()
    if slice_idx is None:
        slice_idx = img.shape[0] // 2

    img_slice = img[slice_idx]
    img_slice_norm = (img_slice - img_slice.mean()) / (img_slice.std() + 1e-8)

    # 1. Attention Evolution
    print("\n1. Generating attention evolution visualization...")
    ref_patch = (img_slice.shape[0] // 14 // 2, img_slice.shape[1] // 14 // 2)
    n_checkpoints = len(checkpoints) + 1

    fig, axes = plt.subplots(2, n_checkpoints, figsize=(4 * n_checkpoints, 8))
    if n_checkpoints == 1:
        axes = axes.reshape(2, 1)

    # Plot original
    axes[0, 0].imshow(img_slice, cmap="gray")
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    # Process checkpoints
    for idx, (stage_name, ckpt_path) in enumerate(checkpoints.items(), start=1):
        model = MedDINOv3VISTA3D(
            num_classes=config["num_classes"], dinov2_backbone=config["dinov2_backbone"]
        )

        if Path(ckpt_path).exists():
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            print(f"   ✓ Loaded {stage_name}")
        else:
            print(f"   ⚠ Checkpoint not found: {stage_name}")
            continue

        features = extract_features(model, img_slice_norm)
        similarity_map = compute_cosine_similarity(features, ref_patch)

        similarity_resized = (
            F.interpolate(
                torch.from_numpy(similarity_map).unsqueeze(0).unsqueeze(0),
                size=img_slice.shape,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

        axes[0, idx].imshow(img_slice, cmap="gray", alpha=0.3)
        im = axes[0, idx].imshow(
            similarity_resized, cmap="jet", alpha=0.7, vmin=0, vmax=1
        )
        axes[0, idx].set_title(stage_name)
        axes[0, idx].axis("off")

        axes[1, idx].imshow(similarity_resized, cmap="jet", vmin=0, vmax=1)
        axes[1, idx].axis("off")

    plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    plt.suptitle(f"Cosine Similarity Evolution - Slice {slice_idx}", fontsize=16)
    plt.tight_layout()

    attn_path = output_dir / "attention_evolution.png"
    plt.savefig(attn_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ Saved: {attn_path}")
    plt.close()

    # 2. Decoder Attention
    print("\n2. Generating decoder attention maps...")
    final_ckpt = list(checkpoints.values())[-1]
    if Path(final_ckpt).exists():
        model = MedDINOv3VISTA3D(
            num_classes=config["num_classes"], dinov2_backbone=config["dinov2_backbone"]
        )
        model.load_state_dict(torch.load(final_ckpt, map_location="cpu"))
        model.eval()

        with torch.no_grad():
            img_norm = (img - img.mean()) / (img.std() + 1e-8)
            img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0)

            # Resize for model
            d, h, w = img_tensor.shape[2:]
            new_d = ((d + 13) // 14) * 14
            new_h = ((h + 13) // 14) * 14
            new_w = ((w + 13) // 14) * 14

            img_resized = F.interpolate(
                img_tensor,
                size=(new_d, new_h, new_w),
                mode="trilinear",
                align_corners=False,
            )
            logits = model(img_resized)
            logits_original = F.interpolate(
                logits, size=(d, h, w), mode="trilinear", align_corners=False
            )
            probs = F.softmax(logits_original, dim=1).cpu().numpy()[0]

        mid_slice = probs.shape[1] // 2
        classes = ["Background", "Kidney", "Tumor", "Cyst"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for idx, (ax, class_name) in enumerate(zip(axes.ravel(), classes)):
            ax.imshow(img[mid_slice], cmap="gray", alpha=0.4)
            im = ax.imshow(probs[idx, mid_slice], cmap="hot", alpha=0.6, vmin=0, vmax=1)
            ax.set_title(f"{class_name} Attention Map")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle("Class-Specific Attention Maps (VISTA3D Decoder)", fontsize=16)
        plt.tight_layout()

        decoder_path = output_dir / "decoder_attention.png"
        plt.savefig(decoder_path, dpi=150, bbox_inches="tight")
        print(f"   ✓ Saved: {decoder_path}")
        plt.close()

    print("\n" + "=" * 60)
    print("✅ Visualization Complete!")
    print("=" * 60)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def train(config):
    print("=" * 60)
    print(
        f"MedDINO-VISTA3D Training (Mode: {config.get('use_warmup', False) and 'Full' or 'Quick'})"
    )
    print(f"Encoder: {config['dinov2_backbone']}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
        torch.cuda.empty_cache()
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(
        config, max_cases=config.get("quick_cases")
    )

    # Model with DINOv2 (proven approach)
    model = MedDINOv3VISTA3D(
        num_classes=config["num_classes"],
        freeze_encoder=True,
        dropout=0.1,
        dinov2_backbone=config["dinov2_backbone"],
        token=config.get("hf_token"),
    ).to(device)
    print(f"Total Params: {sum(p.numel() for p in model.parameters()):,}")

    scaler = torch.amp.GradScaler("cuda")
    loss_fn = HybridLoss().to(device)

    best_dice = 0.0

    # Helper to create scheduler
    def get_scheduler(optimizer, epochs, is_stage2=False):
        if not config.get("use_warmup") or is_stage2:
            # Simple Cosine for Stage 2 or Quick Test
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
        else:
            # Warmup + Cosine for Stage 1 Full Training
            warmup_iters = config["warmup_epochs"]
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_iters
            )
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_iters, eta_min=1e-6
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, main], [warmup_iters]
            )

    # STAGE 1
    print("\n--- STAGE 1: Frozen Encoder ---")
    
    # Initial CUDA cleanup
    torch.cuda.empty_cache()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr_stage1"], weight_decay=config["weight_decay"]
    )
    scheduler = get_scheduler(optimizer, config["num_epochs_stage1"], is_stage2=False)

    early_stopping = EarlyStopping(
        patience=config["patience_stage1"],
        verbose=True,
        path=str(output_dir / "best_model_stage1.pth"),
    )

    for epoch in range(config["num_epochs_stage1"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs_stage1']} - Starting training...")
        
        try:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, scaler, device
            )
            print(f"Training complete. Validating...")
            
            val_loss, mean_dice, report = validate(model, val_loader, loss_fn, device)
            scheduler.step()
        except Exception as e:
            print(f"\n❌ Error in epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue

        print(
            f"Ep {epoch+1}/{config['num_epochs_stage1']} | Loss: {train_loss:.4f}/{val_loss:.4f} | Dice: {mean_dice:.4f}"
        )
        print(report)

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered in Stage 1")
            break

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), output_dir / "best_model_stage1_dice.pth")
            print(f"  -> Saved Best Dice Model (Dice: {best_dice:.4f})")

    # STAGE 2
    if config["num_epochs_stage2"] > 0:
        print("\n--- STAGE 2: Fine-tuning Encoder ---")
        if (output_dir / "best_model_stage1_dice.pth").exists():
            model.load_state_dict(torch.load(output_dir / "best_model_stage1_dice.pth"))
        elif (output_dir / "best_model_stage1.pth").exists():
            model.load_state_dict(torch.load(output_dir / "best_model_stage1.pth"))

        model.unfreeze_encoder(num_layers=4)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr_stage2"],
            weight_decay=config["weight_decay"],
        )
        scheduler = get_scheduler(
            optimizer, config["num_epochs_stage2"], is_stage2=True
        )

        early_stopping = EarlyStopping(
            patience=config["patience_stage2"],
            verbose=True,
            path=str(output_dir / "best_model_final.pth"),
        )

        for epoch in range(config["num_epochs_stage2"]):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, scaler, device
            )
            val_loss, mean_dice, report = validate(model, val_loader, loss_fn, device)
            scheduler.step()

            print(
                f"Ep {epoch+1}/{config['num_epochs_stage2']} | Loss: {train_loss:.4f}/{val_loss:.4f} | Dice: {mean_dice:.4f}"
            )
            print(report)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered in Stage 2")
                break

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), output_dir / "best_model_final_dice.pth")
                print(f"  -> Saved Best Final Dice Model (Dice: {best_dice:.4f})")

    print(f"\nTraining Complete. Best Overall Dice: {best_dice:.4f}")

    # Generate visualizations after training
    checkpoints = {}
    if (output_dir / "best_model_stage1.pth").exists():
        checkpoints["Stage 1: Early"] = str(output_dir / "best_model_stage1.pth")
    if (output_dir / "best_model_stage1_dice.pth").exists():
        checkpoints["Stage 1: Best"] = str(output_dir / "best_model_stage1_dice.pth")
    if (output_dir / "best_model_final.pth").exists():
        checkpoints["Stage 2: Final"] = str(output_dir / "best_model_final.pth")

    if checkpoints:
        visualize_training_results(config, checkpoints, case_idx=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="full", choices=["test", "quick_test", "full"]
    )
    # Token usually not needed for timm DINOv3 (ungated), but kept as compat
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face Access Token (optional)",
    )
    parser.add_argument(
        "--use-dinov2",
        action="store_true",
        help="Fallback to DINOv2 (ungated) if DINOv3 fails",
    )
    args = parser.parse_args()

    # helper
    def update_config(cfg, args):
        if args.hf_token:
            cfg["hf_token"] = args.hf_token
        if args.use_dinov2:
            print(">>> SWITCHING TO DINOv2 (Fallback: Native Transformers) <<<")
            cfg["dinov3_model_name"] = "facebook/dinov2-base"
            # DINOv2 base patch 14
            cfg["patch_size"] = (126, 224, 224)
            # IMPORTANT: Disable remote code for DINOv2 as it is native in transformers
            # We handle this by checking the name in the Encoder class

    if args.mode == "test":
        # Architecture test
        print("Testing Architecture...")
        config = get_config("quick_test")
        update_config(config, args)

        # Force GC
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        try:
            model = MedDINOv3VISTA3D(
                dinov3_model_name=config["dinov3_model_name"],
                token=config.get("hf_token"),
            ).cuda()
            x = torch.randn(1, 1, 32, 128, 128).cuda()
            y = model(x)
            print(f"✓ Input: {x.shape}, Output: {y.shape}")
            print(f"✓ Architecture Test Passed ({config['dinov3_model_name']}).")
            x = torch.randn(1, 1, 32, 128, 128).cuda()
            y = model(x)
            print(f"✓ Input: {x.shape}, Output: {y.shape}")
            print("✓ Architecture Test Passed.")
        except Exception as e:
            print(f"✗ Architecture Test Failed: {e}")
            import traceback

            traceback.print_exc()
            import traceback

            traceback.print_exc()
    else:
        config = get_config(args.mode)
        update_config(config, args)
        train(config)
