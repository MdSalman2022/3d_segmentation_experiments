"""
MedDINO-VISTA3D Enhanced V2
============================

PHASE 1.5 ENHANCEMENTS:
- Deep Supervision: Auxiliary losses at intermediate decoder outputs
- Post-Processing: Connected component filtering (KiTS23 style)
- All Phase 1 features preserved (skip connections, aggressive cyst loss)

Architecture:
- Encoder: DINOv2 (ViT-B/14) with multi-scale features
- Decoder: U-Net with deep supervision at each upsampling stage
- Loss: Multi-level Tversky + Focal + Dice
- Post-Processing: Morphological + connected component cleanup

Expected Results:
- Cyst Dice: 0.10-0.15 (vs 0.064 in V1)
- Better gradient flow via deep supervision
- Higher precision via post-processing

Usage:
    python meddino_vista3d_enhanced_v2.py --mode quick_test
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm
from einops import rearrange
from scipy import ndimage

# Settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["XFORMERS_DISABLED"] = "1"

# ============================================================================
# CONFIGURATION
# ============================================================================

def get_config(mode):
    base = {
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/meddino_vista3d_enhanced_v2",
        "dinov2_backbone": "dinov2_vitb14",
        "hf_token": None,
        "num_classes": 4,
        "batch_size": 2,
        "val_split": 0.2,
        "num_samples_per_volume": 2,
        "patch_size": (140, 224, 224),
        "lr_stage1": 1e-4,
        "lr_stage2": 1e-5,
        "weight_decay": 1e-5,
        "use_deep_supervision": True,  # NEW
        "use_postprocessing": True,    # NEW
        "min_component_size": 50,      # For post-processing
    }

    if mode == "quick_test":
        base.update({
            "num_epochs_stage1": 15,
            "num_epochs_stage2": 10,
            "quick_cases": 100,
            "use_warmup": False,
            "patience_stage1": 3,
            "patience_stage2": 3,
        })
    else:
        base.update({
            "num_epochs_stage1": 100,
            "num_epochs_stage2": 50,
            "quick_cases": None,
            "use_warmup": True,
            "warmup_epochs": 10,
            "patience_stage1": 15,
            "patience_stage2": 10,
        })
    return base


# ============================================================================
# ENCODER (SAME AS V1)
# ============================================================================

class EnhancedMedDINOEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        feature_dim: int = 256,
        freeze_backbone: bool = True,
        token: str = None,
    ):
        super().__init__()
        print(f"Loading Enhanced DINOv2 Encoder: {model_name}...")
        
        self.vit = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True, verbose=False
        )
        self.hidden_dim = 768
        
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        self.scale_layers = [8, 11]
        self.fusion_layers = [2, 5, 8, 11]
        
        self.proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
            )
            for _ in range(4)
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        self.depth_aggregator = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        print(f"✓ Enhanced encoder initialized with {len(self.scale_layers)} skip connection levels")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, D, H, W = x.shape
        x_2d = x.squeeze(1).view(B * D, 1, H, W).repeat(1, 3, 1, 1)
        
        with torch.set_grad_enabled(not next(self.vit.parameters()).requires_grad):
            skip_features_2d = [
                self.vit.get_intermediate_layers(x_2d, n=[i])[0] for i in self.scale_layers
            ]
            fusion_features_2d = [
                self.vit.get_intermediate_layers(x_2d, n=[i])[0] for i in self.fusion_layers
            ]
        
        h, w = H // 14, W // 14
        
        skip_features_3d = []
        for i, feat in enumerate(skip_features_2d):
            num_patches = feat.shape[1]
            patch_h = patch_w = int(num_patches ** 0.5)
            feat_2d = feat.permute(0, 2, 1).reshape(feat.shape[0], self.hidden_dim, patch_h, patch_w)
            feat_2d = F.interpolate(feat_2d, size=(h, w), mode="bilinear", align_corners=False)
            feat_proj = self.proj_heads[i](feat_2d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            feat_3d = rearrange(feat_proj, '(b d) c h w -> b c d h w', b=B, d=D)
            skip_features_3d.append(feat_3d)
        
        multi_scale_2d = []
        for i, feat in enumerate(fusion_features_2d):
            num_patches = feat.shape[1]
            patch_h = patch_w = int(num_patches ** 0.5)
            feat_2d = feat.permute(0, 2, 1).reshape(feat.shape[0], self.hidden_dim, patch_h, patch_w)
            feat_2d = F.interpolate(feat_2d, size=(h, w), mode="bilinear", align_corners=False)
            feat_proj = self.proj_heads[i](feat_2d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            multi_scale_2d.append(feat_proj)
        
        fused_2d = self.fusion(torch.cat(multi_scale_2d, dim=1))
        fused_3d = rearrange(fused_2d, '(b d) c h w -> b c d h w', b=B, d=D)
        bottleneck = self.depth_aggregator(fused_3d)
        
        return {
            'bottleneck': bottleneck,
            'skip1': skip_features_3d[0],
            'skip2': skip_features_3d[1],
        }


# ============================================================================
# SPATIAL ATTENTION
# ============================================================================

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


# ============================================================================
# DECODER WITH DEEP SUPERVISION
# ============================================================================

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, use_attention=True):
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        
        self.skip_conv = nn.Conv3d(skip_channels, out_channels, 1)
        
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.attention = SpatialAttention3D() if use_attention else nn.Identity()
    
    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.skip_conv(skip)
        
        if x.shape != skip.shape:
            skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        
        return x


class DeepSupervisedDecoder(nn.Module):
    """
    Decoder with DEEP SUPERVISION at each upsampling stage.
    Forces early layers to learn meaningful features via auxiliary losses.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 4,
        decoder_channels: List[int] = [128, 64],
        dropout: float = 0.1,
        use_deep_supervision: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_deep_supervision = use_deep_supervision
        
        # Main decoder path
        self.up1 = UpBlock(in_channels, decoder_channels[0], skip_channels=256, use_attention=True)
        self.up2 = UpBlock(decoder_channels[0], decoder_channels[1], skip_channels=256, use_attention=True)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(decoder_channels[1], 32, 2, stride=2),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv3d(32, num_classes, 1)
        
        # Deep supervision heads (auxiliary classifiers)
        if use_deep_supervision:
            # After up1 (1/4 resolution)
            self.aux_head1 = nn.Sequential(
                nn.Conv3d(decoder_channels[0], num_classes, 1),
            )
            # After up2 (1/2 resolution)
            self.aux_head2 = nn.Sequential(
                nn.Conv3d(decoder_channels[1], num_classes, 1),
            )
        
        print(f"✓ Deep-supervised decoder initialized: {decoder_channels}")
        print(f"  Deep Supervision: {'ENABLED' if use_deep_supervision else 'DISABLED'}")
    
    def forward(self, encoder_outputs: Dict[str, torch. Tensor], return_aux: bool = False) -> Dict[str, torch.Tensor]:
        x = encoder_outputs['bottleneck']
        
        # Up1 with skip
        x1 = self.up1(x, encoder_outputs['skip2'])  # Layer 11
        
        # Up2 with skip
        x2 = self.up2(x1, encoder_outputs['skip1'])  # Layer 8
        
        # Final
        x_final = self.final_up(x2)
        logits = self.final(x_final)
        
        outputs = {'main': logits}
        
        # Deep supervision outputs (for training only)
        if self.use_deep_supervision and return_aux:
            outputs['aux1'] = self.aux_head1(x1)  # 1/4 resolution
            outputs['aux2'] = self.aux_head2(x2)  # 1/2 resolution
        
        return outputs


# ============================================================================
# POST-PROCESSING MODULE
# ============================================================================

class PostProcessor:
    """
    Connected component filtering and morphological operations.
    Inspired by KiTS23 2nd place solution.
    """
    
    def __init__(self, min_component_size: int = 50):
        self.min_component_size = min_component_size
    
    def __call__(self, pred: np.ndarray) -> np.ndarray:
        """
        Args:
            pred: (D, H, W) prediction mask with class labels
        Returns:
            cleaned: (D, H, W) cleaned mask
        """
        cleaned = np.zeros_like(pred)
        
        # Process each class separately
        for class_id in [1, 2, 3]:  # Kidney, Tumor, Cyst
            class_mask = (pred == class_id).astype(np.uint8)
            
            if class_mask.sum() == 0:
                continue
            
            # Morphological closing to fill small holes
            if class_id == 3:  # Extra care for cysts
                class_mask = ndimage.binary_closing(class_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
            
            # Connected component analysis
            labeled, num_components = ndimage.label(class_mask)
            
            if num_components == 0:
                continue
            
            # Keep only large enough components
            for component_id in range(1, num_components + 1):
                component_mask = (labeled == component_id)
                component_size = component_mask.sum()
                
                # Adaptive threshold: smaller for cysts
                min_size = self.min_component_size // 2 if class_id == 3 else self.min_component_size
                
                if component_size >= min_size:
                    cleaned[component_mask] = class_id
        
        return cleaned


# ============================================================================
# COMPLETE MODEL
# ============================================================================

class MedDINOVISTA3DEnhancedV2(nn.Module):
    """Enhanced MedDINO-VISTA3D with Deep Supervision + Post-Processing"""
    
    def __init__(
        self,
        num_classes=4,
        freeze_encoder=True,
        dropout=0.1,
        dinov2_backbone="dinov2_vitb14",
        token=None,
        use_deep_supervision=True,
        use_postprocessing=True,
        min_component_size=50,
    ):
        super().__init__()
        self.encoder = EnhancedMedDINOEncoder(
            model_name=dinov2_backbone,
            freeze_backbone=freeze_encoder,
            token=token
        )
        self.decoder = DeepSupervisedDecoder(
            num_classes=num_classes,
            dropout=dropout,
            use_deep_supervision=use_deep_supervision,
        )
        self.use_postprocessing = use_postprocessing
        if use_postprocessing:
            self.postprocessor = PostProcessor(min_component_size=min_component_size)
    
    def forward(self, x, return_aux=False):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out, return_aux=return_aux)
        
        # Resize main output to input size
        if decoder_out['main'].shape[2:] != x.shape[2:]:
            decoder_out['main'] = F.interpolate(
                decoder_out['main'], size=x.shape[2:], mode="trilinear", align_corners=False
            )
        
        # Resize auxiliary outputs if present
        if return_aux and 'aux1' in decoder_out:
            decoder_out['aux1'] = F.interpolate(
                decoder_out['aux1'], size=x.shape[2:], mode="trilinear", align_corners=False
            )
            decoder_out['aux2'] = F.interpolate(
                decoder_out['aux2'], size=x.shape[2:], mode="trilinear", align_corners=False
            )
        
        return decoder_out
    
    def predict_with_postprocessing(self, x):
        """Inference with post-processing"""
        with torch.no_grad():
            outputs = self.forward(x, return_aux=False)
            preds = torch.argmax(outputs['main'], dim=1)  # (B, D, H, W)
            
            if self.use_postprocessing:
                # Apply post-processing to each sample in batch
                cleaned_preds = []
                for pred in preds.cpu().numpy():
                    cleaned = self.postprocessor(pred)
                    cleaned_preds.append(cleaned)
                preds = torch.from_numpy(np.stack(cleaned_preds)).to(preds.device)
            
            return preds
    
    def unfreeze_encoder(self, num_layers=4):
        params = list(self.encoder.vit.parameters())
        for param in params[-num_layers:]:
            param.requires_grad = True
        print(f"Unfroze last {num_layers} layers of encoder")


# ============================================================================
# LOSS FUNCTIONS WITH DEEP SUPERVISION
# ============================================================================

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.15, beta=0.85, smooth=1e-6):
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
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if isinstance(alpha, (float, int, list)):
            alpha = torch.tensor(alpha)
        
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = alpha
        
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets.squeeze(1), weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DeepSupervisedHybridLoss(nn.Module):
    """
    Multi-level loss with deep supervision.
    Combines outputs from multiple decoder levels.
    """
    
    def __init__(self, use_deep_supervision=True):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision
        
        # Class weights: BG, Kidney, Tumor, Cyst
        w = torch.tensor([0.1, 1.0, 12.0, 20.0])
        self.register_buffer("class_weights", w)
        
        self.tversky = TverskyLoss(alpha=0.15, beta=0.85)
        self.focal = FocalLoss(alpha=w, gamma=2.5)
    
    def forward(self, outputs_dict, target):
        """
        Args:
            outputs_dict: {'main': logits, 'aux1': aux_logits1, 'aux2': aux_logits2}
            target: ground truth
        """
        # Main loss
        main_logits = outputs_dict['main']
        tv = self.tversky(main_logits, target)
        fc = self.focal(main_logits, target)
        
        # Dice
        pred_soft = F.softmax(main_logits, dim=1)
        target_oh = F.one_hot(target.squeeze(1).long(), num_classes=main_logits.shape[1])
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()
        
        dice_per_class = 2 * (pred_soft * target_oh).sum(dim=(2,3,4)) / \
                         ((pred_soft + target_oh).sum(dim=(2,3,4)) + 1e-6)
        dice_loss = 1 - dice_per_class.mean()
        
        main_loss = 0.5 * tv + 0.3 * fc + 0.2 * dice_loss
        
        # Deep supervision losses (weighted lower)
        if self.use_deep_supervision and 'aux1' in outputs_dict:
            aux1_loss = 0.3 * self.tversky(outputs_dict['aux1'], target) + \
                        0.2 * self.focal(outputs_dict['aux1'], target)
            aux2_loss = 0.3 * self.tversky(outputs_dict['aux2'], target) + \
                        0.2 * self.focal(outputs_dict['aux2'], target)
            
            # Combine: 70% main + 20% aux2 + 10% aux1
            total_loss = 0.7 * main_loss + 0.2 * aux2_loss + 0.1 * aux1_loss
            return total_loss
        else:
            return main_loss


# ============================================================================
# DATASET - Cyst-Focused Sampling
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
            img, lbl = np.pad(img, pad, mode="constant"), np.pad(lbl, pad, mode="constant")
            d, h, w = img.shape
        
        if self.is_train:
            # PHASE 1: 50% Cyst, 30% Tumor, 10% Kidney, 10% Random
            rand_val = np.random.random()
            
            target_class = None
            if rand_val < 0.5:
                target_class = 3  # Cyst (50%!)
            elif rand_val < 0.8:
                target_class = 2  # Tumor (30%)
            elif rand_val < 0.9:
                target_class = 1  # Kidney (10%)
            
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
                    # Fallback
                    fallback_indices = np.argwhere((lbl == 2) | (lbl == 3))
                    if len(fallback_indices) > 0:
                        c = fallback_indices[np.random.randint(len(fallback_indices))]
                        ds, hs, ws = (
                            np.clip(c[0] - pd // 2, 0, d - pd),
                            np.clip(c[1] - ph // 2, 0, h - ph),
                            np.clip(c[2] - pw // 2, 0, w - pw),
                        )
                    else:
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
                                np.random.randint(0, max(1, d - pd + 1)),
                                np.random.randint(0, max(1, h - ph + 1)),
                                np.random.randint(0, max(1, w - pw + 1)),
                            )
            else:
                # 10% Random
                ds, hs, ws = (
                    np.random.randint(0, max(1, d - pd + 1)),
                    np.random.randint(0, max(1, h - ph + 1)),
                    np.random.randint(0, max(1, w - pw + 1)),
                )
        else:
            ds, hs, ws = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2
        
        return (
            img[ds:ds + pd, hs:hs + ph, ws:ws + pw],
            lbl[ds:ds + pd, hs:hs + ph, ws:ws + pw],
        )


def get_dataloaders(config, max_cases=None):
    data_dir = Path(config["kits23_dir"])
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    if max_cases:
        cases = cases[:max_cases]
    
    data_dicts = [
        {"image": str(c / "imaging.nii.gz"), "label": str(c / "segmentation.nii.gz")}
        for c in cases if (c / "imaging.nii.gz").exists()
    ]
    
    split = int(len(data_dicts) * config["val_split"])
    train_ds = KiTS23Dataset(data_dicts[split:], config["patch_size"], config["num_samples_per_volume"], True)
    val_ds = KiTS23Dataset(data_dicts[:split], config["patch_size"], 1, False)
    
    return (
        DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True),
    )


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
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
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class MetricTracker:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
    
    def update(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        mask = (targets >= 0) & (targets < self.num_classes)
        self.confusion_matrix += torch.bincount(
            self.num_classes * targets[mask].long() + preds[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
    
    def compute(self):
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(0) - tp
        fn = self.confusion_matrix.sum(1) - tp
        
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
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
        
        mean_dice = np.mean(metrics["dice"][1:])
        mean_iou = np.mean(metrics["iou"][1:])
        
        s = f"\n  Mean Dice (Fg): {mean_dice:.4f} | Mean IoU (Fg): {mean_iou:.4f}\n"
        s += f"  {'Class':<10} {'Dice':<8} {'IoU':<8} {'Prec':<8} {'Recall':<8}\n"
        s += "-" * 46 + "\n"
        for i, c in enumerate(classes):
            s += f"  {c:<10} {metrics['dice'][i]:.4f}   {metrics['iou'][i]:.4f}   {metrics['precision'][i]:.4f}   {metrics['recall'][i]:.4f}\n"
        return s, mean_dice


# ============================================================================
# TRAINING FUNCTIONS (MODIFIED FOR DEEP SUPERVISION)
# ============================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, use_deep_supervision):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        try:
            img, lbl = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(img, return_aux=use_deep_supervision)
                loss = loss_fn(outputs, lbl)
            
            if not torch.isfinite(loss):
                print(f"\n⚠ Warning: Non-finite loss at batch {batch_idx}, skipping...")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                del img, lbl, outputs, loss
                torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"\n⚠ Error at batch {batch_idx}: {e}")
            torch.cuda.empty_cache()
            continue
    
    return epoch_loss / max(batch_count, 1)


def validate(model, loader, loss_fn, device, use_postprocessing):
    model.eval()
    val_loss = 0
    tracker = MetricTracker(num_classes=4)
    tracker_postproc = MetricTracker(num_classes=4) if use_postprocessing else None
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
            try:
                img, lbl = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
                
                with torch.amp.autocast("cuda", enabled=True):
                    outputs = model(img, return_aux=False)
                    loss = loss_fn(outputs, lbl)
                    
                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        batch_count += 1
                
                # Standard prediction
                preds = torch.argmax(outputs['main'], dim=1)
                tracker.update(preds, lbl.squeeze(1))
                
                # Post-processed prediction
                if use_postprocessing:
                    preds_postproc = model.predict_with_postprocessing(img)
                    tracker_postproc.update(preds_postproc, lbl.squeeze(1))
                
                if batch_idx % 5 == 0:
                    del img, lbl, outputs, loss, preds
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"\n⚠ Validation error at batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                continue
    
    report, mean_dice = tracker.format_results()
    
    if use_postprocessing:
        report_postproc, mean_dice_postproc = tracker_postproc.format_results()
        print("\n📊 WITH Post-Processing:")
        print(report_postproc)
        return val_loss / max(batch_count, 1), mean_dice_postproc, report
    else:
        return val_loss / max(batch_count, 1), mean_dice, report


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(config):
    print("=" * 70)
    print("MedDINO-VISTA3D ENHANCED V2 (Phase 1.5)")
    print("Features: Deep Supervision + Post-Processing + All Phase 1 enhancements")
    print(f"Encoder: {config['dinov2_backbone']}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    train_loader, val_loader = get_dataloaders(config, max_cases=config.get("quick_cases"))
    
    # Model
    model = MedDINOVISTA3DEnhancedV2(
        num_classes=config["num_classes"],
        freeze_encoder=True,
        dropout=0.1,
        dinov2_backbone=config["dinov2_backbone"],
        token=config.get("hf_token"),
        use_deep_supervision=config["use_deep_supervision"],
        use_postprocessing=config["use_postprocessing"],
        min_component_size=config["min_component_size"],
    ).to(device)
    print(f"Total Params: {sum(p.numel() for p in model.parameters()):,}")
    
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = DeepSupervisedHybridLoss(use_deep_supervision=config["use_deep_supervision"]).to(device)
    
    best_dice = 0.0
    
    def get_scheduler(optimizer, epochs, is_stage2=False):
        if not config.get("use_warmup") or is_stage2:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        else:
            warmup_iters = config["warmup_epochs"]
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
            main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_iters, eta_min=1e-6)
            return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, main], [warmup_iters])
    
    # STAGE 1
    print("\n--- STAGE 1: Frozen Encoder ---")
    torch.cuda.empty_cache()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_stage1"], weight_decay=config["weight_decay"])
    scheduler = get_scheduler(optimizer, config["num_epochs_stage1"], is_stage2=False)
    early_stopping = EarlyStopping(patience=config["patience_stage1"], verbose=True, path=str(output_dir / "best_model_stage1.pth"))
    
    for epoch in range(config["num_epochs_stage1"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs_stage1']} - Starting training...")
        
        try:
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, config["use_deep_supervision"])
            print(f"Training complete. Validating...")
            val_loss, mean_dice, report = validate(model, val_loader, loss_fn, device, config["use_postprocessing"])
            scheduler.step()
        except Exception as e:
            print(f"\n❌ Error in epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
        
        print(f"Ep {epoch+1}/{config['num_epochs_stage1']} | Loss: {train_loss:.4f}/{val_loss:.4f} | Dice: {mean_dice:.4f}")
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_stage2"], weight_decay=config["weight_decay"])
        scheduler = get_scheduler(optimizer, config["num_epochs_stage2"], is_stage2=True)
        early_stopping = EarlyStopping(patience=config["patience_stage2"], verbose=True, path=str(output_dir / "best_model_final.pth"))
        
        for epoch in range(config["num_epochs_stage2"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, config["use_deep_supervision"])
            val_loss, mean_dice, report = validate(model, val_loader, loss_fn, device, config["use_postprocessing"])
            scheduler.step()
            
            print(f"Ep {epoch+1}/{config['num_epochs_stage2']} | Loss: {train_loss:.4f}/{val_loss:.4f} | Dice: {mean_dice:.4f}")
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
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full", choices=["quick_test", "full"])
    args = parser.parse_args()
    
    config = get_config(args.mode)
    train(config)
