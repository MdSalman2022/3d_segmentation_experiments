"""
MedDINO-VISTA3D: IMPROVED Implementation
========================================

Fixes applied:
1. True VISTA3D-style Transformer Decoder (Cross-Attention Class Queries).
2. Aggressive sampling for tumors (100% probability).
3. Higher loss weights for small classes (Tumor/Cyst = 15.0).
4. Learning Rate Scheduler.
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

# Fix deprecated warning
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["XFORMERS_DISABLED"] = "1"

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Paths
    "kits23_dir": "./kits23/dataset",
    "output_dir": "./output/meddino_vista3d_v2",
    
    # Model
    "num_classes": 4,  # background, kidney, tumor, cyst
    "backbone": "dinov2_vitb14",
    
    # Training
    "batch_size": 1,
    "num_epochs_stage1": 5,  # Increased from 3 to see effects
    "num_epochs_stage2": 3,  # Increased from 2
    "lr_stage1": 1e-4,
    "lr_stage2": 1e-5,
    "weight_decay": 1e-5,
    
    # Data
    "patch_size": (140, 224, 224),
    "num_samples_per_volume": 4,
    "val_split": 0.2,
    
    # Quick test mode
    "quick_cases": 20,
    
    # Labels
    "labels": {
        "background": 0,
        "kidney": 1,
        "tumor": 2,
        "cyst": 3
    }
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MedDINOEncoder(nn.Module):
    """MedDINOv3 Encoder (Unchanged logic, keeping stability)"""
    def __init__(self, backbone: str = "dinov2_vitb14", feature_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        print(f"Loading {backbone} backbone...")
        self.vit = torch.hub.load('facebookresearch/dinov2', backbone, pretrained=True, verbose=False)
        self.hidden_dim = 768
        
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
                
        self.scale_layers = [2, 5, 8, 11]
        self.proj_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(self.hidden_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
            for _ in self.scale_layers
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.depth_aggregator = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x = x.squeeze(1).view(B * D, 1, H, W).repeat(1, 3, 1, 1)
        
        # Extract features (No grad if frozen)
        with torch.set_grad_enabled(not next(self.vit.parameters()).requires_grad):
            features = self.vit.get_intermediate_layers(x, n=self.scale_layers, reshape=True)
        
        # Project & Fuse
        h, w = features[0].shape[2], features[0].shape[3]
        multi_scale = []
        for i, feat in enumerate(features):
            if feat.shape[2:] != (h, w):
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B * D * h * w, self.hidden_dim)
            proj = self.proj_heads[i](feat_flat).reshape(B * D, h, w, -1).permute(0, 3, 1, 2)
            multi_scale.append(proj)
            
        fused = self.fusion(torch.cat(multi_scale, dim=1))
        fused_3d = fused.reshape(B, D, -1, h, w).permute(0, 2, 1, 3, 4)
        return self.depth_aggregator(fused_3d)


class VISTA3DTransformerDecoder(nn.Module):
    """
    True VISTA3D-style Decoder using Transformer Cross-Attention.
    """
    def __init__(
        self, 
        in_channels: int = 256, 
        num_classes: int = 4,
        num_queries: int = 4  # Usually equals num_classes (excluding bg or including)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # 1. Upsample features to a reasonable resolution first
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True)
        ) # Output is now roughly (D, H/4, W/4)
        
        # 2. Learnable Class Queries (The VISTA innovation)
        # We treat Kidney, Tumor, Cyst as queries. Background is implicit.
        self.class_queries = nn.Embedding(num_classes - 1, in_channels) 
        
        # 3. Transformer Cross Attention
        # Batch First=True expects input (Batch, Seq_Len, Embed_Dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_channels, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        
        # 4. Feed Forward Network for queries
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        
        # 5. Final projection to segmentation map
        # Projects query features back to spatial map
        self.map_head = nn.Linear(in_channels, 1) 

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, C, D, H, W)
        Returns: (B, num_classes, D', H', W')
        """
        B, C, D, H, W = features.shape
        
        # 1. Upsample features to reduce computational cost for attention
        # and capture finer details. 
        x = self.upsample(features) # (B, C, D, H*4, W*4)
        _, _, D_u, H_u, W_u = x.shape
        
        # 2. Prepare features for Cross-Attention (as Keys and Values)
        # Reshape to sequence: (B, Sequence_Len, C)
        feat_seq = x.view(B, C, -1).permute(0, 2, 1) # (B, D*H*W, C)
        
        # 3. Prepare Queries (Class Embeddings)
        # Repeat class queries for each batch
        # queries shape: (B, num_classes-1, C)
        queries = self.class_queries.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # 4. Cross Attention: Queries attend to Feature Map
        # attn_output: (B, Num_Classes, C)
        attn_out, _ = self.cross_attn(
            query=queries, 
            key=feat_seq, 
            value=feat_seq
        )
        
        # 5. FFN & Residual Connection
        queries = queries + self.norm1(attn_out)
        queries = queries + self.norm2(self.ffn(queries))
        
        # 6. Generate Masks
        # Calculate similarity between enhanced queries and original features
        # (B, Num_Classes, C) @ (B, C, D*H*W) -> (B, Num_Classes, D*H*W)
        masks = torch.bmm(queries, feat_seq.permute(0, 2, 1)) 
        
        # Reshape to spatial dimensions
        masks = masks.view(B, self.num_classes - 1, D_u, H_u, W_u)
        
        # 7. Upsample to original size and append Background
        masks = F.interpolate(masks, size=features.shape[2:], mode='trilinear', align_corners=False)
        
        # Background channel (zeros or learned bias) - implicitly handled by argmax vs 0 if needed,
        # but typically we create a channel.
        bg_map = torch.zeros(B, 1, *masks.shape[2:]).to(masks.device)
        logits = torch.cat([bg_map, masks], dim=1)
        
        return logits


class MedDINOVISTA3D(nn.Module):
    def __init__(
        self, 
        num_classes: int = 4,
        freeze_encoder: bool = True,
        backbone: str = "dinov2_vitb14"
    ):
        super().__init__()
        self.encoder = MedDINOEncoder(
            backbone=backbone,
            feature_dim=256,
            freeze_backbone=freeze_encoder
        )
        
        # Use the new Transformer Decoder
        self.decoder = VISTA3DTransformerDecoder(
            in_channels=256,
            num_classes=num_classes
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.decoder(features)
        
        # Resize to exact input size if slightly off due to convolution padding
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='trilinear', align_corners=False)
            
        return logits
    
    def unfreeze_encoder(self, num_layers: int = 4):
        params = list(self.encoder.vit.parameters())
        for param in params[-num_layers:]:
            param.requires_grad = True
        print(f"Unfroze last {num_layers} layers of encoder")

# ============================================================================
# DATASET (Improved Sampling)
# ============================================================================

class KiTS23Dataset(Dataset):
    def __init__(self, data_dicts: List[Dict], patch_size: Tuple[int, int, int], 
                 num_samples: int = 2, is_train: bool = True):
        self.data_dicts = data_dicts
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data_dicts) * self.num_samples
    
    def __getitem__(self, idx):
        vol_idx = idx // self.num_samples
        data_dict = self.data_dicts[vol_idx]
        
        image = nib.load(data_dict["image"]).get_fdata()
        label = nib.load(data_dict["label"]).get_fdata()
        
        # Normalize
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        patch_img, patch_lbl = self._sample_patch(image, label)
        
        patch_img = torch.from_numpy(patch_img).float().unsqueeze(0)
        patch_lbl = torch.from_numpy(patch_lbl).long().unsqueeze(0)
        return {"image": patch_img, "label": patch_lbl}
    
    def _sample_patch(self, image, label):
        """Aggressive sampling: Force tumor/cyst centering if present"""
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # Pad if needed
        if d < pd or h < ph or w < pw:
            pad_d = max(0, pd - d)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            d, h, w = image.shape
            
        if self.is_train:
            # IMPROVEMENT: 100% chance to center on Tumor or Cyst if they exist
            target_indices = np.argwhere((label == 2) | (label == 3))
            if len(target_indices) > 0:
                center = target_indices[np.random.randint(len(target_indices))]
                d_start = np.clip(center[0] - pd // 2, 0, d - pd)
                h_start = np.clip(center[1] - ph // 2, 0, h - ph)
                w_start = np.clip(center[2] - pw // 2, 0, w - pw)
            else:
                # Fallback to Kidney if no tumor
                kidney_indices = np.argwhere(label == 1)
                if len(kidney_indices) > 0:
                    center = kidney_indices[np.random.randint(len(kidney_indices))]
                    d_start = np.clip(center[0] - pd // 2, 0, d - pd)
                    h_start = np.clip(center[1] - ph // 2, 0, h - ph)
                    w_start = np.clip(center[2] - pw // 2, 0, w - pw)
                else:
                    # Random
                    d_start = np.random.randint(0, max(1, d - pd + 1))
                    h_start = np.random.randint(0, max(1, h - ph + 1))
                    w_start = np.random.randint(0, max(1, w - pw + 1))
        else:
            # Center crop for validation
            d_start = (d - pd) // 2
            h_start = (h - ph) // 2
            w_start = (w - pw) // 2
        
        return image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw], \
               label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]


def get_dataloaders(config, max_cases=None):
    data_dir = Path(config["kits23_dir"])
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    
    if max_cases:
        cases = cases[:max_cases]
    
    data_dicts = []
    for case in cases:
        img_path = case / "imaging.nii.gz"
        seg_path = case / "segmentation.nii.gz"
        if img_path.exists() and seg_path.exists():
            data_dicts.append({"image": str(img_path), "label": str(seg_path)})
    
    val_size = int(len(data_dicts) * config["val_split"])
    train_files = data_dicts[val_size:]
    val_files = data_dicts[:val_size]
    
    train_ds = KiTS23Dataset(train_files, config["patch_size"], config["num_samples_per_volume"], True)
    val_ds = KiTS23Dataset(val_files, config["patch_size"], 1, False)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

# ============================================================================
# LOSS FUNCTION (Improved Weights)
# ============================================================================

class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # IMPROVEMENT: Significantly higher weight for Tumor (2) and Cyst (3)
        # Tumors are tiny, need massive penalty for ignoring them.
        class_weights = torch.tensor([0.1, 1.0, 15.0, 15.0]) 
        self.register_buffer('class_weights', class_weights)
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.squeeze(1).long(), num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_onehot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
        dice = (2. * intersection + smooth) / (union + smooth)
        
        weights = self.class_weights.to(pred.device)
        weighted_dice = (dice * weights.unsqueeze(0)).sum() / weights.sum()
        return 1 - weighted_dice
    
    def forward(self, outputs, target):
        # CE
        ce_loss = F.cross_entropy(outputs, target.squeeze(1), weight=self.class_weights)
        # Dice
        dice_loss = self.dice_loss(outputs, target)
        return ce_loss + dice_loss

# ============================================================================
# TRAINING LOOP
# ============================================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def compute_dice(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    dice_per_class = []
    for c in range(1, 4): 
        pred_c = (pred == c)
        target_c = (target == c)
        if target_c.sum() == 0 and pred_c.sum() == 0:
            dice_per_class.append(1.0)
        elif target_c.sum() == 0:
            dice_per_class.append(0.0)
        else:
            intersection = (pred_c & target_c).sum()
            dice = 2 * intersection / (pred_c.sum() + target_c.sum())
            dice_per_class.append(dice)
    return dice_per_class

def train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    epoch_loss = 0
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return epoch_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0
    dice_scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            
            pred = torch.argmax(outputs, dim=1, keepdim=True)
            dice = compute_dice(pred, labels)
            dice_scores.append(dice)
            
    avg_per_class = np.mean(dice_scores, axis=0)
    mean_dice = np.mean(avg_per_class)
    return val_loss / len(loader), mean_dice, avg_per_class

def train(config, stage1_epochs, stage2_epochs, max_cases=None):
    print("=" * 60)
    print("MedDINO-VISTA3D V2 Training (Improved)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader = get_dataloaders(config, max_cases)
    
    model = MedDINOVISTA3D(
        num_classes=config["num_classes"],
        freeze_encoder=True,
        backbone=config["backbone"]
    ).to(device)
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable: {trainable:,}")
    
    loss_fn = HybridLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_dice = 0
    
    # Stage 1
    print("\nSTAGE 1: Frozen Encoder")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_stage1"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs, eta_min=1e-6)
    
    for epoch in range(stage1_epochs):
        print(f"\nEpoch {epoch+1}/{stage1_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        val_loss, val_dice, class_dices = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Dice: Mean: {val_dice:.4f} (Kidney: {class_dices[0]:.4f}, Tumor: {class_dices[1]:.4f}, Cyst: {class_dices[2]:.4f})")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), output_dir / "best_model_stage1.pth")
            print(f"Saved best model (Dice: {best_dice:.4f})")
            
    # Stage 2
    if stage2_epochs > 0:
        print("\nSTAGE 2: Fine-tuning Encoder")
        model.load_state_dict(torch.load(output_dir / "best_model_stage1.pth"))
        model.unfreeze_encoder(num_layers=4)
        
        total, trainable = count_parameters(model)
        print(f"Trainable (Stage 2): {trainable:,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_stage2"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-7)
        
        for epoch in range(stage2_epochs):
            print(f"\nEpoch {epoch+1}/{stage2_epochs}")
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
            val_loss, val_dice, class_dices = validate(model, val_loader, loss_fn, device)
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Dice: Mean: {val_dice:.4f} (Kidney: {class_dices[0]:.4f}, Tumor: {class_dices[1]:.4f}, Cyst: {class_dices[2]:.4f})")

    print(f"\nTraining Complete. Best Dice: {best_dice:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test", choices=["test", "quick_test", "full"])
    args = parser.parse_args()
    
    if args.mode == "test":
        print("Testing V2 Architecture...")
        # Simplified test
        model = MedDINOVISTA3D().cuda()
        x = torch.randn(1, 1, 32, 128, 128).cuda()
        y = model(x)
        print(f"Input: {x.shape}, Output: {y.shape}")
    elif args.mode == "quick_test":
        train(CONFIG, stage1_epochs=5, stage2_epochs=3, max_cases=CONFIG["quick_cases"])
    else:
        train(CONFIG, stage1_epochs=100, stage2_epochs=50, max_cases=None)

if __name__ == "__main__":
    main() 