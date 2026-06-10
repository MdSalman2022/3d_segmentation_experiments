"""
MedDINO-VISTA3D: Final Production Version
==========================================

Architecture:
- Encoder: MedDINO (Frozen Fine-tuned)
- Decoder: VISTA3D Transformer (Cross-Attention + Class Queries)
- Loss: Tversky + CrossEntropy (Focus on small tumors)

Features:
- Full Training Mode: Includes Warmup Scheduler for Transformer stability.
- Quick Test Mode: 3 epochs, skips warmup for speed.
- API: Uses torch.amp (PyTorch 2.1+).

Usage:
    # Quick Test (Verifies pipeline, 3 epochs)
    python meddino_vista3d_final.py --mode quick_test

    # Full Training (100+50 epochs with Warmup)
    python meddino_vista3d_final.py --mode full
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
        "output_dir": "./output/meddino_vista3d_final",
        "num_classes": 4,
        "backbone": "dinov2_vitb14",
        "batch_size": 2,
        "val_split": 0.2,
        "num_samples_per_volume": 2,
        "patch_size": (140, 224, 224),
        "lr_stage1": 1e-4,
        "lr_stage2": 1e-5,
        "weight_decay": 1e-5,
    }

    if mode == "quick_test":
        base.update({
            "num_epochs_stage1": 3,
            "num_epochs_stage2": 2,
            "quick_cases": 20,
            "use_warmup": False,
            "patience_stage1": 2,
            "patience_stage2": 2
        })
    else:
        base.update({
            "num_epochs_stage1": 50,
            "num_epochs_stage2": 25,
            "quick_cases": None,
            "use_warmup": True,
            "warmup_epochs": 10,
            "patience_stage1": 15,
            "patience_stage2": 10
        })
    return base

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MedDINOEncoder(nn.Module):
    """MedDINOv3 Encoder"""
    def __init__(self, backbone: str = "dinov2_vitb14", feature_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        print(f"Loading {backbone}...")
        self.vit = torch.hub.load('facebookresearch/dinov2', backbone, pretrained=True, verbose=False)
        self.hidden_dim = 768
        
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
                
        # Multi-scale projections
        self.scale_layers = [2, 5, 8, 11]
        self.proj_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(self.hidden_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
            for _ in self.scale_layers
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 3D Aggregation
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
        # Prepare for ViT
        x = x.squeeze(1).view(B * D, 1, H, W).repeat(1, 3, 1, 1)
        
        # Extract features
        with torch.set_grad_enabled(not next(self.vit.parameters()).requires_grad):
            features = self.vit.get_intermediate_layers(x, n=self.scale_layers, reshape=True)
        
        # Project and Fuse
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
    VISTA3D-style Decoder with Cross-Attention and Class Queries.
    Includes Dropout for regularization in full training.
    """
    def __init__(self, in_channels: int = 256, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout
        
        # 1. Upsample features to reduce memory cost before Attention
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True)
        ) # Output: (B, C, D, H/4, W/4)
        
        # 2. Learnable Class Queries
        self.class_queries = nn.Embedding(num_classes - 1, in_channels)
        
        # 3. Transformer Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_channels, 
            num_heads=8, 
            batch_first=True,
            dropout=dropout  # Regularization
        )
        
        # 4. Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout), # Regularization
            nn.Linear(in_channels * 4, in_channels)
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
        feat_seq = x.view(B, C, -1).permute(0, 2, 1) # (B, Seq_Len, C)
        
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
        masks = F.interpolate(masks, size=features.shape[2:], mode='trilinear', align_corners=False)
        bg_map = torch.zeros(B, 1, *masks.shape[2:]).to(masks.device)
        logits = torch.cat([bg_map, masks], dim=1)
        
        return logits


class MedDINOVISTA3D(nn.Module):
    def __init__(self, num_classes=4, freeze_encoder=True, dropout=0.1):
        super().__init__()
        self.encoder = MedDINOEncoder(freeze_backbone=freeze_encoder)
        self.decoder = VISTA3DTransformerDecoder(num_classes=num_classes, dropout=dropout)
        
    def forward(self, x):
        feats = self.encoder(x)
        logits = self.decoder(feats)
        
        # Resize to exact input size if needed
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='trilinear', align_corners=False)
        return logits
    
    def unfreeze_encoder(self, num_layers=4):
        params = list(self.encoder.vit.parameters())
        for param in params[-num_layers:]:
            param.requires_grad = True
        print(f"Unfroze last {num_layers} layers of encoder")

# ============================================================================
# DATASET
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
        return {"image": torch.from_numpy(p_img).float().unsqueeze(0), 
                "label": torch.from_numpy(p_lbl).long().unsqueeze(0)}

    def _sample_patch(self, img, lbl):
        d, h, w = img.shape
        pd, ph, pw = self.patch_size
        
        if d < pd or h < ph or w < pw:
            pad = ((0, max(0, pd-d)), (0, max(0, ph-h)), (0, max(0, pw-w)))
            img, lbl = np.pad(img, pad, mode='constant'), np.pad(lbl, pad, mode='constant')
            d, h, w = img.shape

        if self.is_train:
            # 100% Tumor/Cyst Centering
            tumor_idx = np.argwhere((lbl == 2) | (lbl == 3))
            if len(tumor_idx) > 0:
                c = tumor_idx[np.random.randint(len(tumor_idx))]
                ds, hs, ws = np.clip(c[0]-pd//2, 0, d-pd), np.clip(c[1]-ph//2, 0, h-ph), np.clip(c[2]-pw//2, 0, w-pw)
            else:
                # Fallback to kidney
                kid_idx = np.argwhere(lbl == 1)
                if len(kid_idx) > 0:
                    c = kid_idx[np.random.randint(len(kid_idx))]
                    ds, hs, ws = np.clip(c[0]-pd//2, 0, d-pd), np.clip(c[1]-ph//2, 0, h-ph), np.clip(c[2]-pw//2, 0, w-pw)
                else:
                    ds, hs, ws = np.random.randint(0, d-pd+1), np.random.randint(0, h-ph+1), np.random.randint(0, w-pw+1)
        else:
            ds, hs, ws = (d-pd)//2, (h-ph)//2, (w-pw)//2

        return img[ds:ds+pd, hs:hs+ph, ws:ws+pw], lbl[ds:ds+pd, hs:hs+ph, ws:ws+pw]

def get_dataloaders(config, max_cases=None):
    data_dir = Path(config["kits23_dir"])
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    if max_cases: cases = cases[:max_cases]
    
    data_dicts = [{"image": str(c / "imaging.nii.gz"), "label": str(c / "segmentation.nii.gz")} 
                  for c in cases if (c / "imaging.nii.gz").exists()]
    
    split = int(len(data_dicts) * config["val_split"])
    train_ds = KiTS23Dataset(data_dicts[split:], config["patch_size"], config["num_samples_per_volume"], True)
    val_ds = KiTS23Dataset(data_dicts[:split], config["patch_size"], 1, False)
    
    return (DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2))

# ============================================================================
# LOSS FUNCTION
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
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky.mean()

class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Weights: BG, Kidney, Tumor, Cyst
        w = torch.tensor([0.1, 1.0, 10.0, 10.0])
        self.register_buffer('ce_weights', w)
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
    
    def forward(self, outputs, target):
        ce = F.cross_entropy(outputs, target.squeeze(1), weight=self.ce_weights)
        tv = self.tversky(outputs, target)
        return ce + tv

# ============================================================================
# TRAINING UTILS
# ============================================================================

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
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
            minlength=self.num_classes**2
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
            "recall": recall.numpy()
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
    for batch in tqdm(loader, desc="Training", leave=False):
        img, lbl = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            out = model(img)
            loss = loss_fn(out, lbl)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0
    tracker = MetricTracker(num_classes=4)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            img, lbl = batch["image"].to(device), batch["label"].to(device)
            with torch.amp.autocast('cuda'):
                out = model(img)
                val_loss += loss_fn(out, lbl).item()
            
            preds = torch.argmax(out, dim=1)
            tracker.update(preds, lbl.squeeze(1))
            
    report, mean_dice = tracker.format_results()
    return val_loss / len(loader), mean_dice, report

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(config):
    print("="*60); print(f"MedDINO-VISTA3D Full Training (Mode: {config.get('use_warmup', False) and 'Full' or 'Quick'})"); print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    train_loader, val_loader = get_dataloaders(config, max_cases=config.get("quick_cases"))
    
    # Model
    model = MedDINOVISTA3D(num_classes=config["num_classes"], freeze_encoder=True, dropout=0.1).to(device)
    print(f"Total Params: {sum(p.numel() for p in model.parameters()):,}")
    
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = HybridLoss().to(device)
    
    best_dice = 0.0
    
    # Helper to create scheduler
    def get_scheduler(optimizer, epochs, is_stage2=False):
        if not config.get("use_warmup") or is_stage2:
            # Simple Cosine for Stage 2 or Quick Test
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        else:
            # Warmup + Cosine for Stage 1 Full Training
            warmup_iters = config["warmup_epochs"]
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
            main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_iters, eta_min=1e-6)
            return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, main], [warmup_iters])

    # STAGE 1
    print("\n--- STAGE 1: Frozen Encoder ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_stage1"], weight_decay=config["weight_decay"])
    scheduler = get_scheduler(optimizer, config["num_epochs_stage1"], is_stage2=False)
    
    early_stopping = EarlyStopping(patience=config["patience_stage1"], verbose=True, path=str(output_dir / "best_model_stage1.pth"))

    for epoch in range(config["num_epochs_stage1"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        val_loss, mean_dice, report = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        
        print(f"Ep {epoch+1}/{config['num_epochs_stage1']} | Loss: {train_loss:.4f}/{val_loss:.4f} | Dice: {mean_dice:.4f}")
        print(report)
        
        # Early Stopping monitors validation LOSS or DICE based on implementation. 
        # Here we adapted EarlyStopping to monitor LOSS (lower is better), but usually we want higher Dice.
        # Let's fix EarlyStopping to work with Score (Higher is Better) or Loss (Lower is Better).
        # Our implementation uses -val_loss (so logic is: if -val_loss increases => val_loss decreases => good).
        # To monitor Dice (Higher is better), we should feed it -dice? No, let's stick to Val Loss for stability or use negative Dice.
        # Standard practice: Monitor Loss for Early Stopping, Save Best Model by Dice.
        
        # Actually, let's simply use Val Loss for Early Stopping.
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered in Stage 1")
            break

        # Keep track of best dice separately if needed, but EarlyStopping saves best val_loss model.
        # If we want to save best DICE model, we can modify EarlyStopping or do it manually.
        # Let's do it manually for Best Dice as it's the primary metric.
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), output_dir / "best_model_stage1_dice.pth")
            print(f"  -> Saved Best Dice Model (Dice: {best_dice:.4f})")

    # STAGE 2
    if config["num_epochs_stage2"] > 0:
        print("\n--- STAGE 2: Fine-tuning Encoder ---")
        # Load best model from Stage 1 (Preferably the one with best Dice or best Loss)
        # Let's load the one saved by EarlyStopping (Best Loss) or Best Dice?
        # Usually Best Loss is safer for convergence, Best Dice is better for metrics.
        # Let's use Best Dice for Stage 2 start.
        if (output_dir / "best_model_stage1_dice.pth").exists():
             model.load_state_dict(torch.load(output_dir / "best_model_stage1_dice.pth"))
        elif (output_dir / "best_model_stage1.pth").exists():
             model.load_state_dict(torch.load(output_dir / "best_model_stage1.pth"))
        
        model.unfreeze_encoder(num_layers=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_stage2"], weight_decay=config["weight_decay"])
        scheduler = get_scheduler(optimizer, config["num_epochs_stage2"], is_stage2=True)
        
        # New Early Stopping for Stage 2
        early_stopping = EarlyStopping(patience=config["patience_stage2"], verbose=True, path=str(output_dir / "best_model_final.pth"))

        for epoch in range(config["num_epochs_stage2"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
            val_loss, mean_dice, report = validate(model, val_loader, loss_fn, device)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full", choices=["test", "quick_test", "full"])
    args = parser.parse_args()
    
    if args.mode == "test":
        # Architecture test
        print("Testing Architecture...")
        model = MedDINOVISTA3D().cuda()
        x = torch.randn(1, 1, 32, 128, 128).cuda()
        y = model(x)
        print(f"Input: {x.shape}, Output: {y.shape}")
        print("Test Passed.")
    else:
        config = get_config(args.mode)
        train(config)