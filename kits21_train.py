"""
KiTS21 Training Pipeline
========================
3D Kidney and Tumor Segmentation using MONAI.
Based on KiTS21 nnUNet baseline approach.

Dataset: KiTS23 at ./kits23/dataset
Classes: Background (0), Kidney (1), Tumor (2), Cyst (3)

Usage:
    python kits21_train.py --mode train --epochs 100
    python kits21_train.py --mode test --num_cases 5 --epochs 10
    python kits21_train.py --mode predict --case case_00000
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Paths (relative for portability)
    "dataset_dir": "./kits23/dataset",
    "output_dir": "./output",
    "model_dir": "./models",
    
    # Model architecture
    "num_classes": 4,  # background, kidney, tumor, cyst
    "in_channels": 1,
    "channels": (32, 64, 128, 256, 512),
    "strides": (2, 2, 2, 2),
    "num_res_units": 2,
    
    # Training
    "batch_size": 2,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "val_interval": 5,
    
    # Data
    "spatial_size": [128, 128, 128],
    "spacing": [1.5, 1.5, 1.5],  # Target spacing in mm
    "val_split": 0.2,
    "num_samples": 4,  # Patches per volume during training
    "num_workers": 4,
    
    # Inference
    "sw_batch_size": 4,  # Sliding window batch size
    "overlap": 0.5,  # Sliding window overlap
}


def setup_directories():
    """Create output directories."""
    for key in ["output_dir", "model_dir"]:
        Path(CONFIG[key]).mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL
# ============================================================================
def create_model():
    """Create 3D UNet model using MONAI."""
    from monai.networks.nets import UNet
    
    model = UNet(
        spatial_dims=3,
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["num_classes"],
        channels=CONFIG["channels"],
        strides=CONFIG["strides"],
        num_res_units=CONFIG["num_res_units"],
        norm="batch",
    )
    
    return model


# ============================================================================
# DATA LOADING
# ============================================================================
def create_dataloaders():
    """Create training and validation dataloaders."""
    from monai.data import DataLoader, Dataset
    from kits21_utils import (
        get_data_list, split_train_val,
        get_train_transforms, get_val_transforms,
        print_dataset_stats
    )
    
    # Get all cases
    data_list = get_data_list(CONFIG["dataset_dir"])
    print_dataset_stats(data_list)
    
    if len(data_list) == 0:
        raise ValueError(f"No valid cases found in {CONFIG['dataset_dir']}")
    
    # Split data
    train_files, val_files = split_train_val(data_list, CONFIG["val_split"])
    print(f"Training: {len(train_files)} cases")
    print(f"Validation: {len(val_files)} cases")
    
    # Create transforms
    train_transforms = get_train_transforms(
        CONFIG["spatial_size"], 
        CONFIG["num_samples"]
    )
    val_transforms = get_val_transforms(CONFIG["spacing"])
    
    # Create datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Full volumes for validation
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    
    return train_loader, val_loader


# ============================================================================
# TRAINING
# ============================================================================
def train():
    """Main training loop."""
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference
    
    print("="*60)
    print("🏥 KiTS21/23 3D KIDNEY TUMOR SEGMENTATION")
    print("="*60)
    
    setup_directories()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    
    # Data
    print("\n📊 Loading data...")
    train_loader, val_loader = create_dataloaders()
    
    # Model
    print("\n🧠 Creating model...")
    model = create_model().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    
    # Loss and optimizer
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"]
    )
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    
    # Training loop
    print("\n🚀 Starting training...")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Patch size: {CONFIG['spatial_size']}")
    print("-"*60)
    
    best_dice = 0
    train_losses = []
    val_dices = []
    
    for epoch in range(CONFIG["num_epochs"]):
        # Training
        model.train()
        epoch_loss = 0
        step = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for batch in pbar:
            step += 1
            
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = epoch_loss / max(step, 1)
        train_losses.append(avg_loss)
        
        # Validation
        if (epoch + 1) % CONFIG["val_interval"] == 0:
            model.eval()
            dice_metric.reset()
            
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validating", leave=False):
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    
                    # Sliding window inference
                    val_outputs = sliding_window_inference(
                        val_images,
                        roi_size=CONFIG["spatial_size"],
                        sw_batch_size=CONFIG["sw_batch_size"],
                        predictor=model,
                        overlap=CONFIG["overlap"],
                    )
                    
                    val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                    dice_metric(val_outputs, val_labels)
            
            dice_scores = dice_metric.aggregate()
            mean_dice = dice_scores.mean().item()
            val_dices.append(mean_dice)
            
            # Per-class Dice
            class_names = ["Kidney", "Tumor", "Cyst"]
            dice_str = " | ".join([
                f"{name}: {dice_scores[i].item():.4f}" 
                for i, name in enumerate(class_names) if i < len(dice_scores)
            ])
            
            print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Mean Dice={mean_dice:.4f}")
            print(f"  {dice_str}")
            
            # Save best model
            if mean_dice > best_dice:
                best_dice = mean_dice
                model_path = Path(CONFIG["model_dir"]) / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': mean_dice,
                    'config': CONFIG,
                }, model_path)
                print(f"  ✅ New best model saved! Dice: {best_dice:.4f}")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    # Save final model
    final_path = Path(CONFIG["model_dir"]) / "final_model.pth"
    torch.save({
        'epoch': CONFIG["num_epochs"],
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
    }, final_path)
    
    # Save training history
    history_path = Path(CONFIG["output_dir"]) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_dices": val_dices,
            "best_dice": best_dice,
            "config": CONFIG,
        }, f, indent=2)
    
    print("\n" + "="*60)
    print(f"✅ Training complete!")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Model saved: {final_path}")
    print("="*60)
    
    return model


# ============================================================================
# INFERENCE
# ============================================================================
def predict(case_id: str, model=None, save_prediction: bool = True):
    """Run inference on a single case and optionally save prediction."""
    from monai.inferers import sliding_window_inference
    from kits21_utils import get_inference_transforms
    import nibabel as nib
    
    setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if model is None:
        model = create_model().to(device)
        model_path = Path(CONFIG["model_dir"]) / "best_model.pth"
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"No model found at {model_path}")
    
    model.eval()
    
    # Load image
    img_path = Path(CONFIG["dataset_dir"]) / case_id / "imaging.nii.gz"
    if not img_path.exists():
        raise FileNotFoundError(f"Imaging not found: {img_path}")
    
    transforms = get_inference_transforms(CONFIG["spacing"])
    image = transforms(str(img_path)).unsqueeze(0).to(device)
    
    print(f"Input shape: {image.shape}")
    
    # Inference
    with torch.no_grad():
        output = sliding_window_inference(
            image,
            roi_size=CONFIG["spatial_size"],
            sw_batch_size=CONFIG["sw_batch_size"],
            predictor=model,
            overlap=CONFIG["overlap"],
        )
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique labels: {np.unique(prediction)}")
    
    # Save prediction
    if save_prediction:
        output_path = Path(CONFIG["output_dir"]) / f"{case_id}.nii.gz"
        
        # Load original for affine
        orig_nii = nib.load(img_path)
        pred_nii = nib.Nifti1Image(prediction.astype(np.uint8), orig_nii.affine)
        nib.save(pred_nii, output_path)
        print(f"Saved prediction to {output_path}")
    
    return prediction


def predict_all(output_dir: str = None):
    """Run inference on all cases and save predictions."""
    from kits21_utils import get_data_list
    
    if output_dir is None:
        output_dir = CONFIG["output_dir"]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model once
    model = create_model().to(device)
    model_path = Path(CONFIG["model_dir"]) / "best_model.pth"
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
    
    # Get all cases
    data_list = get_data_list(CONFIG["dataset_dir"])
    print(f"Running inference on {len(data_list)} cases...")
    
    for item in tqdm(data_list):
        case_id = item["case_id"]
        try:
            predict(case_id, model=model, save_prediction=True)
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
    
    print(f"\nPredictions saved to: {output_dir}")


# ============================================================================
# QUICK TEST
# ============================================================================
def quick_test(num_cases: int = 5, epochs: int = 10):
    """Quick test to verify everything works."""
    from monai.losses import DiceCELoss
    from monai.data import DataLoader, Dataset
    from kits21_utils import get_data_list, get_train_transforms
    
    print("="*60)
    print("🧪 QUICK TEST MODE")
    print("="*60)
    
    setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get subset of data
    data_list = get_data_list(CONFIG["dataset_dir"])[:num_cases]
    if len(data_list) == 0:
        print(f"❌ No valid cases found in {CONFIG['dataset_dir']}")
        return None
    
    print(f"Using {len(data_list)} cases for quick test")
    
    # Create simple dataloader
    transforms = get_train_transforms(CONFIG["spatial_size"], num_samples=2)
    dataset = Dataset(data=data_list, transform=transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # Create model
    model = create_model().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
    
    print("\n✅ Quick test complete! Model is working.")
    return model


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize(case_id: str, slice_idx: int = None):
    """Visualize a case with ground truth and prediction overlay."""
    from kits21_utils import visualize_slice
    import nibabel as nib
    
    setup_directories()
    
    # Load data
    img_path = Path(CONFIG["dataset_dir"]) / case_id / "imaging.nii.gz"
    gt_path = Path(CONFIG["dataset_dir"]) / case_id / "segmentation.nii.gz"
    pred_path = Path(CONFIG["output_dir"]) / f"{case_id}.nii.gz"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Imaging not found: {img_path}")
    
    image = nib.load(img_path).get_fdata()
    label = nib.load(gt_path).get_fdata() if gt_path.exists() else None
    prediction = nib.load(pred_path).get_fdata() if pred_path.exists() else None
    
    output_path = Path(CONFIG["output_dir"]) / f"{case_id}_visualization.png"
    
    visualize_slice(
        image, 
        label, 
        prediction, 
        slice_idx=slice_idx,
        output_path=str(output_path)
    )
    
    print(f"Visualization saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KiTS21 Training Pipeline")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "predict", "predict_all", "visualize"],
                        help="Mode to run")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Path to dataset (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--case", type=str, default="case_00000",
                        help="Case ID for predict/visualize mode")
    parser.add_argument("--num_cases", type=int, default=5,
                        help="Number of cases for test mode")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Override config
    if args.dataset_dir:
        CONFIG["dataset_dir"] = args.dataset_dir
    if args.epochs:
        CONFIG["num_epochs"] = args.epochs
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if args.num_workers is not None:
        CONFIG["num_workers"] = args.num_workers
    
    # Run
    if args.mode == "train":
        train()
    elif args.mode == "test":
        quick_test(num_cases=args.num_cases, epochs=args.epochs or 10)
    elif args.mode == "predict":
        predict(args.case)
    elif args.mode == "predict_all":
        predict_all()
    elif args.mode == "visualize":
        visualize(args.case)
