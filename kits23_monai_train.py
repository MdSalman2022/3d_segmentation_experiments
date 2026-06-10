"""
KiTS23 3D Kidney Tumor Segmentation with MONAI
===============================================
Simple 3D U-Net segmentation without heavy preprocessing.
Works directly on the KiTS23 dataset without conversion.

This is MUCH simpler than nnU-Net and doesn't crash WSL!
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Install if needed: pip install monai nibabel

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "kits23_dir": "/mnt/d/Salman/gastric_cancer/kidney/kits23/dataset",
    "output_dir": "/mnt/d/Salman/gastric_cancer/kidney/monai_output",
    "model_dir": "/mnt/d/Salman/gastric_cancer/kidney/monai_models",
    
    # Training params
    "batch_size": 1,  # 3D volumes are large
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "val_split": 0.1,
    
    # Data params
    "spatial_size": [96, 96, 96],  # Patch size (smaller for 2mm spacing)
    "num_samples": 4,  # Patches per volume
    "num_classes": 4,  # background, kidney, tumor, cyst
}

# Create directories
for key in ["output_dir", "model_dir"]:
    Path(CONFIG[key]).mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def get_data_list():
    """Get list of all training cases"""
    kits_dir = Path(CONFIG["kits23_dir"])
    cases = []
    
    for case_dir in sorted(kits_dir.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
            
        img_path = case_dir / "imaging.nii.gz"
        seg_path = case_dir / "segmentation.nii.gz"
        
        if img_path.exists() and seg_path.exists():
            cases.append({
                "image": str(img_path),
                "label": str(seg_path),
                "case_id": case_dir.name
            })
    
    return cases

def create_dataloaders():
    """Create MONAI dataloaders with proper transforms"""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRanged, CropForegroundd,
        RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
        ToTensord, EnsureTyped, SpatialPadd
    )
    from monai.data import DataLoader, CacheDataset, Dataset
    
    # Get all cases
    data_list = get_data_list()
    print(f"Found {len(data_list)} cases")
    
    # Split train/val
    n_val = int(len(data_list) * CONFIG["val_split"])
    train_files = data_list[n_val:]
    val_files = data_list[:n_val]
    
    print(f"Training: {len(train_files)}, Validation: {len(val_files)}")
    
    # Training transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(2.0, 2.0, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=[96, 96, 96]),  # Pad small images
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[96, 96, 96],
            pos=2,
            neg=1,
            num_samples=CONFIG["num_samples"],
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Create datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    # Dataloaders - num_workers=0 for WSL stability
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # No multiprocessing - stable for WSL
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader

# ============================================================================
# MODEL
# ============================================================================
def create_model():
    """Create 3D UNet model using MONAI"""
    from monai.networks.nets import UNet, UNETR, SwinUNETR
    
    # Standard 3D U-Net - works great for kidney segmentation
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=CONFIG["num_classes"],
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    )
    
    return model

# ============================================================================
# TRAINING
# ============================================================================
def train():
    """Main training loop"""
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference
    
    print("="*60)
    print("🏥 KiTS23 MONAI 3D SEGMENTATION")
    print("="*60)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create dataloaders
    print("\n📊 Loading data...")
    train_loader, val_loader = create_dataloaders()
    
    # Create model
    print("\n🧠 Creating model...")
    model = create_model().to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    
    # Loss and optimizer
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Training loop
    print("\n🚀 Starting training...")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print("-"*60)
    
    best_dice = 0
    
    for epoch in range(CONFIG["num_epochs"]):
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
        epoch_loss /= step
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    
                    # Sliding window inference for full volume
                    val_outputs = sliding_window_inference(
                        val_images,
                        roi_size=CONFIG["spatial_size"],
                        sw_batch_size=2,
                        predictor=model,
                        overlap=0.5,
                    )
                    
                    val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                    dice_metric(val_outputs, val_labels)
            
            dice = dice_metric.aggregate().item()
            dice_metric.reset()
            
            print(f"\nEpoch {epoch+1}: Loss={epoch_loss:.4f}, Val Dice={dice:.4f}")
            
            # Save best model
            if dice > best_dice:
                best_dice = dice
                torch.save(model.state_dict(), Path(CONFIG["model_dir"]) / "best_model.pth")
                print(f"  ✅ New best model saved! Dice: {best_dice:.4f}")
        else:
            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}")
    
    print("\n" + "="*60)
    print(f"✅ Training complete! Best Dice: {best_dice:.4f}")
    print("="*60)
    
    return model

# ============================================================================
# INFERENCE
# ============================================================================
def predict(case_id, model=None):
    """Run inference on a single case"""
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst, Orientation,
        Spacing, ScaleIntensityRange, EnsureType
    )
    from monai.inferers import sliding_window_inference
    import nibabel as nib
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model if not provided
    if model is None:
        model = create_model().to(device)
        model_path = Path(CONFIG["model_dir"]) / "best_model.pth"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print("⚠️ No trained model found!")
            return None
    
    model.eval()
    
    # Load and preprocess image
    img_path = Path(CONFIG["kits23_dir"]) / case_id / "imaging.nii.gz"
    
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        ScaleIntensityRange(a_min=-200, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        EnsureType(),
    ])
    
    image = transforms(str(img_path)).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = sliding_window_inference(
            image,
            roi_size=CONFIG["spatial_size"],
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
        )
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Save prediction
    output_path = Path(CONFIG["output_dir"]) / f"{case_id}_pred.nii.gz"
    
    # Load original to get affine
    orig = nib.load(img_path)
    pred_nii = nib.Nifti1Image(prediction.astype(np.uint8), orig.affine)
    nib.save(pred_nii, output_path)
    
    print(f"Saved prediction to {output_path}")
    return prediction

# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_prediction(case_id, slice_idx=None):
    """Visualize prediction vs ground truth"""
    import nibabel as nib
    
    img_path = Path(CONFIG["kits23_dir"]) / case_id / "imaging.nii.gz"
    gt_path = Path(CONFIG["kits23_dir"]) / case_id / "segmentation.nii.gz"
    pred_path = Path(CONFIG["output_dir"]) / f"{case_id}_pred.nii.gz"
    
    img = nib.load(img_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    
    if pred_path.exists():
        pred = nib.load(pred_path).get_fdata()
    else:
        pred = None
    
    # Find slice with most tumor
    if slice_idx is None:
        slice_idx = np.argmax(np.sum(gt == 2, axis=(0, 1)))
    
    n_cols = 3 if pred is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    # Normalize image for display
    img_slice = np.clip(img[:, :, slice_idx], -200, 400)
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
    
    axes[0].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[0].set_title('CT Image')
    axes[0].axis('off')
    
    # Ground truth overlay
    axes[1].imshow(img_slice.T, cmap='gray', origin='lower')
    gt_slice = gt[:, :, slice_idx].T
    overlay = np.zeros((*gt_slice.shape, 4))
    overlay[gt_slice == 1] = [0, 1, 0, 0.4]  # Kidney = green
    overlay[gt_slice == 2] = [1, 0, 0, 0.6]  # Tumor = red
    overlay[gt_slice == 3] = [0, 0, 1, 0.4]  # Cyst = blue
    axes[1].imshow(overlay, origin='lower')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    if pred is not None:
        axes[2].imshow(img_slice.T, cmap='gray', origin='lower')
        pred_slice = pred[:, :, slice_idx].T
        overlay = np.zeros((*pred_slice.shape, 4))
        overlay[pred_slice == 1] = [0, 1, 0, 0.4]
        overlay[pred_slice == 2] = [1, 0, 0, 0.6]
        overlay[pred_slice == 3] = [0, 0, 1, 0.4]
        axes[2].imshow(overlay, origin='lower')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    plt.suptitle(f'{case_id} - Slice {slice_idx}')
    plt.tight_layout()
    plt.savefig(Path(CONFIG["output_dir"]) / f"{case_id}_vis.png", dpi=150)
    plt.show()

# ============================================================================
# QUICK TEST
# ============================================================================
def quick_test(num_cases=5, epochs=10):
    """Quick test on a few cases"""
    from monai.losses import DiceCELoss
    from monai.data import DataLoader, Dataset
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRanged, CropForegroundd,
        RandCropByPosNegLabeld, EnsureTyped
    )
    
    print("="*60)
    print("🧪 QUICK TEST MODE")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get just a few cases
    data_list = get_data_list()[:num_cases]
    print(f"Using {len(data_list)} cases for quick test")
    
    # Simple transforms
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=[96, 96, 96], pos=2, neg=1, num_samples=2
        ),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Create dataloader
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
# MAIN
# ============================================================================
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell in ['ZMQInteractiveShell', 'TerminalInteractiveShell']
    except NameError:
        return False


if __name__ == "__main__":
    if is_notebook():
        # Running in Jupyter - auto-run training
        train()
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["train", "test", "predict", "visualize"], default="train")
        parser.add_argument("--case", type=str, default="case_00000")
        parser.add_argument("--epochs", type=int, default=100)
        args = parser.parse_args()
        
        if args.mode == "test":
            quick_test(num_cases=5, epochs=10)
        elif args.mode == "train":
            CONFIG["num_epochs"] = args.epochs
            train()
        elif args.mode == "predict":
            predict(args.case)
        elif args.mode == "visualize":
            visualize_prediction(args.case)
