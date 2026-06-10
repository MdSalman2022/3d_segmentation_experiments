"""Working code for kits23 kidney segmentation scores 80% dice
"""

import os
import sys
import json
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    print("Warning: nibabel not installed. Install with: pip install nibabel")
    nib = None

try:
    import SimpleITK as sitk
except ImportError:
    print("Warning: SimpleITK not installed. Install with: pip install SimpleITK")
    sitk = None

try:
    from medpy.metric import dc
except ImportError:
    print("Warning: medpy not installed. Install with: pip install medpy")
    dc = None

try:
    from surface_distance import compute_surface_distances
except ImportError:
    print("Warning: surface-distance not installed.")
    print("Install with: pip install git+https://github.com/JoHof/surface-distance.git")
    compute_surface_distances = None


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

# Label mapping
LABEL_NAMES = {0: "background", 1: "kidney", 2: "tumor", 3: "cyst"}

# HEC (Hierarchical Evaluation Class) configuration for KiTS21 evaluation
HEC_CONFIG = {
    "kidney": {"labels": (1,), "tolerance_mm": 2.0},
    "masses": {"labels": (2, 3), "tolerance_mm": 2.0},
    "tumor": {"labels": (2,), "tolerance_mm": 2.0},
}
HEC_NAMES = list(HEC_CONFIG.keys())


def setup_directories():
    """Create output directories."""
    for key in ["output_dir", "model_dir"]:
        Path(CONFIG[key]).mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA UTILITIES
# ============================================================================
def get_data_list(dataset_dir: str = None) -> List[Dict]:
    """Get list of all cases with imaging and segmentation files."""
    if dataset_dir is None:
        dataset_dir = CONFIG["dataset_dir"]
    
    dataset_path = Path(dataset_dir)
    cases = []
    
    for case_dir in sorted(dataset_path.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        
        img_path = case_dir / "imaging.nii.gz"
        seg_path = case_dir / "segmentation.nii.gz"
        
        if img_path.exists() and seg_path.exists():
            cases.append({
                "image": str(img_path),
                "label": str(seg_path),
                "case_id": case_dir.name,
            })
    
    return cases


def split_train_val(data_list: List[Dict], val_split: float = 0.2, seed: int = 42):
    """Split data into training and validation sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(data_list))
    n_val = int(len(data_list) * val_split)
    
    train_files = [data_list[i] for i in indices[n_val:]]
    val_files = [data_list[i] for i in indices[:n_val]]
    
    return train_files, val_files


def get_train_transforms(spatial_size: List[int], num_samples: int = 4):
    """Get training transforms with augmentation."""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRanged, CropForegroundd,
        RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
        RandShiftIntensityd, RandGaussianNoised, EnsureTyped,
        SpatialPadd
    )
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=CONFIG["spacing"],
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=2, neg=1,
            num_samples=num_samples,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms(spacing: List[float] = None):
    """Get validation transforms (no augmentation)."""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRanged, CropForegroundd, EnsureTyped
    )
    
    if spacing is None:
        spacing = CONFIG["spacing"]
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
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


def get_inference_transforms(spacing: List[float] = None):
    """Get inference transforms for a single image."""
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst, Orientation,
        Spacing, ScaleIntensityRange, EnsureType
    )
    
    if spacing is None:
        spacing = CONFIG["spacing"]
    
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=spacing, mode="bilinear"),
        ScaleIntensityRange(a_min=-200, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        EnsureType(),
    ])


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
    
    data_list = get_data_list(CONFIG["dataset_dir"])
    print(f"\nFound {len(data_list)} valid cases")
    
    if len(data_list) == 0:
        raise ValueError(f"No valid cases found in {CONFIG['dataset_dir']}")
    
    train_files, val_files = split_train_val(data_list, CONFIG["val_split"])
    print(f"Training: {len(train_files)} cases")
    print(f"Validation: {len(val_files)} cases")
    
    train_transforms = get_train_transforms(CONFIG["spatial_size"], CONFIG["num_samples"])
    val_transforms = get_val_transforms(CONFIG["spacing"])
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    
    print("\n📊 Loading data...")
    train_loader, val_loader = create_dataloaders()
    
    print("\n🧠 Creating model...")
    model = create_model().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"]
    )
    
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    
    print("\n🚀 Starting training...")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Patch size: {CONFIG['spatial_size']}")
    print("-"*60)
    
    best_dice = 0
    train_losses = []
    val_dices = []
    
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
        avg_loss = epoch_loss / max(step, 1)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % CONFIG["val_interval"] == 0:
            model.eval()
            dice_metric.reset()
            
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validating", leave=False):
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    
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
            
            class_names = ["Kidney", "Tumor", "Cyst"]
            dice_str = " | ".join([
                f"{name}: {dice_scores[i].item():.4f}" 
                for i, name in enumerate(class_names) if i < len(dice_scores)
            ])
            
            print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Mean Dice={mean_dice:.4f}")
            print(f"  {dice_str}")
            
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
    
    final_path = Path(CONFIG["model_dir"]) / "final_model.pth"
    torch.save({
        'epoch': CONFIG["num_epochs"],
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
    }, final_path)
    
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
    """Run inference on a single case."""
    from monai.inferers import sliding_window_inference
    
    setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    img_path = Path(CONFIG["dataset_dir"]) / case_id / "imaging.nii.gz"
    if not img_path.exists():
        raise FileNotFoundError(f"Imaging not found: {img_path}")
    
    transforms = get_inference_transforms(CONFIG["spacing"])
    image = transforms(str(img_path)).unsqueeze(0).to(device)
    
    print(f"Input shape: {image.shape}")
    
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
    
    if save_prediction:
        output_path = Path(CONFIG["output_dir"]) / f"{case_id}.nii.gz"
        orig_nii = nib.load(img_path)
        pred_nii = nib.Nifti1Image(prediction.astype(np.uint8), orig_nii.affine)
        nib.save(pred_nii, output_path)
        print(f"Saved prediction to {output_path}")
    
    return prediction


def predict_all(output_dir: str = None):
    """Run inference on all cases."""
    if output_dir is None:
        output_dir = CONFIG["output_dir"]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model().to(device)
    model_path = Path(CONFIG["model_dir"]) / "best_model.pth"
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
    
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
# EVALUATION
# ============================================================================
def construct_hec_mask(segmentation: np.ndarray, labels: Tuple[int, ...]) -> np.ndarray:
    """Construct a binary mask for a Hierarchical Evaluation Class."""
    if len(labels) == 1:
        return segmentation == labels[0]
    else:
        mask = np.zeros(segmentation.shape, dtype=bool)
        for label in labels:
            mask[segmentation == label] = True
        return mask


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient."""
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)
    
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    
    intersection = np.sum(pred_mask & gt_mask)
    return 2 * intersection / (pred_sum + gt_sum)


def compute_surface_dice(
    pred_mask: np.ndarray, 
    gt_mask: np.ndarray, 
    spacing: Tuple[float, ...],
    tolerance_mm: float
) -> float:
    """Compute Surface Dice with tolerance."""
    if compute_surface_distances is None:
        return 0.0
    
    pred_empty = np.sum(pred_mask) == 0
    gt_empty = np.sum(gt_mask) == 0
    
    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0
    
    dist = compute_surface_distances(gt_mask, pred_mask, spacing)
    
    distances_gt_to_pred = dist["distances_gt_to_pred"]
    distances_pred_to_gt = dist["distances_pred_to_gt"]
    surfel_areas_gt = dist["surfel_areas_gt"]
    surfel_areas_pred = dist["surfel_areas_pred"]
    
    overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
    overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
    
    total_area = np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred)
    if total_area == 0:
        return 1.0
    
    return (overlap_gt + overlap_pred) / total_area


def evaluate_hec(pred_seg: np.ndarray, gt_seg: np.ndarray, spacing: Tuple[float, ...], hec_name: str):
    """Evaluate a single HEC."""
    config = HEC_CONFIG[hec_name]
    
    pred_mask = construct_hec_mask(pred_seg, config["labels"])
    gt_mask = construct_hec_mask(gt_seg, config["labels"])
    
    dice = compute_dice(pred_mask, gt_mask)
    surface_dice = compute_surface_dice(pred_mask, gt_mask, spacing, config["tolerance_mm"])
    
    return {"dice": dice, "surface_dice": surface_dice}


def evaluate_case(pred_path: str, gt_path: str) -> Dict[str, Dict[str, float]]:
    """Evaluate predictions for a single case."""
    if sitk is None:
        raise ImportError("SimpleITK required for evaluation")
    
    pred_img = sitk.ReadImage(pred_path)
    gt_img = sitk.ReadImage(gt_path)
    
    spacing = tuple(pred_img.GetSpacing()[::-1])
    
    pred_seg = sitk.GetArrayFromImage(pred_img)
    gt_seg = sitk.GetArrayFromImage(gt_img)
    
    results = {}
    for hec_name in HEC_NAMES:
        results[hec_name] = evaluate_hec(pred_seg, gt_seg, spacing, hec_name)
    
    return results


def _evaluate_case_wrapper(args):
    """Wrapper for multiprocessing."""
    pred_path, gt_path = args
    try:
        return evaluate_case(pred_path, gt_path)
    except Exception as e:
        print(f"Error evaluating {pred_path}: {e}")
        return None


def evaluate_predictions(
    predictions_dir: str,
    dataset_dir: str = None,
    num_processes: int = 4,
    output_csv: str = None,
):
    """Evaluate all predictions in a directory."""
    if dataset_dir is None:
        dataset_dir = CONFIG["dataset_dir"]
    
    predictions_dir = Path(predictions_dir)
    dataset_dir = Path(dataset_dir)
    
    pred_files = sorted(predictions_dir.glob("case_*.nii.gz"))
    if not pred_files:
        print(f"No prediction files found in {predictions_dir}")
        return {}, []
    
    eval_pairs = []
    case_ids = []
    
    for pred_file in pred_files:
        case_id = pred_file.stem.replace(".nii", "")
        gt_path = dataset_dir / case_id / "segmentation.nii.gz"
        
        if gt_path.exists():
            eval_pairs.append((str(pred_file), str(gt_path)))
            case_ids.append(case_id)
        else:
            print(f"Warning: Ground truth not found for {case_id}")
    
    print(f"Evaluating {len(eval_pairs)} predictions...")
    
    if num_processes > 1:
        with Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(_evaluate_case_wrapper, eval_pairs),
                total=len(eval_pairs),
                desc="Evaluating"
            ))
    else:
        results = [_evaluate_case_wrapper(pair) for pair in tqdm(eval_pairs)]
    
    all_metrics = {hec: {"dice": [], "surface_dice": []} for hec in HEC_NAMES}
    
    for result in results:
        if result is not None:
            for hec in HEC_NAMES:
                all_metrics[hec]["dice"].append(result[hec]["dice"])
                all_metrics[hec]["surface_dice"].append(result[hec]["surface_dice"])
    
    mean_metrics = {}
    for hec in HEC_NAMES:
        mean_metrics[hec] = {
            "dice": np.mean(all_metrics[hec]["dice"]),
            "surface_dice": np.mean(all_metrics[hec]["surface_dice"]),
        }
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"{'HEC':<10} {'Dice':>10} {'Surface Dice':>15}")
    print(f"{'-'*35}")
    for hec in HEC_NAMES:
        print(f"{hec:<10} {mean_metrics[hec]['dice']:>10.4f} {mean_metrics[hec]['surface_dice']:>15.4f}")
    print(f"{'='*60}")
    
    if output_csv is None:
        output_csv = predictions_dir / "evaluation.csv"
    
    with open(output_csv, "w") as f:
        cols = ["case_id"]
        for hec in HEC_NAMES:
            cols.extend([f"Dice_{hec}", f"SD_{hec}"])
        f.write(",".join(cols) + "\n")
        
        for i, case_id in enumerate(case_ids):
            if results[i] is not None:
                row = [case_id]
                for hec in HEC_NAMES:
                    row.append(f"{results[i][hec]['dice']:.4f}")
                    row.append(f"{results[i][hec]['surface_dice']:.4f}")
                f.write(",".join(row) + "\n")
        
        row = ["MEAN"]
        for hec in HEC_NAMES:
            row.append(f"{mean_metrics[hec]['dice']:.4f}")
            row.append(f"{mean_metrics[hec]['surface_dice']:.4f}")
        f.write(",".join(row) + "\n")
    
    print(f"\nResults saved to: {output_csv}")
    
    return mean_metrics, case_ids


# ============================================================================
# QUICK TEST
# ============================================================================
def quick_test(num_cases: int = 5, epochs: int = 10):
    """Quick test to verify everything works."""
    from monai.losses import DiceCELoss
    from monai.data import DataLoader, Dataset
    
    print("="*60)
    print("🧪 QUICK TEST MODE")
    print("="*60)
    
    setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data_list = get_data_list(CONFIG["dataset_dir"])[:num_cases]
    if len(data_list) == 0:
        print(f"❌ No valid cases found in {CONFIG['dataset_dir']}")
        return None
    
    print(f"Using {len(data_list)} cases for quick test")
    
    transforms = get_train_transforms(CONFIG["spatial_size"], num_samples=2)
    dataset = Dataset(data=data_list, transform=transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    model = create_model().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
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
def visualize_slice(
    image: np.ndarray,
    label: np.ndarray = None,
    prediction: np.ndarray = None,
    slice_idx: int = None,
    output_path: str = None,
):
    """Visualize a slice with optional label and prediction overlays."""
    import matplotlib.pyplot as plt
    
    if slice_idx is None:
        if label is not None:
            slice_idx = np.argmax(np.sum(label > 0, axis=(0, 1)))
        else:
            slice_idx = image.shape[2] // 2
    
    n_cols = 1 + (label is not None) + (prediction is not None)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    img_slice = image[:, :, slice_idx]
    img_slice = np.clip(img_slice, -200, 400)
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    
    def create_overlay(seg_slice):
        overlay = np.zeros((*seg_slice.shape, 4))
        overlay[seg_slice == 1] = [0, 1, 0, 0.4]   # Kidney = green
        overlay[seg_slice == 2] = [1, 0, 0, 0.6]   # Tumor = red
        overlay[seg_slice == 3] = [0, 0, 1, 0.4]   # Cyst = blue
        return overlay
    
    col = 0
    axes[col].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[col].set_title('CT Image')
    axes[col].axis('off')
    
    if label is not None:
        col += 1
        axes[col].imshow(img_slice.T, cmap='gray', origin='lower')
        axes[col].imshow(create_overlay(label[:, :, slice_idx].T), origin='lower')
        axes[col].set_title('Ground Truth')
        axes[col].axis('off')
    
    if prediction is not None:
        col += 1
        axes[col].imshow(img_slice.T, cmap='gray', origin='lower')
        axes[col].imshow(create_overlay(prediction[:, :, slice_idx].T), origin='lower')
        axes[col].set_title('Prediction')
        axes[col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize(case_id: str, slice_idx: int = None):
    """Visualize a case with ground truth and prediction overlay."""
    setup_directories()
    
    img_path = Path(CONFIG["dataset_dir"]) / case_id / "imaging.nii.gz"
    gt_path = Path(CONFIG["dataset_dir"]) / case_id / "segmentation.nii.gz"
    pred_path = Path(CONFIG["output_dir"]) / f"{case_id}.nii.gz"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Imaging not found: {img_path}")
    
    image = nib.load(img_path).get_fdata()
    label = nib.load(gt_path).get_fdata() if gt_path.exists() else None
    prediction = nib.load(pred_path).get_fdata() if pred_path.exists() else None
    
    output_path = Path(CONFIG["output_dir"]) / f"{case_id}_visualization.png"
    
    visualize_slice(image, label, prediction, slice_idx=slice_idx, output_path=str(output_path))
    
    print(f"Visualization saved to: {output_path}")


# ============================================================================
# DOWNLOAD IMAGING
# ============================================================================
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_imaging(dataset_dir: str = None, overwrite: bool = False):
    """Download imaging files for all cases."""
    if dataset_dir is None:
        dataset_dir = CONFIG["dataset_dir"]
    
    dataset_dir = Path(dataset_dir)
    KITS_BASE_URL = "https://kits19.sfo2.digitaloceanspaces.com"
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    case_ids = [d.name for d in sorted(dataset_dir.iterdir()) 
                if d.is_dir() and d.name.startswith("case_")]
    print(f"Found {len(case_ids)} cases in {dataset_dir}")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for case_id in case_ids:
        case_dir = dataset_dir / case_id
        imaging_path = case_dir / "imaging.nii.gz"
        
        if imaging_path.exists() and not overwrite:
            skipped += 1
            continue
        
        url = f"{KITS_BASE_URL}/{case_id}/imaging.nii.gz"
        
        print(f"\nDownloading {case_id}...")
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=imaging_path.name) as t:
                urllib.request.urlretrieve(url, str(imaging_path), reporthook=t.update_to)
            downloaded += 1
        except Exception as e:
            print(f"  Failed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KiTS21/23 Kidney Tumor Segmentation")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "predict", "predict_all", "visualize", "evaluate", "download"],
                        help="Mode to run")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Path to dataset (overrides config)")
    parser.add_argument("--predictions_dir", type=str, default=None,
                        help="Path to predictions for evaluation")
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
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Number of processes for evaluation")
    
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
    elif args.mode == "evaluate":
        pred_dir = args.predictions_dir or CONFIG["output_dir"]
        evaluate_predictions(pred_dir, CONFIG["dataset_dir"], args.num_processes)
    elif args.mode == "download":
        download_imaging()


""" 

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/mnt/d/Salman/gastric_cancer/kidney/script.py", line 973, in <module>
    train()
  File "/mnt/d/Salman/gastric_cancer/kidney/script.py", line 309, in train
    from monai.losses import DiceCELoss
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/__init__.py", line 101, in <module>
    load_submodules(sys.modules[__name__], False, exclude_pattern=excludes)
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/utils/module.py", line 187, in load_submodules
    mod = import_module(name)
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/apps/__init__.py", line 14, in <module>
    from .datasets import CrossValidation, DecathlonDataset, MedNISTDataset, TciaDataset
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/apps/datasets.py", line 33, in <module>
    from monai.data import (
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/data/__init__.py", line 29, in <module>
    from .dataset import (
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/data/dataset.py", line 40, in <module>
    from monai.transforms import Compose, Randomizable, RandomizableTrait, Transform, convert_to_contiguous, reset_ops_id
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/transforms/__init__.py", line 93, in <module>
    from .intensity.array import (
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/transforms/intensity/array.py", line 29, in <module>
    from monai.data.ultrasound_confidence_map import UltrasoundConfidenceMap
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/data/ultrasound_confidence_map.py", line 21, in <module>
    cv2, _ = optional_import("cv2")
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/utils/module.py", line 378, in optional_import
    pkg = __import__(module)  # top level module
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/cv2/__init__.py", line 181, in <module>
    bootstrap()
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/cv2/__init__.py", line 153, in bootstrap
    native_module = importlib.import_module("cv2")
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
AttributeError: _ARRAY_API not found
============================================================
🏥 KiTS21/23 3D KIDNEY TUMOR SEGMENTATION
============================================================

Device: cuda
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 34.2 GB

📊 Loading data...

Found 489 valid cases
Training: 392 cases
Validation: 97 cases
/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/utils/deprecate_utils.py:321: FutureWarning: monai.transforms.spatial.dictionary Orientationd.__init__:labels: Current default value of argument `labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))` was changed in version None from `labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))` to `labels=None`. Default value changed to None meaning that the transform now uses the 'space' of a meta-tensor, if applicable, to determine appropriate axis labels.
  warn_deprecated(argname, msg, warning_category)

🧠 Creating model...
Model parameters: 19,225,585

🚀 Starting training...
Epochs: 100
Batch size: 2
Patch size: [128, 128, 128]
------------------------------------------------------------
Epoch 1/100:   4%|████                                                                                                             | 7/196 [00:49<15:33,  4.94s/it, loss=2.5205]Epoch 1/100:   4%|████                                                                                                             | 7/196 [01:06<29:53,  9.49s/it, loss=2.5205]
Traceback (most recent call last):
  File "/mnt/d/Salman/gastric_cancer/kidney/script.py", line 973, in <module>
    train()
  File "/mnt/d/Salman/gastric_cancer/kidney/script.py", line 362, in train
    for batch in pbar:
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 734, in __next__
    data = self._next_data()
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1492, in _next_data
    idx, data = self._get_data()
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1444, in _get_data
    success, data = self._try_get_data()
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1285, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/queue.py", line 180, in get
    self.not_empty.wait(remaining)
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/threading.py", line 324, in wait
    gotit = waiter.acquire(True, timeout)
KeyboardInterrupt
^C
(dl-env-linux) wsluser@DESKTOP-J3ESTLE:/mnt/d/Salman/gastric_cancer/kidney$ python script.py

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/mnt/d/Salman/gastric_cancer/kidney/script.py", line 973, in <module>
    train()
  File "/mnt/d/Salman/gastric_cancer/kidney/script.py", line 309, in train
    from monai.losses import DiceCELoss
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/__init__.py", line 101, in <module>
    load_submodules(sys.modules[__name__], False, exclude_pattern=excludes)
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/utils/module.py", line 187, in load_submodules
    mod = import_module(name)
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/apps/__init__.py", line 14, in <module>
    from .datasets import CrossValidation, DecathlonDataset, MedNISTDataset, TciaDataset
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/apps/datasets.py", line 33, in <module>
    from monai.data import (
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/data/__init__.py", line 29, in <module>
    from .dataset import (
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/data/dataset.py", line 40, in <module>
    from monai.transforms import Compose, Randomizable, RandomizableTrait, Transform, convert_to_contiguous, reset_ops_id
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/transforms/__init__.py", line 93, in <module>
    from .intensity.array import (
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/transforms/intensity/array.py", line 29, in <module>
    from monai.data.ultrasound_confidence_map import UltrasoundConfidenceMap
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/data/ultrasound_confidence_map.py", line 21, in <module>
    cv2, _ = optional_import("cv2")
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/utils/module.py", line 378, in optional_import
    pkg = __import__(module)  # top level module
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/cv2/__init__.py", line 181, in <module>
    bootstrap()
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/cv2/__init__.py", line 153, in bootstrap
    native_module = importlib.import_module("cv2")
  File "/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
AttributeError: _ARRAY_API not found
============================================================
🏥 KiTS21/23 3D KIDNEY TUMOR SEGMENTATION
============================================================

Device: cuda
GPU: NVIDIA GeForce RTX 5090
GPU Memory: 34.2 GB

📊 Loading data...

Found 489 valid cases
Training: 392 cases
Validation: 97 cases
/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/utils/deprecate_utils.py:321: FutureWarning: monai.transforms.spatial.dictionary Orientationd.__init__:labels: Current default value of argument `labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))` was changed in version None from `labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))` to `labels=None`. Default value changed to None meaning that the transform now uses the 'space' of a meta-tensor, if applicable, to determine appropriate axis labels.
  warn_deprecated(argname, msg, warning_category)

🧠 Creating model...
Model parameters: 19,225,585

🚀 Starting training...
Epochs: 10
Batch size: 2
Patch size: [128, 128, 128]
------------------------------------------------------------
Epoch 1/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [18:46<00:00,  5.75s/it, loss=1.6611]
Epoch 1: Loss=1.9542
Epoch 2/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [19:01<00:00,  5.83s/it, loss=0.9785]
Epoch 2: Loss=1.2641
Epoch 3/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [20:06<00:00,  6.16s/it, loss=1.0475]
Epoch 3: Loss=0.8823
Epoch 4/10:   8%|█████████▏                                                                                                       | 16/196 [01:46<12:21,  4.12s/it, loss=0.8748]                                           Epoch 4/10:  26%|█████████████████████████████▍                                                                                   | 51/196 [04:35<24:05,  9.97s/it, loss=0.7948]                                           Epoch 4/10:  28%|███████████████████████████████▏                                                                                 | 54/196 [04:37<08:33,  3.62s/it, loss=0.7808]                     Epoch 4/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [19:26<00:00,  5.95s/it, loss=0.7397]
Epoch 4: Loss=0.7700
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [19:32<00:00,  5.98s/it, loss=0.6809]
Validating:   0%|                                                                                                                                                                                   | 0/97 [00:00<?, ?it/s]/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/inferers/utils.py:226: UserWarning: Using a non-tuple sequence for multidimensional indexing is deprecated and will be changed in pytorch 2.9; use x[tuple(seq)] instead of x[seq]. In pytorch 2.9 this will be interpreted as tensor index, x[torch.tensor(seq)], which will result either in an error or a different result (Triggered internally at /home/conda/feedstock_root/build_artifacts/libtorch_1762103814636/work/torch/csrc/autograd/python_variable_indexing.cpp:306.)
  win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
/home/wsluser/miniconda3/envs/dl-env-linux/lib/python3.10/site-packages/monai/inferers/utils.py:370: UserWarning: Using a non-tuple sequence for multidimensional indexing is deprecated and will be changed in pytorch 2.9; use x[tuple(seq)] instead of x[seq]. In pytorch 2.9 this will be interpreted as tensor index, x[torch.tensor(seq)], which will result either in an error or a different result (Triggered internally at /home/conda/feedstock_root/build_artifacts/libtorch_1762103814636/work/torch/csrc/autograd/python_variable_indexing.cpp:306.)
  out[idx_zm] += p
                                                                                                                                                                                                                           
Epoch 5: Loss=0.7155, Mean Dice=0.7224
  Kidney: 0.7224
  ✅ New best model saved! Dice: 0.7224
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [17:39<00:00,  5.40s/it, loss=0.6484]
Epoch 6: Loss=0.6753
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [19:17<00:00,  5.91s/it, loss=0.5960]
Epoch 7: Loss=0.6555
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [18:00<00:00,  5.51s/it, loss=0.6351]
Epoch 8: Loss=0.6328
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [19:17<00:00,  5.91s/it, loss=0.6091]
Epoch 9: Loss=0.6261
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [19:19<00:00,  5.92s/it, loss=0.6102]
Validating:  86%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍             Validating:  87%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏           Validating:  89%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋        Validating:  90%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍      Validating:  91%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏    Validating:  92%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉   Validating:  93%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋ Validating:  94%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████Validating:  96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████Validating:  97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████Validating:  98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████Validating:  99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████Validating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                                                                           
Epoch 10: Loss=0.6173, Mean Dice=0.8559
  Kidney: 0.8559
  ✅ New best model saved! Dice: 0.8559

============================================================
✅ Training complete!
Best Dice: 0.8559
Model saved: models/final_model.pth
============================================
 """