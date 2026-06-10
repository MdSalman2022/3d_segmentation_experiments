"""
KiTS21 Utilities
================
Data loading, preprocessing, and visualization utilities.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Default configuration
DEFAULT_CONFIG = {
    "dataset_dir": "./kits23/dataset",
    "output_dir": "./output",
    "model_dir": "./models",
    "num_classes": 4,  # background, kidney, tumor, cyst
    "spatial_size": [128, 128, 128],
    "spacing": [1.5, 1.5, 1.5],
    "num_epochs": 100,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "val_split": 0.2,
    "num_workers": 4,
}

# Label mapping following KiTS21 standard
LABEL_NAMES = {
    0: "background",
    1: "kidney",
    2: "tumor", 
    3: "cyst",
}

# Hierarchical Evaluation Classes (HEC) from KiTS21
HEC_MAPPING = {
    "kidney": (1,),           # Kidney only
    "masses": (2, 3),         # Tumor + Cyst
    "tumor": (2,),            # Tumor only
}


def get_data_list(dataset_dir: str = DEFAULT_CONFIG["dataset_dir"]) -> List[Dict]:
    """
    Get list of all cases with imaging and segmentation files.
    
    Returns:
        List of dicts with 'image', 'label', and 'case_id' keys.
    """
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


def split_train_val(
    data_list: List[Dict], 
    val_split: float = 0.2, 
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split data into training and validation sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(data_list))
    
    n_val = int(len(data_list) * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_files = [data_list[i] for i in train_indices]
    val_files = [data_list[i] for i in val_indices]
    
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
            pixdim=DEFAULT_CONFIG["spacing"],
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200, a_max=400,  # CT window for kidney
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=2,
            neg=1,
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
        spacing = DEFAULT_CONFIG["spacing"]
    
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
        spacing = DEFAULT_CONFIG["spacing"]
    
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=spacing, mode="bilinear"),
        ScaleIntensityRange(a_min=-200, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        EnsureType(),
    ])


def visualize_slice(
    image: np.ndarray,
    label: np.ndarray = None,
    prediction: np.ndarray = None,
    slice_idx: int = None,
    output_path: str = None,
):
    """Visualize a slice with optional label and prediction overlays."""
    import matplotlib.pyplot as plt
    
    # Find slice with most content
    if slice_idx is None:
        if label is not None:
            slice_idx = np.argmax(np.sum(label > 0, axis=(0, 1)))
        else:
            slice_idx = image.shape[2] // 2
    
    n_cols = 1 + (label is not None) + (prediction is not None)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    # Normalize image
    img_slice = image[:, :, slice_idx]
    img_slice = np.clip(img_slice, -200, 400)
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    
    col = 0
    axes[col].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[col].set_title('CT Image')
    axes[col].axis('off')
    
    def create_overlay(seg_slice):
        overlay = np.zeros((*seg_slice.shape, 4))
        overlay[seg_slice == 1] = [0, 1, 0, 0.4]   # Kidney = green
        overlay[seg_slice == 2] = [1, 0, 0, 0.6]   # Tumor = red
        overlay[seg_slice == 3] = [0, 0, 1, 0.4]   # Cyst = blue
        return overlay
    
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


def print_dataset_stats(data_list: List[Dict]):
    """Print dataset statistics."""
    print(f"\n{'='*50}")
    print(f"Dataset Statistics")
    print(f"{'='*50}")
    print(f"Total cases: {len(data_list)}")
    
    if len(data_list) > 0:
        print(f"First case: {data_list[0]['case_id']}")
        print(f"Last case: {data_list[-1]['case_id']}")
