"""
KiTS23 2nd Place Solution Reimplementation
==========================================
Complete reimplementation of the MICCAI 2023 KiTS23 Challenge 2nd place solution.
Uses 3-model nnU-Net ensemble with post-processing for kidney and tumor segmentation.

Paper: "Exploring 3D U-Net Training Configurations and Post-Processing Strategies 
        for the MICCAI 2023 Kidney and Tumor Segmentation Challenge"
Authors: Kwang-Hyun Uhm et al.

Key Components:
1. Three nnU-Net models:
   - Lowres PlainConvUNet (batch 2)
   - Lowres ResidualEncoderUNet (batch 2)
   - Fullres PlainConvUNet (batch 4)
2. Kidney post-processing (removes FP from full-res not in low-res)
3. Tumor post-processing
4. Majority voting aggregation

Usage:
    python kits23_2nd_place.py --mode all              # Full pipeline
    python kits23_2nd_place.py --mode train            # Train all models
    python kits23_2nd_place.py --mode inference        # Run inference
    python kits23_2nd_place.py --mode postprocess      # Post-process predictions
    python kits23_2nd_place.py --mode evaluate         # Evaluate results
    python kits23_2nd_place.py --mode quick_test       # Quick test with 5 epochs
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
from multiprocessing import Pool
from datetime import datetime
import warnings
import gc

# Memory optimization for WSL stability
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Disable torch.compile - WSL doesn't have full CUDA toolkit (ptxas)
os.environ["nnUNet_compile"] = "False"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Global logger - will be configured by setup_logging()
logger = None

import numpy as np
import nibabel as nib
from tqdm import tqdm

try:
    from scipy.ndimage import label as scipy_label
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Post-processing may be limited.")

try:
    from surface_distance import compute_surface_distances
    SURFACE_DISTANCE_AVAILABLE = True
except ImportError:
    SURFACE_DISTANCE_AVAILABLE = False
    print("Warning: surface-distance not available. Surface Dice will not be computed.")


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Dataset paths
    "kits23_dir": "./kits23/dataset",
    
    # nnU-Net paths
    "nnunet_raw": "./nnUNet_raw",
    "nnunet_preprocessed": "./nnUNet_preprocessed", 
    "nnunet_results": "./nnUNet_results",
    
    # Output paths
    "output_dir": "./output/kits23_2nd_place",
    
    # Dataset IDs (nnU-Net format)
    "dataset_id": 23,
    "dataset_name": "Dataset023_KiTS23",
    
    # Training settings
    "num_epochs": 1000,  # Paper uses 1000
    "quick_epochs": 5,   # For testing
    "fold": "all",       # Train all folds for best results
    
    # Label definitions
    "labels": {
        "background": 0,
        "kidney": 1,
        "tumor": 2,
        "cyst": 3
    },
    
    # HEC (Hierarchical Evaluation Class) for KiTS evaluation
    "hec_config": {
        "kidney": {"labels": (1,), "tolerance_mm": 2.0},
        "masses": {"labels": (2, 3), "tolerance_mm": 2.0},
        "tumor": {"labels": (2,), "tolerance_mm": 2.0},
    },
}

# Model configurations matching the 2nd place solution
MODEL_CONFIGS = {
    "lowres_plain": {
        "name": "Lowres PlainConvUNet",
        "configuration": "3d_lowres",
        "unet_class": "PlainConvUNet",
        "batch_size": 2,
        "description": "Low-resolution PlainConvUNet - good for localization"
    },
    "lowres_residual": {
        "name": "Lowres ResidualEncoderUNet",
        "configuration": "3d_lowres",
        "unet_class": "ResidualEncoderUNet",
        "batch_size": 2,
        "description": "Low-resolution ResidualUNet - better feature extraction"
    },
    "fullres_batch4": {
        "name": "Fullres PlainConvUNet Batch4",
        "configuration": "3d_fullres",
        "unet_class": "PlainConvUNet",
        "batch_size": 3,  # Reduced from 4 for WSL stability (still good for 32GB VRAM)
        "description": "Full-resolution PlainConvUNet - detailed boundaries"
    }
}


# ============================================================================
# SETUP & UTILITIES
# ============================================================================
def setup_directories():
    """Create all required directories."""
    dirs = [
        CONFIG["nnunet_raw"],
        CONFIG["nnunet_preprocessed"],
        CONFIG["nnunet_results"],
        CONFIG["output_dir"],
        f"{CONFIG['output_dir']}/predictions_lowres",
        f"{CONFIG['output_dir']}/predictions_lowres_residual",
        f"{CONFIG['output_dir']}/predictions_fullres",
        f"{CONFIG['output_dir']}/postprocessed_kidney",
        f"{CONFIG['output_dir']}/postprocessed_tumor",
        f"{CONFIG['output_dir']}/final_predictions",
        f"{CONFIG['output_dir']}/visualizations",
        f"{CONFIG['output_dir']}/logs",  # Log directory
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")


def setup_logging(mode: str = "pipeline"):
    """Setup logging to both console and file."""
    global logger
    
    log_dir = Path(CONFIG["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{mode}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("kits23_2nd_place")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Clear any existing handlers
    
    # File handler - captures everything
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - also show on screen
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"📝 Logging to: {log_file}")
    logger.info(f"   (Check this file if WSL crashes)")
    
    return log_file


def log(message: str, level: str = "info"):
    """Log a message to both console and file."""
    global logger
    if logger is None:
        print(message)  # Fallback to print if logger not initialized
        return
    
    if level == "debug":
        logger.debug(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)


def setup_environment():
    """Setup nnU-Net environment variables."""
    os.environ['nnUNet_raw'] = str(Path(CONFIG["nnunet_raw"]).absolute())
    os.environ['nnUNet_preprocessed'] = str(Path(CONFIG["nnunet_preprocessed"]).absolute())
    os.environ['nnUNet_results'] = str(Path(CONFIG["nnunet_results"]).absolute())
    # Limit nnU-Net data augmentation workers for WSL stability
    os.environ['nnUNet_n_proc_DA'] = '4'  # Limit DA workers (default can be too high)
    print("✅ Environment variables set")


def check_nnunet_installed() -> bool:
    """Check if nnU-Net v2 is installed."""
    try:
        import nnunetv2
        version = getattr(nnunetv2, '__version__', 'installed')
        print(f"✅ nnU-Net v2: {version}")
        return True
    except ImportError:
        print("❌ nnU-Net v2 not installed!")
        print("   Install with: pip install nnunetv2")
        return False


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================================
# DATA CONVERSION
# ============================================================================
def convert_kits23_to_nnunet(max_cases: int = None, clean: bool = False) -> int:
    """
    Convert KiTS23 dataset to nnU-Net format.
    
    Args:
        max_cases: Maximum number of cases to convert (None = all)
        clean: If True, remove existing converted data first
        
    Returns:
        Number of cases converted
    """
    print_header("CONVERTING KiTS23 TO nnU-Net FORMAT")
    
    kits23_dir = Path(CONFIG["kits23_dir"])
    dataset_dir = Path(CONFIG["nnunet_raw"]) / CONFIG["dataset_name"]
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    
    # Clean existing data if requested (important for quick_test mode)
    if clean and dataset_dir.exists():
        print(f"🧹 Cleaning existing data in {dataset_dir}...")
        shutil.rmtree(dataset_dir)
    
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    # Find all cases
    cases = sorted([d for d in kits23_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('case_')])
    
    if max_cases:
        cases = cases[:max_cases]
    
    print(f"Found {len(cases)} cases to convert")
    
    training_cases = []
    skipped = []
    
    for case_dir in tqdm(cases, desc="Converting"):
        case_id = case_dir.name
        img_path = case_dir / "imaging.nii.gz"
        seg_path = case_dir / "segmentation.nii.gz"
        
        if not img_path.exists() or not seg_path.exists():
            skipped.append(case_id)
            continue
        
        # nnU-Net naming: case_XXXXX_0000.nii.gz for images
        dst_img = images_tr / f"{case_id}_0000.nii.gz"
        dst_seg = labels_tr / f"{case_id}.nii.gz"
        
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        if not dst_seg.exists():
            shutil.copy2(seg_path, dst_seg)
        
        training_cases.append(case_id)
    
    # Count ACTUAL files in the folder (handles case where files already exist)
    actual_image_count = len(list(images_tr.glob("*_0000.nii.gz")))
    
    # Create dataset.json with actual file count
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": CONFIG["labels"],
        "numTraining": actual_image_count,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    
    with open(dataset_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\n✅ Converted {len(training_cases)} cases")
    print(f"📊 Total cases in folder: {actual_image_count}")
    if skipped:
        print(f"⚠️  Skipped {len(skipped)} incomplete cases")
    print(f"📁 Output: {dataset_dir}")
    
    return actual_image_count


# ============================================================================
# NNUNET PLANS MODIFICATION
# ============================================================================
def modify_nnunet_plans(model_key: str):
    """
    Modify nnUNetPlans.json for specific model configuration.
    
    Args:
        model_key: One of 'lowres_plain', 'lowres_residual', 'fullres_batch4'
    """
    config = MODEL_CONFIGS[model_key]
    
    plans_path = (
        Path(CONFIG["nnunet_preprocessed"]) / 
        CONFIG["dataset_name"] / 
        "nnUNetPlans.json"
    )
    
    if not plans_path.exists():
        print(f"⚠️ Plans file not found: {plans_path}")
        print("   Run preprocessing first!")
        return
    
    with open(plans_path, 'r') as f:
        plans = json.load(f)
    
    configuration_key = config["configuration"]
    
    if configuration_key in plans["configurations"]:
        plans["configurations"][configuration_key]["UNet_class_name"] = config["unet_class"]
        plans["configurations"][configuration_key]["batch_size"] = config["batch_size"]
        
        with open(plans_path, 'w') as f:
            json.dump(plans, f, indent=4)
        
        print(f"✅ Modified plans for {config['name']}")
        print(f"   UNet_class_name: {config['unet_class']}")
        print(f"   batch_size: {config['batch_size']}")
    else:
        print(f"⚠️ Configuration {configuration_key} not found in plans")


# ============================================================================
# PREPROCESSING
# ============================================================================
def run_preprocessing(configurations: List[str] = None, num_processes: int = 2):
    """
    Run nnU-Net preprocessing.
    
    Args:
        configurations: List of configurations to preprocess (default: both lowres and fullres)
        num_processes: Number of parallel processes (reduced for WSL stability)
    """
    print_header("RUNNING nnU-Net PREPROCESSING")
    
    setup_environment()
    
    if configurations is None:
        configurations = ["3d_lowres", "3d_fullres"]
    
    dataset_id = CONFIG["dataset_id"]
    
    for config in configurations:
        print(f"\n📊 Preprocessing {config}...")
        
        cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id} -c {config} -np {num_processes} --verify_dataset_integrity"
        print(f"Command: {cmd}")
        
        result = os.system(cmd)
        
        # Clean up memory after each preprocessing step
        gc.collect()
        
        if result != 0:
            print(f"⚠️ Preprocessing {config} may have issues (return code: {result})")
        else:
            print(f"✅ Preprocessing {config} complete")


# ============================================================================
# TRAINING
# ============================================================================
def create_custom_trainer(num_epochs: int):
    """
    Create a custom nnUNet trainer with specified number of epochs.
    nnUNet v2 doesn't support --num_epochs via CLI, so we create a custom trainer.
    
    Args:
        num_epochs: Number of training epochs
        
    Returns:
        Name of the custom trainer class
    """
    trainer_name = f"nnUNetTrainer_{num_epochs}epochs"
    
    # Find nnunetv2 trainers directory
    try:
        import nnunetv2
        nnunet_path = Path(nnunetv2.__file__).parent
        trainers_dir = nnunet_path / "training" / "nnUNetTrainer" / "variants" / "training_length"
        trainers_dir.mkdir(parents=True, exist_ok=True)
        
        trainer_file = trainers_dir / f"{trainer_name}.py"
        
        # Create custom trainer - SIMPLEST approach: just override num_epochs as class attribute
        trainer_code = f'''"""
Custom nnUNet trainer with {num_epochs} epochs.
Auto-generated by kits23_2nd_place.py
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class {trainer_name}(nnUNetTrainer):
    """Custom trainer that runs for only {num_epochs} epochs."""
    
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.num_epochs = {num_epochs}
'''
        
        with open(trainer_file, 'w') as f:
            f.write(trainer_code)
        
        print(f"✅ Created custom trainer: {trainer_name}")
        print(f"   File: {trainer_file}")
        return trainer_name
        
    except Exception as e:
        print(f"⚠️ Could not create custom trainer: {e}")
        print(f"   Using default 1000 epochs")
        return None


def train_model(model_key: str, num_epochs: int = None, fold: Union[int, str] = None):
    """
    Train a specific model configuration.
    
    Args:
        model_key: One of 'lowres_plain', 'lowres_residual', 'fullres_batch4'
        num_epochs: Number of training epochs
        fold: Fold to train (0-4 or 'all')
    """
    config = MODEL_CONFIGS[model_key]
    
    if num_epochs is None:
        num_epochs = CONFIG["num_epochs"]
    if fold is None:
        fold = CONFIG["fold"]
    
    print_header(f"TRAINING: {config['name']}")
    print(f"Configuration: {config['configuration']}")
    print(f"UNet class: {config['unet_class']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {num_epochs}")
    print(f"Fold: {fold}")
    
    # Log training start for crash recovery analysis
    log(f"=== Training {config['name']} ===")
    log(f"Config: {config['configuration']}, Batch: {config['batch_size']}, Epochs: {num_epochs}")
    
    setup_environment()
    
    # Modify plans for this model configuration
    modify_nnunet_plans(model_key)
    
    dataset_id = CONFIG["dataset_id"]
    configuration = config["configuration"]
    
    # Build training command
    # Note: Removed --npz flag to avoid blosc2 validation export issues with numpy 2.x
    cmd = f"nnUNetv2_train {dataset_id} {configuration} {fold}"
    
    # Create custom trainer if epochs != 1000
    if num_epochs != 1000:
        trainer_name = create_custom_trainer(num_epochs)
        if trainer_name:
            cmd += f" -tr {trainer_name}"
        else:
            print(f"⚠️ Running with default 1000 epochs (custom trainer failed)")
    
    print(f"\nCommand: {cmd}")
    log(f"Command: {cmd}")
    print("\n🚀 Starting training...\n")
    log(f"Training started at {datetime.now().strftime('%H:%M:%S')}")
    
    result = os.system(cmd)
    
    # Log result
    log(f"Training finished at {datetime.now().strftime('%H:%M:%S')}, return code: {result}")
    
    # Clean up memory after training subprocess completes
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    
    if result == 0:
        print(f"\n✅ Training complete: {config['name']}")
        log(f"✅ Training SUCCESS: {config['name']}")
    else:
        print(f"\n⚠️ Training may have issues (return code: {result})")
        log(f"⚠️ Training FAILED: {config['name']} (return code: {result})", level="error")
    
    return result


def train_all_models(num_epochs: int = None, fold: Union[int, str] = None):
    """Train all three models sequentially."""
    print_header("TRAINING ALL MODELS")
    
    results = {}
    for model_key in ["lowres_plain", "lowres_residual", "fullres_batch4"]:
        print(f"\n{'='*50}")
        print(f"Training model: {model_key}")
        print(f"{'='*50}")
        results[model_key] = train_model(model_key, num_epochs, fold)
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for model_key, result in results.items():
        status = "✅" if result == 0 else "⚠️"
        print(f"  {status} {MODEL_CONFIGS[model_key]['name']}")
    
    return results


# ============================================================================
# INFERENCE
# ============================================================================
def run_inference_single(model_key: str, input_folder: str = None, 
                         output_folder: str = None, fold: Union[int, str] = None,
                         trainer_name: str = None):
    """
    Run inference for a single model.
    
    Args:
        model_key: One of 'lowres_plain', 'lowres_residual', 'fullres_batch4'
        input_folder: Input folder with images
        output_folder: Output folder for predictions
        fold: Fold to use for inference
        trainer_name: Custom trainer name (e.g., 'nnUNetTrainer_5epochs')
    """
    config = MODEL_CONFIGS[model_key]
    
    if fold is None:
        fold = CONFIG["fold"]
    
    print_header(f"INFERENCE: {config['name']}")
    
    setup_environment()
    
    if input_folder is None:
        input_folder = Path(CONFIG["nnunet_raw"]) / CONFIG["dataset_name"] / "imagesTr"
    
    if output_folder is None:
        output_map = {
            "lowres_plain": "predictions_lowres",
            "lowres_residual": "predictions_lowres_residual",
            "fullres_batch4": "predictions_fullres"
        }
        output_folder = Path(CONFIG["output_dir"]) / output_map[model_key]
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    dataset_id = CONFIG["dataset_id"]
    configuration = config["configuration"]
    
    # Modify plans for this model (in case of using different architecture)
    modify_nnunet_plans(model_key)
    
    # Build command with optional trainer name
    cmd = f"nnUNetv2_predict -i {input_folder} -o {output_folder} -d {dataset_id} -c {configuration} -f {fold}"
    if trainer_name:
        cmd += f" -tr {trainer_name}"
    
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"\nCommand: {cmd}")
    
    result = os.system(cmd)
    
    # Clean up memory after inference subprocess
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    if result == 0:
        print(f"\n✅ Inference complete: {config['name']}")
    else:
        print(f"\n⚠️ Inference may have issues (return code: {result})")
    
    return result


def run_inference_all(fold: Union[int, str] = None, trainer_name: str = None):
    """Run inference for all three models."""
    print_header("RUNNING INFERENCE FOR ALL MODELS")
    
    results = {}
    for model_key in ["lowres_plain", "lowres_residual", "fullres_batch4"]:
        results[model_key] = run_inference_single(model_key, fold=fold, trainer_name=trainer_name)
    
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)
    for model_key, result in results.items():
        status = "✅" if result == 0 else "⚠️"
        print(f"  {status} {MODEL_CONFIGS[model_key]['name']}")
    
    return results


# ============================================================================
# POST-PROCESSING
# ============================================================================
def get_connected_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Get connected components of a binary mask.
    
    Args:
        binary_mask: Binary numpy array
        
    Returns:
        Tuple of (labeled array, number of components)
    """
    if not SCIPY_AVAILABLE:
        return binary_mask.astype(np.int32), 1
    
    structure = generate_binary_structure(3, 1)  # 6-connectivity
    labeled, num_features = scipy_label(binary_mask, structure=structure)
    return labeled, num_features


def remove_unmatched_blobs(fullres_mask: np.ndarray, lowres_mask: np.ndarray, 
                          label_value: int) -> np.ndarray:
    """
    Remove blobs from fullres that don't have corresponding blobs in lowres.
    This is the key insight from the 2nd place solution.
    
    Args:
        fullres_mask: Full resolution segmentation
        lowres_mask: Low resolution segmentation (will be resized if needed)
        label_value: Label value to process (1=kidney, 2=tumor, 3=cyst)
        
    Returns:
        Cleaned fullres mask
    """
    if not SCIPY_AVAILABLE:
        return fullres_mask
    
    # Binary masks for the specific label
    fullres_binary = (fullres_mask == label_value)
    lowres_binary = (lowres_mask == label_value)
    
    # Get connected components in fullres
    fullres_labeled, num_fullres = get_connected_components(fullres_binary)
    
    # For each component in fullres, check if it overlaps with any component in lowres
    result_mask = fullres_mask.copy()
    
    for component_id in range(1, num_fullres + 1):
        component_mask = (fullres_labeled == component_id)
        
        # Check overlap with lowres
        overlap = np.any(component_mask & lowres_binary)
        
        if not overlap:
            # Remove this component - it's a false positive
            result_mask[component_mask] = 0
    
    return result_mask


def postprocess_kidney(lowres_folder: str, fullres_folder: str, output_folder: str):
    """
    Post-process kidney predictions.
    Remove kidney FPs from fullres that don't appear in lowres.
    
    Args:
        lowres_folder: Folder with lowres predictions
        fullres_folder: Folder with fullres predictions
        output_folder: Output folder for processed predictions
    """
    print_header("POST-PROCESSING: KIDNEY")
    
    lowres_path = Path(lowres_folder)
    fullres_path = Path(fullres_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fullres_files = sorted(fullres_path.glob("*.nii.gz"))
    print(f"Processing {len(fullres_files)} predictions...")
    
    for fullres_file in tqdm(fullres_files, desc="Kidney post-processing"):
        case_name = fullres_file.name
        lowres_file = lowres_path / case_name
        
        if not lowres_file.exists():
            print(f"⚠️ No lowres prediction for {case_name}")
            shutil.copy2(fullres_file, output_path / case_name)
            continue
        
        # Load predictions
        fullres_nii = nib.load(fullres_file)
        fullres_data = fullres_nii.get_fdata().astype(np.int32)
        lowres_data = nib.load(lowres_file).get_fdata().astype(np.int32)
        
        # Resize lowres if needed
        if fullres_data.shape != lowres_data.shape:
            from scipy.ndimage import zoom
            zoom_factors = np.array(fullres_data.shape) / np.array(lowres_data.shape)
            lowres_data = zoom(lowres_data, zoom_factors, order=0)
        
        # Process kidney (label 1)
        processed = remove_unmatched_blobs(fullres_data, lowres_data, label_value=1)
        
        # Save processed prediction
        processed_nii = nib.Nifti1Image(processed.astype(np.uint8), fullres_nii.affine)
        nib.save(processed_nii, output_path / case_name)
    
    print(f"✅ Kidney post-processing complete: {output_path}")


def postprocess_tumor(kidney_processed_lowres: str, kidney_processed_fullres: str, 
                     output_folder: str):
    """
    Post-process tumor predictions.
    Remove tumor FPs from fullres that don't appear in lowres (after kidney processing).
    
    Args:
        kidney_processed_lowres: Folder with kidney-processed lowres predictions
        kidney_processed_fullres: Folder with kidney-processed fullres predictions
        output_folder: Output folder for processed predictions
    """
    print_header("POST-PROCESSING: TUMOR")
    
    lowres_path = Path(kidney_processed_lowres)
    fullres_path = Path(kidney_processed_fullres)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fullres_files = sorted(fullres_path.glob("*.nii.gz"))
    print(f"Processing {len(fullres_files)} predictions...")
    
    for fullres_file in tqdm(fullres_files, desc="Tumor post-processing"):
        case_name = fullres_file.name
        lowres_file = lowres_path / case_name
        
        if not lowres_file.exists():
            shutil.copy2(fullres_file, output_path / case_name)
            continue
        
        # Load predictions
        fullres_nii = nib.load(fullres_file)
        fullres_data = fullres_nii.get_fdata().astype(np.int32)
        lowres_data = nib.load(lowres_file).get_fdata().astype(np.int32)
        
        # Resize lowres if needed
        if fullres_data.shape != lowres_data.shape:
            from scipy.ndimage import zoom
            zoom_factors = np.array(fullres_data.shape) / np.array(lowres_data.shape)
            lowres_data = zoom(lowres_data, zoom_factors, order=0)
        
        # Process tumor (label 2) and cyst (label 3)
        processed = remove_unmatched_blobs(fullres_data, lowres_data, label_value=2)
        processed = remove_unmatched_blobs(processed, lowres_data, label_value=3)
        
        # Save processed prediction
        processed_nii = nib.Nifti1Image(processed.astype(np.uint8), fullres_nii.affine)
        nib.save(processed_nii, output_path / case_name)
    
    print(f"✅ Tumor post-processing complete: {output_path}")


def majority_voting(input_folders: List[str], output_folder: str):
    """
    Perform majority voting on multiple predictions.
    
    Args:
        input_folders: List of folders containing predictions to combine
        output_folder: Output folder for final predictions
    """
    print_header("MAJORITY VOTING AGGREGATION")
    
    input_paths = [Path(f) for f in input_folders]
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all prediction files from first folder
    pred_files = sorted(input_paths[0].glob("*.nii.gz"))
    print(f"Combining predictions from {len(input_folders)} sources")
    print(f"Processing {len(pred_files)} cases...")
    
    for pred_file in tqdm(pred_files, desc="Majority voting"):
        case_name = pred_file.name
        
        # Load all predictions for this case
        predictions = []
        affine = None
        
        for input_path in input_paths:
            case_file = input_path / case_name
            if case_file.exists():
                nii = nib.load(case_file)
                if affine is None:
                    affine = nii.affine
                predictions.append(nii.get_fdata().astype(np.int32))
        
        if not predictions:
            print(f"⚠️ No predictions found for {case_name}")
            continue
        
        # Stack predictions and compute mode (majority vote)
        stacked = np.stack(predictions, axis=0)
        
        # For each voxel, find the most common label
        final_pred = np.zeros(predictions[0].shape, dtype=np.int32)
        
        for label in range(4):  # background, kidney, tumor, cyst
            label_votes = np.sum(stacked == label, axis=0)
            # Only assign if it's the majority (more than half)
            is_majority = label_votes > len(predictions) / 2
            final_pred[is_majority] = label
        
        # Handle ties: if no majority, use first prediction
        no_majority = (final_pred == 0) & (stacked[0] > 0)
        final_pred[no_majority] = stacked[0][no_majority]
        
        # Save final prediction
        final_nii = nib.Nifti1Image(final_pred.astype(np.uint8), affine)
        nib.save(final_nii, output_path / case_name)
    
    print(f"✅ Majority voting complete: {output_path}")


def run_full_postprocessing():
    """Run the complete post-processing pipeline."""
    print_header("RUNNING FULL POST-PROCESSING PIPELINE")
    
    output_dir = Path(CONFIG["output_dir"])
    
    # Step 1: Kidney post-processing for lowres + fullres pair
    postprocess_kidney(
        lowres_folder=output_dir / "predictions_lowres",
        fullres_folder=output_dir / "predictions_fullres",
        output_folder=output_dir / "postprocessed_kidney" / "plain_fullres"
    )
    
    # Step 2: Kidney post-processing for lowres_residual + fullres pair
    postprocess_kidney(
        lowres_folder=output_dir / "predictions_lowres_residual",
        fullres_folder=output_dir / "predictions_fullres",
        output_folder=output_dir / "postprocessed_kidney" / "residual_fullres"
    )
    
    # Step 3: Tumor post-processing for first pair
    postprocess_tumor(
        kidney_processed_lowres=output_dir / "predictions_lowres",
        kidney_processed_fullres=output_dir / "postprocessed_kidney" / "plain_fullres",
        output_folder=output_dir / "postprocessed_tumor" / "plain_fullres"
    )
    
    # Step 4: Tumor post-processing for second pair
    postprocess_tumor(
        kidney_processed_lowres=output_dir / "predictions_lowres_residual",
        kidney_processed_fullres=output_dir / "postprocessed_kidney" / "residual_fullres",
        output_folder=output_dir / "postprocessed_tumor" / "residual_fullres"
    )
    
    # Step 5: Majority voting aggregation
    majority_voting(
        input_folders=[
            output_dir / "postprocessed_tumor" / "plain_fullres",
            output_dir / "postprocessed_tumor" / "residual_fullres"
        ],
        output_folder=output_dir / "final_predictions"
    )
    
    print("\n✅ Full post-processing pipeline complete!")


# ============================================================================
# EVALUATION
# ============================================================================
def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient."""
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)
    
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    
    intersection = np.sum(pred_mask & gt_mask)
    return 2.0 * intersection / (pred_sum + gt_sum)


def compute_surface_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                        spacing: Tuple[float, ...], tolerance_mm: float) -> float:
    """Compute Surface Dice with tolerance."""
    if not SURFACE_DISTANCE_AVAILABLE:
        return 0.0
    
    pred_empty = np.sum(pred_mask) == 0
    gt_empty = np.sum(gt_mask) == 0
    
    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0
    
    try:
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
    except Exception as e:
        print(f"Surface dice error: {e}")
        return 0.0


def construct_hec_mask(segmentation: np.ndarray, labels: Tuple[int, ...]) -> np.ndarray:
    """Construct binary mask for Hierarchical Evaluation Class."""
    if len(labels) == 1:
        return segmentation == labels[0]
    else:
        mask = np.zeros(segmentation.shape, dtype=bool)
        for label in labels:
            mask[segmentation == label] = True
        return mask


def evaluate_case(pred_path: str, gt_path: str) -> Dict[str, Dict[str, float]]:
    """Evaluate a single case."""
    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)
    
    pred_data = pred_nii.get_fdata().astype(np.int32)
    gt_data = gt_nii.get_fdata().astype(np.int32)
    
    spacing = tuple(pred_nii.header.get_zooms()[:3])
    
    results = {}
    for hec_name, hec_config in CONFIG["hec_config"].items():
        pred_mask = construct_hec_mask(pred_data, hec_config["labels"])
        gt_mask = construct_hec_mask(gt_data, hec_config["labels"])
        
        dice = compute_dice(pred_mask, gt_mask)
        surface_dice = compute_surface_dice(
            pred_mask, gt_mask, spacing, hec_config["tolerance_mm"]
        )
        
        results[hec_name] = {"dice": dice, "surface_dice": surface_dice}
    
    return results


def evaluate_predictions(predictions_dir: str = None, dataset_dir: str = None):
    """
    Evaluate all predictions against ground truth.
    
    Args:
        predictions_dir: Directory containing predictions
        dataset_dir: Directory containing ground truth segmentations
    """
    print_header("EVALUATING PREDICTIONS")
    
    if predictions_dir is None:
        predictions_dir = Path(CONFIG["output_dir"]) / "final_predictions"
    else:
        predictions_dir = Path(predictions_dir)
    
    if dataset_dir is None:
        dataset_dir = Path(CONFIG["kits23_dir"])
    else:
        dataset_dir = Path(dataset_dir)
    
    pred_files = sorted(predictions_dir.glob("*.nii.gz"))
    print(f"Evaluating {len(pred_files)} predictions...")
    
    all_results = []
    hec_names = list(CONFIG["hec_config"].keys())
    
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        case_name = pred_file.stem.replace(".nii", "")
        gt_path = dataset_dir / case_name / "segmentation.nii.gz"
        
        if not gt_path.exists():
            continue
        
        try:
            case_results = evaluate_case(str(pred_file), str(gt_path))
            case_results["case_id"] = case_name
            all_results.append(case_results)
        except Exception as e:
            print(f"Error evaluating {case_name}: {e}")
    
    if not all_results:
        print("❌ No valid predictions to evaluate")
        return {}
    
    # Compute averages
    mean_results = {}
    for hec_name in hec_names:
        dice_scores = [r[hec_name]["dice"] for r in all_results]
        sd_scores = [r[hec_name]["surface_dice"] for r in all_results]
        
        mean_results[hec_name] = {
            "dice": np.mean(dice_scores),
            "dice_std": np.std(dice_scores),
            "surface_dice": np.mean(sd_scores),
            "surface_dice_std": np.std(sd_scores)
        }
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'HEC':<12} {'Dice':<20} {'Surface Dice':<20}")
    print("-" * 52)
    
    for hec_name in hec_names:
        dice = mean_results[hec_name]["dice"]
        dice_std = mean_results[hec_name]["dice_std"]
        sd = mean_results[hec_name]["surface_dice"]
        sd_std = mean_results[hec_name]["surface_dice_std"]
        print(f"{hec_name:<12} {dice:.4f} ± {dice_std:.4f}    {sd:.4f} ± {sd_std:.4f}")
    
    # Composite score (average of all HECs)
    all_dice = [mean_results[h]["dice"] for h in hec_names]
    all_sd = [mean_results[h]["surface_dice"] for h in hec_names]
    print("-" * 52)
    print(f"{'AVERAGE':<12} {np.mean(all_dice):.4f}               {np.mean(all_sd):.4f}")
    print("=" * 70)
    
    # Save results
    results_file = predictions_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "per_case": all_results,
            "mean": mean_results,
            "num_cases": len(all_results)
        }, f, indent=2, default=float)
    
    print(f"\n📁 Results saved to: {results_file}")
    
    return mean_results


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_case(case_id: str, predictions_dir: str = None, save: bool = True):
    """Visualize a single case with prediction overlay."""
    import matplotlib.pyplot as plt
    
    dataset_dir = Path(CONFIG["kits23_dir"])
    if predictions_dir is None:
        predictions_dir = Path(CONFIG["output_dir"]) / "final_predictions"
    else:
        predictions_dir = Path(predictions_dir)
    
    vis_dir = Path(CONFIG["output_dir"]) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = dataset_dir / case_id / "imaging.nii.gz"
    gt_path = dataset_dir / case_id / "segmentation.nii.gz"
    pred_path = predictions_dir / f"{case_id}.nii.gz"
    
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return
    
    # Load data
    img = nib.load(img_path).get_fdata()
    gt = nib.load(gt_path).get_fdata() if gt_path.exists() else None
    pred = nib.load(pred_path).get_fdata() if pred_path.exists() else None
    
    # Find slice with maximum tumor content
    if gt is not None:
        tumor_per_slice = np.sum(gt == 2, axis=(0, 1))
        slice_idx = np.argmax(tumor_per_slice)
        if tumor_per_slice[slice_idx] == 0:
            kidney_per_slice = np.sum(gt == 1, axis=(0, 1))
            slice_idx = np.argmax(kidney_per_slice)
    else:
        slice_idx = img.shape[2] // 2
    
    # Create visualization
    n_cols = 2 if pred is None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    # Normalize CT for display
    img_slice = img[:, :, slice_idx].T
    img_norm = np.clip(img_slice, -200, 400)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    
    # Colors for overlay
    colors = {1: [0, 1, 0, 0.4], 2: [1, 0, 0, 0.6], 3: [0, 0, 1, 0.4]}
    
    def create_overlay(seg_slice):
        overlay = np.zeros((*seg_slice.shape, 4))
        for label, color in colors.items():
            overlay[seg_slice == label] = color
        return overlay
    
    # Plot CT
    axes[0].imshow(img_norm, cmap='gray', origin='lower')
    axes[0].set_title('CT Image')
    axes[0].axis('off')
    
    # Plot GT
    if gt is not None:
        axes[1].imshow(img_norm, cmap='gray', origin='lower')
        axes[1].imshow(create_overlay(gt[:, :, slice_idx].T), origin='lower')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
    
    # Plot Prediction
    if pred is not None:
        axes[2].imshow(img_norm, cmap='gray', origin='lower')
        axes[2].imshow(create_overlay(pred[:, :, slice_idx].T), origin='lower')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    plt.suptitle(f'{case_id} - Slice {slice_idx}', fontsize=14)
    plt.tight_layout()
    
    if save:
        save_path = vis_dir / f"{case_id}_visualization.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📁 Saved: {save_path}")
    
    plt.close()


def visualize_multiple_cases(n_cases: int = 5, predictions_dir: str = None):
    """Visualize multiple cases."""
    print_header("GENERATING VISUALIZATIONS")
    
    dataset_dir = Path(CONFIG["kits23_dir"])
    cases = sorted([d.name for d in dataset_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('case_')])[:n_cases]
    
    for case_id in tqdm(cases, desc="Visualizing"):
        visualize_case(case_id, predictions_dir)
    
    print(f"✅ Visualizations saved to: {CONFIG['output_dir']}/visualizations")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_full_pipeline(num_epochs: int = None, quick_run: bool = False, 
                     max_cases: int = None):
    """
    Run the complete KiTS23 2nd place pipeline.
    
    Args:
        num_epochs: Number of training epochs
        quick_run: If True, use minimal epochs for testing
        max_cases: Maximum cases to process (for testing)
    """
    print_header("KiTS23 2nd PLACE SOLUTION - FULL PIPELINE")
    
    if quick_run:
        num_epochs = CONFIG["quick_epochs"]
        max_cases = max_cases or 10
        print("\n⚡ QUICK RUN MODE - Using minimal settings for testing\n")
    
    if not check_nnunet_installed():
        print("\n❌ Cannot proceed without nnU-Net. Please install and retry.")
        return
    
    setup_directories()
    
    # Initialize logging - saves all output to file
    log_file = setup_logging(mode="quick_test" if quick_run else "full_pipeline")
    log(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Epochs: {num_epochs}, Max cases: {max_cases}, Quick run: {quick_run}")
    
    setup_environment()
    
    # Step 1: Convert data (clean old data in quick_run mode to avoid integrity issues)
    print("\n📋 Step 1/6: Data Conversion")
    convert_kits23_to_nnunet(max_cases, clean=quick_run)
    
    # Step 2: Preprocess
    print("\n📋 Step 2/6: Preprocessing")
    run_preprocessing(["3d_lowres", "3d_fullres"])
    
    # Step 3: Train all models
    print("\n📋 Step 3/6: Training")
    train_all_models(num_epochs)
    
    # Step 4: Run inference (use custom trainer name if using custom epochs)
    print("\n📋 Step 4/6: Inference")
    # Use custom trainer if epochs != 1000 (default)
    if num_epochs and num_epochs != 1000:
        trainer_name = f"nnUNetTrainer_{num_epochs}epochs"
        print(f"   Using custom trainer: {trainer_name}")
    else:
        trainer_name = None
        print("   Using default trainer: nnUNetTrainer")
    run_inference_all(trainer_name=trainer_name)
    
    # Step 5: Post-processing
    print("\n📋 Step 5/6: Post-processing")
    run_full_postprocessing()
    
    # Step 6: Evaluation
    print("\n📋 Step 6/6: Evaluation")
    evaluate_predictions()
    
    # Visualization
    print("\n📋 Bonus: Visualization")
    visualize_multiple_cases(n_cases=10)
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='KiTS23 2nd Place Solution Reimplementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kits23_2nd_place.py --mode all              # Full pipeline
  python kits23_2nd_place.py --mode quick_test       # Quick test (5 epochs)
  python kits23_2nd_place.py --mode train            # Train all models
  python kits23_2nd_place.py --mode train_single --model lowres_plain
  python kits23_2nd_place.py --mode postprocess      # Run post-processing
  python kits23_2nd_place.py --mode evaluate         # Evaluate predictions
        """
    )
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'quick_test', 'setup', 'preprocess', 
                               'train', 'train_single', 'inference', 
                               'postprocess', 'evaluate', 'visualize'],
                       help='Mode to run')
    parser.add_argument('--model', type=str, default=None,
                       choices=['lowres_plain', 'lowres_residual', 'fullres_batch4'],
                       help='Model to train (for train_single mode)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--fold', type=str, default='all',
                       help='Fold to train (0-4 or all)')
    parser.add_argument('--max_cases', type=int, default=None,
                       help='Maximum cases to process')
    parser.add_argument('--predictions_dir', type=str, default=None,
                       help='Directory with predictions to evaluate')
    
    args = parser.parse_args()
    
    # Handle fold argument
    try:
        fold = int(args.fold)
    except ValueError:
        fold = args.fold
    
    # Execute based on mode
    if args.mode == 'all':
        run_full_pipeline(args.epochs, quick_run=False, max_cases=args.max_cases)
    
    elif args.mode == 'quick_test':
        run_full_pipeline(quick_run=True, max_cases=args.max_cases or 10)
    
    elif args.mode == 'setup':
        setup_directories()
        setup_environment()
        convert_kits23_to_nnunet(args.max_cases)
    
    elif args.mode == 'preprocess':
        setup_directories()
        setup_environment()
        run_preprocessing()
    
    elif args.mode == 'train':
        setup_directories()
        train_all_models(args.epochs, fold)
    
    elif args.mode == 'train_single':
        if args.model is None:
            print("❌ Please specify --model for train_single mode")
            return
        setup_directories()
        train_model(args.model, args.epochs, fold)
    
    elif args.mode == 'inference':
        setup_directories()
        run_inference_all(fold)
    
    elif args.mode == 'postprocess':
        setup_directories()
        run_full_postprocessing()
    
    elif args.mode == 'evaluate':
        evaluate_predictions(args.predictions_dir)
    
    elif args.mode == 'visualize':
        visualize_multiple_cases(args.max_cases or 10, args.predictions_dir)


if __name__ == "__main__":
    main()
