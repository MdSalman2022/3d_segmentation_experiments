"""
KiTS23 2nd Place Solution - Simplified Version (V2)
====================================================
Simplified version focusing on speed and proper validation.

Key Changes from V1:
- Single model only (3d_fullres PlainConvUNet)
- 20 cases for quick testing
- Proper train/validation split (fold 0 = 80/20 split)
- No post-processing (simpler evaluation)
- Separate output folder

Usage:
    python kits23_2nd_place_v2.py --mode quick_test     # 5 epochs, 20 cases
    python kits23_2nd_place_v2.py --mode full           # 1000 epochs, all cases
    python kits23_2nd_place_v2.py --mode inference      # Run inference only
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime
import gc

# Memory optimization for WSL stability
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["nnUNet_compile"] = "False"

import numpy as np
import nibabel as nib
from tqdm import tqdm


# ============================================================================
# CONFIGURATION - V2 (Simplified)
# ============================================================================
CONFIG = {
    # Dataset paths
    "kits23_dir": "./kits23/dataset",
    
    # nnU-Net paths - SEPARATE from V1 to avoid conflicts
    "nnunet_raw": "./nnUNet_raw_v2",
    "nnunet_preprocessed": "./nnUNet_preprocessed_v2", 
    "nnunet_results": "./nnUNet_results_v2",
    
    # Output paths
    "output_dir": "./output/kits23_v2",
    
    # Dataset IDs
    "dataset_id": 24,  # Different ID from V1 (23)
    "dataset_name": "Dataset024_KiTS23_V2",
    
    # Training settings
    "num_epochs": 1000,
    "quick_epochs": 5,
    "fold": 0,  # Use fold 0 for proper 80/20 split
    
    # Simplified: only fullres model
    "configuration": "3d_fullres",
    "batch_size": 2,
    
    # Case limits
    "quick_cases": 20,  # 20 cases for quick testing
    
    # Labels
    "labels": {
        "background": 0,
        "kidney": 1,
        "tumor": 2,
        "cyst": 3
    },
}


# ============================================================================
# SETUP
# ============================================================================
def setup_directories():
    """Create all required directories."""
    dirs = [
        CONFIG["nnunet_raw"],
        CONFIG["nnunet_preprocessed"],
        CONFIG["nnunet_results"],
        CONFIG["output_dir"],
        f"{CONFIG['output_dir']}/predictions",
        f"{CONFIG['output_dir']}/visualizations",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created (V2 - separate from V1)")


def setup_environment():
    """Setup nnU-Net environment variables."""
    os.environ['nnUNet_raw'] = str(Path(CONFIG["nnunet_raw"]).absolute())
    os.environ['nnUNet_preprocessed'] = str(Path(CONFIG["nnunet_preprocessed"]).absolute())
    os.environ['nnUNet_results'] = str(Path(CONFIG["nnunet_results"]).absolute())
    os.environ['nnUNet_n_proc_DA'] = '4'
    print("✅ Environment variables set")


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================================
# DATA CONVERSION
# ============================================================================
def convert_kits23_to_nnunet(max_cases: int = None) -> int:
    """Convert KiTS23 dataset to nnU-Net format."""
    print_header("CONVERTING KiTS23 TO nnU-Net FORMAT (V2)")
    
    kits23_dir = Path(CONFIG["kits23_dir"])
    dataset_dir = Path(CONFIG["nnunet_raw"]) / CONFIG["dataset_name"]
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    
    # Clean and recreate
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    # Find all cases
    cases = sorted([d for d in kits23_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('case_')])
    
    if max_cases:
        cases = cases[:max_cases]
    
    print(f"Converting {len(cases)} cases...")
    
    converted = 0
    for case_dir in tqdm(cases, desc="Converting"):
        case_id = case_dir.name
        img_path = case_dir / "imaging.nii.gz"
        seg_path = case_dir / "segmentation.nii.gz"
        
        if not img_path.exists() or not seg_path.exists():
            continue
        
        shutil.copy2(img_path, images_tr / f"{case_id}_0000.nii.gz")
        shutil.copy2(seg_path, labels_tr / f"{case_id}.nii.gz")
        converted += 1
    
    # Create dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": CONFIG["labels"],
        "numTraining": converted,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    
    with open(dataset_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"✅ Converted {converted} cases to {dataset_dir}")
    return converted


# ============================================================================
# PREPROCESSING
# ============================================================================
def run_preprocessing():
    """Run nnU-Net preprocessing for fullres only."""
    print_header("PREPROCESSING (3d_fullres only)")
    
    setup_environment()
    
    cmd = f"nnUNetv2_plan_and_preprocess -d {CONFIG['dataset_id']} -c 3d_fullres -np 8 --verify_dataset_integrity"
    print(f"Command: {cmd}")
    
    result = os.system(cmd)
    gc.collect()
    
    if result == 0:
        print("✅ Preprocessing complete")
    else:
        print(f"⚠️ Preprocessing may have issues (return code: {result})")


# ============================================================================
# TRAINING
# ============================================================================
def create_custom_trainer(num_epochs: int) -> str:
    """Create custom trainer with specified epochs."""
    trainer_name = f"nnUNetTrainer_{num_epochs}epochs"
    
    try:
        import nnunetv2
        nnunet_path = Path(nnunetv2.__file__).parent
        trainers_dir = nnunet_path / "training" / "nnUNetTrainer" / "variants" / "training_length"
        trainers_dir.mkdir(parents=True, exist_ok=True)
        
        trainer_file = trainers_dir / f"{trainer_name}.py"
        trainer_code = f'''"""Custom trainer with {num_epochs} epochs."""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class {trainer_name}(nnUNetTrainer):
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.num_epochs = {num_epochs}
'''
        
        with open(trainer_file, 'w') as f:
            f.write(trainer_code)
        
        print(f"✅ Created trainer: {trainer_name}")
        return trainer_name
        
    except Exception as e:
        print(f"⚠️ Could not create custom trainer: {e}")
        return None


def train_model(num_epochs: int = None, fold: int = None):
    """Train the fullres model."""
    print_header("TRAINING: 3d_fullres PlainConvUNet")
    
    if num_epochs is None:
        num_epochs = CONFIG["num_epochs"]
    if fold is None:
        fold = CONFIG["fold"]
    
    print(f"Configuration: 3d_fullres")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {num_epochs}")
    print(f"Fold: {fold} (80% train, 20% validation)")
    
    setup_environment()
    
    # Build command (no --npz to avoid blosc2 issues)
    cmd = f"nnUNetv2_train {CONFIG['dataset_id']} 3d_fullres {fold}"
    
    if num_epochs != 1000:
        trainer_name = create_custom_trainer(num_epochs)
        if trainer_name:
            cmd += f" -tr {trainer_name}"
    
    print(f"\nCommand: {cmd}")
    print("\n🚀 Starting training...\n")
    
    result = os.system(cmd)
    gc.collect()
    
    if result == 0:
        print("\n✅ Training complete!")
    else:
        print(f"\n⚠️ Training finished with return code: {result}")
    
    return result


# ============================================================================
# INFERENCE
# ============================================================================
def run_inference(num_epochs: int = None, fold: int = None):
    """Run inference on training images."""
    print_header("RUNNING INFERENCE")
    
    if fold is None:
        fold = CONFIG["fold"]
    
    setup_environment()
    
    input_folder = Path(CONFIG["nnunet_raw"]) / CONFIG["dataset_name"] / "imagesTr"
    output_folder = Path(CONFIG["output_dir"]) / "predictions"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    cmd = f"nnUNetv2_predict -i {input_folder} -o {output_folder} -d {CONFIG['dataset_id']} -c 3d_fullres -f {fold}"
    
    if num_epochs and num_epochs != 1000:
        cmd += f" -tr nnUNetTrainer_{num_epochs}epochs"
    
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Command: {cmd}")
    
    result = os.system(cmd)
    gc.collect()
    
    if result == 0:
        print("\n✅ Inference complete!")
    else:
        print(f"\n⚠️ Inference finished with return code: {result}")
    
    return result


# ============================================================================
# EVALUATION
# ============================================================================
def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient."""
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 1.0
    intersection = np.sum(pred & gt)
    return 2.0 * intersection / (np.sum(pred) + np.sum(gt))


def evaluate_predictions():
    """Evaluate predictions against ground truth."""
    print_header("EVALUATING PREDICTIONS")
    
    pred_dir = Path(CONFIG["output_dir"]) / "predictions"
    dataset_dir = Path(CONFIG["kits23_dir"])
    
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    print(f"Evaluating {len(pred_files)} predictions...")
    
    results = {"kidney": [], "tumor": [], "cyst": []}
    
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        case_name = pred_file.stem.replace(".nii", "")
        gt_path = dataset_dir / case_name / "segmentation.nii.gz"
        
        if not gt_path.exists():
            continue
        
        pred = nib.load(pred_file).get_fdata().astype(np.int32)
        gt = nib.load(gt_path).get_fdata().astype(np.int32)
        
        results["kidney"].append(compute_dice(pred == 1, gt == 1))
        results["tumor"].append(compute_dice(pred == 2, gt == 2))
        results["cyst"].append(compute_dice(pred == 3, gt == 3))
    
    if not results["kidney"]:
        print("❌ No predictions to evaluate")
        return {}
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"{'Class':<10} {'Dice':<15} {'Std':<10}")
    print("-" * 35)
    
    for label, scores in results.items():
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"{label:<10} {mean:.4f}          ±{std:.4f}")
    
    avg = np.mean([np.mean(v) for v in results.values()])
    print("-" * 35)
    print(f"{'AVERAGE':<10} {avg:.4f}")
    print("=" * 50)
    
    # Save results
    with open(Path(CONFIG["output_dir"]) / "evaluation_results.json", 'w') as f:
        json.dump({k: {"mean": float(np.mean(v)), "std": float(np.std(v))} 
                   for k, v in results.items()}, f, indent=2)
    
    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_quick_test():
    """Run quick test with 20 cases, 5 epochs, fold 0."""
    print_header("V2 QUICK TEST - Single Model, Proper Validation")
    print(f"Cases: {CONFIG['quick_cases']}")
    print(f"Epochs: {CONFIG['quick_epochs']}")
    print(f"Fold: 0 (80% train, 20% val)")
    print(f"Model: 3d_fullres only")
    
    setup_directories()
    setup_environment()
    
    # Step 1: Convert data
    print("\n📋 Step 1/4: Data Conversion")
    convert_kits23_to_nnunet(CONFIG["quick_cases"])
    
    # Step 2: Preprocess
    print("\n📋 Step 2/4: Preprocessing")
    run_preprocessing()
    
    # Step 3: Train
    print("\n📋 Step 3/4: Training")
    train_model(CONFIG["quick_epochs"], fold=0)
    
    # Step 4: Inference
    print("\n📋 Step 4/4: Inference")
    run_inference(CONFIG["quick_epochs"], fold=0)
    
    # Evaluate
    print("\n📋 Evaluation")
    evaluate_predictions()
    
    print("\n" + "=" * 60)
    print("✅ V2 PIPELINE COMPLETE!")
    print("=" * 60)


def run_full_training():
    """Run full training with all cases, 1000 epochs."""
    print_header("V2 FULL TRAINING")
    
    setup_directories()
    setup_environment()
    
    convert_kits23_to_nnunet(None)  # All cases
    run_preprocessing()
    train_model(1000, fold=0)
    run_inference(1000, fold=0)
    evaluate_predictions()


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='KiTS23 V2 - Simplified Pipeline')
    parser.add_argument('--mode', type=str, default='quick_test',
                       choices=['quick_test', 'full', 'preprocess', 'train', 'inference', 'evaluate'],
                       help='Mode to run')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--cases', type=int, default=None, help='Max cases')
    parser.add_argument('--fold', type=int, default=0, help='Fold to use (0-4)')
    
    args = parser.parse_args()
    
    if args.mode == 'quick_test':
        run_quick_test()
    elif args.mode == 'full':
        run_full_training()
    elif args.mode == 'preprocess':
        setup_directories()
        setup_environment()
        convert_kits23_to_nnunet(args.cases or CONFIG["quick_cases"])
        run_preprocessing()
    elif args.mode == 'train':
        setup_directories()
        train_model(args.epochs or CONFIG["quick_epochs"], args.fold)
    elif args.mode == 'inference':
        setup_directories()
        run_inference(args.epochs, args.fold)
    elif args.mode == 'evaluate':
        evaluate_predictions()


if __name__ == "__main__":
    main()
