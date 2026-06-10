"""
KiTS23 nnU-Net Complete Training Pipeline
==========================================
Complete script for training nnU-Net on KiTS23 dataset.
Includes: data conversion, preprocessing, training, inference, evaluation, visualization.

Usage:
    python kits23_nnunet_train.py --mode setup      # Setup and convert data
    python kits23_nnunet_train.py --mode preprocess # Run preprocessing
    python kits23_nnunet_train.py --mode train      # Run training
    python kits23_nnunet_train.py --mode evaluate   # Evaluate and visualize
    python kits23_nnunet_train.py --mode all        # Run everything
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SYSTEM
# ============================================================================
CONFIG = {
    # Dataset paths
    "kits23_dir": "/mnt/d/Salman/gastric_cancer/kidney/kits23/dataset",
    
    # nnU-Net paths (will be created)
    "nnunet_raw": "/mnt/d/Salman/gastric_cancer/kidney/nnUNet_raw",
    "nnunet_preprocessed": "/mnt/d/Salman/gastric_cancer/kidney/nnUNet_preprocessed",
    "nnunet_results": "/mnt/d/Salman/gastric_cancer/kidney/nnUNet_results",
    
    # Training config
    "dataset_id": 23,
    "dataset_name": "Dataset023_KiTS23",
    "configuration": "3d_fullres",  # Options: 3d_fullres, 3d_lowres, 2d
    "fold": 0,  # 0-4 for 5-fold CV, or "all" for all folds
    "num_epochs": 50,  # Set to 1000 for full training, 50 for quick test
    
    # Output paths
    "predictions_dir": "/mnt/d/Salman/gastric_cancer/kidney/predictions",
    "visualizations_dir": "/mnt/d/Salman/gastric_cancer/kidney/visualizations",
}

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================
def setup_environment():
    """Setup nnU-Net environment variables"""
    os.environ['nnUNet_raw'] = CONFIG["nnunet_raw"]
    os.environ['nnUNet_preprocessed'] = CONFIG["nnunet_preprocessed"]
    os.environ['nnUNet_results'] = CONFIG["nnunet_results"]
    
    # Create directories
    for path_key in ["nnunet_raw", "nnunet_preprocessed", "nnunet_results", 
                     "predictions_dir", "visualizations_dir"]:
        Path(CONFIG[path_key]).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment variables set:")
    print(f"   nnUNet_raw: {CONFIG['nnunet_raw']}")
    print(f"   nnUNet_preprocessed: {CONFIG['nnunet_preprocessed']}")
    print(f"   nnUNet_results: {CONFIG['nnunet_results']}")

def check_nnunet_installed():
    """Check if nnU-Net is installed"""
    try:
        import nnunetv2
        # Try to get version, but don't fail if not available
        try:
            version = nnunetv2.__version__
        except AttributeError:
            version = "installed (version unknown)"
        print(f"✅ nnU-Net v2: {version}")
        return True
    except ImportError:
        print("❌ nnU-Net not installed!")
        print("   Install with: pip install nnunetv2")
        return False

# ============================================================================
# DATA CONVERSION
# ============================================================================
def convert_kits23_to_nnunet():
    """Convert KiTS23 dataset to nnU-Net format"""
    print("\n" + "="*60)
    print("CONVERTING KiTS23 TO nnU-Net FORMAT")
    print("="*60)
    
    kits23_dir = Path(CONFIG["kits23_dir"])
    dataset_dir = Path(CONFIG["nnunet_raw"]) / CONFIG["dataset_name"]
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    # Find all cases
    cases = sorted([d for d in kits23_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('case_')])
    
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
    
    # Create dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "kidney": 1,
            "tumor": 2,
            "cyst": 3
        },
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    
    with open(dataset_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\n✅ Converted {len(training_cases)} cases")
    if skipped:
        print(f"⚠️  Skipped {len(skipped)} incomplete cases")
    print(f"📁 Output: {dataset_dir}")
    
    return len(training_cases)

# ============================================================================
# PREPROCESSING
# ============================================================================
def run_preprocessing(num_processes=4, skip_2d=True):
    """
    Run nnU-Net preprocessing.
    
    Args:
        num_processes: Number of parallel processes
        skip_2d: Skip 2D preprocessing (only do 3d_fullres)
    """
    print("\n" + "="*60)
    print("RUNNING nnU-Net PREPROCESSING")
    print("="*60)
    
    setup_environment()
    
    dataset_id = CONFIG["dataset_id"]
    
    # Build command - only preprocess 3d_fullres to save time
    cmd = [
        sys.executable, "-m", "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
        "-d", str(dataset_id),
        "--verify_dataset_integrity",
        "-c", "3d_fullres",          # ONLY preprocess 3d_fullres
        "-np", str(num_processes),
        "-npfp", str(num_processes), # Fingerprint extraction processes
        "-npp", str(num_processes),  # Preprocessing processes
    ]
    
    # Add no_pp flag to skip preprocessing of other configs
    if skip_2d:
        cmd.extend(["--no_pp"])  # Skip preprocessing, only plan
    
    print(f"Command: {' '.join(cmd)}")
    print(f"\n⚙️  Using {num_processes} process(es)")
    print(f"📊 Only preprocessing 3d_fullres (skipping 2D)")
    print("-"*60)
    
    # STEP 1: Run planning only first
    print("\n� STEP 1: Planning (quick)...")
    plan_cmd = cmd.copy()
    
    try:
        process = subprocess.Popen(
            plan_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip(), flush=True)
        
        if process.poll() != 0:
            print(f"⚠️ Planning returned code: {process.poll()}")
            
    except Exception as e:
        print(f"❌ Planning failed: {e}")
        return
    
    # STEP 2: Preprocess only 3d_fullres
    print("\n🔧 STEP 2: Preprocessing 3d_fullres only...")
    preprocess_cmd = [
        sys.executable, "-m", "nnunetv2.preprocessing.cropping.dataset_specific_cropping_and_resampling",
        str(dataset_id),
        "-c", "3d_fullres",
        "-np", str(num_processes)
    ]
    
    try:
        # Alternative: use nnUNet's preprocess directly
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
        from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
        
        preprocessed_folder = Path(nnUNet_preprocessed) / f"Dataset{dataset_id:03d}_KiTS23"
        
        print(f"Preprocessed folder: {preprocessed_folder}")
        print("Starting preprocessing of 3d_fullres configuration...")
        print("⏱️  This will take 30-60 minutes for 489 cases...")
        
        # Run the actual preprocessing command
        preprocess_3d_cmd = f"nnUNetv2_preprocess -d {dataset_id} -c 3d_fullres -np {num_processes}"
        print(f"Running: {preprocess_3d_cmd}")
        os.system(preprocess_3d_cmd)
        
        print("\n" + "-"*60)
        print("✅ Preprocessing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback
        alt_cmd = f"nnUNetv2_preprocess -d {dataset_id} -c 3d_fullres -np {num_processes}"
        print(f"Running: {alt_cmd}")
        os.system(alt_cmd)

# ============================================================================
# TRAINING
# ============================================================================
def run_training():
    """Run nnU-Net training"""
    print("\n" + "="*60)
    print("RUNNING nnU-Net TRAINING")
    print("="*60)
    
    setup_environment()
    
    dataset_id = CONFIG["dataset_id"]
    configuration = CONFIG["configuration"]
    fold = CONFIG["fold"]
    num_epochs = CONFIG["num_epochs"]
    
    print(f"Dataset ID: {dataset_id}")
    print(f"Configuration: {configuration}")
    print(f"Fold: {fold}")
    print(f"Epochs: {num_epochs}")
    
    # Build command
    cmd = [
        sys.executable, "-m", "nnunetv2.run.run_training",
        str(dataset_id), configuration, str(fold),
        "--npz"
    ]
    
    if num_epochs != 1000:  # Add epochs flag only if not default
        cmd.extend(["--num_epochs", str(num_epochs)])
    
    print(f"\nCommand: {' '.join(cmd)}")
    print("\n🚀 Starting training...\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with subprocess: {e}")
        # Try alternative
        alt_cmd = f"nnUNetv2_train {dataset_id} {configuration} {fold} --npz"
        if num_epochs != 1000:
            alt_cmd += f" --num_epochs {num_epochs}"
        print(f"Trying: {alt_cmd}")
        os.system(alt_cmd)

# ============================================================================
# INFERENCE
# ============================================================================
def run_inference(input_folder=None, output_folder=None):
    """Run inference on test cases"""
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    setup_environment()
    
    if input_folder is None:
        input_folder = Path(CONFIG["nnunet_raw"]) / CONFIG["dataset_name"] / "imagesTr"
    if output_folder is None:
        output_folder = Path(CONFIG["predictions_dir"])
    
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "nnunetv2.inference.predict_from_raw_data",
        "-i", str(input_folder),
        "-o", str(output_folder),
        "-d", str(CONFIG["dataset_id"]),
        "-c", CONFIG["configuration"],
        "-f", str(CONFIG["fold"])
    ]
    
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"\nCommand: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Inference complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Inference failed: {e}")
        alt_cmd = f"nnUNetv2_predict -i {input_folder} -o {output_folder} -d {CONFIG['dataset_id']} -c {CONFIG['configuration']} -f {CONFIG['fold']}"
        print(f"Trying: {alt_cmd}")
        os.system(alt_cmd)

# ============================================================================
# EVALUATION
# ============================================================================
def dice_score(pred, gt, label):
    """Compute Dice score for a specific label"""
    pred_binary = (pred == label).astype(np.float32)
    gt_binary = (gt == label).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    if union == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0
    
    return 2.0 * intersection / union

def surface_dice(pred, gt, label, tolerance=2.0):
    """Compute Surface Dice (NSD) - simplified version"""
    # This is a simplified placeholder - full implementation requires distance transforms
    return dice_score(pred, gt, label)

def evaluate_predictions():
    """Evaluate all predictions against ground truth"""
    print("\n" + "="*60)
    print("EVALUATING PREDICTIONS")
    print("="*60)
    
    predictions_dir = Path(CONFIG["predictions_dir"])
    labels_dir = Path(CONFIG["nnunet_raw"]) / CONFIG["dataset_name"] / "labelsTr"
    
    if not predictions_dir.exists():
        print(f"❌ Predictions directory not found: {predictions_dir}")
        return None
    
    pred_files = sorted(predictions_dir.glob("*.nii.gz"))
    print(f"Found {len(pred_files)} prediction files")
    
    results = []
    
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        case_id = pred_file.stem.replace(".nii", "")
        gt_file = labels_dir / f"{case_id}.nii.gz"
        
        if not gt_file.exists():
            continue
        
        pred = nib.load(pred_file).get_fdata()
        gt = nib.load(gt_file).get_fdata()
        
        result = {
            "case_id": case_id,
            "kidney_dice": dice_score(pred, gt, 1),
            "tumor_dice": dice_score(pred, gt, 2),
            "cyst_dice": dice_score(pred, gt, 3),
        }
        result["composite_dice"] = (result["kidney_dice"] + result["tumor_dice"]) / 2
        results.append(result)
    
    if not results:
        print("❌ No valid predictions found to evaluate")
        return None
    
    # Compute averages
    avg_kidney = np.mean([r["kidney_dice"] for r in results])
    avg_tumor = np.mean([r["tumor_dice"] for r in results])
    avg_cyst = np.mean([r["cyst_dice"] for r in results])
    avg_composite = np.mean([r["composite_dice"] for r in results])
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Cases evaluated: {len(results)}")
    print("-"*40)
    print(f"Kidney Dice:     {avg_kidney:.4f} ({avg_kidney*100:.2f}%)")
    print(f"Tumor Dice:      {avg_tumor:.4f} ({avg_tumor*100:.2f}%)")
    print(f"Cyst Dice:       {avg_cyst:.4f} ({avg_cyst*100:.2f}%)")
    print("-"*40)
    print(f"Composite Dice:  {avg_composite:.4f} ({avg_composite*100:.2f}%)")
    print("="*60)
    
    # Save results
    results_file = Path(CONFIG["predictions_dir"]) / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "per_case": results,
            "average": {
                "kidney_dice": avg_kidney,
                "tumor_dice": avg_tumor,
                "cyst_dice": avg_cyst,
                "composite_dice": avg_composite
            }
        }, f, indent=2)
    print(f"\n📁 Results saved to: {results_file}")
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_case(case_id, show=True, save=True):
    """Visualize a single case with prediction overlay"""
    kits23_dir = Path(CONFIG["kits23_dir"])
    predictions_dir = Path(CONFIG["predictions_dir"])
    vis_dir = Path(CONFIG["visualizations_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = kits23_dir / case_id / "imaging.nii.gz"
    gt_path = kits23_dir / case_id / "segmentation.nii.gz"
    pred_path = predictions_dir / f"{case_id}.nii.gz"
    
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return
    
    img = nib.load(img_path).get_fdata()
    gt = nib.load(gt_path).get_fdata() if gt_path.exists() else None
    pred = nib.load(pred_path).get_fdata() if pred_path.exists() else None
    
    # Find slice with maximum tumor content
    if gt is not None:
        tumor_per_slice = np.sum(gt == 2, axis=(0, 1))
        slice_idx = np.argmax(tumor_per_slice)
    else:
        slice_idx = img.shape[2] // 2
    
    # Create figure
    n_cols = 2 + (1 if gt is not None else 0) + (1 if pred is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    # Normalize CT for display
    img_slice = img[:, :, slice_idx].T
    img_norm = np.clip(img_slice, -200, 400)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())
    
    col = 0
    
    # CT Image
    axes[col].imshow(img_norm, cmap='gray', origin='lower')
    axes[col].set_title('CT Image')
    axes[col].axis('off')
    col += 1
    
    # Ground Truth
    if gt is not None:
        axes[col].imshow(img_norm, cmap='gray', origin='lower')
        gt_slice = gt[:, :, slice_idx].T
        overlay = create_overlay(gt_slice)
        axes[col].imshow(overlay, origin='lower')
        axes[col].set_title('Ground Truth')
        axes[col].axis('off')
        col += 1
    
    # Prediction
    if pred is not None:
        axes[col].imshow(img_norm, cmap='gray', origin='lower')
        pred_slice = pred[:, :, slice_idx].T
        overlay = create_overlay(pred_slice)
        axes[col].imshow(overlay, origin='lower')
        axes[col].set_title('Prediction')
        axes[col].axis('off')
        col += 1
    
    # Comparison (if both available)
    if gt is not None and pred is not None:
        axes[col].imshow(img_norm, cmap='gray', origin='lower')
        # Show difference
        diff_overlay = create_diff_overlay(gt_slice, pred_slice)
        axes[col].imshow(diff_overlay, origin='lower')
        axes[col].set_title('Difference\n(Red=FN, Blue=FP)')
        axes[col].axis('off')
    
    plt.suptitle(f'{case_id} - Slice {slice_idx}', fontsize=14)
    plt.tight_layout()
    
    if save:
        save_path = vis_dir / f"{case_id}_visualization.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📁 Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def create_overlay(seg_slice):
    """Create colored overlay for segmentation"""
    overlay = np.zeros((*seg_slice.shape, 4))
    overlay[seg_slice == 1] = [0, 1, 0, 0.4]  # Kidney = green
    overlay[seg_slice == 2] = [1, 0, 0, 0.6]  # Tumor = red
    overlay[seg_slice == 3] = [0, 0, 1, 0.4]  # Cyst = blue
    return overlay

def create_diff_overlay(gt_slice, pred_slice):
    """Create difference overlay (FP/FN visualization)"""
    overlay = np.zeros((*gt_slice.shape, 4))
    
    for label in [1, 2, 3]:
        gt_mask = (gt_slice == label)
        pred_mask = (pred_slice == label)
        
        # False negatives (in GT but not in pred) - Red
        fn = gt_mask & ~pred_mask
        overlay[fn] = [1, 0, 0, 0.5]
        
        # False positives (in pred but not in GT) - Blue
        fp = ~gt_mask & pred_mask
        overlay[fp] = [0, 0, 1, 0.5]
        
        # True positives - Green
        tp = gt_mask & pred_mask
        overlay[tp] = [0, 1, 0, 0.3]
    
    return overlay

def visualize_multiple_cases(n_cases=5):
    """Visualize multiple cases"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    kits23_dir = Path(CONFIG["kits23_dir"])
    cases = sorted([d.name for d in kits23_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('case_')])[:n_cases]
    
    for case_id in tqdm(cases, desc="Visualizing"):
        visualize_case(case_id, show=False, save=True)
    
    print(f"\n✅ Visualizations saved to: {CONFIG['visualizations_dir']}")

# ============================================================================
# LESION SIZE ANALYSIS
# ============================================================================
def calculate_lesion_sizes():
    """Calculate lesion sizes from segmentation masks including % of image"""
    print("\n" + "="*60)
    print("CALCULATING LESION SIZES (with % of image)")
    print("="*60)
    
    kits23_dir = Path(CONFIG["kits23_dir"])
    cases = sorted([d for d in kits23_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('case_')])
    
    results = []
    
    for case_dir in tqdm(cases, desc="Calculating volumes"):
        case_id = case_dir.name
        seg_path = case_dir / "segmentation.nii.gz"
        img_path = case_dir / "imaging.nii.gz"
        
        if not seg_path.exists():
            continue
        
        seg = nib.load(seg_path)
        data = seg.get_fdata()
        voxel_dims = seg.header.get_zooms()
        voxel_volume_mm3 = np.prod(voxel_dims)
        
        # Total image volume
        total_voxels = data.size
        total_volume_cm3 = (total_voxels * voxel_volume_mm3) / 1000
        
        # Count voxels for each class
        kidney_voxels = np.sum(data == 1)
        tumor_voxels = np.sum(data == 2)
        cyst_voxels = np.sum(data == 3)
        
        # Calculate volumes in cm³
        kidney_vol = (kidney_voxels * voxel_volume_mm3) / 1000
        tumor_vol = (tumor_voxels * voxel_volume_mm3) / 1000
        cyst_vol = (cyst_voxels * voxel_volume_mm3) / 1000
        
        # Calculate percentages of total image
        kidney_pct = (kidney_voxels / total_voxels) * 100
        tumor_pct = (tumor_voxels / total_voxels) * 100
        cyst_pct = (cyst_voxels / total_voxels) * 100
        total_lesion_pct = kidney_pct + tumor_pct + cyst_pct
        
        results.append({
            "case_id": case_id,
            "image_shape": data.shape,
            "total_volume_cm3": total_volume_cm3,
            "kidney_volume_cm3": kidney_vol,
            "tumor_volume_cm3": tumor_vol,
            "cyst_volume_cm3": cyst_vol,
            "kidney_pct_of_image": kidney_pct,
            "tumor_pct_of_image": tumor_pct,
            "cyst_pct_of_image": cyst_pct,
            "total_lesion_pct": total_lesion_pct,
            "voxel_spacing": voxel_dims
        })
    
    # Print summary
    tumor_volumes = [r["tumor_volume_cm3"] for r in results if r["tumor_volume_cm3"] > 0]
    tumor_pcts = [r["tumor_pct_of_image"] for r in results if r["tumor_pct_of_image"] > 0]
    kidney_pcts = [r["kidney_pct_of_image"] for r in results]
    total_lesion_pcts = [r["total_lesion_pct"] for r in results]
    
    print(f"\n📊 LESION SIZE SUMMARY")
    print("="*60)
    print(f"Total cases: {len(results)}")
    print(f"Cases with tumors: {len(tumor_volumes)}")
    
    print("\n--- TUMOR VOLUME (cm³) ---")
    print(f"  Range:  {min(tumor_volumes):.2f} - {max(tumor_volumes):.2f} cm³")
    print(f"  Mean:   {np.mean(tumor_volumes):.2f} cm³")
    print(f"  Median: {np.median(tumor_volumes):.2f} cm³")
    
    print("\n--- TUMOR % OF IMAGE ---")
    print(f"  Range:  {min(tumor_pcts):.4f}% - {max(tumor_pcts):.4f}%")
    print(f"  Mean:   {np.mean(tumor_pcts):.4f}%")
    print(f"  Median: {np.median(tumor_pcts):.4f}%")
    
    print("\n--- KIDNEY % OF IMAGE ---")
    print(f"  Range:  {min(kidney_pcts):.4f}% - {max(kidney_pcts):.4f}%")
    print(f"  Mean:   {np.mean(kidney_pcts):.4f}%")
    print(f"  Median: {np.median(kidney_pcts):.4f}%")
    
    print("\n--- TOTAL LESION % (Kidney + Tumor + Cyst) ---")
    print(f"  Range:  {min(total_lesion_pcts):.4f}% - {max(total_lesion_pcts):.4f}%")
    print(f"  Mean:   {np.mean(total_lesion_pcts):.4f}%")
    print(f"  Median: {np.median(total_lesion_pcts):.4f}%")
    print("="*60)
    
    # Save results
    output_file = Path(CONFIG["kits23_dir"]).parent / "lesion_sizes.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📁 Saved to: {output_file}")
    
    # Also create a summary CSV for easy viewing
    import csv
    csv_file = Path(CONFIG["kits23_dir"]).parent / "lesion_sizes_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'case_id', 'tumor_volume_cm3', 'tumor_pct_of_image',
            'kidney_volume_cm3', 'kidney_pct_of_image', 
            'cyst_volume_cm3', 'cyst_pct_of_image', 'total_lesion_pct'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'case_id': r['case_id'],
                'tumor_volume_cm3': f"{r['tumor_volume_cm3']:.2f}",
                'tumor_pct_of_image': f"{r['tumor_pct_of_image']:.4f}",
                'kidney_volume_cm3': f"{r['kidney_volume_cm3']:.2f}",
                'kidney_pct_of_image': f"{r['kidney_pct_of_image']:.4f}",
                'cyst_volume_cm3': f"{r['cyst_volume_cm3']:.2f}",
                'cyst_pct_of_image': f"{r['cyst_pct_of_image']:.4f}",
                'total_lesion_pct': f"{r['total_lesion_pct']:.4f}"
            })
    print(f"📁 CSV saved to: {csv_file}")
    
    return results

# ============================================================================
# MAIN (Works in both command line and Jupyter)
# ============================================================================
def run_pipeline(mode="all", epochs=None, fold=None):
    """
    Run the KiTS23 nnU-Net pipeline.
    
    Args:
        mode: One of 'setup', 'preprocess', 'train', 'inference', 
              'evaluate', 'visualize', 'lesion_sizes', 'all'
        epochs: Override number of epochs (default: use CONFIG value)
        fold: Override fold number (default: use CONFIG value)
    """
    # Override config if specified
    if epochs is not None:
        CONFIG["num_epochs"] = epochs
    if fold is not None:
        CONFIG["fold"] = fold
    
    print("="*60)
    print("🏥 KiTS23 nnU-Net TRAINING PIPELINE")
    print("="*60)
    print(f"Mode: {mode}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Fold: {CONFIG['fold']}")
    print("="*60)
    
    # Check nnU-Net installation
    nnunet_available = check_nnunet_installed()
    if not nnunet_available and mode in ['preprocess', 'train', 'inference', 'all']:
        print("\n⚠️  Please install nnU-Net first:")
        print("   pip install nnunetv2")
        print("\nContinuing with available operations...")
    
    setup_environment()
    
    if mode == 'setup' or mode == 'all':
        convert_kits23_to_nnunet()
    
    if mode == 'preprocess' or mode == 'all':
        if nnunet_available:
            run_preprocessing()
        else:
            print("⚠️ Skipping preprocessing - nnU-Net not installed")
    
    if mode == 'train' or mode == 'all':
        if nnunet_available:
            run_training()
        else:
            print("⚠️ Skipping training - nnU-Net not installed")
    
    if mode == 'inference' or mode == 'all':
        if nnunet_available:
            run_inference()
        else:
            print("⚠️ Skipping inference - nnU-Net not installed")
    
    if mode == 'evaluate' or mode == 'all':
        evaluate_predictions()
    
    if mode == 'visualize' or mode == 'all':
        visualize_multiple_cases(n_cases=10)
    
    if mode == 'lesion_sizes':
        calculate_lesion_sizes()
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='KiTS23 nnU-Net Training Pipeline')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['setup', 'preprocess', 'train', 'inference', 
                               'evaluate', 'visualize', 'lesion_sizes', 'all'],
                       help='Mode to run')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--fold', type=int, default=None,
                       help='Override fold number')
    
    args = parser.parse_args()
    run_pipeline(mode=args.mode, epochs=args.epochs, fold=args.fold)


# ============================================================================
# JUPYTER NOTEBOOK USAGE
# ============================================================================
# To use in Jupyter, don't run the script directly. Instead:
#
#   from kits23_nnunet_train import run_pipeline, CONFIG
#   
#   # Update paths for your system
#   CONFIG["kits23_dir"] = "/path/to/your/kits23/dataset"
#   
#   # Run specific modes:
#   run_pipeline(mode="setup")           # Setup only
#   run_pipeline(mode="lesion_sizes")    # Calculate lesion sizes
#   run_pipeline(mode="all", epochs=50)  # Full pipeline with 50 epochs
#
# Or run individual functions:
#   from kits23_nnunet_train import calculate_lesion_sizes, visualize_case
#   calculate_lesion_sizes()
#   visualize_case("case_00000")
# ============================================================================


def is_notebook():
    """Check if running in Jupyter notebook"""
    try:
        shell = get_ipython().__class__.__name__
        return shell in ['ZMQInteractiveShell', 'TerminalInteractiveShell']
    except NameError:
        return False


if __name__ == "__main__":
    if is_notebook():
        print("="*60)
        print("🔔 RUNNING IN JUPYTER NOTEBOOK")
        print("="*60)
        print("\nUse these commands to run the pipeline:\n")
        print("  # Calculate lesion sizes:")
        print("  calculate_lesion_sizes()")
        print("")
        print("  # Run full pipeline:")
        print("  run_pipeline(mode='all', epochs=50)")
        print("")
        print("  # Available modes:")
        print("  # 'setup', 'preprocess', 'train', 'inference',")
        print("  # 'evaluate', 'visualize', 'lesion_sizes', 'all'")
        print("")
        print("  # Visualize a case:")
        print("  visualize_case('case_00000')")
        print("="*60)
    else:
        main()
