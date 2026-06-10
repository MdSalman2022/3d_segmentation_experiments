"""
KiTS23 Manual Preprocessing for nnU-Net
========================================
This script preprocesses cases ONE BY ONE to avoid WSL memory crashes.
Run this AFTER the planning step is complete.
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "kits23_dir": "/mnt/d/Salman/gastric_cancer/kidney/kits23/dataset",
    "nnunet_raw": "/mnt/d/Salman/gastric_cancer/kidney/nnUNet_raw",
    "nnunet_preprocessed": "/mnt/d/Salman/gastric_cancer/kidney/nnUNet_preprocessed",
    "dataset_id": 23,
    "dataset_name": "Dataset023_KiTS23",
}

# Set environment
os.environ['nnUNet_raw'] = CONFIG["nnunet_raw"]
os.environ['nnUNet_preprocessed'] = CONFIG["nnunet_preprocessed"]

def preprocess_single_case(case_id, plans, configuration="3d_fullres"):
    """Preprocess a single case manually"""
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    
    dataset_name = maybe_convert_to_dataset_name(CONFIG["dataset_id"])
    
    raw_folder = Path(CONFIG["nnunet_raw"]) / dataset_name
    preprocessed_folder = Path(CONFIG["nnunet_preprocessed"]) / dataset_name
    
    # Get configuration from plans
    config_plans = plans['configurations'][configuration]
    
    # Initialize preprocessor
    preprocessor = DefaultPreprocessor(verbose=False)
    
    # Input files
    image_file = raw_folder / "imagesTr" / f"{case_id}_0000.nii.gz"
    seg_file = raw_folder / "labelsTr" / f"{case_id}.nii.gz"
    
    if not image_file.exists() or not seg_file.exists():
        print(f"  ⚠️ Files not found for {case_id}")
        return False
    
    # Output folder
    output_folder = preprocessed_folder / config_plans['data_identifier']
    output_folder.mkdir(parents=True, exist_ok=True)
    
    output_file = output_folder / f"{case_id}.npz"
    output_pkl = output_folder / f"{case_id}.pkl"
    
    # Skip if already processed
    if output_file.exists() and output_pkl.exists():
        return True  # Already done
    
    try:
        # Run preprocessing
        data, seg, properties = preprocessor.run_case_npy(
            [str(image_file)],
            str(seg_file),
            plans,
            configuration,
            dataset_name
        )
        
        # Save
        np.savez_compressed(output_file, data=data, seg=seg)
        with open(output_pkl, 'wb') as f:
            pickle.dump(properties, f)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {case_id}: {e}")
        return False


def manual_preprocess_all(start_from=0):
    """Preprocess all cases one by one"""
    print("="*60)
    print("MANUAL PREPROCESSING (ONE CASE AT A TIME)")
    print("="*60)
    print("This avoids WSL memory crashes by processing sequentially.\n")
    
    # Load plans
    preprocessed_folder = Path(CONFIG["nnunet_preprocessed"]) / CONFIG["dataset_name"]
    plans_file = preprocessed_folder / "nnUNetPlans.json"
    
    if not plans_file.exists():
        print(f"❌ Plans file not found: {plans_file}")
        print("Please run planning step first:")
        print("  nnUNetv2_plan_and_preprocess -d 23 --verify_dataset_integrity --no_pp")
        return
    
    with open(plans_file, 'r') as f:
        plans = json.load(f)
    
    print(f"✅ Loaded plans from: {plans_file}")
    
    # Find all cases
    raw_folder = Path(CONFIG["nnunet_raw"]) / CONFIG["dataset_name"] / "imagesTr"
    cases = sorted([f.stem.replace("_0000", "") for f in raw_folder.glob("*_0000.nii.gz")])
    
    print(f"📊 Found {len(cases)} cases to preprocess")
    print(f"⏱️  Estimated time: ~1-2 minutes per case")
    print(f"⏱️  Total: ~{len(cases) * 1.5 / 60:.1f} hours")
    print("-"*60)
    
    # Skip already processed
    config = "3d_fullres"
    output_folder = preprocessed_folder / plans['configurations'][config]['data_identifier']
    
    already_done = 0
    for case_id in cases:
        if (output_folder / f"{case_id}.npz").exists():
            already_done += 1
    
    if already_done > 0:
        print(f"ℹ️  {already_done} cases already preprocessed, will skip them")
    
    # Process each case
    success = 0
    failed = []
    
    for i, case_id in enumerate(tqdm(cases[start_from:], desc="Preprocessing")):
        idx = i + start_from
        
        # Skip if already done
        if (output_folder / f"{case_id}.npz").exists():
            success += 1
            continue
        
        result = preprocess_single_case(case_id, plans, config)
        
        if result:
            success += 1
        else:
            failed.append(case_id)
        
        # Periodic save point
        if (idx + 1) % 50 == 0:
            print(f"\n  📍 Checkpoint: {idx + 1}/{len(cases)} done, {len(failed)} failed")
    
    print("\n" + "="*60)
    print(f"✅ PREPROCESSING COMPLETE")
    print(f"   Success: {success}/{len(cases)}")
    if failed:
        print(f"   Failed: {len(failed)} cases")
        print(f"   Failed cases: {failed[:10]}...")
    print("="*60)


def check_preprocessing_status():
    """Check how many cases are preprocessed"""
    preprocessed_folder = Path(CONFIG["nnunet_preprocessed"]) / CONFIG["dataset_name"]
    
    # Check for plans
    plans_file = preprocessed_folder / "nnUNetPlans.json"
    if not plans_file.exists():
        print("❌ Planning not done yet")
        return
    
    with open(plans_file, 'r') as f:
        plans = json.load(f)
    
    # Check 3d_fullres
    config = "3d_fullres"
    output_folder = preprocessed_folder / plans['configurations'][config]['data_identifier']
    
    if output_folder.exists():
        npz_files = list(output_folder.glob("*.npz"))
        print(f"3d_fullres: {len(npz_files)} cases preprocessed")
    else:
        print("3d_fullres: 0 cases preprocessed")
    
    # Total expected
    raw_folder = Path(CONFIG["nnunet_raw"]) / CONFIG["dataset_name"] / "imagesTr"
    total = len(list(raw_folder.glob("*_0000.nii.gz")))
    print(f"Total expected: {total}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check status only")
    parser.add_argument("--start", type=int, default=0, help="Start from case index")
    args = parser.parse_args()
    
    if args.check:
        check_preprocessing_status()
    else:
        manual_preprocess_all(start_from=args.start)
