
import os
import shutil
import json
import argparse
import subprocess
import glob
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_KiTS23"
BASE_DIR = Path("./output/kits23_nnunet").absolute()
RAW_DIR = BASE_DIR / "nnUNet_raw"
PREPROCESSED_DIR = BASE_DIR / "nnUNet_preprocessed"
RESULTS_DIR = BASE_DIR / "nnUNet_results"

# Source Data
SOURCE_DIR = Path("./kits23/dataset").absolute()

# Set environment variables for nnU-Net
os.environ["nnUNet_raw"] = str(RAW_DIR)
os.environ["nnUNet_preprocessed"] = str(PREPROCESSED_DIR)
os.environ["nnUNet_results"] = str(RESULTS_DIR)

def setup_directories():
    """Creates the necessary nnU-Net directory structure."""
    print(f"Creating directories at {BASE_DIR}...")
    for d in [RAW_DIR, PREPROCESSED_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Dataset specific folders
    (RAW_DIR / DATASET_NAME / "imagesTr").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / DATASET_NAME / "labelsTr").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / DATASET_NAME / "imagesTs").mkdir(parents=True, exist_ok=True)

def convert_dataset():
    """Converts the KiTS23 dataset to nnU-Net format."""
    print("Converting dataset to nnU-Net format...")
    
    cases = sorted(list(SOURCE_DIR.glob("case_*")))
    train_cases = cases # For now, use all available as training (split handled by nnUNet)
    
    print(f"Found {len(cases)} cases.")
    
    # 1. Create dataset.json
    json_dict = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "kidney": 1,
            "tumor": 2,
            "cyst": 3
        },
        "numTraining": len(train_cases),
        "file_ending": ".nii.gz"
    }
    
    with open(RAW_DIR / DATASET_NAME / "dataset.json", 'w') as f:
        json.dump(json_dict, f, indent=4)
        
    # 2. Copy files
    print("Copying and renaming files...")
    for case_path in tqdm(train_cases):
        case_id = case_path.name
        
        # Input paths
        img_src = case_path / "imaging.nii.gz"
        seg_src = case_path / "segmentation.nii.gz"
        
        if not img_src.exists() or not seg_src.exists():
            print(f"Skipping {case_id}: Missing imaging or segmentation file.")
            continue
            
        # Target paths (nnU-Net convention)
        # Image: case_Identifier_0000.nii.gz
        # Label: case_Identifier.nii.gz
        img_dst = RAW_DIR / DATASET_NAME / "imagesTr" / f"{case_id}_0000.nii.gz"
        seg_dst = RAW_DIR / DATASET_NAME / "labelsTr" / f"{case_id}.nii.gz"
        
        if not img_dst.exists():
            shutil.copy(img_src, img_dst)
        if not seg_dst.exists():
            shutil.copy(seg_src, seg_dst)

def run_command(cmd):
    """Runs a shell command with real-time output."""
    print(f"\nExample running: {cmd}")
    # Pass environment variables
    env = os.environ.copy()
    
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        env=env,
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with code {process.returncode}")

def main():
    parser = argparse.ArgumentParser(description="KiTS23 Official nnU-Net Pipeline")
    parser.add_argument("--step", type=str, choices=["all", "convert", "plan", "train_lowres", "train_fullres", "predict"], default="all")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs (default: 1000)")
    parser.add_argument("--fold", type=int, default=0, help="Fold to train (0-4)")
    args = parser.parse_args()
    
    setup_directories()
    
    if args.step in ["all", "convert"]:
        convert_dataset()
        
    if args.step in ["all", "plan"]:
        print("\n\n=== Running Verification & Planning ===")
        # verify_dataset_integrity is crucial
        run_command(f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity -c 3d_fullres 3d_lowres")
        
    if args.step in ["all", "train_lowres"]:
        print(f"\n\n=== Training 3D Lowres (Fold {args.fold}) ===")
        # Note: nnU-Net doesn't easily convert epochs via CLI, usually requires modifying plans or trainer.
        # But we can just run it. If user wants fewer epochs, we might need a custom trainer, 
        # but for now we run standard to fix the empty Dice issue.
        if args.epochs != 1000:
            print(f"WARNING: Custom epochs {args.epochs} requested. This requires a custom trainer class or direct code modification.")
            print("Running with default 1000 epochs (Ctrl+C to stop early - nnU-Net saves checkpoints frequently).")
        
        run_command(f"nnUNetv2_train {DATASET_ID} 3d_lowres {args.fold}")
        
    if args.step in ["all", "train_fullres"]:
        print(f"\n\n=== Training 3D Fullres (Fold {args.fold}) ===")
        run_command(f"nnUNetv2_train {DATASET_ID} 3d_fullres {args.fold}")

if __name__ == "__main__":
    main()
