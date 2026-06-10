
"""
KiTS23 Official nnU-Net Pipeline Orchestrator (Single File Solution)

This script manages the entire lifecycle of the KiTS23 segmentation task using the official nnU-Net v2 library.
It is designed to be portable and run on high-performance hardware (e.g., RTX 5090).

Workflow:
1. `check_setup`: Verifies/installs dependencies (nnunetv2, etc.)
2. `convert`: Transforms KiTS23 dataset -> nnU-Net raw format
3. `plan`: Runs nnUNetv2_plan_and_preprocess (auto-tunes for hardware)
4. `train`: Orchestrates training (supports 3d_fullres, 3d_lowres)
5. `predict`: Runs inference on test cases
6. `evaluate`: Computes Dice scores against ground truth

Usage:
    python kits23_nnunet_pipeline.py --step all
    python kits23_nnunet_pipeline.py --step train --epochs 1000
    python kits23_nnunet_pipeline.py --step train --quick        (Runs 20 epochs)
    python kits23_nnunet_pipeline.py --step predict
"""

import os
import shutil
import json
import argparse
import subprocess
import sys
import glob
import time
from pathlib import Path

# Try importing standard libraries; if missing, they will be checked in check_setup
try:
    from tqdm import tqdm
    import numpy as np
    import nibabel as nib
except ImportError:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_ID = 1  # Unique ID for nnU-Net
DATASET_NAME = f"Dataset{DATASET_ID:03d}_KiTS23"
DEFAULT_BASE_DIR = Path("./output/kits23_nnunet").absolute()
DEFAULT_DATA_DIR = Path("./kits23/dataset").absolute()

# ============================================================================
# HELPERS
# ============================================================================
def run_command(cmd, env=None, check=True):
    """Runs a shell command with real-time output stream."""
    print(f"\n>>> Running: {cmd}")
    if env is None:
        env = os.environ.copy()

    process = subprocess.Popen(
        cmd, 
        shell=True, 
        env=env,
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if check and process.returncode != 0:
        print(f"\n!!! Command failed with code {process.returncode} !!!")
        sys.exit(process.returncode)
    return process.returncode

def verify_dependencies():
    """Checks and installs required Python packages."""
    required = ["nnunetv2", "monai", "nibabel", "tqdm", "numpy", "batchgenerators", "torch"]
    missing = []
    
    print("Checking dependencies...")
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            try:
                __import__(pkg.replace("-", "_"))
            except ImportError:
                missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}. Installing...")
        cmd = f"{sys.executable} -m pip install " + " ".join(missing)
        subprocess.check_call(cmd, shell=True)
        print("Dependencies installed. Please restart the script to load them.")
        sys.exit(0)
    else:
        print("All dependencies verified.")

# ============================================================================
# PIPELINE STEPS
# ============================================================================
def setup_directories(base_dir):
    """Creates nnU-Net folder structure."""
    raw_dir = base_dir / "nnUNet_raw"
    preprocessed_dir = base_dir / "nnUNet_preprocessed"
    results_dir = base_dir / "nnUNet_results"
    
    for d in [raw_dir, preprocessed_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Dataset specific folders
    (raw_dir / DATASET_NAME / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw_dir / DATASET_NAME / "labelsTr").mkdir(parents=True, exist_ok=True)
    (raw_dir / DATASET_NAME / "imagesTs").mkdir(parents=True, exist_ok=True)
    
    return raw_dir, preprocessed_dir, results_dir

def set_env_vars(raw_dir, preprocessed_dir, results_dir):
    """Sets environment variables required by nnU-Net."""
    os.environ["nnUNet_raw"] = str(raw_dir)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_dir)
    os.environ["nnUNet_results"] = str(results_dir)
    # Optimization for 5090
    os.environ["nnUNet_n_proc_DA"] = "12"
    
    # Enable internal nnU-Net logging to stdout
    os.environ["nnUNet_print_to_stdout"] = "1"

def convert_dataset(source_dir, raw_dir):
    """Converts KiTS23 file structure to nnU-Net format."""
    print(f"\n=== Converting Dataset from {source_dir} ===")
    
    cases = sorted(list(source_dir.glob("case_*")))
    if not cases:
        raise FileNotFoundError(f"No 'case_*' directories found in {source_dir}")
        
    print(f"Found {len(cases)} cases to process.")
    
    # Generate dataset.json
    json_dict = {
        "channel_names": { "0": "CT" },
        "labels": {
            "background": 0,
            "kidney": 1,
            "tumor": 2,
            "cyst": 3
        },
        "numTraining": len(cases),
        "file_ending": ".nii.gz"
    }
    
    json_path = raw_dir / DATASET_NAME / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4)
    print(f"Created dataset.json at {json_path}")
    
    # Copy files
    from tqdm import tqdm
    success_count = 0
    for case_path in tqdm(cases, desc="Copying files"):
        case_id = case_path.name
        img_src = case_path / "imaging.nii.gz"
        seg_src = case_path / "segmentation.nii.gz"
        
        if not img_src.exists() or not seg_src.exists():
            print(f"Skipping {case_id}: Missing files")
            continue
            
        img_dst = raw_dir / DATASET_NAME / "imagesTr" / f"{case_id}_0000.nii.gz"
        seg_dst = raw_dir / DATASET_NAME / "labelsTr" / f"{case_id}.nii.gz"
        
        try:
            if not img_dst.exists(): shutil.copy2(img_src, img_dst)
            if not seg_dst.exists(): shutil.copy2(seg_src, seg_dst)
            success_count += 1
        except IOError as e:
            print(f"Error: {e}")
            
    print(f"Successfully converted {success_count}/{len(cases)} cases.")

def run_planning(env, data_dir, raw_dir):
    """
    Runs nnU-Net planning.
    Auto-runs conversion if dataset.json is missing.
    """
    print("\n=== Running Planning & Preprocessing ===")
    
    # Check if dataset.json exists
    dataset_json_path = raw_dir / DATASET_NAME / "dataset.json"
    if not dataset_json_path.exists():
        print(f"⚠️ dataset.json not found at {dataset_json_path}")
        print(">> Auto-triggering conversion step...")
        convert_dataset(data_dir, raw_dir)
    
    print("This auto-tunes patch size, batch size, and network topology for your hardware.")
    cmd = f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity -c 3d_fullres 3d_lowres"
    run_command(cmd, env)

def run_training_python(config, fold, epochs, env, data_dir, raw_dir):
    """
    Runs training using nnU-Net Python API to allow custom epoch counts.
    Auto-runs planning/preprocessing if missing.
    """
    print(f"\n=== Running Training ({config}, Fold {fold}, Epochs={epochs}) ===")
    
    # Check if plans exist
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    
    try:
        dataset_name = maybe_convert_to_dataset_name(DATASET_ID)
    except:
        dataset_name = DATASET_NAME
        
    preprocessed_dataset_dir = Path(nnUNet_preprocessed) / dataset_name
    plans_file = preprocessed_dataset_dir / "nnUNetPlans.json"
    
    if not plans_file.exists():
        print(f"⚠️ Plans file not found at {plans_file}")
        print(">> Auto-triggering planning step...")
        run_planning(env, data_dir, raw_dir)
    
    try:
        import torch
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        
        # 1. Load Plans
        plans_manager = PlansManager(plans_file)
        
        # 2. Get configuration
        if config not in plans_manager.available_configurations:
            raise ValueError(f"Configuration {config} not found in plans.")
            
        configuration_manager = plans_manager.get_configuration(config)
        
        # 3. Load Dataset JSON
        dataset_json_file = preprocessed_dataset_dir / "dataset.json"
        with open(dataset_json_file, 'r') as f:
            dataset_json = json.load(f)
            
        # 4. Instantiate Trainer
        trainer = nnUNetTrainer(
            plans=plans_manager, 
            configuration=config, 
            fold=fold, 
            dataset_json=dataset_json, 
            unpack_dataset=True, 
            device=torch.device('cuda')
        )
        
        # 5. OVERRIDE EPOCHS
        if epochs != 1000:
            print(f"⚡ OVERRIDING default epochs (1000) -> {epochs}")
            trainer.num_epochs = epochs
            
        # 6. Run
        trainer.initialize()
        trainer.run_training()
        print("Training finished.")

    except ImportError as e:
        print(f"⚠️ Could not import nnU-Net Python API ({e}). Falling back to CLI command.")
        run_command(f"nnUNetv2_train {DATASET_ID} {config} {fold}", env)
    except Exception as e:
        print(f"❌ Error during Python API training: {e}")
        print("Falling back to CLI command...")
        run_command(f"nnUNetv2_train {DATASET_ID} {config} {fold}", env)

def run_inference(env, config, fold, base_dir):
    """Runs inference."""
    print(f"\n=== Running Inference ({config}) ===")
    input_folder = base_dir / "nnUNet_raw" / DATASET_NAME / "imagesTr"
    output_folder = base_dir / "predictions" / config
    output_folder.mkdir(parents=True, exist_ok=True)
    
    cmd = f"nnUNetv2_predict -i {input_folder} -o {output_folder} -d {DATASET_ID} -c {config} -f {fold} --save_probabilities"
    run_command(cmd, env)
    return output_folder

def evaluate_predictions(base_dir, pred_folder):
    """Computes Dice scores."""
    print(f"\n=== Evaluating Predictions ===")
    gt_folder = base_dir / "nnUNet_raw" / DATASET_NAME / "labelsTr"
    
    pred_files = sorted(list(pred_folder.glob("*.nii.gz")))
    if not pred_files:
        print("No predictions found.")
        return
        
    import nibabel as nib
    import numpy as np
    
    scores = {"kidney": [], "tumor": [], "cyst": []}
    
    for pred_path in tqdm(pred_files, desc="Eval"):
        case_id = pred_path.name
        gt_path = gt_folder / case_id
        
        if not gt_path.exists(): continue
            
        pred = nib.load(pred_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()
        
        for cls_name, cls_idx in [("kidney", 1), ("tumor", 2), ("cyst", 3)]:
            p = (pred == cls_idx)
            g = (gt == cls_idx)
            intersection = np.logical_and(p, g).sum()
            union = p.sum() + g.sum()
            dice = (2.0 * intersection) / (union + 1e-8)
            scores[cls_name].append(dice)
            
    print("\nRESULTS:")
    for k, v in scores.items():
        print(f"{k.capitalize()}: {np.mean(v):.4f}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="KiTS23 nnU-Net Pipeline")
    parser.add_argument("--step", type=str, choices=["all", "setup", "convert", "plan", "train", "predict", "evaluate"], default="all")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--config", type=str, default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="Run in quick mode (20 epochs)")
    
    args = parser.parse_args()
    
    # Handle Quick Mode
    if args.quick:
        args.epochs = 20
        print(f"🚀 Quick Mode Enabled: Epochs set to {args.epochs}")
    
    verify_dependencies()
    
    base_dir = Path(args.base_dir).absolute()
    data_dir = Path(args.data_dir).absolute()
    
    raw_dir, preprocessed_dir, results_dir = setup_directories(base_dir)
    set_env_vars(raw_dir, preprocessed_dir, results_dir)
    env = os.environ.copy()
    
    if args.step in ["all", "setup", "convert"]:
        convert_dataset(data_dir, raw_dir)
        
    if args.step in ["all", "setup", "plan"]:
        run_planning(env, data_dir, raw_dir)
        
    if args.step in ["all", "train"]:
        run_training_python(args.config, args.fold, args.epochs, env, data_dir, raw_dir)
        
    if args.step in ["predict"]:
        out_dir = run_inference(env, args.config, args.fold, base_dir)
        evaluate_predictions(base_dir, out_dir)
    
    if args.step in ["evaluate"]:
        out_dir = base_dir / "predictions" / args.config
        evaluate_predictions(base_dir, out_dir)

if __name__ == "__main__":
    main()
