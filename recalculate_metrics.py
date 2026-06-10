"""
Recalculate Metrics (V4 Fair Evaluation)
========================================

Recalculates Dice scores from existing predictions with "Fair" logic:
1. Handles empty cases correctly (Dice = 1.0 if both GT and Pred are empty).
2. Reports "Tumor-Positive Dice" (performance on cases that actually have tumors).
3. Does NOT require GPU or re-running inference (uses saved NIfTI predictions).

Usage:
    python recalculate_metrics.py
"""

import os
import json
from pathlib import Path
import time
import numpy as np
import nibabel as nib
from tqdm import tqdm

def compute_fair_metrics(pred, target, num_classes=4):
    """
    Compute Dice with proper handling for empty classes.
    """
    metrics = {
        'dice': [],
        'has_fg': [],  # Whether the class exists in GT
        'vol_gt': [],
        'vol_pred': []
    }
    
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        tp = np.sum(pred_c & target_c)
        fp = np.sum(pred_c & ~target_c)
        fn = np.sum(~pred_c & target_c)
        
        target_vol = np.sum(target_c)
        pred_vol = np.sum(pred_c)
        
        # Logic for empty cases
        if target_vol == 0:
            if pred_vol == 0:
                 # Correct rejection! (Both empty)
                dice = 1.0
            else:
                 # False positive (GT empty, Pred has stuff)
                dice = 0.0
            has_fg = False
        else:
            # Standard Dice
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            has_fg = True
            
        metrics['dice'].append(float(dice))
        metrics['has_fg'].append(has_fg)
        metrics['vol_gt'].append(int(target_vol))
        metrics['vol_pred'].append(int(pred_vol))
    
    return metrics

def run_recalculation():
    # Paths
    base_dir = Path("./")
    kits_dir = base_dir / "kits23/dataset"
    pred_dir = base_dir / "output/meddino_vista3d_v4_tumor_focused/predictions"
    output_file = base_dir / "output/meddino_vista3d_v4_tumor_focused/fair_results.json"
    
    if not pred_dir.exists():
        print(f"Error: Predictions directory not found: {pred_dir}")
        return

    # Find all prediction files
    pred_files = sorted(list(pred_dir.glob("*_prediction.nii.gz")))
    print(f"Found {len(pred_files)} predictions.")
    
    results = []
    
    print("Recalculating metrics...")
    for p_path in tqdm(pred_files):
        # Extract case name (e.g., "case_00012_prediction.nii.gz" -> "case_00012")
        case_name = p_path.name.replace("_prediction.nii.gz", "")
        
        # Load Prediction
        pred = nib.load(p_path).get_fdata().astype(np.uint8)
        
        # Load Ground Truth
        gt_path = kits_dir / case_name / "segmentation.nii.gz"
        if not gt_path.exists():
            print(f"Warning: No GT found for {case_name}")
            continue
            
        gt = nib.load(gt_path).get_fdata().astype(np.uint8)
        
        # Verify shapes
        if pred.shape != gt.shape:
            print(f"Warning: Shape mismatch for {case_name} {pred.shape} vs {gt.shape}")
            continue
            
        # Compute Stats
        m = compute_fair_metrics(pred, gt)
        
        results.append({
            "case": case_name,
            "metrics": m
        })
        
    # --- Aggregation ---
    classes = ["Background", "Kidney", "Tumor", "Cyst"]
    
    print("\n" + "="*60)
    print("FAIR EVALUATION REPORT (V4)")
    print("="*60)
    
    summary = {}
    
    for idx, cls_name in enumerate(classes):
        if idx == 0: continue # Skip BG
        
        all_dice = [r['metrics']['dice'][idx] for r in results]
        
        # 1. Overall Mean (including empty cases as 1.0 or 0.0)
        mean_all = np.mean(all_dice)
        
        # 2. Positive-Only Mean (Only cases where GT has this class)
        #    This tells us "When a tumor exists, how well do we seg it?"
        pos_indices = [i for i, r in enumerate(results) if r['metrics']['has_fg'][idx]]
        if pos_indices:
            pos_dice = np.mean([all_dice[i] for i in pos_indices])
            pos_count = len(pos_indices)
        else:
            pos_dice = 0.0
            pos_count = 0
            
        summary[cls_name] = {
            "mean_all": mean_all,
            "mean_positive": pos_dice,
            "count_positive": pos_count
        }
        
        print(f"\n{cls_name.upper()}:")
        print(f"  Overall Dice (Fair):       {mean_all:.4f}  (Includes correct empty predictions)")
        print(f"  Tumor-Positive Dice:       {pos_dice:.4f}  (Only cases with {cls_name}, N={pos_count})")

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump({"summary": summary, "cases": results}, f, indent=2)
    print(f"\nSaved detailed analysis to: {output_file}")

if __name__ == "__main__":
    run_recalculation()
