"""
KiTS21 Evaluation
=================
Official KiTS21 evaluation metrics: Dice and Surface Dice.
Evaluates on Hierarchical Evaluation Classes (HECs):
- Kidney: Label 1
- Masses: Labels 2+3 (tumor + cyst)  
- Tumor: Label 2
"""

import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from typing import Tuple, Union, List, Dict
from tqdm import tqdm

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

# HEC (Hierarchical Evaluation Class) configuration
HEC_CONFIG = {
    "kidney": {
        "labels": (1,),
        "tolerance_mm": 2.0,
    },
    "masses": {
        "labels": (2, 3),
        "tolerance_mm": 2.0,
    },
    "tumor": {
        "labels": (2,),
        "tolerance_mm": 2.0,
    },
}

HEC_NAMES = list(HEC_CONFIG.keys())


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
    """Compute Dice coefficient between predicted and ground truth masks."""
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
    """
    Compute Surface Dice with tolerance.
    
    Surface Dice measures the fraction of surface points within a 
    specified tolerance distance from the other surface.
    """
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


def evaluate_hec(
    pred_seg: np.ndarray,
    gt_seg: np.ndarray,
    spacing: Tuple[float, ...],
    hec_name: str,
) -> Dict[str, float]:
    """Evaluate a single HEC (Hierarchical Evaluation Class)."""
    config = HEC_CONFIG[hec_name]
    
    pred_mask = construct_hec_mask(pred_seg, config["labels"])
    gt_mask = construct_hec_mask(gt_seg, config["labels"])
    
    dice = compute_dice(pred_mask, gt_mask)
    surface_dice = compute_surface_dice(
        pred_mask, gt_mask, spacing, config["tolerance_mm"]
    )
    
    return {
        "dice": dice,
        "surface_dice": surface_dice,
    }


def evaluate_case(
    pred_path: str,
    gt_path: str,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictions for a single case.
    
    Returns:
        Dict mapping HEC names to their metrics (dice, surface_dice).
    """
    if sitk is None:
        raise ImportError("SimpleITK required for evaluation")
    
    # Load images
    pred_img = sitk.ReadImage(pred_path)
    gt_img = sitk.ReadImage(gt_path)
    
    # Get spacing (reversed because SimpleITK uses xyz, numpy uses zyx)
    spacing = tuple(pred_img.GetSpacing()[::-1])
    
    # Convert to numpy
    pred_seg = sitk.GetArrayFromImage(pred_img)
    gt_seg = sitk.GetArrayFromImage(gt_img)
    
    # Evaluate each HEC
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
    dataset_dir: str = "./kits23/dataset",
    num_processes: int = 4,
    output_csv: str = None,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Evaluate all predictions in a directory.
    
    Args:
        predictions_dir: Directory containing prediction files (case_XXXXX.nii.gz)
        dataset_dir: Directory containing ground truth
        num_processes: Number of parallel processes
        output_csv: Optional path for CSV output
    
    Returns:
        Tuple of (metrics dict, list of case IDs)
    """
    predictions_dir = Path(predictions_dir)
    dataset_dir = Path(dataset_dir)
    
    # Find prediction files
    pred_files = sorted(predictions_dir.glob("case_*.nii.gz"))
    if not pred_files:
        print(f"No prediction files found in {predictions_dir}")
        return {}, []
    
    # Build evaluation pairs
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
    
    # Run evaluation
    if num_processes > 1:
        with Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(_evaluate_case_wrapper, eval_pairs),
                total=len(eval_pairs),
                desc="Evaluating"
            ))
    else:
        results = [_evaluate_case_wrapper(pair) for pair in tqdm(eval_pairs)]
    
    # Aggregate results
    all_metrics = {hec: {"dice": [], "surface_dice": []} for hec in HEC_NAMES}
    
    for result in results:
        if result is not None:
            for hec in HEC_NAMES:
                all_metrics[hec]["dice"].append(result[hec]["dice"])
                all_metrics[hec]["surface_dice"].append(result[hec]["surface_dice"])
    
    # Compute means
    mean_metrics = {}
    for hec in HEC_NAMES:
        mean_metrics[hec] = {
            "dice": np.mean(all_metrics[hec]["dice"]),
            "surface_dice": np.mean(all_metrics[hec]["surface_dice"]),
        }
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"{'HEC':<10} {'Dice':>10} {'Surface Dice':>15}")
    print(f"{'-'*35}")
    for hec in HEC_NAMES:
        print(f"{hec:<10} {mean_metrics[hec]['dice']:>10.4f} {mean_metrics[hec]['surface_dice']:>15.4f}")
    print(f"{'='*60}")
    
    # Write CSV if requested
    if output_csv is None:
        output_csv = predictions_dir / "evaluation.csv"
    
    with open(output_csv, "w") as f:
        # Header
        cols = ["case_id"]
        for hec in HEC_NAMES:
            cols.extend([f"Dice_{hec}", f"SD_{hec}"])
        f.write(",".join(cols) + "\n")
        
        # Per-case results
        for i, case_id in enumerate(case_ids):
            if results[i] is not None:
                row = [case_id]
                for hec in HEC_NAMES:
                    row.append(f"{results[i][hec]['dice']:.4f}")
                    row.append(f"{results[i][hec]['surface_dice']:.4f}")
                f.write(",".join(row) + "\n")
        
        # Mean row
        row = ["MEAN"]
        for hec in HEC_NAMES:
            row.append(f"{mean_metrics[hec]['dice']:.4f}")
            row.append(f"{mean_metrics[hec]['surface_dice']:.4f}")
        f.write(",".join(row) + "\n")
    
    print(f"\nResults saved to: {output_csv}")
    
    return mean_metrics, case_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate KiTS21 predictions")
    parser.add_argument("predictions_dir", type=str,
                        help="Directory containing predictions (case_XXXXX.nii.gz)")
    parser.add_argument("--dataset_dir", type=str, default="./kits23/dataset",
                        help="Path to dataset with ground truth")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Number of parallel processes")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: predictions_dir/evaluation.csv)")
    
    args = parser.parse_args()
    
    evaluate_predictions(
        args.predictions_dir,
        args.dataset_dir,
        args.num_processes,
        args.output_csv,
    )
