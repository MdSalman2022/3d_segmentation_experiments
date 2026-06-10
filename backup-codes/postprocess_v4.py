"""
Post-Processing for MedDINO-VISTA3D V4 Predictions
====================================================

Applies post-processing to EXISTING saved predictions (no retraining needed).
Also provides a fixed re-inference option that corrects the normalization bug.

Fixes applied:
1. Remove small spurious connected components (especially cyst false positives)
2. Ensure tumor/cyst predictions are within or adjacent to kidney mask
3. Remove isolated tiny clusters per class
4. Optional: Re-inference with FIXED normalization (volume-level, matching training)

Usage:
    # Post-process existing predictions only (0 GPU hours):
    python postprocess_v4.py --mode postprocess

    # Re-inference with fixed normalization + post-process (~3h on 5090):
    python postprocess_v4.py --mode reinference

    # Re-inference with TTA + post-process (~6-8h on 5090):
    python postprocess_v4.py --mode reinference --tta
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm

# ============================================================================
# POST-PROCESSING FUNCTIONS
# ============================================================================


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Remove connected components smaller than min_voxels."""
    if mask.sum() == 0:
        return mask
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask

    # Get sizes of each component
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))

    # Keep only components >= min_voxels
    cleaned = np.zeros_like(mask)
    for i, size in enumerate(component_sizes):
        if size >= min_voxels:
            cleaned[labeled == (i + 1)] = 1
    return cleaned


def keep_largest_n_components(mask: np.ndarray, n: int = 2) -> np.ndarray:
    """Keep only the largest N connected components."""
    if mask.sum() == 0:
        return mask
    labeled, num_features = ndimage.label(mask)
    if num_features <= n:
        return mask

    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    sorted_indices = np.argsort(component_sizes)[::-1]

    cleaned = np.zeros_like(mask)
    for i in range(min(n, len(sorted_indices))):
        cleaned[labeled == (sorted_indices[i] + 1)] = 1
    return cleaned


def get_dilated_kidney_mask(
    kidney_mask: np.ndarray,
    dilation_mm: float = 5.0,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Get dilated kidney mask to constrain tumor/cyst predictions."""
    if kidney_mask.sum() == 0:
        return kidney_mask

    # Calculate dilation radius in voxels for each dimension
    struct_size = [max(1, int(round(dilation_mm / s))) for s in voxel_spacing]
    struct = ndimage.generate_binary_structure(3, 1)

    # Iterative dilation
    dilated = ndimage.binary_dilation(
        kidney_mask, structure=struct, iterations=max(struct_size)
    )
    return dilated.astype(np.uint8)


def postprocess_prediction(
    pred: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    kidney_min_voxels: int = 1000,
    tumor_min_voxels: int = 50,
    cyst_min_voxels: int = 100,
    kidney_dilation_mm: float = 10.0,
    constrain_to_kidney: bool = True,
) -> np.ndarray:
    """
    Apply post-processing to a single prediction volume.

    Steps:
    1. Clean kidney: keep largest 2 components (left + right kidney), remove small noise
    2. Clean tumor: remove tiny components, constrain to kidney region
    3. Clean cyst: remove tiny components, constrain to kidney region
    4. Resolve overlaps (kidney > tumor > cyst priority for spatial consistency)
    """
    result = pred.copy()

    # Extract per-class masks
    kidney_mask = (pred == 1).astype(np.uint8)
    tumor_mask = (pred == 2).astype(np.uint8)
    cyst_mask = (pred == 3).astype(np.uint8)

    # ---- Step 1: Clean kidney ----
    # Keep largest 2 components (left + right kidney)
    kidney_clean = keep_largest_n_components(kidney_mask, n=2)
    kidney_clean = remove_small_components(kidney_clean, kidney_min_voxels)

    # ---- Step 2: Kidney containment region ----
    if constrain_to_kidney and kidney_clean.sum() > 0:
        # Tumor and cyst should be within dilated kidney region
        kidney_region = get_dilated_kidney_mask(
            kidney_clean, kidney_dilation_mm, voxel_spacing
        )
    else:
        kidney_region = np.ones_like(pred, dtype=np.uint8)

    # ---- Step 3: Clean tumor ----
    tumor_clean = remove_small_components(tumor_mask, tumor_min_voxels)
    if constrain_to_kidney:
        tumor_clean = tumor_clean * kidney_region  # Must be near kidney

    # ---- Step 4: Clean cyst ----
    cyst_clean = remove_small_components(cyst_mask, cyst_min_voxels)
    if constrain_to_kidney:
        cyst_clean = cyst_clean * kidney_region  # Must be near kidney

    # ---- Step 5: Reassemble (priority: kidney > tumor > cyst > background) ----
    result = np.zeros_like(pred)
    result[kidney_clean > 0] = 1
    result[tumor_clean > 0] = 2  # Tumor overwrites kidney (tumors are within kidney)
    result[cyst_clean > 0] = 3  # Cyst overwrites kidney (cysts are within kidney)

    return result


def compute_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int = 4) -> Dict:
    """Compute Dice, IoU, Precision, Recall per class."""
    metrics = {"dice": [], "iou": [], "precision": [], "recall": []}

    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c

        tp = np.sum(pred_c & target_c)
        fp = np.sum(pred_c & ~target_c)
        fn = np.sum(~pred_c & target_c)

        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        metrics["dice"].append(float(dice))
        metrics["iou"].append(float(iou))
        metrics["precision"].append(float(precision))
        metrics["recall"].append(float(recall))

    return metrics


def compute_fair_metrics(pred: np.ndarray, target: np.ndarray) -> Dict:
    """Compute metrics with has_fg and volume info for fair comparison."""
    metrics = compute_metrics(pred, target)

    has_fg = []
    vol_gt = []
    vol_pred = []
    for c in range(4):
        has_fg.append(bool(np.any(target == c)))
        vol_gt.append(int(np.sum(target == c)))
        vol_pred.append(int(np.sum(pred == c)))

    metrics["has_fg"] = has_fg
    metrics["vol_gt"] = vol_gt
    metrics["vol_pred"] = vol_pred

    return metrics


# ============================================================================
# MAIN POSTPROCESSING PIPELINE
# ============================================================================


def postprocess_existing_predictions(config):
    """Apply post-processing to already-saved prediction NIfTI files."""

    data_dir = Path(config["kits23_dir"])
    pred_dir = Path(config["output_dir"]) / "predictions"
    pp_dir = Path(config["output_dir"]) / "predictions_postprocessed"
    pp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Post-Processing V4 Predictions")
    print(f"{'='*60}")
    print(f"Input:  {pred_dir}")
    print(f"Output: {pp_dir}")
    print(f"Kidney min: {config['kidney_min_voxels']} voxels")
    print(f"Tumor min:  {config['tumor_min_voxels']} voxels")
    print(f"Cyst min:   {config['cyst_min_voxels']} voxels")
    print(f"Kidney dilation: {config['kidney_dilation_mm']}mm")
    print(f"Constrain to kidney: {config['constrain_to_kidney']}")
    print(f"{'='*60}\n")

    # Find all prediction files
    pred_files = sorted(pred_dir.glob("*_prediction.nii.gz"))
    print(f"Found {len(pred_files)} prediction files\n")

    results_before = []
    results_after = []

    for pred_path in tqdm(pred_files, desc="Post-processing"):
        case_name = pred_path.name.replace("_prediction.nii.gz", "")

        # Load prediction
        pred_nii = nib.load(pred_path)
        pred = pred_nii.get_fdata().astype(np.int32)

        # Get voxel spacing
        voxel_spacing = tuple(abs(s) for s in pred_nii.header.get_zooms()[:3])
        if any(s == 0 for s in voxel_spacing):
            voxel_spacing = (1.0, 1.0, 1.0)

        # Load ground truth
        gt_path = data_dir / case_name / "segmentation.nii.gz"
        gt = None
        if gt_path.exists():
            gt = nib.load(gt_path).get_fdata().astype(np.int32)

        # Metrics before post-processing
        if gt is not None:
            metrics_before = compute_fair_metrics(pred, gt)
            results_before.append({"case": case_name, "metrics": metrics_before})

        # Apply post-processing
        pred_pp = postprocess_prediction(
            pred,
            voxel_spacing=voxel_spacing,
            kidney_min_voxels=config["kidney_min_voxels"],
            tumor_min_voxels=config["tumor_min_voxels"],
            cyst_min_voxels=config["cyst_min_voxels"],
            kidney_dilation_mm=config["kidney_dilation_mm"],
            constrain_to_kidney=config["constrain_to_kidney"],
        )

        # Metrics after post-processing
        if gt is not None:
            metrics_after = compute_fair_metrics(pred_pp, gt)
            results_after.append({"case": case_name, "metrics": metrics_after})

        # Save post-processed prediction
        pp_nii = nib.Nifti1Image(
            pred_pp.astype(np.uint8), pred_nii.affine, pred_nii.header
        )
        nib.save(pp_nii, pp_dir / pred_path.name)

    # Print comparison
    if results_before and results_after:
        print_comparison(results_before, results_after, config)

    return results_after


def print_comparison(results_before, results_after, config):
    """Print before/after comparison of metrics."""

    class_names = ["Background", "Kidney", "Tumor", "Cyst"]

    print(f"\n{'='*70}")
    print(f"BEFORE vs AFTER Post-Processing")
    print(f"{'='*70}")

    for c_idx, c_name in enumerate(class_names):
        if c_idx == 0:
            continue  # Skip background

        dice_before = [r["metrics"]["dice"][c_idx] for r in results_before]
        dice_after = [r["metrics"]["dice"][c_idx] for r in results_after]

        # Fair metrics: only count cases where GT has the class
        has_class = [r["metrics"]["has_fg"][c_idx] for r in results_before]

        all_before = np.mean(dice_before)
        all_after = np.mean(dice_after)

        pos_before = (
            np.mean([d for d, h in zip(dice_before, has_class) if h])
            if any(has_class)
            else 0
        )
        pos_after = (
            np.mean([d for d, h in zip(dice_after, has_class) if h])
            if any(has_class)
            else 0
        )

        count_zero_before = sum(1 for d in dice_before if d == 0)
        count_zero_after = sum(1 for d in dice_after if d == 0)

        delta_all = all_after - all_before
        delta_pos = pos_after - pos_before

        sign_all = "+" if delta_all >= 0 else ""
        sign_pos = "+" if delta_pos >= 0 else ""

        print(f"\n  {c_name}:")
        print(
            f"    Mean (all):      {all_before:.4f} → {all_after:.4f} ({sign_all}{delta_all:.4f})"
        )
        print(
            f"    Mean (positive): {pos_before:.4f} → {pos_after:.4f} ({sign_pos}{delta_pos:.4f})"
        )
        print(f"    Zero-dice cases: {count_zero_before} → {count_zero_after}")

    # Overall Mean FG Dice
    fg_before = np.mean([np.mean(r["metrics"]["dice"][1:]) for r in results_before])
    fg_after = np.mean([np.mean(r["metrics"]["dice"][1:]) for r in results_after])
    delta_fg = fg_after - fg_before
    sign_fg = "+" if delta_fg >= 0 else ""

    print(
        f"\n  Mean FG Dice: {fg_before:.4f} → {fg_after:.4f} ({sign_fg}{delta_fg:.4f})"
    )
    print(f"{'='*70}")

    # Save results
    output_dir = Path(config["output_dir"])
    summary = {"summary": {}, "cases": []}

    for c_idx, c_name in enumerate(["Kidney", "Tumor", "Cyst"]):
        real_idx = c_idx + 1
        dice_after = [r["metrics"]["dice"][real_idx] for r in results_after]
        has_class = [r["metrics"]["has_fg"][real_idx] for r in results_after]

        pos_dice = [d for d, h in zip(dice_after, has_class) if h]

        summary["summary"][c_name] = {
            "mean_all": float(np.mean(dice_after)),
            "mean_positive": float(np.mean(pos_dice)) if pos_dice else 0.0,
            "count_positive": sum(has_class),
            "before_mean_all": float(
                np.mean([r["metrics"]["dice"][real_idx] for r in results_before])
            ),
        }

    summary["cases"] = results_after

    with open(output_dir / "postprocessed_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir / 'postprocessed_results.json'}")


# ============================================================================
# RE-INFERENCE WITH FIXED NORMALIZATION
# ============================================================================


def reinference_fixed(config):
    """
    Re-run inference with FIXED normalization:
    - Normalize entire volume FIRST (matching training behavior)
    - Then extract patches from the normalized volume
    - Optional: Test-Time Augmentation (TTA) with flips
    """
    import torch
    import torch.nn.functional as F
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from meddino_vista3d_v4_tumor_focused import MedDINOVISTA3DEnhanced

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output_dir"])
    pred_dir = output_dir / "predictions_fixed"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Re-Inference with FIXED Normalization")
    print(f"{'='*60}")
    print(f"TTA: {config.get('tta', False)}")
    print(f"Overlap: {config.get('overlap', 0.5)}")
    print(f"Output: {pred_dir}")
    print(f"{'='*60}\n")

    # Load model
    model = MedDINOVISTA3DEnhanced(
        num_classes=4, freeze_encoder=False, dinov2_backbone="dinov2_vitb14"
    ).to(device)

    checkpoint_path = output_dir / "best_model_final.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✓ Loaded: {checkpoint_path.name}\n")

    # Get test cases (same split as original evaluation)
    data_dir = Path(config["kits23_dir"])
    all_cases = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")]
    )
    split = int(len(all_cases) * 0.2)
    test_cases = all_cases[:split]

    if config.get("num_cases"):
        test_cases = test_cases[: config["num_cases"]]

    print(f"Processing {len(test_cases)} cases...\n")

    results = []

    for case_dir in test_cases:
        case_name = case_dir.name
        img_path = case_dir / "imaging.nii.gz"
        lbl_path = case_dir / "segmentation.nii.gz"

        if not img_path.exists():
            continue

        print(f"Processing {case_name}...")
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()

        gt = None
        if lbl_path.exists():
            gt = nib.load(lbl_path).get_fdata().astype(np.int32)

        case_start = time.time()

        # ====== FIXED: Normalize entire volume first (matching training) ======
        img_normalized = (img - img.mean()) / (img.std() + 1e-8)

        # Run inference
        pred_probs = sliding_window_inference_fixed(
            model,
            img_normalized,
            roi_size=config.get("patch_size", (140, 224, 224)),
            overlap=config.get("overlap", 0.5),
            device=device,
            use_tta=config.get("tta", False),
        )

        # Argmax to get prediction
        pred = np.argmax(pred_probs, axis=0)

        # Apply post-processing
        voxel_spacing = tuple(abs(s) for s in img_nii.header.get_zooms()[:3])
        if any(s == 0 for s in voxel_spacing):
            voxel_spacing = (1.0, 1.0, 1.0)

        pred_pp = postprocess_prediction(
            pred,
            voxel_spacing=voxel_spacing,
            kidney_min_voxels=config.get("kidney_min_voxels", 1000),
            tumor_min_voxels=config.get("tumor_min_voxels", 50),
            cyst_min_voxels=config.get("cyst_min_voxels", 100),
            kidney_dilation_mm=config.get("kidney_dilation_mm", 10.0),
            constrain_to_kidney=config.get("constrain_to_kidney", True),
        )

        case_time = time.time() - case_start

        # Save prediction
        pred_nii = nib.Nifti1Image(
            pred_pp.astype(np.uint8), img_nii.affine, img_nii.header
        )
        nib.save(pred_nii, pred_dir / f"{case_name}_prediction.nii.gz")

        # Metrics
        if gt is not None:
            metrics = compute_fair_metrics(pred_pp, gt)
            results.append({"case": case_name, "metrics": metrics})
            print(
                f"  ✓ Dice: K={metrics['dice'][1]:.3f} T={metrics['dice'][2]:.3f} C={metrics['dice'][3]:.3f} | {case_time:.0f}s"
            )
        else:
            print(f"  ✓ Saved | {case_time:.0f}s")

    # Print summary
    if results:
        print(f"\n{'='*60}")
        print("FIXED NORMALIZATION RESULTS")
        print(f"{'='*60}")
        for c_idx, c_name in enumerate(["Background", "Kidney", "Tumor", "Cyst"]):
            if c_idx == 0:
                continue
            dice_vals = [r["metrics"]["dice"][c_idx] for r in results]
            print(f"  {c_name}: {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")

        fg_dice = np.mean([np.mean(r["metrics"]["dice"][1:]) for r in results])
        print(f"  Mean FG: {fg_dice:.4f}")
        print(f"{'='*60}")

        # Save
        with open(output_dir / "fixed_inference_results.json", "w") as f:
            json.dump(
                {
                    "summary": {
                        "Kidney": {
                            "mean": float(
                                np.mean([r["metrics"]["dice"][1] for r in results])
                            )
                        },
                        "Tumor": {
                            "mean": float(
                                np.mean([r["metrics"]["dice"][2] for r in results])
                            )
                        },
                        "Cyst": {
                            "mean": float(
                                np.mean([r["metrics"]["dice"][3] for r in results])
                            )
                        },
                    },
                    "cases": results,
                },
                f,
                indent=2,
            )

    return results


def sliding_window_inference_fixed(
    model,
    img_normalized,
    roi_size=(140, 224, 224),
    overlap=0.5,
    device="cuda",
    use_tta=False,
):
    """
    Sliding window inference with FIXED normalization.
    Key difference: img is ALREADY normalized at volume level. No per-patch normalization.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    d, h, w = img_normalized.shape
    pd, ph, pw = roi_size

    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))

    # Pad if needed
    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        img_normalized = np.pad(
            img_normalized, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant"
        )
        d, h, w = img_normalized.shape

    output = np.zeros((4, d, h, w), dtype=np.float32)
    count = np.zeros((d, h, w), dtype=np.float32)

    # Generate positions
    positions = set()
    for ds in range(0, max(1, d - pd + 1), stride_d):
        for hs in range(0, max(1, h - ph + 1), stride_h):
            for ws in range(0, max(1, w - pw + 1), stride_w):
                positions.add((ds, hs, ws))

    # Edge positions
    if d > pd:
        for hs in range(0, max(1, h - ph + 1), stride_h):
            for ws in range(0, max(1, w - pw + 1), stride_w):
                positions.add((d - pd, hs, ws))
    if h > ph:
        for ds in range(0, max(1, d - pd + 1), stride_d):
            for ws in range(0, max(1, w - pw + 1), stride_w):
                positions.add((ds, h - ph, ws))
    if w > pw:
        for ds in range(0, max(1, d - pd + 1), stride_d):
            for hs in range(0, max(1, h - ph + 1), stride_h):
                positions.add((ds, hs, w - pw))
    # Corner
    if d > pd and h > ph and w > pw:
        positions.add((d - pd, h - ph, w - pw))

    positions = list(positions)

    # Gaussian weighting for smooth blending
    sigma = 0.125
    gauss_d = np.exp(-0.5 * ((np.arange(pd) - pd / 2) / (pd * sigma)) ** 2)
    gauss_h = np.exp(-0.5 * ((np.arange(ph) - ph / 2) / (ph * sigma)) ** 2)
    gauss_w = np.exp(-0.5 * ((np.arange(pw) - pw / 2) / (pw * sigma)) ** 2)
    gaussian_weight = (
        gauss_d[:, None, None] * gauss_h[None, :, None] * gauss_w[None, None, :]
    )
    gaussian_weight = gaussian_weight.astype(np.float32)

    with torch.no_grad():
        for ds, hs, ws in tqdm(positions, desc="  Inference", leave=False):
            patch = img_normalized[ds : ds + pd, hs : hs + ph, ws : ws + pw]

            # NO per-patch normalization — already normalized at volume level
            patch_tensor = (
                torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
            )

            with torch.amp.autocast("cuda", enabled=True):
                logits = model(patch_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            if use_tta:
                # TTA: flip along each axis and average
                for flip_dims in [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                    flipped_input = torch.flip(patch_tensor, dims=flip_dims)
                    with torch.amp.autocast("cuda", enabled=True):
                        flipped_logits = model(flipped_input)
                        flipped_probs = (
                            torch.softmax(
                                torch.flip(flipped_logits, dims=flip_dims), dim=1
                            )
                            .cpu()
                            .numpy()[0]
                        )
                    probs += flipped_probs
                probs /= 8.0  # Average over original + 7 flips

            # Gaussian-weighted accumulation
            output[:, ds : ds + pd, hs : hs + ph, ws : ws + pw] += (
                probs * gaussian_weight[None, :, :, :]
            )
            count[ds : ds + pd, hs : hs + ph, ws : ws + pw] += gaussian_weight

    count[count == 0] = 1
    output = output / count[None, :, :, :]

    # Remove padding
    orig_d = output.shape[1] - pad_d if pad_d > 0 else output.shape[1]
    orig_h = output.shape[2] - pad_h if pad_h > 0 else output.shape[2]
    orig_w = output.shape[3] - pad_w if pad_w > 0 else output.shape[3]
    output = output[:, :orig_d, :orig_h, :orig_w]

    return output


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process V4 predictions")
    parser.add_argument(
        "--mode",
        type=str,
        default="postprocess",
        choices=["postprocess", "reinference"],
        help="postprocess: fix saved predictions (0 GPU). "
        "reinference: re-run with fixed normalization (~3h)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation (doubles inference time)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding window overlap (default: 0.5)",
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=None,
        help="Limit number of cases (for testing)",
    )

    # Post-processing parameters
    parser.add_argument(
        "--kidney_min",
        type=int,
        default=1000,
        help="Min voxels for kidney component (default: 1000)",
    )
    parser.add_argument(
        "--tumor_min",
        type=int,
        default=50,
        help="Min voxels for tumor component (default: 50)",
    )
    parser.add_argument(
        "--cyst_min",
        type=int,
        default=100,
        help="Min voxels for cyst component (default: 100)",
    )
    parser.add_argument(
        "--kidney_dilation",
        type=float,
        default=10.0,
        help="Kidney dilation in mm for containment (default: 10.0)",
    )
    parser.add_argument(
        "--no_constrain",
        action="store_true",
        help="Disable constraining tumor/cyst to kidney region",
    )

    args = parser.parse_args()

    config = {
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/meddino_vista3d_v4_tumor_focused",
        "patch_size": (140, 224, 224),
        "overlap": args.overlap,
        "num_cases": args.num_cases,
        "tta": args.tta,
        "kidney_min_voxels": args.kidney_min,
        "tumor_min_voxels": args.tumor_min,
        "cyst_min_voxels": args.cyst_min,
        "kidney_dilation_mm": args.kidney_dilation,
        "constrain_to_kidney": not args.no_constrain,
    }

    if args.mode == "postprocess":
        # Zero GPU time — just process saved NIfTI files
        postprocess_existing_predictions(config)

    elif args.mode == "reinference":
        # Re-inference with fixed normalization (~3h on 5090)
        reinference_fixed(config)
