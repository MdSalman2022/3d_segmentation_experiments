"""
Evaluation Script for MedDINO-VISTA3D V5
=========================================

Uses the same HU normalization as V5 training for consistent inference.
Performs full-volume sliding window inference.

Usage:
    python evaluate_v5.py --checkpoint best_final.pth
    python evaluate_v5.py --checkpoint best_stage1.pth --num_cases 10
"""

import os
import json
import argparse
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent))
from meddino_vista3d_v5 import MedDINOVISTA3DV5, normalize_ct


def sliding_window_inference(
    model, image, roi_size=(140, 224, 224), overlap=0.5, device="cuda"
):
    """Full-volume sliding window inference with V5-consistent normalization."""
    model.eval()
    d, h, w = image.shape
    pd, ph, pw = roi_size

    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))

    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
        d, h, w = image.shape

    output = np.zeros((4, d, h, w), dtype=np.float32)
    count = np.zeros((d, h, w), dtype=np.float32)

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

    positions = list(positions)
    print(f"  {len(positions)} windows ...")

    with torch.no_grad():
        for ds, hs, ws in tqdm(positions, desc="  Inference", leave=False):
            patch = image[ds : ds + pd, hs : hs + ph, ws : ws + pw]
            patch_tensor = (
                torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
            )

            with torch.amp.autocast("cuda", enabled=True):
                logits = model(patch_tensor)

            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            output[:, ds : ds + pd, hs : hs + ph, ws : ws + pw] += probs
            count[ds : ds + pd, hs : hs + ph, ws : ws + pw] += 1

    count[count == 0] = 1
    output = output / count[None]
    prediction = np.argmax(output, axis=0)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        orig_d, orig_h, orig_w = d - pad_d, h - pad_h, w - pad_w
        prediction = prediction[:orig_d, :orig_h, :orig_w]

    return prediction


def compute_fair_metrics(pred, target, num_classes=4):
    """
    Compute metrics with proper handling for empty classes.
    
    Returns:
        dict with 'dice', 'iou', 'precision', 'recall', 'has_fg', 'vol_gt', 'vol_pred'
    """
    metrics = {
        "dice": [], "iou": [], "precision": [], "recall": [],
        "has_fg": [],  # Whether the class exists in GT
        "vol_gt": [], "vol_pred": []
    }
    
    for c in range(num_classes):
        pc, tc = (pred == c), (target == c)
        tp = float(np.sum(pc & tc))
        fp = float(np.sum(pc & ~tc))
        fn = float(np.sum(~pc & tc))
        
        target_vol = int(np.sum(tc))
        pred_vol = int(np.sum(pc))
        
        # Logic for empty cases (Fix from V4 evaluation)
        if target_vol == 0:
            if pred_vol == 0:
                # Correct rejection! (Both empty)
                dice, iou, precision, recall = 1.0, 1.0, 1.0, 1.0
            else:
                # False positive (GT empty, Pred has stuff)
                dice, iou, precision, recall = 0.0, 0.0, 0.0, 1.0
            has_fg = False
        else:
            # Standard metrics
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            has_fg = True
        
        metrics["dice"].append(dice)
        metrics["iou"].append(iou)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["has_fg"].append(has_fg)
        metrics["vol_gt"].append(target_vol)
        metrics["vol_pred"].append(pred_vol)
    
    return metrics


def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output_dir"])
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"MedDINO-VISTA3D V5 Evaluation")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {config['checkpoint']}")
    print(f"{'=' * 60}\n")

    model = MedDINOVISTA3DV5(
        num_classes=4, freeze_encoder=False, dinov2_backbone="dinov2_vitb14"
    ).to(device)

    ckpt_path = output_dir / config["checkpoint"]
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✓ Loaded {ckpt_path.name}\n")

    data_dir = Path(config["kits23_dir"])
    all_cases = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")]
    )
    split = int(len(all_cases) * 0.2)
    test_cases = all_cases[:split]
    if config.get("num_cases"):
        test_cases = test_cases[: config["num_cases"]]

    print(f"Evaluating {len(test_cases)} cases ...\n")

    results = []
    total_start = time.time()

    for case_dir in test_cases:
        case_name = case_dir.name
        img_path = case_dir / "imaging.nii.gz"
        lbl_path = case_dir / "segmentation.nii.gz"
        if not img_path.exists():
            continue

        print(f"{case_name} ...", end=" ")
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()

        # V5: HU windowing normalization (consistent with training)
        img = normalize_ct(img)

        lbl = nib.load(lbl_path).get_fdata() if lbl_path.exists() else None

        t0 = time.time()
        pred = sliding_window_inference(
            model,
            img,
            roi_size=config["patch_size"],
            overlap=config.get("overlap", 0.5),
            device=device,
        )
        dt = time.time() - t0

        # Save prediction
        pred_nii = nib.Nifti1Image(
            pred.astype(np.uint8), img_nii.affine, img_nii.header
        )
        nib.save(pred_nii, pred_dir / f"{case_name}_prediction.nii.gz")

        result = {"case": case_name, "shape": list(img.shape), "time_s": round(dt, 1)}
        if lbl is not None:
            m = compute_fair_metrics(pred, lbl)
            result["metrics"] = m
            print(
                f"K={m['dice'][1]:.3f} T={m['dice'][2]:.3f} C={m['dice'][3]:.3f} | "
                f"GT: K={m['vol_gt'][1]} T={m['vol_gt'][2]} C={m['vol_gt'][3]} ({dt:.0f}s)"
            )
        else:
            print(f"({dt:.0f}s)")
        results.append(result)

    total_time = time.time() - total_start

    # Aggregate with both overall and positive-only metrics
    if any("metrics" in r for r in results):
        all_m = [r["metrics"] for r in results if "metrics" in r]
        agg = {}
        
        for ci, cn in enumerate(["background", "kidney", "tumor", "cyst"]):
            dices = [m["dice"][ci] for m in all_m]
            
            # Overall metrics (including empty cases)
            mean_all = float(np.mean(dices))
            std_all = float(np.std(dices))
            
            # Positive-only metrics (only cases where GT has this class)
            pos_indices = [i for i, m in enumerate(all_m) if m["has_fg"][ci]]
            if pos_indices:
                pos_dices = [dices[i] for i in pos_indices]
                mean_pos = float(np.mean(pos_dices))
                std_pos = float(np.std(pos_dices))
                count_pos = len(pos_indices)
            else:
                mean_pos = 0.0
                std_pos = 0.0
                count_pos = 0
            
            agg[cn] = {
                "mean_dice_all": mean_all,
                "std_dice_all": std_all,
                "mean_dice_positive": mean_pos,
                "std_dice_positive": std_pos,
                "count_positive": count_pos,
                "median_dice": float(np.median(dices)),
                "min_dice": float(np.min(dices)),
                "max_dice": float(np.max(dices)),
            }
        
        # Mean foreground dice (overall)
        agg["mean_fg_dice_all"] = float(np.mean([np.mean(m["dice"][1:]) for m in all_m]))
        
        # Mean foreground dice (positive-only: average of classes that exist in each case)
        fg_pos_dices = []
        for m in all_m:
            case_fg = [m["dice"][i] for i in range(1, 4) if m["has_fg"][i]]
            if case_fg:
                fg_pos_dices.append(np.mean(case_fg))
        agg["mean_fg_dice_positive"] = float(np.mean(fg_pos_dices)) if fg_pos_dices else 0.0

        print(f"\n{'=' * 70}")
        print("EVALUATION SUMMARY (V5)")
        print(f"{'=' * 70}")
        print(f"Total cases: {len(results)} | Total time: {total_time/60:.1f} min")
        print(f"\n{'Class':<12} {'Overall Dice':<20} {'Positive-Only Dice':<25} {'N+':<5}")
        print("-" * 70)
        for cn in ["kidney", "tumor", "cyst"]:
            m = agg[cn]
            print(f"{cn.capitalize():<12} "
                  f"{m['mean_dice_all']:.4f} ± {m['std_dice_all']:.4f}      "
                  f"{m['mean_dice_positive']:.4f} ± {m['std_dice_positive']:.4f}      "
                  f"{m['count_positive']:<5}")
        
        print(f"\nMean Foreground Dice:")
        print(f"  Overall:       {agg['mean_fg_dice_all']:.4f}")
        print(f"  Positive-Only: {agg['mean_fg_dice_positive']:.4f}")
        print(f"{'=' * 70}\n")

    out_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "V5",
        "checkpoint": config["checkpoint"],
        "per_case": results,
    }
    if any("metrics" in r for r in results):
        out_json["aggregate"] = agg

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"✓ Saved to {output_dir / 'test_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_final.pth")
    parser.add_argument("--num_cases", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    evaluate_model(
        {
            "kits23_dir": "./kits23/dataset",
            "output_dir": "./output/meddino_vista3d_v5",
            "checkpoint": args.checkpoint,
            "num_cases": args.num_cases,
            "patch_size": (140, 224, 224),
            "overlap": args.overlap,
        }
    )
