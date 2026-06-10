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
from swinunetr_vista3d_vlm_v6b import MetricTracker, VLMLoopSegNet, SwinUNETREncoder, TextGuidedDecoder, MultiGranularityPromptEncoder, SpatialAttention3D, TextVisualAlignment, PromptAttentionModule, TextGuidedUpBlock


def compute_fair_metrics(pred, target, num_classes=4):
    """
    Compute metrics with proper handling for empty classes.
    """
    metrics = {
        "dice": [], "iou": [], "precision": [], "recall": [],
        "has_fg": [],
        "vol_gt": [], "vol_pred": []
    }
    
    for c in range(num_classes):
        pc, tc = (pred == c), (target == c)
        tp = float(np.sum(pc & tc))
        fp = float(np.sum(pc & ~tc))
        fn = float(np.sum(~pc & tc))
        
        target_vol = int(np.sum(tc))
        pred_vol = int(np.sum(pc))
        
        if target_vol == 0:
            if pred_vol == 0:
                dice, iou, precision, recall = 1.0, 1.0, 1.0, 1.0
            else:
                dice, iou, precision, recall = 0.0, 0.0, 0.0, 1.0
            has_fg = False
        else:
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
    from monai.inferers import sliding_window_inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output_dir"])
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\\n{'=' * 60}")
    print(f"SwinUNETR-VISTA3D V6b Evaluation")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {config['checkpoint']}")
    print(f"{'=' * 60}\\n")

    model = VLMLoopSegNet(
        num_classes=4,
        feature_dim=256,
        swin_feature_size=48,
        freeze_encoder=False,
        use_mgpl=True,
    ).to(device)

    ckpt_path = output_dir / config["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print(f"✓ Loaded {ckpt_path.name}\\n")

    data_dir = Path(config["kits23_dir"])
    all_cases = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")]
    )
    
    # Validation split same as training
    split = int(len(all_cases) * 0.2)
    test_cases = all_cases[:split]
    
    if config.get("num_cases"):
        test_cases = test_cases[: config["num_cases"]]

    print(f"Evaluating {len(test_cases)} cases ...\\n")

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
        img = img_nii.get_fdata().astype(np.float32)

        # Baseline CT windowing (-175, 250)
        img = np.clip(img, -175, 250)
        img = (img - (-175)) / (250 - (-175))

        lbl = nib.load(lbl_path).get_fdata() if lbl_path.exists() else None

        pd, ph, pw = config["patch_size"]
        d, h, w = img.shape
        if d < pd or h < ph or w < pw:
            pad = ((0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w)))
            img_padded = np.pad(img, pad, mode="constant")
        else:
            img_padded = img
            
        img_t = torch.from_numpy(img_padded).float().unsqueeze(0).unsqueeze(0).to(device)

        t0 = time.time()
        
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = sliding_window_inference(
                inputs=img_t,
                roi_size=config["patch_size"],
                sw_batch_size=4,
                predictor=model,
                overlap=config.get("overlap", 0.5),
            )
        
        pred = torch.argmax(out, dim=1)[0].cpu().numpy()
        
        # Trim padding
        if d < pd or h < ph or w < pw:
            pred = pred[:d, :h, :w]
            
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

    # Aggregate
    if any("metrics" in r for r in results):
        all_m = [r["metrics"] for r in results if "metrics" in r]
        agg = {}
        
        for ci, cn in enumerate(["background", "kidney", "tumor", "cyst"]):
            dices = [m["dice"][ci] for m in all_m]
            
            mean_all = float(np.mean(dices))
            std_all = float(np.std(dices))
            
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
        
        agg["mean_fg_dice_all"] = float(np.mean([np.mean(m["dice"][1:]) for m in all_m]))
        
        fg_pos_dices = []
        for m in all_m:
            case_fg = [m["dice"][i] for i in range(1, 4) if m["has_fg"][i]]
            if case_fg:
                fg_pos_dices.append(np.mean(case_fg))
        agg["mean_fg_dice_positive"] = float(np.mean(fg_pos_dices)) if fg_pos_dices else 0.0

        print(f"\\n{'=' * 70}")
        print("EVALUATION SUMMARY (V6b)")
        print(f"{'=' * 70}")
        print(f"Total cases: {len(results)} | Total time: {total_time/60:.1f} min")
        print(f"\\n{'Class':<12} {'Overall Dice':<20} {'Positive-Only Dice':<25} {'N+':<5}")
        print("-" * 70)
        for cn in ["kidney", "tumor", "cyst"]:
            m = agg[cn]
            print(f"{cn.capitalize():<12} "
                  f"{m['mean_dice_all']:.4f} ± {m['std_dice_all']:.4f}      "
                  f"{m['mean_dice_positive']:.4f} ± {m['std_dice_positive']:.4f}      "
                  f"{m['count_positive']:<5}")
        
        print(f"\\nMean Foreground Dice:")
        print(f"  Overall:       {agg['mean_fg_dice_all']:.4f}")
        print(f"  Positive-Only: {agg['mean_fg_dice_positive']:.4f}")
        print(f"{'=' * 70}\\n")

    out_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "V6b",
        "checkpoint": config["checkpoint"],
        "per_case": results,
    }
    if any("metrics" in r for r in results):
        out_json["aggregate"] = agg

    with open(output_dir / "fair_results.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"✓ Saved to {output_dir / 'fair_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_final.pth")
    parser.add_argument("--num_cases", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--output_dir", default="./output/vlm_loop_v6b")
    args = parser.parse_args()

    evaluate_model(
        {
            "kits23_dir": "./kits23/dataset",
            "output_dir": args.output_dir,
            "checkpoint": args.checkpoint,
            "num_cases": args.num_cases,
            "patch_size": (96, 192, 192),
            "overlap": args.overlap,
            "model": "swinunetr"
        }
    )
