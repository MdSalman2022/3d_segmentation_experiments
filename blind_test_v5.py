"""
blind_test_v5.py — Blind evaluation of MedDINO-VISTA3D V5
==========================================================
Loads a trained checkpoint and runs full-volume sliding-window inference on
a held-out test set.  Reports per-case and aggregate Dice/IoU/Precision/Recall,
saves predicted NIfTI segmentations, and renders side-by-side comparison figures.

IMPORTANT: Uses SLIDING-WINDOW inference (not center-crop patches) so every
voxel is predicted and Dice reflects real performance, not just the patch center.

Usage
-----
# Evaluate best stage-1 model on ./kits23/dataset test cases
python blind_test_v5.py --checkpoint output/meddino_vista3d_v5_vlm_critic/best_model_stage1_dice.pth

# Evaluate best final model (stage 2)
python blind_test_v5.py --checkpoint output/meddino_vista3d_v5_vlm_critic/best_model_final_dice.pth

# Custom dataset dir, save outputs to custom folder
python blind_test_v5.py \\
    --checkpoint path/to/model.pth \\
    --data_dir ./kits23/dataset \\
    --output_dir ./blind_test_results \\
    --cases 20          # evaluate first 20 cases (omit for all)
    --no_save_nifti     # skip writing segmentation .nii.gz files
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Import model from main training file ──────────────────────────────────────
# We re-use the exact same MedDINOVISTA3D_V5 class so weights load cleanly.
sys.path.insert(0, str(Path(__file__).parent))
from meddino_vista3d_v5_vlm_critic import MedDINOVISTA3D_V5

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]
CLASS_COLORS = {
    0: np.array([0, 0, 0]),  # background: black
    1: np.array([0, 255, 0]),  # kidney:     green
    2: np.array([255, 0, 0]),  # tumor:      red
    3: np.array([0, 0, 255]),  # cyst:       blue
}

# HU windowing must match training exactly
HU_MIN, HU_MAX = -175.0, 250.0

# Default patch / stride for sliding-window inference
DEFAULT_PATCH = (48, 154, 154)
DEFAULT_STRIDE = (24, 77, 77)  # 50% overlap → smooth predictions


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDING-WINDOW INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


def sliding_window_inference(
    model: torch.nn.Module,
    volume: np.ndarray,  # (D, H, W) normalised float32
    patch_size: tuple,
    stride: tuple,
    num_classes: int,
    device: torch.device,
    batch_size: int = 2,
) -> np.ndarray:
    """
    Tile the volume with overlapping patches and average softmax predictions.
    Returns integer label map of shape (D, H, W).
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # Pad volume so it is divisible by stride
    pad_d = max(0, pd - D) if D < pd else (sd - (D - pd) % sd) % sd
    pad_h = max(0, ph - H) if H < ph else (sh - (H - ph) % sh) % sh
    pad_w = max(0, pw - W) if W < pw else (sw - (W - pw) % sw) % sw

    vol_pad = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="reflect")
    Dp, Hp, Wp = vol_pad.shape

    # Accumulator tensors
    prob_sum = np.zeros((num_classes, Dp, Hp, Wp), dtype=np.float32)
    count_sum = np.zeros((Dp, Hp, Wp), dtype=np.float32)

    # Collect all patch start positions
    starts = []
    for ds in range(0, Dp - pd + 1, sd):
        for hs in range(0, Hp - ph + 1, sh):
            for ws in range(0, Wp - pw + 1, sw):
                starts.append((ds, hs, ws))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i : i + batch_size]
            patches = []
            for ds, hs, ws in batch_starts:
                p = vol_pad[ds : ds + pd, hs : hs + ph, ws : ws + pw]
                patches.append(p)

            # Stack → (B, 1, D, H, W)
            batch_t = (
                torch.from_numpy(np.stack(patches)[:, np.newaxis]).float().to(device)
            )

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(batch_t)  # (B, C, D, H, W)
                probs = F.softmax(logits, dim=1).float()  # avoid fp16 accumulation

            probs_np = probs.cpu().numpy()

            for j, (ds, hs, ws) in enumerate(batch_starts):
                prob_sum[:, ds : ds + pd, hs : hs + ph, ws : ws + pw] += probs_np[j]
                count_sum[ds : ds + pd, hs : hs + ph, ws : ws + pw] += 1.0

            del batch_t, logits, probs

    # Average and argmax
    count_sum = np.maximum(count_sum, 1e-8)
    avg_prob = prob_sum / count_sum[np.newaxis]
    pred = np.argmax(avg_prob, axis=0).astype(np.uint8)

    # Crop back to original size
    pred = pred[:D, :H, :W]
    return pred


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict:
    """Per-class Dice, IoU, Precision, Recall."""
    results = {}
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        tp = int((p & g).sum())
        fp = int((p & ~g).sum())
        fn = int((~p & g).sum())
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        results[CLASS_NAMES[c]] = {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(prec),
            "recall": float(rec),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return results


def print_metrics_table(metrics: dict, header: str = ""):
    if header:
        print(f"\n  {header}")
    print(f"  {'Class':<10} {'Dice':>7} {'IoU':>7} {'Prec':>7} {'Recall':>7}")
    print("  " + "-" * 42)
    for cls in CLASS_NAMES:
        m = metrics[cls]
        print(
            f"  {cls:<10} {m['dice']:>7.4f} {m['iou']:>7.4f} "
            f"{m['precision']:>7.4f} {m['recall']:>7.4f}"
        )
    fg = [metrics[c]["dice"] for c in CLASS_NAMES[1:]]
    print(f"  {'Mean (Fg)':<10} {sum(fg)/len(fg):>7.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════


def make_overlay(label_slice: np.ndarray) -> np.ndarray:
    h, w = label_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        if cls_id == 0:
            continue
        mask = label_slice == cls_id
        rgba[mask, :3] = color
        rgba[mask, 3] = 180
    return rgba


def save_case_figure(
    case_id: str,
    image: np.ndarray,  # (D, H, W) normalised
    pred: np.ndarray,  # (D, H, W) int labels
    gt: np.ndarray,  # (D, H, W) int labels
    metrics: dict,
    save_path: Path,
    num_slices: int = 4,
):
    """Save a comparison figure: CT | GT | Pred | Error map."""
    D = image.shape[0]

    # Pick slices with most tumor content; fall back to equally spaced
    tumor_content = [
        int((pred[z] == 2).sum()) + int((gt[z] == 2).sum())
        for z in range(D)
    ]
    top = sorted(range(D), key=lambda z: tumor_content[z], reverse=True)
    selected, seen = [], []
    for z in top:
        if all(abs(z - s) > D // 8 for s in seen):
            selected.append(z)
            seen.append(z)
        if len(selected) >= num_slices:
            break
    if not selected:
        selected = [D * i // (num_slices + 1) for i in range(1, num_slices + 1)]

    fig, axes = plt.subplots(len(selected), 4, figsize=(16, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis]

    col_titles = ["CT", "Ground Truth", "Prediction", "Errors (R=FP, B=FN)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, z in enumerate(selected):
        ct = image[z]
        ct_norm = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)

        # CT
        axes[row, 0].imshow(ct_norm, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"z={z}", fontsize=9)

        # GT
        axes[row, 1].imshow(ct_norm, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(make_overlay(gt[z]), alpha=0.55)

        # Prediction
        axes[row, 2].imshow(ct_norm, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(make_overlay(pred[z]), alpha=0.55)

        # Error map: FP=red, FN=blue
        err = np.zeros((*pred[z].shape, 3), dtype=np.uint8)
        fp = (pred[z] > 0) & (gt[z] == 0)
        fn = (pred[z] == 0) & (gt[z] > 0)
        err[fp] = [255, 0, 0]
        err[fn] = [0, 0, 255]
        axes[row, 3].imshow(ct_norm, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(err, alpha=0.6)

        for col in range(4):
            axes[row, col].axis("off")

    dice_str = " | ".join(f"{c}: {metrics[c]['dice']:.3f}" for c in CLASS_NAMES[1:])
    fig.suptitle(f"{case_id}    {dice_str}", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def get_holdout_cases(data_dir: Path, train_mode: str, val_split: float = 0.2) -> list:
    """
    Return the 40 cases permanently reserved as the blind test set.
    These are always the LAST 40 alphabetically-sorted cases.
    They are excluded from train/val splits in the training script.

    --train_mode is kept for backward compatibility but no longer changes
    which cases are selected; the test set is always the same 40 cases.
    """
    N_TEST = 40
    all_cases = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("case_")
        and (d / "imaging.nii.gz").exists()
        and (d / "segmentation.nii.gz").exists()
    )
    holdout = all_cases[-N_TEST:]
    print(f"  Blind test set : last {N_TEST} cases")
    print(f"  Range          : {holdout[0].name} → {holdout[-1].name}")
    print(f"  (These cases were excluded from all training and validation.)")
    return holdout




# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load MedDINOVISTA3D_V5 from a checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Checkpoint may be a plain state_dict or a full save dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        print(
            f"  Saved at epoch {meta.get('epoch', '?')}, "
            f"best_dice {meta.get('best_dice', '?'):.4f}"
        )
    else:
        state_dict = ckpt
        meta = {}

    model = MedDINOVISTA3D_V5(
        num_classes=4,
        feature_dim=256,
        cnn_channels=[32, 64, 128, 256],
        dinov2_backbone="dinov2_vitb14",
        clip_model="ViT-B-32",
        clip_pretrained="openai",
        freeze_encoder=True,
        use_mgpl=True,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠  Missing keys  ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  ⚠  Unexpected ({len(unexpected)}): {unexpected[:5]} ...")

    model.eval()
    print("  ✓ Model loaded\n")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Blind test for MedDINO-VISTA3D V5")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pth file (best_model_stage1_dice.pth or best_model_final_dice.pth)",
    )
    parser.add_argument(
        "--data_dir",
        default="./kits23/dataset",
        help="Root directory with case_XXXXX/ sub-folders",
    )
    parser.add_argument(
        "--output_dir",
        default="./blind_test_results",
        help="Where to write predictions, figures, and the summary JSON",
    )
    parser.add_argument(
        "--train_mode",
        default="quick_test",
        choices=["quick_test", "full"],
        help="Match the --mode used during training so the correct cases are excluded. "
             "quick_test: model saw cases 0-49; blind test uses 50+. "
             "full: model saw all cases; blind test uses val split (first 20%%).",
    )
    parser.add_argument(
        "--all_cases",
        action="store_true",
        help="Ignore train/val split and evaluate ALL cases (not recommended for blind test).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Val split fraction used during training (default 0.2). Only used with --train_mode full.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=None,
        help="Limit to first N holdout cases (default: all holdout cases)",
    )
    parser.add_argument(
        "--patch_size",
        nargs=3,
        type=int,
        default=list(DEFAULT_PATCH),
        metavar=("D", "H", "W"),
    )
    parser.add_argument(
        "--stride",
        nargs=3,
        type=int,
        default=list(DEFAULT_STRIDE),
        metavar=("D", "H", "W"),
    )
    parser.add_argument(
        "--sw_batch",
        type=int,
        default=2,
        help="Number of patches per sliding-window forward pass",
    )
    parser.add_argument(
        "--no_save_nifti",
        action="store_true",
        help="Skip writing predicted segmentation .nii.gz files",
    )
    parser.add_argument(
        "--no_save_figures",
        action="store_true",
        help="Skip saving comparison PNG figures",
    )
    parser.add_argument(
        "--figures_per_case",
        type=int,
        default=4,
        help="Number of slices per comparison figure",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    nifti_dir = output_dir / "predictions"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_nifti:
        nifti_dir.mkdir(exist_ok=True)
    if not args.no_save_figures:
        figure_dir.mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  MedDINO-VISTA3D V5 — Blind Evaluation")
    print(f"{'='*65}")
    print(f"  Device   : {device}")
    print(f"  Data     : {args.data_dir}")
    print(f"  Output   : {output_dir}")
    print(f"  Patch    : {args.patch_size}  Stride: {args.stride}")
    print(f"{'='*65}\n")

    # ── Load model ──────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── Discover holdout cases ──────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    if args.all_cases:
        all_cases = sorted(
            d for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("case_")
            and (d / "imaging.nii.gz").exists()
            and (d / "segmentation.nii.gz").exists()
        )
        print(f"  ⚠  --all_cases set: evaluating ALL {len(all_cases)} cases (includes training data)")
    else:
        all_cases = get_holdout_cases(data_dir, args.train_mode, args.val_split)

    if args.cases:
        all_cases = all_cases[: args.cases]

    print(f"  Evaluating {len(all_cases)} holdout cases\n")

    # ── Per-case evaluation ─────────────────────────────────────────────────
    all_results = {}
    # Confusion matrix accumulators for global metrics
    global_tp = np.zeros(4)
    global_fp = np.zeros(4)
    global_fn = np.zeros(4)

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    for case_dir in tqdm(all_cases, desc="Evaluating", unit="case"):
        case_id = case_dir.name
        t0 = time.time()

        # Load & preprocess
        img_nib = nib.load(str(case_dir / "imaging.nii.gz"))
        img_raw = img_nib.get_fdata().astype(np.float32)
        gt_raw = (
            nib.load(str(case_dir / "segmentation.nii.gz")).get_fdata().astype(np.int64)
        )

        # HU windowing + normalize (must match training)
        img = np.clip(img_raw, HU_MIN, HU_MAX)
        img = (img - HU_MIN) / (HU_MAX - HU_MIN)

        # Sliding-window inference
        pred = sliding_window_inference(
            model=model,
            volume=img,
            patch_size=patch_size,
            stride=stride,
            num_classes=4,
            device=device,
            batch_size=args.sw_batch,
        )

        elapsed = time.time() - t0

        # Metrics
        metrics = compute_metrics(pred, gt_raw, num_classes=4)
        for c_idx, cls in enumerate(CLASS_NAMES):
            global_tp[c_idx] += metrics[cls]["tp"]
            global_fp[c_idx] += metrics[cls]["fp"]
            global_fn[c_idx] += metrics[cls]["fn"]

        fg_dice = np.mean([metrics[c]["dice"] for c in CLASS_NAMES[1:]])
        print_metrics_table(
            metrics, header=f"{case_id}  (inf {elapsed:.0f}s | fg dice {fg_dice:.4f})"
        )

        all_results[case_id] = {
            "inference_time_s": round(elapsed, 1),
            "metrics": metrics,
            "mean_dice_fg": float(fg_dice),
        }

        # Save NIfTI prediction
        if not args.no_save_nifti:
            pred_nib = nib.Nifti1Image(
                pred.astype(np.uint8), img_nib.affine, img_nib.header
            )
            nib.save(pred_nib, str(nifti_dir / f"{case_id}_pred.nii.gz"))

        # Save comparison figure
        if not args.no_save_figures:
            save_case_figure(
                case_id=case_id,
                image=img,
                pred=pred,
                gt=gt_raw,
                metrics=metrics,
                save_path=figure_dir / f"{case_id}.png",
                num_slices=args.figures_per_case,
            )

        # Free GPU memory
        torch.cuda.empty_cache()

    # ── Global / aggregate metrics ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  AGGREGATE RESULTS ({len(all_cases)} cases)")
    print(f"{'='*65}")

    # Global dice (pooled TP/FP/FN across all cases)
    global_dice = 2 * global_tp / (2 * global_tp + global_fp + global_fn + 1e-8)
    global_iou = global_tp / (global_tp + global_fp + global_fn + 1e-8)
    global_prec = global_tp / (global_tp + global_fp + 1e-8)
    global_rec = global_tp / (global_tp + global_fn + 1e-8)

    print(f"\n  Pooled (global) metrics:")
    print(f"  {'Class':<10} {'Dice':>7} {'IoU':>7} {'Prec':>7} {'Recall':>7}")
    print("  " + "-" * 42)
    for i, cls in enumerate(CLASS_NAMES):
        print(
            f"  {cls:<10} {global_dice[i]:>7.4f} {global_iou[i]:>7.4f} "
            f"{global_prec[i]:>7.4f} {global_rec[i]:>7.4f}"
        )
    mean_fg_global = float(np.mean(global_dice[1:]))
    print(f"  {'Mean (Fg)':<10} {mean_fg_global:>7.4f}")

    # Mean-of-cases dice
    case_dices = [r["mean_dice_fg"] for r in all_results.values()]
    per_class_dices = {
        cls: np.mean([r["metrics"][cls]["dice"] for r in all_results.values()])
        for cls in CLASS_NAMES
    }

    print(f"\n  Mean-of-cases foreground Dice : {np.mean(case_dices):.4f}")
    print(f"  Std  (across cases)           : {np.std(case_dices):.4f}")
    print(f"  Per-class mean-of-cases Dice  :")
    for cls in CLASS_NAMES:
        print(f"    {cls:<10}: {per_class_dices[cls]:.4f}")

    # ── Save summary JSON ────────────────────────────────────────────────────
    summary = {
        "checkpoint": str(args.checkpoint),
        "num_cases": len(all_cases),
        "global_metrics": {
            cls: {
                "dice": float(global_dice[i]),
                "iou": float(global_iou[i]),
                "precision": float(global_prec[i]),
                "recall": float(global_rec[i]),
            }
            for i, cls in enumerate(CLASS_NAMES)
        },
        "mean_fg_dice_global": mean_fg_global,
        "mean_fg_dice_per_case": float(np.mean(case_dices)),
        "std_fg_dice_per_case": float(np.std(case_dices)),
        "per_class_mean_of_cases": {
            cls: float(per_class_dices[cls]) for cls in CLASS_NAMES
        },
        "per_case": all_results,
    }

    summary_path = output_dir / "blind_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved → {summary_path}")
    if not args.no_save_nifti:
        print(f"  NIfTI preds  → {nifti_dir}/")
    if not args.no_save_figures:
        print(f"  Figures      → {figure_dir}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
