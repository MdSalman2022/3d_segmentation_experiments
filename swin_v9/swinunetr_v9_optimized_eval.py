"""
swinunetr_v9_optimized_eval.py
=====================
Runs the saved best_final.pth on the KiTS23 test set (last 40 cases),
computes per-case metrics, saves NIfTI predictions, and produces
publication-ready tables + figures.

Usage
-----
  python swinunetr_v9_optimized_eval.py                              # auto-find best_final.pth
  python swinunetr_v9_optimized_eval.py --checkpoint path/to/ckpt   # explicit checkpoint
  python swinunetr_v9_optimized_eval.py --split test                 # test | val | all
  python swinunetr_v9_optimized_eval.py --no-save-nifti             # skip NIfTI export

Outputs  (./output/swin_v9_optimized/predict_export/)
  per_case_metrics.csv          per-case Dice / IoU / Prec / Rec for all classes
  summary_metrics.csv           aggregate mean ± std per class
  figures/
    01_per_case_dice.png         grouped bar chart – Dice per case × class
    02_class_boxplots.png        box-whisker per class
    03_summary_radar.png         radar chart – K / T / C / MeanFG
    04_scatter_kidney_tumor.png  per-case kidney vs tumour Dice scatter
    05_slice_overlay_*.png       axial + coronal overlays for top-5 + worst-5 + random-5
  predictions/
    case_XXXXX_pred.nii.gz       integer label map (0=bg 1=kidney 2=tumor 3=cyst)
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["CUDA_MODULE_LOADING"]     = "LAZY"

# ── matplotlib (non-interactive) ──────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import pandas as pd

# ── import shared model / utilities from training script ──────────────────
sys.path.insert(0, str(Path(__file__).parent))
from swinunetr_v9_optimized import (
    SwinUNETR_LLM, get_config, _load_volume, _cache_dir_for,
    _strip_compiled_prefix, postprocess, _sliding_window_inference,
    CLASS_NAMES, _DTYPE,
)

# ── colour palette for segmentation overlays ─────────────────────────────
_SEG_CMAP = ListedColormap(["none", "#4FC3F7", "#EF5350", "#AB47BC"])
_CLASS_COLORS = ["#4FC3F7", "#EF5350", "#AB47BC"]  # kidney, tumor, cyst
_CLASS_LABELS  = ["Kidney", "Tumor", "Cyst"]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path: Path, config: dict, device: torch.device):
    model = SwinUNETR_LLM(
        num_classes    = config["num_classes"],
        feature_size   = config["feature_size"],
        use_llm        = config.get("use_llm", True),
        llm_model_name = config["llm_model_name"],
        llm_layer_idx  = config["llm_layer_idx"],
        lora_rank      = config["lora_rank"],
        use_checkpoint = False,          # no gradient checkpointing at inference
    ).to(device)

    # Apply S2 LoRA adapters so the state-dict keys match the checkpoint
    if config.get("lora_s2", True):
        model.unfreeze_encoder_with_lora(lora_rank=config["lora_s2_rank"])

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = _strip_compiled_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] Missing keys: {len(missing)}")
    if unexpected:
        print(f"  [warn] Unexpected keys: {len(unexpected)}")
    # Re-move everything to device: unfreeze_encoder_with_lora adds new modules
    # that default to CPU; load_state_dict does not re-pin them.
    model.to(device)
    model.eval()
    print(f"  Checkpoint loaded: {checkpoint_path.name}")
    return model


# =============================================================================
# TEST-SET CASE LIST
# =============================================================================

def get_test_cases(config: dict, split: str = "test"):
    data_dir  = Path(config["kits23_dir"])
    all_cases = sorted([c for c in data_dir.iterdir()
                        if c.is_dir() and c.name.startswith("case_")])
    n_test = config.get("test_cases", 40)
    n_val  = config.get("val_cases",  40)

    if split == "test":
        selected = all_cases[-n_test:]
    elif split == "val":
        selected = all_cases[-(n_test + n_val):-n_test]
    else:  # "all"
        selected = all_cases

    cases = [
        {"image": str(c / "imaging.nii.gz"),
         "label": str(c / "segmentation.nii.gz"),
         "case_id": c.name}
        for c in selected
        if (c / "imaging.nii.gz").exists() and (c / "segmentation.nii.gz").exists()
    ]
    print(f"  {len(cases)} cases selected for split='{split}'")
    return cases


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(pred: np.ndarray, gt: np.ndarray, n_cls: int = 4):
    eps = 1e-8
    dice, iou, prec, rec = [], [], [], []
    for c in range(n_cls):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        d  = (2 * tp) / (2 * tp + fp + fn + eps)
        i  = tp / (tp + fp + fn + eps)
        p  = tp / (tp + fp + eps)
        r  = tp / (tp + fn + eps)
        dice.append(d); iou.append(i); prec.append(p); rec.append(r)
    return np.array(dice), np.array(iou), np.array(prec), np.array(rec)


# =============================================================================
# INFERENCE LOOP
# =============================================================================

def run_inference(model, cases, config, device, pred_dir: Path, save_nifti: bool = True):
    cache_dir = _cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists():
        cache_dir = None

    rows = []
    for d in tqdm(cases, desc="Inference"):
        case_id = d["case_id"]
        try:
            img, lbl_gt = _load_volume(d, cache_dir)
            img_t = (torch.from_numpy(img).float()
                     .unsqueeze(0).unsqueeze(0).to(device))

            with torch.no_grad():
                logits = _sliding_window_inference(
                    img_t,
                    roi_size      = config["patch_size"],
                    sw_batch_size = config.get("sw_batch_size", 4),
                    predictor     = model,
                    overlap       = config.get("sw_overlap", 0.5),
                    mode          = "gaussian",
                )
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int32)
            pred = postprocess(pred, config)

            dice, iou, prec, rec = compute_metrics(pred, lbl_gt)
            row = {"case_id": case_id}
            for ci, cn in enumerate(CLASS_NAMES):
                row[f"dice_{cn}"]  = round(float(dice[ci]),  4)
                row[f"iou_{cn}"]   = round(float(iou[ci]),   4)
                row[f"prec_{cn}"]  = round(float(prec[ci]),  4)
                row[f"rec_{cn}"]   = round(float(rec[ci]),   4)
            row["mean_fg_dice"] = round(float(dice[1:].mean()), 4)
            row["mean_fg_iou"]  = round(float(iou[1:].mean()),  4)
            rows.append(row)

            if save_nifti:
                orig_nii = nib.load(d["image"])
                pred_nii = nib.Nifti1Image(pred.astype(np.uint8), orig_nii.affine, orig_nii.header)
                nib.save(pred_nii, str(pred_dir / f"{case_id}_pred.nii.gz"))

            del img_t, logits
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error on {case_id}: {e}")

    return pd.DataFrame(rows)


# =============================================================================
# FIGURE 1 — per-case Dice grouped bar chart
# =============================================================================

def fig_per_case_dice(df: pd.DataFrame, out_path: Path):
    cases  = df["case_id"].tolist()
    x      = np.arange(len(cases))
    width  = 0.28

    fig, ax = plt.subplots(figsize=(max(14, len(cases) * 0.35), 5))
    for i, (col, label, color) in enumerate(zip(
            ["dice_kidney", "dice_tumor", "dice_cyst"],
            _CLASS_LABELS, _CLASS_COLORS)):
        ax.bar(x + i * width, df[col], width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("case_", "") for c in cases],
                       rotation=75, fontsize=7)
    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Case Dice by Class")
    ax.set_ylim(0, 1.05)
    ax.axhline(df["dice_kidney"].mean(), color=_CLASS_COLORS[0],
               linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(df["dice_tumor"].mean(),  color=_CLASS_COLORS[1],
               linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(df["dice_cyst"].mean(),   color=_CLASS_COLORS[2],
               linestyle="--", linewidth=1, alpha=0.6)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =============================================================================
# FIGURE 2 — box-whisker per class
# =============================================================================

def fig_class_boxplots(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dice boxplot
    data_dice = [df["dice_kidney"], df["dice_tumor"], df["dice_cyst"]]
    bp = axes[0].boxplot(data_dice, patch_artist=True, notch=False,
                         medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], _CLASS_COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[0].set_xticklabels(_CLASS_LABELS)
    axes[0].set_ylabel("Dice Score")
    axes[0].set_title("Dice Distribution by Class")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)
    for i, vals in enumerate(data_dice):
        axes[0].text(i + 1, vals.median() + 0.02, f"{vals.median():.3f}",
                     ha="center", fontsize=8, fontweight="bold")

    # IoU boxplot
    data_iou = [df["iou_kidney"], df["iou_tumor"], df["iou_cyst"]]
    bp2 = axes[1].boxplot(data_iou, patch_artist=True, notch=False,
                          medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp2["boxes"], _CLASS_COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[1].set_xticklabels(_CLASS_LABELS)
    axes[1].set_ylabel("IoU Score")
    axes[1].set_title("IoU Distribution by Class")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)
    for i, vals in enumerate(data_iou):
        axes[1].text(i + 1, vals.median() + 0.02, f"{vals.median():.3f}",
                     ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Per-Class Performance Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =============================================================================
# FIGURE 3 — radar chart
# =============================================================================

def fig_radar(df: pd.DataFrame, out_path: Path):
    metrics = {
        "Kidney\nDice":   df["dice_kidney"].mean(),
        "Tumor\nDice":    df["dice_tumor"].mean(),
        "Cyst\nDice":     df["dice_cyst"].mean(),
        "Mean FG\nDice":  df["mean_fg_dice"].mean(),
        "Kidney\nIoU":    df["iou_kidney"].mean(),
        "Tumor\nIoU":     df["iou_tumor"].mean(),
        "Kidney\nRecall": df["rec_kidney"].mean(),
        "Tumor\nRecall":  df["rec_tumor"].mean(),
    }
    labels = list(metrics.keys())
    vals   = list(metrics.values())
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals   += vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, "o-", linewidth=2, color="#1565C0")
    ax.fill(angles, vals, alpha=0.25, color="#1565C0")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Model Performance Radar", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =============================================================================
# FIGURE 4 — kidney vs tumour scatter
# =============================================================================

def fig_scatter(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["dice_kidney"], df["dice_tumor"],
                    c=df["dice_cyst"], cmap="plasma",
                    s=60, alpha=0.8, edgecolors="grey", linewidths=0.4)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Cyst Dice", fontsize=9)
    ax.set_xlabel("Kidney Dice", fontsize=11)
    ax.set_ylabel("Tumor Dice",  fontsize=11)
    ax.set_title("Kidney vs Tumor Dice (colour = Cyst Dice)", fontsize=12)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.axvline(df["dice_kidney"].mean(), color=_CLASS_COLORS[0],
               linestyle="--", linewidth=1, alpha=0.7,
               label=f"Kidney mean={df['dice_kidney'].mean():.3f}")
    ax.axhline(df["dice_tumor"].mean(),  color=_CLASS_COLORS[1],
               linestyle="--", linewidth=1, alpha=0.7,
               label=f"Tumor mean={df['dice_tumor'].mean():.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =============================================================================
# FIGURE 5 — slice overlays  (axial + coronal for selected cases)
# =============================================================================

def _best_slice(vol_3d: np.ndarray, axis: int = 0):
    """Return the slice index along `axis` with most foreground voxels."""
    counts = (vol_3d > 0).sum(axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(counts))


def fig_slice_overlay(case_dict: dict, pred_path: Path, label: str,
                      out_path: Path, cache_dir):
    img, lbl_gt = _load_volume(case_dict, cache_dir)
    pred        = nib.load(str(pred_path)).get_fdata().astype(np.int32)

    # pick best axial and coronal slices from ground truth
    ax_sl  = _best_slice(lbl_gt, axis=0)
    cor_sl = _best_slice(lbl_gt, axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(f"{case_dict['case_id']}  [{label}]", fontsize=12)

    views = [
        (img[ax_sl],   lbl_gt[ax_sl],   pred[ax_sl],   "Axial"),
        (img[:, cor_sl], lbl_gt[:, cor_sl], pred[:, cor_sl], "Coronal"),
    ]
    for row, (im_sl, gt_sl, pr_sl, view_name) in enumerate(views):
        vmin, vmax = np.percentile(im_sl, [1, 99])

        axes[row, 0].imshow(im_sl, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[row, 0].set_title(f"{view_name} — Image", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(im_sl, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[row, 1].imshow(np.ma.masked_equal(gt_sl, 0), cmap=_SEG_CMAP,
                            vmin=0, vmax=3, alpha=0.55, origin="lower")
        axes[row, 1].set_title(f"{view_name} — Ground Truth", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(im_sl, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[row, 2].imshow(np.ma.masked_equal(pr_sl, 0), cmap=_SEG_CMAP,
                            vmin=0, vmax=3, alpha=0.55, origin="lower")
        axes[row, 2].set_title(f"{view_name} — Prediction", fontsize=9)
        axes[row, 2].axis("off")

    legend_patches = [
        mpatches.Patch(color="#4FC3F7", label="Kidney"),
        mpatches.Patch(color="#EF5350", label="Tumor"),
        mpatches.Patch(color="#AB47BC", label="Cyst"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def make_slice_figures(df: pd.DataFrame, cases: list, pred_dir: Path,
                       fig_dir: Path, config: dict, n_top=5, n_worst=5, n_rand=3):
    cache_dir = _cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists():
        cache_dir = None

    sorted_df  = df.sort_values("mean_fg_dice", ascending=False).reset_index(drop=True)
    top_ids    = sorted_df.head(n_top)["case_id"].tolist()
    worst_ids  = sorted_df.tail(n_worst)["case_id"].tolist()
    rng        = np.random.default_rng(42)
    mid_pool   = sorted_df.iloc[n_top:-n_worst]["case_id"].tolist()
    rand_ids   = rng.choice(mid_pool, size=min(n_rand, len(mid_pool)), replace=False).tolist()

    case_map   = {c["case_id"]: c for c in cases}
    groups     = [("top",   top_ids), ("worst", worst_ids), ("random", rand_ids)]

    for group_name, ids in groups:
        for cid in ids:
            pred_path = pred_dir / f"{cid}_pred.nii.gz"
            if not pred_path.exists():
                print(f"  [skip] No prediction for {cid}")
                continue
            row   = df[df["case_id"] == cid].iloc[0]
            label = (f"MeanFG={row['mean_fg_dice']:.3f} "
                     f"K={row['dice_kidney']:.3f} "
                     f"T={row['dice_tumor']:.3f} "
                     f"C={row['dice_cyst']:.3f} [{group_name}]")
            out   = fig_dir / f"05_slice_{group_name}_{cid}.png"
            try:
                fig_slice_overlay(case_map[cid], pred_path, label, out, cache_dir)
                print(f"  Saved: {out.name}")
            except Exception as e:
                print(f"  Overlay error {cid}: {e}")


# =============================================================================
# SUMMARY TABLE  (printed + CSV)
# =============================================================================

def print_summary(df: pd.DataFrame, out_csv: Path):
    fg_cols = ["dice_kidney", "dice_tumor", "dice_cyst",
               "iou_kidney",  "iou_tumor",  "iou_cyst",
               "prec_kidney", "prec_tumor", "prec_cyst",
               "rec_kidney",  "rec_tumor",  "rec_cyst",
               "mean_fg_dice", "mean_fg_iou"]
    stats = df[fg_cols].agg(["mean", "std", "median", "min", "max"]).T
    stats.columns = ["Mean", "Std", "Median", "Min", "Max"]
    stats = stats.round(4)

    print("\n" + "=" * 72)
    print("  TEST SET SUMMARY")
    print("=" * 72)
    print(stats.to_string())
    print("=" * 72)
    print(f"  Cases evaluated: {len(df)}")
    print(f"  MeanFG Dice : {df['mean_fg_dice'].mean():.4f} ± {df['mean_fg_dice'].std():.4f}")
    print(f"  Kidney Dice : {df['dice_kidney'].mean():.4f} ± {df['dice_kidney'].std():.4f}")
    print(f"  Tumor  Dice : {df['dice_tumor'].mean():.4f}  ± {df['dice_tumor'].std():.4f}")
    print(f"  Cyst   Dice : {df['dice_cyst'].mean():.4f}  ± {df['dice_cyst'].std():.4f}")
    print("=" * 72 + "\n")

    stats.to_csv(out_csv)
    print(f"  Summary CSV → {out_csv}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Predict + export KiTS23 test results")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to .pth checkpoint (default: auto-find best_final.pth)")
    ap.add_argument("--output-dir", default="./output/swin_v9_optimized",
                    help="Training output directory")
    ap.add_argument("--export-dir", default=None,
                    help="Where to write results (default: <output_dir>/predict_export)")
    ap.add_argument("--split", default="test", choices=["test", "val", "all"],
                    help="Which split to evaluate")
    ap.add_argument("--no-save-nifti", action="store_true",
                    help="Skip saving NIfTI predictions")
    ap.add_argument("--no-overlays", action="store_true",
                    help="Skip slice overlay figures (faster)")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="Sliding window overlap (default 0.5)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── directories ──────────────────────────────────────────────────────
    out_dir    = Path(args.output_dir)
    export_dir = Path(args.export_dir) if args.export_dir else out_dir / "predict_export"
    pred_dir   = export_dir / "predictions"
    fig_dir    = export_dir / "figures"
    export_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    # ── checkpoint ───────────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = out_dir / "best_final.pth"
    if not ckpt_path.exists():
        sys.exit(f"  [error] Checkpoint not found: {ckpt_path}")
    print(f"  Checkpoint: {ckpt_path}")

    # ── config ───────────────────────────────────────────────────────────
    cfg = get_config("full")
    cfg["output_dir"] = str(out_dir)
    cfg["sw_overlap"] = args.overlap

    # Load model
    model = load_model(ckpt_path, cfg, device)
    model.eval()

    # Get cases
    cases = get_test_cases(cfg, split=args.split)
    if not cases:
        sys.exit("  [error] No cases found — check kits23_dir in config.")

    # ── inference ────────────────────────────────────────────────────────
    print(f"\n  Running sliding-window inference (overlap={args.overlap}) ...")
    df = run_inference(model, cases, cfg, device,
                       pred_dir=pred_dir,
                       save_nifti=not args.no_save_nifti)
    if df.empty:
        sys.exit("  [error] No predictions produced.")

    # ── save per-case CSV ────────────────────────────────────────────────
    pc_csv = export_dir / "per_case_metrics.csv"
    df.to_csv(pc_csv, index=False)
    print(f"  Per-case CSV → {pc_csv}")

    # ── summary ──────────────────────────────────────────────────────────
    print_summary(df, export_dir / "summary_metrics.csv")

    # ── figures ──────────────────────────────────────────────────────────
    print("\n  Generating figures ...")
    fig_per_case_dice(df, fig_dir / "01_per_case_dice.png")
    fig_class_boxplots(df, fig_dir / "02_class_boxplots.png")
    fig_radar(df, fig_dir / "03_summary_radar.png")
    fig_scatter(df, fig_dir / "04_scatter_kidney_tumor.png")

    if not args.no_overlays:
        print("  Generating slice overlays (top/worst/random cases) ...")
        make_slice_figures(df, cases, pred_dir, fig_dir, cfg)

    # ── save JSON summary ─────────────────────────────────────────────────
    summary = {
        "checkpoint":    str(ckpt_path),
        "split":         args.split,
        "n_cases":       len(df),
        "mean_fg_dice":  round(float(df["mean_fg_dice"].mean()), 4),
        "mean_fg_iou":   round(float(df["mean_fg_iou"].mean()),  4),
        "kidney_dice":   round(float(df["dice_kidney"].mean()),  4),
        "tumor_dice":    round(float(df["dice_tumor"].mean()),   4),
        "cyst_dice":     round(float(df["dice_cyst"].mean()),    4),
        "kidney_dice_std": round(float(df["dice_kidney"].std()), 4),
        "tumor_dice_std":  round(float(df["dice_tumor"].std()),  4),
        "cyst_dice_std":   round(float(df["dice_cyst"].std()),   4),
    }
    (export_dir / "predict_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print(f"  Done!  Results in: {export_dir}")
    print(f"  MeanFG Dice = {summary['mean_fg_dice']:.4f}")
    print(f"  Kidney      = {summary['kidney_dice']:.4f}  "
          f"Tumor = {summary['tumor_dice']:.4f}  "
          f"Cyst  = {summary['cyst_dice']:.4f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
