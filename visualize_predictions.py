"""
visualize_predictions.py
========================
Runs sliding-window inference with the best saved checkpoint from
swinunetr_v9_optimized.py and produces supervisor-ready figures.

Outputs (saved to <output_dir>/visualizations/):
  - Per-case PNG:  axial / coronal / sagittal panels (CT | GT | Pred | Error map)
  - summary.png :  bar chart of per-class Dice across all visualised cases
  - metrics.csv :  table with Dice / IoU / Precision / Recall per case and class

Usage (on Linux, activate your env first):
  python visualize_predictions.py                         # auto-selects checkpoint
  python visualize_predictions.py --ckpt best_stage2.pth  # specific checkpoint
  python visualize_predictions.py --n_cases 10            # number of val cases
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import csv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import model + helpers from the training script (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from swinunetr_v9_optimized import (
    SwinUNETR_LLM,
    get_config,
    _load_volume,
    _cache_dir_for,
    _sliding_window_inference,
    postprocess,
    CLASS_NAMES,
    _DTYPE,
)

# ---------------------------------------------------------------------------
# Colour map:  0=bg(transparent)  1=kidney(blue)  2=tumor(red)  3=cyst(green)
# ---------------------------------------------------------------------------
SEG_COLORS = np.array([
    [0,   0,   0,   0  ],   # background – transparent
    [30,  144, 255, 180],   # kidney     – dodger blue
    [255, 50,  50,  200],   # tumor      – red
    [50,  205, 50,  200],   # cyst       – lime green
], dtype=np.uint8)

SEG_CMAP = ListedColormap(SEG_COLORS[:, :3] / 255.0, name="seg")

LEGEND_PATCHES = [
    mpatches.Patch(color=SEG_COLORS[1, :3]/255, label="Kidney"),
    mpatches.Patch(color=SEG_COLORS[2, :3]/255, label="Tumor"),
    mpatches.Patch(color=SEG_COLORS[3, :3]/255, label="Cyst"),
]


# ---------------------------------------------------------------------------
# Dice / IoU helpers
# ---------------------------------------------------------------------------

def dice_per_class(pred: np.ndarray, gt: np.ndarray, n_cls: int = 4) -> np.ndarray:
    eps = 1e-8
    out = np.zeros(n_cls)
    for c in range(n_cls):
        p = pred == c; g = gt == c
        out[c] = (2 * (p & g).sum()) / (p.sum() + g.sum() + eps)
    return out


def iou_per_class(pred: np.ndarray, gt: np.ndarray, n_cls: int = 4) -> np.ndarray:
    eps = 1e-8
    out = np.zeros(n_cls)
    for c in range(n_cls):
        p = pred == c; g = gt == c
        out[c] = (p & g).sum() / ((p | g).sum() + eps)
    return out


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _overlay_seg(ax, ct_slice: np.ndarray, seg_slice: np.ndarray, title: str):
    """Show CT greyscale with coloured seg overlay."""
    ax.imshow(ct_slice, cmap="gray", vmin=0, vmax=1, interpolation="none")
    # Build RGBA overlay
    rgba = np.zeros((*seg_slice.shape, 4), dtype=np.uint8)
    for c in range(1, 4):
        mask = seg_slice == c
        rgba[mask] = SEG_COLORS[c]
    ax.imshow(rgba, interpolation="none")
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")


def _ct_slice(ax, ct_slice: np.ndarray, title: str):
    ax.imshow(ct_slice, cmap="gray", vmin=0, vmax=1, interpolation="none")
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")


def _error_map(ax, pred_slice: np.ndarray, gt_slice: np.ndarray, title: str):
    """
    Green = correct foreground,  Red = false positive,  Blue = false negative.
    """
    h, w = pred_slice.shape
    rgb = np.zeros((h, w, 3))
    # correct fg (any class > 0)
    correct = (pred_slice > 0) & (pred_slice == gt_slice)
    fp      = (pred_slice > 0) & (gt_slice == 0)
    fn      = (pred_slice == 0) & (gt_slice > 0)
    rgb[correct] = [0.2, 0.9, 0.2]   # green
    rgb[fp]      = [0.9, 0.1, 0.1]   # red
    rgb[fn]      = [0.1, 0.3, 0.9]   # blue
    ax.imshow(rgb, interpolation="none")
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Find "best" slice for a volume (highest foreground count)
# ---------------------------------------------------------------------------

def _best_slice(seg: np.ndarray, axis: int) -> int:
    counts = np.array([(seg.take(i, axis=axis) > 0).sum()
                       for i in range(seg.shape[axis])])
    return int(counts.argmax()) if counts.max() > 0 else seg.shape[axis] // 2


# ---------------------------------------------------------------------------
# Per-case figure
# ---------------------------------------------------------------------------

def save_case_figure(case_id: str, ct: np.ndarray, gt: np.ndarray,
                     pred: np.ndarray, dice: np.ndarray, out_dir: Path):
    """
    4 columns × 3 rows:
      cols: CT | GT overlay | Pred overlay | Error map
      rows: axial (best) | coronal (best) | sagittal (best)
    """
    ax_idx = _best_slice(gt, 0)
    co_idx = _best_slice(gt, 1)
    sa_idx = _best_slice(gt, 2)

    views = [
        ("Axial",    ax_idx, 0),
        ("Coronal",  co_idx, 1),
        ("Sagittal", sa_idx, 2),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.patch.set_facecolor("#1a1a2e")

    dice_str = (f"K={dice[1]:.3f}  T={dice[2]:.3f}  C={dice[3]:.3f}  "
                f"MeanFG={dice[1:].mean():.3f}")
    fig.suptitle(f"{case_id}   |   Dice → {dice_str}", color="white",
                 fontsize=11, y=1.01)

    for row, (view_name, idx, axis) in enumerate(views):
        ct_sl  = np.take(ct,  idx, axis=axis)
        gt_sl  = np.take(gt,  idx, axis=axis)
        pr_sl  = np.take(pred, idx, axis=axis)

        _ct_slice(axes[row, 0],   ct_sl,               f"{view_name} – CT")
        _overlay_seg(axes[row, 1], ct_sl, gt_sl,        f"{view_name} – Ground Truth")
        _overlay_seg(axes[row, 2], ct_sl, pr_sl,        f"{view_name} – Prediction")
        _error_map(axes[row, 3],  pr_sl, gt_sl,         f"{view_name} – Error Map")

    # Legend
    fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=3,
               framealpha=0.3, labelcolor="white", fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    # Error map legend
    err_patches = [
        mpatches.Patch(color=[0.2,0.9,0.2], label="Correct FG"),
        mpatches.Patch(color=[0.9,0.1,0.1], label="False Positive"),
        mpatches.Patch(color=[0.1,0.3,0.9], label="False Negative"),
    ]
    ax_extra = fig.add_axes([0.76, -0.06, 0.22, 0.04])
    ax_extra.axis("off")
    ax_extra.legend(handles=err_patches, loc="center", ncol=3,
                    framealpha=0, fontsize=7, labelcolor="white")

    plt.tight_layout()
    fig.savefig(str(out_dir / f"{case_id}.png"), dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def save_summary_figure(results: list, out_dir: Path):
    """Bar chart of per-class Dice for every case, plus mean lines."""
    if not results:
        return
    case_ids = [r["case_id"] for r in results]
    dice_k   = np.array([r["dice"][1] for r in results])
    dice_t   = np.array([r["dice"][2] for r in results])
    dice_c   = np.array([r["dice"][3] for r in results])
    mean_fg  = np.array([r["mean_fg"]  for r in results])

    x = np.arange(len(case_ids))
    w = 0.22

    fig, ax = plt.subplots(figsize=(max(12, len(case_ids)*0.8), 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.bar(x - w,     dice_k,  w, label="Kidney", color="#1E90FF", alpha=0.85)
    ax.bar(x,         dice_t,  w, label="Tumor",  color="#FF3232", alpha=0.85)
    ax.bar(x + w,     dice_c,  w, label="Cyst",   color="#32CD32", alpha=0.85)
    ax.bar(x + 2*w,  mean_fg,  w, label="MeanFG", color="#FFD700", alpha=0.85)

    # Mean lines
    for val, col, ls, lbl in [(dice_k.mean(), "#1E90FF", "--", f"Kidney mean={dice_k.mean():.3f}"),
                               (dice_t.mean(), "#FF3232", "-.", f"Tumor mean={dice_t.mean():.3f}"),
                               (dice_c.mean(), "#32CD32", ":",  f"Cyst mean={dice_c.mean():.3f}"),
                               (mean_fg.mean(),"#FFD700", "-",  f"MeanFG={mean_fg.mean():.3f}")]:
        ax.axhline(val, color=col, linestyle=ls, linewidth=1.2, label=lbl)

    ax.set_xticks(x + w/2)
    ax.set_xticklabels(case_ids, rotation=45, ha="right", fontsize=7, color="white")
    ax.set_ylabel("Dice Score", color="white")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-case Dice — SwinUNETR V9 Optimized", color="white", fontsize=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=7, labelcolor="white", framealpha=0.2,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(str(out_dir / "summary.png"), dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved summary chart -> {out_dir / 'summary.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_model(config: dict, ckpt_path: Path, device: torch.device) -> SwinUNETR_LLM:
    print(f"\nBuilding model ...")
    model = SwinUNETR_LLM(
        num_classes   = config["num_classes"],
        feature_size  = config["feature_size"],
        use_llm       = config["use_llm"],
        llm_model_name= config["llm_model_name"],
        llm_layer_idx = config["llm_layer_idx"],
        lora_rank     = config["lora_rank"],
        use_checkpoint= False,          # no gradient checkpointing during inference
    )

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Accept both raw state_dict and our checkpoint dicts
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        stage = ckpt.get("stage", "unknown")
        epoch = ckpt.get("epoch", "?")
        print(f"  Checkpoint stage={stage}  epoch={epoch}")
    else:
        state = ckpt

    # torch.compile() adds '_orig_mod.' prefix to all keys — strip it
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        print("  Stripping torch.compile '_orig_mod.' prefix from state dict ...")
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    # If checkpoint came from after Stage-2 LoRA was applied the keys differ;
    # detect by presence of "lora_A" and apply LoRA before loading.
    has_lora_encoder = any("swinViT" in k and "lora_A" in k for k in state.keys())
    if has_lora_encoder:
        print("  Detected S2 LoRA encoder weights — applying LoRA before load ...")
        from swinunetr_v9_optimized import apply_lora_to_linears
        apply_lora_to_linears(model.net.swinViT, rank=config["lora_s2_rank"],
                              target_names=("qkv", "proj", "fc1", "fc2"))

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded: {len(state)-len(unexpected)} keys  "
          f"| missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")

    model.to(device)
    model.eval()
    return model


def pick_checkpoint(output_dir: Path) -> Path:
    """Return best_stage2.pth if exists, else best_stage1.pth."""
    for name in ("best_stage2.pth", "best_stage1.pth", "resume_state.pth"):
        p = output_dir / name
        if p.exists():
            print(f"  Auto-selected checkpoint: {p}")
            return p
    raise FileNotFoundError(
        f"No checkpoint found in {output_dir}. "
        "Train at least one stage or pass --ckpt explicitly."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default=None,
                        help="Checkpoint filename (relative to output_dir) or full path")
    parser.add_argument("--n_cases",  type=int, default=8,
                        help="Number of validation cases to visualize (default 8)")
    parser.add_argument("--output_dir", default="./output/swin_v9_optimized",
                        help="Model output directory (same as training config)")
    parser.add_argument("--kits23_dir", default="./kits23/dataset",
                        help="KiTS23 dataset root (case_XXXXX folders)")
    parser.add_argument("--device",   default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--no_save_nii", action="store_true",
                        help="Skip saving NIfTI prediction masks")
    parser.add_argument("--sw_overlap", type=float, default=0.0,
                        help="Sliding-window overlap fraction (default 0.0 = fastest, use 0.25 for best quality)")
    parser.add_argument("--sw_batch_size", type=int, default=4,
                        help="Patches per SW inference batch (default 4, increase if VRAM allows)")
    args = parser.parse_args()

    device     = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    vis_dir    = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint ───────────────────────────────────────────────────────────
    if args.ckpt:
        ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else output_dir / args.ckpt
    else:
        ckpt_path = pick_checkpoint(output_dir)

    # ── Config ───────────────────────────────────────────────────────────────
    config = get_config("full")
    config["kits23_dir"] = args.kits23_dir

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(config, ckpt_path, device)

    # ── Val cases ────────────────────────────────────────────────────────────
    data_dir  = Path(config["kits23_dir"])
    cache_dir = _cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists():
        cache_dir = None

    all_cases = sorted([c for c in data_dir.iterdir()
                        if c.is_dir() and c.name.startswith("case_")])
    # mirror training split: last 40 are test, first n_val are val
    all_cases = all_cases[:-config.get("test_cases", 40)]
    n_val     = int(len(all_cases) * config["val_split"])
    val_cases  = all_cases[:n_val]

    n_show = min(args.n_cases, len(val_cases))
    # Pick evenly spaced cases so we see variety
    indices   = np.linspace(0, len(val_cases) - 1, n_show, dtype=int)
    show_cases = [val_cases[i] for i in indices]

    print(f"\nRunning inference on {n_show} val cases -> {vis_dir}\n")

    results = []
    for case_dir in tqdm(show_cases, desc="Inference"):
        case_id  = case_dir.name
        img_path = case_dir / "imaging.nii.gz"
        lbl_path = case_dir / "segmentation.nii.gz"
        if not img_path.exists():
            print(f"  Skipping {case_id} (no imaging.nii.gz)"); continue

        d = {"image": str(img_path), "label": str(lbl_path), "case_id": case_id}
        ct, lbl_gt = _load_volume(d, cache_dir)

        # Sliding window inference
        img_t = torch.from_numpy(ct).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_logits = _sliding_window_inference(
                img_t,
                roi_size      = config["patch_size"],
                sw_batch_size = args.sw_batch_size,
                predictor     = model,
                overlap       = args.sw_overlap,
                mode          = "constant",
            )
        pred = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()
        pred = postprocess(pred, config)

        # Metrics
        dice = dice_per_class(pred, lbl_gt)
        iou  = iou_per_class(pred, lbl_gt)
        mean_fg = float(dice[1:].mean())
        print(f"  {case_id}: K={dice[1]:.3f}  T={dice[2]:.3f}  "
              f"C={dice[3]:.3f}  MeanFG={mean_fg:.3f}")

        results.append({
            "case_id": case_id,
            "dice":    dice.tolist(),
            "iou":     iou.tolist(),
            "mean_fg": mean_fg,
        })

        # Figure
        save_case_figure(case_id, ct, lbl_gt, pred, dice, vis_dir)

        # Optional NIfTI mask
        if not args.no_save_nii:
            nii_pred = nib.Nifti1Image(pred.astype(np.uint8), affine=np.eye(4))
            nib.save(nii_pred, str(vis_dir / f"{case_id}_pred.nii.gz"))

        del img_t, pred_logits
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────────
    if results:
        save_summary_figure(results, vis_dir)

        # CSV
        csv_path = vis_dir / "metrics.csv"
        fieldnames = ["case_id",
                      "dice_bg", "dice_kidney", "dice_tumor", "dice_cyst", "mean_fg_dice",
                      "iou_bg",  "iou_kidney",  "iou_tumor",  "iou_cyst"]
        with open(str(csv_path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow({
                    "case_id":     r["case_id"],
                    "dice_bg":     f"{r['dice'][0]:.4f}",
                    "dice_kidney": f"{r['dice'][1]:.4f}",
                    "dice_tumor":  f"{r['dice'][2]:.4f}",
                    "dice_cyst":   f"{r['dice'][3]:.4f}",
                    "mean_fg_dice":f"{r['mean_fg']:.4f}",
                    "iou_bg":      f"{r['iou'][0]:.4f}",
                    "iou_kidney":  f"{r['iou'][1]:.4f}",
                    "iou_tumor":   f"{r['iou'][2]:.4f}",
                    "iou_cyst":    f"{r['iou'][3]:.4f}",
                })
        print(f"  Saved metrics CSV -> {csv_path}")

        # Print summary table
        print("\n" + "="*60)
        print(f"{'Case':<20} {'Kidney':>8} {'Tumor':>8} {'Cyst':>8} {'MeanFG':>8}")
        print("-"*60)
        for r in results:
            d = r["dice"]
            print(f"{r['case_id']:<20} {d[1]:>8.3f} {d[2]:>8.3f} {d[3]:>8.3f} {r['mean_fg']:>8.3f}")
        print("-"*60)
        all_dice = np.array([r["dice"] for r in results])
        print(f"{'MEAN':<20} {all_dice[:,1].mean():>8.3f} {all_dice[:,2].mean():>8.3f} "
              f"{all_dice[:,3].mean():>8.3f} {np.array([r['mean_fg'] for r in results]).mean():>8.3f}")
        print("="*60)

    print(f"\nDone.  All outputs saved to:  {vis_dir}")


if __name__ == "__main__":
    main()
