"""
WaLM-Net evaluation + visualization
====================================
Load a trained checkpoint (best.pth) and, for each case, produce:

  * a per-class figure (CT | Kidney | Tumor | Cyst, rows = Ground truth / Prediction)
  * the predicted segmentation as a NIfTI volume (kidney=1, tumor=2, cyst=3)
  * per-case Dice for kidney / tumor / cyst, plus a metrics CSV + JSON

Only needs `walmnet_kits23.py` next to this file (PyTorch + MONAI + nibabel +
matplotlib + scipy). No Modal required. Cases are always drawn from the held-out
validation split (first val_split fraction) — never training data.

Two selection modes
--------------------
1. First-N (default):
       python walmnet_eval.py --num-cases 8

2. Class-targeted — pick the cases that contain the most of each structure and
   render each figure on the slice where that structure is largest:
       python walmnet_eval.py --n-kidney 4 --n-tumor 8 --n-cyst 8
   Figures land in eval_outputs/figures/<kidney|tumor|cyst>/.

Examples
--------
    python walmnet_eval.py --checkpoint output/walmnet_medium/best.pth --mode medium \
                           --n-kidney 4 --n-tumor 8 --n-cyst 8
    python walmnet_eval.py --num-cases 0            # 0 = all val cases (first-N mode)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _import_W():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import walmnet_kits23
    return walmnet_kits23


CLASS_NAMES = ["Kidney", "Tumor", "Cyst"]            # index 0..2 -> label 1..3
# RGBA per class: kidney=blue, tumor=red, cyst=green
CLASS_RGBA = [(0.15, 0.45, 1.0, 0.55), (1.0, 0.15, 0.15, 0.60), (0.10, 0.80, 0.25, 0.60)]


def _per_class_dice(pred, gt, num_classes=4):
    out = []
    for c in range(1, num_classes):
        p, g = (pred == c), (gt == c)
        denom = p.sum() + g.sum()
        out.append(float("nan") if denom == 0 else 2.0 * (p & g).sum() / denom)
    return out


def _overlay(ax, ct_slice, mask_slice, rgba):
    import numpy as np
    ax.imshow(ct_slice, cmap="gray")
    rgb = np.zeros((*mask_slice.shape, 4), dtype=float)
    rgb[mask_slice.astype(bool)] = rgba
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")


def _save_figure(case, ct, gt, pred, dice, target_cls, fig_path):
    """2x4 panel (rows GT/Pred, cols All+each class), sliced at target_cls's peak."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # slice = where the target class is largest; fall back to any FG, then middle
    ref = (gt == target_cls) if (gt == target_cls).any() else (pred == target_cls)
    if not ref.any():
        ref = (gt > 0) if (gt > 0).any() else (pred > 0)
    z = int(np.argmax(ref.sum(axis=(0, 1)))) if ref.any() else ct.shape[2] // 2

    ct_s, gt_s, pr_s = ct[:, :, z].T, gt[:, :, z].T, pred[:, :, z].T
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for r, (lbl, name) in enumerate([(gt_s, "GT (all)"), (pr_s, "Pred (all)")]):
        axs[r, 0].imshow(ct_s, cmap="gray")
        comp = np.zeros((*lbl.shape, 4))
        for c in range(3):
            comp[lbl == c + 1] = CLASS_RGBA[c]
        axs[r, 0].imshow(comp, interpolation="nearest"); axs[r, 0].axis("off")
        axs[r, 0].set_title(name)
    for c in range(3):
        _overlay(axs[0, c + 1], ct_s, gt_s == c + 1, CLASS_RGBA[c])
        _overlay(axs[1, c + 1], ct_s, pr_s == c + 1, CLASS_RGBA[c])
        axs[0, c + 1].set_title(f"GT {CLASS_NAMES[c]}")
        axs[1, c + 1].set_title(f"Pred {CLASS_NAMES[c]}  (Dice={dice[c]:.3f})")
    tname = CLASS_NAMES[target_cls - 1]
    fig.suptitle(f"{case}  -  {tname} view (axial z={z})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(str(fig_path) + "." + ext, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _save_multislice(case, ct, gt, pred, target_cls, n_rows, fig_path):
    """Rows = N axial slices, cols = CT | Ground Truth | Prediction (overlays)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ref = (gt == target_cls) if (gt == target_cls).any() else (gt > 0)
    if not ref.any():
        ref = (pred > 0)
    zs = np.where(ref.sum(axis=(0, 1)) > 0)[0]
    if len(zs) == 0:
        zs = np.array([ct.shape[2] // 2])
    # evenly spaced slices across the structure's extent
    idx = np.linspace(0, len(zs) - 1, min(n_rows, len(zs))).round().astype(int)
    chosen = [int(zs[i]) for i in idx]

    rows = len(chosen)
    fig, axs = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    if rows == 1:
        axs = axs[None, :]
    for r, z in enumerate(chosen):
        ct_s, gt_s, pr_s = ct[:, :, z].T, gt[:, :, z].T, pred[:, :, z].T
        for col, lbl in enumerate([None, gt_s, pr_s]):
            ax = axs[r, col]
            ax.imshow(ct_s, cmap="gray"); ax.axis("off")
            if lbl is not None:
                comp = np.zeros((*lbl.shape, 4))
                for c in range(3):
                    comp[lbl == c + 1] = CLASS_RGBA[c]
                ax.imshow(comp, interpolation="nearest")
            if r == 0:
                ax.set_title(["CT", "Ground Truth", "WaLM-Net"][col], fontsize=13)
        axs[r, 0].text(-0.08, 0.5, f"z={z}", transform=axs[r, 0].transAxes,
                       rotation=90, va="center", fontsize=10)
    fig.suptitle(f"{case}  -  {CLASS_NAMES[target_cls-1]}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(str(fig_path) + "_multislice." + ext, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _scan_class_counts(val_files):
    """GT voxel count per class (1,2,3) for each case (native space, for ranking)."""
    import numpy as np
    import nibabel as nib
    counts = {}
    for i, item in enumerate(val_files):
        lbl = np.asanyarray(nib.load(item["label"]).dataobj)
        counts[item["image"]] = {c: int((lbl == c).sum()) for c in (1, 2, 3)}
        print(f"  scan {i+1}/{len(val_files)} {Path(item['image']).parent.name}", flush=True)
    return counts


def _build_tf(cfg):
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, EnsureTyped,
    )
    a_min, a_max = cfg["ct_window"]
    return Compose([
        LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])


def _infer(model, item, cfg, device, tf):
    """Run inference on one case -> (ct, gt, pred) numpy arrays."""
    import numpy as np
    import torch
    from monai.data import Dataset, DataLoader
    from monai.inferers import sliding_window_inference
    ld = DataLoader(Dataset([item], tf), batch_size=1, num_workers=0)
    for b in ld:
        img = b["image"].to(device)
        gt = b["label"][0, 0].cpu().numpy().astype(np.int16)
        lg = sliding_window_inference(img, cfg["patch_size"], cfg["sw_batch_size"],
                                      model, overlap=cfg["sw_overlap"])
        pred = torch.argmax(lg[0], 0).cpu().numpy().astype(np.int16)
        return img[0, 0].cpu().numpy(), gt, pred


def _select_by_dice(model, val_files, cfg, device, wanted, worst):
    """Prescan: infer all val cases, rank each class by predicted Dice."""
    import torch
    tf = _build_tf(cfg)
    dice_by_path = {}
    model.eval()
    with torch.no_grad():
        for i, it in enumerate(val_files):
            _, gt, pred = _infer(model, it, cfg, device, tf)
            dice_by_path[it["image"]] = _per_class_dice(pred, gt, cfg["num_classes"])
            print(f"  prescan {i+1}/{len(val_files)} {Path(it['image']).parent.name}", flush=True)
    case2targets: dict[str, list[int]] = {}
    for cls, n in wanted.items():
        if n <= 0:
            continue
        cand = [p for p in dice_by_path if dice_by_path[p][cls - 1] == dice_by_path[p][cls - 1]]
        cand.sort(key=lambda p: dice_by_path[p][cls - 1], reverse=not worst)
        for p in cand[:n]:
            case2targets.setdefault(p, []).append(cls)
        tag = "worst" if worst else "best"
        print(f"  {CLASS_NAMES[cls-1]} ({tag}): "
              f"{[Path(p).parent.name for p in cand[:n]]}", flush=True)
    return case2targets


def evaluate(args):
    import numpy as np
    import nibabel as nib
    import torch
    from monai.data import Dataset, DataLoader
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, EnsureTyped,
    )

    W = _import_W()
    cfg = W.get_config(args.mode)
    if args.patch:
        cfg["patch_size"] = tuple(int(v) for v in args.patch.split(","))
    cfg["kits23_dir"] = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    nii_dir = out_dir / "pred_nifti"
    nii_dir.mkdir(parents=True, exist_ok=True)

    model = W.build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} "
          f"(epoch {ckpt.get('epoch')}, score {ckpt.get('score')})", flush=True)

    # held-out validation split (same deterministic split as training)
    items = W._list_cases(cfg["kits23_dir"])
    if not items:
        raise FileNotFoundError(f"No KiTS23 cases under {cfg['kits23_dir']}")
    n_val = max(1, int(len(items) * cfg["val_split"]))
    val_files = items[:n_val]
    print(f"Validation pool: {len(val_files)} held-out cases", flush=True)

    targeted = (args.n_kidney + args.n_tumor + args.n_cyst) > 0

    # ---- decide which (case -> [target classes]) to render -------------
    case2targets: dict[str, list[int]] = {}
    by_path = {it["image"]: it for it in val_files}
    if targeted:
        wanted = {1: args.n_kidney, 2: args.n_tumor, 3: args.n_cyst}
        if args.select in ("best", "worst"):
            print(f"Prescanning all val cases to rank by {args.select} Dice ...", flush=True)
            case2targets = _select_by_dice(model, val_files, cfg, device, wanted,
                                           worst=(args.select == "worst"))
        else:  # size: rank by ground-truth structure volume (default, unbiased)
            print("Scanning GT for size-based selection ...", flush=True)
            counts = _scan_class_counts(val_files)
            for cls, n in wanted.items():
                if n <= 0:
                    continue
                ranked = sorted((p for p in counts if counts[p][cls] > 0),
                                key=lambda p: counts[p][cls], reverse=True)
                picked = ranked[:n]
                cname = CLASS_NAMES[cls - 1]
                if len(picked) < n:
                    print(f"  [warn] only {len(picked)} val cases contain {cname} "
                          f"(asked {n})", flush=True)
                print(f"  {cname}: {[Path(p).parent.name for p in picked]}", flush=True)
                for p in picked:
                    case2targets.setdefault(p, []).append(cls)
        selected = list(case2targets.keys())
    else:
        sel = val_files if args.num_cases == 0 else val_files[:args.num_cases]
        selected = [it["image"] for it in sel]
        for p in selected:
            case2targets[p] = [1]   # generic view, sliced on kidney/any-FG
    print(f"Rendering {len(selected)} unique cases", flush=True)

    # ---- run inference once per selected case --------------------------
    a_min, a_max = cfg["ct_window"]
    tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    sel_items = [by_path[p] for p in selected]
    loader = DataLoader(Dataset(sel_items, tf), batch_size=1, num_workers=2)

    rows = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            path = sel_items[idx]["image"]
            case = Path(path).parent.name
            img = batch["image"].to(device)
            gt = batch["label"][0, 0].cpu().numpy().astype(np.int16)
            logits = sliding_window_inference(
                img, cfg["patch_size"], cfg["sw_batch_size"], model, overlap=cfg["sw_overlap"])
            pred = torch.argmax(logits[0], dim=0).cpu().numpy().astype(np.int16)
            ct = img[0, 0].cpu().numpy()

            dice = _per_class_dice(pred, gt, cfg["num_classes"])
            rows.append({"case": case, "dice_kidney": dice[0],
                         "dice_tumor": dice[1], "dice_cyst": dice[2]})
            print(f"  {case}: kidney={dice[0]:.4f} tumor={dice[1]:.4f} cyst={dice[2]:.4f}",
                  flush=True)

            for cls in case2targets[path]:
                sub = (out_dir / "figures" / CLASS_NAMES[cls - 1].lower()) if targeted \
                    else (out_dir / "figures")
                sub.mkdir(parents=True, exist_ok=True)
                _save_figure(case, ct, gt, pred, dice, cls, sub / case)
                if args.n_slices > 0:
                    _save_multislice(case, ct, gt, pred, cls, args.n_slices, sub / case)

            nib.save(nib.Nifti1Image(pred.astype(np.uint8), np.eye(4)),
                     str(nii_dir / f"{case}_pred.nii.gz"))

    # ---- aggregate ------------------------------------------------------
    def _nanmean(key):
        vals = [r[key] for r in rows if r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    means = {"dice_kidney": _nanmean("dice_kidney"),
             "dice_tumor": _nanmean("dice_tumor"),
             "dice_cyst": _nanmean("dice_cyst")}
    means["mean_fg"] = sum(v for v in means.values() if v == v) / 3.0

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["case", "dice_kidney", "dice_tumor", "dice_cyst"])
        wri.writeheader(); wri.writerows(rows)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "mode": args.mode, "targeted": targeted,
         "n_cases": len(rows), "per_case": rows, "mean": means}, indent=2))

    print(f"\n=== Mean Dice over {len(rows)} rendered cases ===")
    print(f"  Kidney = {means['dice_kidney']:.4f}")
    print(f"  Tumor  = {means['dice_tumor']:.4f}")
    print(f"  Cyst   = {means['dice_cyst']:.4f}")
    print(f"  MeanFG = {means['mean_fg']:.4f}")
    print(f"\nFigures -> {out_dir/'figures'}\nNIfTI -> {nii_dir}\nMetrics -> {out_dir/'metrics.csv'}")


def main(argv=None):
    p = argparse.ArgumentParser(description="WaLM-Net eval + per-class visualization")
    p.add_argument("--checkpoint", default="output/walmnet_medium/best.pth")
    p.add_argument("--mode", default="medium", choices=["full", "medium", "quick_test"])
    p.add_argument("--data-dir", default="./kits23/dataset")
    p.add_argument("--num-cases", type=int, default=8,
                   help="first-N mode: how many val cases (0 = all). Ignored if --n-* used.")
    p.add_argument("--n-kidney", type=int, default=0, help="targeted: # cases for kidney figures")
    p.add_argument("--n-tumor", type=int, default=0, help="targeted: # cases for tumor figures")
    p.add_argument("--n-cyst", type=int, default=0, help="targeted: # cases for cyst figures")
    p.add_argument("--select", default="size", choices=["size", "best", "worst"],
                   help="targeted ranking: size=most GT voxels (default), "
                        "best/worst=by predicted Dice (prescans all val cases)")
    p.add_argument("--patch", default="", help="override sliding-window patch, e.g. 128,128,128")
    p.add_argument("--n-slices", type=int, default=3,
                   help="also save a CT|GT|Pred multi-slice figure with this many rows (0 = off)")
    p.add_argument("--out", default="eval_outputs")
    evaluate(p.parse_args(argv))


if __name__ == "__main__":
    main()
