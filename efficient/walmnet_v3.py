"""
WaLM-Net v3 — corrected evaluation + inference/training improvements
====================================================================
v3 is a driver around the WaLM-Net model in `walmnet_kits23.py` (unchanged
architecture) that fixes the correctness issues found reviewing v2 and adds the
improvements most likely to lift the remaining weak points (cyst, boundary/NSD).

WHAT v2 GOT WRONG / LEFT ON THE TABLE  ->  WHAT v3 DOES
-------------------------------------------------------
1. Metrics computed at 1.5 mm resampled resolution (not comparable to KiTS).
   -> `evaluate_native`: run inference at 1.5 mm, resample the prediction back to
      the ORIGINAL label grid (nearest), and score Dice/NSD in native space.
2. NSD tolerance was in voxels (SurfaceDiceMetric called without `spacing`).
   -> pass the native voxel `spacing` and KiTS-style tolerances to the metric.
3. Checkpoint selected with `score:null` never recorded during training.
   -> validation runs on the native KiTS HEC metric every `val_every`, the score
      is logged to history.json, and best.pth is chosen on it.
4. Cyst hurt by false positives (GT-empty cases scored 0.0) and tiny debris.
   -> inference post-processing: keep the 2 largest kidney+mass components,
      drop tumor/cyst far from kidney, and remove sub-threshold T/C components.
5. No test-time augmentation.
   -> optional flip TTA (softmax-averaged), on by default at eval.
6. Reported vs cherry-picked: v2 also emitted "targeted" (largest-structure)
   metrics that are optimistically biased.
   -> v3 reports ONLY the full held-out val set in KiTS HEC format.

Optional (needs a fresh train, invalidates old checkpoints):
   --ct-norm znorm  ->  nnU-Net-style clip-to-HU-range + z-score normalisation,
   which typically improves low-contrast tumor/cyst contrast over plain min-max.

The model, loss, blocks, and helpers are imported from `walmnet_kits23.py`.

Usage
-----
    python walmnet_v3.py --selftest                                  # CPU logic check (no data)
    python walmnet_v3.py --evaluate --checkpoint output/walmnet_medium/best.pth
    python walmnet_v3.py --train --mode full
    python walmnet_v3.py --evaluate --checkpoint <ckpt> --no-tta --no-postproc  # ablate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def _import_W():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import walmnet_kits23
    return walmnet_kits23


# KiTS23 Hierarchical Evaluation Classes
HECS = ["Kidney+Masses", "Masses", "Tumor"]
HEC_LABELS = [(1, 2, 3), (2, 3), (2,)]
# KiTS official surface-dice tolerances (mm) per HEC.
HEC_NSD_TOL = {"Kidney+Masses": 1.0, "Masses": 1.0, "Tumor": 1.0}


# ===========================================================================
# Configuration
# ===========================================================================

def get_config(mode: str = "full") -> dict:
    W = _import_W()
    cfg = W.get_config(mode)
    cfg.update({
        "output_dir": f"./output/walmnet_v3_{mode}",
        # ---- evaluation (v3 corrections) ----
        "eval_native": True,         # score at original resolution, not 1.5 mm
        "selection": "kits_mean",    # mean of the 3 HEC Dice (native)
        "use_tta": True,             # flip TTA at final/held-out inference
        "val_tta": False,            # keep per-epoch validation fast (no TTA)
        "tta_flips": ((), (2,), (3,), (4,)),   # identity + 3 single-axis flips
        "sw_overlap": 0.6,           # (was 0.5) higher overlap helps boundary/NSD
        "sw_mode": "gaussian",       # Gaussian window blending (was 'constant')
        # ---- post-processing (cyst FP suppression; recall-safe defaults) ----
        "postprocess": True,
        "keep_n_kidney": 2,          # keep 2 largest kidney components (two kidneys)
        "tumor_kidney_max_dist": 15.0,
        "min_tumor_mm3": 30.0,       # drop only tiny tumor debris
        "min_cyst_mm3": 20.0,        # conservative: protects small real cysts
        # ---- intensity normalisation ----
        "ct_norm": "minmax",         # "minmax" (v2-compatible) | "znorm" (nnU-Net-style)
        "ct_clip": (-79.0, 304.0),   # KiTS foreground HU range for znorm
        "ct_znorm_stats": (101.0, 76.9),  # (mean, std) fallback for znorm
        # ---- extra augmentation (only affects fresh training) ----
        "extra_aug": True,
    })
    if mode == "full":
        cfg.update({
            "num_epochs": 250,
            "iters_per_epoch": 250,
            "val_every": 10,
            "val_cases": 12,
            "min_epochs_before_stop": 100,
            # nudge the loss further toward the rare classes (cyst especially)
            "ce_weight": (0.1, 1.0, 6.0, 6.0),
            "fg_dice_weights": (1.0, 3.0, 3.0),
        })
    elif mode == "medium":
        cfg.update({
            "num_epochs": 140,
            "val_every": 10,
            "val_cases": 10,
            "min_epochs_before_stop": 50,
        })
    return cfg


# ===========================================================================
# Data (adds optional z-score normalisation + light extra augmentation)
# ===========================================================================

def _intensity_transforms(cfg, training: bool):
    """Return the intensity-normalisation transform(s) for image key."""
    if cfg.get("ct_norm") == "znorm":
        lo, hi = cfg["ct_clip"]
        mean, std = cfg["ct_znorm_stats"]
        # clip to KiTS foreground HU range, then fixed z-score (dataset-level stats)
        from monai.transforms import ThresholdIntensityd, ShiftIntensityd, ScaleIntensityd
        return [
            ThresholdIntensityd(keys=["image"], threshold=hi, above=False, cval=hi),
            ThresholdIntensityd(keys=["image"], threshold=lo, above=True, cval=lo),
            ShiftIntensityd(keys=["image"], offset=-mean),
            ScaleIntensityd(keys=["image"], factor=1.0 / std),
        ]
    from monai.transforms import ScaleIntensityRanged
    a_min, a_max = cfg["ct_window"]
    return [ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max,
                                 b_min=0.0, b_max=1.0, clip=True)]


def get_dataloaders(cfg):
    """Training/val loaders. Val loader keeps the label in NATIVE space (image
    resampled to cfg['spacing'] only) so metrics can be computed at original res."""
    from monai.data import CacheDataset, DataLoader
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        CropForegroundd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
        RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised,
        RandAdjustContrastd, EnsureTyped,
    )
    W = _import_W()
    items = W._list_cases(cfg["kits23_dir"])
    if not items:
        raise FileNotFoundError(f"No KiTS23 cases under {cfg['kits23_dir']}")
    n_val = max(1, int(len(items) * cfg["val_split"]))
    val_files, train_files = items[:n_val], items[n_val:]
    pos, neg = cfg["pos_neg_ratio"]
    inten = _intensity_transforms(cfg, training=True)

    train_pre = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        *inten,
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
    aug = [
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=cfg["patch_size"], pos=pos, neg=neg,
                               num_samples=cfg["num_samples_per_volume"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
    ]
    if cfg.get("extra_aug"):
        aug += [
            RandGaussianNoised(keys=["image"], prob=0.15, std=0.02),
            RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.8, 1.25)),
        ]
    train_tf = Compose(train_pre + aug + [EnsureTyped(keys=["image", "label"])])

    # Validation: resample ONLY the image; keep the label at native resolution.
    val_tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=cfg["spacing"], mode="bilinear"),
        *_intensity_transforms(cfg, training=False),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(train_files, train_tf, cache_rate=cfg["cache_rate"],
                            num_workers=cfg["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    return train_loader, val_files, val_tf


# ===========================================================================
# Inference: TTA + native-resolution prediction + post-processing
# ===========================================================================

def _spacing_from_affine(affine):
    import numpy as np
    a = np.asarray(affine, dtype=float)[:3, :3]
    return tuple(float(np.linalg.norm(a[:, i])) for i in range(3))


def _label_spacing(lbl_t):
    """Extract native voxel spacing (mm) from a (possibly batched) MetaTensor."""
    import numpy as np
    aff = getattr(lbl_t, "affine", None)
    if aff is None:
        return (1.0, 1.0, 1.0)
    if hasattr(aff, "cpu"):
        aff = aff.cpu().numpy()
    aff = np.asarray(aff, dtype=float)
    if aff.ndim == 3:            # batched -> (B,4,4)
        aff = aff[0]
    return _spacing_from_affine(aff)


def predict_logits(model, img, cfg, device):
    """Sliding-window inference with optional flip TTA -> softmax probs (1,C,...)."""
    import torch
    import torch.nn.functional as F
    from monai.inferers import sliding_window_inference

    flips = cfg["tta_flips"] if cfg.get("use_tta") else ((),)
    prob = None
    for fl in flips:
        x = torch.flip(img, dims=list(fl)) if fl else img
        lg = sliding_window_inference(x, cfg["patch_size"], cfg["sw_batch_size"],
                                      model, overlap=cfg.get("sw_overlap", 0.5),
                                      mode=cfg.get("sw_mode", "gaussian"))
        if fl:
            lg = torch.flip(lg, dims=list(fl))
        p = F.softmax(lg, dim=1)
        prob = p if prob is None else prob + p
    return prob / len(flips)


def resample_pred_to(prob, ref_shape):
    """Resample softmax probs (1,C,D,H,W) to ref_shape (D0,H0,W0) then argmax."""
    import torch
    import torch.nn.functional as F
    if prob.shape[2:] != tuple(ref_shape):
        prob = F.interpolate(prob, size=tuple(ref_shape), mode="trilinear", align_corners=False)
    return torch.argmax(prob[0], dim=0).cpu().numpy()


def postprocess(pred, cfg, spacing=(1.0, 1.0, 1.0)):
    """Recall-safe false-positive suppression.

    Order matters and is deliberately conservative so real tumor/cyst are NOT
    deleted:
      1. Kidney speck removal — keep only the N largest *kidney* components
         (removes spurious kidney blobs; never touches tumor/cyst).
      2. Distance filter — drop tumor/cyst components with no voxel within
         `tumor_kidney_max_dist` mm of remaining kidney (only if kidney exists).
      3. Small-component filter — drop tumor/cyst components below a *small*
         volume threshold (debris only; thresholds kept low to protect small
         real cysts, which are common in KiTS).
    pred: int numpy array (D,H,W), labels 0..3. Returns a cleaned copy.
    """
    import numpy as np
    from scipy import ndimage

    if not cfg.get("postprocess"):
        return pred
    out = pred.copy()
    voxel_mm3 = float(spacing[0] * spacing[1] * spacing[2])

    # 1. keep the N largest KIDNEY components (does not risk deleting a tumor)
    keep_n = int(cfg.get("keep_n_kidney", 2))
    kidney = out == 1
    if keep_n > 0 and kidney.any():
        lab, n = ndimage.label(kidney)
        if n > keep_n:
            sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
            keep = set((np.argsort(sizes)[::-1][:keep_n] + 1).tolist())
            drop = kidney & ~np.isin(lab, list(keep))
            out[drop] = 0

    # 2. remove tumor/cyst far from any (remaining) kidney voxel
    max_dist = cfg.get("tumor_kidney_max_dist", 0)
    kidney = out == 1
    if max_dist and max_dist > 0 and kidney.any():
        dist = ndimage.distance_transform_edt(~kidney, sampling=spacing)
        allowed = dist <= float(max_dist)
        for cid in (2, 3):
            mask = out == cid
            if not mask.any():
                continue
            lab, n = ndimage.label(mask)
            if n == 0:
                continue
            keep = np.zeros(n + 1, dtype=bool)
            keep[np.unique(lab[mask & allowed])] = True
            keep[0] = True
            out[mask & ~keep[lab]] = 0

    # 3. drop sub-threshold tumor / cyst debris (thresholds kept conservative)
    for cid, key in ((2, "min_tumor_mm3"), (3, "min_cyst_mm3")):
        thr = cfg.get(key, 0.0)
        if thr and thr > 0:
            mask = out == cid
            if not mask.any():
                continue
            lab, n = ndimage.label(mask)
            if n == 0:
                continue
            sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1)) * voxel_mm3
            small = [i + 1 for i, s in enumerate(sizes) if s < thr]
            if small:
                out[np.isin(lab, small)] = 0
    return out


# ===========================================================================
# Native-resolution KiTS HEC evaluation (the correctness centrepiece)
# ===========================================================================

def _dice(pm, gm):
    denom = pm.sum() + gm.sum()
    return float("nan") if denom == 0 else float(2.0 * (pm & gm).sum() / denom)


def _binary_stats(pm, gm):
    """Dice, IoU, recall, precision for a binary pred/gt pair (NaN when undefined)."""
    tp = float((pm & gm).sum()); pp = float(pm.sum()); gp = float(gm.sum())
    union = pp + gp - tp
    return {
        "dice": float("nan") if (pp + gp) == 0 else 2.0 * tp / (pp + gp),
        "iou": float("nan") if union == 0 else tp / union,
        "recall": float("nan") if gp == 0 else tp / gp,       # sensitivity to real lesions
        "precision": float("nan") if pp == 0 else tp / pp,    # 1-FP-rate proxy
    }


def evaluate_native(model, val_files, cfg, device, n_cases=None, logger=None,
                    out_dir=None):
    """Score the full held-out val split in native space, KiTS HEC format."""
    import numpy as np
    import torch
    from monai.data import Dataset, DataLoader

    log = (logger.info if logger else print)
    _, _, val_tf = get_dataloaders_for_eval(cfg)
    subset = val_files[:n_cases] if n_cases else val_files
    loader = DataLoader(Dataset(subset, val_tf), batch_size=1, num_workers=2)

    try:
        from monai.metrics import SurfaceDiceMetric
        have_nsd = True
    except Exception:  # noqa: BLE001
        have_nsd = False

    model.eval()
    per_case = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            case = Path(subset[i]["image"]).parent.name
            img = b["image"].to(device)
            lbl_t = b["label"]
            gt = lbl_t[0, 0].cpu().numpy().astype(np.int16)
            spacing = _label_spacing(lbl_t)
            prob = predict_logits(model, img, cfg, device)
            pred = resample_pred_to(prob, gt.shape).astype(np.int16)
            pred = postprocess(pred, cfg, spacing=spacing)

            rec = {"case": case}
            # KiTS Hierarchical Evaluation Classes (the headline metric)
            for name, labs in zip(HECS, HEC_LABELS):
                pm, gm = np.isin(pred, labs), np.isin(gt, labs)
                rec[f"{name}_dice"] = _dice(pm, gm)
                if have_nsd:
                    rec[f"{name}_nsd"] = _nsd(pm, gm, spacing, HEC_NSD_TOL[name])
            # Per-class stats (kidney/tumor/cyst) so tables can cite REAL numbers
            # and cyst recall vs precision (FP) can be read separately.
            for cid, cname in ((1, "kidney"), (2, "tumor"), (3, "cyst")):
                st = _binary_stats(pred == cid, gt == cid)
                for k, v in st.items():
                    rec[f"{cname}_{k}"] = v
            per_case.append(rec)
            log(f"  {case}: " + " ".join(f"{h}={rec[h+'_dice']:.3f}" for h in HECS)
                + f" | cyst(rec/prec)={rec['cyst_recall']}/{rec['cyst_precision']}")

    def _mean(k):
        vals = [r[k] for r in per_case if r.get(k) == r.get(k)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    summary = {h: {"dice": _mean(f"{h}_dice")} for h in HECS}
    if have_nsd:
        for h in HECS:
            summary[h]["nsd"] = _mean(f"{h}_nsd")
    avg_dice = sum(summary[h]["dice"] for h in HECS) / 3.0

    per_class = {c: {k: _mean(f"{c}_{k}") for k in ("dice", "iou", "recall", "precision")}
                 for c in ("kidney", "tumor", "cyst")}

    mixer = "mamba" if _has_mamba() and cfg.get("mixer", "auto") != "lite" else "lite"
    metrics = {"avg_dice": avg_dice, "hec": summary, "per_class": per_class}

    if out_dir is not None:
        out = {"avg_dice": avg_dice, "hec": summary, "per_class": per_class,
               "n_cases": len(per_case), "native": True, "mixer": mixer,
               "tta": cfg.get("use_tta"), "postproc": cfg.get("postprocess"),
               "sw_overlap": cfg.get("sw_overlap"), "ct_norm": cfg.get("ct_norm"),
               "per_case": per_case}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "kits_metrics_native.json").write_text(json.dumps(out, indent=2))
    return avg_dice, metrics


def _has_mamba():
    try:
        return bool(_import_W()._HAS_MAMBA)
    except Exception:  # noqa: BLE001
        return False


def _nsd(pm, gm, spacing, tol):
    import numpy as np
    import torch
    try:
        from monai.metrics import compute_surface_dice
        # proper 2-channel one-hot [background, foreground]; score foreground only
        p = np.stack([~pm, pm]).astype(np.float32)[None]   # (1,2,D,H,W)
        g = np.stack([~gm, gm]).astype(np.float32)[None]
        val = compute_surface_dice(torch.from_numpy(p), torch.from_numpy(g),
                                   class_thresholds=[float(tol)],
                                   include_background=False, spacing=spacing)
        return float(val.item())
    except Exception:  # noqa: BLE001
        return float("nan")


def get_dataloaders_for_eval(cfg):
    """Val-only transform builder (no training loaders); reuses get_dataloaders logic."""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, EnsureTyped,
    )
    val_tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=cfg["spacing"], mode="bilinear"),
        *_intensity_transforms(cfg, training=False),
        EnsureTyped(keys=["image", "label"]),
    ])
    W = _import_W()
    items = W._list_cases(cfg["kits23_dir"])
    n_val = max(1, int(len(items) * cfg["val_split"]))
    return None, items[:n_val], val_tf


# ===========================================================================
# Training (validation uses the native KiTS metric; score is recorded)
# ===========================================================================

def train(cfg, resume: bool = False):
    import numpy as np
    import torch
    import torch.nn as nn
    W = _import_W()

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"]); torch.backends.cudnn.benchmark = True

    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(out_dir / "training.log"),
                                  logging.StreamHandler(sys.stdout)], force=True)
    logger = logging.getLogger("walmnet_v3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_files, _ = get_dataloaders(cfg)
    model = W.build_model(cfg).to(device)
    total, _ = W.count_parameters(model)
    loss_fn = W.build_loss(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = W.cosine_warmup_scheduler(optimizer, cfg["num_epochs"], cfg["warmup_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    ema = W.ModelEMA(model, cfg["ema_decay"]) if cfg.get("ema_decay", 0) > 0 else None
    es = W.EarlyStopping(cfg["patience"], out_dir / "best.pth")

    logger.info("=" * 72)
    logger.info(f"WaLM-Net v3 training | params={total/1e6:.2f}M | ct_norm={cfg['ct_norm']} "
                f"| tta={cfg['use_tta']} | postproc={cfg['postprocess']}")
    logger.info(f"Train {len(train_loader.dataset)} | Val {len(val_files)} | selection={cfg['selection']} (native)")
    logger.info("=" * 72)

    data_iter = W._cycle(train_loader)
    accum = cfg["grad_accum_steps"]
    history = {"epochs": []}

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train(); t0, running = time.time(), 0.0
        optimizer.zero_grad(set_to_none=True)
        for it in range(cfg["iters_per_epoch"]):
            batch = next(data_iter)
            img = batch["image"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"] and device.type == "cuda"):
                loss = loss_fn(model(img), lbl) / accum
            scaler.scale(loss).backward(); running += loss.item() * accum
            if (it + 1) % accum == 0:
                if cfg.get("grad_clip", 0) > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
        scheduler.step()
        train_loss = running / cfg["iters_per_epoch"]
        elapsed = (time.time() - t0) / 60

        score, metrics = None, None
        if epoch % cfg["val_every"] == 0 or epoch == cfg["num_epochs"]:
            eval_model = ema.module if ema is not None else model
            val_cfg = {**cfg, "use_tta": cfg.get("val_tta", False)}  # fast per-epoch val
            score, metrics = evaluate_native(eval_model, val_files, val_cfg, device,
                                             n_cases=cfg["val_cases"], logger=logger)
            improved = es.step(score, eval_model, epoch)
            logger.info(f"  KiTS mean Dice (native) = {score:.4f}"
                        + (" *BEST*" if improved else f" (no-improve {es.counter}/{es.patience})"))
        logger.info(f"[E{epoch:03d}/{cfg['num_epochs']}] train_loss={train_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.1f}min")
        history["epochs"].append({"epoch": epoch, "train_loss": train_loss,
                                  "score": score, "metrics": metrics, "time_min": elapsed})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.module.state_dict() if ema else None,
                    "epoch": epoch, "score": score}, out_dir / "last.pth")
        if es.early_stop and epoch >= cfg["min_epochs_before_stop"]:
            logger.info(f"Early stopping at epoch {epoch} (best={es.best:.4f})"); break

    # final full-val native evaluation with the best checkpoint
    final_model = ema.module if ema is not None else model
    if (out_dir / "best.pth").exists():
        best = torch.load(out_dir / "best.pth", map_location=device)
        final_model.load_state_dict(best["model_state_dict"])
        logger.info(f"Loaded best (epoch {best.get('epoch')}, score {best.get('score')})")
    avg, metrics = evaluate_native(final_model, val_files, cfg, device, n_cases=None,
                                   logger=logger, out_dir=out_dir)
    summary = {"version": "walmnet_v3", "mode": cfg.get("mode"),
               "params_millions": round(total / 1e6, 3),
               "best_val_subset_score": es.best, "full_val_avg_dice": avg,
               "full_val_hec": metrics["hec"], "n_val_cases": len(val_files),
               "total_hours": round(sum(e["time_min"] for e in history["epochs"]) / 60, 2),
               "config": {k: str(v) for k, v in cfg.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"FULL val native KiTS mean Dice = {avg:.4f} over {len(val_files)} cases")
    return summary


def evaluate(cfg, checkpoint: str):
    import torch
    W = _import_W()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = W.build_model(cfg).to(device)
    ck = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ck.get("ema_state_dict") or ck.get("model_state_dict") or ck)
    print(f"Loaded {checkpoint} (epoch {ck.get('epoch')})", flush=True)
    _, val_files, _ = get_dataloaders_for_eval(cfg)
    out_dir = Path(cfg["output_dir"]);
    avg, metrics = evaluate_native(model, val_files, cfg, device, n_cases=None,
                                   out_dir=out_dir)
    print(f"\n=== KiTS23 HEC (native resolution, {len(val_files)} val cases) ===")
    print(f"  Average Dice = {avg:.4f}   [tta={cfg['use_tta']} postproc={cfg['postprocess']} "
          f"overlap={cfg['sw_overlap']} ct_norm={cfg['ct_norm']}]")
    for h in HECS:
        line = f"  {h:<14} Dice={metrics['hec'][h]['dice']:.4f}"
        if "nsd" in metrics["hec"][h]:
            line += f"  NSD={metrics['hec'][h]['nsd']:.4f}"
        print(line)
    print("  --- per-class (native; use THESE for tables, not hardcoded values) ---")
    for c in ("kidney", "tumor", "cyst"):
        pc = metrics["per_class"][c]
        print(f"  {c:<7} Dice={pc['dice']:.4f} IoU={pc['iou']:.4f} "
              f"recall={pc['recall']:.4f} precision={pc['precision']:.4f}")
    print(f"\nSaved -> {out_dir/'kits_metrics_native.json'}")


# ===========================================================================
# Self-test (CPU, no MONAI/data): validates the v3 logic that we can run here
# ===========================================================================

def selftest():
    import numpy as np
    print("=" * 60)
    print("WaLM-Net v3 self-test (CPU, logic only)")
    print("=" * 60)
    cfg = get_config("quick_test")

    # (a) post-processing (recall-safe): real tumor + near cyst kept; far/tiny FP removed
    vol = np.zeros((64, 64, 64), dtype=np.int16)
    vol[20:40, 20:40, 20:40] = 1          # kidney blob
    vol[24:32, 24:32, 40:48] = 2          # tumor next to kidney (kept: near + big)
    vol[24:30, 24:30, 24:30] = 3          # cyst inside kidney, 216 mm^3 (kept)
    vol[0:2, 0:2, 0:2] = 3                # tiny far cyst FP, 8 mm^3 (removed: far + tiny)
    vol[54:62, 54:62, 54:62] = 3          # large far cyst FP, 512 mm^3 (removed: far)
    cleaned = postprocess(vol, cfg, spacing=(1.0, 1.0, 1.0))
    tumor_kept = (cleaned[24:32, 24:32, 40:48] == 2).all()
    near_cyst = (cleaned[24:30, 24:30, 24:30] == 3).all()
    far_gone = (cleaned[54:62, 54:62, 54:62] == 3).sum() == 0
    tiny_gone = (cleaned[0:2, 0:2, 0:2] == 3).sum() == 0
    print(f"[postproc] tumor kept={tumor_kept} near cyst kept={near_cyst} "
          f"far FP removed={far_gone} tiny FP removed={tiny_gone}")
    assert tumor_kept and near_cyst and far_gone and tiny_gone, "postprocess logic failed"

    # (b) native resample: probs (1,C,8,8,8) -> native (16,16,16) argmax
    import torch
    prob = torch.zeros(1, 4, 8, 8, 8); prob[:, 2] = 1.0     # all tumor
    pred = resample_pred_to(prob, (16, 16, 16))
    print(f"[resample] out shape={pred.shape} unique={np.unique(pred).tolist()} (expect all 2)")
    assert pred.shape == (16, 16, 16) and set(np.unique(pred).tolist()) == {2}

    # (c) HEC dice + spacing-from-affine
    gt = np.zeros((10, 10, 10), dtype=np.int16); gt[2:8, 2:8, 2:8] = 2
    d = _dice(np.isin(gt, (2,)), np.isin(gt, (2,)))
    sp = _spacing_from_affine(np.diag([1.5, 1.5, 2.0, 1.0]))
    print(f"[metrics] self-dice(tumor)={d:.3f} (expect 1.000)  spacing={sp} (expect ~1.5,1.5,2.0)")
    assert abs(d - 1.0) < 1e-6 and abs(sp[2] - 2.0) < 1e-6

    print("\nAll v3 logic checks passed.")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="WaLM-Net v3 — corrected native eval + improvements")
    p.add_argument("--train", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--mode", default="full", choices=["full", "medium", "quick_test"])
    p.add_argument("--checkpoint", default="output/walmnet_medium/best.pth")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--ct-norm", choices=["minmax", "znorm"], default=None,
                   help="znorm = nnU-Net-style clip+z-score (requires fresh training)")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--no-postproc", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.selftest:
        selftest(); return

    cfg = get_config(args.mode)
    cfg["mode"] = args.mode
    if args.data_dir:
        cfg["kits23_dir"] = args.data_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.ct_norm:
        cfg["ct_norm"] = args.ct_norm
    if args.no_tta:
        cfg["use_tta"] = False
    if args.no_postproc:
        cfg["postprocess"] = False

    if args.evaluate:
        evaluate(cfg, args.checkpoint)
    elif args.train:
        train(cfg, resume=args.resume)
    else:
        p.error("choose one of --train / --evaluate / --selftest")


if __name__ == "__main__":
    main()
