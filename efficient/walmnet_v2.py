"""
WaLM-Net v2 — runs on Modal *or* locally, with resume + paper figures
=====================================================================
A simpler, publication-oriented launcher for `walmnet_kits23.py`. The model /
data / loss code in `walmnet_kits23.py` is reused unchanged. This file adds:

  * dead-simple commands (one flag per mode, download skipped by default)
  * the SAME commands work on Modal and on a local GPU box
  * exact training resume (optimizer + scheduler + scaler + EMA + early-stop)
  * automatic paper figures (PNG @300dpi + vector PDF) at the end of training
  * qualitative CT / GT / prediction overlay panels + predicted NIfTI volumes
  * a final FULL validation-set evaluation (not the 8-case training subset)

------------------------------------------------------------------ Modal -----
    modal run walmnet_v2.py --selftest                  # CPU sanity check
    modal run walmnet_v2.py --download                  # fetch KiTS23 into Volume
    modal run --detach walmnet_v2.py --quick            # 4-epoch smoke
    modal run --detach walmnet_v2.py --medium           # 120-epoch run
    modal run --detach walmnet_v2.py --full             # full run
    modal run --detach walmnet_v2.py --full --download  # download then train
    modal run --detach walmnet_v2.py --full --resume    # resume full
    modal run --detach walmnet_v2.py --medium --resume  # resume medium
    modal run walmnet_v2.py --full --evaluate           # full-val eval + figures

------------------------------------------------------------------ Local -----
    python walmnet_v2.py --selftest
    python walmnet_v2.py --download                      # clone + download KiTS23
    python walmnet_v2.py --quick
    python walmnet_v2.py --medium
    python walmnet_v2.py --full
    python walmnet_v2.py --full --resume                 # resume full
    python walmnet_v2.py --medium --resume               # resume medium
    python walmnet_v2.py --full --evaluate               # full-val eval + figures

Local paths default to ./kits23/dataset (data) and ./output/<mode> (results);
override with --data-dir and --output-dir.

Download Modal results back to this machine:
    modal volume get walmnet-kits23-data output/walmnet_full ./walmnet_full_results
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import modal
    HAS_MODAL = True
except ImportError:  # local box without the modal package
    modal = None
    HAS_MODAL = False


# ---------------------------------------------------------------------------
# Infrastructure constants (Modal Volume layout)
# ---------------------------------------------------------------------------
APP_NAME = "walmnet-kits23-v2"
VOLUME_NAME = "walmnet-kits23-data"
VOLUME_MOUNT = Path("/data")
KITS23_REPO_DIR = VOLUME_MOUNT / "kits23"
KITS23_DATASET_DIR = KITS23_REPO_DIR / "dataset"
OUTPUT_ROOT = VOLUME_MOUNT / "output"

# Modal GPU for training. Options: "A100", "A100-80GB", "L40S", "L4", "T4", "H100".
# A100 is best for this bandwidth-bound 3D conv workload; L40S is cheaper/faster
# to schedule but slower per step. Edit and re-run to switch.
TRAIN_GPU = "A100"
MAX_TIMEOUT = 24 * 60 * 60

# Output sub-directory per mode (kept identical to v1 so existing runs resume).
MODE_SUBDIR = {
    "full": "walmnet_full",
    "medium": "walmnet_medium",
    "quick_test": "walmnet_quick",
}

# ===========================================================================
# >>> EDIT HERE <<<  Central training knobs per mode.
# ---------------------------------------------------------------------------
# These override the architecture defaults in walmnet_kits23.get_config().
# Precedence (later wins):  get_config() defaults  <  TUNE[mode]  <  CLI flags
# (--patch / --batch / --sw-batch / --epochs on the local CLI).
# Only the commonly-tuned values live here; anything not listed keeps the
# get_config() default (warmup_epochs, ema_decay, loss weights, spacing, ...).
# patch_size is (D, H, W). Lower it / lower batch for a smaller GPU.
# ===========================================================================
TUNE = {
    "full": {
        "batch_size": 2,
        "patch_size": (128, 128, 128),
        "num_epochs": 300,
        "iters_per_epoch": 250,
        "lr": 2e-4,
        "val_cases": 12,            # cases used for the per-epoch val score
        "num_workers": 4,
        "cache_rate": 0.1,          # fraction of dataset cached in RAM
        "sw_batch_size": 2,         # sliding-window batch at validation
    },
    "medium": {
        "batch_size": 2,
        "patch_size": (128, 128, 128),
        "num_epochs": 80,
        "iters_per_epoch": 150,
        "lr": 2e-4,
        "val_cases": 20,
        "num_workers": 4,
        "cache_rate": 0.1,
        "sw_batch_size": 2,
    },
    "quick_test": {
        "batch_size": 2,
        "patch_size": (96, 96, 96),
        "num_epochs": 4,
        "iters_per_epoch": 8,
        "lr": 2e-4,
        "val_cases": 3,
        "num_workers": 4,
        "cache_rate": 0.0,
        "sw_batch_size": 2,
    },
}


# ===========================================================================
# Shared helpers (no torch / no modal) — usable locally and remotely
# ===========================================================================
def _import_W():
    """Import walmnet_kits23 from /root (Modal) or this file's dir (local)."""
    here = str(Path(__file__).resolve().parent)
    for p in ("/root", here):
        if p not in sys.path:
            sys.path.insert(0, p)
    import walmnet_kits23
    return walmnet_kits23


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _dataset_stats(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return {"dataset_dir": str(dataset_dir), "cases": 0, "images": 0, "segmentations": 0}
    cases = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("case_"))
    images = sum((c / "imaging.nii.gz").exists() for c in cases)
    segs = sum((c / "segmentation.nii.gz").exists() for c in cases)
    return {"dataset_dir": str(dataset_dir), "cases": len(cases),
            "images": images, "segmentations": segs}


def _make_config(mode: str, dataset_dir: Path, output_dir: Path, local: bool = False):
    """Build a mode config pointed at the given data / output paths."""
    W = _import_W()
    cfg = W.get_config(mode)
    cfg.update(TUNE.get(mode, {}))          # central per-mode overrides
    cfg["mode"] = mode
    cfg["kits23_dir"] = str(dataset_dir)
    cfg["output_dir"] = str(output_dir)
    cfg["_local"] = local
    return cfg


def _apply_overrides(cfg: dict, patch: str = "", batch: int = 0,
                     sw_batch: int = 0, epochs: int = 0) -> dict:
    """Optional config overrides (e.g. to fit a small GPU)."""
    if patch:
        cfg["patch_size"] = tuple(int(v.strip()) for v in patch.split(","))
    if batch > 0:
        cfg["batch_size"] = batch
    if sw_batch > 0:
        cfg["sw_batch_size"] = sw_batch
    if epochs > 0:
        cfg["num_epochs"] = epochs
    return cfg


def _maybe_commit(config: dict | None = None) -> None:
    """Commit the Modal Volume if we are running inside Modal; no-op locally."""
    if config is not None and config.get("_local"):
        return
    vol = globals().get("data_volume")
    if vol is None:
        return
    try:
        vol.commit()
    except Exception:  # noqa: BLE001
        pass


def _download_kits23(dataset_dir: Path, force: bool = False) -> dict[str, Any]:
    """Clone the official KiTS23 repo (next to dataset_dir) and download imaging."""
    dataset_dir = Path(dataset_dir)
    repo = dataset_dir.parent
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        if repo.exists() and any(repo.iterdir()):
            raise RuntimeError(f"{repo} exists but is not a git checkout; remove it first.")
        _run(["git", "clone", "--depth", "1", "https://github.com/neheller/kits23", str(repo)])
    else:
        _run(["git", "pull", "--ff-only"], cwd=repo)

    if force and dataset_dir.exists():
        for p in dataset_dir.glob("case_*/imaging.nii.gz"):
            p.unlink()

    before = _dataset_stats(dataset_dir)
    print("Before: " + json.dumps(before), flush=True)
    expected = max(before["cases"], before["segmentations"])
    if expected == 0 or before["images"] < expected:
        _run([sys.executable, "-m", "kits23.download"], cwd=repo)
    else:
        print("Imaging already complete; skipping download.", flush=True)
    after = _dataset_stats(dataset_dir)
    print("After: " + json.dumps(after), flush=True)
    return after


# ===========================================================================
# Paper-quality plotting (matplotlib only, no torch)
# ===========================================================================
def make_plots(out_dir: Path) -> list[str]:
    """Generate publication figures from history.json. Returns saved paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    hist_path = out_dir / "history.json"
    if not hist_path.exists():
        print(f"[plot] no history.json at {hist_path}; skipping", flush=True)
        return []
    hist = json.loads(hist_path.read_text())["epochs"]
    if not hist:
        return []

    CLASSES = ["Kidney", "Tumor", "Cyst"]
    COLORS = ["#2563eb", "#dc2626", "#16a34a"]

    ep = [e["epoch"] for e in hist]
    loss = [e["train_loss"] for e in hist]
    lr = [e.get("lr") for e in hist]

    vep = [e["epoch"] for e in hist if e.get("score") is not None]
    have_val = len(vep) > 0
    if have_val:
        dice = {c: [e["metrics"]["dice"][i] for e in hist if e.get("score") is not None]
                for i, c in enumerate(CLASSES)}
        hd95 = {c: [e["metrics"]["hd95"][i] for e in hist if e.get("score") is not None]
                for i, c in enumerate(CLASSES)}
        mean_fg = [e["metrics"]["mean_fg_dice"] for e in hist if e.get("score") is not None]
        rare = [e["metrics"]["rare_mean"] for e in hist if e.get("score") is not None]
        best_i = max(range(len(rare)), key=lambda i: rare[i])

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "figure.dpi": 120,
    })

    saved: list[str] = []

    def _save(fig, name):
        for ext in ("png", "pdf"):
            p = out_dir / f"{name}.{ext}"
            fig.savefig(p, bbox_inches="tight")
            saved.append(str(p))
        plt.close(fig)

    # --- Fig 1: training loss (+ LR twin axis) -----------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, loss, color="#7c3aed", lw=2, label="Train loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
    ax.set_title("WaLM-Net — Training Loss")
    lr_pairs = [(e, v) for e, v in zip(ep, lr) if v is not None]
    if lr_pairs:
        ax2 = ax.twinx()
        ax2.plot([e for e, _ in lr_pairs], [v for _, v in lr_pairs],
                 color="#9ca3af", lw=1.2, ls=":", label="LR")
        ax2.set_ylabel("Learning rate"); ax2.grid(False)
    _save(fig, "fig_loss")

    if have_val:
        # --- Fig 2: per-class validation Dice ------------------------------
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for c, col in zip(CLASSES, COLORS):
            ax.plot(vep, dice[c], marker="o", ms=4, lw=2, color=col, label=c)
        ax.plot(vep, mean_fg, marker="s", ms=4, lw=2, ls="--", color="#111827", label="Mean FG")
        ax.set(xlabel="Epoch", ylabel="Dice", ylim=(0, 1),
               title="WaLM-Net — Validation Dice per Class")
        ax.legend(loc="lower right", frameon=False)
        _save(fig, "fig_dice")

        # --- Fig 3: aggregate scores ---------------------------------------
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(vep, mean_fg, marker="s", ms=4, lw=2, color="#0891b2", label="Mean FG Dice")
        ax.plot(vep, rare, marker="^", ms=4, lw=2, color="#ea580c", label="Rare (Tumor+Cyst)")
        ax.annotate(f"best={rare[best_i]:.3f} @E{vep[best_i]}",
                    (vep[best_i], rare[best_i]), textcoords="offset points",
                    xytext=(-12, 16), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#ea580c"))
        ax.set(xlabel="Epoch", ylabel="Dice", ylim=(0, 1),
               title="WaLM-Net — Aggregate Dice")
        ax.legend(loc="lower right", frameon=False)
        _save(fig, "fig_aggregate")

        # --- Fig 4: HD95 ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for c, col in zip(CLASSES, COLORS):
            ax.plot(vep, hd95[c], marker="o", ms=4, lw=2, color=col, label=c)
        ax.set(xlabel="Epoch", ylabel="HD95 (mm)",
               title="WaLM-Net — HD95 per Class (lower is better)")
        ax.legend(frameon=False)
        _save(fig, "fig_hd95")

        # --- Fig 5: combined 2x2 panel -------------------------------------
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        axs[0, 0].plot(ep, loss, color="#7c3aed", lw=2)
        axs[0, 0].set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
        for c, col in zip(CLASSES, COLORS):
            axs[0, 1].plot(vep, dice[c], marker="o", ms=3, color=col, label=c)
        axs[0, 1].plot(vep, mean_fg, ls="--", color="#111827", label="Mean FG")
        axs[0, 1].set(title="Validation Dice", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
        axs[0, 1].legend(fontsize=8, frameon=False)
        axs[1, 0].plot(vep, mean_fg, marker="s", ms=3, color="#0891b2", label="Mean FG")
        axs[1, 0].plot(vep, rare, marker="^", ms=3, color="#ea580c", label="Rare")
        axs[1, 0].set(title="Aggregate Dice", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
        axs[1, 0].legend(fontsize=8, frameon=False)
        for c, col in zip(CLASSES, COLORS):
            axs[1, 1].plot(vep, hd95[c], marker="o", ms=3, color=col, label=c)
        axs[1, 1].set(title="HD95 (mm)", xlabel="Epoch", ylabel="HD95")
        axs[1, 1].legend(fontsize=8, frameon=False)
        fig.suptitle("WaLM-Net KiTS23 — Training Summary", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        _save(fig, "fig_summary")

    print(f"[plot] wrote {len(saved)} files to {out_dir}", flush=True)
    return saved


# ===========================================================================
# Qualitative overlays + predicted NIfTI (needs torch + monai)
# ===========================================================================
def make_qualitative(model, val_files, config, device, out_dir: Path, n_cases: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import nibabel as nib
    import torch
    from matplotlib.colors import ListedColormap
    from monai.data import Dataset, DataLoader
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, EnsureTyped,
    )

    out_dir = Path(out_dir)
    qdir = out_dir / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    a_min, a_max = config["ct_window"]
    tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=config["spacing"], mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    subset = val_files[:n_cases]
    loader = DataLoader(Dataset(subset, tf), batch_size=1, num_workers=2)

    overlay_cmap = ListedColormap([(0, 0, 0, 0), (0.15, 0.4, 1.0, 0.55),
                                   (1.0, 0.15, 0.15, 0.6), (0.1, 0.8, 0.2, 0.6)])
    model.eval()
    saved = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            img = batch["image"].to(device)
            lbl = batch["label"][0, 0].cpu().numpy().astype(np.int16)
            logits = sliding_window_inference(
                img, config["patch_size"], config["sw_batch_size"], model,
                overlap=config["sw_overlap"])
            pred = torch.argmax(logits[0], dim=0).cpu().numpy().astype(np.int16)
            ct = img[0, 0].cpu().numpy()

            fg = (lbl > 0)
            ref = fg if fg.any() else (pred > 0)
            z = int(np.argmax(ref.sum(axis=(0, 1)))) if ref.any() else ct.shape[2] // 2

            ct_s, gt_s, pr_s = ct[:, :, z].T, lbl[:, :, z].T, pred[:, :, z].T
            fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
            for a in axs:
                a.imshow(ct_s, cmap="gray"); a.axis("off")
            axs[0].set_title("CT")
            axs[1].imshow(gt_s, cmap=overlay_cmap, vmin=0, vmax=3, interpolation="nearest")
            axs[1].set_title("Ground truth")
            axs[2].imshow(pr_s, cmap=overlay_cmap, vmin=0, vmax=3, interpolation="nearest")
            axs[2].set_title("WaLM-Net prediction")
            case = Path(subset[idx]["image"]).parent.name
            fig.suptitle(f"{case}  (axial z={z})", fontsize=12)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                p = qdir / f"{case}_overlay.{ext}"
                fig.savefig(p, bbox_inches="tight", dpi=300)
                saved.append(str(p))
            plt.close(fig)

            nib.save(nib.Nifti1Image(pred.astype(np.uint8), np.eye(4)),
                     str(qdir / f"{case}_pred.nii.gz"))
    print(f"[qual] wrote {len(saved)} overlays + NIfTI to {qdir}", flush=True)
    return saved


# ===========================================================================
# Resume-capable training (device-agnostic: CPU / local GPU / Modal GPU)
# ===========================================================================
def train_resumable(config: dict, resume: bool):
    import logging
    import time

    import numpy as np
    import torch
    import torch.nn as nn

    W = _import_W()

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = True

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(out_dir / "training.log"),
                                  logging.StreamHandler(sys.stdout)], force=True)
    logger = logging.getLogger("walmnet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, val_files = W.get_dataloaders(config)
    model = W.build_model(config).to(device)
    total, trainable = W.count_parameters(model)
    mixer_kind = "mamba" if W._HAS_MAMBA and config.get("mixer", "auto") != "lite" else "lite"

    loss_fn = W.build_loss(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = W.cosine_warmup_scheduler(optimizer, config["num_epochs"], config["warmup_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    ema = W.ModelEMA(model, config["ema_decay"]) if config.get("ema_decay", 0) > 0 else None
    es = W.EarlyStopping(config["patience"], out_dir / "best.pth")

    history = {"epochs": []}
    start_epoch = 1

    # ---- resume ---------------------------------------------------------
    ckpt_full = out_dir / "ckpt.pth"
    ckpt_last = out_dir / "last.pth"
    if resume and ckpt_full.exists():
        st = torch.load(ckpt_full, map_location=device)
        model.load_state_dict(st["model_state_dict"])
        optimizer.load_state_dict(st["optimizer_state_dict"])
        scheduler.load_state_dict(st["scheduler_state_dict"])
        scaler.load_state_dict(st["scaler_state_dict"])
        if ema is not None and st.get("ema_state_dict"):
            ema.module.load_state_dict(st["ema_state_dict"])
        es.best = st.get("best"); es.counter = st.get("es_counter", 0)
        start_epoch = int(st["epoch"]) + 1
        if (out_dir / "history.json").exists():
            history = json.loads((out_dir / "history.json").read_text())
        logger.info(f"[resume] full state from {ckpt_full} -> start epoch {start_epoch}")
    elif resume and ckpt_last.exists():
        # warm resume from a v1 checkpoint (weights + ema + epoch only)
        st = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(st["model_state_dict"])
        if ema is not None and st.get("ema_state_dict"):
            ema.module.load_state_dict(st["ema_state_dict"])
        start_epoch = int(st.get("epoch", 0)) + 1
        for _ in range(start_epoch - 1):
            scheduler.step()
        if (out_dir / "history.json").exists():
            history = json.loads((out_dir / "history.json").read_text())
            for e in history["epochs"]:
                if e.get("score") is not None and (es.best is None or e["score"] > es.best):
                    es.best = e["score"]
        logger.info(f"[resume] WARM start from v1 {ckpt_last} (weights+EMA only) "
                    f"-> start epoch {start_epoch}; optimizer reinitialised")
    elif resume:
        logger.info("[resume] requested but no checkpoint found; starting fresh")

    logger.info("=" * 72)
    logger.info("WaLM-Net v2 training")
    logger.info(f"Device: {device} | Params: {total/1e6:.2f}M | Mixer: {mixer_kind}")
    logger.info(f"Patch: {config['patch_size']} | Epochs: {start_epoch}->{config['num_epochs']}")
    logger.info(f"Train cases: {len(train_loader.dataset)} | Val cases: {len(val_files)}")
    logger.info(f"Output: {out_dir}")
    logger.info("=" * 72)

    data_iter = W._cycle(train_loader)
    accum = config["grad_accum_steps"]

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        model.train()
        t0, running = time.time(), 0.0
        optimizer.zero_grad(set_to_none=True)
        for it in range(config["iters_per_epoch"]):
            batch = next(data_iter)
            img = batch["image"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=config["amp"] and device.type == "cuda"):
                loss = loss_fn(model(img), lbl) / accum
            scaler.scale(loss).backward()
            running += loss.item() * accum
            if (it + 1) % accum == 0:
                if config.get("grad_clip", 0) > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

        scheduler.step()
        train_loss = running / config["iters_per_epoch"]
        elapsed = (time.time() - t0) / 60

        do_val = (epoch % config["val_every"] == 0) or (epoch == config["num_epochs"])
        score, metrics = None, None
        if do_val:
            eval_model = ema.module if ema is not None else model
            score, metrics = W.validate(eval_model, val_files, config, device,
                                        n_cases=config["val_cases"], logger=logger)
            improved = es.step(score, eval_model, epoch)
            tag = " *BEST*" if improved else f" (no-improve {es.counter}/{es.patience})"
            logger.info(f"  {config['selection']} score={score:.4f}{tag}")

        cur_lr = scheduler.get_last_lr()[0]
        logger.info(f"[E{epoch:03d}/{config['num_epochs']}] train_loss={train_loss:.4f} "
                    f"lr={cur_lr:.2e} time={elapsed:.1f}min")
        history["epochs"].append({"epoch": epoch, "train_loss": train_loss, "lr": cur_lr,
                                  "score": score, "metrics": metrics, "time_min": elapsed})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # full-state checkpoint for exact resume
        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.module.state_dict() if ema else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch, "best": es.best, "es_counter": es.counter,
            "config": {k: str(v) for k, v in config.items()},
        }, out_dir / "ckpt.pth")
        # lightweight weights-only checkpoint (v1-compatible)
        torch.save({"model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.module.state_dict() if ema else None,
                    "epoch": epoch}, out_dir / "last.pth")

        _maybe_commit(config)  # persist to Modal Volume mid-run; no-op locally

        if es.early_stop and epoch >= config["min_epochs_before_stop"]:
            logger.info(f"Early stopping at epoch {epoch} (best={es.best:.4f})")
            break

    # ---- final FULL validation-set evaluation (unbiased) ----------------
    logger.info("Running final evaluation on the FULL validation set ...")
    final_model = ema.module if ema is not None else model
    if (out_dir / "best.pth").exists():
        best = torch.load(out_dir / "best.pth", map_location=device)
        final_model.load_state_dict(best["model_state_dict"])
        logger.info(f"Loaded best checkpoint (epoch {best.get('epoch')}, score {best.get('score')})")
    full_score, full_metrics = W.validate(final_model, val_files, config, device,
                                          n_cases=None, logger=logger)
    summary = {
        "version": "walmnet_v2", "mode": config.get("mode"),
        "params_millions": round(total / 1e6, 3), "mixer": mixer_kind,
        "best_val_subset_score": es.best,
        "full_val_score": full_score, "full_val_metrics": full_metrics,
        "n_val_cases_full": len(val_files),
        "total_hours": round(sum(e["time_min"] for e in history["epochs"]) / 60, 2),
        "config": {k: str(v) for k, v in config.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"FULL val: {full_metrics.get('mean_fg_dice'):.4f} meanFG | "
                f"rare={full_metrics.get('rare_mean'):.4f} over {len(val_files)} cases")

    # ---- figures + qualitative overlays ---------------------------------
    make_plots(out_dir)
    try:
        make_qualitative(final_model, val_files, config, device, out_dir, n_cases=4)
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[qual] skipped due to: {exc}")

    _maybe_commit(config)
    return summary


def evaluate_full(config: dict, checkpoint: str = ""):
    """Full validation-set eval + figures + overlays from a checkpoint."""
    import torch
    W = _import_W()

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = checkpoint or str(out_dir / "best.pth")
    model = W.build_model(config).to(device)
    st = torch.load(ckpt_path, map_location=device)
    state = st.get("ema_state_dict") or st.get("model_state_dict") or st
    model.load_state_dict(state)
    print(f"Loaded {ckpt_path}", flush=True)

    _, _, val_files = W.get_dataloaders(config)
    score, metrics = W.validate(model, val_files, config, device, n_cases=None, logger=None)
    (out_dir / "evaluation.json").write_text(json.dumps(
        {"full_val_score": score, "full_val_metrics": metrics,
         "n_val_cases": len(val_files)}, indent=2))
    make_plots(out_dir)
    try:
        make_qualitative(model, val_files, config, device, out_dir, n_cases=4)
    except Exception as exc:  # noqa: BLE001
        print(f"[qual] skipped: {exc}", flush=True)
    print(f"FULL val score = {score:.4f} over {len(val_files)} cases", flush=True)
    return {"full_val_score": score, "metrics": metrics, "output_dir": str(out_dir)}


# ===========================================================================
# Modal app + remote functions (only defined when modal is installed)
# ===========================================================================
if HAS_MODAL:
    data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git")
        .uv_pip_install("torch", "monai", "nibabel", "numpy", "scipy", "tqdm", "matplotlib")
        .add_local_file(Path(__file__).with_name("walmnet_kits23.py"), "/root/walmnet_kits23.py")
    )

    app = modal.App(APP_NAME, image=image)

    @app.function(volumes={VOLUME_MOUNT: data_volume}, cpu=4, memory=16384, timeout=MAX_TIMEOUT)
    def do_download(force: bool = False) -> dict[str, Any]:
        res = _download_kits23(KITS23_DATASET_DIR, force=force)
        _maybe_commit()
        return res

    @app.function(timeout=30 * 60, cpu=2, memory=8192)
    def run_selftest() -> None:
        _import_W().selftest()

    @app.function(volumes={VOLUME_MOUNT: data_volume}, gpu=TRAIN_GPU,
                  cpu=8, memory=65536, timeout=MAX_TIMEOUT)
    def run_train(mode: str, resume: bool = False) -> dict[str, Any]:
        data_volume.reload()
        stats = _dataset_stats(KITS23_DATASET_DIR)
        if stats["images"] == 0 or stats["segmentations"] == 0:
            raise FileNotFoundError(
                f"KiTS23 not ready at {KITS23_DATASET_DIR}. Run with --download first.")
        cfg = _make_config(mode, KITS23_DATASET_DIR, OUTPUT_ROOT / MODE_SUBDIR[mode], local=False)
        return train_resumable(cfg, resume)

    @app.function(volumes={VOLUME_MOUNT: data_volume}, gpu=TRAIN_GPU,
                  cpu=8, memory=65536, timeout=MAX_TIMEOUT)
    def run_eval(mode: str, checkpoint: str = "") -> dict[str, Any]:
        data_volume.reload()
        cfg = _make_config(mode, KITS23_DATASET_DIR, OUTPUT_ROOT / MODE_SUBDIR[mode], local=False)
        res = evaluate_full(cfg, checkpoint)
        _maybe_commit()
        return res

    @app.local_entrypoint()
    def main(
        full: bool = False, medium: bool = False, quick: bool = False,
        selftest: bool = False, evaluate: bool = False,
        download: bool = False, resume: bool = False, checkpoint: str = "",
    ):
        if selftest:
            run_selftest.remote()
            return
        chosen = [m for m, f in (("full", full), ("medium", medium), ("quick_test", quick)) if f]
        if download and not chosen and not evaluate:
            print(do_download.remote())
            return
        if not chosen:
            print("Specify a mode: --full | --medium | --quick  "
                  "(+ --download / --resume / --evaluate; or --download / --selftest alone)")
            return
        if len(chosen) > 1:
            print(f"Pick ONE mode, got: {chosen}")
            return
        mode = chosen[0]
        if download:
            print(do_download.remote())
        if evaluate:
            print(run_eval.remote(mode=mode, checkpoint=checkpoint))
            return
        print(run_train.remote(mode=mode, resume=resume))


# ===========================================================================
# Local CLI (python walmnet_v2.py ...) — same behaviour without Modal
# ===========================================================================
def _local_main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="WaLM-Net v2 - local runner (resume + paper figures)")
    p.add_argument("--full", action="store_true")
    p.add_argument("--medium", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--data-dir", default="./kits23/dataset",
                   help="local KiTS23 dataset dir (default ./kits23/dataset)")
    p.add_argument("--output-dir", default="",
                   help="override output dir (default ./output/<mode>)")
    p.add_argument("--checkpoint", default="")
    # fit-a-small-GPU overrides
    p.add_argument("--patch", default="", help="e.g. 96,96,96 (lower for small VRAM)")
    p.add_argument("--batch", type=int, default=0, help="training batch size")
    p.add_argument("--sw-batch", type=int, default=0, help="sliding-window batch size")
    p.add_argument("--epochs", type=int, default=0, help="override number of epochs")
    args = p.parse_args(argv)

    if args.selftest:
        _import_W().selftest()
        return

    data_dir = Path(args.data_dir)
    chosen = [m for m, f in (("full", args.full), ("medium", args.medium),
                             ("quick_test", args.quick)) if f]

    if args.download and not chosen and not args.evaluate:
        print(_download_kits23(data_dir, force=args.force_download))
        return
    if not chosen:
        p.print_help()
        return
    if len(chosen) > 1:
        print(f"Pick ONE mode, got: {chosen}")
        return
    mode = chosen[0]

    if args.download:
        _download_kits23(data_dir, force=args.force_download)

    stats = _dataset_stats(data_dir)
    if stats["images"] == 0 or stats["segmentations"] == 0:
        raise FileNotFoundError(
            f"No KiTS23 data at {data_dir}. Use --download or pass --data-dir.")

    out_dir = Path(args.output_dir) if args.output_dir else Path("./output") / MODE_SUBDIR[mode]
    cfg = _make_config(mode, data_dir, out_dir, local=True)
    cfg = _apply_overrides(cfg, args.patch, args.batch, args.sw_batch, args.epochs)

    if args.evaluate:
        print(evaluate_full(cfg, args.checkpoint))
        return
    print(train_resumable(cfg, args.resume))


if __name__ == "__main__":
    _local_main()
