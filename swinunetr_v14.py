"""
SwinUNETR V14 -- DINOv3 adapter + two-stage unfreezing + deep supervision
                 + tumor-in-kidney post-processing.

V14 builds on V13's adapter + CT-branch architecture and adds the key missing
pieces needed to push tumor Dice toward 70+:

  1. TWO-STAGE TRAINING: The DINOv3 encoder is frozen for the first
     `unfreeze_after` epochs (adapter warm-up), then the top-N ViT blocks are
     unfrozen with a lower LR for domain adaptation.
  2. DEEP SUPERVISION: Auxiliary segmentation heads on intermediate decoder
     outputs to improve gradient flow for rare classes (tumor, cyst).
  3. TUMOR-IN-KIDNEY POST-PROCESSING: Tumor and cyst voxels that are too far
     from the nearest kidney prediction are removed, suppressing FPs.
  4. LONGER TRAINING (200 epochs) and increased patience (20) to let the
     unfrozen encoder converge.

Recommended:
  python swinunetr_v14.py --quick --no-resume
  python swinunetr_v14.py --build-cache
  python swinunetr_v14.py --full
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import swinunetr_v12 as v12
import swinunetr_v13 as v13

base = v12.base
_BASE_VALIDATE_SLIDING_WINDOW = base.validate_sliding_window


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_config(mode: str) -> dict:
    cfg = v13.get_config(mode)
    cfg.update({
        "output_dir": "./output/swin_v14_twostage_tumor",
        "arch": "dinov3_v14",
        "experiment_name": "SwinUNETR V14 -- two-stage unfreeze + deep supervision",

        # Two-stage unfreezing schedule
        "dinov3_freeze": True,           # frozen for stage 1
        "unfreeze_after": 50,            # unfreeze top ViT blocks after this epoch
        "unfreeze_top_n_blocks": 6,      # how many transformer blocks to unfreeze
        "encoder_lr_factor": 0.1,        # LR multiplier for unfrozen encoder params

        # Deep supervision
        "deep_supervision": True,
        "ds_weights": (1.0, 0.4, 0.2),  # full, half, quarter resolution

        # Longer training with more patience for stage 2
        "num_epochs": 200,
        "patience": 20,
        "fast_val_every": 10,       # cheap patch-val loss for monitoring/early stop
        "fast_patience": 12,        # 12 fast checks = ~120 epochs by default
        "sw_val_every": 0,          # expensive full-volume Dice validation; off during training by default
        "sw_val_cases": 10,
        "sw_val_on_final": False,
        "min_epochs_before_stop": 80,

        # Tumor-in-kidney post-processing
        "tumor_kidney_max_dist": 15.0,   # max voxel distance from kidney for T/C
        "context_max_dist": 15.0,
        "context_filtered_classes": (2, 3),

        # Slightly tuned loss for the longer run
        "ce_weight": (0.1, 1.0, 18.0, 12.0),
        "fg_dice_weights": (1.0, 5.0, 3.0),
        "tversky_alpha": 0.12,
        "tversky_beta": 0.88,
        "focal_gamma": 3.0,
    })
    if mode == "quick_test":
        cfg.update({
            "output_dir": "./output/swin_v14_twostage_tumor_quick",
            "num_epochs": 4,
            "unfreeze_after": 2,
            "unfreeze_top_n_blocks": 2,
            "fast_val_every": 1,
            "sw_val_every": 0,
            "sw_val_on_final": False,
            "min_epochs_before_stop": 999999,
        })
    elif mode == "medium":
        cfg.update({
            "output_dir": "./output/swin_v14_twostage_tumor_medium",
            "num_epochs": 120,
            "unfreeze_after": 30,
            "sw_val_every": 0,
            "sw_val_cases": 8,
            "min_epochs_before_stop": 60,
        })
    return cfg


# ---------------------------------------------------------------------------
# Model: V14 with deep supervision
# ---------------------------------------------------------------------------

class DINOv3V14(nn.Module):
    """V13 adapter architecture with added deep supervision heads."""

    def __init__(self, config: dict):
        super().__init__()
        fd = int(config["dinov3_feature_dim"])
        full_ch, half_ch = tuple(config.get("ct_skip_channels", (32, 64)))
        self.deep_supervision = bool(config.get("deep_supervision", True))

        # Reuse V13's adapter encoder + CT branch
        self.encoder = v13.DINOv3AdapterSliceEncoder(
            repo_path=config["dinov3_repo"],
            model_name=config["dinov3_model"],
            weights_path=config.get("dinov3_weights"),
            feature_dim=fd,
            layers=tuple(config["dinov3_layers"]),
            hf_model=config.get("dinov3_hf_model", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
            hf_layers=tuple(config.get("dinov3_hf_layers", (-9, -5, -1))),
            source=config.get("dinov3_source", "hf"),
            patch_size=int(config.get("dinov3_patch_size", 16)),
            freeze=bool(config.get("dinov3_freeze", True)),
            slice_batch=int(config.get("dinov3_slice_batch", 8)),
            input_norm=config.get("dinov3_norm", "imagenet"),
            adapter_scale_init=float(config.get("adapter_scale_init", 0.10)),
            adapter_dropout=float(config.get("adapter_dropout", 0.05)),
        )
        self.ct_branch = v13.CTHighResBranch(full_channels=full_ch, half_channels=half_ch)

        # Decoder (same as V13)
        self.up1 = v12.AnisotropicUpBlock(fd, 192, skip_channels=fd)
        self.up2 = v12.AnisotropicUpBlock(192, 128, skip_channels=fd)
        self.up3 = v12.AnisotropicUpBlock(128, 64, skip_channels=half_ch)
        self.up4 = v12.AnisotropicUpBlock(64, 32, skip_channels=full_ch)

        # Main segmentation head
        nc = int(config["num_classes"])
        self.head = nn.Conv3d(32, nc, 1)

        # Deep supervision auxiliary heads
        self.head_half = nn.Conv3d(64, nc, 1)
        self.head_quarter = nn.Conv3d(128, nc, 1)

    def forward(self, x: torch.Tensor):
        target_size = x.shape[2:]
        dino = self.encoder(x)
        ct = self.ct_branch(x)

        y1 = self.up1(dino["bottleneck"], dino["skip_high"])
        y2 = self.up2(y1, dino["skip_low"])
        y3 = self.up3(y2, ct["half"])
        y4 = self.up4(y3, ct["full"])

        if y4.shape[2:] != target_size:
            y4 = F.interpolate(y4, size=target_size, mode="trilinear", align_corners=False)
        full_out = self.head(y4)

        if self.training and self.deep_supervision:
            return [
                full_out,
                self.head_half(y3),     # 1/2 resolution
                self.head_quarter(y2),  # 1/4 resolution
            ]
        return full_out


# ---------------------------------------------------------------------------
# Two-stage unfreezing logic
# ---------------------------------------------------------------------------

def unfreeze_top_vit_blocks(model: nn.Module, n_blocks: int, logger=None):
    """Unfreeze the top-N transformer blocks of the DINOv3 ViT encoder."""
    log = logger.info if logger else print
    if n_blocks <= 0:
        log("  Stage 2 requested 0 ViT blocks; encoder remains frozen.")
        return []
    encoder = model.encoder if hasattr(model, "encoder") else model

    # Navigate to the ViT backbone inside the DINOv3 encoder
    vit = encoder.vit if hasattr(encoder, "vit") else None
    if vit is None:
        log("  WARNING: Could not find ViT backbone for unfreezing.")
        return []

    # Find the transformer block list. HF DINOv3/ViT variants and the local
    # facebookresearch/dinov3 repo use slightly different paths.
    block_list = None
    block_path = None
    for attr in [
        "encoder.layer",
        "encoder.layers",
        "encoder.blocks",
        "layer",
        "layers",
        "blocks",
        "backbone.encoder.layer",
        "backbone.encoder.layers",
        "backbone.blocks",
    ]:
        obj = vit
        found = True
        for part in attr.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                found = False
                break
        if found and hasattr(obj, "__len__") and len(obj) > 0:
            block_list = obj
            block_path = attr
            break

    if block_list is None:
        candidates = []
        for name, module in vit.named_modules():
            if not isinstance(module, (nn.ModuleList, nn.Sequential)):
                continue
            if len(module) < max(2, min(4, n_blocks)):
                continue
            if not all(isinstance(child, nn.Module) for child in module):
                continue
            param_count = sum(p.numel() for child in module for p in child.parameters())
            if param_count <= 0:
                continue
            lname = name.lower()
            score = len(module)
            if any(token in lname for token in ("encoder", "block", "layer")):
                score += 1000
            candidates.append((score, name, module))
        if candidates:
            _, block_path, block_list = max(candidates, key=lambda x: x[0])

    if block_list is None:
        available = [name for name, module in vit.named_modules() if isinstance(module, (nn.ModuleList, nn.Sequential))]
        preview = ", ".join(available[:12]) if available else "none"
        log(f"  WARNING: Could not locate transformer blocks for unfreezing. ModuleList candidates: {preview}")
        return []

    total_blocks = len(block_list)
    n_to_unfreeze = min(n_blocks, total_blocks)
    unfrozen_params = []

    # Unfreeze the last N blocks
    for i in range(total_blocks - n_to_unfreeze, total_blocks):
        for param in block_list[i].parameters():
            param.requires_grad = True
            unfrozen_params.append(param)

    # Also unfreeze the layernorm at the end if present
    for name in ["layernorm", "norm", "encoder.ln"]:
        obj = vit
        found = True
        for part in name.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                found = False
                break
        if found and isinstance(obj, nn.Module):
            for param in obj.parameters():
                param.requires_grad = True
                unfrozen_params.append(param)

    n_tensors = len(unfrozen_params)
    n_values = sum(p.numel() for p in unfrozen_params)
    log(f"  Unfroze top {n_to_unfreeze}/{total_blocks} ViT blocks from {block_path} "
        f"({n_tensors} tensors, {n_values:,} parameters)")
    return unfrozen_params


def rebuild_optimizer_for_stage2(model, config, unfrozen_encoder_params, logger=None):
    """Create a new optimizer with differential LR for unfrozen encoder params."""
    log = logger.info if logger else print

    encoder_lr = config["lr"] * config.get("encoder_lr_factor", 0.1)
    unfrozen_set = set(id(p) for p in unfrozen_encoder_params)

    # Group 1: all existing trainable params (adapters, decoder, CT branch, heads)
    base_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in unfrozen_set]
    # Group 2: newly unfrozen encoder params (lower LR)
    encoder_params = [p for p in unfrozen_encoder_params if p.requires_grad]

    param_groups = [
        {"params": base_params, "lr": config["lr"]},
        {"params": encoder_params, "lr": encoder_lr, "name": "encoder_unfrozen"},
    ]

    log(f"  Stage 2 optimizer: {len(base_params)} base params (lr={config['lr']:.1e}), "
        f"{len(encoder_params)} encoder params (lr={encoder_lr:.1e})")

    return torch.optim.AdamW(param_groups, lr=config["lr"],
                             weight_decay=config["weight_decay"])


# ---------------------------------------------------------------------------
# Enhanced post-processing: tumor-in-kidney constraint
# ---------------------------------------------------------------------------

def postprocess_tumor_in_kidney(prediction, config):
    """Remove tumor/cyst voxels that are too far from any kidney voxel."""
    import numpy as np
    from scipy import ndimage

    output = prediction.copy()
    kidney_mask = output == 1
    if not kidney_mask.any():
        return output

    max_dist = config.get("tumor_kidney_max_dist", 15.0)
    if max_dist <= 0:
        return output

    target_mask = (output == 2) | (output == 3)
    if not target_mask.any():
        return output

    # Full-volume EDT + per-component boolean masks are slow on noisy early
    # predictions. Crop to kidney/T/C foreground plus margin, then decide in a
    # vectorized way which connected components intersect the allowed distance.
    fg = kidney_mask | target_mask
    coords = np.argwhere(fg)
    margin = int(np.ceil(float(max_dist))) + 2
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, np.array(output.shape))
    crop = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))

    kidney_crop = kidney_mask[crop]
    output_crop = output[crop].copy()
    dist_to_kidney = ndimage.distance_transform_edt(~kidney_crop)
    allowed_zone = dist_to_kidney <= float(max_dist)

    for class_id in (2, 3):  # tumor, cyst
        mask = output_crop == class_id
        if not mask.any():
            continue
        labeled, n_comp = ndimage.label(mask)
        if n_comp == 0:
            continue
        keep_labels = np.unique(labeled[mask & allowed_zone])
        keep = np.zeros(n_comp + 1, dtype=bool)
        keep[keep_labels] = True
        keep[0] = True
        output_crop[mask & ~keep[labeled]] = 0

    output[crop] = output_crop

    return output


# ---------------------------------------------------------------------------
# Build model and loss - override base functions
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    if config.get("arch") == "dinov3_v14":
        return DINOv3V14(config)
    return v13.build_model(config)


def build_loss(config: dict, device) -> nn.Module:
    """Build loss with deep supervision support for V14."""
    method_cfg = base._method_config(config)
    base_loss = base.RareClassSegLoss(method_cfg).to(device)
    if config.get("deep_supervision", True):
        return base.DeepSupervisionLoss(base_loss, tuple(config["ds_weights"]))
    return base_loss


def validate_sliding_window(model, val_dicts, config, device, n_cases=None, logger=None):
    """V12 validation with selection override + V14 post-processing."""
    # Temporarily patch postprocess_prediction to include tumor-in-kidney
    original_postprocess = base.postprocess_prediction

    def _enhanced_postprocess(prediction, cfg):
        output = original_postprocess(prediction, cfg)
        if cfg.get("tumor_kidney_max_dist", 0) > 0:
            output = postprocess_tumor_in_kidney(output, cfg)
        return output

    base.postprocess_prediction = _enhanced_postprocess
    try:
        score, metrics = _BASE_VALIDATE_SLIDING_WINDOW(
            model, val_dicts, config, device, n_cases=n_cases, logger=logger
        )
    finally:
        base.postprocess_prediction = original_postprocess

    # Apply tumor/kidney_tumor selection override (from V12)
    if config.get("selection") in {"tumor", "kidney_tumor"}:
        metrics["selection_metric"] = v12._selection_from_metrics(metrics, config)
        metrics["selection"] = config["selection"]
        log = logger.info if logger else print
        log(f"  Selection override ({config['selection']}): {metrics['selection_metric']:.4f}")
        return metrics["selection_metric"], metrics
    return score, metrics


# ---------------------------------------------------------------------------
# Custom training loop with two-stage unfreezing
# ---------------------------------------------------------------------------

def train_v14(config: dict, resume_path=None, skip_resume=False):
    """V14 training loop: wraps V11's train() but intercepts for stage-2 unfreeze."""
    import copy
    import json
    import os
    import sys
    import time
    import numpy as np

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(output_dir / "training.log"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("v14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff_batch = config["batch_size"] * config.get("grad_accum_steps", 1)

    logger.info("=" * 72)
    logger.info("SwinUNETR V14 -- two-stage unfreeze + deep supervision + TIK post-proc")
    logger.info(f"Started: {__import__('datetime').datetime.now().isoformat()}")
    logger.info(f"Device: {device}" +
                (f" | GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    logger.info(f"Output: {output_dir}")
    logger.info(f"Arch: {config['arch']} | Selection: {config['selection']} | Eff.Batch: {eff_batch}")
    logger.info(f"Stage 1: frozen encoder for {config['unfreeze_after']} epochs")
    logger.info(f"Stage 2: unfreeze top {config['unfreeze_top_n_blocks']} ViT blocks "
                f"(lr_factor={config['encoder_lr_factor']})")
    logger.info(f"Deep supervision: {config['deep_supervision']} | "
                f"Tumor-kidney max_dist: {config.get('tumor_kidney_max_dist', 'off')}")
    sw_every = int(config.get("sw_val_every", 0))
    sw_text = f"SW every {sw_every} epochs" if sw_every > 0 else "SW off during training"
    logger.info(
        f"Validation: fast patch every {config.get('fast_val_every', 10)} epochs | {sw_text}"
    )
    logger.info("=" * 72)

    train_loader, val_loader, _, val_dicts, _ = base.get_dataloaders(config)
    model = build_model(config).to(device)

    loss_fn = build_loss(config, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = base.cosine_warmup_scheduler(optimizer, config["num_epochs"], config["warmup_epochs"])
    ema = base.ModelEMA(model, config["ema_decay"]) if config.get("ema_decay", 0.0) > 0 else None

    # Resume
    resume_state = None
    auto_resume = output_dir / "resume_state.pth"
    if resume_path:
        resume_state = base._load_resume(Path(resume_path), device)
    elif not skip_resume:
        resume_state = base._load_resume(auto_resume, device)

    start_epoch = 1
    sw_es = base.EarlyStopping(config["patience"], output_dir / "best_final.pth")
    fast_es = base.EarlyStopping(config.get("fast_patience", config["patience"]), output_dir / "best_fast.pth")
    stage2_activated = False

    if resume_state:
        model.load_state_dict(base._strip_state(resume_state["model_state_dict"]))
        if ema is not None and resume_state.get("ema_state_dict"):
            ema.module.load_state_dict(base._strip_state(resume_state["ema_state_dict"]))
        optimizer_restored = False
        try:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            optimizer_restored = True
        except ValueError as exc:
            logger.info(
                "Resume optimizer/scheduler state skipped because the parameter "
                f"groups changed ({exc}). Model weights and epoch were restored."
            )
        start_epoch = int(resume_state["epoch"]) + 1
        sw_es.best = resume_state.get("sw_best_metric", resume_state.get("best_metric"))
        sw_es.counter = int(resume_state.get("sw_es_counter", resume_state.get("es_counter", 0)))
        fast_es.best = resume_state.get("fast_best_metric")
        fast_es.counter = int(resume_state.get("fast_es_counter", 0))
        stage2_activated = resume_state.get("stage2_activated", False)
        logger.info(f"Resumed from epoch {resume_state['epoch']} | sw_best={base._fmt(sw_es.best)} "
                    f"| fast_best={base._fmt(fast_es.best)} "
                    f"| stage2={'active' if stage2_activated else 'pending'} "
                    f"| optimizer={'restored' if optimizer_restored else 'rebuilt'}")

    # If resuming past unfreeze_after, re-unfreeze
    if stage2_activated or start_epoch > config["unfreeze_after"]:
        unfrozen = unfreeze_top_vit_blocks(model, config["unfreeze_top_n_blocks"], logger)
        if unfrozen and not stage2_activated:
            optimizer = rebuild_optimizer_for_stage2(model, config, unfrozen, logger)
            scheduler = base.cosine_warmup_scheduler(
                optimizer, config["num_epochs"] - config["unfreeze_after"],
                warmup_epochs=3
            )
        stage2_activated = True

    for epoch in range(start_epoch, int(config["num_epochs"]) + 1):
        t0 = time.time()

        # --- Stage 2 transition ---
        if epoch == config["unfreeze_after"] + 1 and not stage2_activated:
            logger.info("=" * 40)
            logger.info(f"STAGE 2: Unfreezing top {config['unfreeze_top_n_blocks']} ViT blocks")
            logger.info("=" * 40)

            unfrozen = unfreeze_top_vit_blocks(model, config["unfreeze_top_n_blocks"], logger)
            if unfrozen:
                optimizer = rebuild_optimizer_for_stage2(model, config, unfrozen, logger)
                # Reset scheduler for remaining epochs with short warmup
                remaining = config["num_epochs"] - epoch + 1
                scheduler = base.cosine_warmup_scheduler(optimizer, remaining, warmup_epochs=3)
            stage2_activated = True

            # Also update EMA if present
            if ema is not None:
                ema = base.ModelEMA(model, config["ema_decay"])
                logger.info("  Rebuilt EMA for stage 2")

        train_loss = base.train_one_epoch(model, train_loader, optimizer, loss_fn,
                                          device, config, ema=ema)
        scheduler.step()
        elapsed = time.time() - t0

        fast_val_every = int(config.get("fast_val_every", config.get("val_every", 10)))
        sw_val_every = int(config.get("sw_val_every", 60))
        final_epoch = epoch == int(config["num_epochs"])
        do_sw_val = (sw_val_every > 0 and epoch % sw_val_every == 0) or (
            final_epoch and bool(config.get("sw_val_on_final", True))
        )
        do_fast_val = (
            fast_val_every > 0 and epoch % fast_val_every == 0
        ) or do_sw_val or final_epoch
        val_loss = None
        sw_metrics = None
        fast_improved = False
        sw_improved = False
        eval_model = ema.module if ema is not None else model

        if do_fast_val:
            val_loss = base.validate_patch_loss(eval_model, val_loader, loss_fn, device)
            if val_loss is not None:
                # EarlyStopping maximizes, so negate validation loss.
                fast_improved = fast_es.step(-float(val_loss), eval_model, epoch, config)
                if fast_improved:
                    logger.info(f"  New best fast val_loss = {val_loss:.4f}")
                elif fast_es.best is not None:
                    logger.info(
                        f"  Fast val no improvement {fast_es.counter}/{fast_es.patience} "
                        f"| best_loss={-fast_es.best:.4f}"
                    )

        if do_sw_val:
            score, sw_metrics = validate_sliding_window(
                eval_model, val_dicts, config, device,
                n_cases=config.get("sw_val_cases"), logger=logger,
            )
            sw_improved = sw_es.step(score, eval_model, epoch, config)
            if sw_improved:
                logger.info(f"  New best {config['selection']} metric = {score:.4f}")
            elif sw_es.best is not None:
                logger.info(f"  SW no improvement {sw_es.counter}/{sw_es.patience} | best={sw_es.best:.4f}")

        stage_label = "S2" if stage2_activated else "S1"
        val_text = f"val_loss={base._fmt(val_loss)}"
        if not do_fast_val:
            next_fast = epoch + (fast_val_every - (epoch % fast_val_every)) if fast_val_every > 0 else None
            next_fast = min(int(config["num_epochs"]), next_fast) if next_fast else None
            val_text = f"val_loss=N/A(next fast E{next_fast:03d})" if next_fast else "val_loss=N/A"
        elif sw_metrics is not None and sw_metrics.get("dice"):
            dice = sw_metrics["dice"]
            val_text += (
                f" K={dice[1]:.4f} T={dice[2]:.4f} C={dice[3]:.4f}"
                f" Rare={sw_metrics.get('rare_mean', 0.0):.4f}"
                f" MeanFG={sw_metrics.get('mean_fg_dice', 0.0):.4f}"
            )
        elif not do_sw_val:
            next_sw = epoch + (sw_val_every - (epoch % sw_val_every)) if sw_val_every > 0 else None
            if next_sw is not None:
                if bool(config.get("sw_val_on_final", True)):
                    next_sw = min(int(config["num_epochs"]), next_sw)
                val_text += f" SW=N/A(next E{next_sw:03d})"

        lr_str = scheduler.get_last_lr()[0]
        logger.info(
            f"[E{epoch:03d}/{config['num_epochs']}|{stage_label}] train={train_loss:.4f} "
            f"{val_text} lr={lr_str:.2e} time={elapsed / 60:.1f}min"
        )
        base.save_history(output_dir, epoch, train_loss, val_loss, sw_metrics, elapsed, sw_improved or fast_improved)

        # Save resume state
        resume_tmp = output_dir / "resume_state.tmp"
        torch.save({
            "model_state_dict": base._strip_state(model.state_dict()),
            "ema_state_dict": base._strip_state(ema.module.state_dict()) if ema is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": sw_es.best,
            "sw_best_metric": sw_es.best,
            "sw_es_counter": sw_es.counter,
            "fast_best_metric": fast_es.best,
            "fast_es_counter": fast_es.counter,
            "stage2_activated": stage2_activated,
            "config": {k: str(v) for k, v in config.items()},
        }, resume_tmp)
        os.replace(resume_tmp, output_dir / "resume_state.pth")

        if fast_es.early_stop and epoch >= int(config.get("min_epochs_before_stop", 0)):
            logger.info(f"Early stopping at epoch {epoch} by fast validation loss")
            break

    history = json.loads((output_dir / "history.json").read_text()) if (output_dir / "history.json").exists() else {"epochs": []}
    total_min = sum(e.get("epoch_time_min", 0) for e in history["epochs"])
    summary = {
        "version": "swinunetr_v14",
        "method": "two-stage unfreeze + deep supervision + TIK postproc",
        "arch": config["arch"],
        "selection": config["selection"],
        "best_metric": sw_es.best,
        "best_sw_metric": sw_es.best,
        "best_fast_val_loss": -fast_es.best if fast_es.best is not None else None,
        "total_hours": round(total_min / 60, 2),
        "config": {k: str(v) for k, v in config.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Training complete. Best SW {config['selection']} metric: {base._fmt(sw_es.best)}")
    logger.info(f"Summary -> {output_dir / 'summary.json'}")


# ---------------------------------------------------------------------------
# Patch base module
# ---------------------------------------------------------------------------

def _patch_base_for_v14():
    base.build_model = build_model
    base.build_loss = build_loss
    base.validate_sliding_window = validate_sliding_window


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_mode(args, parser) -> str:
    shortcut_modes = [
        ("full", args.full),
        ("medium", args.medium),
        ("quick_test", args.quick),
        ("build_cache", args.build_cache),
        ("audit", args.audit),
        ("evaluate", args.evaluate),
    ]
    selected = [mode for mode, enabled in shortcut_modes if enabled]
    if len(selected) > 1:
        parser.error("Use only one mode shortcut, e.g. --full or --build-cache")
    return selected[0] if selected else (args.mode or "full")


def main():
    parser = argparse.ArgumentParser(
        description="SwinUNETR V14 two-stage + deep supervision tumor-focused training"
    )
    parser.add_argument("--mode", choices=["full", "medium", "quick_test", "evaluate", "build_cache", "audit"], default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--kits23-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-dinov3", action="store_true")
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--dinov3-repo", default=None)
    parser.add_argument("--dinov3-model", default=None, choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"])
    parser.add_argument("--dinov3-weights", default=None)
    parser.add_argument("--unfreeze-dinov3", action="store_true", help="Force unfreeze from epoch 1 (skip stage 1)")
    parser.add_argument("--unfreeze-after", type=int, default=None, help="Epoch to unfreeze encoder (default: 50)")
    parser.add_argument("--unfreeze-blocks", type=int, default=None, help="Number of top ViT blocks to unfreeze")
    parser.add_argument("--encoder-lr-factor", type=float, default=None, help="LR multiplier for encoder (default: 0.1)")
    parser.add_argument("--no-deep-supervision", action="store_true")
    parser.add_argument("--slice-batch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--patch-size", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--iters-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-every", type=int, default=None, help="Alias for --fast-val-every")
    parser.add_argument("--fast-val-every", type=int, default=None, help="Cheap patch-validation interval")
    parser.add_argument("--sw-val-every", type=int, default=None, help="Expensive full sliding-window validation interval")
    parser.add_argument("--no-sw-final", action="store_true", help="Do not force SW validation on the final epoch")
    parser.add_argument("--fast-patience", type=int, default=None, help="Early-stop patience measured in fast-val checks")
    parser.add_argument("--min-epochs-before-stop", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--sw-val-cases", type=int, default=None)
    parser.add_argument("--sw-batch-size", type=int, default=None)
    parser.add_argument("--selection", choices=["tumor", "kidney_tumor", "rare", "fg"], default=None)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--tumor-kidney-dist", type=float, default=None, help="Max distance from kidney for tumor/cyst (0=off)")
    args = parser.parse_args()

    mode = _resolve_mode(args, parser)
    cfg = get_config(mode if mode not in ("evaluate", "build_cache", "audit") else "full")

    # Apply CLI overrides
    if args.local_dinov3:
        cfg["dinov3_source"] = "local"
    if args.hf_model:
        cfg["dinov3_hf_model"] = args.hf_model
    if args.kits23_dir:
        cfg["kits23_dir"] = args.kits23_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.dinov3_repo:
        cfg["dinov3_repo"] = args.dinov3_repo
    if args.dinov3_model:
        cfg["dinov3_model"] = args.dinov3_model
    if args.dinov3_weights:
        cfg["dinov3_weights"] = args.dinov3_weights
    if args.unfreeze_dinov3:
        cfg["unfreeze_after"] = 0  # unfreeze immediately
    if args.unfreeze_after is not None:
        cfg["unfreeze_after"] = args.unfreeze_after
    if args.unfreeze_blocks is not None:
        cfg["unfreeze_top_n_blocks"] = args.unfreeze_blocks
    if args.encoder_lr_factor is not None:
        cfg["encoder_lr_factor"] = args.encoder_lr_factor
    if args.no_deep_supervision:
        cfg["deep_supervision"] = False
    if args.slice_batch:
        cfg["dinov3_slice_batch"] = args.slice_batch
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.grad_accum:
        cfg["grad_accum_steps"] = args.grad_accum
    if args.patch_size:
        cfg["patch_size"] = tuple(int(x) for x in args.patch_size.split(","))
    if args.num_epochs:
        cfg["num_epochs"] = args.num_epochs
    if args.iters_per_epoch:
        cfg["iters_per_epoch"] = args.iters_per_epoch
    if args.lr:
        cfg["lr"] = args.lr
    if args.val_every:
        cfg["fast_val_every"] = args.val_every
        cfg["val_every"] = args.val_every
    if args.fast_val_every is not None:
        cfg["fast_val_every"] = args.fast_val_every
        cfg["val_every"] = args.fast_val_every
    if args.sw_val_every is not None:
        cfg["sw_val_every"] = args.sw_val_every
    if args.no_sw_final:
        cfg["sw_val_on_final"] = False
    if args.fast_patience is not None:
        cfg["fast_patience"] = args.fast_patience
    if args.min_epochs_before_stop is not None:
        cfg["min_epochs_before_stop"] = args.min_epochs_before_stop
    if args.patience:
        cfg["patience"] = args.patience
    if args.sw_val_cases is not None:
        cfg["sw_val_cases"] = args.sw_val_cases
    if args.sw_batch_size:
        cfg["sw_batch_size"] = args.sw_batch_size
    if args.selection:
        cfg["selection"] = args.selection
    if args.no_tta:
        cfg["use_tta"] = False
        cfg["val_tta"] = False
    if args.tumor_kidney_dist is not None:
        cfg["tumor_kidney_max_dist"] = args.tumor_kidney_dist
        cfg["context_max_dist"] = args.tumor_kidney_dist

    _patch_base_for_v14()
    logging.getLogger("v14").info("Patched runtime with V14 model + two-stage training.")

    if mode == "build_cache":
        base.build_cache(cfg["kits23_dir"], base._cache_dir_for(cfg["kits23_dir"]))
    elif mode == "audit":
        base.audit_labels(cfg)
    elif mode == "evaluate":
        if not args.checkpoint:
            parser.error("--checkpoint required for evaluate mode")
        base.evaluate(cfg, args.checkpoint)
    else:
        if args.no_resume:
            resume_file = Path(cfg["output_dir"]) / "resume_state.pth"
            if resume_file.exists():
                resume_file.unlink()
        # Use V14's custom training loop instead of base.train()
        train_v14(cfg, resume_path=args.resume, skip_resume=args.no_resume)


if __name__ == "__main__":
    main()
