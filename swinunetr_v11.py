"""
SwinUNETR V11 -- Rare-class optimized KiTS23 kidney segmentation.

This is a self-contained v11 training entry point. It embeds the cache,
metrics, rare-class sampling, loss, model, and inference helpers it needs so
the file can be copied to another machine without the old v9/review files.

Core changes from v9:
  - single-stage, fully-trainable model by default
  - no LLM bottleneck, no frozen encoder, no LoRA schedule
  - component-uniform rare-class patch sampling with quota batches
  - present-class Focal-Tversky + Focal-CE loss
  - Gaussian softmax sliding-window inference with optional TTA
  - rescue thresholds and rare-class checkpoint selection

Recommended workflow:
  python swinunetr_v11.py --mode build_cache
  python swinunetr_v11.py --mode quick_test --no-resume
  python swinunetr_v11.py --mode full
  python swinunetr_v11.py --mode evaluate --checkpoint ./output/swin_v11/best_final.pth
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
import sys
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm


CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]
_DTYPE = torch.bfloat16


def get_config(mode: str) -> dict:
    n_cpu = min(multiprocessing.cpu_count(), 12)
    base = {
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/swin_v11",
        "num_classes": 4,
        "arch": "res_unet",  # "res_unet" or "swin_unetr"
        "feature_size": 48,
        "channels": (32, 64, 128, 256, 320),
        "blocks_per_stage": 2,
        "deep_supervision": True,
        "ds_weights": (1.0, 0.5, 0.25),
        "patch_size": (128, 128, 128),
        "batch_size": 2,
        "grad_accum_steps": 2,
        "iters_per_epoch": 180,
        "num_epochs": 160,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 8,
        "patience": 8,
        "val_every": 3,
        "grad_clip_norm": 1.0,
        "ema_decay": 0.999,
        "val_split": 0.20,
        "val_cases": 40,
        "test_cases": 40,
        "quick_cases": None,
        "seed": 42,
        "num_workers": n_cpu,
        "prefetch_factor": 4,
        "quota_classes": (2, 3),
        "class_sampling_bias": (0.15, 0.30, 0.35, 0.20),
        "min_target_voxels": (0, 500, 50, 20),
        "jitter_fraction": 0.25,
        "max_sample_attempts": 5,
        "max_stored_coords_per_component": 1000,
        "tversky_alpha": 0.3,
        "tversky_beta": 0.7,
        "tversky_gamma": 0.75,
        "tversky_smooth": 1.0,
        "fg_dice_weights": (1.0, 2.0, 3.0),
        "focal_gamma": 2.0,
        "ce_weight": (0.5, 1.0, 4.0, 8.0),
        "sw_val_cases": 40,
        "sw_overlap": 0.5,
        "sw_batch_size": 4,
        "gaussian_sigma_scale": 0.125,
        "use_tta": True,
        "val_tta": False,
        "rescue_tau": (0.0, 0.0, 0.35, 0.30),
        "class_keep_min_voxels": (0, 1000, 20, 10),
        "context_max_dist": None,
        "context_filtered_classes": (2, 3),
        "selection": "rare",
        "use_checkpoint": True,
        "use_compile": False,
    }

    if mode == "quick_test":
        base.update({
            "output_dir": "./output/swin_v11_quick",
            "quick_cases": 30,
            "num_epochs": 2,
            "iters_per_epoch": 12,
            "warmup_epochs": 1,
            "patience": 3,
            "val_every": 1,
            "sw_val_cases": 3,
            "sw_overlap": 0.0,
            "grad_accum_steps": 1,
            "num_workers": min(n_cpu, 4),
            "use_tta": False,
        })
    elif mode == "showcase":
        base.update({
            "output_dir": "./output/swin_v11_showcase",
            "quick_cases": 150,
            "num_epochs": 60,
            "iters_per_epoch": 160,
            "warmup_epochs": 4,
            "patience": 15,
            "val_every": 5,
            "sw_val_cases": 8,
            "use_tta": False,
        })
    elif mode == "medium":
        base.update({
            "output_dir": "./output/swin_v11_medium",
            "num_epochs": 140,
            "iters_per_epoch": 220,
            "warmup_epochs": 6,
            "patience": 25,
            "val_every": 5,
            "sw_val_cases": 12,
            "use_tta": False,
        })
    else:
        base["quick_cases"] = None

    return base


@dataclass
class MethodConfig:
    num_classes: int = 4
    in_channels: int = 1
    arch: str = "res_unet"
    channels: tuple[int, ...] = (32, 64, 128, 256, 320)
    blocks_per_stage: int = 2
    deep_supervision: bool = True
    ds_weights: tuple[float, ...] = (1.0, 0.5, 0.25)
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 2
    grad_accum_steps: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 400
    iters_per_epoch: int = 250
    warmup_epochs: int = 10
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    val_interval: int = 5
    quota_classes: tuple[int, ...] = (2, 3)
    class_sampling_bias: tuple[float, float, float, float] = (0.15, 0.30, 0.35, 0.20)
    min_target_voxels: tuple[int, int, int, int] = (0, 500, 50, 20)
    jitter_fraction: float = 0.25
    max_sample_attempts: int = 5
    max_stored_coords_per_component: int = 1000
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_gamma: float = 0.75
    tversky_smooth: float = 1.0
    fg_dice_weights: tuple[float, float, float] = (1.0, 2.0, 3.0)
    focal_gamma: float = 2.0
    ce_weight: tuple[float, float, float, float] = (0.5, 1.0, 4.0, 8.0)
    sw_batch_size: int = 4
    sw_overlap: float = 0.5
    gaussian_sigma_scale: float = 0.125
    use_tta: bool = True
    rescue_tau: tuple[float, float, float, float] = (0.0, 0.0, 0.35, 0.30)
    class_keep_min_voxels: tuple[int, int, int, int] = (0, 1000, 20, 10)
    context_max_dist: Optional[float] = None
    context_filtered_classes: tuple[int, ...] = (2, 3)
    selection: str = "rare"


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + identity)


def _make_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
    blocks = [ResidualBlock3D(in_channels, out_channels, stride=stride)]
    blocks += [ResidualBlock3D(out_channels, out_channels) for _ in range(num_blocks - 1)]
    return nn.Sequential(*blocks)


class ResidualUNet3D(nn.Module):
    def __init__(self, config: MethodConfig):
        super().__init__()
        ch = config.channels
        nb = config.blocks_per_stage
        self.deep_supervision = config.deep_supervision
        self.encoder = nn.ModuleList(
            [_make_stage(config.in_channels, ch[0], nb, stride=1)]
            + [_make_stage(ch[i], ch[i + 1], nb, stride=2) for i in range(len(ch) - 1)]
        )
        self.ups = nn.ModuleList(
            [nn.ConvTranspose3d(ch[i + 1], ch[i], kernel_size=2, stride=2) for i in range(len(ch) - 1)]
        )
        self.decoder = nn.ModuleList(
            [_make_stage(ch[i] * 2, ch[i], nb, stride=1) for i in range(len(ch) - 1)]
        )
        self.head_full = nn.Conv3d(ch[0], config.num_classes, 1)
        self.head_half = nn.Conv3d(ch[1], config.num_classes, 1)
        self.head_quarter = nn.Conv3d(ch[2], config.num_classes, 1)

    def forward(self, x: torch.Tensor):
        skips = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        x = skips[-1]
        decoder_outputs: dict[int, torch.Tensor] = {}
        for i in reversed(range(len(self.ups))):
            x = self.ups[i](x)
            x = torch.cat([x, skips[i]], dim=1)
            x = self.decoder[i](x)
            decoder_outputs[i] = x
        full = self.head_full(decoder_outputs[0])
        if self.training and self.deep_supervision:
            return [
                full,
                self.head_half(decoder_outputs[1]),
                self.head_quarter(decoder_outputs[2]),
            ]
        return full


class RareClassSegLoss(nn.Module):
    def __init__(self, config: MethodConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.alpha = config.tversky_alpha
        self.beta = config.tversky_beta
        self.gamma_tv = config.tversky_gamma
        self.smooth = config.tversky_smooth
        self.focal_gamma = config.focal_gamma
        self.register_buffer("ce_weight", torch.tensor(config.ce_weight, dtype=torch.float32))
        self.register_buffer("fg_weight", torch.tensor(config.fg_dice_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == logits.dim():
            target = target.squeeze(1)
        target = target.long()
        logits = logits.float()
        prob = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        tp = (prob * target_oh).sum(dim=(2, 3, 4))
        fp = (prob * (1.0 - target_oh)).sum(dim=(2, 3, 4))
        fn = ((1.0 - prob) * target_oh).sum(dim=(2, 3, 4))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1.0 - tversky).clamp(min=0.0).pow(self.gamma_tv)

        present = (target_oh.sum(dim=(2, 3, 4)) > 0).float()[:, 1:]
        weights = self.fg_weight.unsqueeze(0) * present
        loss_tv = (focal_tversky[:, 1:] * weights).sum() / weights.sum().clamp(min=1e-8)

        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        voxel_w = self.ce_weight[target]
        focal_ce = voxel_w * (1.0 - pt).pow(self.focal_gamma) * (-log_pt)
        loss_ce = focal_ce.sum() / voxel_w.sum().clamp(min=1e-8)
        return 0.5 * loss_tv + 0.5 * loss_ce


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, weights: Sequence[float]):
        super().__init__()
        self.base_loss = base_loss
        self.weights = tuple(weights)

    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, target)
        if target.dim() == outputs[0].dim() - 1:
            target = target.unsqueeze(1)
        total = outputs[0].new_zeros(())
        weight_sum = 0.0
        for out, weight in zip(outputs, self.weights):
            scaled = target.float()
            if out.shape[2:] != target.shape[2:]:
                scaled = F.interpolate(scaled, size=out.shape[2:], mode="nearest")
            total = total + weight * self.base_loss(out, scaled.long())
            weight_sum += weight
        return total / weight_sum


def _match_shape(array: np.ndarray, shape: Sequence[int], pad_value: float) -> np.ndarray:
    slices = []
    for current, wanted in zip(array.shape, shape):
        if current > wanted:
            start = (current - wanted) // 2
            slices.append(slice(start, start + wanted))
        else:
            slices.append(slice(0, current))
    array = array[tuple(slices)]
    pad = [(0, max(0, wanted - current)) for current, wanted in zip(array.shape, shape)]
    if any(p[1] for p in pad):
        array = np.pad(array, pad, mode="constant", constant_values=pad_value)
    return array


def augment_patch(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    patch_shape = image.shape
    for axis in range(3):
        if np.random.random() < 0.5:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    if np.random.random() < 0.40:
        k = np.random.randint(1, 4)
        axes = tuple(np.random.choice(3, 2, replace=False).tolist())
        image = np.rot90(image, k=k, axes=axes)
        label = np.rot90(label, k=k, axes=axes)
    if np.random.random() < 0.15:
        scale = float(np.random.uniform(0.88, 1.12))
        image = ndimage.zoom(image.astype(np.float32), scale, order=1)
        label = ndimage.zoom(label.astype(np.float32), scale, order=0)
        image = _match_shape(image, patch_shape, pad_value=0.0)
        label = _match_shape(label, patch_shape, pad_value=0)
    if np.random.random() < 0.25:
        image = np.clip(
            image + np.random.randn(*image.shape).astype(np.float32) * float(np.random.uniform(0.01, 0.08)),
            0.0,
            1.0,
        )
    if np.random.random() < 0.20:
        image = np.clip(image + float(np.random.uniform(-0.1, 0.1)), 0.0, 1.0)
    if np.random.random() < 0.20:
        image = np.power(np.clip(image, 0.0, 1.0), float(np.random.uniform(0.7, 1.5)))
    if np.random.random() < 0.15:
        image = ndimage.gaussian_filter(image.astype(np.float32), float(np.random.uniform(0.5, 1.5)))
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(label).astype(np.int64)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                ema_state[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                ema_state[key].copy_(value)


def _method_config(config: dict):
    return MethodConfig(
        num_classes=config["num_classes"],
        in_channels=1,
        arch=config["arch"],
        channels=tuple(config["channels"]),
        blocks_per_stage=config["blocks_per_stage"],
        deep_supervision=config["deep_supervision"],
        ds_weights=tuple(config["ds_weights"]),
        patch_size=tuple(config["patch_size"]),
        batch_size=config["batch_size"],
        grad_accum_steps=config["grad_accum_steps"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        num_epochs=config["num_epochs"],
        iters_per_epoch=config["iters_per_epoch"],
        warmup_epochs=config["warmup_epochs"],
        grad_clip_norm=config["grad_clip_norm"],
        ema_decay=config["ema_decay"],
        val_interval=config["val_every"],
        quota_classes=tuple(config["quota_classes"]),
        class_sampling_bias=tuple(config["class_sampling_bias"]),
        min_target_voxels=tuple(config["min_target_voxels"]),
        jitter_fraction=config["jitter_fraction"],
        max_sample_attempts=config["max_sample_attempts"],
        max_stored_coords_per_component=config["max_stored_coords_per_component"],
        tversky_alpha=config["tversky_alpha"],
        tversky_beta=config["tversky_beta"],
        tversky_gamma=config["tversky_gamma"],
        tversky_smooth=config["tversky_smooth"],
        fg_dice_weights=tuple(config["fg_dice_weights"]),
        focal_gamma=config["focal_gamma"],
        ce_weight=tuple(config["ce_weight"]),
        sw_batch_size=config["sw_batch_size"],
        sw_overlap=config["sw_overlap"],
        gaussian_sigma_scale=config["gaussian_sigma_scale"],
        use_tta=config["use_tta"],
        rescue_tau=tuple(config["rescue_tau"]),
        class_keep_min_voxels=tuple(config["class_keep_min_voxels"]),
        context_max_dist=config["context_max_dist"],
        context_filtered_classes=tuple(config["context_filtered_classes"]),
        selection=config["selection"],
    )


def _amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=_DTYPE)
    return nullcontext()


def _unwrap_logits(outputs):
    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def build_model(config: dict) -> nn.Module:
    method_cfg = _method_config(config)
    if config["arch"] == "res_unet":
        return ResidualUNet3D(method_cfg)
    if config["arch"] == "swin_unetr":
        from monai.networks.nets import SwinUNETR

        return SwinUNETR(
            in_channels=1,
            out_channels=config["num_classes"],
            feature_size=config["feature_size"],
            use_checkpoint=config.get("use_checkpoint", True),
            spatial_dims=3,
        )
    raise ValueError(f"Unknown architecture: {config['arch']}")


def _strip_state(state_dict: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def _cache_dir_for(kits23_dir: str) -> Path:
    return Path(kits23_dir).parent / "kits23_cache_bf16"


def build_cache(kits23_dir: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted([c for c in Path(kits23_dir).iterdir() if c.is_dir() and c.name.startswith("case_")])
    print(f"Building cache for {len(cases)} cases -> {cache_dir}")
    skipped = 0
    for case in tqdm(cases, desc="Caching"):
        img_c = cache_dir / f"{case.name}_img.npy"
        lbl_c = cache_dir / f"{case.name}_lbl.npy"
        if img_c.exists() and lbl_c.exists():
            skipped += 1
            continue
        img_path = case / "imaging.nii.gz"
        lbl_path = case / "segmentation.nii.gz"
        if not img_path.exists() or not lbl_path.exists():
            continue
        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)
        img = (np.clip(img, -175.0, 250.0) + 175.0) / 425.0
        np.save(str(img_c), img.astype(np.float16))
        np.save(str(lbl_c), lbl)
    print(f"  Cache ready: {cache_dir} ({skipped} already cached)")


def _load_volume(d: dict, cache_dir: Optional[Path]):
    case_id = d.get("case_id", "")
    if cache_dir:
        img_c = cache_dir / f"{case_id}_img.npy"
        lbl_c = cache_dir / f"{case_id}_lbl.npy"
        if img_c.exists() and lbl_c.exists():
            img = np.array(np.load(str(img_c), mmap_mode="r"), dtype=np.float32)
            lbl = np.array(np.load(str(lbl_c), mmap_mode="r"), dtype=np.int64)
            return img, lbl
    img = nib.load(d["image"]).get_fdata().astype(np.float32)
    lbl = nib.load(d["label"]).get_fdata().astype(np.int64)
    img = (np.clip(img, -175.0, 250.0) + 175.0) / 425.0
    return img, lbl


def _load_label(d: dict, cache_dir: Optional[Path]) -> np.ndarray:
    case_id = d.get("case_id", "")
    if cache_dir:
        lbl_c = cache_dir / f"{case_id}_lbl.npy"
        if lbl_c.exists():
            return np.array(np.load(str(lbl_c), mmap_mode="r"), dtype=np.int64)
    return nib.load(d["label"]).get_fdata().astype(np.int64)


def _parse_tuple(text: str, cast=float) -> tuple:
    return tuple(cast(x.strip()) for x in text.split(",") if x.strip())


class QuotaPatchDataset(Dataset):
    """Lazy KiTS23 patch dataset with methodology_v2 rare-class composition.

    The component index stores only sampled coordinates per connected component,
    while images are loaded lazily from the v9 cache on each patch request.
    With multiple workers each worker maintains its own quota queue; set
    --num-workers 0 for exact global batch ordering during debugging.
    """

    def __init__(self, data_dicts: Sequence[dict], config: dict, is_train: bool, cache_dir: Optional[Path]):
        self.data_dicts = list(data_dicts)
        self.config = config
        self.is_train = is_train
        self.cache_dir = cache_dir
        self.patch_size = tuple(config["patch_size"])
        self.num_classes = config["num_classes"]
        self.base_seed = int(config.get("seed", 42))
        self.rng = np.random.default_rng(self.base_seed)
        self.worker_id: Optional[int] = None
        self.queue: deque[int] = deque()
        self.effective_batch = int(config["batch_size"] * config.get("grad_accum_steps", 1))
        self.length = (
            int(config["iters_per_epoch"] * config["batch_size"])
            if is_train
            else len(self.data_dicts)
        )

        self.components: dict[int, list[tuple[int, np.ndarray]]] = {
            c: [] for c in range(1, self.num_classes)
        }
        if self.is_train:
            self._index_components()

    def __len__(self) -> int:
        return self.length

    def _ensure_worker_rng(self) -> None:
        info = get_worker_info()
        wid = info.id if info is not None else 0
        if self.worker_id != wid:
            self.worker_id = wid
            self.rng = np.random.default_rng(self.base_seed + 100_003 * wid + (0 if self.is_train else 17))
            self.queue.clear()

    def _component_index_cache_path(self) -> Path:
        case_ids = [d.get("case_id", "") for d in self.data_dicts]
        digest = hashlib.sha1("\n".join(case_ids).encode("utf-8")).hexdigest()[:12]
        base_dir = self.cache_dir if self.cache_dir is not None else Path(self.config["output_dir"]) / "component_index"
        base_dir.mkdir(parents=True, exist_ok=True)
        max_coords = int(self.config["max_stored_coords_per_component"])
        return base_dir / f"v11_components_{digest}_c{self.num_classes}_m{max_coords}.pkl"

    def _load_component_index_cache(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        case_ids = [d.get("case_id", "") for d in self.data_dicts]
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            meta = payload.get("meta", {})
            if (
                meta.get("case_ids") != case_ids
                or meta.get("num_classes") != self.num_classes
                or meta.get("max_stored_coords_per_component") != int(self.config["max_stored_coords_per_component"])
            ):
                return False
            self.components = payload["components"]
            summary = {c: len(v) for c, v in self.components.items()}
            print(f"  Component index loaded from cache: {cache_path} {summary}")
            return True
        except Exception as exc:
            print(f"  Component index cache could not be loaded ({exc}); rebuilding.")
            return False

    def _save_component_index_cache(self, cache_path: Path) -> None:
        payload = {
            "meta": {
                "case_ids": [d.get("case_id", "") for d in self.data_dicts],
                "num_classes": self.num_classes,
                "max_stored_coords_per_component": int(self.config["max_stored_coords_per_component"]),
            },
            "components": self.components,
        }
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            print(f"  Component index cached: {cache_path}")
        except Exception as exc:
            print(f"  Component index cache save skipped ({exc}).")

    def _index_components(self) -> None:
        cache_path = self._component_index_cache_path()
        if self._load_component_index_cache(cache_path):
            return

        max_coords = int(self.config["max_stored_coords_per_component"])
        iterator = tqdm(self.data_dicts, desc="Index components", leave=False)
        for case_idx, d in enumerate(iterator):
            label = _load_label(d, self.cache_dir)
            for class_id in range(1, self.num_classes):
                mask = label == class_id
                if not mask.any():
                    continue
                labeled, n_comp = ndimage.label(mask)
                objects = ndimage.find_objects(labeled)
                for comp_id in range(1, n_comp + 1):
                    box = objects[comp_id - 1]
                    if box is None:
                        continue
                    local = np.argwhere(labeled[box] == comp_id)
                    offset = np.array([s.start for s in box], dtype=np.int64)
                    coords = local + offset
                    if len(coords) > max_coords:
                        keep = self.rng.choice(len(coords), size=max_coords, replace=False)
                        coords = coords[keep]
                    self.components[class_id].append((case_idx, coords.astype(np.int32, copy=False)))

        summary = {c: len(v) for c, v in self.components.items()}
        print(f"  Component index: {summary}")
        missing = [c for c, comps in self.components.items() if not comps]
        if missing:
            print(f"  WARNING: no components found for classes {missing}; draws fall back to available foreground.")
        self._save_component_index_cache(cache_path)

    def _pad_if_needed(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pad = [(0, max(0, p - s)) for p, s in zip(self.patch_size, image.shape)]
        if any(p[1] for p in pad):
            image = np.pad(image, pad, mode="constant", constant_values=0.0)
            label = np.pad(label, pad, mode="constant", constant_values=0)
        return image, label

    def _draw_from_bias(self) -> int:
        probs = np.array(self.config["class_sampling_bias"], dtype=np.float64)
        probs = probs / probs.sum()
        choice = int(self.rng.choice(4, p=probs))
        return [1, 2, 3, 0][choice]

    def _refill_queue(self) -> None:
        composition = [c for c in self.config["quota_classes"] if self.components.get(c)]
        while len(composition) < self.effective_batch:
            composition.append(self._draw_from_bias())
        self.rng.shuffle(composition)
        self.queue.extend(composition)

    def _fallback_class(self) -> int:
        for class_id in sorted(self.components, reverse=True):
            if self.components[class_id]:
                return class_id
        return 0

    def _random_start(self, shape: Sequence[int]) -> tuple[int, int, int]:
        return tuple(
            int(self.rng.integers(0, max(1, size - patch + 1)))
            for size, patch in zip(shape, self.patch_size)
        )

    def _class_start(self, target: int, label: np.ndarray, coords: np.ndarray) -> tuple[int, int, int]:
        shape = label.shape
        jitter_max = [max(1, int(p * self.config["jitter_fraction"])) for p in self.patch_size]
        min_voxels = int(self.config["min_target_voxels"][target])
        best_start = self._random_start(shape)
        best_count = -1

        for _ in range(int(self.config["max_sample_attempts"])):
            center = coords[int(self.rng.integers(len(coords)))]
            start = tuple(
                int(np.clip(
                    int(center[i])
                    + int(self.rng.integers(-jitter_max[i], jitter_max[i] + 1))
                    - self.patch_size[i] // 2,
                    0,
                    shape[i] - self.patch_size[i],
                ))
                for i in range(3)
            )
            window = tuple(slice(start[i], start[i] + self.patch_size[i]) for i in range(3))
            count = int(np.count_nonzero(label[window] == target))
            if count > best_count:
                best_count = count
                best_start = start
            if count >= min_voxels:
                break
        return best_start

    def _next_target(self) -> int:
        if not self.queue:
            self._refill_queue()
        target = self.queue.popleft()
        if target != 0 and not self.components.get(target):
            target = self._fallback_class()
        return target

    def _sample_train_patch(self) -> tuple[np.ndarray, np.ndarray]:
        target = self._next_target()
        coords = None
        if target == 0:
            case_idx = int(self.rng.integers(len(self.data_dicts)))
        else:
            comps = self.components[target]
            case_idx, coords = comps[int(self.rng.integers(len(comps)))]

        image, label = _load_volume(self.data_dicts[case_idx], self.cache_dir)
        image, label = self._pad_if_needed(image, label)
        if target == 0:
            start = self._random_start(label.shape)
        else:
            start = self._class_start(target, label, coords)
        window = tuple(slice(start[i], start[i] + self.patch_size[i]) for i in range(3))
        image = image[window].copy()
        label = label[window].copy()
        return augment_patch(image, label)

    def _center_crop(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, label = self._pad_if_needed(image, label)
        starts = [(s - p) // 2 for s, p in zip(image.shape, self.patch_size)]
        window = tuple(slice(starts[i], starts[i] + self.patch_size[i]) for i in range(3))
        return image[window].copy(), label[window].copy()

    def __getitem__(self, idx: int) -> dict:
        self._ensure_worker_rng()
        if self.is_train:
            image, label = self._sample_train_patch()
            case_id = "quota_patch"
        else:
            d = self.data_dicts[idx]
            image, label = _load_volume(d, self.cache_dir)
            image, label = self._center_crop(image, label)
            case_id = d.get("case_id", "unknown")
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0),
            "label": torch.from_numpy(np.ascontiguousarray(label)).long().unsqueeze(0),
            "case_id": case_id,
        }


def _loader_kwargs(config: dict, train: bool) -> dict:
    workers = int(config.get("num_workers", 0))
    if not train:
        workers = max(0, workers // 2)
    kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": int(config.get("prefetch_factor", 2 if not train else 4)),
        })
    return kwargs


def get_data_dicts(config: dict) -> tuple[list[dict], list[dict], list[Path]]:
    data_dir = Path(config["kits23_dir"])
    all_cases = sorted([c for c in data_dir.iterdir() if c.is_dir() and c.name.startswith("case_")])
    test_cases = all_cases[-int(config.get("test_cases", 40)):]
    train_val_cases = all_cases[:-int(config.get("test_cases", 40))]
    if config.get("quick_cases"):
        train_val_cases = train_val_cases[: int(config["quick_cases"])]
    data_dicts = [
        {"image": str(c / "imaging.nii.gz"), "label": str(c / "segmentation.nii.gz"), "case_id": c.name}
        for c in train_val_cases
        if (c / "imaging.nii.gz").exists() and (c / "segmentation.nii.gz").exists()
    ]
    n_val = int(config["val_cases"]) if config.get("val_cases") else int(len(data_dicts) * config["val_split"])
    n_val = min(max(n_val, 1), max(len(data_dicts) - 1, 1))
    val_dicts = data_dicts[:n_val]
    train_dicts = data_dicts[n_val:]
    return train_dicts, val_dicts, test_cases


def get_dataloaders(config: dict):
    cache_dir = _cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists():
        print(f"  Cache not found at {cache_dir}")
        print("  Run: python swinunetr_v11.py --mode build_cache")
        print("  Training will fall back to NIfTI reads and will be slower.")
        cache_dir = None

    train_dicts, val_dicts, test_cases = get_data_dicts(config)
    print(f"  Split: {len(train_dicts)} train | {len(val_dicts)} val | {len(test_cases)} test")

    train_ds = QuotaPatchDataset(train_dicts, config, is_train=True, cache_dir=cache_dir)
    val_ds = QuotaPatchDataset(val_dicts, config, is_train=False, cache_dir=cache_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True,
        **_loader_kwargs(config, train=True),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(config, train=False),
    )
    return train_loader, val_loader, train_dicts, val_dicts, test_cases


def build_loss(config: dict, device: torch.device) -> nn.Module:
    method_cfg = _method_config(config)
    base_loss = RareClassSegLoss(method_cfg).to(device)
    if config["arch"] == "res_unet" and config.get("deep_supervision", True):
        return DeepSupervisionLoss(base_loss, method_cfg.ds_weights)
    return base_loss


def cosine_warmup_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.01 + 0.5 * 0.99 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class EarlyStopping:
    def __init__(self, patience: int, ckpt_path: Path):
        self.patience = patience
        self.ckpt_path = Path(ckpt_path)
        self.best: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def step(self, score: float, model: nn.Module, epoch: int, config: dict) -> bool:
        if self.best is None or score > self.best:
            self.best = score
            self.counter = 0
            torch.save({
                "model_state_dict": _strip_state(model.state_dict()),
                "epoch": epoch,
                "selection_metric": score,
                "selection": config["selection"],
                "version": "swinunetr_v11",
                "config": {k: str(v) for k, v in config.items()},
            }, self.ckpt_path)
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def train_one_epoch(model, loader, optimizer, loss_fn, device, config, ema=None) -> float:
    torch.cuda.empty_cache()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    count = 0
    pending = False
    accum = int(config.get("grad_accum_steps", 1))

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        try:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with _amp_context(device):
                outputs = model(image)
                loss = loss_fn(outputs, label) / accum
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            total_loss += loss.item() * accum
            count += 1
            pending = True
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                pending = False
            del image, label, outputs, loss
        except RuntimeError as exc:
            print(f"  Train batch error: {exc}")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    if pending:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
    return total_loss / max(count, 1)


def validate_patch_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="QuickVal", leave=False):
            try:
                image = batch["image"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                with _amp_context(device):
                    outputs = model(image)
                    loss = loss_fn(outputs, label)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    count += 1
                del image, label, outputs, loss
            except RuntimeError:
                torch.cuda.empty_cache()
    return total_loss / max(count, 1)


def _gaussian_importance_map(patch_size: Sequence[int], sigma_scale: float) -> torch.Tensor:
    axes = []
    for size in patch_size:
        coords = torch.arange(size, dtype=torch.float32)
        center = (size - 1) / 2.0
        sigma = max(size * sigma_scale, 1e-3)
        axes.append(torch.exp(-0.5 * ((coords - center) / sigma) ** 2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    weight = weight / weight.max()
    return weight.clamp(min=1e-3)


def sliding_window_probabilities(model, inputs: torch.Tensor, config: dict, device: torch.device) -> torch.Tensor:
    assert inputs.shape[0] == 1
    pd, ph, pw = tuple(config["patch_size"])
    _, _, d_orig, h_orig, w_orig = inputs.shape
    pad_d, pad_h, pad_w = max(0, pd - d_orig), max(0, ph - h_orig), max(0, pw - w_orig)
    if pad_d or pad_h or pad_w:
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h, 0, pad_d), value=0.0)

    _, _, depth, height, width = inputs.shape
    step_d = max(1, int(pd * (1.0 - config["sw_overlap"])))
    step_h = max(1, int(ph * (1.0 - config["sw_overlap"])))
    step_w = max(1, int(pw * (1.0 - config["sw_overlap"])))

    def starts(size: int, patch: int, step: int) -> list[int]:
        points = list(range(0, size - patch + 1, step))
        if not points or points[-1] + patch < size:
            points.append(size - patch)
        return points

    coords = [
        (d0, h0, w0)
        for d0 in starts(depth, pd, step_d)
        for h0 in starts(height, ph, step_h)
        for w0 in starts(width, pw, step_w)
    ]
    importance = _gaussian_importance_map(config["patch_size"], config["gaussian_sigma_scale"])
    output = None
    norm = None
    sw_batch_size = int(config.get("sw_batch_size", 4))

    for start in range(0, len(coords), sw_batch_size):
        batch_coords = coords[start:start + sw_batch_size]
        batch_in = torch.cat(
            [inputs[:, :, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw] for d0, h0, w0 in batch_coords],
            dim=0,
        )
        with torch.no_grad():
            with _amp_context(device):
                logits = _unwrap_logits(model(batch_in))
            probs = F.softmax(logits.float(), dim=1).cpu()

        if output is None:
            n_classes = probs.shape[1]
            output = torch.zeros(1, n_classes, depth, height, width)
            norm = torch.zeros(1, 1, depth, height, width)

        weighted = probs * importance
        for i, (d0, h0, w0) in enumerate(batch_coords):
            output[:, :, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw] += weighted[i:i + 1]
            norm[:, :, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw] += importance
        del batch_in, probs, weighted

    output = output / norm.clamp(min=1e-8)
    return output[:, :, :d_orig, :h_orig, :w_orig]


def predict_probabilities(model, volume: np.ndarray, config: dict, device: torch.device, tta: bool) -> np.ndarray:
    model.eval()
    image = torch.from_numpy(np.ascontiguousarray(volume)).float().unsqueeze(0).unsqueeze(0).to(device)
    flip_sets = [(False, False, False)]
    if tta:
        import itertools

        flip_sets = list(itertools.product([False, True], repeat=3))
    accumulated = None
    for flips in flip_sets:
        axes = [i + 2 for i, flag in enumerate(flips) if flag]
        flipped = torch.flip(image, dims=axes) if axes else image
        probs = sliding_window_probabilities(model, flipped, config, device)
        if axes:
            probs = torch.flip(probs, dims=axes)
        accumulated = probs if accumulated is None else accumulated + probs
    return (accumulated / len(flip_sets)).squeeze(0).numpy()


def apply_rescue_thresholds(probs: np.ndarray, prediction: np.ndarray, config: dict) -> np.ndarray:
    output = prediction.copy()
    for class_id, tau in enumerate(config["rescue_tau"]):
        if class_id == 0 or tau <= 0:
            continue
        output[(probs[class_id] > tau) & (output == 0)] = class_id
    return output


def postprocess_prediction(prediction: np.ndarray, config: dict) -> np.ndarray:
    output = prediction.copy()
    for class_id, min_voxels in enumerate(config["class_keep_min_voxels"]):
        if class_id == 0 or min_voxels <= 0:
            continue
        mask = output == class_id
        if not mask.any():
            continue
        labeled, _ = ndimage.label(mask)
        sizes = np.bincount(labeled.ravel())
        remove = sizes < min_voxels
        remove[0] = False
        output[remove[labeled]] = 0

    if config.get("context_max_dist") is not None and (output == 1).any():
        dist_to_class1 = ndimage.distance_transform_edt(output != 1)
        for class_id in config["context_filtered_classes"]:
            mask = output == class_id
            if not mask.any():
                continue
            labeled, n_comp = ndimage.label(mask)
            for comp_id in range(1, n_comp + 1):
                comp = labeled == comp_id
                if dist_to_class1[comp].min() > config["context_max_dist"]:
                    output[comp] = 0
    return output


def predict_volume(model, volume: np.ndarray, config: dict, device: torch.device, tta: bool, postprocess: bool = True):
    probs = predict_probabilities(model, volume, config, device, tta=tta)
    pred = np.argmax(probs, axis=0)
    pred = apply_rescue_thresholds(probs, pred, config)
    if postprocess:
        pred = postprocess_prediction(pred, config)
    return pred


def per_class_dice(pred: np.ndarray, label: np.ndarray, num_classes: int) -> np.ndarray:
    dice = np.full(num_classes, np.nan, dtype=np.float64)
    for class_id in range(num_classes):
        gt = label == class_id
        pr = pred == class_id
        denom = gt.sum() + pr.sum()
        if denom > 0:
            dice[class_id] = 2.0 * np.logical_and(gt, pr).sum() / denom
    return dice


def validate_sliding_window(model, val_dicts, config, device, n_cases=None, logger=None):
    model.eval()
    cache_dir = _cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists():
        cache_dir = None
    cases = list(val_dicts)
    if n_cases and n_cases < len(cases):
        rng = np.random.default_rng(config.get("seed", 42))
        idx = rng.choice(len(cases), size=n_cases, replace=False)
        cases = [cases[i] for i in idx]

    all_dice = []
    log = logger.info if logger else print
    for d in tqdm(cases, desc="SW-Val", leave=False):
        try:
            image, label = _load_volume(d, cache_dir)
            pred = predict_volume(model, image, config, device, tta=bool(config.get("val_tta", False)))
            all_dice.append(per_class_dice(pred, label, config["num_classes"]))
            del image, label, pred
            torch.cuda.empty_cache()
        except Exception as exc:
            log(f"  SW-Val error {d.get('case_id', '?')}: {exc}")

    if not all_dice:
        return 0.0, {"dice": [0.0] * config["num_classes"], "selection_metric": 0.0}

    dice_arr = np.stack(all_dice)
    mean = np.nanmean(dice_arr, axis=0)
    std = np.nanstd(dice_arr, axis=0)
    rare_mean = float(np.nan_to_num(mean[2:4]).mean())
    fg_mean = float(np.nan_to_num(mean[1:]).mean())
    selection = rare_mean if config["selection"] == "rare" else fg_mean
    log(
        f"  SW-Val ({len(cases)} cases): "
        f"K={mean[1]:.4f} T={mean[2]:.4f} C={mean[3]:.4f} "
        f"Rare={rare_mean:.4f} MeanFG={fg_mean:.4f} Select={selection:.4f}"
    )
    return selection, {
        "dice": mean.tolist(),
        "dice_std": std.tolist(),
        "rare_mean": rare_mean,
        "mean_fg_dice": fg_mean,
        "selection_metric": selection,
        "selection": config["selection"],
    }


def save_history(output_dir, epoch, train_loss, val_loss, sw_metrics, elapsed_s, is_best):
    hist_file = Path(output_dir) / "history.json"
    hist = json.loads(hist_file.read_text()) if hist_file.exists() else {"epochs": []}
    hist["epochs"].append({
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "epoch_time_min": round(elapsed_s / 60, 2),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "sw_metrics": sw_metrics,
        "is_best": is_best,
    })
    hist_file.write_text(json.dumps(hist, indent=2))


def _fmt(v, fmt=".4f") -> str:
    return f"{v:{fmt}}" if v is not None else "N/A"


def _load_resume(path: Path, device: torch.device):
    """Load a resume checkpoint, tolerating a corrupted primary file.

    A run killed mid-write can leave a truncated zip ("failed finding central
    directory"). We try the primary file, then a ``.bak`` rotated backup, and
    finally give up gracefully so training can start fresh instead of crashing.
    """
    candidates = [p for p in (path, path.with_suffix(path.suffix + ".bak")) if p.exists()]
    for cand in candidates:
        try:
            state = torch.load(str(cand), map_location=device, weights_only=False)
            if cand != path:
                logging.getLogger("v11").warning(
                    f"Primary resume file unreadable; recovered from backup {cand.name}"
                )
            return state
        except Exception as e:
            logging.getLogger("v11").warning(
                f"Resume checkpoint {cand.name} is corrupted ({type(e).__name__}: {e}); "
                "trying next candidate."
            )
    if candidates:
        logging.getLogger("v11").warning(
            "No readable resume checkpoint found; starting from scratch."
        )
    return None


def train(config: dict, resume_path=None, skip_resume=False):
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
        handlers=[logging.FileHandler(output_dir / "training.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("v11")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff_batch = config["batch_size"] * config.get("grad_accum_steps", 1)

    logger.info("=" * 72)
    logger.info("SwinUNETR V11 -- methodology_v2 rare-class optimized training")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Device: {device}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    logger.info(f"Output: {output_dir}")
    logger.info(f"Arch: {config['arch']} | Selection: {config['selection']} | Eff.Batch: {eff_batch}")
    logger.info("No LLM bottleneck, no frozen encoder, no LoRA schedule.")
    logger.info("=" * 72)

    train_loader, val_loader, _, val_dicts, _ = get_dataloaders(config)
    model = build_model(config).to(device)

    if config.get("use_compile") and hasattr(torch, "compile"):
        if config.get("ema_decay", 0.0) > 0:
            logger.warning("Disabling EMA because --compile is enabled.")
            config["ema_decay"] = 0.0
        logger.info("Compiling model with torch.compile(mode='reduce-overhead')")
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

    loss_fn = build_loss(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = cosine_warmup_scheduler(optimizer, config["num_epochs"], config["warmup_epochs"])
    ema = ModelEMA(model, config["ema_decay"]) if config.get("ema_decay", 0.0) > 0 else None

    resume_state = None
    auto_resume = output_dir / "resume_state.pth"
    if resume_path:
        resume_state = _load_resume(Path(resume_path), device)
    elif not skip_resume:
        resume_state = _load_resume(auto_resume, device)

    start_epoch = 1
    es = EarlyStopping(config["patience"], output_dir / "best_final.pth")
    if resume_state:
        model.load_state_dict(_strip_state(resume_state["model_state_dict"]))
        if ema is not None and resume_state.get("ema_state_dict"):
            ema.module.load_state_dict(_strip_state(resume_state["ema_state_dict"]))
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = int(resume_state["epoch"]) + 1
        es.best = resume_state.get("best_metric")
        es.counter = int(resume_state.get("es_counter", 0))
        logger.info(f"Resumed from epoch {resume_state['epoch']} | best={_fmt(es.best)}")

    for epoch in range(start_epoch, int(config["num_epochs"]) + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, config, ema=ema)
        scheduler.step()
        elapsed = time.time() - t0
        do_val = epoch % int(config["val_every"]) == 0 or epoch == int(config["num_epochs"])
        val_loss = None
        sw_metrics = None
        improved = False
        eval_model = ema.module if ema is not None else model

        if do_val:
            val_loss = validate_patch_loss(eval_model, val_loader, loss_fn, device)
            score, sw_metrics = validate_sliding_window(
                eval_model,
                val_dicts,
                config,
                device,
                n_cases=config.get("sw_val_cases"),
                logger=logger,
            )
            improved = es.step(score, eval_model, epoch, config)
            if improved:
                logger.info(f"  New best {config['selection']} metric = {score:.4f}")
            elif es.best is not None:
                logger.info(f"  No improvement {es.counter}/{es.patience} | best={es.best:.4f}")

        val_text = f"val_loss={_fmt(val_loss)}"
        if not do_val:
            next_val = min(
                int(config["num_epochs"]),
                epoch + (int(config["val_every"]) - (epoch % int(config["val_every"]))),
            )
            val_text = f"val_loss=N/A(next E{next_val:03d})"
        elif sw_metrics is not None and sw_metrics.get("dice"):
            dice = sw_metrics["dice"]
            val_text += (
                f" K={dice[1]:.4f} T={dice[2]:.4f} C={dice[3]:.4f}"
                f" Rare={sw_metrics.get('rare_mean', 0.0):.4f}"
                f" MeanFG={sw_metrics.get('mean_fg_dice', 0.0):.4f}"
            )
        logger.info(
            f"[E{epoch:03d}/{config['num_epochs']}] train={train_loss:.4f} "
            f"{val_text} lr={scheduler.get_last_lr()[0]:.2e} "
            f"time={elapsed / 60:.1f}min"
        )
        save_history(output_dir, epoch, train_loss, val_loss, sw_metrics, elapsed, improved)

        resume_tmp = output_dir / "resume_state.tmp"
        torch.save({
            "model_state_dict": _strip_state(model.state_dict()),
            "ema_state_dict": _strip_state(ema.module.state_dict()) if ema is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": es.best,
            "es_counter": es.counter,
            "config": {k: str(v) for k, v in config.items()},
        }, resume_tmp)
        resume_final = output_dir / "resume_state.pth"
        if resume_final.exists():
            try:
                os.replace(resume_final, output_dir / "resume_state.pth.bak")
            except OSError:
                pass
        os.replace(resume_tmp, resume_final)

        if es.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    history = json.loads((output_dir / "history.json").read_text()) if (output_dir / "history.json").exists() else {"epochs": []}
    total_min = sum(e.get("epoch_time_min", 0) for e in history["epochs"])
    summary = {
        "version": "swinunetr_v11",
        "method": "methodology_v2 rare-class optimized",
        "arch": config["arch"],
        "selection": config["selection"],
        "best_metric": es.best,
        "total_hours": round(total_min / 60, 2),
        "config": {k: str(v) for k, v in config.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Training complete. Best {config['selection']} metric: {_fmt(es.best)}")
    logger.info(f"Summary -> {output_dir / 'summary.json'}")


def compute_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    if not pred_mask.any() or not gt_mask.any():
        return float("nan")
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt

        pred_border = pred_mask ^ binary_erosion(pred_mask)
        gt_border = gt_mask ^ binary_erosion(gt_mask)
        distances = np.concatenate([
            distance_transform_edt(~gt_mask)[pred_border],
            distance_transform_edt(~pred_mask)[gt_border],
        ])
        return float(np.percentile(distances, 95))
    except Exception:
        return float("nan")


def aggregate_metrics(per_case_metrics: list, num_classes: int = 4) -> dict:
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    for metrics in per_case_metrics:
        raw_counts = metrics.get("raw_counts", {})
        if raw_counts:
            tp += np.array(raw_counts["tp"])
            fp += np.array(raw_counts["fp"])
            fn += np.array(raw_counts["fn"])
    eps = 1e-8
    dice = (2 * tp / (2 * tp + fp + fn + eps)).tolist()
    iou = (tp / (tp + fp + fn + eps)).tolist()
    precision = (tp / (tp + fp + eps)).tolist()
    recall = (tp / (tp + fn + eps)).tolist()
    hd95_arr = np.array([m.get("hd95", [float("nan")] * num_classes) for m in per_case_metrics])
    hd95_mean = np.nanmean(hd95_arr, axis=0).tolist()
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "hd95_mean": hd95_mean,
        "mean_fg_dice": float(np.mean(dice[1:])),
        "mean_fg_iou": float(np.mean(iou[1:])),
        "mean_fg_precision": float(np.mean(precision[1:])),
        "mean_fg_recall": float(np.mean(recall[1:])),
        "mean_fg_hd95": float(np.nanmean(hd95_mean[1:])),
    }


def compute_seg_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> dict:
    dice_l, iou_l, precision_l, recall_l = [], [], [], []
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        gt_mask = gt == class_id
        tp = int((pred_mask & gt_mask).sum())
        fp = int((pred_mask & ~gt_mask).sum())
        fn = int((~pred_mask & gt_mask).sum())
        dice_l.append(2 * tp / (2 * tp + fp + fn + 1e-8))
        iou_l.append(tp / (tp + fp + fn + 1e-8))
        precision_l.append(tp / (tp + fp + 1e-8))
        recall_l.append(tp / (tp + fn + 1e-8))
    return {"dice": dice_l, "iou": iou_l, "precision": precision_l, "recall": recall_l}


def evaluate(config: dict, checkpoint: str):
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
    logger = logging.getLogger("v11_eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(_strip_state(state))
    model.eval()
    logger.info(f"Loaded: {checkpoint}")

    cache_dir = _cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists():
        cache_dir = None
    data_dir = Path(config["kits23_dir"])
    test_cases = sorted([c for c in data_dir.iterdir() if c.is_dir() and c.name.startswith("case_")])[-int(config["test_cases"]):]
    per_case = []
    conf_matrix = np.zeros((config["num_classes"], config["num_classes"]), dtype=np.int64)
    logger.info(f"\nEvaluating {len(test_cases)} test cases | TTA={config.get('use_tta', True)}")

    with torch.no_grad():
        for case_dir in test_cases:
            d = {"image": str(case_dir / "imaging.nii.gz"), "label": str(case_dir / "segmentation.nii.gz"), "case_id": case_dir.name}
            if not Path(d["image"]).exists() or not Path(d["label"]).exists():
                continue
            t0 = time.time()
            image, label = _load_volume(d, cache_dir)
            pred = predict_volume(model, image, config, device, tta=bool(config.get("use_tta", True)))
            metrics = compute_seg_metrics(pred, label, num_classes=config["num_classes"])
            hd95 = [float("nan")] + [compute_hd95(pred == c, label == c) for c in range(1, config["num_classes"])]
            for gt_class in range(config["num_classes"]):
                for pred_class in range(config["num_classes"]):
                    conf_matrix[gt_class, pred_class] += int(((label == gt_class) & (pred == pred_class)).sum())
            tp_raw = [int(((pred == c) & (label == c)).sum()) for c in range(config["num_classes"])]
            fp_raw = [int(((pred == c) & (label != c)).sum()) for c in range(config["num_classes"])]
            fn_raw = [int(((pred != c) & (label == c)).sum()) for c in range(config["num_classes"])]
            elapsed = time.time() - t0
            per_case.append({
                "case_id": case_dir.name,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "hd95": hd95,
                "raw_counts": {"tp": tp_raw, "fp": fp_raw, "fn": fn_raw},
                "elapsed_s": round(elapsed, 1),
            })
            logger.info(
                f"{case_dir.name} | K={metrics['dice'][1]:.3f} "
                f"T={metrics['dice'][2]:.3f} C={metrics['dice'][3]:.3f} ({elapsed:.0f}s)"
            )
            del image, label, pred
            torch.cuda.empty_cache()

    agg = aggregate_metrics(per_case, num_classes=config["num_classes"])
    logger.info("\n" + "=" * 72)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 72)
    logger.info(f"{'Class':<12} {'Dice':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'HD95':>8}")
    logger.info("-" * 72)
    for i, class_name in enumerate(CLASS_NAMES):
        logger.info(
            f"{class_name:<12} {agg['dice'][i]:>8.4f} {agg['iou'][i]:>8.4f} "
            f"{agg['precision'][i]:>10.4f} {agg['recall'][i]:>8.4f} {agg['hd95_mean'][i]:>8.2f}"
        )
    logger.info("-" * 72)
    logger.info(
        f"{'MeanFG':<12} {agg['mean_fg_dice']:>8.4f} {agg['mean_fg_iou']:>8.4f} "
        f"{agg['mean_fg_precision']:>10.4f} {agg['mean_fg_recall']:>8.4f} {agg['mean_fg_hd95']:>8.2f}"
    )

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(json.dumps({"global_metrics": agg, "per_case": per_case}, indent=2))
    csv_path = out_dir / "eval_per_case.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id",
            "dice_bg", "dice_kidney", "dice_tumor", "dice_cyst",
            "iou_bg", "iou_kidney", "iou_tumor", "iou_cyst",
            "prec_bg", "prec_kidney", "prec_tumor", "prec_cyst",
            "rec_bg", "rec_kidney", "rec_tumor", "rec_cyst",
            "hd95_kidney", "hd95_tumor", "hd95_cyst", "elapsed_s",
        ])
        for case_metrics in per_case:
            writer.writerow([
                case_metrics["case_id"],
                *case_metrics["dice"],
                *case_metrics["iou"],
                *case_metrics["precision"],
                *case_metrics["recall"],
                case_metrics["hd95"][1],
                case_metrics["hd95"][2],
                case_metrics["hd95"][3],
                case_metrics["elapsed_s"],
            ])
    (out_dir / "confusion_matrix.json").write_text(json.dumps({
        "class_names": CLASS_NAMES,
        "matrix_rows_GT_cols_pred": conf_matrix.tolist(),
    }, indent=2))
    (out_dir / "summary.json").write_text(json.dumps({
        "checkpoint": checkpoint,
        "global_metrics": agg,
        "n_test_cases": len(per_case),
    }, indent=2))
    logger.info(f"Results -> {out_dir / 'eval_results.json'}")
    logger.info(f"Per-case CSV -> {csv_path}")


def audit_labels(config: dict):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists():
        cache_dir = None
    train_dicts, val_dicts, _ = get_data_dicts(config)
    data_dicts = train_dicts + val_dicts
    voxel_counts = np.zeros(config["num_classes"], dtype=np.float64)
    component_sizes: dict[int, list[int]] = {c: [] for c in range(1, config["num_classes"])}

    for d in tqdm(data_dicts, desc="Audit labels"):
        label = _load_label(d, cache_dir)
        for class_id in range(config["num_classes"]):
            voxel_counts[class_id] += np.count_nonzero(label == class_id)
        for class_id in range(1, config["num_classes"]):
            mask = label == class_id
            if not mask.any():
                continue
            labeled, _ = ndimage.label(mask)
            sizes = np.bincount(labeled.ravel())[1:]
            component_sizes[class_id].extend(int(s) for s in sizes if s > 0)

    freq = voxel_counts / voxel_counts.sum()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(freq, 1e-12))
    suggested_ce = tuple(float(round(w, 3)) for w in (inv_sqrt / inv_sqrt[1]))
    suggested_min_keep = [0]
    for class_id in range(1, config["num_classes"]):
        sizes = np.array(component_sizes[class_id]) if component_sizes[class_id] else np.array([0])
        suggested_min_keep.append(max(2, int(0.5 * float(np.percentile(sizes, 5)))))

    result = {
        "n_cases": len(data_dicts),
        "voxel_frequency": freq.tolist(),
        "suggested_ce_weight": suggested_ce,
        "suggested_class_keep_min_voxels": tuple(suggested_min_keep),
        "component_size_percentiles": {
            class_id: {
                "p5": float(np.percentile(component_sizes[class_id], 5)) if component_sizes[class_id] else 0.0,
                "p50": float(np.percentile(component_sizes[class_id], 50)) if component_sizes[class_id] else 0.0,
                "p95": float(np.percentile(component_sizes[class_id], 95)) if component_sizes[class_id] else 0.0,
                "n_components": len(component_sizes[class_id]),
            }
            for class_id in range(1, config["num_classes"])
        },
    }
    (output_dir / "label_audit.json").write_text(json.dumps(result, indent=2))
    print("[audit] voxel_frequency:", np.array2string(freq, precision=6))
    print("[audit] suggested ce_weight:", suggested_ce)
    print("[audit] suggested class_keep_min_voxels:", tuple(suggested_min_keep))
    print(f"[audit] wrote {output_dir / 'label_audit.json'}")


def main():
    parser = argparse.ArgumentParser(description="SwinUNETR V11 rare-class optimized KiTS23 training")
    parser.add_argument("--mode", choices=["full", "medium", "showcase", "quick_test", "evaluate", "build_cache", "audit"], default="full")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--kits23-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--arch", choices=["res_unet", "swin_unetr"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--patch-size", default=None, help="D,H,W, for example 128,128,128")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--iters-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sw-overlap", type=float, default=None)
    parser.add_argument("--sw-batch-size", type=int, default=None)
    parser.add_argument("--sw-val-cases", type=int, default=None, help="Number of validation cases for sliding-window Dice during training")
    parser.add_argument("--selection", choices=["rare", "fg"], default=None)
    parser.add_argument("--ce-weight", default=None, help="Comma tuple, for example 0.5,1,4,8")
    parser.add_argument("--rescue-tau", default=None, help="Comma tuple, for example 0,0,0.35,0.30")
    parser.add_argument("--class-keep-min-voxels", default=None, help="Comma tuple, for example 0,1000,20,10")
    parser.add_argument("--context-max-dist", type=float, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="use_compile")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable SwinUNETR gradient checkpointing")
    parser.add_argument("--resume", default=None, metavar="PATH")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.mode if args.mode not in ("evaluate", "build_cache", "audit") else "full")
    if args.kits23_dir:
        cfg["kits23_dir"] = args.kits23_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.arch:
        cfg["arch"] = args.arch
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
        cfg["val_every"] = args.val_every
    if args.patience:
        cfg["patience"] = args.patience
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.sw_overlap is not None:
        cfg["sw_overlap"] = args.sw_overlap
    if args.sw_batch_size:
        cfg["sw_batch_size"] = args.sw_batch_size
    if args.sw_val_cases is not None:
        cfg["sw_val_cases"] = args.sw_val_cases
    if args.selection:
        cfg["selection"] = args.selection
    if args.ce_weight:
        cfg["ce_weight"] = _parse_tuple(args.ce_weight, float)
    if args.rescue_tau:
        cfg["rescue_tau"] = _parse_tuple(args.rescue_tau, float)
    if args.class_keep_min_voxels:
        cfg["class_keep_min_voxels"] = _parse_tuple(args.class_keep_min_voxels, int)
    if args.context_max_dist is not None:
        cfg["context_max_dist"] = args.context_max_dist
    if args.no_ema:
        cfg["ema_decay"] = 0.0
    if args.no_tta:
        cfg["use_tta"] = False
        cfg["val_tta"] = False
    if args.use_compile:
        cfg["use_compile"] = True
    if args.no_checkpoint:
        cfg["use_checkpoint"] = False

    if args.mode == "build_cache":
        build_cache(cfg["kits23_dir"], _cache_dir_for(cfg["kits23_dir"]))
    elif args.mode == "audit":
        audit_labels(cfg)
    elif args.mode == "evaluate":
        if not args.checkpoint:
            parser.error("--checkpoint required for evaluate mode")
        evaluate(cfg, args.checkpoint)
    else:
        if args.no_resume:
            resume_file = Path(cfg["output_dir"]) / "resume_state.pth"
            if resume_file.exists():
                resume_file.unlink()
        train(cfg, resume_path=args.resume, skip_resume=args.no_resume)


if __name__ == "__main__":
    main()
