"""
Methodology v2: 3D multi-class segmentation rebuilt for rare-class (class 2 / class 3) performance.

Label meaning (abstract):
  0 -> background
  1 -> large foreground region
  2 -> small sparse foreground region
  3 -> very small sparse foreground region

What changed vs. methodology_only.py and why:

  1. Architecture: residual 3D U-Net with deep supervision (nnU-Net recipe), fully
     trainable from the first step. No frozen encoder, no LoRA, no LLM bottleneck.
     (The frozen encoder + rank-2 LoRA was the single largest cause of rare-class
     failure: small objects depend on trainable high-resolution features.)
     A fully-trainable SwinUNETR (no prior block) is available via build_model()
     as a comparison arm.

  2. Sampling: quota-based composition. Every effective batch is guaranteed at
     least one class-2 patch and one class-3 patch. Patch centers are chosen
     component-uniformly (tiny components get equal exposure), jittered (no
     object-at-center bias), and resampled until a minimum number of target-class
     voxels is inside the window.

  3. Loss: masked Focal-Tversky (beta > alpha penalizes false negatives) computed
     only over classes PRESENT in the patch, plus Focal-CE with corrected class
     weights (class 3 > class 2 > class 1). The old loss spent most of its
     rare-class gradient suppressing classes 2/3 in patches where they were absent.

  4. Inference: sliding window with Gaussian importance weighting, overlap 0.5,
     softmax (not logit) averaging, optional 8-flip TTA, and per-class rescue
     thresholds so rare classes do not have to beat the background prior at argmax.

  5. Cleanup: per-class minimum component sizes meant to be derived from the GT
     component-size distribution (see audit_labels / verify_postprocess_preserves_truth)
     instead of a blanket 50 voxels that can delete true class-3 objects. Optional
     class-1 proximity filter for isolated rare-class false positives.

  6. Validation: per-volume, per-class Dice (matches the test metric), checkpoint
     selection on the rare-class mean (classes 2 and 3), EMA weights for evaluation.

Wiring example (data loading stays external, as before):

    volumes, labels = load_training_arrays()          # lists of 3D np.ndarray, image in [0,1], label in {0..3}
    config = MethodConfig()
    audit = audit_labels(labels_train, config)        # run once; copy suggested values into config
    sampler = QuotaPatchSampler(volumes, labels, config)
    model = build_model(config).to(device)
    result = run_training(model, sampler, val_volumes, val_labels, config, device)
    model.load_state_dict(result["best_state"])
    report = evaluate_test(model, test_volumes, test_labels, config, device)
"""

from __future__ import annotations

import copy
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MethodConfig:
    num_classes: int = 4
    in_channels: int = 1
    arch: str = "res_unet"  # "res_unet" | "swin_unetr"

    # Residual U-Net
    channels: tuple[int, ...] = (32, 64, 128, 256, 320)
    blocks_per_stage: int = 2
    deep_supervision: bool = True
    ds_weights: tuple[float, ...] = (1.0, 0.5, 0.25)

    # Patches / batches. Effective batch = batch_size * grad_accum_steps.
    # Patch dims must be divisible by 2 ** (len(channels) - 1) = 16.
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 2
    grad_accum_steps: int = 2

    # Optimization (single stage; everything trainable)
    lr: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 400
    iters_per_epoch: int = 250
    warmup_epochs: int = 10
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    val_interval: int = 5

    # Sampling. quota_classes are guaranteed in every effective batch; the
    # remaining slots are drawn from class_sampling_bias over (class1, class2,
    # class3, random). Class 3 is sampled at least as often as class 2.
    quota_classes: tuple[int, ...] = (2, 3)
    class_sampling_bias: tuple[float, float, float, float] = (0.15, 0.30, 0.35, 0.20)
    min_target_voxels: tuple[int, int, int, int] = (0, 500, 50, 20)
    jitter_fraction: float = 0.25
    max_sample_attempts: int = 5
    max_stored_coords_per_component: int = 1000

    # Loss
    tversky_alpha: float = 0.3   # false-positive weight
    tversky_beta: float = 0.7    # false-negative weight (recall emphasis)
    tversky_gamma: float = 0.75  # focal exponent on (1 - tversky)
    tversky_smooth: float = 1.0
    fg_dice_weights: tuple[float, float, float] = (1.0, 2.0, 3.0)  # classes 1, 2, 3
    focal_gamma: float = 2.0
    # Recompute from audit_labels(); class 3 must get the largest weight.
    ce_weight: tuple[float, float, float, float] = (0.5, 1.0, 4.0, 8.0)

    # Inference
    sw_batch_size: int = 4
    sw_overlap: float = 0.5
    gaussian_sigma_scale: float = 0.125
    use_tta: bool = True
    # Rescue: assign class k to background voxels where softmax_k > tau_k.
    # 0 disables. Tune on validation.
    rescue_tau: tuple[float, float, float, float] = (0.0, 0.0, 0.35, 0.30)

    # Cleanup. Derive from audit_labels(); must stay below the smallest true
    # component or the filter deletes ground truth.
    class_keep_min_voxels: tuple[int, int, int, int] = (0, 1000, 20, 10)
    # Optional: drop class-2/3 components farther than this (voxels) from the
    # class-1 mask. None disables; validate the co-location assumption first.
    context_max_dist: Optional[float] = None
    context_filtered_classes: tuple[int, ...] = (2, 3)

    # Checkpoint selection: "rare" = mean(class2, class3), "fg" = mean(1, 2, 3)
    selection: str = "rare"


# ---------------------------------------------------------------------------
# Model: residual 3D U-Net with deep supervision
# ---------------------------------------------------------------------------


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )
        else:
            self.skip = None

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
    """Encoder-decoder with strided-conv downsampling, transpose-conv upsampling,
    and deep-supervision heads at output strides 1, 2 and 4 (training only)."""

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


def build_model(config: MethodConfig) -> nn.Module:
    if config.arch == "res_unet":
        return ResidualUNet3D(config)
    if config.arch == "swin_unetr":
        # Comparison arm: plain SwinUNETR, fully trainable, no prior block,
        # no freezing, no LoRA. No deep supervision in this arm.
        from monai.networks.nets import SwinUNETR

        return SwinUNETR(
            in_channels=config.in_channels,
            out_channels=config.num_classes,
            feature_size=48,
            use_checkpoint=True,
            spatial_dims=3,
        )
    raise ValueError(f"unknown arch: {config.arch}")


# ---------------------------------------------------------------------------
# Loss: masked Focal-Tversky + Focal-CE, deep-supervision wrapper
# ---------------------------------------------------------------------------


class RareClassSegLoss(nn.Module):
    """0.5 * class-weighted Focal-Tversky over PRESENT foreground classes
    + 0.5 * class-weighted Focal-CE.

    Masking absent classes out of the Tversky average is the critical fix: a
    class with zero GT voxels in the patch must not contribute a saturated
    (1 - dice) = 1 penalty that teaches the model to suppress it everywhere.
    False positives for absent classes are still penalized through the CE term
    and through the alpha*fp term of patches where the class is present.
    """

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

        # Focal-Tversky over present foreground classes
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

        # Focal-CE, normalized by the summed per-voxel class weights
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
        for out, w in zip(outputs, self.weights):
            scaled = target.float()
            if out.shape[2:] != target.shape[2:]:
                scaled = F.interpolate(scaled, size=out.shape[2:], mode="nearest")
            total = total + w * self.base_loss(out, scaled.long())
            weight_sum += w
        return total / weight_sum


# ---------------------------------------------------------------------------
# Augmentation (shape-preserving)
# ---------------------------------------------------------------------------


def _match_shape(
    array: np.ndarray, shape: Sequence[int], pad_value: float
) -> np.ndarray:
    """Center-crop and/or pad to an exact shape (fixes the zoom shape drift)."""
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


# ---------------------------------------------------------------------------
# Quota patch sampler
# ---------------------------------------------------------------------------


class QuotaPatchSampler:
    """Patch sampler with three guarantees the old sampler lacked:

    1. Quota: every effective batch contains at least one patch for each class
       in config.quota_classes (default: class 2 and class 3).
    2. Component-uniform centers: a connected component of the target class is
       chosen uniformly first, then a voxel inside it, so tiny components are
       sampled as often as large ones. Centers are jittered to remove the
       object-at-patch-center bias.
    3. Minimum positive volume: the window is resampled (up to
       max_sample_attempts) until at least min_target_voxels[class] voxels of
       the target class are inside; the best attempt is kept otherwise.
    """

    def __init__(
        self,
        volumes: Sequence[np.ndarray],
        labels: Sequence[np.ndarray],
        config: MethodConfig,
        seed: int = 0,
    ):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.patch_size = tuple(config.patch_size)
        self.effective_batch = config.batch_size * config.grad_accum_steps
        self.queue: deque[int] = deque()  # class id per upcoming patch; 0 = random

        self.volumes: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        for image, label in zip(volumes, labels):
            pad = [(0, max(0, p - s)) for p, s in zip(self.patch_size, image.shape)]
            if any(p[1] for p in pad):
                image = np.pad(image, pad, mode="constant", constant_values=0.0)
                label = np.pad(label, pad, mode="constant", constant_values=0)
            self.volumes.append(np.ascontiguousarray(image, dtype=np.float32))
            self.labels.append(np.ascontiguousarray(label).astype(np.int64))

        # components[c] = list of (volume_index, coords[N, 3]) per connected component
        self.components: dict[int, list[tuple[int, np.ndarray]]] = {
            c: [] for c in range(1, config.num_classes)
        }
        max_coords = config.max_stored_coords_per_component
        for vol_idx, label in enumerate(self.labels):
            for c in range(1, config.num_classes):
                mask = label == c
                if not mask.any():
                    continue
                labeled, n_comp = ndimage.label(mask)
                objects = ndimage.find_objects(labeled)
                for comp_id in range(1, n_comp + 1):
                    box = objects[comp_id - 1]
                    local = np.argwhere(labeled[box] == comp_id)
                    offset = np.array([s.start for s in box])
                    coords = local + offset
                    if len(coords) > max_coords:
                        keep = self.rng.choice(len(coords), size=max_coords, replace=False)
                        coords = coords[keep]
                    self.components[c].append((vol_idx, coords))

        missing = [c for c, comps in self.components.items() if not comps]
        if missing:
            print(f"[sampler] WARNING: no voxels found for classes {missing}; "
                  f"their quota/bias draws fall back to other foreground classes")

    def _draw_from_bias(self) -> int:
        b1, b2, b3, br = self.config.class_sampling_bias
        probs = np.array([b1, b2, b3, br], dtype=np.float64)
        probs /= probs.sum()
        choice = self.rng.choice(4, p=probs)
        return [1, 2, 3, 0][choice]

    def _refill_queue(self) -> None:
        composition = [c for c in self.config.quota_classes if self.components.get(c)]
        while len(composition) < self.effective_batch:
            composition.append(self._draw_from_bias())
        self.rng.shuffle(composition)
        self.queue.extend(composition)

    def _fallback_class(self, target: int) -> int:
        # Prefer the rarest classes first instead of silently drifting to class 1.
        for c in sorted(self.components, reverse=True):
            if self.components[c]:
                return c
        return 0

    def _random_window(self, vol_idx: int) -> tuple[int, int, int]:
        shape = self.labels[vol_idx].shape
        return tuple(
            int(self.rng.integers(0, max(1, s - p + 1))) for s, p in zip(shape, self.patch_size)
        )

    def _class_window(self, target: int) -> tuple[int, tuple[int, int, int]]:
        comps = self.components[target]
        vol_idx, coords = comps[int(self.rng.integers(len(comps)))]
        shape = self.labels[vol_idx].shape
        jitter_max = [max(1, int(p * self.config.jitter_fraction)) for p in self.patch_size]
        min_voxels = self.config.min_target_voxels[target]
        label = self.labels[vol_idx]

        best_start: Optional[tuple[int, int, int]] = None
        best_count = -1
        for _ in range(self.config.max_sample_attempts):
            center = coords[int(self.rng.integers(len(coords)))]
            start = tuple(
                int(np.clip(
                    center[i] + self.rng.integers(-jitter_max[i], jitter_max[i] + 1) - self.patch_size[i] // 2,
                    0,
                    shape[i] - self.patch_size[i],
                ))
                for i in range(3)
            )
            window = tuple(slice(start[i], start[i] + self.patch_size[i]) for i in range(3))
            count = int(np.count_nonzero(label[window] == target))
            if count > best_count:
                best_count, best_start = count, start
            if count >= min_voxels:
                break
        return vol_idx, best_start

    def sample_patch(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.queue:
            self._refill_queue()
        target = self.queue.popleft()
        if target != 0 and not self.components.get(target):
            target = self._fallback_class(target)

        if target == 0:
            vol_idx = int(self.rng.integers(len(self.volumes)))
            start = self._random_window(vol_idx)
        else:
            vol_idx, start = self._class_window(target)

        window = tuple(slice(start[i], start[i] + self.patch_size[i]) for i in range(3))
        image = self.volumes[vol_idx][window]
        label = self.labels[vol_idx][window]
        return augment_patch(image, label)

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        images, labels = [], []
        for _ in range(batch_size):
            image, label = self.sample_patch()
            images.append(torch.from_numpy(image))
            labels.append(torch.from_numpy(label))
        return {
            "image": torch.stack(images).unsqueeze(1),
            "label": torch.stack(labels).unsqueeze(1),
        }


# ---------------------------------------------------------------------------
# Inference: Gaussian sliding window + TTA + rescue thresholds
# ---------------------------------------------------------------------------


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


def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Sequence[int],
    sw_batch_size: int,
    predictor: nn.Module,
    overlap: float = 0.5,
    sigma_scale: float = 0.125,
) -> torch.Tensor:
    """Returns softmax PROBABILITIES (1, C, D, H, W) on CPU, Gaussian-blended."""
    assert inputs.shape[0] == 1
    pd, ph, pw = roi_size
    _, _, d_orig, h_orig, w_orig = inputs.shape
    pad_d, pad_h, pad_w = max(0, pd - d_orig), max(0, ph - h_orig), max(0, pw - w_orig)
    if pad_d or pad_h or pad_w:
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h, 0, pad_d), value=0.0)

    _, _, depth, height, width = inputs.shape
    step_d = max(1, int(pd * (1 - overlap)))
    step_h = max(1, int(ph * (1 - overlap)))
    step_w = max(1, int(pw * (1 - overlap)))

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
    importance = _gaussian_importance_map(roi_size, sigma_scale)  # CPU, (pd, ph, pw)
    output: Optional[torch.Tensor] = None
    norm: Optional[torch.Tensor] = None

    for start in range(0, len(coords), sw_batch_size):
        batch_coords = coords[start : start + sw_batch_size]
        batch_in = torch.cat(
            [inputs[:, :, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw] for d0, h0, w0 in batch_coords],
            dim=0,
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=DTYPE):
                logits = predictor(batch_in)
            probs = F.softmax(logits.float(), dim=1).cpu()

        if output is None:
            n_classes = probs.shape[1]
            output = torch.zeros(1, n_classes, depth, height, width)
            norm = torch.zeros(1, 1, depth, height, width)

        weighted = probs * importance
        for i, (d0, h0, w0) in enumerate(batch_coords):
            output[:, :, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw] += weighted[i : i + 1]
            norm[:, :, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw] += importance

    output = output / norm.clamp(min=1e-8)
    return output[:, :, :d_orig, :h_orig, :w_orig]


def predict_probabilities(
    model: nn.Module,
    volume: np.ndarray,
    config: MethodConfig,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    """Full-volume class probabilities (C, D, H, W), optionally 8-flip TTA-averaged."""
    model.eval()
    image = torch.from_numpy(np.ascontiguousarray(volume)).float().unsqueeze(0).unsqueeze(0).to(device)
    flip_sets = list(itertools.product([False, True], repeat=3)) if tta else [(False, False, False)]
    accumulated: Optional[torch.Tensor] = None
    for flips in flip_sets:
        axes = [i + 2 for i, f in enumerate(flips) if f]
        flipped = torch.flip(image, dims=axes) if axes else image
        probs = sliding_window_inference(
            flipped,
            roi_size=config.patch_size,
            sw_batch_size=config.sw_batch_size,
            predictor=model,
            overlap=config.sw_overlap,
            sigma_scale=config.gaussian_sigma_scale,
        )
        if axes:
            probs = torch.flip(probs, dims=axes)
        accumulated = probs if accumulated is None else accumulated + probs
    return (accumulated / len(flip_sets)).squeeze(0).numpy()


def apply_rescue_thresholds(probs: np.ndarray, prediction: np.ndarray, config: MethodConfig) -> np.ndarray:
    """Assign rare class k to BACKGROUND voxels where softmax_k > tau_k, so a
    rare class does not have to beat the background prior at argmax."""
    output = prediction.copy()
    for class_id, tau in enumerate(config.rescue_tau):
        if class_id == 0 or tau <= 0:
            continue
        output[(probs[class_id] > tau) & (output == 0)] = class_id
    return output


def postprocess_prediction(prediction: np.ndarray, config: MethodConfig) -> np.ndarray:
    output = prediction.copy()
    for class_id, min_voxels in enumerate(config.class_keep_min_voxels):
        if class_id == 0 or min_voxels <= 0:
            continue
        mask = output == class_id
        if not mask.any():
            continue
        labeled, _ = ndimage.label(mask)
        component_sizes = np.bincount(labeled.ravel())
        remove = component_sizes < min_voxels
        remove[0] = False
        output[remove[labeled]] = 0

    if config.context_max_dist is not None and (output == 1).any():
        dist_to_class1 = ndimage.distance_transform_edt(output != 1)
        for class_id in config.context_filtered_classes:
            mask = output == class_id
            if not mask.any():
                continue
            labeled, n_comp = ndimage.label(mask)
            for comp_id in range(1, n_comp + 1):
                comp = labeled == comp_id
                if dist_to_class1[comp].min() > config.context_max_dist:
                    output[comp] = 0
    return output


def predict_volume(
    model: nn.Module,
    volume: np.ndarray,
    config: MethodConfig,
    device: torch.device,
    tta: Optional[bool] = None,
    postprocess: bool = True,
) -> np.ndarray:
    probs = predict_probabilities(
        model, volume, config, device, tta=config.use_tta if tta is None else tta
    )
    pred = np.argmax(probs, axis=0)
    pred = apply_rescue_thresholds(probs, pred, config)
    if postprocess:
        pred = postprocess_prediction(pred, config)
    return pred


# ---------------------------------------------------------------------------
# Metrics, validation, test evaluation
# ---------------------------------------------------------------------------


def per_class_dice(pred: np.ndarray, label: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-volume Dice per class; NaN when the class is absent from both GT and
    prediction (excluded from means via nanmean — matches common test protocol)."""
    dice = np.full(num_classes, np.nan, dtype=np.float64)
    for class_id in range(num_classes):
        gt = label == class_id
        pr = pred == class_id
        denom = gt.sum() + pr.sum()
        if denom == 0:
            continue
        dice[class_id] = 2.0 * np.logical_and(gt, pr).sum() / denom
    return dice


def validate(
    model: nn.Module,
    volumes: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    config: MethodConfig,
    device: torch.device,
    tta: bool = False,
    postprocess: bool = True,
) -> dict:
    model.eval()
    all_dice = []
    for volume, label in zip(volumes, labels):
        pred = predict_volume(model, volume, config, device, tta=tta, postprocess=postprocess)
        all_dice.append(per_class_dice(pred, label, config.num_classes))
    all_dice = np.stack(all_dice)  # (n_volumes, num_classes)
    mean = np.nanmean(all_dice, axis=0)
    std = np.nanstd(all_dice, axis=0)
    rare_mean = float(np.nan_to_num(mean[2:4]).mean())
    fg_mean = float(np.nan_to_num(mean[1:]).mean())
    return {
        "dice_mean": mean,
        "dice_std": std,
        "rare_mean": rare_mean,
        "fg_mean": fg_mean,
        "selection_metric": rare_mean if config.selection == "rare" else fg_mean,
        "per_volume": all_dice,
    }


def evaluate_test(
    model: nn.Module,
    volumes: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    config: MethodConfig,
    device: torch.device,
) -> dict:
    report = validate(model, volumes, labels, config, device, tta=config.use_tta, postprocess=True)
    mean, std = report["dice_mean"], report["dice_std"]
    print(f"Test volumes evaluated: {len(volumes)}")
    print(f"Test mean foreground Dice: {report['fg_mean']:.4f}")
    for class_id in range(1, config.num_classes):
        print(f"  class {class_id}: {mean[class_id]:.4f} +- {std[class_id]:.4f}")
    return report


# ---------------------------------------------------------------------------
# Dataset audits (run BEFORE training; copy results into MethodConfig)
# ---------------------------------------------------------------------------


def audit_labels(labels: Sequence[np.ndarray], config: MethodConfig) -> dict:
    """Derives data-driven config values:
    - ce_weight from inverse-sqrt voxel frequency (class 3 should come out largest)
    - class_keep_min_voxels from the 5th-percentile GT component size
    """
    voxel_counts = np.zeros(config.num_classes, dtype=np.float64)
    component_sizes: dict[int, list[int]] = {c: [] for c in range(1, config.num_classes)}
    for label in labels:
        for c in range(config.num_classes):
            voxel_counts[c] += np.count_nonzero(label == c)
        for c in range(1, config.num_classes):
            mask = label == c
            if not mask.any():
                continue
            labeled, _ = ndimage.label(mask)
            sizes = np.bincount(labeled.ravel())[1:]
            component_sizes[c].extend(int(s) for s in sizes if s > 0)

    freq = voxel_counts / voxel_counts.sum()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(freq, 1e-12))
    suggested_ce = tuple(float(round(w, 3)) for w in (inv_sqrt / inv_sqrt[1]))

    suggested_min_keep = [0]
    for c in range(1, config.num_classes):
        sizes = np.array(component_sizes[c]) if component_sizes[c] else np.array([0])
        p5 = float(np.percentile(sizes, 5))
        suggested_min_keep.append(max(2, int(0.5 * p5)))

    result = {
        "voxel_frequency": freq,
        "suggested_ce_weight": suggested_ce,
        "component_size_percentiles": {
            c: {
                "p5": float(np.percentile(component_sizes[c], 5)) if component_sizes[c] else 0.0,
                "p50": float(np.percentile(component_sizes[c], 50)) if component_sizes[c] else 0.0,
                "p95": float(np.percentile(component_sizes[c], 95)) if component_sizes[c] else 0.0,
                "n_components": len(component_sizes[c]),
            }
            for c in range(1, config.num_classes)
        },
        "suggested_class_keep_min_voxels": tuple(suggested_min_keep),
    }
    print("[audit] voxel frequency:", np.array2string(freq, precision=6))
    print("[audit] suggested ce_weight:", result["suggested_ce_weight"])
    print("[audit] suggested class_keep_min_voxels:", result["suggested_class_keep_min_voxels"])
    print("[audit] component size percentiles:", result["component_size_percentiles"])
    return result


def verify_postprocess_preserves_truth(labels: Sequence[np.ndarray], config: MethodConfig) -> bool:
    """Runs GT through the cleanup. Any per-class Dice < 1.0 proves the filter
    deletes true objects and class_keep_min_voxels must be lowered."""
    ok = True
    for i, label in enumerate(labels):
        cleaned = postprocess_prediction(label.astype(np.int64), config)
        dice = per_class_dice(cleaned, label, config.num_classes)
        for class_id in range(1, config.num_classes):
            if not np.isnan(dice[class_id]) and dice[class_id] < 1.0:
                print(f"[audit] volume {i}: cleanup destroys class {class_id} GT "
                      f"(dice {dice[class_id]:.4f}) -> lower class_keep_min_voxels[{class_id}]")
                ok = False
    if ok:
        print("[audit] cleanup preserves all GT components")
    return ok


# ---------------------------------------------------------------------------
# Training: single stage, everything trainable, EMA, rare-class selection
# ---------------------------------------------------------------------------


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                ema_state[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                ema_state[key].copy_(value)


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.01 + 0.5 * 0.99 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: nn.Module,
    sampler: QuotaPatchSampler,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: MethodConfig,
    device: torch.device,
    ema: Optional[ModelEMA] = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    pending = False

    for it in range(1, config.iters_per_epoch + 1):
        batch = sampler.sample_batch(config.batch_size)
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=DTYPE):
            outputs = model(image)
        loss = loss_fn(outputs, label) / config.grad_accum_steps
        loss.backward()
        total_loss += loss.item() * config.grad_accum_steps
        pending = True

        if it % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            pending = False

    if pending:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
    return total_loss / max(config.iters_per_epoch, 1)


def run_training(
    model: nn.Module,
    sampler: QuotaPatchSampler,
    val_volumes: Sequence[np.ndarray],
    val_labels: Sequence[np.ndarray],
    config: MethodConfig,
    device: torch.device,
) -> dict:
    base_loss = RareClassSegLoss(config).to(device)
    loss_fn: nn.Module = (
        DeepSupervisionLoss(base_loss, config.ds_weights)
        if config.deep_supervision and config.arch == "res_unet"
        else base_loss
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = cosine_warmup_scheduler(optimizer, config.num_epochs, config.warmup_epochs)
    ema = ModelEMA(model, config.ema_decay) if config.ema_decay > 0 else None

    best_metric = -1.0
    best_state: Optional[dict[str, torch.Tensor]] = None
    history: list[dict] = []

    for epoch in range(1, config.num_epochs + 1):
        epoch_loss = train_one_epoch(model, sampler, optimizer, loss_fn, config, device, ema=ema)
        scheduler.step()

        if epoch % config.val_interval == 0 or epoch == config.num_epochs:
            eval_model = ema.module if ema is not None else model
            report = validate(eval_model, val_volumes, val_labels, config, device,
                              tta=False, postprocess=True)
            mean = report["dice_mean"]
            print(
                f"epoch {epoch:4d} | loss {epoch_loss:.4f} | "
                f"dice c1 {mean[1]:.4f} c2 {mean[2]:.4f} c3 {mean[3]:.4f} | "
                f"rare {report['rare_mean']:.4f} | fg {report['fg_mean']:.4f}"
            )
            history.append({"epoch": epoch, "loss": epoch_loss, **{
                "dice_mean": mean.tolist(), "rare_mean": report["rare_mean"],
                "fg_mean": report["fg_mean"],
            }})
            if report["selection_metric"] > best_metric:
                best_metric = report["selection_metric"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()
                }

    return {"best_metric": best_metric, "best_state": best_state, "history": history}
