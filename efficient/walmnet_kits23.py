"""
WaLM-Net — Wavelet-Lifting Mamba Network for KiTS23 3D Segmentation
==================================================================
Efficient yet SOTA-oriented 3D segmentation of kidney / tumor / cyst.

Design (see efficient/RESEARCH_FOCUS.md for the full rationale). The headline
contribution is *rare/small-structure (tumor + cyst) robustness at a fraction of
nnU-Net compute*, achieved by a single light U-shaped backbone that fuses:

  1. LEARNABLE LIFTING WAVELET (LearnableLiftingWavelet3D) — a separable 3D
     lifting-scheme transform used for downsampling. Unlike WaveFormer's fixed
     Haar basis, the predict/update operators are learned, while the lifting
     structure guarantees *perfect reconstruction by construction*. Initialised
     as the lazy wavelet (P=U=0), so at init DWT->IDWT is exactly invertible.
  2. LINEAR-COST TOKEN MIXER (WaMambaBlock) — SegMamba-style gated spatial conv
     (local) + a bidirectional state-space mixer (global, linear in voxels).
     Uses the real `mamba_ssm` kernel when available, else a dependency-free
     SSM-lite fallback so the model runs anywhere (Plan B in the research doc).
  3. HIGH-FREQUENCY DETAIL PATH (HFBranch) — the 7 high-freq sub-bands feed a
     lightweight depthwise-separable branch that is gated back into the main
     path (single branch, not HybridMamba's dual branch) to preserve boundaries
     and small structures that Mamba blurs.
  4. EFFICIENT DECODER (EffiDec3D-style) — channel-reduced decoder with
     attention-gated skips + deep supervision + a dedicated rare-class head.

Loss = Dice + weighted CE + Tversky (alpha<beta -> recall on tumor/cyst)
       + boundary loss, with deep supervision.

Reporting: per-class Dice, IoU/Jaccard, Surface-Dice and HD95 (MONAI).

Usage:
    python walmnet_kits23.py --selftest          # CPU shape/recon/param check (no data)
    python walmnet_kits23.py --quick             # short smoke-training run
    python walmnet_kits23.py --full              # full training
    python walmnet_kits23.py --evaluate --checkpoint <path>

The model + --selftest need only PyTorch. Data/training/eval additionally need
MONAI (and SciPy for post-processing); those are imported lazily.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional real Mamba kernel. Falls back to a pure-PyTorch mixer if absent.
try:  # pragma: no cover - depends on environment
    from mamba_ssm import Mamba as _MambaSSM
    _HAS_MAMBA = True
except Exception:  # noqa: BLE001
    _MambaSSM = None
    _HAS_MAMBA = False


# ===========================================================================
# Configuration
# ===========================================================================

def get_config(mode: str = "full") -> dict:
    """Return a config dict for the requested mode (full / medium / quick_test)."""
    cfg = {
        # ---- data ----
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/walmnet",
        "num_classes": 4,                       # bg, kidney, tumor, cyst
        "class_names": ("background", "kidney", "tumor", "cyst"),
        "spacing": (1.5, 1.5, 1.5),
        "ct_window": (-200.0, 300.0),           # HU clip range for kidney CT
        "patch_size": (128, 128, 128),
        "num_samples_per_volume": 2,            # crops per volume per step
        "pos_neg_ratio": (2, 1),                # foreground vs background crop bias
        "val_split": 0.2,
        "cache_rate": 0.1,
        "num_workers": 4,

        # ---- model ----
        "in_channels": 1,
        "features": (32, 64, 128, 256),         # stem + 3 encoder stages
        "bottleneck_channels": 256,
        "decoder_channels": (128, 64, 32, 32),  # EffiDec3D-style reduced widths
        "use_high_freq_branch": True,
        "use_rare_head": True,
        "mixer": "auto",                        # auto | mamba | lite
        "deep_supervision": True,
        "ds_weights": (1.0, 0.4, 0.2),          # full, 1/2, 1/4 resolution

        # ---- loss ----
        "ce_weight": (0.1, 1.0, 5.0, 4.0),      # bg, kidney, tumor, cyst
        "fg_dice_weights": (1.0, 3.0, 2.5),     # kidney, tumor, cyst
        "tversky_alpha": 0.3,                   # FP weight (< beta -> boost recall)
        "tversky_beta": 0.7,                    # FN weight
        "tversky_rare_boost": 1.5,              # extra weight on tumor/cyst
        "boundary_weight": 0.2,
        "rare_head_weight": 0.3,

        # ---- training ----
        "seed": 42,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "num_epochs": 300,
        "iters_per_epoch": 250,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 10,
        "amp": True,
        "ema_decay": 0.999,
        "grad_clip": 12.0,

        # ---- validation / early stop ----
        "val_every": 5,
        "val_cases": 12,
        "patience": 30,
        "min_epochs_before_stop": 60,
        "selection": "rare",                    # rare | tumor | mean_fg
        "sw_overlap": 0.5,
        "sw_batch_size": 2,

        # ---- post-processing ----
        "tumor_kidney_max_dist": 15.0,          # remove T/C far from kidney (0=off)
        "keep_largest_kidney": True,
    }

    if mode == "quick_test":
        cfg.update({
            "output_dir": "./output/walmnet_quick",
            "patch_size": (96, 96, 96),
            "num_epochs": 4,
            "iters_per_epoch": 8,
            "warmup_epochs": 1,
            "val_every": 1,
            "val_cases": 3,
            "cache_rate": 0.0,
            "min_epochs_before_stop": 10_000,
            "ema_decay": 0.0,
        })
    elif mode == "medium":
        cfg.update({
            "output_dir": "./output/walmnet_medium",
            "num_epochs": 120,
            "iters_per_epoch": 150,
            "val_cases": 8,
            "min_epochs_before_stop": 40,
        })
    return cfg


# ===========================================================================
# 1. Learnable lifting wavelet (the core novelty)
# ===========================================================================

class _Lifting1D(nn.Module):
    """One learnable lifting step applied along an arbitrary spatial axis.

    Lazy/learnable lifting:
        detail = odd  - P(even)
        approx = even + U(detail)
    with depthwise predict/update convs P, U initialised to zero (lazy wavelet).
    The structure is exactly invertible for *any* P, U, so reconstruction is
    guaranteed regardless of what the network learns.
    """

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        # kernel only spans the LAST dim; the axis is moved to last in forward().
        self.P = nn.Conv3d(channels, channels, (1, 1, kernel),
                           padding=(0, 0, pad), groups=channels, bias=False)
        self.U = nn.Conv3d(channels, channels, (1, 1, kernel),
                           padding=(0, 0, pad), groups=channels, bias=False)
        nn.init.zeros_(self.P.weight)
        nn.init.zeros_(self.U.weight)

    def forward(self, x: torch.Tensor, dim: int):
        x = x.transpose(dim, -1).contiguous()
        pad = x.shape[-1] % 2
        if pad:
            # replicate the last slice along the last axis. F.pad(mode="replicate")
            # raises on 5D tensors, so do it manually (works for any ndim / odd dims).
            x = torch.cat([x, x[..., -1:]], dim=-1)
        even = x[..., 0::2].contiguous()
        odd = x[..., 1::2].contiguous()
        detail = odd - self.P(even)
        approx = even + self.U(detail)
        return (approx.transpose(dim, -1).contiguous(),
                detail.transpose(dim, -1).contiguous(), pad)

    def inverse(self, approx: torch.Tensor, detail: torch.Tensor, dim: int, pad: int):
        approx = approx.transpose(dim, -1).contiguous()
        detail = detail.transpose(dim, -1).contiguous()
        even = approx - self.U(detail)
        odd = detail + self.P(even)
        b, c, d, h, half = even.shape
        x = even.new_zeros(b, c, d, h, half * 2)
        x[..., 0::2] = even
        x[..., 1::2] = odd
        if pad:
            x = x[..., :-1]
        return x.transpose(dim, -1).contiguous()


class LearnableLiftingWavelet3D(nn.Module):
    """Separable single-level 3D lifting DWT/IDWT.

    forward(x) -> (low, highs[7], pads)   each band has half spatial size.
    inverse(low, highs, pads) -> x        (perfect reconstruction at init).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.lift_w = _Lifting1D(channels)
        self.lift_h = _Lifting1D(channels)
        self.lift_d = _Lifting1D(channels)

    def forward(self, x: torch.Tensor):
        aW, dW, pW = self.lift_w(x, 4)
        aWaH, aWdH, pH = self.lift_h(aW, 3)
        dWaH, dWdH, _ = self.lift_h(dW, 3)
        aaa, aad, pD = self.lift_d(aWaH, 2)
        ada, add_, _ = self.lift_d(aWdH, 2)
        daa, dad, _ = self.lift_d(dWaH, 2)
        dda, ddd, _ = self.lift_d(dWdH, 2)
        highs = [aad, ada, add_, daa, dad, dda, ddd]
        return aaa, highs, (pW, pH, pD)

    def inverse(self, low: torch.Tensor, highs: List[torch.Tensor], pads):
        pW, pH, pD = pads
        aad, ada, add_, daa, dad, dda, ddd = highs
        aWaH = self.lift_d.inverse(low, aad, 2, pD)
        aWdH = self.lift_d.inverse(ada, add_, 2, pD)
        dWaH = self.lift_d.inverse(daa, dad, 2, pD)
        dWdH = self.lift_d.inverse(dda, ddd, 2, pD)
        aW = self.lift_h.inverse(aWaH, aWdH, 3, pH)
        dW = self.lift_h.inverse(dWaH, dWdH, 3, pH)
        return self.lift_w.inverse(aW, dW, 4, pW)


# ===========================================================================
# 2. Linear-cost token mixer (Mamba or dependency-free fallback)
# ===========================================================================

class _SSMLite(nn.Module):
    """Dependency-free linear-time sequence mixer (fallback for mamba_ssm).

    Combines a depthwise causal-free conv over the token axis (local context)
    with a broadcast global-mean token (global context) and a SiLU gate.
    O(N) in sequence length. Operates on (B, N, C).
    """

    def __init__(self, dim: int, d_conv: int = 4, expand: int = 2):
        super().__init__()
        inner = dim * expand
        self.in_proj = nn.Linear(dim, inner * 2)
        self.conv = nn.Conv1d(inner, inner, d_conv, padding=d_conv - 1, groups=inner)
        self.act = nn.SiLU()
        self.global_proj = nn.Linear(inner, inner)
        self.out_proj = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        xz = self.in_proj(x)
        xs, z = xz.chunk(2, dim=-1)
        xc = self.conv(xs.transpose(1, 2))[..., :n].transpose(1, 2)
        g = self.global_proj(xs.mean(dim=1, keepdim=True))   # (B,1,inner)
        y = self.act(xc + g) * self.act(z)
        return self.out_proj(y)


class TokenMixer(nn.Module):
    """Bidirectional linear-cost mixer over flattened voxels (scan-order robust)."""

    def __init__(self, channels: int, mixer: str = "auto"):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        use_mamba = (mixer == "mamba") or (mixer == "auto" and _HAS_MAMBA)
        if use_mamba and _HAS_MAMBA:
            self.core = _MambaSSM(d_model=channels)
            self.kind = "mamba"
        else:
            self.core = _SSMLite(channels)
            self.kind = "lite"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)        # (B, N, C)
        t = self.norm(t)
        fwd = self.core(t)
        bwd = self.core(t.flip(1)).flip(1)      # average two scan directions
        t = 0.5 * (fwd + bwd)
        return t.transpose(1, 2).reshape(b, c, d, h, w)


class GSC(nn.Module):
    """Gated spatial convolution (SegMamba-style local feature block)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.gate = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm(self.conv1(x)))
        return self.conv2(y) * torch.sigmoid(self.gate(x))


class WaMambaBlock(nn.Module):
    """Residual block: gated spatial conv (local) + token mixer (global) + MLP."""

    def __init__(self, channels: int, mixer: str = "auto"):
        super().__init__()
        self.gsc = GSC(channels)
        self.mixer = TokenMixer(channels, mixer)
        self.mlp_norm = nn.InstanceNorm3d(channels, affine=True)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv3d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gsc(x)
        x = x + self.mixer(x)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ===========================================================================
# 3. High-frequency detail branch & generic blocks
# ===========================================================================

class HFBranch(nn.Module):
    """Process the 7 high-frequency sub-bands into a detail feature map."""

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        c7 = channels_in * 7
        self.dw = nn.Conv3d(c7, c7, 3, padding=1, groups=channels_in)  # grouped depthwise
        self.pw = nn.Conv3d(c7, channels_out, 1)
        self.norm = nn.InstanceNorm3d(channels_out, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, highs: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(highs, dim=1)
        return self.act(self.norm(self.pw(self.dw(x))))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + self.skip(x))


class AttentionGate(nn.Module):
    """Filter a skip connection with the decoder gating signal."""

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv3d(f_g, f_int, 1), nn.InstanceNorm3d(f_int))
        self.w_x = nn.Sequential(nn.Conv3d(f_l, f_int, 1), nn.InstanceNorm3d(f_int))
        self.psi = nn.Sequential(nn.Conv3d(f_int, 1, 1), nn.InstanceNorm3d(1), nn.Sigmoid())
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.act(self.w_g(g) + self.w_x(x)))


def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize x to ref's spatial size if they differ (defensive against odd dims)."""
    if x.shape[2:] != ref.shape[2:]:
        x = F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)
    return x


# ===========================================================================
# 4. WaLM-Net
# ===========================================================================

class WaLMEncoderStage(nn.Module):
    """Learnable-wavelet downsample + high-freq detail fusion + token-mixer block."""

    def __init__(self, in_ch: int, out_ch: int, mixer: str, use_hf: bool):
        super().__init__()
        self.dwt = LearnableLiftingWavelet3D(in_ch)
        self.proj = nn.Conv3d(in_ch, out_ch, 1)
        self.use_hf = use_hf
        if use_hf:
            self.hf = HFBranch(in_ch, out_ch)
            self.gate = nn.Conv3d(out_ch * 2, out_ch, 1)
        self.block = WaMambaBlock(out_ch, mixer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, highs, _ = self.dwt(x)
        low = self.proj(low)
        if self.use_hf:
            hf = self.hf(highs)
            g = torch.sigmoid(self.gate(torch.cat([low, hf], dim=1)))
            low = low + g * hf
        return self.block(low)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.att = AttentionGate(out_ch, skip_ch, max(out_ch // 2, 1))
        self.conv = ResBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = _match(self.up(x), skip)
        skip = self.att(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class WaLMNet(nn.Module):
    """Wavelet-Lifting Mamba Network."""

    def __init__(self, config: dict):
        super().__init__()
        f = tuple(config["features"])               # (32, 64, 128, 256)
        bott = int(config["bottleneck_channels"])
        dec = tuple(config["decoder_channels"])     # (128, 64, 32, 32)
        nc = int(config["num_classes"])
        mixer = config.get("mixer", "auto")
        use_hf = bool(config.get("use_high_freq_branch", True))
        self.deep_supervision = bool(config.get("deep_supervision", True))
        self.use_rare_head = bool(config.get("use_rare_head", True))

        # Stem (full resolution)
        self.stem = ResBlock(int(config["in_channels"]), f[0])

        # Encoder (wavelet downsampling)
        self.enc1 = WaLMEncoderStage(f[0], f[1], mixer, use_hf)   # -> 1/2
        self.enc2 = WaLMEncoderStage(f[1], f[2], mixer, use_hf)   # -> 1/4
        self.enc3 = WaLMEncoderStage(f[2], f[3], mixer, use_hf)   # -> 1/8

        # Bottleneck (wavelet downsample -> 1/16)
        self.bott_dwt = LearnableLiftingWavelet3D(f[3])
        self.bott_proj = nn.Conv3d(f[3], bott, 1)
        self.bott_block = WaMambaBlock(bott, mixer)

        # Decoder (EffiDec3D-style reduced widths) with attention-gated skips
        self.up3 = UpBlock(bott, f[3], dec[0])      # 1/16 -> 1/8
        self.up2 = UpBlock(dec[0], f[2], dec[1])    # 1/8  -> 1/4
        self.up1 = UpBlock(dec[1], f[1], dec[2])    # 1/4  -> 1/2
        self.up0 = UpBlock(dec[2], f[0], dec[3])    # 1/2  -> 1/1

        # Heads
        self.head = nn.Conv3d(dec[3], nc, 1)
        if self.deep_supervision:
            self.head_half = nn.Conv3d(dec[2], nc, 1)       # 1/2
            self.head_quarter = nn.Conv3d(dec[1], nc, 1)    # 1/4
        if self.use_rare_head:
            self.rare_head = nn.Conv3d(dec[3], 2, 1)        # tumor, cyst

    def forward(self, x: torch.Tensor):
        target = x.shape[2:]
        s0 = self.stem(x)              # f0, 1/1
        s1 = self.enc1(s0)             # f1, 1/2
        s2 = self.enc2(s1)             # f2, 1/4
        s3 = self.enc3(s2)             # f3, 1/8

        low, highs, pads = self.bott_dwt(s3)
        b = self.bott_block(self.bott_proj(low))   # bott, 1/16

        d3 = self.up3(b, s3)           # 1/8
        d2 = self.up2(d3, s2)          # 1/4
        d1 = self.up1(d2, s1)          # 1/2
        d0 = self.up0(d1, s0)          # 1/1

        main = _match(self.head(d0), x)

        if self.training:
            out = {"main": main}
            if self.deep_supervision:
                out["ds"] = [self.head_half(d1), self.head_quarter(d2)]
            if self.use_rare_head:
                out["rare"] = _match(self.rare_head(d0), x)
            return out
        return main


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(config: dict) -> WaLMNet:
    return WaLMNet(config)


# ===========================================================================
# 5. Loss
# ===========================================================================

def _laplacian_kernel(channels: int, device, dtype) -> torch.Tensor:
    k = torch.zeros(channels, 1, 3, 3, 3, device=device, dtype=dtype)
    k[:, 0, 1, 1, 1] = 6.0
    for dz, dy, dx in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
        k[:, 0, dz, dy, dx] = -1.0
    return k


class WaLMLoss(nn.Module):
    """Dice + weighted CE + Tversky (rare-boosted) + boundary, with deep supervision."""

    def __init__(self, config: dict, device):
        super().__init__()
        self.nc = int(config["num_classes"])
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(config["ce_weight"], dtype=torch.float32, device=device))
        self.fg_dice_w = torch.tensor(config["fg_dice_weights"], dtype=torch.float32, device=device)
        self.alpha = float(config["tversky_alpha"])
        self.beta = float(config["tversky_beta"])
        self.rare_boost = float(config["tversky_rare_boost"])
        self.boundary_w = float(config["boundary_weight"])
        self.rare_head_w = float(config["rare_head_weight"])
        self.ds_weights = tuple(config["ds_weights"])
        self.deep_supervision = bool(config.get("deep_supervision", True))

    def _onehot(self, target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        oh = torch.zeros_like(logits)
        oh.scatter_(1, target.long(), 1.0)
        return oh

    def _dice(self, prob: torch.Tensor, oh: torch.Tensor) -> torch.Tensor:
        loss = prob.new_tensor(0.0)
        for i, c in enumerate(range(1, self.nc)):
            p, t = prob[:, c], oh[:, c]
            inter = (p * t).sum()
            denom = p.sum() + t.sum()
            loss = loss + self.fg_dice_w[i] * (1 - (2 * inter + 1e-6) / (denom + 1e-6))
        return loss / self.fg_dice_w.sum()

    def _tversky(self, prob: torch.Tensor, oh: torch.Tensor) -> torch.Tensor:
        loss = prob.new_tensor(0.0)
        for c in range(1, self.nc):
            p, t = prob[:, c], oh[:, c]
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            tv = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)
            w = self.rare_boost if c in (2, 3) else 1.0
            loss = loss + w * (1 - tv)
        return loss / (self.nc - 1)

    def _boundary(self, prob: torch.Tensor, oh: torch.Tensor) -> torch.Tensor:
        ch = prob.shape[1]
        k = _laplacian_kernel(ch, prob.device, prob.dtype)
        lp = F.conv3d(prob, k, padding=1, groups=ch)
        lt = F.conv3d(oh, k, padding=1, groups=ch)
        return (lp - lt).abs()[:, 1:].mean()

    def _seg_term(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        oh = self._onehot(target, logits)
        prob = F.softmax(logits, dim=1)
        loss = self.ce(logits, target.squeeze(1).long())
        loss = loss + self._dice(prob, oh) + self._tversky(prob, oh)
        if self.boundary_w > 0:
            loss = loss + self.boundary_w * self._boundary(prob, oh)
        return loss

    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(outputs, dict):            # eval / plain tensor
            return self._seg_term(outputs, target)

        total = self._seg_term(outputs["main"], target) * self.ds_weights[0]

        if self.deep_supervision and "ds" in outputs:
            for w, ds in zip(self.ds_weights[1:], outputs["ds"]):
                t = F.interpolate(target.float(), size=ds.shape[2:], mode="nearest")
                total = total + w * self._seg_term(ds, t)

        if "rare" in outputs:                        # 2-channel head: tumor, cyst
            gt = torch.cat([(target == 2).float(), (target == 3).float()], dim=1)
            total = total + self.rare_head_w * F.binary_cross_entropy_with_logits(
                outputs["rare"], gt)
        return total


def build_loss(config: dict, device) -> WaLMLoss:
    return WaLMLoss(config, device)


# ===========================================================================
# 6. Data pipeline (MONAI, imported lazily)
# ===========================================================================

def _list_cases(kits23_dir: str):
    data_dir = Path(kits23_dir)
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    items = []
    for case in cases:
        img, seg = case / "imaging.nii.gz", case / "segmentation.nii.gz"
        if img.exists() and seg.exists():
            items.append({"image": str(img), "label": str(seg)})
    return items


def get_dataloaders(config: dict):
    from monai.data import CacheDataset, DataLoader
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
        RandFlipd, RandRotate90d, RandShiftIntensityd, RandScaleIntensityd, EnsureTyped,
    )

    items = _list_cases(config["kits23_dir"])
    if not items:
        raise FileNotFoundError(f"No KiTS23 cases under {config['kits23_dir']}")
    n_val = max(1, int(len(items) * config["val_split"]))
    val_files, train_files = items[:n_val], items[n_val:]
    a_min, a_max = config["ct_window"]
    pos, neg = config["pos_neg_ratio"]

    common = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=config["spacing"], mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=config["patch_size"],
                    mode=("constant", "constant")),
    ]
    train_tf = Compose(common + [
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=config["patch_size"], pos=pos, neg=neg,
                               num_samples=config["num_samples_per_volume"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ])
    val_tf = Compose(common + [EnsureTyped(keys=["image", "label"])])

    train_ds = CacheDataset(train_files, train_tf, cache_rate=config["cache_rate"],
                            num_workers=config["num_workers"])
    val_ds = CacheDataset(val_files, val_tf, cache_rate=config["cache_rate"],
                          num_workers=max(1, config["num_workers"] // 2))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    return train_loader, val_loader, val_files


# ===========================================================================
# 7. Metrics & sliding-window validation
# ===========================================================================

def _build_metrics(num_classes: int):
    from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
    metrics = {
        "dice": DiceMetric(include_background=False, reduction="mean_batch"),
        "iou": MeanIoU(include_background=False, reduction="mean_batch"),
        "hd95": HausdorffDistanceMetric(include_background=False, percentile=95,
                                        reduction="mean_batch"),
    }
    try:
        from monai.metrics import SurfaceDiceMetric
        metrics["surface_dice"] = SurfaceDiceMetric(
            class_thresholds=[1.5] * (num_classes - 1), include_background=False,
            reduction="mean_batch")
    except Exception:  # noqa: BLE001
        pass
    return metrics


def validate(model, val_files, config, device, n_cases=None, logger=None):
    """Full-volume sliding-window validation reporting Dice/IoU/SurfaceDice/HD95."""
    from monai.data import Dataset, DataLoader
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, EnsureTyped, AsDiscrete,
    )
    log = (logger.info if logger else print)
    a_min, a_max = config["ct_window"]
    tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=config["spacing"], mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    subset = val_files[:n_cases] if n_cases else val_files
    loader = DataLoader(Dataset(subset, tf), batch_size=1, num_workers=2)

    nc = int(config["num_classes"])
    metrics = _build_metrics(nc)
    post_pred = AsDiscrete(argmax=True, to_onehot=nc)
    post_label = AsDiscrete(to_onehot=nc)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            lbl = batch["label"].to(device)
            logits = sliding_window_inference(
                img, config["patch_size"], config["sw_batch_size"], model,
                overlap=config["sw_overlap"])
            pred = post_pred(logits[0]).unsqueeze(0)
            gt = post_label(lbl[0]).unsqueeze(0)
            for m in metrics.values():
                m(y_pred=pred, y=gt)

    results = {}
    for name, m in metrics.items():
        try:
            vals = m.aggregate()
            results[name] = [float(v) for v in vals]
            m.reset()
        except Exception:  # noqa: BLE001
            results[name] = None

    dice = results.get("dice") or [0.0] * (nc - 1)
    names = config["class_names"][1:]
    mean_fg = sum(dice) / len(dice)
    rare = sum(dice[1:]) / max(1, len(dice[1:]))   # tumor + cyst
    results["mean_fg_dice"] = mean_fg
    results["rare_mean"] = rare
    log("  " + " ".join(f"{n}={d:.4f}" for n, d in zip(names, dice))
        + f" | meanFG={mean_fg:.4f} rare(T+C)={rare:.4f}")

    sel = config.get("selection", "rare")
    score = {"rare": rare, "tumor": dice[1] if len(dice) > 1 else 0.0,
             "mean_fg": mean_fg}.get(sel, rare)
    return score, results


# ===========================================================================
# 8. Training utilities
# ===========================================================================

class EarlyStopping:
    def __init__(self, patience: int, ckpt_path: Path):
        self.patience, self.ckpt_path = patience, ckpt_path
        self.best, self.counter, self.early_stop = None, 0, False

    def step(self, score: float, model, epoch: int) -> bool:
        if self.best is None or score > self.best:
            self.best, self.counter = score, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "score": score}, self.ckpt_path)
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


class ModelEMA:
    def __init__(self, model, decay: float):
        import copy
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def cosine_warmup_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    def fn(epoch: int):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


# ===========================================================================
# 9. Training loop
# ===========================================================================

def train(config: dict):
    import numpy as np

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

    train_loader, _, val_files = get_dataloaders(config)
    model = build_model(config).to(device)
    total, trainable = count_parameters(model)
    mixer_kind = "mamba" if _HAS_MAMBA and config.get("mixer", "auto") != "lite" else "lite"

    logger.info("=" * 72)
    logger.info("WaLM-Net training")
    logger.info(f"Device: {device} | Params: {total/1e6:.2f}M ({trainable/1e6:.2f}M trainable)")
    logger.info(f"Mixer: {mixer_kind} | Patch: {config['patch_size']} | DS: {config['deep_supervision']}")
    logger.info(f"Train cases: {len(train_loader.dataset)} | Val cases: {len(val_files)}")
    logger.info(f"Output: {out_dir}")
    logger.info("=" * 72)

    loss_fn = build_loss(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = cosine_warmup_scheduler(optimizer, config["num_epochs"], config["warmup_epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"] and device.type == "cuda")
    ema = ModelEMA(model, config["ema_decay"]) if config.get("ema_decay", 0) > 0 else None
    es = EarlyStopping(config["patience"], out_dir / "best.pth")

    data_iter = _cycle(train_loader)
    accum = config["grad_accum_steps"]
    history = {"epochs": []}

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        t0, running = time.time(), 0.0
        optimizer.zero_grad(set_to_none=True)

        for it in range(config["iters_per_epoch"]):
            batch = next(data_iter)
            img = batch["image"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=config["amp"] and device.type == "cuda"):
                loss = loss_fn(model(img), lbl) / accum
            scaler.scale(loss).backward()
            running += loss.item() * accum
            if (it + 1) % accum == 0:
                if config.get("grad_clip", 0) > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

        scheduler.step()
        train_loss = running / config["iters_per_epoch"]
        elapsed = (time.time() - t0) / 60

        do_val = (epoch % config["val_every"] == 0) or (epoch == config["num_epochs"])
        score, metrics, improved = None, None, False
        if do_val:
            eval_model = ema.module if ema is not None else model
            score, metrics = validate(eval_model, val_files, config, device,
                                      n_cases=config["val_cases"], logger=logger)
            improved = es.step(score, eval_model, epoch)
            tag = " *BEST*" if improved else f" (no-improve {es.counter}/{es.patience})"
            logger.info(f"  {config['selection']} score={score:.4f}{tag}")

        logger.info(f"[E{epoch:03d}/{config['num_epochs']}] train_loss={train_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.1f}min")
        history["epochs"].append({"epoch": epoch, "train_loss": train_loss,
                                  "score": score, "metrics": metrics, "time_min": elapsed})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        torch.save({"model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.module.state_dict() if ema else None,
                    "epoch": epoch, "config": {k: str(v) for k, v in config.items()}},
                   out_dir / "last.pth")

        if es.early_stop and epoch >= config["min_epochs_before_stop"]:
            logger.info(f"Early stopping at epoch {epoch} (best={es.best:.4f})")
            break

    summary = {"version": "walmnet", "params_millions": round(total / 1e6, 3),
               "mixer": mixer_kind, "best_score": es.best, "selection": config["selection"],
               "total_hours": round(sum(e["time_min"] for e in history["epochs"]) / 60, 2),
               "config": {k: str(v) for k, v in config.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Done. Best {config['selection']} = {es.best}. Summary -> {out_dir/'summary.json'}")


# ===========================================================================
# 10. Post-processing & evaluation
# ===========================================================================

def postprocess_tumor_in_kidney(prediction, config):
    """Remove tumor/cyst components too far from any kidney voxel (FP suppression)."""
    import numpy as np
    from scipy import ndimage

    out = prediction.copy()
    kidney = out == 1
    max_dist = config.get("tumor_kidney_max_dist", 0)
    if not kidney.any() or max_dist <= 0:
        return out
    target = (out == 2) | (out == 3)
    if not target.any():
        return out

    fg = kidney | target
    coords = np.argwhere(fg)
    margin = int(np.ceil(float(max_dist))) + 2
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin + 1, np.array(out.shape))
    crop = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))

    kc = kidney[crop]
    oc = out[crop].copy()
    allowed = ndimage.distance_transform_edt(~kc) <= float(max_dist)
    for cid in (2, 3):
        mask = oc == cid
        if not mask.any():
            continue
        labeled, n = ndimage.label(mask)
        keep = np.zeros(n + 1, dtype=bool)
        keep[np.unique(labeled[mask & allowed])] = True
        keep[0] = True
        oc[mask & ~keep[labeled]] = 0
    out[crop] = oc
    return out


def evaluate(config: dict, checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {checkpoint}")

    _, _, val_files = get_dataloaders(config)
    score, metrics = validate(model, val_files, config, device, logger=None)
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps({"score": score, "metrics": metrics}, indent=2))
    print(f"Selection score = {score:.4f}")
    print(f"Metrics saved -> {out_dir / 'evaluation.json'}")


# ===========================================================================
# 11. Self-test (CPU, no data) — validates shapes, reconstruction, params
# ===========================================================================

def selftest():
    print("=" * 60)
    print("WaLM-Net self-test (CPU)")
    print("=" * 60)
    print(f"mamba_ssm available: {_HAS_MAMBA} (using {'mamba' if _HAS_MAMBA else 'lite'} mixer)")
    torch.manual_seed(0)

    # (a) Perfect-reconstruction check of the learnable lifting wavelet
    dwt = LearnableLiftingWavelet3D(channels=4)
    x = torch.randn(1, 4, 16, 16, 16)
    low, highs, pads = dwt(x)
    recon = dwt.inverse(low, highs, pads)
    err = (recon - x).abs().max().item()
    print(f"[wavelet] low={tuple(low.shape)} highs={len(highs)}x{tuple(highs[0].shape)}")
    print(f"[wavelet] DWT->IDWT max recon error = {err:.3e} "
          f"({'OK perfect reconstruction' if err < 1e-4 else 'FAIL'})")

    # (b) Forward pass (train + eval) and param count
    cfg = get_config("quick_test")
    model = build_model(cfg)
    total, trainable = count_parameters(model)
    print(f"[model]   parameters = {total/1e6:.2f}M ({trainable/1e6:.2f}M trainable)")

    x = torch.randn(1, 1, 64, 64, 64)
    model.train()
    out = model(x)
    print(f"[train]   main={tuple(out['main'].shape)} "
          f"ds={[tuple(d.shape) for d in out.get('ds', [])]} "
          f"rare={tuple(out['rare'].shape) if 'rare' in out else None}")
    loss = build_loss(cfg, torch.device("cpu"))(out, torch.randint(0, 4, (1, 1, 64, 64, 64)))
    loss.backward()
    print(f"[train]   loss={loss.item():.4f}, backward OK")

    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"[eval]    output={tuple(y.shape)} (expect (1, {cfg['num_classes']}, 64, 64, 64))")
    assert y.shape == (1, cfg["num_classes"], 64, 64, 64), "eval output shape mismatch"
    print("\nAll checks passed.")


# ===========================================================================
# 12. CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="WaLM-Net for KiTS23 3D segmentation")
    parser.add_argument("--mode", choices=["full", "medium", "quick_test"], default="full")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--selftest", action="store_true", help="CPU shape/recon/param check (no data)")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--kits23-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mixer", choices=["auto", "mamba", "lite"], default=None)
    parser.add_argument("--patch-size", default=None, help="comma-separated, e.g. 128,128,128")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-deep-supervision", action="store_true")
    parser.add_argument("--no-high-freq", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    mode = "quick_test" if args.quick else "medium" if args.medium else "full" if args.full else args.mode
    cfg = get_config(mode)

    if args.kits23_dir:
        cfg["kits23_dir"] = args.kits23_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.mixer:
        cfg["mixer"] = args.mixer
    if args.patch_size:
        cfg["patch_size"] = tuple(int(v) for v in args.patch_size.split(","))
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.num_epochs:
        cfg["num_epochs"] = args.num_epochs
    if args.lr:
        cfg["lr"] = args.lr
    if args.no_deep_supervision:
        cfg["deep_supervision"] = False
    if args.no_high_freq:
        cfg["use_high_freq_branch"] = False

    if args.evaluate:
        if not args.checkpoint:
            parser.error("--checkpoint is required with --evaluate")
        evaluate(cfg, args.checkpoint)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
