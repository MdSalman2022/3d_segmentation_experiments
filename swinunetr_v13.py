"""
SwinUNETR V13 -- DINOv3-HF + spatial adapters + CT high-res branch.

V13 keeps the V12 tumor-focused training setup, then adds two SOTA-guide ideas:
  - trainable lightweight spatial adapters on frozen DINOv3 dense patch features
  - a raw CT high-resolution branch fused into late decoder stages for small tumors

Recommended:
  python swinunetr_v13.py --quick --no-resume
  python swinunetr_v13.py --build-cache
  python swinunetr_v13.py --full
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import swinunetr_v12 as v12

base = v12.base


def get_config(mode: str) -> dict:
    cfg = v12.get_config(mode)
    cfg.update({
        "output_dir": "./output/swin_v13_dinov3_adapter_tumor",
        "arch": "dinov3_adapter_vista",
        "dinov3_source": "hf",
        "adapter_scale_init": 0.10,
        "adapter_dropout": 0.05,
        "ct_skip_channels": (32, 64),
        "experiment_name": "SwinUNETR V13 -- DINOv3 adapters + CT high-res tumor branch",
    })
    if mode == "quick_test":
        cfg["output_dir"] = "./output/swin_v13_dinov3_adapter_tumor_quick"
    elif mode == "medium":
        cfg["output_dir"] = "./output/swin_v13_dinov3_adapter_tumor_medium"
    return cfg


class SpatialAdapter3D(nn.Module):
    """Small PEFT-style adapter for dense DINOv3 patch volumes."""

    def __init__(self, channels: int, scale_init: float = 0.10, dropout: float = 0.05):
        super().__init__()
        hidden = max(32, channels // 4)
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.reduce = nn.Conv3d(channels, hidden, 1, bias=False)
        self.dw3 = nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.dw5 = nn.Conv3d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden, bias=False)
        self.mix = nn.Conv3d(hidden * 2, channels, 1, bias=False)
        self.drop = nn.Dropout3d(dropout)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.reduce(self.norm(x)))
        y = torch.cat([self.dw3(y), self.dw5(y)], dim=1)
        y = self.drop(self.mix(self.act(y)))
        return x + self.scale * y


class CTHighResBranch(nn.Module):
    """Preserves local CT intensity/detail cues that ViT patch tokens can smooth out."""

    def __init__(self, full_channels: int = 32, half_channels: int = 64):
        super().__init__()
        self.full = nn.Sequential(
            nn.Conv3d(1, full_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(full_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(full_channels, full_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(full_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.half = nn.Sequential(
            nn.Conv3d(full_channels, half_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(half_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(half_channels, half_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(half_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        full = self.full(x)
        # "half" is also a built-in nn.Module dtype-conversion method, so access
        # the registered child module directly to avoid calling Module.half().
        half = self._modules["half"](full)
        return {"full": full, "half": half}


class DINOv3AdapterSliceEncoder(v12.DINOv3SliceEncoder):
    """V12 DINOv3 slice encoder plus trainable 3D spatial adapters."""

    def __init__(self, *args, feature_dim: int, adapter_scale_init: float, adapter_dropout: float, **kwargs):
        super().__init__(*args, feature_dim=feature_dim, **kwargs)
        self.adapters = nn.ModuleList([
            SpatialAdapter3D(feature_dim, scale_init=adapter_scale_init, dropout=adapter_dropout)
            for _ in self.layers
        ])
        self.fusion_adapter = SpatialAdapter3D(feature_dim, scale_init=adapter_scale_init, dropout=adapter_dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, d, h, w = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * d, 1, h, w)
        x2d = x2d.repeat(1, 3, 1, 1)
        x2d = self._normalize_slices(x2d)

        layer_outputs = [[] for _ in self.layers]
        ctx = torch.no_grad() if self.freeze else v12.nullcontext()
        with ctx:
            for start in range(0, x2d.shape[0], self.slice_batch):
                chunk = x2d[start:start + self.slice_batch]
                feats = self._hf_features(chunk) if self.source == "hf" else self._local_features(chunk)
                for i, feat in enumerate(feats):
                    layer_outputs[i].append(feat)

        projected = []
        for i, parts in enumerate(layer_outputs):
            feat2d = torch.cat(parts, dim=0)
            feat2d = self.proj[i](feat2d)
            _, c, fh, fw = feat2d.shape
            feat3d = feat2d.reshape(b, d, c, fh, fw).permute(0, 2, 1, 3, 4).contiguous()
            projected.append(self.adapters[i](feat3d))

        bottleneck = self.fusion_adapter(self.fusion(torch.cat(projected, dim=1)))
        return {
            "bottleneck": bottleneck,
            "skip_low": projected[-2] if len(projected) >= 2 else projected[-1],
            "skip_high": projected[-1],
        }


class DINOv3AdapterVISTA3D(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        fd = int(config["dinov3_feature_dim"])
        full_ch, half_ch = tuple(config.get("ct_skip_channels", (32, 64)))
        self.encoder = DINOv3AdapterSliceEncoder(
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
        self.ct_branch = CTHighResBranch(full_channels=full_ch, half_channels=half_ch)
        self.up1 = v12.AnisotropicUpBlock(fd, 192, skip_channels=fd)
        self.up2 = v12.AnisotropicUpBlock(192, 128, skip_channels=fd)
        self.up3 = v12.AnisotropicUpBlock(128, 64, skip_channels=half_ch)
        self.up4 = v12.AnisotropicUpBlock(64, 32, skip_channels=full_ch)
        self.head = nn.Conv3d(32, int(config["num_classes"]), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[2:]
        dino = self.encoder(x)
        ct = self.ct_branch(x)
        y = self.up1(dino["bottleneck"], dino["skip_high"])
        y = self.up2(y, dino["skip_low"])
        y = self.up3(y, ct["half"])
        y = self.up4(y, ct["full"])
        if y.shape[2:] != target_size:
            y = F.interpolate(y, size=target_size, mode="trilinear", align_corners=False)
        return self.head(y)


def build_model(config: dict) -> nn.Module:
    if config.get("arch") == "dinov3_adapter_vista":
        return DINOv3AdapterVISTA3D(config)
    return v12.build_model(config)


def _patch_base_for_v13():
    base.build_model = build_model
    base.validate_sliding_window = v12.validate_sliding_window


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
    parser = argparse.ArgumentParser(description="SwinUNETR V13 DINOv3 adapter tumor-focused training")
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
    parser.add_argument("--unfreeze-dinov3", action="store_true")
    parser.add_argument("--slice-batch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--patch-size", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--iters-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--sw-val-cases", type=int, default=None)
    parser.add_argument("--sw-batch-size", type=int, default=None)
    parser.add_argument("--selection", choices=["tumor", "kidney_tumor", "rare", "fg"], default=None)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    mode = _resolve_mode(args, parser)
    cfg = get_config(mode if mode not in ("evaluate", "build_cache", "audit") else "full")
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
        cfg["dinov3_freeze"] = False
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
        cfg["val_every"] = args.val_every
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

    _patch_base_for_v13()
    logging.getLogger("v13").info("Patched runtime with V13 DINOv3 adapter model.")

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
        base.train(cfg, resume_path=args.resume, skip_resume=args.no_resume)


if __name__ == "__main__":
    main()
