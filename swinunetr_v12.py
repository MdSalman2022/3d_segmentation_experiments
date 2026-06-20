"""
SwinUNETR V12 -- DINOv3 + VISTA-style tumor-focused KiTS23 segmentation.

V12 combines the strongest ideas we found:
  - V4: frozen pretrained 2D foundation encoder, tumor-focused sampling/loss,
        larger in-plane patches, attention decoder.
  - V11: cache/restart support, component index cache, rare-class full-volume
        sliding-window validation, rescue thresholds, robust CLI.

Important:
  - This file reuses the mature training/evaluation infrastructure from
    swinunetr_v11.py and replaces the model/config with a DINOv3-VISTA variant.
  - HuggingFace Transformers is the primary DINOv3 path, matching dermavit_v16.
  - The cloned facebookresearch/dinov3 repo path remains available only as an
    explicit local fallback via --local-dinov3.

Recommended:
  python swinunetr_v12.py --quick --no-resume
  python swinunetr_v12.py --build-cache
  python swinunetr_v12.py --full
  python swinunetr_v12.py --full --local-dinov3 --dinov3-weights ./dinov3_weights/dinov3_vitb16.pth
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import swinunetr_v11 as base

_BASE_BUILD_MODEL = base.build_model
_BASE_VALIDATE_SLIDING_WINDOW = base.validate_sliding_window


def _find_dinov3_weights(model_name: str) -> Optional[str]:
    candidates = [
        f"./dinov3_weights/{model_name}.pth",
        f"./checkpoints/{model_name}.pth",
        f"./{model_name}.pth",
        "./dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "./dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.stat().st_size > 1024 * 1024:
            return str(path)
    return None


def get_config(mode: str) -> dict:
    cfg = base.get_config(mode)
    cfg.update({
        "output_dir": "./output/swin_v12_dinov3_tumor",
        "arch": "dinov3_vista",
        "patch_size": (128, 224, 224),
        "batch_size": 1,
        "grad_accum_steps": 4,
        "num_epochs": 120,
        "iters_per_epoch": 160,
        "warmup_epochs": 6,
        "patience": 10,
        "val_every": 10,
        "sw_val_cases": 40,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        # V4-inspired tumor focus: tumor receives the strongest pressure.
        "class_sampling_bias": (0.15, 0.50, 0.25, 0.10),  # kidney, tumor, cyst, random
        "quota_classes": (2, 3),
        "ce_weight": (0.1, 1.0, 15.0, 10.0),
        "fg_dice_weights": (1.0, 4.0, 2.0),
        "tversky_alpha": 0.15,
        "tversky_beta": 0.85,
        "focal_gamma": 2.5,
        "selection": "tumor",
        "rescue_tau": (0.0, 0.0, 0.25, 0.30),
        "class_keep_min_voxels": (0, 1000, 20, 10),
        # DINOv3 model options
        "dinov3_source": "hf",
        "dinov3_repo": "./dinov3",
        "dinov3_model": "dinov3_vitb16",
        "dinov3_hf_model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "dinov3_weights": None,
        "dinov3_freeze": True,
        "dinov3_slice_batch": 8,
        "dinov3_feature_dim": 256,
        "dinov3_layers": (5, 8, 11),
        "dinov3_hf_layers": (-9, -5, -1),
        "dinov3_patch_size": 16,
        "dinov3_norm": "imagenet",
    })
    if mode == "quick_test":
        cfg.update({
            "output_dir": "./output/swin_v12_dinov3_tumor_quick",
            "patch_size": (32, 112, 112),
            "num_epochs": 2,
            "iters_per_epoch": 8,
            "grad_accum_steps": 1,
            "sw_val_cases": 2,
            "dinov3_model": "dinov3_vits16",
            "dinov3_hf_model": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "dinov3_slice_batch": 4,
            "use_tta": False,
        })
    elif mode == "medium":
        cfg.update({
            "output_dir": "./output/swin_v12_dinov3_tumor_medium",
            "num_epochs": 80,
            "iters_per_epoch": 140,
            "sw_val_cases": 20,
            "use_tta": False,
        })
    return cfg


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class DINOv3SliceEncoder(nn.Module):
    """Frozen 2D DINOv3 slice encoder projected into 3D feature volumes.

    Local mode uses facebookresearch/dinov3 and get_intermediate_layers().
    HF mode mirrors dermavit_v16's AutoModel.from_pretrained() path, but uses
    patch tokens instead of CLS so the decoder still receives spatial maps.
    """

    def __init__(
        self,
        repo_path: str,
        model_name: str,
        weights_path: Optional[str],
        feature_dim: int,
        layers: tuple[int, ...],
        hf_model: str,
        hf_layers: tuple[int, ...],
        source: str,
        patch_size: int,
        freeze: bool,
        slice_batch: int,
        input_norm: str = "imagenet",
    ):
        super().__init__()
        self.source = source
        self.layers = tuple(hf_layers if source == "hf" else layers)
        self.patch_size = int(patch_size)
        self.slice_batch = int(slice_batch)
        self.input_norm = input_norm

        if source == "hf":
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "DINOv3 HF mode needs transformers installed. "
                    "Use --dinov3-source local or install transformers in this environment."
                ) from exc

            print(f"  DINOv3 HF: loading {hf_model}")
            self.vit = AutoModel.from_pretrained(hf_model, ignore_mismatched_sizes=True)
            self.pretrained = True
            self.embed_dim = int(getattr(self.vit.config, "hidden_size", 1024))
            self.patch_size = int(getattr(self.vit.config, "patch_size", self.patch_size))
        else:
            repo = Path(repo_path).resolve()
            if repo.exists() and str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from dinov3.hub import backbones

            factory = getattr(backbones, model_name)
            weights = weights_path or _find_dinov3_weights(model_name)
            if weights:
                print(f"  DINOv3 local: loading {model_name} pretrained weights from {weights}")
                self.vit = factory(pretrained=True, weights=weights)
                self.pretrained = True
            else:
                print(
                    f"  DINOv3 local: no cached weights found for {model_name}; "
                    "downloading public pretrained weights from Meta."
                )
                try:
                    self.vit = factory(pretrained=True)
                    self.pretrained = True
                except Exception as exc:
                    print(
                        "  WARNING: public DINOv3 weight download failed; "
                        f"falling back to random init. Reason: {exc}"
                    )
                    self.vit = factory(pretrained=False)
                    self.pretrained = False
            self.embed_dim = int(getattr(self.vit, "embed_dim", getattr(self.vit, "num_features", 768)))
            self.patch_size = int(getattr(self.vit, "patch_size", self.patch_size))

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
        self.freeze = freeze

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, feature_dim, 1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.GELU(),
            )
            for _ in self.layers
        ])
        self.fusion = nn.Sequential(
            nn.Conv3d(feature_dim * len(self.layers), feature_dim, 1, bias=False),
            nn.InstanceNorm3d(feature_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(feature_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def _normalize_slices(self, x2d: torch.Tensor) -> torch.Tensor:
        if self.input_norm == "imagenet":
            mean = x2d.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x2d.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            return (x2d - mean) / std
        if self.input_norm == "zscore":
            mean = x2d.mean(dim=(2, 3), keepdim=True)
            std = x2d.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            return (x2d - mean) / std
        return x2d

    def _local_features(self, chunk: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.vit.get_intermediate_layers(chunk, n=self.layers, reshape=True)

    def _hf_features(self, chunk: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.vit(pixel_values=chunk, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        grid_h = chunk.shape[-2] // self.patch_size
        grid_w = chunk.shape[-1] // self.patch_size
        n_patches = grid_h * grid_w
        feats = []
        for layer in self.layers:
            tokens = hidden_states[layer]
            # DINOv3 HF outputs CLS/register tokens plus patch tokens. The final
            # grid_h*grid_w tokens are the spatial patch sequence we need here.
            patch_tokens = tokens[:, -n_patches:, :]
            feat = patch_tokens.reshape(chunk.shape[0], grid_h, grid_w, self.embed_dim)
            feats.append(feat.permute(0, 3, 1, 2).contiguous())
        return tuple(feats)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, d, h, w = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * d, 1, h, w)
        x2d = x2d.repeat(1, 3, 1, 1)
        x2d = self._normalize_slices(x2d)

        layer_outputs = [[] for _ in self.layers]
        ctx = torch.no_grad() if self.freeze else nullcontext()
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
            projected.append(feat3d)
        bottleneck = self.fusion(torch.cat(projected, dim=1))
        return {
            "bottleneck": bottleneck,
            "skip_low": projected[-2] if len(projected) >= 2 else projected[-1],
            "skip_high": projected[-1],
        }


class AnisotropicUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.skip_adapter = nn.Conv3d(skip_channels, out_channels, 1) if skip_channels else None
        conv_in = out_channels * 2 if skip_channels else out_channels
        self.refine = nn.Sequential(
            nn.Conv3d(conv_in, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.attn = SpatialAttention3D()

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None and self.skip_adapter is not None:
            skip = self.skip_adapter(skip)
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.attn(self.refine(x))


class DINOv3VISTA3D(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        fd = int(config["dinov3_feature_dim"])
        self.encoder = DINOv3SliceEncoder(
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
        )
        self.up1 = AnisotropicUpBlock(fd, 192, skip_channels=fd)
        self.up2 = AnisotropicUpBlock(192, 128, skip_channels=fd)
        self.up3 = AnisotropicUpBlock(128, 64)
        self.up4 = AnisotropicUpBlock(64, 32)
        self.head = nn.Conv3d(32, int(config["num_classes"]), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[2:]
        feats = self.encoder(x)
        y = self.up1(feats["bottleneck"], feats["skip_high"])
        y = self.up2(y, feats["skip_low"])
        y = self.up3(y)
        y = self.up4(y)
        if y.shape[2:] != target_size:
            y = F.interpolate(y, size=target_size, mode="trilinear", align_corners=False)
        return self.head(y)


def build_model(config: dict) -> nn.Module:
    if config.get("arch") == "dinov3_vista":
        return DINOv3VISTA3D(config)
    return _BASE_BUILD_MODEL(config)


def _selection_from_metrics(sw_metrics: dict, config: dict) -> float:
    dice = sw_metrics.get("dice") or [0.0, 0.0, 0.0, 0.0]
    if config.get("selection") == "tumor":
        return float(dice[2])
    if config.get("selection") == "kidney_tumor":
        return float((dice[1] + dice[2]) / 2.0)
    return float(sw_metrics.get("selection_metric", sw_metrics.get("rare_mean", 0.0)))


def validate_sliding_window(model, val_dicts, config, device, n_cases=None, logger=None):
    score, metrics = _BASE_VALIDATE_SLIDING_WINDOW(model, val_dicts, config, device, n_cases=n_cases, logger=logger)
    if config.get("selection") in {"tumor", "kidney_tumor"}:
        metrics["selection_metric"] = _selection_from_metrics(metrics, config)
        metrics["selection"] = config["selection"]
        log = logger.info if logger else print
        log(f"  Selection override ({config['selection']}): {metrics['selection_metric']:.4f}")
        return metrics["selection_metric"], metrics
    return score, metrics


def _patch_base_for_v12():
    base.build_model = build_model
    base.validate_sliding_window = validate_sliding_window


def main():
    parser = argparse.ArgumentParser(description="SwinUNETR V12 DINOv3-VISTA tumor-focused training")
    parser.add_argument("--mode", choices=["full", "medium", "quick_test", "evaluate", "build_cache", "audit"], default=None)
    parser.add_argument("--full", action="store_true", help="Run full training (recommended command: python swinunetr_v12.py --full)")
    parser.add_argument("--medium", action="store_true", help="Run medium training profile")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test profile")
    parser.add_argument("--build-cache", action="store_true", help="Build/reuse preprocessing cache and component index")
    parser.add_argument("--audit", action="store_true", help="Audit labels and split metadata")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--kits23-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-dinov3", action="store_true", help="Use local facebookresearch/dinov3 repo instead of HuggingFace AutoModel")
    parser.add_argument("--hf-dinov3", action="store_true", help="Use HuggingFace DINOv3 checkpoints instead of the local repo path")
    parser.add_argument("--hf-model", default=None, help="HuggingFace DINOv3 model id")
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

    shortcut_modes = [
        ("full", args.full),
        ("medium", args.medium),
        ("quick_test", args.quick),
        ("build_cache", args.build_cache),
        ("audit", args.audit),
        ("evaluate", args.evaluate),
    ]
    selected_shortcuts = [mode for mode, enabled in shortcut_modes if enabled]
    if len(selected_shortcuts) > 1:
        parser.error("Use only one mode shortcut, e.g. --full or --build-cache")
    mode = selected_shortcuts[0] if selected_shortcuts else (args.mode or "full")

    cfg = get_config(mode if mode not in ("evaluate", "build_cache", "audit") else "full")
    if args.local_dinov3:
        cfg["dinov3_source"] = "local"
    if args.hf_dinov3 or args.hf_model:
        cfg["dinov3_source"] = "hf"
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

    _patch_base_for_v12()
    logging.getLogger("v12").info("Patched v11 runtime with V12 DINOv3 model.")

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
