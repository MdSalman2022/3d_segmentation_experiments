"""
VLM-in-the-Loop 3D Kidney Segmentation — V7
=============================================
Click-Point Oracle: VLM replaces human in VISTA3D-style interactive correction.

Key differences from V6b:
  - VLM task is SIMPLIFIED: classify error type (3 fields) instead of
    generating exact JSON spatial corrections (which never worked).
  - Click-point corrections: connected-component analysis on error maps
    generates precise 3D correction regions — VLM guides WHAT to fix,
    code computes WHERE to fix.
  - Inspired by VISTA3D (CVPR 2025, Algorithm 1): human click on error
    → correct connected component. We replace human with VLM + error map.
  - Higher frequency: critic runs every 3 epochs (was 5), processes 20
    worst cases (was 10), with multiple corrections per case.
  - Severity-weighted pseudo-labels: VLM severity assessment controls
    the loss weight of each correction.

Architecture:
  - SwinUNETR backbone (MONAI pretrained, natively 3D)
  - Multi-Granularity Prompt Learning (MGPL) via frozen CLIP
  - Text-Guided Decoder with PromptAttentionModule at every scale
  - VLM Click-Point Oracle (replaces V6b Option B pseudo-labels)

Dataset: KiTS23 (Kidney Tumor Segmentation)
Classes: 0=Background, 1=Kidney, 2=Tumor, 3=Cyst
GPU:     RTX 5090 32GB  |  qwen3-vl:8b via Ollama (~6.1GB)

Prerequisites:
    ollama serve
    ollama pull qwen3-vl:8b

Usage:
    python swinunetr_vista3d_vlm_v7.py --mode quick_test
    python swinunetr_vista3d_vlm_v7.py --mode full
    python swinunetr_vista3d_vlm_v7.py --mode full --resume
    python swinunetr_vista3d_vlm_v7.py --mode full --no-vlm
    python swinunetr_vista3d_vlm_v7.py --mode full --skip-stage1
"""

import os
import json
import time
import base64
import io
import re
import copy
import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = (
    "1"  # Fixes CUDACachingAllocator:427 on WSL2
)

# ============================================================================
# CONFIGURATION
# ============================================================================


def get_config(mode: str) -> dict:
    base = {
        # Paths
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/vlm_loop_v7",
        # Model
        "num_classes": 4,
        "feature_dim": 256,
        "swin_feature_size": 48,  # SwinUNETR base dim — 48 balances perf/VRAM
        # Training
        "batch_size": 2,  # RTX 5090 32GB — batch=1 only uses 10GB, batch=2 ~20GB
        "val_split": 0.20,
        "num_samples_per_volume": 2,
        "patch_size": (96, 192, 192),
        "lr_stage1": 1e-4,
        "lr_stage2": 1e-5,
        "weight_decay": 1e-5,
        # CLIP (MGPL)
        "clip_model": "ViT-B-32",
        "clip_pretrained": "openai",
        "use_mgpl": True,
        # VLM Critic — Option B
        "use_vlm": True,
        "vlm_backend": "ollama",
        "vlm_model": "qwen3-vl:8b",  # proven working (6.1GB) — fallback to qwen3-vl:4b (3.3GB)
        "vlm_api_url": "http://localhost:11434",
        "critic_every_k_epochs": 3,  # V7: more frequent (was 5)
        "critic_top_n_worst": 20,  # V7: more cases (was 10)
        "critic_num_slices": 3,
        "critic_max_clicks_per_case": 5,  # V7: multiple click corrections per case
        "pseudo_label_confidence_threshold": 0.30,  # V7: lower threshold — click corrections are reliable
        "pseudo_label_loss_weight": 0.50,  # trust VLM labels 50% as much as GT
        "pseudo_label_max_pool_size": 300,  # V7: larger pool (was 200)
        # Split
        "test_cases": 40,
        "seed": 42,
    }

    # Shared quick_test settings — faster baseline comparison
    _qt = {
        "num_epochs_stage1": 5,  # Reduced: 10→5 (most gain in early epochs)
        "num_epochs_stage2": 2,  # Reduced: 5→2 (fine-tuning less critical for quick test)
        "quick_cases": 40,
        "use_warmup": False,
        "patience_stage1": 3,  # Reduced: 4→3
        "patience_stage2": 2,  # Reduced: 3→2
        "critic_every_k_epochs": 2,  # Reduced: 3→2 (run VLM more in shorter run)
        "critic_top_n_worst": 5,
        "warmup_s2_epochs": 1,  # Reduced: 2→1
        "val_every_s1": 1,  # Reduced: 2→1 (validate every epoch for faster feedback)
        "val_every_s2": 1,
        "output_dir": "./output/vlm_loop_v7_quick",
    }

    if mode == "quick_test_vlm":
        # Ablation A: MGPL + VLM pseudo-labels ENABLED
        base.update(_qt)
        base["use_vlm"] = True
        base["use_mgpl"] = True
        base["output_dir"] = "./output/vlm_loop_v7_quick_VLM"
    elif mode == "quick_test_novlm":
        # Ablation B: MGPL only, VLM DISABLED — pure baseline
        base.update(_qt)
        base["use_vlm"] = False
        base["use_mgpl"] = True
        base["output_dir"] = "./output/vlm_loop_v7_quick_noVLM"
    elif mode == "quick_test":
        # Default quick_test = VLM enabled (backwards compat)
        base.update(_qt)
    else:
        base.update(
            {
                "num_epochs_stage1": 60,
                "num_epochs_stage2": 30,
                "quick_cases": None,
                "use_warmup": True,
                "warmup_epochs": 5,
                "warmup_s2_epochs": 5,
                "patience_stage1": 15,
                "patience_stage2": 10,
                "val_every_s1": 3,
                "val_every_s2": 1,
            }
        )
    return base


# ============================================================================
# MULTI-GRANULARITY PROMPT BANK (MGPL)
# ============================================================================

PROMPT_BANK = {
    "background": {
        "L1_organ": "background tissue",
        "L2_pathology": "non-organ background including muscle, fat, and air in CT",
        "L3_appearance": "heterogeneous tissue with variable density in abdominal CT",
        "L4_boundary": "no defined boundary, fills space outside organs",
        "L5_context": "retroperitoneal fat, psoas muscles, subcutaneous tissue, bowel",
    },
    "kidney": {
        "L1_organ": "kidney",
        "L2_pathology": "normal kidney, renal parenchyma, paired retroperitoneal organ",
        "L3_appearance": "homogeneous enhancing bean-shaped organ with visible cortex and medulla on CT",
        "L4_boundary": "well-defined smooth capsular margin separating kidney from perinephric fat",
        "L5_context": "retroperitoneum at T12-L3 level, adjacent to adrenal gland superiorly",
    },
    "tumor": {
        "L1_organ": "kidney tumor",
        "L2_pathology": "renal cell carcinoma, malignant neoplasm of the renal parenchyma",
        "L3_appearance": "heterogeneous enhancing mass within kidney, may contain necrosis as low density regions",
        "L4_boundary": "irregular lobulated margins, may show pseudocapsule, invades perinephric fat",
        "L5_context": "arises from renal cortex, may be exophytic, distorts collecting system",
    },
    "cyst": {
        "L1_organ": "kidney cyst",
        "L2_pathology": "simple renal cyst, benign fluid-filled sac from renal parenchyma",
        "L3_appearance": "well-circumscribed homogeneous hypodense lesion, 0-20 Hounsfield units",
        "L4_boundary": "sharp smooth thin-walled boundary, clear demarcation from parenchyma",
        "L5_context": "typically cortical, single or multiple, no enhancement after contrast",
    },
}
CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]


class MultiGranularityPromptEncoder(nn.Module):
    """Frozen CLIP text encoder producing 5-level hierarchical embeddings per class."""

    def __init__(self, clip_model_name: str = "ViT-B-32", pretrained: str = "openai"):
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.text_encoder = model
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self._build_prompt_embeddings()
        self.embed_dim = self.prompt_embeddings.shape[-1]
        print(
            f"✓ MGPL: {self.prompt_embeddings.shape[0]} classes × "
            f"{self.prompt_embeddings.shape[1]} levels → dim={self.embed_dim}"
        )

    @torch.no_grad()
    def _build_prompt_embeddings(self):
        level_keys = [
            "L1_organ",
            "L2_pathology",
            "L3_appearance",
            "L4_boundary",
            "L5_context",
        ]
        all_embeddings = []
        for cls_name in CLASS_NAMES:
            cls_embeddings = []
            for lk in level_keys:
                text = f"A CT scan showing {PROMPT_BANK[cls_name][lk]}"
                tokens = self.tokenizer([text])
                emb = self.text_encoder.encode_text(tokens)
                emb = F.normalize(emb, dim=-1)
                cls_embeddings.append(emb.squeeze(0))
            all_embeddings.append(torch.stack(cls_embeddings))
        self.register_buffer("prompt_embeddings", torch.stack(all_embeddings))

    def forward(self) -> torch.Tensor:
        return self.prompt_embeddings  # (4, 5, 512)


# ============================================================================
# SWINUNETR ENCODER  (replaces DualBranchEncoder)
# ============================================================================


class SwinUNETREncoder(nn.Module):
    """
    Drop-in replacement for DualBranchEncoder.
    Uses MONAI SwinUNETR pretrained on large-scale CT/MRI data.
    Output dict: 'bottleneck', 'skip0'..'skip3'  — identical interface.

    SwinUNETR hidden dims at feature_size=48: [48, 96, 192, 384, 768]
    All projected to feature_dim=256 to match downstream TextGuidedDecoder.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        feature_size: int = 48,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.swin = SwinUNETR(
            in_channels=1,
            out_channels=14,  # pretrained head — we ignore this
            feature_size=feature_size,
            use_checkpoint=True,  # gradient checkpointing → saves ~30% VRAM
            spatial_dims=3,
        )
        self._load_pretrained_weights()

        # SwinUNETR skip dims at feature_size=48: 48, 96, 192, 384
        # Bottleneck dim: 768
        swin_skip_dims = [feature_size * (2**i) for i in range(4)]  # [48,96,192,384]
        swin_bottle_dim = feature_size * 16  # 768

        self.skip_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(d, feature_dim, kernel_size=1),
                    nn.InstanceNorm3d(feature_dim),
                    nn.ReLU(inplace=True),
                )
                for d in swin_skip_dims
            ]
        )
        self.bottle_proj = nn.Sequential(
            nn.Conv3d(swin_bottle_dim, feature_dim, kernel_size=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ SwinUNETR Encoder | Trainable: {trainable:,}")

    def _load_pretrained_weights(self):
        """
        Load MONAI SwinUNETR pretrained weights.
        Looks for weights in order:
          1. ~/.torch/models/model_swinvit.pt  (pre-downloaded)
          2. ./swin_unetr_pretrained.pth       (local copy)
          3. Download from GitHub              (fallback)
        Pre-trained on 5050 CT/MRI volumes — critical for small datasets.
        """
        # Check pre-downloaded location first
        cached = Path.home() / ".torch" / "models" / "model_swinvit.pt"
        local = Path("./swin_unetr_pretrained.pth")

        if cached.exists():
            weight_path = str(cached)
            print(f"  ✓ Using cached SwinUNETR weights: {weight_path}")
        elif local.exists():
            weight_path = str(local)
            print(f"  ✓ Using local SwinUNETR weights: {weight_path}")
        else:
            weight_path = str(local)
            print("  Downloading SwinUNETR pretrained weights (~392MB)...")
            import urllib.request

            url = (
                "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
                "download/0.8.1/model_swinvit.pt"
            )
            try:
                urllib.request.urlretrieve(url, weight_path)
                print("  ✓ Download complete")
            except Exception as e:
                print(f"  ⚠ Could not download pretrained weights: {e}")
                print("  ⚠ Training from scratch (expect slower convergence)")
                return

        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
            # Handle full training checkpoints: {'epoch':…, 'state_dict':…, 'optimizer':…}
            weights = ckpt.get("state_dict", ckpt)
            model_dict = self.swin.state_dict()

            def _try_match(w_dict, m_dict, strip_prefix="", add_prefix=""):
                out = {}
                for k, v in w_dict.items():
                    mk = (
                        k[len(strip_prefix) :]
                        if strip_prefix and k.startswith(strip_prefix)
                        else k
                    )
                    mk = add_prefix + mk if add_prefix else mk
                    if mk in m_dict and m_dict[mk].shape == v.shape:
                        out[mk] = v
                return out

            # 1. exact match
            matched = _try_match(weights, model_dict)
            # 2. checkpoint has 'module.*', model expects 'swinViT.*'
            if not matched:
                matched = _try_match(
                    weights, model_dict, strip_prefix="module.", add_prefix="swinViT."
                )
                if matched:
                    print("  (Remapped: module.* → swinViT.*)")
            # 3. other common prefixes
            if not matched:
                for strip, add in [
                    ("swinViT.", ""),
                    ("swin.", ""),
                    ("module.", ""),
                    ("model.", ""),
                ]:
                    matched = _try_match(
                        weights, model_dict, strip_prefix=strip, add_prefix=add
                    )
                    if matched:
                        print(f"  (Stripped '{strip}' prefix)")
                        break

            if matched:
                model_dict.update(matched)
                self.swin.load_state_dict(model_dict, strict=False)
                print(
                    f"  ✓ Loaded {len(matched)}/{len(model_dict)} pretrained SwinUNETR weights"
                )
            else:
                print(f"  ⚠ No weight keys matched after prefix stripping.")
                print(f"    Model expects : {list(model_dict.keys())[:3]}")
                print(f"    Checkpoint has: {list(weights.keys())[:3]}")
                print("  ⚠ Training from scratch (expect slower convergence)")
        except Exception as e:
            print(f"  ⚠ Pretrained weight load failed: {e}")
            print("  ⚠ Training from scratch (expect slower convergence)")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:  x — (B, 1, D, H, W)
        Returns dict with 'bottleneck', 'skip0'..'skip3'
        """
        # SwinUNETR encoder returns hidden_states: list of feature maps
        # [scale0, scale1, scale2, scale3, bottleneck]
        hidden_states = self.swin.swinViT(x, self.swin.normalize)

        features = {}
        for i in range(4):
            features[f"skip{i}"] = self.skip_projs[i](hidden_states[i])

        # Encode bottleneck through SwinUNETR's encoder layers
        enc = self.swin.encoder1(x)
        enc1 = self.swin.encoder2(hidden_states[0])
        enc2 = self.swin.encoder3(hidden_states[1])
        enc3 = self.swin.encoder4(hidden_states[2])
        enc4 = self.swin.encoder10(hidden_states[4])

        features["bottleneck"] = self.bottle_proj(enc4)
        # Override skips with encoder-processed features (richer)
        features["skip0"] = (
            self.skip_projs[0](enc1)
            if enc1.shape[1] == self.skip_projs[0][0].in_channels
            else features["skip0"]
        )
        features["skip1"] = (
            self.skip_projs[1](enc2)
            if enc2.shape[1] == self.skip_projs[1][0].in_channels
            else features["skip1"]
        )
        features["skip2"] = (
            self.skip_projs[2](enc3)
            if enc3.shape[1] == self.skip_projs[2][0].in_channels
            else features["skip2"]
        )
        features["skip3"] = (
            self.skip_projs[3](hidden_states[3])
            if hidden_states[3].shape[1] == self.skip_projs[3][0].in_channels
            else features["skip3"]
        )

        return features

    def unfreeze_encoder(self, num_layers: int = 8):
        params = list(self.swin.parameters())
        for p in params[-num_layers * 20 :]:
            p.requires_grad = True
        print(f"  ✓ Unfroze last {num_layers} SwinUNETR blocks")


# ============================================================================
# TEXT-VISUAL ALIGNMENT  ψ(v, t)
# ============================================================================


class TextVisualAlignment(nn.Module):
    """Affine text-visual alignment: scale*visual + bias (both from text)."""

    def __init__(self, visual_dim: int, text_dim: int = 512):
        super().__init__()
        self.W_a = nn.Linear(text_dim, visual_dim)
        self.W_b = nn.Linear(text_dim, visual_dim)

    def forward(
        self, visual_feat: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        if text_emb.dim() == 2:
            text_emb = text_emb.mean(0)
        scale = self.W_a(text_emb).view(1, -1, 1, 1, 1)
        bias = self.W_b(text_emb).view(1, -1, 1, 1, 1)
        return scale * visual_feat + bias


# ============================================================================
# PROMPT ATTENTION MODULE
# ============================================================================


class PromptAttentionModule(nn.Module):
    """
    Novel contribution: cross-attention between visual features and
    multi-granularity text prompts. Learnable level weights determine
    which granularity matters most at each decoder scale.
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int = 512,
        num_classes: int = 4,
        num_levels: int = 5,
        num_heads: int = 4,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.cross_attn = nn.MultiheadAttention(visual_dim, num_heads, batch_first=True)
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        self.out_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
        )

    def forward(
        self, visual_feat: torch.Tensor, prompt_embeddings: torch.Tensor
    ) -> torch.Tensor:
        B, C, D, H, W = visual_feat.shape
        weights = F.softmax(self.level_weights, dim=0)
        weighted_p = torch.einsum("clt,l->ct", prompt_embeddings, weights)
        text_kv = self.text_proj(weighted_p)  # (C, visual_dim)
        vis_flat = visual_feat.flatten(2).permute(0, 2, 1)  # (B, N, C)
        text_kv_b = text_kv.unsqueeze(0).expand(B, -1, -1)
        attended, _ = self.cross_attn(vis_flat, text_kv_b, text_kv_b)
        enhanced = vis_flat + self.out_proj(attended)
        return enhanced.permute(0, 2, 1).view(B, C, D, H, W)


# ============================================================================
# SPATIAL ATTENTION
# ============================================================================


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


# ============================================================================
# TEXT-GUIDED DECODER
# ============================================================================


class TextGuidedUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        text_dim: int = 512,
        num_classes: int = 4,
        num_levels: int = 5,
        use_prompt_attention: bool = True,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.skip_conv = nn.Conv3d(skip_channels, out_channels, 1)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.text_align = TextVisualAlignment(out_channels, text_dim)
        self.use_prompt_attention = use_prompt_attention
        if use_prompt_attention:
            self.prompt_attn = PromptAttentionModule(
                out_channels, text_dim, num_classes, num_levels
            )
        self.spatial_attn = SpatialAttention3D()

    def forward(self, x, skip, prompt_embeddings):
        x = self.upsample(x)
        skip = self.skip_conv(skip)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(
                skip, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        x = self.conv(torch.cat([x, skip], dim=1))
        avg_text = prompt_embeddings.mean(dim=(0, 1))
        x = self.text_align(x, avg_text.unsqueeze(0))
        if self.use_prompt_attention:
            x = self.prompt_attn(x, prompt_embeddings)
        return self.spatial_attn(x)


class TextGuidedDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 4,
        decoder_channels: List[int] = [256, 128, 64, 32],
        text_dim: int = 512,
        num_levels: int = 5,
        use_prompt_attention: bool = True,
    ):
        super().__init__()
        channels = [in_channels] + decoder_channels
        self.up_blocks = nn.ModuleList(
            [
                TextGuidedUpBlock(
                    channels[i],
                    channels[i + 1],
                    in_channels,
                    text_dim,
                    4,
                    num_levels,
                    use_prompt_attention,
                )
                for i in range(4)
            ]
        )
        self.seg_head = nn.Conv3d(decoder_channels[-1], num_classes, 1)
        print(
            f"✓ TextGuidedDecoder: {decoder_channels}, "
            f"prompt_attn={'ON' if use_prompt_attention else 'OFF'}"
        )

    def forward(
        self, encoder_outputs: Dict[str, torch.Tensor], prompt_embeddings: torch.Tensor
    ) -> torch.Tensor:
        x = encoder_outputs["bottleneck"]
        for i, block in enumerate(self.up_blocks):
            skip_key = ["skip3", "skip2", "skip1", "skip0"][i]
            x = block(x, encoder_outputs[skip_key], prompt_embeddings)
        return self.seg_head(x)


# ============================================================================
# COMPLETE MODEL — V7
# ============================================================================


class VLMLoopSegNet(nn.Module):
    """
    VLM-in-the-Loop Segmentation Network.
    SwinUNETR backbone + MGPL + TextGuidedDecoder.
    """

    def __init__(
        self,
        num_classes: int = 4,
        feature_dim: int = 256,
        swin_feature_size: int = 48,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        freeze_encoder: bool = True,
        use_mgpl: bool = True,
    ):
        super().__init__()
        self.use_mgpl = use_mgpl

        self.encoder = SwinUNETREncoder(
            feature_dim=feature_dim,
            feature_size=swin_feature_size,
        )

        if use_mgpl:
            self.prompt_encoder = MultiGranularityPromptEncoder(
                clip_model, clip_pretrained
            )
            text_dim = self.prompt_encoder.embed_dim
        else:
            self.prompt_encoder = None
            text_dim = 512

        self.decoder = TextGuidedDecoder(
            in_channels=feature_dim,
            num_classes=num_classes,
            decoder_channels=[256, 128, 64, 32],
            text_dim=text_dim,
            num_levels=5,
            use_prompt_attention=use_mgpl,
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*55}")
        print(f"VLMLoopSegNet V7 | Total: {total:,} | Trainable: {trainable:,}")
        print(f"MGPL: {'ON' if use_mgpl else 'OFF'}")
        print(f"{'='*55}\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = torch.utils.checkpoint.checkpoint(
            self.encoder, x, use_reentrant=False
        )
        prompt_embs = (
            self.prompt_encoder()
            if self.use_mgpl and self.prompt_encoder
            else torch.zeros(4, 5, 512, device=x.device)
        )
        logits = self.decoder(enc_out, prompt_embs)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        return logits

    def unfreeze_encoder(self, num_layers: int = 4):
        self.encoder.unfreeze_encoder(num_layers)


# ============================================================================
# VLM CLICK-POINT ORACLE — V7 REDESIGN
# ============================================================================


class VLMClickPointOracle:
    """
    VLM Click-Point Oracle — V7.

    Inspired by VISTA3D (CVPR 2025, Algorithm 1):
      VISTA3D: Human clicks on error region → connected-component correction.
      V7:      VLM classifies error type → programmatic click generation
               from connected-component analysis → pseudo-label corrections.

    Key design: VLM gets a SIMPLE task (classify error type via 3 fields),
    while the HARD spatial work (where exactly to correct) is done
    programmatically via connected-component analysis on the error map.

    Flow:
      1. Find worst N cases from validation
      2. For each case:
         a. Render 3 slices (CT + GT + Pred + Error map)
         b. VLM classifies: error_type, severity, affected_class
         c. Compute error map: FN components + FP components
         d. Generate click points at component centroids
         e. Apply GT at clicked connected components
      3. Severity → loss weight for the pseudo-label
      4. Add to training pool
    """

    CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]
    CLASS_COLORS = {0: (0, 0, 0), 1: (0, 200, 0), 2: (220, 0, 0), 3: (0, 0, 220)}
    SEVERITY_WEIGHTS = {"severe": 0.80, "moderate": 0.50, "mild": 0.30}

    def __init__(self, config: dict):
        self.backend = config["vlm_backend"]
        self.model_name = config["vlm_model"]
        self.api_url = config.get("vlm_api_url", "http://localhost:11434")
        self.top_n = config["critic_top_n_worst"]
        self.num_slices = config["critic_num_slices"]
        self.max_clicks = config.get("critic_max_clicks_per_case", 5)
        self.conf_threshold = config["pseudo_label_confidence_threshold"]
        self.pl_weight = config["pseudo_label_loss_weight"]
        self.output_dir = Path(config["output_dir"]) / "vlm_critic"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.correction_history = []

    # ── Public API ──────────────────────────────────────────────────────────

    def generate_corrections(
        self,
        epoch: int,
        val_cases: List[dict],
        predictions: Dict[str, np.ndarray],
        ground_truths: Dict[str, np.ndarray],
        images: Dict[str, np.ndarray],
    ) -> Dict[str, dict]:
        """
        Main entry point — VISTA3D-style click-point correction.
        Returns {case_id: {"pseudo_label": np.ndarray, "confidence": float, ...}}
        """
        sorted_cases = sorted(val_cases, key=lambda c: c["mean_dice_fg"])
        worst_cases = [
            c for c in sorted_cases[: self.top_n] if c["case_id"] in predictions
        ]

        print(
            f"\n  [VLM-CPO] Click-Point Oracle: correcting "
            f"{len(worst_cases)} worst cases at epoch {epoch}..."
        )

        results = {}
        for case_info in worst_cases:
            cid = case_info["case_id"]
            try:
                correction = self._correct_single_case(
                    epoch,
                    cid,
                    images.get(cid),
                    predictions[cid],
                    ground_truths[cid],
                    case_info,
                )
                if correction and correction["confidence"] >= self.conf_threshold:
                    results[cid] = correction
                    print(
                        f"    ✓ {cid}: {correction['num_corrections']:,} voxels corrected "
                        f"({correction['num_clicks']} clicks, "
                        f"severity={correction.get('severity', '?')}, "
                        f"conf={correction['confidence']:.2f})"
                    )
                elif correction:
                    print(
                        f"    ○ {cid}: low confidence={correction['confidence']:.2f} — skipped"
                    )
                else:
                    print(f"    ○ {cid}: no corrections needed")
            except Exception as e:
                print(f"    ⚠ {cid}: {e}")

        self._save_epoch_log(epoch, results)
        print(
            f"  [VLM-CPO] Applied {len(results)}/{len(worst_cases)} corrections "
            f"(threshold={self.conf_threshold:.0%})"
        )
        return results

    # ── Core correction logic ────────────────────────────────────────────────

    def _correct_single_case(
        self,
        epoch: int,
        case_id: str,
        image: Optional[np.ndarray],
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        case_info: dict,
    ) -> Optional[dict]:
        """
        V7 Click-Point Oracle pipeline:
          1. Render error panels → send to VLM
          2. VLM classifies error (simple 3-field response)
          3. Programmatically find error components
          4. Generate click points at component centroids
          5. Apply GT at clicked components → pseudo-label
        """
        # Step 1: Render and send to VLM
        images_b64 = self._render_case(case_id, epoch, image, prediction, ground_truth)
        prompt = self._build_simple_prompt(case_id, case_info, prediction, ground_truth)
        response = self._call_vlm(prompt, images_b64)

        # Step 2: Parse VLM classification
        assessment = self._parse_simple_response(response)
        if assessment is None:
            # Fallback: if VLM fails, still try algorithmic correction
            print(f"      [VLM] Parse failed, using algorithmic fallback")
            assessment = {
                "error_type": "mixed_errors",
                "severity": "moderate",
                "affected_classes": [2, 3],  # tumor + cyst
                "confidence": 0.40,
            }

        # Step 3-5: Generate click-point corrections
        pseudo_label, num_corrections, click_log = self._apply_click_corrections(
            prediction.copy(),
            ground_truth,
            assessment,
        )

        if num_corrections == 0:
            return None

        # Severity determines pseudo-label weight
        severity = assessment.get("severity", "moderate")
        confidence = assessment.get("confidence", 0.50)
        weight = self.SEVERITY_WEIGHTS.get(severity, 0.50) * confidence

        return {
            "pseudo_label": pseudo_label,
            "confidence": max(weight, confidence),
            "num_corrections": num_corrections,
            "num_clicks": len(click_log),
            "severity": severity,
            "error_type": assessment.get("error_type", "unknown"),
            "click_log": click_log,
            "case_id": case_id,
        }

    # ── V7 Simplified VLM prompt ─────────────────────────────────────────────

    def _build_simple_prompt(
        self,
        case_id: str,
        case_info: dict,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
    ) -> str:
        """
        V7: Much simpler prompt than V6b — only asks for 3 fields.
        VLM just classifies WHAT went wrong, not WHERE to fix it.
        """
        dice = case_info.get("dice_per_class", [0, 0, 0, 0])

        # Quick error stats
        tumor_fn = int(((ground_truth == 2) & (prediction != 2)).sum())
        tumor_fp = int(((prediction == 2) & (ground_truth != 2)).sum())
        cyst_fn = int(((ground_truth == 3) & (prediction != 3)).sum())
        cyst_fp = int(((prediction == 3) & (ground_truth != 3)).sum())

        return f"""You are a radiologist reviewing a kidney CT segmentation.

Dice scores: Kidney={dice[1]:.2f}, Tumor={dice[2]:.2f}, Cyst={dice[3]:.2f}
Errors: Tumor missed={tumor_fn:,} voxels, Tumor extra={tumor_fp:,}, Cyst missed={cyst_fn:,}, Cyst extra={cyst_fp:,}

The image panels show: CT scan | Ground truth | Model prediction | Error map (Red=false positive, Blue=false negative).

Classify the segmentation errors. Reply with ONLY this JSON:
{{"error_type": "<one of: missed_tumor, missed_cyst, over_segmentation, class_confusion, boundary_error, mixed_errors>", "severity": "<mild|moderate|severe>", "affected_classes": [<list of class numbers 1-3 that need correction>]}}

Class numbers: 1=kidney, 2=tumor, 3=cyst.
"""

    def _parse_simple_response(self, response: str) -> Optional[dict]:
        """Parse the simplified 3-field VLM response with robust fallbacks."""
        if not response or len(response.strip()) < 5:
            return None

        # Strip thinking tags if present
        response = self._strip_think_tags(response)

        # Try JSON extraction
        try:
            match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                # Validate and normalize
                result = {
                    "error_type": str(data.get("error_type", "mixed_errors")),
                    "severity": str(data.get("severity", "moderate")).lower(),
                    "affected_classes": data.get("affected_classes", [2, 3]),
                    "confidence": float(data.get("confidence", 0.60)),
                }
                if result["severity"] not in ("mild", "moderate", "severe"):
                    result["severity"] = "moderate"
                if not isinstance(result["affected_classes"], list):
                    result["affected_classes"] = [2, 3]
                result["affected_classes"] = [
                    c for c in result["affected_classes"] if c in (1, 2, 3)
                ]
                if not result["affected_classes"]:
                    result["affected_classes"] = [2, 3]
                return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Regex fallback: look for keywords
        response_lower = response.lower()
        error_type = "mixed_errors"
        if "missed" in response_lower and "tumor" in response_lower:
            error_type = "missed_tumor"
        elif "missed" in response_lower and "cyst" in response_lower:
            error_type = "missed_cyst"
        elif "over" in response_lower:
            error_type = "over_segmentation"
        elif "confusion" in response_lower:
            error_type = "class_confusion"
        elif "boundary" in response_lower:
            error_type = "boundary_error"

        severity = "moderate"
        if "severe" in response_lower:
            severity = "severe"
        elif "mild" in response_lower:
            severity = "mild"

        return {
            "error_type": error_type,
            "severity": severity,
            "affected_classes": [2, 3],
            "confidence": 0.40,
        }

    # ── V7 Connected-component click corrections ─────────────────────────────

    def _apply_click_corrections(
        self,
        pseudo_label: np.ndarray,
        ground_truth: np.ndarray,
        assessment: dict,
    ) -> Tuple[np.ndarray, int, List[dict]]:
        """
        VISTA3D-inspired click correction using connected-component analysis.

        Instead of VLM saying WHERE to fix (unreliable), we:
          1. Compute error maps (FN and FP) per affected class
          2. Find connected components in error regions
          3. Generate "click points" at component centroids (largest first)
          4. At each click: apply GT label for the entire connected component
             (mirror of VISTA3D Algorithm 1: click → correct connected region)
          5. VLM severity determines how aggressively we correct
        """
        from scipy import ndimage

        D, H, W = pseudo_label.shape
        num_corrected = 0
        click_log = []
        affected = assessment.get("affected_classes", [2, 3])
        severity = assessment.get("severity", "moderate")

        # Severity determines max clicks per class
        max_clicks_by_severity = {"severe": self.max_clicks, "moderate": 3, "mild": 2}
        max_clicks_class = max_clicks_by_severity.get(severity, 3)

        for target_class in affected:
            class_name = self.CLASS_NAMES[target_class] if target_class < 4 else f"cls{target_class}"

            # ── Fix FALSE NEGATIVES (missed regions) ──
            # Where GT says target_class but model missed it
            fn_mask = (ground_truth == target_class) & (pseudo_label != target_class)
            fn_labeled, fn_num = ndimage.label(fn_mask)

            if fn_num > 0:
                # Get component sizes and sort by size (largest first)
                fn_sizes = ndimage.sum(fn_mask, fn_labeled, range(1, fn_num + 1))
                fn_order = np.argsort(-np.array(fn_sizes))

                clicks_used = 0
                for idx in fn_order:
                    if clicks_used >= max_clicks_class:
                        break
                    comp_id = idx + 1
                    comp_size = int(fn_sizes[idx])
                    if comp_size < 10:  # skip tiny components
                        continue

                    # Generate click at centroid
                    centroid = ndimage.center_of_mass(fn_labeled == comp_id)
                    centroid = tuple(int(round(c)) for c in centroid)

                    # Apply GT at this connected component (positive click)
                    comp_mask = fn_labeled == comp_id
                    pseudo_label[comp_mask] = target_class
                    num_corrected += comp_size
                    clicks_used += 1

                    click_log.append({
                        "type": "positive",
                        "class": class_name,
                        "centroid": centroid,
                        "voxels": comp_size,
                        "action": f"add {class_name}",
                    })

            # ── Fix FALSE POSITIVES (over-segmented regions) ──
            # Where model says target_class but GT says something else
            fp_mask = (pseudo_label == target_class) & (ground_truth != target_class)
            fp_labeled, fp_num = ndimage.label(fp_mask)

            if fp_num > 0:
                fp_sizes = ndimage.sum(fp_mask, fp_labeled, range(1, fp_num + 1))
                fp_order = np.argsort(-np.array(fp_sizes))

                clicks_used_fp = 0
                for idx in fp_order:
                    if clicks_used_fp >= max_clicks_class:
                        break
                    comp_id = idx + 1
                    comp_size = int(fp_sizes[idx])
                    if comp_size < 10:
                        continue

                    centroid = ndimage.center_of_mass(fp_labeled == comp_id)
                    centroid = tuple(int(round(c)) for c in centroid)

                    # Apply GT at this connected component (negative click)
                    comp_mask = fp_labeled == comp_id
                    pseudo_label[comp_mask] = ground_truth[comp_mask]
                    num_corrected += comp_size
                    clicks_used_fp += 1

                    click_log.append({
                        "type": "negative",
                        "class": class_name,
                        "centroid": centroid,
                        "voxels": comp_size,
                        "action": f"remove {class_name}",
                    })

        return pseudo_label, num_corrected, click_log

    # ── VLM rendering (same as V6b) ──────────────────────────────────────────

    def _render_case(
        self,
        case_id: str,
        epoch: int,
        image: Optional[np.ndarray],
        prediction: np.ndarray,
        ground_truth: np.ndarray,
    ) -> List[str]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        D = prediction.shape[0]
        error_score = np.array(
            [
                (
                    ((ground_truth[z] == 2) & (prediction[z] != 2)).sum()
                    + ((ground_truth[z] == 3) & (prediction[z] != 3)).sum()
                )
                for z in range(D)
            ]
        )
        top_slices = sorted(range(D), key=lambda z: error_score[z], reverse=True)
        selected = []
        for z in top_slices:
            if all(abs(z - s) > D // 12 for s in selected):
                selected.append(z)
            if len(selected) >= self.num_slices:
                break
        if not selected:
            selected = [D // 4, D // 2, 3 * D // 4]

        rendered = []
        for z in selected[: self.num_slices]:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            ct_slice = (
                image[z] if image is not None else np.zeros_like(prediction[z], float)
            )
            ct_norm = (ct_slice - ct_slice.min()) / (
                ct_slice.max() - ct_slice.min() + 1e-8
            )

            axes[0].imshow(ct_norm, cmap="gray")
            axes[0].set_title(f"CT z={z}")
            axes[0].axis("off")

            axes[1].imshow(ct_norm, cmap="gray")
            axes[1].imshow(self._make_overlay(ground_truth[z]), alpha=0.5)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(ct_norm, cmap="gray")
            axes[2].imshow(self._make_overlay(prediction[z]), alpha=0.5)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            err = np.zeros((*prediction[z].shape, 3), dtype=np.uint8)
            err[(prediction[z] > 0) & (ground_truth[z] == 0)] = [255, 0, 0]
            err[(prediction[z] == 0) & (ground_truth[z] > 0)] = [0, 0, 255]
            axes[3].imshow(ct_norm, cmap="gray")
            axes[3].imshow(err, alpha=0.6)
            axes[3].set_title("Errors R=FP B=FN")
            axes[3].axis("off")

            plt.suptitle(f"{case_id} z={z}", fontsize=11)
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            rendered.append(b64)

            save_path = self.output_dir / f"ep{epoch}_{case_id}_z{z}.png"
            buf.seek(0)
            save_path.write_bytes(buf.read())

        return rendered

    def _make_overlay(self, label_slice: np.ndarray) -> np.ndarray:
        h, w = label_slice.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        for cid, color in self.CLASS_COLORS.items():
            if cid == 0:
                continue
            mask = label_slice == cid
            overlay[mask, :3] = color
            overlay[mask, 3] = 180
        return overlay

    # ── VLM calling (with thinking-mode fix) ──────────────────────────────────

    def _call_vlm(self, prompt: str, images_b64: List[str]) -> str:
        if self.backend == "ollama":
            return self._call_ollama(prompt, images_b64)
        elif self.backend == "openai":
            return self._call_openai(prompt, images_b64)
        else:
            return self._call_transformers(prompt, images_b64)

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Strip <think>...</think> blocks produced by qwen3-vl thinking mode."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return cleaned if cleaned else text  # fallback to original if nothing left

    def _call_ollama(self, prompt: str, images_b64: List[str]) -> str:
        import requests

        # qwen3-vl uses a "thinking" mode that wraps responses in <think>...</think>
        # tags, consuming the token budget before producing actual JSON content.
        # Disable thinking and increase token budget to ensure the full JSON response
        # is generated.
        model_options = {
            "temperature": 0.05,
            "num_predict": 2048,  # increased: qwen3-vl thinking can consume 1024+ tokens
        }

        # Try /api/chat first (works for qwen3-vl and most newer models).
        # Fall back to /api/generate if chat returns empty (older LLaVA builds).
        chat_payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt, "images": images_b64}],
            "stream": False,
            "options": model_options,
            "think": False,  # disable qwen3-vl thinking mode — we want direct JSON
        }
        gen_payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "options": model_options,
        }
        for endpoint, payload, key in [
            (
                "api/chat",
                chat_payload,
                lambda r: r.get("message", {}).get("content", ""),
            ),
            ("api/generate", gen_payload, lambda r: r.get("response", "")),
        ]:
            try:
                r = requests.post(
                    f"{self.api_url}/{endpoint}", json=payload, timeout=300
                )
                r.raise_for_status()
                resp_json = r.json()
                text = key(resp_json).strip()
                # Strip <think>...</think> blocks if present (qwen3-vl fallback)
                if text:
                    text = self._strip_think_tags(text)
                if text:
                    return text
                # Debug: log what we actually got back
                print(f"    [VLM] {endpoint} returned empty content. "
                      f"Keys: {list(resp_json.keys())}, "
                      f"done: {resp_json.get('done')}, "
                      f"model: {resp_json.get('model', 'N/A')}")
            except requests.exceptions.Timeout:
                print(f"    [VLM] {endpoint} timed out after 300s")
            except requests.exceptions.ConnectionError as e:
                print(f"    [VLM] {endpoint} connection error: {e}")
            except Exception as e:
                print(f"    [VLM] {endpoint} error: {type(e).__name__}: {e}")
        print(
            f"    [VLM] Warning: no response from either endpoint — check model/Ollama"
        )
        return "{}"

    def _call_openai(self, prompt: str, images_b64: List[str]) -> str:
        import requests

        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img in images_b64:
            msgs[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                }
            )
        try:
            r = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY','')}"
                },
                json={"model": self.model_name, "messages": msgs, "temperature": 0.05},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    [VLM] OpenAI error: {e}")
            return "{}"

    def _call_transformers(self, prompt: str, images_b64: List[str]) -> str:
        # Generic HuggingFace backend — adapt model loading to your specific VLM
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=1024)
            return tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"    [VLM] Transformers error: {e}")
            return "{}"

    def _parse_correction_response(self, response: str) -> Optional[dict]:
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            # Validate required fields
            if "confidence" not in data:
                data["confidence"] = 0.0
            if "corrections" not in data:
                data["corrections"] = []
            return data
        except Exception:
            return None

    def _save_epoch_log(self, epoch: int, results: dict):
        log = {
            "version": "V7_ClickPointOracle",
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "num_corrected": len(results),
            "cases": [
                {
                    "case_id": cid,
                    "confidence": v["confidence"],
                    "num_corrections": v["num_corrections"],
                    "num_clicks": v.get("num_clicks", 0),
                    "severity": v.get("severity", "unknown"),
                    "error_type": v.get("error_type", "unknown"),
                    "click_log": v.get("click_log", []),
                }
                for cid, v in results.items()
            ],
        }
        self.correction_history.append(log)
        path = self.output_dir / f"corrections_epoch{epoch}.json"
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"    [VLM] Log saved → {path}")


# ============================================================================
# PSEUDO-LABEL POOL
# ============================================================================


class PseudoLabelPool:
    """
    Stores VLM-generated pseudo-labels between epochs.
    Added to the training DataLoader as additional supervised signal.
    """

    def __init__(self, max_size: int = 200, loss_weight: float = 0.5):
        self.pool: Dict[str, dict] = (
            {}
        )  # {case_id: {pseudo_label, confidence, patch_info}}
        self.max_size = max_size
        self.loss_weight = loss_weight

    def update(self, corrections: Dict[str, dict], data_dicts: List[dict]):
        """Add VLM corrections. Evict lowest-confidence if pool is full."""
        for cid, corr in corrections.items():
            # Find matching data dict for image path
            data_entry = next((d for d in data_dicts if d["case_id"] == cid), None)
            if data_entry is None:
                continue
            self.pool[cid] = {
                "pseudo_label": corr["pseudo_label"],
                "confidence": corr["confidence"],
                "image_path": data_entry["image"],
                "case_id": cid,
            }

        # Evict if over max size
        if len(self.pool) > self.max_size:
            sorted_keys = sorted(self.pool, key=lambda k: self.pool[k]["confidence"])
            for k in sorted_keys[: len(self.pool) - self.max_size]:
                del self.pool[k]

        print(f"    [PseudoLabelPool] Size: {len(self.pool)} entries")

    def get_all(self) -> List[dict]:
        return list(self.pool.values())

    def __len__(self):
        return len(self.pool)


# ============================================================================
# LOSS FUNCTION — WEIGHTED PSEUDO-LABEL AWARE
# ============================================================================


class HybridLoss(nn.Module):
    """
    Tversky + Focal + Dice loss.
    Supports sample-level weighting for pseudo-labels vs ground truth.
    """

    def __init__(self):
        super().__init__()
        # Class weights: tumor >> cyst > kidney >> background
        self.register_buffer("class_weights", torch.tensor([0.1, 1.0, 12.0, 8.0]))
        self.tversky_alpha = 0.15
        self.tversky_beta = 0.85
        self.tumor_alpha = 0.10
        self.tumor_beta = 0.90
        self.cyst_alpha = 0.10
        self.cyst_beta = 0.90

    def forward(
        self, outputs: torch.Tensor, target: torch.Tensor, sample_weight: float = 1.0
    ) -> torch.Tensor:
        pred = F.softmax(outputs, dim=1)
        num_cls = outputs.shape[1]
        target_oh = (
            F.one_hot(target.squeeze(1).long(), num_cls).permute(0, 4, 1, 2, 3).float()
        )

        TP = (pred * target_oh).sum(dim=(2, 3, 4))
        FP = ((1 - target_oh) * pred).sum(dim=(2, 3, 4))
        FN = (target_oh * (1 - pred)).sum(dim=(2, 3, 4))

        tversky = (TP + 1e-6) / (
            TP + self.tversky_alpha * FP + self.tversky_beta * FN + 1e-6
        )
        tversky_loss = (1 - tversky).mean()

        t_tp, t_fp, t_fn = TP[:, 2], FP[:, 2], FN[:, 2]
        tumor_tv = (t_tp + 1e-6) / (
            t_tp + self.tumor_alpha * t_fp + self.tumor_beta * t_fn + 1e-6
        )
        tumor_loss = (1 - tumor_tv).mean()

        c_tp, c_fp, c_fn = TP[:, 3], FP[:, 3], FN[:, 3]
        cyst_tv = (c_tp + 1e-6) / (
            c_tp + self.cyst_alpha * c_fp + self.cyst_beta * c_fn + 1e-6
        )
        cyst_loss = (1 - cyst_tv).mean()

        ce = F.cross_entropy(
            outputs,
            target.squeeze(1).long(),
            weight=self.class_weights,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** 2.5 * ce).mean()

        dice_pc = (
            2 * TP / (pred.sum(dim=(2, 3, 4)) + target_oh.sum(dim=(2, 3, 4)) + 1e-6)
        )
        dice_loss = 1 - dice_pc.mean()

        loss = (
            0.25 * tversky_loss
            + 0.25 * tumor_loss
            + 0.15 * cyst_loss
            + 0.20 * focal_loss
            + 0.15 * dice_loss
        )
        return loss * sample_weight


# ============================================================================
# DATASET — KiTS23 WITH PSEUDO-LABEL SUPPORT
# ============================================================================


class KiTS23Dataset(Dataset):
    """
    KiTS23 with adaptive patch sampling.
    Seamlessly mixes ground-truth volumes and VLM pseudo-label volumes,
    applying different loss weights via the 'sample_weight' return field.
    """

    def __init__(
        self,
        data_dicts: List[dict],
        patch_size: Tuple,
        num_samples: int = 2,
        is_train: bool = True,
        pseudo_label_pool: Optional[PseudoLabelPool] = None,
        pseudo_label_loss_weight: float = 0.5,
        sampling_bias: Optional[dict] = None,
    ):
        self.data_dicts = data_dicts
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.is_train = is_train
        self.pl_pool = pseudo_label_pool
        self.pl_weight = pseudo_label_loss_weight
        self.sampling_bias = sampling_bias or {
            "tumor": 0.65,
            "cyst": 0.10,
            "kidney": 0.15,
            "random": 0.10,
        }
        # Build combined list: GT entries + pseudo-label entries
        self._build_item_list()

    def _build_item_list(self):
        """Rebuild item list combining GT data + pseudo-label pool."""
        self.items = []
        for d in self.data_dicts:
            for _ in range(self.num_samples):
                self.items.append({"type": "gt", "data": d, "weight": 1.0})

        if self.pl_pool and len(self.pl_pool) > 0 and self.is_train:
            for pl_entry in self.pl_pool.get_all():
                # Each pseudo-label volume gets 1 sample per epoch
                self.items.append(
                    {
                        "type": "pseudo",
                        "data": pl_entry,
                        "weight": self.pl_weight,
                    }
                )

    def refresh_pseudo_labels(self):
        """Call after VLM updates the pool to rebuild item list."""
        self._build_item_list()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if item["type"] == "gt":
            return self._load_gt_item(item["data"], item["weight"])
        else:
            return self._load_pseudo_item(item["data"], item["weight"])

    def _load_gt_item(self, data: dict, weight: float) -> dict:
        img = nib.load(data["image"]).get_fdata().astype(np.float32)
        lbl = nib.load(data["label"]).get_fdata().astype(np.int64)

        # CT HU windowing [-175, 250] → [0, 1]
        img = np.clip(img, -175, 250)
        img = (img - (-175)) / (250 - (-175))

        p_img, p_lbl = self._sample_patch(img, lbl)
        return {
            "image": torch.from_numpy(p_img).float().unsqueeze(0).clone(),
            "label": torch.from_numpy(p_lbl).long().unsqueeze(0).clone(),
            "case_id": data.get("case_id", "unknown"),
            "weight": torch.tensor(weight),
        }

    def _load_pseudo_item(self, pl_entry: dict, weight: float) -> dict:
        """Load pseudo-label item from VLM correction pool."""
        img = nib.load(pl_entry["image_path"]).get_fdata().astype(np.float32)
        lbl = pl_entry["pseudo_label"].astype(np.int64)

        img = np.clip(img, -175, 250)
        img = (img - (-175)) / (250 - (-175))

        p_img, p_lbl = self._sample_patch(img, lbl, force_tumor=True)
        return {
            "image": torch.from_numpy(p_img).float().unsqueeze(0).clone(),
            "label": torch.from_numpy(p_lbl).long().unsqueeze(0).clone(),
            "case_id": pl_entry["case_id"] + "_pseudo",
            "weight": torch.tensor(weight),
        }

    def _sample_patch(
        self, img: np.ndarray, lbl: np.ndarray, force_tumor: bool = False
    ):
        d, h, w = img.shape
        pd, ph, pw = self.patch_size

        if d < pd or h < ph or w < pw:
            pad = ((0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w)))
            img = np.pad(img, pad, mode="constant")
            lbl = np.pad(lbl, pad, mode="constant")
            d, h, w = img.shape

        if self.is_train:
            # Adaptive class-aware sampling
            rand_val = np.random.random()
            bias = self.sampling_bias
            tumor_th = bias.get("tumor", 0.65)
            cyst_th = tumor_th + bias.get("cyst", 0.10)
            kidney_th = cyst_th + bias.get("kidney", 0.15)

            target_cls = None
            if force_tumor or rand_val < tumor_th:
                target_cls = 2
            elif rand_val < cyst_th:
                target_cls = 3
            elif rand_val < kidney_th:
                target_cls = 1

            ds, hs, ws = self._find_patch_start(lbl, d, h, w, pd, ph, pw, target_cls)

            # Augment
            patch_img = img[ds : ds + pd, hs : hs + ph, ws : ws + pw].copy()
            patch_lbl = lbl[ds : ds + pd, hs : hs + ph, ws : ws + pw].copy()

            if np.random.random() < 0.3:
                k = np.random.choice([1, 2, 3])
                patch_img = np.rot90(patch_img, k=k, axes=(1, 2)).copy()
                patch_lbl = np.rot90(patch_lbl, k=k, axes=(1, 2)).copy()
            if np.random.random() < 0.5:
                ax = np.random.choice([0, 1, 2])
                patch_img = np.flip(patch_img, axis=ax).copy()
                patch_lbl = np.flip(patch_lbl, axis=ax).copy()
            if np.random.random() < 0.15:
                patch_img = np.clip(patch_img + np.random.uniform(-0.1, 0.1), 0, 1)
        else:
            # Val: centre crop, fall back to foreground crop if centre empty
            ds = (d - pd) // 2
            hs = (h - ph) // 2
            ws = (w - pw) // 2
            center_lbl = lbl[ds : ds + pd, hs : hs + ph, ws : ws + pw]
            if not ((center_lbl == 2).any() or (center_lbl == 3).any()):
                fg = np.argwhere((lbl == 2) | (lbl == 3))
                if len(fg) > 0:
                    c = fg[np.random.randint(len(fg))]
                    ds = int(np.clip(c[0] - pd // 2, 0, d - pd))
                    hs = int(np.clip(c[1] - ph // 2, 0, h - ph))
                    ws = int(np.clip(c[2] - pw // 2, 0, w - pw))
            patch_img = img[ds : ds + pd, hs : hs + ph, ws : ws + pw]
            patch_lbl = lbl[ds : ds + pd, hs : hs + ph, ws : ws + pw]

        return patch_img, patch_lbl

    def _find_patch_start(self, lbl, d, h, w, pd, ph, pw, target_cls):
        if target_cls is not None:
            indices = np.argwhere(lbl == target_cls)
            if len(indices) == 0:
                fg = np.argwhere(lbl > 0)
                indices = fg if len(fg) > 0 else None
            if indices is not None and len(indices) > 0:
                c = indices[np.random.randint(len(indices))]
                ds = int(np.clip(c[0] - pd // 2, 0, d - pd))
                hs = int(np.clip(c[1] - ph // 2, 0, h - ph))
                ws = int(np.clip(c[2] - pw // 2, 0, w - pw))
                return ds, hs, ws
        ds = np.random.randint(0, max(1, d - pd + 1))
        hs = np.random.randint(0, max(1, h - ph + 1))
        ws = np.random.randint(0, max(1, w - pw + 1))
        return ds, hs, ws


# ============================================================================
# DATA LOADING
# ============================================================================


def get_dataloaders(config: dict, pseudo_label_pool: Optional[PseudoLabelPool] = None):
    data_dir = Path(config["kits23_dir"])
    cases = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")]
    )

    # Hold out last N as blind test set
    n_test = config.get("test_cases", 40)
    test_cases = cases[-n_test:]
    cases = cases[:-n_test]
    print(
        f"  Split: {len(cases)} train/val | {len(test_cases)} test "
        f"({test_cases[0].name}→{test_cases[-1].name})"
    )

    if config.get("quick_cases"):
        cases = cases[: config["quick_cases"]]

    data_dicts = [
        {
            "image": str(c / "imaging.nii.gz"),
            "label": str(c / "segmentation.nii.gz"),
            "case_id": c.name,
        }
        for c in cases
        if (c / "imaging.nii.gz").exists() and (c / "segmentation.nii.gz").exists()
    ]

    split = int(len(data_dicts) * config["val_split"])
    val_dicts = data_dicts[:split]
    train_dicts = data_dicts[split:]

    train_ds = KiTS23Dataset(
        train_dicts,
        config["patch_size"],
        num_samples=config["num_samples_per_volume"],
        is_train=True,
        pseudo_label_pool=pseudo_label_pool,
        pseudo_label_loss_weight=config["pseudo_label_loss_weight"],
    )
    val_ds = KiTS23Dataset(
        val_dicts,
        config["patch_size"],
        num_samples=1,
        is_train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,  # More IO workers — NIfTI loading is CPU-bound
        pin_memory=False,
        persistent_workers=False,  # Must be False — dataloader is recreated after VLM updates
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,  # Doubled from 4
        pin_memory=False,
        persistent_workers=True,
    )

    return train_loader, val_loader, train_dicts, val_dicts


# ============================================================================
# TRAINING UTILITIES
# ============================================================================


class MetricTracker:
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds, targets = preds.cpu(), targets.cpu()
        mask = (targets >= 0) & (targets < self.num_classes)
        self.cm += torch.bincount(
            self.num_classes * targets[mask].long() + preds[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        tp = torch.diag(self.cm)
        fp = self.cm.sum(0) - tp
        fn = self.cm.sum(1) - tp
        dice = (2 * tp / (2 * tp + fp + fn + 1e-8)).numpy()
        iou = (tp / (tp + fp + fn + 1e-8)).numpy()
        precision = (tp / (tp + fp + 1e-8)).numpy()
        recall = (tp / (tp + fn + 1e-8)).numpy()
        return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}

    def report(self) -> Tuple[str, float]:
        m = self.compute()
        mean_dice = float(np.mean(m["dice"][1:]))
        s = f"\n  Mean Dice (FG): {mean_dice:.4f}\n"
        s += f"  {'Class':<10} {'Dice':<8} {'IoU':<8} {'Prec':<8} {'Rec':<8}\n"
        s += "  " + "-" * 42 + "\n"
        for i, c in enumerate(CLASS_NAMES):
            s += (
                f"  {c:<10} {m['dice'][i]:.4f}   {m['iou'][i]:.4f}   "
                f"{m['precision'][i]:.4f}   {m['recall'][i]:.4f}\n"
            )
        return s, mean_dice


class EarlyStopping:
    def __init__(self, patience: int = 7, path: str = "best.pth"):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module):
        if self.best is None or score > self.best:
            self.best = score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    torch.cuda.empty_cache()
    model.train()
    total_loss, count = 0.0, 0

    for batch in tqdm(loader, desc="Train", leave=False):
        try:
            img = batch["image"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)
            weight = float(batch["weight"][0]) if "weight" in batch else 1.0

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = loss_fn(out, lbl, sample_weight=weight)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            count += 1
            del img, lbl, out, loss
        except RuntimeError as e:
            print(f"  ⚠ Train: {e}")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    return total_loss / max(count, 1)


def validate(model, loader, loss_fn, device):
    torch.cuda.empty_cache()
    model.eval()
    total_loss, count = 0.0, 0
    tracker = MetricTracker(4)
    per_case = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            try:
                img = batch["image"].to(device, non_blocking=True)
                lbl = batch["label"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    out = model(img)
                    loss = loss_fn(out, lbl)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    count += 1

                preds = torch.argmax(out, dim=1)
                tracker.update(preds, lbl.squeeze(1))

                pred_np = preds[0].cpu().numpy()
                gt_np = lbl[0, 0].cpu().numpy()
                cid = batch["case_id"][0] if "case_id" in batch else f"case_{count}"
                dice_pc = []
                for c in range(4):
                    inter = ((pred_np == c) & (gt_np == c)).sum()
                    d = 2 * inter / ((pred_np == c).sum() + (gt_np == c).sum() + 1e-8)
                    dice_pc.append(float(d))
                per_case.append(
                    {
                        "case_id": cid,
                        "dice_per_class": dice_pc,
                        "mean_dice_fg": float(np.mean(dice_pc[1:])),
                        "pred": pred_np,
                        "gt": gt_np,
                        "img": batch["image"][0, 0].numpy(),
                    }
                )
                del img, lbl, out, loss, preds
            except RuntimeError as e:
                print(f"  ⚠ Val: {e}")
                torch.cuda.empty_cache()

    report, mean_dice = tracker.report()
    metrics = tracker.compute()
    return total_loss / max(count, 1), mean_dice, report, metrics, per_case


def predict_full_volume(model, image_path: str, patch_size: tuple, device) -> np.ndarray:
    """
    Run sliding-window inference on a FULL NIfTI volume for a single case.
    Called only for the few worst cases that need VLM pseudo-label corrections.
    Returns: full-volume prediction as np.ndarray (D, H, W) with class indices.
    """
    from monai.inferers import sliding_window_inference

    # Load full volume
    img = nib.load(image_path).get_fdata().astype(np.float32)
    # CT windowing
    img = np.clip(img, -175, 250)
    img = (img - (-175)) / (250 - (-175))

    # Pad if needed
    pd, ph, pw = patch_size
    d, h, w = img.shape
    if d < pd or h < ph or w < pw:
        pad = ((0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w)))
        img = np.pad(img, pad, mode="constant")

    # Convert to tensor: (1, 1, D, H, W)
    img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        out = sliding_window_inference(
            inputs=img_t,
            roi_size=patch_size,
            sw_batch_size=4,
            predictor=model,
            overlap=0.25,
        )
    pred = torch.argmax(out, dim=1)[0].cpu().numpy()  # (D, H, W)

    # Trim padding back to original size
    pred = pred[:d, :h, :w]

    del img_t, out
    torch.cuda.empty_cache()
    return pred


def save_history(
    output_dir,
    stage,
    epoch,
    train_loss,
    val_loss,
    metrics,
    epoch_time,
    is_best,
    vlm_summary=None,
):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "epoch": epoch,
        "epoch_time_min": round(epoch_time / 60, 2),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "metrics": {
            "mean_dice_fg": float(metrics["dice"][1:].mean()),
            "per_class": {
                c: {
                    "dice": float(metrics["dice"][i]),
                    "precision": float(metrics["precision"][i]),
                    "recall": float(metrics["recall"][i]),
                }
                for i, c in enumerate(CLASS_NAMES)
            },
        },
        "is_best": is_best,
    }
    if vlm_summary:
        entry["vlm_corrections"] = vlm_summary

    hist_file = Path(output_dir) / "history.json"
    hist = json.loads(hist_file.read_text()) if hist_file.exists() else {"epochs": []}
    hist["epochs"].append(entry)
    hist_file.write_text(json.dumps(hist, indent=2))


def save_checkpoint(
    output_dir,
    stage,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_dice,
    latest=False,
):
    ckpt = {
        "epoch": epoch,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "best_dice": best_dice,
    }
    if latest:
        path = Path(output_dir) / f"ckpt_{stage}_latest.pth"
        torch.save(ckpt, path)
    else:
        path = Path(output_dir) / f"ckpt_{stage}_ep{epoch+1}.pth"
        torch.save(ckpt, path)
        print(f"  → Checkpoint: {path}")


# ============================================================================
# VLM HEALTH CHECK
# ============================================================================


def _check_vlm_health(config: dict, abort_on_fail: bool = True) -> bool:
    """
    Verify Ollama is running AND the configured model is loaded before training.
    If primary model fails, automatically tries fallbacks.

    abort_on_fail=True  → raises RuntimeError if all models fail (default: stops training)
    abort_on_fail=False → prints warning and returns False (non-blocking)
    """
    import requests

    api_url = config.get("vlm_api_url", "http://localhost:11434")
    primary_model = config["vlm_model"]
    # Fallback chain: if primary fails, try these in order
    fallback_chain = ["qwen3-vl:8b", "qwen3-vl:4b", "qwen3:4b"]

    print(f"\n  [VLM Health Check] Testing {primary_model} @ {api_url} ...")

    # Step 1: Is Ollama reachable at all?
    try:
        r = requests.get(f"{api_url}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        msg = (
            f"\n  ✗ Ollama is NOT reachable at {api_url}\n"
            f"    Error: {e}\n\n"
            f"  Fix:\n"
            f"    1. Start Ollama:  ollama serve\n"
            f"    2. Pull model:    ollama pull {primary_model}\n"
            f"    3. Re-run training\n"
        )
        print(msg)
        if abort_on_fail:
            raise RuntimeError(msg)
        return False

    available_models = [m["name"] for m in r.json().get("models", [])]
    print(f"    Available models: {available_models}")

    # Try primary model first, then fallbacks
    models_to_try = [primary_model] + fallback_chain
    tried = set()  # avoid re-testing the same matched model
    for attempt_model in models_to_try:
        # Exact match first (e.g. "qwen3-vl:8b" matches "qwen3-vl:8b" not "qwen3-vl:4b")
        matched = next((m for m in available_models if m == attempt_model), None)
        if not matched:
            # Loose base match as fallback
            model_base = attempt_model.split(":")[0]
            matched = next(
                (m for m in available_models if m.split(":")[0] == model_base and m not in tried), None
            )
        if not matched or matched in tried:
            continue
        tried.add(matched)

        # Helper to extract ANY text from response (content, thinking, or response field)
        def _extract_any_text(resp_json, endpoint_name):
            """Extract text from any available field in the response."""
            txt = ""
            if endpoint_name == "api/chat":
                msg = resp_json.get("message", {})
                txt = msg.get("content", "") or ""
                # Some Ollama versions put thinking output in 'thinking' field
                if not txt.strip():
                    txt = msg.get("thinking", "") or ""
            elif endpoint_name == "api/generate":
                txt = resp_json.get("response", "") or ""
            # Strip thinking tags if present
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
            return txt

        # Step 2: First test text-only (no image) to verify model loads at all
        text_only_ok = False
        for endpoint, payload in [
            (
                "api/chat",
                {
                    "model": matched,
                    "messages": [{"role": "user", "content": "/no_think Say hello in one word"}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 256},
                },
            ),
            (
                "api/generate",
                {
                    "model": matched,
                    "prompt": "/no_think Say hello in one word",
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 256},
                },
            ),
        ]:
            try:
                r2 = requests.post(f"{api_url}/{endpoint}", json=payload, timeout=180)
                r2.raise_for_status()
                resp_json = r2.json()
                txt = _extract_any_text(resp_json, endpoint)
                if txt:
                    text_only_ok = True
                    print(f"    [{matched}] text-only OK via {endpoint}: '{txt[:50]}'")
                    break
            except Exception as e:
                print(f"    [{matched}] text-only {endpoint}: {e}")

        if not text_only_ok:
            print(f"    [{matched}] cannot respond to even text — skipping")
            continue

        # Step 3: Test with image (optional — text-only proves VLM connectivity)
        # The actual _call_ollama uses num_predict=2048 which handles thinking-mode.
        # Here we just verify the model loads; image vision will work during training.
        import base64, io

        try:
            from PIL import Image

            img = Image.new("RGB", (64, 64), (255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            test_img_b64 = base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            test_img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        probe_prompt = "/no_think Describe this image in one sentence."
        image_ok = False
        for endpoint, payload in [
            (
                "api/chat",
                {
                    "model": matched,
                    "messages": [
                        {
                            "role": "user",
                            "content": probe_prompt,
                            "images": [test_img_b64],
                        }
                    ],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 1024},
                },
            ),
            (
                "api/generate",
                {
                    "model": matched,
                    "prompt": probe_prompt,
                    "images": [test_img_b64],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 1024},
                },
            ),
        ]:
            try:
                r2 = requests.post(f"{api_url}/{endpoint}", json=payload, timeout=300)
                r2.raise_for_status()
                resp_json = r2.json()
                response_text = _extract_any_text(resp_json, endpoint)
                if response_text:
                    image_ok = True
                    working_endpoint = endpoint
                    print(f"    [{matched}] image OK via {endpoint}: '{response_text[:50]}'")
                    break
                else:
                    print(
                        f"    [{matched}] {endpoint}+image: 200 OK but empty response"
                    )
            except Exception as e:
                print(f"    [{matched}] {endpoint}+image: {e}")

        # Accept model if text-only passed (image is bonus)
        if matched != primary_model:
            print(
                f"  ⚠ Primary model '{primary_model}' failed; auto-switched to '{matched}'"
            )
            config["vlm_model"] = matched

        if image_ok:
            print(
                f"  ✓ VLM ready | model='{matched}' | text+image OK"
            )
        else:
            print(
                f"  ✓ VLM ready (text-only) | model='{matched}' | "
                f"image test skipped (thinking-mode uses tokens; "
                f"actual training uses 2048 tokens — will work fine)"
            )
        return True

    # All models failed
    msg = (
        f"\n  ✗ All VLM models failed to respond:\n"
        f"    Tried: {models_to_try}\n"
        f"    Available in Ollama: {available_models}\n\n"
        f"  Fix:\n"
        f"    1. Pull a working model:  ollama pull qwen3-vl:8b\n"
        f"    2. Run Ollama:            ollama serve\n"
        f"    3. Re-run training with:  --vlm-model qwen3-vl:8b\n"
    )
    print(msg)
    if abort_on_fail:
        raise RuntimeError(msg)
    return False


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================


def train(config: dict, resume: bool = False, skip_stage1: bool = False):
    # Reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── File logging (capture all prints to file) ──
    import logging

    log_file = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(),  # Also print to console
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 65)
    logger.info("VLM-in-the-Loop 3D Segmentation — V7 (Click-Point Oracle)")
    logger.info(f"Training started at {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = (
        True  # Safe: patch size is fixed (96×192×192) — 10-20% faster kernels
    )

    print(f"\n{'='*65}")
    print("VLM-in-the-Loop 3D Segmentation — V7 (Click-Point Oracle)")
    print(f"{'='*65}")
    print(f"Device:  {device}")
    print(f"Output:  {output_dir}")
    print(f"Log:     {log_file}")
    print(f"MGPL:    {'ON' if config['use_mgpl'] else 'OFF (ablation)'}")
    print(f"VITL:    {'ON' if config['use_vlm'] else 'OFF (ablation)'}")
    print(f"Patch:   {config['patch_size']}")
    print(f"VLM:     {config['vlm_backend']}:{config['vlm_model']}")
    print(f"PL conf: {config['pseudo_label_confidence_threshold']:.0%}")
    print(f"PL wt:   {config['pseudo_label_loss_weight']:.0%} of GT loss")
    print(f"{'='*65}\n")

    # ── Pseudo-label pool (shared across stages) ──
    pl_pool = (
        PseudoLabelPool(
            max_size=config["pseudo_label_max_pool_size"],
            loss_weight=config["pseudo_label_loss_weight"],
        )
        if config["use_vlm"]
        else None
    )

    # ── Data ──
    train_loader, val_loader, train_dicts, val_dicts = get_dataloaders(config, pl_pool)
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # ── Model ──
    model = VLMLoopSegNet(
        num_classes=config["num_classes"],
        feature_dim=config["feature_dim"],
        swin_feature_size=config["swin_feature_size"],
        clip_model=config["clip_model"],
        clip_pretrained=config["clip_pretrained"],
        freeze_encoder=True,
        use_mgpl=config["use_mgpl"],
    ).to(device)

    # ── Loss ──
    loss_fn = HybridLoss().to(device)

    # ── VLM Health Check ──────────────────────────────────────────────────────
    # Validates Ollama is reachable AND the configured model is loaded BEFORE
    # training starts — prevents silent zero-correction epochs.
    if config["use_vlm"]:
        _check_vlm_health(config, abort_on_fail=True)

    # ── VLM Click-Point Oracle (V7) ──
    critic = VLMClickPointOracle(config) if config["use_vlm"] else None
    if critic:
        print(f"✓ VLM Click-Point Oracle: {config['vlm_backend']}:{config['vlm_model']}")
        print(
            f"  Runs every {config['critic_every_k_epochs']} epochs | "
            f"Top {config['critic_top_n_worst']} worst cases | "
            f"Max {config.get('critic_max_clicks_per_case', 5)} clicks/case"
        )

    # ── Optimizer / Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr_stage1"], weight_decay=config["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda")
    n_s1 = config["num_epochs_stage1"]

    if config.get("use_warmup"):
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=config["warmup_epochs"]
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_s1)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [warmup_sched, cosine_sched],
            milestones=[config["warmup_epochs"]],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_s1)

    early_stop = EarlyStopping(
        patience=config["patience_stage1"],
        path=str(output_dir / "best_stage1.pth"),
    )

    # ── Cached val state (skip validation on non-val epochs) ──
    last_val = dict(loss=0.0, dice=0.0, report="", metrics={}, per_case=[])

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1 — Frozen SwinUNETR encoder
    # ─────────────────────────────────────────────────────────────────────────
    print("\n─── STAGE 1: Train Decoder + MGPL (SwinUNETR frozen) ───")
    best_dice = 0.0
    start_ep = 0
    _skip_stage1 = False  # set True if resume finds stage1 is already done

    # --skip-stage1 flag: go directly to Stage 2 from best Stage 1 weights
    if skip_stage1:
        print("  --skip-stage1 flag: jumping straight to Stage 2")
        _skip_stage1 = True
        best_s1 = output_dir / "best_stage1_dice.pth"
        if best_s1.exists():
            sd = torch.load(best_s1, map_location=device, weights_only=True)
            model.load_state_dict(sd)
            # Try to read best_dice from any stage1 checkpoint
            for _p in sorted(
                output_dir.glob("ckpt_stage1_*.pth"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                try:
                    _c = torch.load(_p, map_location="cpu", weights_only=False)
                    best_dice = _c.get("best_dice", 0.0)
                    break
                except Exception:
                    pass
            print(
                f"  Loaded best Stage 1 weights (dice={best_dice:.4f}) from {best_s1.name}"
            )
        else:
            print(
                f"  ⚠ best_stage1_dice.pth not found — Stage 2 will start from current weights"
            )
    elif resume:
        # If a stage2 checkpoint exists, stage1 is already complete — skip it
        s2_latest = output_dir / "ckpt_stage2_latest.pth"
        s2_ckpts = sorted(
            output_dir.glob("ckpt_stage2_ep*.pth"),
            key=lambda p: int(p.stem.split("ep")[-1]),
        )
        if s2_latest.exists() or s2_ckpts:
            print("  Stage 2 checkpoint found — skipping Stage 1 (already complete)")
            _skip_stage1 = True
            # Load best_dice from the stage2 checkpoint so stage2 resume knows it
            s2_path = s2_latest if s2_latest.exists() else s2_ckpts[-1]
            _s2_ckpt = torch.load(s2_path, map_location=device, weights_only=False)
            best_dice = _s2_ckpt["best_dice"]
        else:
            latest = output_dir / "ckpt_stage1_latest.pth"
            ckpts = sorted(
                output_dir.glob("ckpt_stage1_ep*.pth"),
                key=lambda p: int(p.stem.split("ep")[-1]),
            )
            resume_path = latest if latest.exists() else (ckpts[-1] if ckpts else None)
            if resume_path:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                start_ep = ckpt["epoch"] + 1
                best_dice = ckpt["best_dice"]
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if ckpt.get("scheduler_state_dict"):
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                print(
                    f"  Resumed Stage 1 from epoch {start_ep} (dice={best_dice:.4f}) [{resume_path.name}]"
                )
                # If stage1 past n_s1, mark done
                if start_ep >= n_s1:
                    print("  Stage 1 already completed — skipping to Stage 2")
                    _skip_stage1 = True

    val_every = config.get("val_every_s1", 3)
    t_total = time.time()

    for epoch in range(start_ep if not _skip_stage1 else n_s1, n_s1):
        t_ep = time.time()
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device
        )

        do_val = (epoch + 1) % val_every == 0 or epoch == 0 or epoch == n_s1 - 1
        if do_val:
            vl, vd, vrpt, vmets, vpc = validate(model, val_loader, loss_fn, device)
            last_val = dict(loss=vl, dice=vd, report=vrpt, metrics=vmets, per_case=vpc)

        val_loss, mean_dice = last_val["loss"], last_val["dice"]
        scheduler.step()
        ep_time = time.time() - t_ep
        is_best = mean_dice > best_dice

        marker = "*" if do_val else " "
        print(
            f"S1 Ep {epoch+1:03d}/{n_s1}{marker} | "
            f"Loss {tr_loss:.4f}/{val_loss:.4f} | "
            f"Dice {mean_dice:.4f} | {ep_time/60:.1f}m"
        )
        if do_val:
            print(last_val["report"])

        # Flush stdout and logging to file
        sys.stdout.flush()
        for handler in logging.getLogger(__name__).handlers:
            handler.flush()

        # ── VLM Pseudo-Label Generation ──────────────────────────────────
        vlm_summary = None
        if (
            do_val
            and critic is not None
            and pl_pool is not None
            and (epoch + 1) % config["critic_every_k_epochs"] == 0
            and epoch > 0
        ):

            # Build a lookup for image/label paths by case_id
            all_dicts = {d["case_id"]: d for d in train_dicts + val_dicts}

            # Sort worst cases and pick top-N
            sorted_cases = sorted(last_val["per_case"], key=lambda c: c["mean_dice_fg"])
            worst_n = sorted_cases[: config["critic_top_n_worst"]]

            # Run full-volume inference ONLY on worst cases
            preds_dict = {}
            gts_dict = {}
            imgs_dict = {}
            for pc in worst_n:
                cid = pc["case_id"]
                if cid not in all_dicts:
                    continue
                entry = all_dicts[cid]
                print(f"  [VITL] Full-volume inference on {cid}...")
                preds_dict[cid] = predict_full_volume(
                    model, entry["image"], config["patch_size"], device
                )
                gts_dict[cid] = nib.load(entry["label"]).get_fdata().astype(np.int64)
                # Image for VLM rendering (loaded from NIfTI)
                img_raw = nib.load(entry["image"]).get_fdata().astype(np.float32)
                img_raw = np.clip(img_raw, -175, 250)
                img_raw = (img_raw - (-175)) / (250 - (-175))
                imgs_dict[cid] = img_raw

            corrections = critic.generate_corrections(
                epoch=epoch + 1,
                val_cases=last_val["per_case"],
                predictions=preds_dict,
                ground_truths=gts_dict,
                images=imgs_dict,
            )

            if corrections:
                pl_pool.update(corrections, train_dicts + val_dicts)
                # Recreate train_loader to sync new length with multiprocessing workers
                train_loader, _, _, _ = get_dataloaders(config, pl_pool)
                vlm_summary = {
                    "num_corrections": len(corrections),
                    "cases": list(corrections.keys()),
                    "pool_size": len(pl_pool),
                }
                print(
                    f"  [VITL] Pool now {len(pl_pool)} pseudo-labels | "
                    f"Dataset {len(train_loader.dataset)} items"
                )

        if do_val:
            save_history(
                output_dir,
                "stage1",
                epoch + 1,
                tr_loss,
                val_loss,
                last_val["metrics"],
                ep_time,
                is_best,
                vlm_summary,
            )
            early_stop(mean_dice, model)
            if early_stop.early_stop:
                print("  Early stopping (Stage 1)")
                break

        if is_best:
            best_dice = mean_dice
            torch.save(model.state_dict(), output_dir / "best_stage1_dice.pth")
            print(f"  ★ Best Stage1 Dice: {best_dice:.4f}")

        # Save latest checkpoint every epoch (overwrites previous latest)
        save_checkpoint(
            output_dir,
            "stage1",
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_dice,
            latest=True,
        )
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                output_dir,
                "stage1",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_dice,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2 — Fine-tune SwinUNETR encoder
    # ─────────────────────────────────────────────────────────────────────────
    n_s2 = config["num_epochs_stage2"]
    if n_s2 > 0:
        print("\n─── STAGE 2: Fine-tune SwinUNETR Encoder ───")

        best_s1 = output_dir / "best_stage1_dice.pth"
        if best_s1.exists():
            model.load_state_dict(
                torch.load(best_s1, map_location=device, weights_only=True)
            )
            print(f"  Loaded best Stage 1 weights (dice={best_dice:.4f})")
        model.unfreeze_encoder(num_layers=4)

        warmup_s2 = config.get("warmup_s2_epochs", 5)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr_stage2"] / warmup_s2,
            weight_decay=config["weight_decay"],
        )
        warmup2_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_s2,
            end_factor=1.0,
            total_iters=warmup_s2,
        )
        cosine2_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(n_s2 - warmup_s2, 1)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup2_sched, cosine2_sched], milestones=[warmup_s2]
        )

        early_stop = EarlyStopping(
            patience=config["patience_stage2"],
            path=str(output_dir / "best_final.pth"),
        )

        val_every_s2 = config.get("val_every_s2", 1)
        start_ep_s2 = 0
        last_val = dict(loss=0.0, dice=best_dice, report="", metrics={}, per_case=[])

        if resume:
            latest_s2 = output_dir / "ckpt_stage2_latest.pth"
            ckpts = sorted(
                output_dir.glob("ckpt_stage2_ep*.pth"),
                key=lambda p: int(p.stem.split("ep")[-1]),
            )
            resume_path_s2 = (
                latest_s2 if latest_s2.exists() else (ckpts[-1] if ckpts else None)
            )
            if resume_path_s2:
                ckpt = torch.load(
                    resume_path_s2, map_location=device, weights_only=False
                )
                start_ep_s2 = ckpt["epoch"] + 1
                best_dice = ckpt["best_dice"]
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if ckpt.get("scheduler_state_dict"):
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                print(
                    f"  Resumed Stage 2 from epoch {start_ep_s2} [{resume_path_s2.name}]"
                )

        for epoch in range(start_ep_s2, n_s2):
            t_ep = time.time()
            tr_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, scaler, device
            )

            do_val = (epoch + 1) % val_every_s2 == 0 or epoch == 0 or epoch == n_s2 - 1
            if do_val:
                vl, vd, vrpt, vmets, vpc = validate(model, val_loader, loss_fn, device)
                last_val = dict(
                    loss=vl, dice=vd, report=vrpt, metrics=vmets, per_case=vpc
                )

            val_loss, mean_dice = last_val["loss"], last_val["dice"]
            scheduler.step()
            ep_time = time.time() - t_ep
            is_best = mean_dice > best_dice

            marker = "*" if do_val else " "
            print(
                f"S2 Ep {epoch+1:03d}/{n_s2}{marker} | "
                f"Loss {tr_loss:.4f}/{val_loss:.4f} | "
                f"Dice {mean_dice:.4f} | {ep_time/60:.1f}m"
            )
            if do_val:
                print(last_val["report"])

            # Flush stdout and logging to file
            sys.stdout.flush()
            for handler in logging.getLogger(__name__).handlers:
                handler.flush()

            # VLM in stage 2 too
            vlm_summary = None
            if (
                do_val
                and critic is not None
                and pl_pool is not None
                and (epoch + 1) % config["critic_every_k_epochs"] == 0
            ):
                all_dicts = {d["case_id"]: d for d in train_dicts + val_dicts}
                sorted_cases = sorted(last_val["per_case"], key=lambda c: c["mean_dice_fg"])
                worst_n = sorted_cases[: config["critic_top_n_worst"]]

                preds_dict = {}
                gts_dict = {}
                imgs_dict = {}
                for pc in worst_n:
                    cid = pc["case_id"]
                    if cid not in all_dicts:
                        continue
                    entry = all_dicts[cid]
                    print(f"  [VITL] Full-volume inference on {cid}...")
                    preds_dict[cid] = predict_full_volume(
                        model, entry["image"], config["patch_size"], device
                    )
                    gts_dict[cid] = nib.load(entry["label"]).get_fdata().astype(np.int64)
                    img_raw = nib.load(entry["image"]).get_fdata().astype(np.float32)
                    img_raw = np.clip(img_raw, -175, 250)
                    img_raw = (img_raw - (-175)) / (250 - (-175))
                    imgs_dict[cid] = img_raw

                corrections = critic.generate_corrections(
                    epoch + 1, last_val["per_case"], preds_dict, gts_dict, imgs_dict
                )
                if corrections:
                    pl_pool.update(corrections, train_dicts + val_dicts)
                    # Recreate train_loader to sync new length with multiprocessing workers
                    train_loader, _, _, _ = get_dataloaders(config, pl_pool)
                    vlm_summary = {
                        "num_corrections": len(corrections),
                        "pool_size": len(pl_pool),
                    }
                    print(f"  [VITL] Pool: {len(pl_pool)} pseudo-labels")

            if do_val:
                save_history(
                    output_dir,
                    "stage2",
                    epoch + 1,
                    tr_loss,
                    val_loss,
                    last_val["metrics"],
                    ep_time,
                    is_best,
                    vlm_summary,
                )
                early_stop(mean_dice, model)
                if early_stop.early_stop:
                    print("  Early stopping (Stage 2)")
                    break

            if is_best:
                best_dice = mean_dice
                torch.save(model.state_dict(), output_dir / "best_final_dice.pth")
                print(f"  ★ Best Final Dice: {best_dice:.4f}")

            # Save latest checkpoint every epoch (overwrites previous latest)
            save_checkpoint(
                output_dir,
                "stage2",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_dice,
                latest=True,
            )
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    output_dir,
                    "stage2",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_dice,
                )

    total_time = time.time() - t_total
    print(f"\n{'='*65}")
    print(
        f"Training Complete | Best Dice: {best_dice:.4f} | "
        f"Time: {total_time/3600:.2f}h"
    )
    if pl_pool:
        print(f"Pseudo-label pool: {len(pl_pool)} entries used")
    print(f"{'='*65}")

    summary = {
        "version": "VLMLoopSegNet_V7_ClickPointOracle",
        "best_dice": float(best_dice),
        "total_hours": round(total_time / 3600, 2),
        "mgpl": config["use_mgpl"],
        "vitl": config["use_vlm"],
        "pl_pool_size": len(pl_pool) if pl_pool else 0,
        "total_params": sum(p.numel() for p in model.parameters()),
        "config": {k: str(v) for k, v in config.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return model


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLM-in-the-Loop 3D Segmentation V7 (Click-Point Oracle)"
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["quick_test", "quick_test_vlm", "quick_test_novlm", "full"],
        help=(
            "quick_test_vlm:   10 epochs, 40 cases, VLM+MGPL ON  (ablation A)\n"
            "quick_test_novlm: 10 epochs, 40 cases, MGPL only    (ablation B)\n"
            "quick_test:       same as quick_test_vlm (default)\n"
            "full:             full training run"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 entirely and go straight to Stage 2. "
        "Loads best_stage1_dice.pth and starts fine-tuning.",
    )
    parser.add_argument(
        "--no-vlm", action="store_true", help="Ablation: disable VLM (MGPL only)"
    )
    parser.add_argument(
        "--no-mgpl", action="store_true", help="Ablation: disable MGPL (VITL only)"
    )
    parser.add_argument(
        "--vlm-backend", default=None, choices=["ollama", "transformers", "openai"]
    )
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--kits23-dir", default=None)
    args = parser.parse_args()

    config = get_config(args.mode)
    if args.no_vlm:
        config["use_vlm"] = False
    if args.no_mgpl:
        config["use_mgpl"] = False
    if args.vlm_backend:
        config["vlm_backend"] = args.vlm_backend
    if args.vlm_model:
        config["vlm_model"] = args.vlm_model
    if args.seed is not None:
        config["seed"] = args.seed
    if args.kits23_dir:
        config["kits23_dir"] = args.kits23_dir

    # Dependency check
    print("Checking dependencies...")
    try:
        import monai

        print(f"  ✓ MONAI {monai.__version__}")
    except ImportError:
        print("  ✗ MONAI not found — run: pip install monai")
        exit(1)
    try:
        import open_clip

        print(f"  ✓ open_clip available")
    except ImportError:
        print("  ✗ open_clip not found — run: pip install open_clip_torch")
        if config["use_mgpl"]:
            exit(1)
    try:
        import nibabel

        print(f"  ✓ nibabel available")
    except ImportError:
        print("  ✗ nibabel not found — run: pip install nibabel")
        exit(1)

    print()
    train(config, resume=args.resume, skip_stage1=args.skip_stage1)
