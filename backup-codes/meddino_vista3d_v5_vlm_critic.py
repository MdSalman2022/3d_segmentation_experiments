"""
MedDINO-VISTA3D V5 — VLM-as-Critic + Multi-Granularity Prompt Learning
========================================================================

NOVEL CONTRIBUTIONS (Q1 Journal Target):
1. Multi-Granularity Prompt Learning (MGPL): LLM-generated hierarchical
   anatomical descriptions encoded via CLIP, aligned with visual features
   at every decoder scale via a learnable projection function ψ.
2. VLM-in-the-Loop Training (VITL): A local Vision-Language Model acts
   as an automated critic every K epochs, analyzing segmentation errors
   and producing structured feedback that adaptively modifies loss weights,
   sampling strategy, and spatial attention.

ARCHITECTURE:
   ┌──────────────┐
   │  DINOv2 ViT  │ (frozen, 2D slices → 3D adaptor)
   │  Multi-scale  │ Layers [2,5,8,11] → skip connections
   └──────┬───────┘
          │ per-scale features
          ▼
   ┌──────────────┐     ┌──────────────────┐
   │  3D CNN      │────▶│ Cross-Attention   │◀── CLIP Multi-Granularity
   │  Encoder     │     │ Fusion + ψ align  │    Text Embeddings
   └──────────────┘     └───────┬──────────┘
                                │
                         ┌──────▼──────┐
                         │  U-Net      │
                         │  Decoder    │ ← text-aligned skip connections
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │ Segmentation│
                         │ Head        │
                         └─────────────┘

   Every K epochs:
   ┌─────────────────────────────────────────────┐
   │  VLM CRITIC (Qwen2-VL / LLaVA)             │
   │  Renders worst val cases → structured JSON  │
   │  → Adapts: loss weights, sampling,          │
   │    spatial attention, curriculum             │
   └─────────────────────────────────────────────┘

Dataset: KiTS23 (Kidney Tumor Segmentation)
Classes: 0=Background, 1=Kidney, 2=Tumor, 3=Cyst

Usage:
    python meddino_vista3d_v5_vlm_critic.py --mode quick_test
    python meddino_vista3d_v5_vlm_critic.py --mode full
    python meddino_vista3d_v5_vlm_critic.py --mode full --resume
    python meddino_vista3d_v5_vlm_critic.py --mode full --no-vlm-critic  # Ablation: MGPL only
    python meddino_vista3d_v5_vlm_critic.py --mode full --no-mgpl        # Ablation: VITL only
"""

import os
import json
import time
import base64
import io
import re
from datetime import datetime
import sys
import argparse
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
from einops import rearrange

# Settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Disable caching allocator IPC — fixes CUDACachingAllocator:427 on WSL2
# (pin_memory workers create handles in child CUDA contexts that get destroyed
# on worker exit, leaving dangling IPC handles the parent can't reclaim)
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["XFORMERS_DISABLED"] = "1"


# ============================================================================
# CONFIGURATION
# ============================================================================


def get_config(mode: str) -> dict:
    """Returns config based on mode (quick_test or full)."""
    base = {
        # Paths
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/meddino_vista3d_v5_vlm_critic",
        # Model
        "dinov2_backbone": "dinov2_vitb14",
        "hf_token": None,
        "num_classes": 4,
        "feature_dim": 256,
        "cnn_channels": [32, 64, 128, 256],
        # Training
        "batch_size": 1,
        "val_split": 0.15,  # use 15% of (train+val pool) for validation
        "num_samples_per_volume": 2,
        "patch_size": (
            96,   # Reduced from 140: V5 is 2× heavier than V4 (266M vs 130M params)
            192,  # Divisible by 14 after padding (14×14=196); fits 32GB with V5 arch
            192,
        ),  # ~50% of V4 volume — safe for V5's dual-branch + CLIP + cross-attn
        "lr_stage1": 1e-4,
        "lr_stage2": 5e-6,
        "weight_decay": 1e-5,
        # CLIP
        "clip_model": "ViT-B-32",
        "clip_pretrained": "openai",
        # VLM Critic
        "vlm_backend": "ollama",  # "ollama", "transformers", or "openai"
        "vlm_model": "qwen3-vl:4b",  # 4B is enough for critic, saves ~10GB vs 8B
        "vlm_api_url": "http://localhost:11434",  # Ollama API endpoint
        "critic_every_k_epochs": 5,  # Run VLM critic every K epochs
        "critic_top_n_worst": 10,  # Analyze top N worst cases
        "critic_num_slices": 3,  # Number of slices to show per case
        # Multi-Granularity Prompts
        "prompt_levels": 5,  # L1-L5 granularity levels
        "use_mgpl": True,  # Enable MGPL module
        "use_vlm_critic": True,  # Enable VLM critic
        # Split
        "test_cases": 40,  # Last N cases always held out as blind test — never used in train/val
        # Reproducibility
        "seed": 42,
    }

    if mode == "quick_test":
        base.update(
            {
                "num_epochs_stage1": 15,
                "num_epochs_stage2": 10,
                "quick_cases": 50,
                "use_warmup": False,
                "patience_stage1": 5,
                "patience_stage2": 5,
                "critic_every_k_epochs": 3,
                "critic_top_n_worst": 5,
                "warmup_s2_epochs": 2,
                "val_every_s1": 2,  # validate every 2 epochs in stage 1
                "val_every_s2": 1,  # validate every epoch in stage 2
            }
        )
    else:  # full
        base.update(
            {
                "num_epochs_stage1": 20,
                "num_epochs_stage2": 10,
                "quick_cases": None,
                "use_warmup": True,
                "warmup_epochs": 5,
                "warmup_s2_epochs": 5,
                "patience_stage1": 7,  # 20 epochs total → 7 gives ~35% window
                "patience_stage2": 4,
                "val_every_s1": 3,  # validate every 3 epochs in stage 1 (~66% less val time)
                "val_every_s2": 1,  # validate every epoch in stage 2 (only 40 epochs)
            }
        )
    return base


# ============================================================================
# MULTI-GRANULARITY PROMPT BANK (MGPL)
# ============================================================================

# The prompt bank is generated ONE TIME using an LLM. These descriptions
# encode hierarchical anatomical knowledge at 5 granularity levels.
# In a paper, you would describe this as "LLM-generated" prompts.

PROMPT_BANK = {
    "background": {
        "L1_organ": "background tissue",
        "L2_pathology": "non-organ background regions including muscle, fat, and air in a CT scan",
        "L3_appearance": "heterogeneous tissue with variable density, typically dark air or grey soft tissue",
        "L4_boundary": "no defined boundary, fills all space not occupied by organs or pathology",
        "L5_context": "surrounds all abdominal organs, includes retroperitoneal fat, psoas muscles, subcutaneous tissue, and bowel",
    },
    "kidney": {
        "L1_organ": "kidney",
        "L2_pathology": "normal kidney, also known as renal parenchyma, a paired retroperitoneal organ",
        "L3_appearance": "a homogeneous enhancing bean-shaped organ with cortex and medulla visible on contrast CT",
        "L4_boundary": "well-defined smooth capsular margin separating kidney from perinephric fat, distinct cortical rim",
        "L5_context": "located in the retroperitoneum at T12-L3 level, right kidney slightly lower than left, adjacent to adrenal gland superiorly, contains hilum with renal artery and vein medially",
    },
    "tumor": {
        "L1_organ": "kidney tumor",
        "L2_pathology": "kidney tumor, most commonly renal cell carcinoma, a malignant neoplasm of the renal parenchyma",
        "L3_appearance": "a heterogeneous enhancing mass within the kidney, may contain areas of necrosis appearing as low-density regions, often with irregular internal architecture",
        "L4_boundary": "irregular or lobulated margins, may show pseudocapsule, can invade perinephric fat or renal sinus, boundary with normal parenchyma often shows contrast difference",
        "L5_context": "typically arises from the renal cortex, may be exophytic extending beyond kidney contour, can distort the collecting system, small tumors less than 1 centimeter may appear as subtle focal hyperdensity or hypodensity requiring careful window adjustment to detect",
    },
    "cyst": {
        "L1_organ": "kidney cyst",
        "L2_pathology": "simple kidney cyst, a benign fluid-filled sac arising from the renal parenchyma, also known as a simple renal cyst",
        "L3_appearance": "a well-circumscribed homogeneous hypodense round or oval lesion with water density between 0 and 20 Hounsfield units, no internal enhancement",
        "L4_boundary": "sharp smooth thin-walled boundary with imperceptible wall, clear demarcation from adjacent renal parenchyma, no irregular margins",
        "L5_context": "typically cortical in location, can be single or multiple, may be parapelvic near the renal sinus, does not enhance after contrast administration unlike tumors, may cause mild mass effect on adjacent structures when large",
    },
}

CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]


class MultiGranularityPromptEncoder(nn.Module):
    """
    Encodes the Multi-Granularity Prompt Bank using a frozen CLIP text encoder.
    Produces a prompt embedding tensor of shape (num_classes, num_levels, embed_dim).

    This module is frozen — CLIP weights are never trained. The embeddings
    serve as fixed semantic priors that guide the segmentation decoder via
    learnable alignment projections.
    """

    def __init__(self, clip_model_name: str = "ViT-B-32", pretrained: str = "openai"):
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.text_encoder = model  # We only use encode_text
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Pre-compute all embeddings
        self._build_prompt_embeddings()

        # Expose dim
        self.embed_dim = self.prompt_embeddings.shape[-1]  # 512 for ViT-B-32
        print(
            f"✓ MGPL: Encoded {self.prompt_embeddings.shape[0]} classes × "
            f"{self.prompt_embeddings.shape[1]} levels → dim={self.embed_dim}"
        )

    @torch.no_grad()
    def _build_prompt_embeddings(self):
        """Encode all prompts from the bank into CLIP embeddings."""
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
            for level_key in level_keys:
                text = (
                    f"A computerized tomography of {PROMPT_BANK[cls_name][level_key]}"
                )
                tokens = self.tokenizer([text])
                emb = self.text_encoder.encode_text(tokens)
                emb = F.normalize(emb, dim=-1)
                cls_embeddings.append(emb.squeeze(0))
            all_embeddings.append(torch.stack(cls_embeddings))

        # Shape: (num_classes=4, num_levels=5, embed_dim=512)
        self.register_buffer("prompt_embeddings", torch.stack(all_embeddings))

    def forward(self) -> torch.Tensor:
        """Returns prompt embeddings: (num_classes, num_levels, embed_dim)."""
        return self.prompt_embeddings


# ============================================================================
# TEXT-VISUAL ALIGNMENT FUNCTION ψ
# ============================================================================


class TextVisualAlignment(nn.Module):
    """
    Alignment function ψ(v, t) from CDPDNet (Eq. 6):
        ψ(v, t) = (W_a · t + b_a) ⊙ v + W_b · t + b_b

    The text embedding modulates visual features via learned affine transforms.
    This is applied at each decoder scale independently.
    """

    def __init__(self, visual_dim: int, text_dim: int = 512):
        super().__init__()
        # Multiplicative path: text → scale for visual features
        self.W_a = nn.Linear(text_dim, visual_dim)
        # Additive path: text → bias for visual features
        self.W_b = nn.Linear(text_dim, visual_dim)

    def forward(
        self, visual_feat: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_feat: (B, C, D, H, W) — visual features at some decoder level
            text_emb:    (C_text,)  or (num_classes, C_text) — text embedding(s)
        Returns:
            aligned:     (B, C, D, H, W) — text-aligned visual features
        """
        # If text_emb is per-class, average over classes for spatial modulation
        if text_emb.dim() == 2:
            text_emb = text_emb.mean(dim=0)  # (C_text,)

        scale = self.W_a(text_emb)  # (visual_dim,)
        bias = self.W_b(text_emb)  # (visual_dim,)

        # Reshape for broadcasting: (1, C, 1, 1, 1)
        scale = scale.view(1, -1, 1, 1, 1)
        bias = bias.view(1, -1, 1, 1, 1)

        return scale * visual_feat + bias


# ============================================================================
# PROMPT ATTENTION MODULE (Novel Contribution)
# ============================================================================


class PromptAttentionModule(nn.Module):
    """
    Novel Prompt Attention Module for Multi-Granularity Prompt Learning.

    At each decoder scale, visual features attend to ALL prompt levels.
    Learnable attention weights select which granularity is most useful:
      - Low-resolution features (bottleneck) → L1-L2 (organ/pathology)
      - High-resolution features (skip)      → L4-L5 (boundary/context)

    The model LEARNS which text descriptions help at which spatial scale.
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
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.num_levels = num_levels

        # Project text embeddings to visual dim
        self.text_proj = nn.Linear(text_dim, visual_dim)

        # Cross-attention: visual queries attend to text keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Learnable level-importance weights (initialized uniform, learned during training)
        # These weights determine which granularity level matters at this scale
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
        )

    def forward(
        self,
        visual_feat: torch.Tensor,
        prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_feat:       (B, C, D, H, W) — visual features at some scale
            prompt_embeddings: (num_classes, num_levels, text_dim) — from MGPL
        Returns:
            enhanced:          (B, C, D, H, W) — prompt-enhanced visual features
        """
        B, C, D, H, W = visual_feat.shape

        # Weighted combination of prompt levels
        weights = F.softmax(self.level_weights, dim=0)  # (num_levels,)
        # (num_classes, num_levels, text_dim) × (num_levels,) → (num_classes, text_dim)
        weighted_prompts = torch.einsum("clt, l -> ct", prompt_embeddings, weights)

        # Project to visual dim: (num_classes, visual_dim)
        text_kv = self.text_proj(weighted_prompts)  # (num_classes, visual_dim)

        # Flatten spatial dims for cross-attention
        # (B, C, D*H*W) → (B, D*H*W, C)
        vis_flat = visual_feat.flatten(2).permute(0, 2, 1)  # (B, N, C)

        # Expand text for batch: (num_classes, C) → (B, num_classes, C)
        text_kv_batch = text_kv.unsqueeze(0).expand(B, -1, -1)

        # Cross attention: visual attends to text prompts
        attended, _ = self.cross_attn(
            query=vis_flat,
            key=text_kv_batch,
            value=text_kv_batch,
        )  # (B, N, C)

        # Residual connection + projection
        enhanced = vis_flat + self.out_proj(attended)

        # Reshape back: (B, N, C) → (B, C, D, H, W)
        enhanced = enhanced.permute(0, 2, 1).view(B, C, D, H, W)

        return enhanced


# ============================================================================
# 3D CNN ENCODER (Parallel Branch)
# ============================================================================


class CNN3DEncoder(nn.Module):
    """
    Lightweight 3D CNN encoder that processes volumes natively in 3D.
    Runs in parallel with DINOv2 to capture 3D spatial context that
    the 2D ViT misses.

    Architecture follows CDPDNet Table I style:
      Stem → 4 ConvBlocks with progressive downsampling.
    """

    def __init__(self, in_channels: int = 1, channels: List[int] = [32, 64, 128, 256]):
        super().__init__()
        self.channels = channels

        # Stem block
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels[0]),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels[0]),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # Convolutional blocks with downsampling (stride-2 conv)
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1
                    ),
                    nn.InstanceNorm3d(channels[i + 1]),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Conv3d(
                        channels[i + 1], channels[i + 1], kernel_size=3, padding=1
                    ),
                    nn.InstanceNorm3d(channels[i + 1]),
                    nn.LeakyReLU(0.01, inplace=True),
                )
            )

        print(f"✓ 3D CNN Encoder: {channels}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns multi-scale features matching DINOv2 skip connection levels.
        """
        features = {}

        x = self.stem(x)  # (B, 32, D, H, W)
        features["scale0"] = x

        for i, block in enumerate(self.blocks):
            x = block(x)  # Progressive downsampling
            features[f"scale{i + 1}"] = x

        return features


# ============================================================================
# CROSS-ATTENTION FUSION (DINOv2 ↔ CNN)
# ============================================================================


class GatedFusion(nn.Module):
    """
    Lightweight gated fusion for large-resolution feature maps.
    Uses a learned gate to blend DINOv2 and CNN features:
        fused = gate * dino + (1 - gate) * cnn
    Memory-efficient: O(N) not O(N²) like cross-attention.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1),
            nn.InstanceNorm3d(dim),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, dino_feat: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
        gate = self.gate_conv(torch.cat([dino_feat, cnn_feat], dim=1))
        fused = gate * dino_feat + (1 - gate) * cnn_feat
        return self.refine(fused)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion for small-resolution feature maps (bottleneck only).
    Sequence length must be manageable (< ~2000 tokens).
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm_dino = nn.LayerNorm(dim)
        self.norm_cnn = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, dino_feat: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = dino_feat.shape

        dino_flat = dino_feat.flatten(2).permute(0, 2, 1)  # (B, N, C)
        cnn_flat = cnn_feat.flatten(2).permute(0, 2, 1)

        q = self.norm_dino(dino_flat)
        kv = self.norm_cnn(cnn_flat)
        attended, _ = self.cross_attn(query=q, key=kv, value=kv)

        fused = dino_flat + attended
        fused = fused + self.ffn(self.norm_out(fused))

        return fused.permute(0, 2, 1).view(B, C, D, H, W)


# ============================================================================
# ENHANCED ENCODER — DINOv2 + 3D CNN + Fusion
# ============================================================================


class DualBranchEncoder(nn.Module):
    """
    Dual-branch encoder combining:
      1. DINOv2 (frozen, 2D slices → 3D via adaptor) — global semantic features
      2. 3D CNN — native volumetric spatial features

    Features are fused via cross-attention at matching spatial scales.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        feature_dim: int = 256,
        cnn_channels: List[int] = [32, 64, 128, 256],
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # ── DINOv2 Branch ──
        print(f"Loading DINOv2: {model_name}...")
        self.vit = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True, verbose=False
        )
        self.hidden_dim = 768  # ViT-B/14

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Multi-scale layer indices for skip connections
        self.skip_layer_indices = [2, 5, 8, 11]

        # Per-layer projection: ViT hidden_dim → feature_dim
        self.dino_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )

        # 3D adaptor per scale: depthwise conv + 3D conv (CDPDNet Eq. 4)
        self.dino_3d_adaptors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        feature_dim,
                        feature_dim,
                        kernel_size=3,
                        padding=1,
                        groups=feature_dim,
                    ),  # depthwise
                    nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(feature_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )

        # ── 3D CNN Branch ──
        self.cnn = CNN3DEncoder(in_channels=1, channels=cnn_channels)

        # Projection to match feature_dim at each scale
        self.cnn_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(ch, feature_dim, kernel_size=1),
                    nn.InstanceNorm3d(feature_dim),
                    nn.ReLU(inplace=True),
                )
                for ch in cnn_channels
            ]
        )

        # ── Scale-Aware Fusion ──
        # Gated fusion at large scales (0-2) to avoid O(N²) attention on huge maps
        # Cross-attention only at scale3 (bottleneck) where N is small (~784 tokens)
        self.fusion_modules = nn.ModuleList(
            [
                GatedFusion(
                    dim=feature_dim
                ),  # scale0: 32×112×112 = 401K tokens → gated
                GatedFusion(dim=feature_dim),  # scale1: 16×56×56  = 50K tokens  → gated
                GatedFusion(dim=feature_dim),  # scale2: 8×28×28   = 6K tokens   → gated
                CrossAttentionFusion(
                    dim=feature_dim
                ),  # scale3: 4×14×14   = 784 tokens  → cross-attn ✓
            ]
        )

        # ── Bottleneck aggregator ──
        self.bottleneck = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_dim),
            nn.ReLU(inplace=True),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"✓ Dual-branch encoder: DINOv2 + 3D CNN, trainable params: {trainable:,}"
        )

    def _extract_dino_multiscale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from DINOv2 (2D slice-by-slice, chunked)."""
        B, C, D, H, W = x.shape

        # Pad H, W to nearest multiple of 14 (ViT-B/14 patch size requirement)
        patch_sz = 14
        pad_h = (patch_sz - H % patch_sz) % patch_sz
        pad_w = (patch_sz - W % patch_sz) % patch_sz
        H_pad, W_pad = H + pad_h, W + pad_w
        h, w = H_pad // 14, W_pad // 14

        # Process slices in chunks to avoid OOM (ViT is heavy per-slice)
        CHUNK = (
            8   # 8 slices per forward pass — halved from 16 to reduce peak VRAM
            # V5's dual-branch architecture doubles activation memory vs V4
        )
        num_layers = len(self.skip_layer_indices)
        all_feats = [[] for _ in range(num_layers)]

        grad_enabled = not next(self.vit.parameters()).requires_grad
        for d_start in range(0, B * D, CHUNK):
            d_end = min(d_start + CHUNK, B * D)

            # Get chunk of slices → (chunk, 1, H, W)
            x_flat = x.squeeze(1).reshape(B * D, 1, H, W)
            chunk_2d = x_flat[d_start:d_end].repeat(1, 3, 1, 1)
            if pad_h > 0 or pad_w > 0:
                chunk_2d = F.pad(
                    chunk_2d, (0, pad_w, 0, pad_h), mode="constant", value=0
                )

            with torch.set_grad_enabled(not grad_enabled):
                chunk_outputs = self.vit.get_intermediate_layers(
                    chunk_2d, n=self.skip_layer_indices
                )

            for i, feat in enumerate(chunk_outputs):
                all_feats[i].append(feat)

            del chunk_2d, chunk_outputs

        # Concatenate chunks back
        features_3d = []
        for i in range(num_layers):
            feat = torch.cat(all_feats[i], dim=0)  # (B*D, num_patches, hidden_dim)
            num_patches = feat.shape[1]
            patch_side = int(num_patches**0.5)

            feat_2d = feat.permute(0, 2, 1).reshape(
                feat.shape[0], self.hidden_dim, patch_side, patch_side
            )
            feat_2d = F.interpolate(
                feat_2d, size=(h, w), mode="bilinear", align_corners=False
            )

            feat_proj = self.dino_projections[i](feat_2d.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )

            feat_3d = rearrange(feat_proj, "(b d) c h w -> b c d h w", b=B, d=D)
            feat_3d = self.dino_3d_adaptors[i](feat_3d)
            features_3d.append(feat_3d)

        return features_3d

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, D, H, W) — input CT volume patch
        Returns:
            dict with 'bottleneck', 'skip0', 'skip1', 'skip2', 'skip3'
        """
        # ── DINOv2 multi-scale features ──
        dino_features = self._extract_dino_multiscale(x)  # 4 tensors

        # ── 3D CNN multi-scale features ──
        cnn_features = self.cnn(x)  # dict: scale0..scale3

        # ── Cross-attention fusion at each scale ──
        fused_features = {}
        for i in range(4):
            dino_f = dino_features[i]
            cnn_f = self.cnn_projections[i](cnn_features[f"scale{i}"])

            # IMPORTANT: Upsample DINOv2 features to match CNN's multi-scale
            # spatial hierarchy — NOT the other way around! CNN provides the
            # proper multi-resolution pyramid (112→56→28→14), DINOv2 gives
            # the same resolution at all layers (8×8 tokens).
            if dino_f.shape[2:] != cnn_f.shape[2:]:
                dino_f = F.interpolate(
                    dino_f, size=cnn_f.shape[2:], mode="trilinear", align_corners=False
                )

            fused = self.fusion_modules[i](dino_f, cnn_f)
            fused_features[f"skip{i}"] = fused

        # Bottleneck = deepest fused features
        fused_features["bottleneck"] = self.bottleneck(fused_features["skip3"])

        return fused_features

    def unfreeze_encoder(self, num_layers: int = 4):
        """Unfreeze last N layers of DINOv2 for fine-tuning (Stage 2)."""
        params = list(self.vit.parameters())
        for param in params[-num_layers:]:
            param.requires_grad = True
        print(f"  Unfroze last {num_layers} layers of DINOv2")


# ============================================================================
# TEXT-GUIDED DECODER
# ============================================================================


class TextGuidedUpBlock(nn.Module):
    """
    Decoder upsampling block with:
      1. Skip connection from encoder
      2. Text-visual alignment ψ
      3. Prompt attention module (optional)
      4. Spatial attention
    """

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

        # Transposed conv for upsampling
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        # Skip connection adapter
        self.skip_conv = nn.Conv3d(skip_channels, out_channels, 1)

        # Refinement conv (after concat)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Text-visual alignment ψ
        self.text_align = TextVisualAlignment(
            visual_dim=out_channels, text_dim=text_dim
        )

        # Prompt attention module (novel contribution)
        self.use_prompt_attention = use_prompt_attention
        if use_prompt_attention:
            self.prompt_attn = PromptAttentionModule(
                visual_dim=out_channels,
                text_dim=text_dim,
                num_classes=num_classes,
                num_levels=num_levels,
            )

        # Spatial attention
        self.spatial_attn = SpatialAttention3D()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, C_in, D, H, W) — from previous decoder level
            skip:             (B, C_skip, D', H', W') — encoder skip connection
            prompt_embeddings: (num_classes, num_levels, text_dim) — from MGPL
        """
        x = self.upsample(x)
        skip = self.skip_conv(skip)

        # Match sizes
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(
                skip, size=x.shape[2:], mode="trilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        # Text-visual alignment: average over classes and levels for ψ
        avg_text = prompt_embeddings.mean(dim=(0, 1))  # (text_dim,)
        x = self.text_align(x, avg_text.unsqueeze(0))

        # Prompt attention (if enabled)
        if self.use_prompt_attention and prompt_embeddings is not None:
            x = self.prompt_attn(x, prompt_embeddings)

        # Spatial attention
        x = self.spatial_attn(x)

        return x


class TextGuidedDecoder(nn.Module):
    """
    U-Net style decoder with text-guided skip connections
    and multi-granularity prompt attention at each scale.
    """

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

        # 4 upsampling blocks matching 4 encoder skip connections
        channels = [in_channels] + decoder_channels
        self.up_blocks = nn.ModuleList()
        for i in range(4):
            self.up_blocks.append(
                TextGuidedUpBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    skip_channels=in_channels,  # all encoder skips are feature_dim
                    text_dim=text_dim,
                    num_classes=num_classes,
                    num_levels=num_levels,
                    use_prompt_attention=use_prompt_attention,
                )
            )

        # Final segmentation head
        self.seg_head = nn.Conv3d(decoder_channels[-1], num_classes, kernel_size=1)

        print(
            f"✓ Text-guided decoder: {decoder_channels}, "
            f"prompt_attention={'ON' if use_prompt_attention else 'OFF'}"
        )

    def forward(
        self,
        encoder_outputs: Dict[str, torch.Tensor],
        prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_outputs: dict with 'bottleneck', 'skip0'..'skip3'
            prompt_embeddings: (num_classes, num_levels, text_dim)
        Returns:
            logits: (B, num_classes, D, H, W)
        """
        x = encoder_outputs["bottleneck"]

        # Decode with skip connections (deepest first)
        skip_keys = ["skip3", "skip2", "skip1", "skip0"]
        for i, block in enumerate(self.up_blocks):
            skip = encoder_outputs[skip_keys[i]]
            x = block(x, skip, prompt_embeddings)

        logits = self.seg_head(x)
        return logits


# ============================================================================
# SPATIAL ATTENTION MODULE (reused from V4)
# ============================================================================


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


# ============================================================================
# COMPLETE MODEL — V5
# ============================================================================


class MedDINOVISTA3D_V5(nn.Module):
    """
    MedDINO-VISTA3D V5 with:
      - Dual-branch encoder (DINOv2 + 3D CNN + cross-attention fusion)
      - Multi-Granularity Prompt Learning (MGPL)
      - Text-guided decoder with prompt attention at each scale
    """

    def __init__(
        self,
        num_classes: int = 4,
        feature_dim: int = 256,
        cnn_channels: List[int] = [32, 64, 128, 256],
        dinov2_backbone: str = "dinov2_vitb14",
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        freeze_encoder: bool = True,
        use_mgpl: bool = True,
    ):
        super().__init__()
        self.use_mgpl = use_mgpl

        # Encoder
        self.encoder = DualBranchEncoder(
            model_name=dinov2_backbone,
            feature_dim=feature_dim,
            cnn_channels=cnn_channels,
            freeze_backbone=freeze_encoder,
        )

        # MGPL (frozen CLIP)
        if use_mgpl:
            self.prompt_encoder = MultiGranularityPromptEncoder(
                clip_model_name=clip_model, pretrained=clip_pretrained
            )
            text_dim = self.prompt_encoder.embed_dim
        else:
            self.prompt_encoder = None
            text_dim = 512  # dummy

        # Decoder
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
        print(f"\n{'='*50}")
        print(f"MedDINO-VISTA3D V5 | Total: {total:,} | Trainable: {trainable:,}")
        print(f"MGPL: {'ON' if use_mgpl else 'OFF'}")
        print(f"{'='*50}\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use gradient checkpointing on encoder to save memory (~30-40% reduction)
        # Trade compute for memory: recompute activations during backward
        encoder_out = torch.utils.checkpoint.checkpoint(
            self.encoder,
            x,
            use_reentrant=False,  # More memory-efficient on modern PyTorch
        )

        if self.use_mgpl and self.prompt_encoder is not None:
            prompt_embs = self.prompt_encoder()  # (4, 5, 512)
        else:
            prompt_embs = torch.zeros(4, 5, 512, device=x.device)

        logits = self.decoder(encoder_out, prompt_embs)

        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        return logits

    def unfreeze_encoder(self, num_layers: int = 4):
        self.encoder.unfreeze_encoder(num_layers)


# ============================================================================
# VLM-AS-CRITIC MODULE (Novel Contribution)
# ============================================================================


class VLMCritic:
    """
    VLM-in-the-Loop Training (VITL) Critic.

    Every K epochs, this module:
    1. Identifies the worst-performing validation cases
    2. Renders 2D slices with prediction overlays
    3. Feeds them to a local VLM (Ollama / Transformers / OpenAI)
    4. Parses structured JSON feedback
    5. Returns adaptive training signals:
       - Per-class loss weight adjustments
       - Per-case difficulty scores (for curriculum learning)
       - Sampling bias updates (oversample error-prone regions)
       - Spatial attention hints
    """

    CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]
    CLASS_COLORS = {
        0: (0, 0, 0),  # background: black
        1: (0, 255, 0),  # kidney: green
        2: (255, 0, 0),  # tumor: red
        3: (0, 0, 255),  # cyst: blue
    }

    def __init__(self, config: dict):
        self.backend = config["vlm_backend"]
        self.model_name = config["vlm_model"]
        self.api_url = config.get("vlm_api_url", "http://localhost:11434")
        self.top_n = config["critic_top_n_worst"]
        self.num_slices = config["critic_num_slices"]
        self.output_dir = Path(config["output_dir"]) / "vlm_critic"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Current adaptive weights — match AdaptiveHybridLoss defaults
        self.adaptive_class_weights = torch.tensor([0.1, 1.0, 12.0, 4.0])
        self.case_difficulty_scores = {}  # case_id → float
        # Tumor-focused sampling: 65% patches centred on tumor
        self.sampling_bias = {
            "tumor": 0.65,
            "cyst": 0.10,
            "kidney": 0.15,
            "random": 0.10,
        }
        self.error_history = []

    def analyze_epoch(
        self,
        epoch: int,
        val_cases: List[dict],
        predictions: Dict[str, torch.Tensor],
        ground_truths: Dict[str, torch.Tensor],
        images: Dict[str, torch.Tensor],
        agg_metrics: Optional[dict] = None,
    ) -> dict:
        """
        Run VLM critic analysis on the worst validation cases.

        Args:
            epoch: Current epoch number
            val_cases: List of case dicts with 'case_id', 'dice_per_class', etc.
            predictions: {case_id: pred_tensor}
            ground_truths: {case_id: gt_tensor}
            images: {case_id: image_tensor}
            agg_metrics: MetricTracker.compute() result — authoritative dice/prec/recall

        Returns:
            feedback: dict with adaptive training signals
        """
        # Sort by foreground mean dice (ascending = worst first)
        sorted_cases = sorted(val_cases, key=lambda c: c["mean_dice_fg"])
        worst_cases = sorted_cases[: self.top_n]

        print(
            f"\n  [VLM CRITIC] Analyzing {len(worst_cases)} worst cases at epoch {epoch}..."
        )

        all_analyses = []
        for case_info in worst_cases:
            case_id = case_info["case_id"]

            # Only attempt VLM image analysis when full-volume tensors are available
            if case_id in predictions and case_id in ground_truths:
                pred = predictions[case_id]
                gt = ground_truths[case_id]
                img = images[case_id] if case_id in images else None

                # Render slices and get VLM analysis
                analysis = self._analyze_single_case(
                    case_id, img, pred, gt, case_info, epoch
                )
                if analysis is not None:
                    all_analyses.append(analysis)

        # Aggregate feedback — pass val_cases + aggregated metrics for rules
        feedback = self._aggregate_feedback(all_analyses, epoch, val_cases, agg_metrics)

        # Save feedback
        self._save_feedback(feedback, epoch)

        return feedback

    def _analyze_single_case(
        self,
        case_id: str,
        image: Optional[torch.Tensor],
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        case_info: dict,
        epoch: int,
    ) -> Optional[dict]:
        """Analyze a single case using the VLM."""
        try:
            # Render comparison images
            rendered_images = self._render_case_slices(
                case_id, image, prediction, ground_truth
            )

            # Build the prompt for the VLM
            prompt = self._build_vlm_prompt(case_id, case_info)

            # Call VLM
            response = self._call_vlm(prompt, rendered_images)

            # Parse structured response
            analysis = self._parse_vlm_response(response, case_id)

            return analysis

        except Exception as e:
            print(f"    [VLM CRITIC] Error analyzing {case_id}: {e}")
            return None

    def _render_case_slices(
        self,
        case_id: str,
        image: Optional[torch.Tensor],
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> List[str]:
        """
        Render comparison slices as base64 PNG images.
        Shows: CT | Ground Truth overlay | Prediction overlay | Error map
        """
        from PIL import Image
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # prediction and ground_truth: (D, H, W) integer labels
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().numpy()
        if image is not None and isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        D = prediction.shape[0]
        # Pick slices with most pathology
        tumor_per_slice = [
            (prediction[z] == 2).sum() + (ground_truth[z] == 2).sum() for z in range(D)
        ]
        # Pick top slices by tumor content, spread out
        top_slices = sorted(range(D), key=lambda z: tumor_per_slice[z], reverse=True)
        # Remove duplicates close together
        selected = []
        for z in top_slices:
            if all(abs(z - s) > D // 10 for s in selected):
                selected.append(z)
            if len(selected) >= self.num_slices:
                break
        if not selected:
            selected = [D // 4, D // 2, 3 * D // 4]

        rendered = []
        for z in selected[: self.num_slices]:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # CT slice
            if image is not None:
                ct_slice = image[z] if image.ndim == 3 else image[0, z]
                ct_norm = (ct_slice - ct_slice.min()) / (
                    ct_slice.max() - ct_slice.min() + 1e-8
                )
            else:
                ct_norm = np.zeros_like(prediction[z], dtype=float)

            axes[0].imshow(ct_norm, cmap="gray")
            axes[0].set_title(f"CT (z={z})")
            axes[0].axis("off")

            # Ground truth overlay
            axes[1].imshow(ct_norm, cmap="gray")
            gt_overlay = self._create_overlay(ground_truth[z])
            axes[1].imshow(gt_overlay, alpha=0.5)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            # Prediction overlay
            axes[2].imshow(ct_norm, cmap="gray")
            pred_overlay = self._create_overlay(prediction[z])
            axes[2].imshow(pred_overlay, alpha=0.5)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            # Error map: FP (red), FN (blue)
            axes[3].imshow(ct_norm, cmap="gray")
            error_map = np.zeros((*prediction[z].shape, 3), dtype=np.uint8)
            fp = (prediction[z] > 0) & (ground_truth[z] == 0)  # false positive
            fn = (prediction[z] == 0) & (ground_truth[z] > 0)  # false negative
            error_map[fp] = [255, 0, 0]  # red
            error_map[fn] = [0, 0, 255]  # blue
            axes[3].imshow(error_map, alpha=0.6)
            axes[3].set_title("Errors (R=FP, B=FN)")
            axes[3].axis("off")

            plt.suptitle(f"{case_id} — Slice {z}", fontsize=12)
            plt.tight_layout()

            # Save to buffer → base64
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            rendered.append(b64)

            # Also save to disk for logging
            save_path = self.output_dir / f"epoch{epoch}_{case_id}_z{z}.png"
            buf.seek(0)
            with open(save_path, "wb") as f:
                f.write(buf.read())

        return rendered

    def _create_overlay(self, label_slice: np.ndarray) -> np.ndarray:
        """Create RGBA overlay from label map."""
        h, w = label_slice.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        for cls_id, color in self.CLASS_COLORS.items():
            if cls_id == 0:
                continue
            mask = label_slice == cls_id
            overlay[mask, :3] = color
            overlay[mask, 3] = 180
        return overlay

    def _build_vlm_prompt(self, case_id: str, case_info: dict) -> str:
        """Build the structured analysis prompt for the VLM."""
        dice_str = ", ".join(
            f"{cls}: {d:.4f}"
            for cls, d in zip(
                self.CLASS_NAMES, case_info.get("dice_per_class", [0, 0, 0, 0])
            )
        )
        return f"""You are an expert medical imaging analyst reviewing a kidney CT segmentation result.

Case: {case_id}
Current Dice scores: {dice_str}
Mean foreground Dice: {case_info.get('mean_dice_fg', 0):.4f}

The images show:
- Panel 1: Raw CT slice
- Panel 2: Ground truth segmentation (Green=Kidney, Red=Tumor, Blue=Cyst)
- Panel 3: Model prediction overlay
- Panel 4: Error map (Red=False Positive, Blue=False Negative)

Analyze the segmentation errors and respond ONLY with valid JSON in this exact format:
{{
  "error_types": {{
    "kidney": {{"type": "none|boundary|missed|over_segmented|class_confusion", "severity": "none|mild|moderate|severe"}},
    "tumor": {{"type": "none|boundary|missed|over_segmented|class_confusion", "severity": "none|mild|moderate|severe"}},
    "cyst": {{"type": "none|boundary|missed|over_segmented|class_confusion", "severity": "none|mild|moderate|severe"}}
  }},
  "dominant_failure": "missed_small_tumor|over_segmentation|boundary_error|class_confusion|no_pathology_detected",
  "suggested_focus": "increase_recall|increase_precision|improve_boundary|detect_small_objects",
  "difficulty": "easy|medium|hard|very_hard",
  "spatial_region": "upper_pole|lower_pole|hilum|cortex|medulla|whole_kidney"
}}"""

    def _call_vlm(self, prompt: str, images_b64: List[str]) -> str:
        """Call the VLM backend."""
        if self.backend == "ollama":
            return self._call_ollama(prompt, images_b64)
        elif self.backend == "transformers":
            return self._call_transformers(prompt, images_b64)
        elif self.backend == "openai":
            return self._call_openai(prompt, images_b64)
        else:
            raise ValueError(f"Unknown VLM backend: {self.backend}")

    def _call_ollama(self, prompt: str, images_b64: List[str]) -> str:
        """Call Ollama API (local VLM server)."""
        import requests

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
            },
        }

        try:
            resp = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            print(f"    [VLM] Ollama call failed: {e}")
            return "{}"

    def _call_transformers(self, prompt: str, images_b64: List[str]) -> str:
        """Call a local HuggingFace transformers VLM."""
        # This is a placeholder — implementation depends on the specific model.
        # For Qwen2-VL or LLaVA, you'd load the model and generate here.
        # This keeps the code modular so you can swap backends.
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from PIL import Image

            # Decode first image
            img_bytes = base64.b64decode(images_b64[0])
            img = Image.open(io.BytesIO(img_bytes))

            # This is model-specific; adapt for your chosen VLM
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
                outputs = model.generate(**inputs, max_new_tokens=1024)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            print(f"    [VLM] Transformers call failed: {e}")
            return "{}"

    def _call_openai(self, prompt: str, images_b64: List[str]) -> str:
        """Call OpenAI-compatible API (GPT-4o, etc.)."""
        import requests

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img in images_b64:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                }
            )

        try:
            resp = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.1,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    [VLM] OpenAI call failed: {e}")
            return "{}"

    def _parse_vlm_response(self, response: str, case_id: str) -> Optional[dict]:
        """Parse the VLM's JSON response into structured feedback."""
        try:
            # Extract JSON from response (VLMs sometimes add text around it)
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                print(f"    [VLM] No JSON found in response for {case_id}")
                return None

            data = json.loads(json_match.group())
            data["case_id"] = case_id
            return data

        except json.JSONDecodeError as e:
            print(f"    [VLM] JSON parse error for {case_id}: {e}")
            return None

    def _aggregate_feedback(
        self,
        analyses: List[dict],
        epoch: int,
        val_cases: Optional[List[dict]] = None,
        agg_metrics: Optional[dict] = None,
    ) -> dict:
        """
        Aggregate individual case analyses into training-level feedback.
        Applies both metric-driven rules (always) and VLM-driven rules (when available).
        """
        if not analyses and not val_cases:
            return self._default_feedback()

        # Count error types per class
        error_counts = {
            cls: {
                "missed": 0,
                "over_segmented": 0,
                "boundary": 0,
                "class_confusion": 0,
                "none": 0,
            }
            for cls in ["kidney", "tumor", "cyst"]
        }

        difficulty_scores = {}
        dominant_failures = []
        suggested_focuses = []

        for a in analyses:
            case_id = a.get("case_id", "unknown")

            # Parse error types
            for cls in ["kidney", "tumor", "cyst"]:
                err = a.get("error_types", {}).get(cls, {})
                if isinstance(err, str):
                    try:
                        err = json.loads(err)
                    except (json.JSONDecodeError, TypeError):
                        err = {"type": "none", "severity": "none"}
                err_type = err.get("type", "none")
                if err_type in error_counts[cls]:
                    error_counts[cls][err_type] += 1

            # Difficulty
            diff_map = {"easy": 0.25, "medium": 0.5, "hard": 0.75, "very_hard": 1.0}
            difficulty_scores[case_id] = diff_map.get(
                a.get("difficulty", "medium"), 0.5
            )

            dominant_failures.append(a.get("dominant_failure", ""))
            suggested_focuses.append(a.get("suggested_focus", ""))

        # ── Compute adaptive loss weights ──
        new_weights = self.adaptive_class_weights.clone()

        # ── Rule-based metric-driven adjustments (always applied, no VLM needed) ──
        # Use aggregated MetricTracker results (authoritative) when available;
        # fall back to per-patch val_cases average only if agg_metrics is absent.
        if agg_metrics is not None:
            agg_dice = agg_metrics["dice"]  # shape (4,)
            agg_prec = agg_metrics["precision"]  # shape (4,)
            agg_recall = agg_metrics["recall"]  # shape (4,)
        elif val_cases:
            # per-patch fallback — pessimistic because center-crop patches often lack tumor
            arr = np.zeros(4)
            for vc in val_cases:
                arr += np.array(vc.get("dice_per_class", [0.0, 0.0, 0.0, 0.0]))
            arr /= max(len(val_cases), 1)
            agg_dice = agg_prec = agg_recall = arr
        else:
            agg_dice = agg_prec = agg_recall = np.zeros(4)

        # Cyst — escalate only when genuinely undetected
        if agg_dice[3] < 0.05:
            old_w = new_weights[3].item()
            new_weights[3] = min(new_weights[3] * 1.2, 20.0)
            print(
                f"    [METRIC RULE] Cyst dice={agg_dice[3]:.4f} → weight {old_w:.1f}→{new_weights[3]:.1f}"
            )

        # Tumor — tiered escalation, but ONLY when the issue is under-detection.
        # If recall is already high (>0.40) but precision is low (<0.08), it's
        # over-segmentation — escalating further makes it worse, so skip/reduce.
        tumor_dice = float(agg_dice[2])
        tumor_prec = float(agg_prec[2])
        tumor_recall = float(agg_recall[2])
        old_w = new_weights[2].item()

        if tumor_recall > 0.40 and tumor_prec < 0.08:
            # Over-segmentation: model already finds tumors but floods FPs
            new_weights[2] = max(new_weights[2] * 0.85, 8.0)
            print(
                f"    [METRIC RULE] Tumor over-seg (rec={tumor_recall:.2f} prec={tumor_prec:.3f}) → weight {old_w:.1f}→{new_weights[2]:.1f}"
            )
        elif tumor_recall > 0.25 and tumor_dice < 0.20:
            # Recall is present but precision is the bottleneck — escalating CE weight
            # won't help and causes instability; hold weight steady
            print(
                f"    [METRIC RULE] Tumor precision-limited (rec={tumor_recall:.2f} dice={tumor_dice:.4f}) → weight held at {new_weights[2]:.1f}"
            )
        elif tumor_dice < 0.15:
            new_weights[2] = min(new_weights[2] * 1.25, 20.0)
            print(
                f"    [METRIC RULE] Tumor dice={tumor_dice:.4f} (critical) → weight {old_w:.1f}→{new_weights[2]:.1f}"
            )
        elif tumor_dice < 0.35:
            new_weights[2] = min(new_weights[2] * 1.15, 20.0)
            print(
                f"    [METRIC RULE] Tumor dice={tumor_dice:.4f} (low) → weight {old_w:.1f}→{new_weights[2]:.1f}"
            )
        elif tumor_dice < 0.55:
            new_weights[2] = min(new_weights[2] * 1.08, 20.0)
            print(
                f"    [METRIC RULE] Tumor dice={tumor_dice:.4f} (improving) → weight {old_w:.1f}→{new_weights[2]:.1f}"
            )

        # ── VLM-driven weight adjustments (when VLM image analyses available) ──
        tumor_misses = error_counts["tumor"]["missed"]
        tumor_overseg = error_counts["tumor"]["over_segmented"]
        total_analyzed = len(analyses)

        if total_analyzed > 0:
            if tumor_misses > total_analyzed * 0.1:
                # Many missed tumors: increase tumor weight
                new_weights[2] = min(new_weights[2] * 1.3, 50.0)
                print(
                    f"    [VLM CRITIC] Tumor frequently missed → weight {new_weights[2]:.1f}"
                )
            elif tumor_overseg > total_analyzed * 0.1:
                # Over-segmented but only reduce if tumor dice is already healthy
                if (
                    analyses
                    and np.mean(
                        [
                            a.get("dice_per_class", [0, 0, 0, 0])[2]
                            for a in analyses
                            if isinstance(a.get("dice_per_class"), list)
                        ]
                    )
                    > 0.50
                ):
                    new_weights[2] = max(new_weights[2] * 0.9, 10.0)
                    print(
                        f"    [VLM CRITIC] Tumor over-segmented (dice healthy) → weight {new_weights[2]:.1f}"
                    )

        # Same for cyst (VLM-driven)
        cyst_misses = error_counts["cyst"]["missed"]
        if total_analyzed > 0 and cyst_misses > total_analyzed * 0.1:
            new_weights[3] = min(new_weights[3] * 1.3, 60.0)
            print(
                f"    [VLM CRITIC] Cyst frequently missed → weight {new_weights[3]:.1f}"
            )

        # ── Compute sampling bias ──
        new_sampling = dict(self.sampling_bias)
        if (
            "missed_small_tumor" in dominant_failures
            or "detect_small_objects" in suggested_focuses
        ):
            # Shift sampling even more toward tumor
            new_sampling["tumor"] = min(new_sampling["tumor"] + 0.05, 0.80)
            new_sampling["random"] = max(new_sampling["random"] - 0.05, 0.05)
            print(
                f"    [VLM CRITIC] Small objects missed → tumor sampling {new_sampling['tumor']:.0%}"
            )

        feedback = {
            "epoch": epoch,
            "adaptive_class_weights": new_weights.tolist(),
            "case_difficulty_scores": difficulty_scores,
            "sampling_bias": new_sampling,
            "error_counts": {k: dict(v) for k, v in error_counts.items()},
            "dominant_failures": dominant_failures,
            "num_analyzed": total_analyzed,
        }

        # Update internal state
        self.adaptive_class_weights = new_weights
        self.case_difficulty_scores.update(difficulty_scores)
        self.sampling_bias = new_sampling
        self.error_history.append(feedback)

        return feedback

    def _default_feedback(self) -> dict:
        """Default feedback when VLM analysis fails."""
        return {
            "adaptive_class_weights": self.adaptive_class_weights.tolist(),
            "case_difficulty_scores": {},
            "sampling_bias": self.sampling_bias,
            "error_counts": {},
            "dominant_failures": [],
            "num_analyzed": 0,
        }

    def _save_feedback(self, feedback: dict, epoch: int):
        """Save critic feedback to disk."""
        path = self.output_dir / f"feedback_epoch{epoch}.json"
        with open(path, "w") as f:
            json.dump(feedback, f, indent=2)
        print(f"    [VLM CRITIC] Feedback saved to {path}")

    def get_current_weights(self) -> torch.Tensor:
        """Get current adaptive class weights."""
        return self.adaptive_class_weights

    def get_case_sampling_weight(self, case_id: str) -> float:
        """Get curriculum sampling weight for a case."""
        return self.case_difficulty_scores.get(case_id, 0.5)


# ============================================================================
# LOSS FUNCTION — ADAPTIVE (VLM-Critic aware)
# ============================================================================


class AdaptiveHybridLoss(nn.Module):
    """
    Adaptive Hybrid Loss that can be updated by VLM critic feedback.
    Combines Tversky + Focal + Dice with dynamically adjustable class weights.
    """

    def __init__(self, initial_weights: Optional[torch.Tensor] = None):
        super().__init__()
        if initial_weights is None:
            # Tumor-primary: tumor >> cyst > kidney >> background
            # Only affects focal/CE — Tversky stays unweighted to prevent collapse
            initial_weights = torch.tensor([0.1, 1.0, 12.0, 4.0])
        self.register_buffer("class_weights", initial_weights.clone())

        # Shared Tversky params (all classes, background-safe)
        self.tversky_alpha = 0.15  # FP penalty
        self.tversky_beta = 0.85  # FN penalty

        # Dedicated tumor Tversky: maximize recall — penalise FN heavily
        self.tumor_alpha = (
            0.10  # mild FP penalty — recall-biased but avoids full precision collapse
        )
        self.tumor_beta = (
            0.90  # heavily penalise FN (missing tumor is the worst failure)
        )

    def update_weights(self, new_weights: torch.Tensor):
        """Update class weights from VLM critic feedback."""
        self.class_weights.copy_(new_weights.to(self.class_weights.device))

    def update_tversky_params(self, alpha: float, beta: float):
        """Shift Tversky FP/FN balance based on critic feedback."""
        self.tversky_alpha = alpha
        self.tversky_beta = beta

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(outputs, dim=1)
        target_oh = F.one_hot(target.squeeze(1).long(), num_classes=outputs.shape[1])
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()

        TP = (pred * target_oh).sum(dim=(2, 3, 4))
        FP = ((1 - target_oh) * pred).sum(dim=(2, 3, 4))
        FN = (target_oh * (1 - pred)).sum(dim=(2, 3, 4))

        # ── Shared Tversky (unweighted) — keeps background stable ──
        tversky = (TP + 1e-6) / (
            TP + self.tversky_alpha * FP + self.tversky_beta * FN + 1e-6
        )
        tversky_loss = (1 - tversky).mean()

        # ── Dedicated tumor Tversky — high-recall, penalise FN heavily ──
        # Only computed over the tumor class (index 2)
        t_tp = TP[:, 2]  # (B,)
        t_fp = FP[:, 2]
        t_fn = FN[:, 2]
        tumor_tversky = (t_tp + 1e-6) / (
            t_tp + self.tumor_alpha * t_fp + self.tumor_beta * t_fn + 1e-6
        )
        tumor_tversky_loss = (1 - tumor_tversky).mean()

        # ── Focal CE — weighted toward tumor/cyst ──
        ce = F.cross_entropy(
            outputs,
            target.squeeze(1).long(),
            weight=self.class_weights,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** 2.5 * ce).mean()

        # ── Dice (unweighted mean) — general overlap signal ──
        dice_per_class = (
            2 * TP / ((pred.sum(dim=(2, 3, 4)) + target_oh.sum(dim=(2, 3, 4))) + 1e-6)
        )
        dice_loss = 1 - dice_per_class.mean()

        # 0.30 shared Tversky + 0.30 tumor-recall Tversky + 0.25 focal + 0.15 dice
        return (
            0.30 * tversky_loss
            + 0.30 * tumor_tversky_loss
            + 0.25 * focal_loss
            + 0.15 * dice_loss
        )


# ============================================================================
# DATASET — WITH ADAPTIVE SAMPLING
# ============================================================================


class AdaptiveKiTS23Dataset(Dataset):
    """
    KiTS23 dataset with VLM-critic adaptive sampling:
      - Adjustable class sampling probabilities
      - Per-case difficulty-based oversampling
      - CT HU windowing + proper normalization
    """

    def __init__(
        self,
        data_dicts: List[dict],
        patch_size: Tuple[int, ...],
        num_samples: int = 2,
        is_train: bool = True,
        sampling_bias: Optional[dict] = None,
        case_weights: Optional[Dict[str, float]] = None,
    ):
        self.data_dicts = data_dicts
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.is_train = is_train

        # Adaptive sampling from VLM critic
        self.sampling_bias = sampling_bias or {
            "tumor": 0.65,
            "cyst": 0.10,
            "kidney": 0.15,
            "random": 0.10,
        }
        self.case_weights = case_weights or {}

    def update_sampling(self, new_bias: dict, new_case_weights: dict):
        """Update sampling strategy from VLM critic feedback."""
        self.sampling_bias = new_bias
        self.case_weights.update(new_case_weights)

    def __len__(self):
        return len(self.data_dicts) * self.num_samples

    def __getitem__(self, idx):
        vol_idx = idx // self.num_samples
        data = self.data_dicts[vol_idx]

        img = nib.load(data["image"]).get_fdata().astype(np.float32)
        lbl = nib.load(data["label"]).get_fdata().astype(np.int64)

        # CT HU windowing + normalization (fixed from V4 normalization bug)
        img = np.clip(img, -175, 250)
        img = (img - (-175)) / (250 - (-175))  # Normalize to [0, 1]

        p_img, p_lbl = self._sample_patch(img, lbl)
        return {
            "image": torch.from_numpy(p_img).float().unsqueeze(0),
            "label": torch.from_numpy(p_lbl).long().unsqueeze(0),
            "case_id": data.get("case_id", f"case_{vol_idx:05d}"),
        }

    def _sample_patch(self, img: np.ndarray, lbl: np.ndarray):
        d, h, w = img.shape
        pd, ph, pw = self.patch_size

        # Pad if needed
        if d < pd or h < ph or w < pw:
            pad = ((0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w)))
            img = np.pad(img, pad, mode="constant")
            lbl = np.pad(lbl, pad, mode="constant")
            d, h, w = img.shape

        if self.is_train:
            # Adaptive sampling using VLM critic probabilities
            rand_val = np.random.random()
            tumor_th = self.sampling_bias.get("tumor", 0.50)
            cyst_th = tumor_th + self.sampling_bias.get("cyst", 0.25)
            kidney_th = cyst_th + self.sampling_bias.get("kidney", 0.15)

            target_class = None
            if rand_val < tumor_th:
                target_class = 2
            elif rand_val < cyst_th:
                target_class = 3
            elif rand_val < kidney_th:
                target_class = 1

            if target_class is not None:
                indices = np.argwhere(lbl == target_class)
                if len(indices) > 0:
                    c = indices[np.random.randint(len(indices))]
                    ds = np.clip(c[0] - pd // 2, 0, d - pd)
                    hs = np.clip(c[1] - ph // 2, 0, h - ph)
                    ws = np.clip(c[2] - pw // 2, 0, w - pw)
                else:
                    # Fallback: try any foreground
                    fg = np.argwhere(lbl > 0)
                    if len(fg) > 0:
                        c = fg[np.random.randint(len(fg))]
                        ds = np.clip(c[0] - pd // 2, 0, d - pd)
                        hs = np.clip(c[1] - ph // 2, 0, h - ph)
                        ws = np.clip(c[2] - pw // 2, 0, w - pw)
                    else:
                        ds = np.random.randint(0, max(1, d - pd + 1))
                        hs = np.random.randint(0, max(1, h - ph + 1))
                        ws = np.random.randint(0, max(1, w - pw + 1))
            else:
                # Random
                ds = np.random.randint(0, max(1, d - pd + 1))
                hs = np.random.randint(0, max(1, h - ph + 1))
                ws = np.random.randint(0, max(1, w - pw + 1))

            # Random augmentations
            patch_img = img[ds : ds + pd, hs : hs + ph, ws : ws + pw].copy()
            patch_lbl = lbl[ds : ds + pd, hs : hs + ph, ws : ws + pw].copy()

            # Random 90-degree rotations (2 of 3 axes)
            if np.random.random() < 0.3:
                k = np.random.choice([1, 2, 3])
                axes = (1, 2)  # H, W axes
                patch_img = np.rot90(patch_img, k=k, axes=axes).copy()
                patch_lbl = np.rot90(patch_lbl, k=k, axes=axes).copy()

            # Random flip
            if np.random.random() < 0.5:
                axis = np.random.choice([0, 1, 2])
                patch_img = np.flip(patch_img, axis=axis).copy()
                patch_lbl = np.flip(patch_lbl, axis=axis).copy()

            # Random intensity shift
            if np.random.random() < 0.15:
                shift = np.random.uniform(-0.1, 0.1)
                patch_img = np.clip(patch_img + shift, 0, 1)

            return patch_img, patch_lbl
        else:
            # Foreground-aware validation crop:
            # Try center crop first; if it contains no tumor/cyst AND the full
            # volume has foreground, fall back to a foreground-centered crop so
            # validation dice reflects actual tumor/cyst detection performance.
            ds = (d - pd) // 2
            hs = (h - ph) // 2
            ws = (w - pw) // 2
            center_lbl = lbl[ds : ds + pd, hs : hs + ph, ws : ws + pw]
            has_fg_in_center = (center_lbl == 2).any() or (center_lbl == 3).any()
            has_fg_in_vol = (lbl == 2).any() or (lbl == 3).any()

            if not has_fg_in_center and has_fg_in_vol:
                # Pick a random tumor/cyst voxel and center the patch on it
                fg_indices = np.argwhere((lbl == 2) | (lbl == 3))
                c = fg_indices[np.random.randint(len(fg_indices))]
                ds = int(np.clip(c[0] - pd // 2, 0, d - pd))
                hs = int(np.clip(c[1] - ph // 2, 0, h - ph))
                ws = int(np.clip(c[2] - pw // 2, 0, w - pw))

            return (
                img[ds : ds + pd, hs : hs + ph, ws : ws + pw],
                lbl[ds : ds + pd, hs : hs + ph, ws : ws + pw],
            )


# ============================================================================
# TRAINING UTILITIES
# ============================================================================


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
            self.counter = 0
            self.val_loss_min = val_loss


class MetricTracker:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, preds, targets):
        preds, targets = preds.cpu(), targets.cpu()
        mask = (targets >= 0) & (targets < self.num_classes)
        self.confusion_matrix += torch.bincount(
            self.num_classes * targets[mask].long() + preds[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(0) - tp
        fn = self.confusion_matrix.sum(1) - tp
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return {
            "dice": dice.numpy(),
            "iou": iou.numpy(),
            "precision": precision.numpy(),
            "recall": recall.numpy(),
        }

    def format_results(self):
        m = self.compute()
        mean_dice = np.mean(m["dice"][1:])
        s = f"\n  Mean Dice (Fg): {mean_dice:.4f}\n"
        s += f"  {'Class':<10} {'Dice':<8} {'IoU':<8} {'Prec':<8} {'Recall':<8}\n"
        s += "-" * 46 + "\n"
        for i, c in enumerate(CLASS_NAMES):
            s += (
                f"  {c:<10} {m['dice'][i]:.4f}   {m['iou'][i]:.4f}   "
                f"{m['precision'][i]:.4f}   {m['recall'][i]:.4f}\n"
            )
        return s, mean_dice


def get_dataloaders(config, sampling_bias=None, case_weights=None):
    """Create train/val dataloaders with adaptive sampling."""
    data_dir = Path(config["kits23_dir"])
    cases = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")]
    )

    # ── Reserve last N cases as blind test set (never touched during training) ──
    n_test = config.get("test_cases", 40)
    test_cases = cases[-n_test:]
    cases = cases[:-n_test]  # train+val pool
    print(
        f"  Split: {len(cases)} train/val cases | {len(test_cases)} held-out test cases"
        f" ({test_cases[0].name} → {test_cases[-1].name})"
    )

    max_cases = config.get("quick_cases")
    if max_cases:
        cases = cases[:max_cases]

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

    train_ds = AdaptiveKiTS23Dataset(
        train_dicts,
        config["patch_size"],
        config["num_samples_per_volume"],
        is_train=True,
        sampling_bias=sampling_bias,
        case_weights=case_weights,
    )
    val_ds = AdaptiveKiTS23Dataset(
        val_dicts,
        config["patch_size"],
        1,
        is_train=False,
    )

    # WSL2 fix: pin_memory=False is the critical change.
    # The CUDACachingAllocator:427 crash is caused by pin_memory=True:
    # it forces workers to allocate pinned GPU-accessible memory via CUDA IPC
    # handles; when worker processes exit, those handles become dangling.
    # Workers doing CPU-only NIfTI IO + augmentation create NO CUDA contexts,
    # so parallel workers are safe as long as pin_memory stays False.
    num_workers = 8  # Static 8 workers for parallel IO
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,  # Half for validation
        pin_memory=False,
        persistent_workers=True,
    )

    return train_loader, val_loader, val_dicts


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    """Train for one epoch with AMP."""
    # Flush any allocator state left from previous val pass
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    model.train()
    total_loss = 0
    count = 0
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 10  # abort epoch if allocator is in unrecoverable state

    for batch in tqdm(loader, desc="Training", leave=False):
        try:
            img = batch["image"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = loss_fn(out, lbl)

            if not torch.isfinite(loss):
                consecutive_errors += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            count += 1
            consecutive_errors = 0  # reset on success

            del img, lbl, out, loss
        except RuntimeError as e:
            consecutive_errors += 1
            print(f"  ⚠ Training error: {e}")
            for var in ["img", "lbl", "out", "loss"]:
                if var in locals():
                    del locals()[var]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(
                    f"  ❌ {consecutive_errors} consecutive errors — aborting epoch to prevent corrupt metrics"
                )
                break
            continue

    return total_loss / max(count, 1)


def validate(model, loader, loss_fn, device):
    """Validate and return per-case results for VLM critic."""
    # Flush allocator before switching model to eval mode
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    model.eval()
    total_loss = 0
    count = 0
    tracker = MetricTracker(num_classes=4)

    # For VLM critic: collect per-case predictions
    per_case_results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
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

                # Per-case metrics for VLM critic
                pred_np = preds[0].cpu().numpy()
                gt_np = lbl[0, 0].cpu().numpy()
                case_id = batch["case_id"][0] if "case_id" in batch else f"case_{count}"

                dice_per_class = []
                for c in range(4):
                    p_mask = pred_np == c
                    g_mask = gt_np == c
                    inter = (p_mask & g_mask).sum()
                    dice = 2 * inter / (p_mask.sum() + g_mask.sum() + 1e-8)
                    dice_per_class.append(float(dice))

                per_case_results.append(
                    {
                        "case_id": case_id,
                        "dice_per_class": dice_per_class,
                        "mean_dice_fg": float(np.mean(dice_per_class[1:])),
                    }
                )

                del img, lbl, out, loss, preds
            except RuntimeError as e:
                print(f"  ⚠ Validation error: {e}")
                torch.cuda.empty_cache()
                continue

    report, mean_dice = tracker.format_results()
    metrics_dict = tracker.compute()
    return total_loss / max(count, 1), mean_dice, report, metrics_dict, per_case_results


def save_epoch_results(
    output_dir,
    stage,
    epoch,
    train_loss,
    val_loss,
    metrics,
    epoch_time,
    is_best=False,
    critic_feedback=None,
):
    """Save epoch results to JSON."""
    epoch_data = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "epoch": epoch,
        "epoch_time_minutes": round(epoch_time / 60, 2),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "metrics": {
            "mean_dice_fg": float(metrics["dice"][1:].mean()),
            "classes": {
                cls: {
                    "dice": float(metrics["dice"][i]),
                    "iou": float(metrics["iou"][i]),
                    "precision": float(metrics["precision"][i]),
                    "recall": float(metrics["recall"][i]),
                }
                for i, cls in enumerate(CLASS_NAMES)
            },
        },
        "is_best": bool(is_best),
    }
    if critic_feedback:
        epoch_data["vlm_critic_feedback"] = critic_feedback

    history_file = Path(output_dir) / "training_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = {"version": "V5_VLM_CRITIC_MGPL", "epochs": []}

    history["epochs"].append(epoch_data)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def save_checkpoint(
    output_dir,
    epoch,
    stage,
    model,
    optimizer,
    scheduler,
    best_dice,
    scaler,
    critic=None,
):
    """Save training checkpoint including VLM critic state."""
    ckpt = {
        "epoch": epoch,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "best_dice": best_dice,
    }
    if critic:
        ckpt["critic_state"] = {
            "adaptive_class_weights": critic.adaptive_class_weights.tolist(),
            "sampling_bias": critic.sampling_bias,
            "case_difficulty_scores": critic.case_difficulty_scores,
        }

    path = Path(output_dir) / f"checkpoint_{stage}_epoch{epoch + 1}.pth"
    torch.save(ckpt, path)
    print(f"  → Checkpoint saved: {path}")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================


def train(config: dict, resume: bool = False):
    # ── Reproducibility ─────────────────────────────────────────────────────
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Force-clear any orphaned allocations from previous OOM crashes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed patch size → cuDNN auto-tunes optimal conv algorithms once then reuses
    # NOTE: disabled on WSL2 — cudnn.benchmark + AMP corrupts CUDACachingAllocator
    # torch.backends.cudnn.benchmark = True  # only safe on bare-metal Linux
    torch.backends.cudnn.benchmark = False

    print(f"\n{'=' * 70}")
    print(f"MedDINO-VISTA3D V5 — VLM-as-Critic + Multi-Granularity Prompts")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"MGPL:  {'ENABLED' if config['use_mgpl'] else 'DISABLED (ablation)'}")
    print(f"VITL:  {'ENABLED' if config['use_vlm_critic'] else 'DISABLED (ablation)'}")
    print(f"Patch: {config['patch_size']}")
    print(f"{'=' * 70}\n")

    # ── Data ──
    train_loader, val_loader, val_dicts = get_dataloaders(config)
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # ── Model ──
    # For a 32GB GPU, gradient checkpointing is essential for this large architecture
    model = MedDINOVISTA3D_V5(
        num_classes=config["num_classes"],
        feature_dim=config["feature_dim"],
        cnn_channels=config["cnn_channels"],
        dinov2_backbone=config["dinov2_backbone"],
        clip_model=config["clip_model"],
        clip_pretrained=config["clip_pretrained"],
        freeze_encoder=True,
        use_mgpl=config["use_mgpl"],
    ).to(device)

    # ── Loss ──
    loss_fn = AdaptiveHybridLoss().to(device)

    # ── VLM Critic ──
    critic = None
    if config["use_vlm_critic"]:
        critic = VLMCritic(config)
        print(f"✓ VLM Critic enabled: {config['vlm_backend']}:{config['vlm_model']}")
        print(
            f"  Runs every {config['critic_every_k_epochs']} epochs, "
            f"analyzes top {config['critic_top_n_worst']} worst cases"
        )

    # ── Optimizer / Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr_stage1"], weight_decay=config["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda")

    num_epochs_s1 = config["num_epochs_stage1"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs_s1
    )
    if config.get("use_warmup"):
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=config["warmup_epochs"]
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, scheduler], milestones=[config["warmup_epochs"]]
        )

    early_stopping = EarlyStopping(
        patience=config["patience_stage1"],
        verbose=True,
        path=str(output_dir / "best_model_stage1.pth"),
    )

    # ── STAGE 1: Train Decoder + CNN (Frozen DINOv2) ──
    print("\n─── STAGE 1: Train Decoder + 3D CNN (DINOv2 Frozen) ───")
    best_dice = 0.0
    t_start = time.time()
    start_epoch = 0
    val_every_s1 = config.get("val_every_s1", 2)
    # Cache last known val results so we can still log on skipped epochs
    last_val_loss, last_mean_dice, last_report = 0.0, 0.0, ""
    last_metrics_dict, last_per_case = {}, []

    # ── RESUME LOGIC ──
    if resume:
        ckpt_files = sorted(output_dir.glob("checkpoint_stage1_epoch*.pth"))
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]
            print(f"\n  📂 Resuming from: {latest_ckpt.name}")
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)

            start_epoch = ckpt.get("epoch", 0) + 1  # resume from next epoch
            best_dice = ckpt.get("best_dice", 0.0)

            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])

            print(f"  ✓ Resumed at epoch {start_epoch} (best_dice={best_dice:.4f})\n")
        else:
            print(
                f"\n  ⚠ --resume set but no stage1 checkpoints found; starting from scratch\n"
            )

    for epoch in range(start_epoch, num_epochs_s1):
        t_ep = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device
        )

        # Validate every val_every_s1 epochs + always on first and last epoch
        do_val = (
            (epoch + 1) % val_every_s1 == 0 or epoch == 0 or epoch == num_epochs_s1 - 1
        )
        if do_val:
            (
                last_val_loss,
                last_mean_dice,
                last_report,
                last_metrics_dict,
                last_per_case,
            ) = validate(model, val_loader, loss_fn, device)
        val_loss, mean_dice, report, metrics_dict, per_case = (
            last_val_loss,
            last_mean_dice,
            last_report,
            last_metrics_dict,
            last_per_case,
        )

        scheduler.step()
        ep_time = time.time() - t_ep

        is_best = mean_dice > best_dice
        val_marker = "*" if do_val else " "
        print(
            f"Ep {epoch + 1}/{num_epochs_s1}{val_marker}| "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Dice: {mean_dice:.4f} | Time: {ep_time / 60:.1f}m"
        )
        if do_val:
            print(report)

        # ── VLM CRITIC ──
        critic_feedback = None
        if (
            do_val
            and critic is not None
            and (epoch + 1) % config["critic_every_k_epochs"] == 0
            and epoch > 0
        ):
            # For critic: need prediction/GT volumes (using per-case patch results)
            critic_feedback = critic.analyze_epoch(
                epoch=epoch + 1,
                val_cases=per_case,
                predictions={},
                ground_truths={},
                images={},
                agg_metrics=metrics_dict,  # authoritative aggregated dice/prec/recall
            )

            # Apply feedback to loss + dataset
            if critic_feedback.get("adaptive_class_weights"):
                new_w = torch.tensor(critic_feedback["adaptive_class_weights"])
                loss_fn.update_weights(new_w)
                print(f"  [VITL] Loss weights updated: {new_w.tolist()}")

            if critic_feedback.get("sampling_bias"):
                train_loader.dataset.update_sampling(
                    critic_feedback["sampling_bias"],
                    critic_feedback.get("case_difficulty_scores", {}),
                )
                print(
                    f"  [VITL] Sampling bias updated: {critic_feedback['sampling_bias']}"
                )

        # Save epoch results (only on val epochs — others have no new val metrics)
        if do_val:
            save_epoch_results(
                output_dir,
                "stage1",
                epoch + 1,
                train_loss,
                val_loss,
                metrics_dict,
                ep_time,
                is_best,
                critic_feedback,
            )

        # Track dice (negated so EarlyStopping's "lower is better" logic works)
        # Only update when we actually ran validation this epoch
        if do_val:
            early_stopping(-mean_dice, model)
            if early_stopping.early_stop:
                print("Early stopping triggered (Stage 1)")
                break

        if is_best:
            best_dice = mean_dice
            torch.save(model.state_dict(), output_dir / "best_model_stage1_dice.pth")
            print(f"  → Best Dice: {best_dice:.4f}")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                output_dir,
                epoch,
                "stage1",
                model,
                optimizer,
                scheduler,
                best_dice,
                scaler,
                critic,
            )

    # ── STAGE 2: Fine-tune DINOv2 ──
    if config["num_epochs_stage2"] > 0:
        print("\n─── STAGE 2: Fine-tune DINOv2 Encoder ───")

        # Load best stage 1 model
        best_path = output_dir / "best_model_stage1_dice.pth"
        if best_path.exists():
            model.load_state_dict(
                torch.load(best_path, map_location=device, weights_only=True)
            )
        elif (output_dir / "best_model_stage1.pth").exists():
            model.load_state_dict(
                torch.load(
                    output_dir / "best_model_stage1.pth",
                    map_location=device,
                    weights_only=True,
                )
            )

        model.unfreeze_encoder(num_layers=4)

        # Start from near-zero LR and warm up to lr_stage2 over a few epochs.
        # This prevents the sudden gradient shock from newly unfrozen DINOv2 layers
        # dropping tumor Dice by 0.12+ at Stage 2 epoch 1.
        warmup_s2 = config.get("warmup_s2_epochs", 3)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr_stage2"] / warmup_s2,  # start at fraction of target LR
            weight_decay=config["weight_decay"],
        )
        warmup_s2_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_s2,
            end_factor=1.0,
            total_iters=warmup_s2,
        )
        cosine_s2_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config["num_epochs_stage2"] - warmup_s2, 1)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_s2_sched, cosine_s2_sched], milestones=[warmup_s2]
        )
        early_stopping = EarlyStopping(
            patience=config["patience_stage2"],
            verbose=True,
            path=str(output_dir / "best_model_final.pth"),
        )

        val_every_s2 = config.get("val_every_s2", 1)
        s2_last_val_loss, s2_last_dice, s2_last_report = 0.0, best_dice, ""
        s2_last_metrics, s2_last_per_case = {}, []

        # ── RESUME LOGIC FOR STAGE 2 ──
        start_epoch_s2 = 0
        if resume:
            ckpt_files = sorted(output_dir.glob("checkpoint_stage2_epoch*.pth"))
            if ckpt_files:
                latest_ckpt = ckpt_files[-1]
                print(f"\n  📂 Resuming Stage 2 from: {latest_ckpt.name}")
                ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)

                start_epoch_s2 = ckpt.get("epoch", 0) + 1  # resume from next epoch
                best_dice = ckpt.get("best_dice", 0.0)

                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                scaler.load_state_dict(ckpt["scaler_state_dict"])

                print(
                    f"  ✓ Resumed at Stage 2 epoch {start_epoch_s2} (best_dice={best_dice:.4f})\n"
                )

        for epoch in range(start_epoch_s2, config["num_epochs_stage2"]):
            t_ep = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, scaler, device
            )

            do_val = (
                (epoch + 1) % val_every_s2 == 0
                or epoch == 0
                or epoch == config["num_epochs_stage2"] - 1
            )
            if do_val:
                (
                    s2_last_val_loss,
                    s2_last_dice,
                    s2_last_report,
                    s2_last_metrics,
                    s2_last_per_case,
                ) = validate(model, val_loader, loss_fn, device)
            val_loss, mean_dice, report, metrics_dict, per_case = (
                s2_last_val_loss,
                s2_last_dice,
                s2_last_report,
                s2_last_metrics,
                s2_last_per_case,
            )

            scheduler.step()
            ep_time = time.time() - t_ep

            is_best = mean_dice > best_dice
            val_marker = "*" if do_val else " "
            print(
                f"Ep {epoch + 1}/{config['num_epochs_stage2']}{val_marker}| "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Dice: {mean_dice:.4f} | Time: {ep_time / 60:.1f}m"
            )
            if do_val:
                print(report)

            # VLM critic in stage 2 also
            critic_feedback = None
            if (
                do_val
                and critic is not None
                and (epoch + 1) % config["critic_every_k_epochs"] == 0
            ):
                critic_feedback = critic.analyze_epoch(
                    epoch + 1, per_case, {}, {}, {}, agg_metrics=metrics_dict
                )
                if critic_feedback.get("adaptive_class_weights"):
                    loss_fn.update_weights(
                        torch.tensor(critic_feedback["adaptive_class_weights"])
                    )

            if do_val:
                save_epoch_results(
                    output_dir,
                    "stage2",
                    epoch + 1,
                    train_loss,
                    val_loss,
                    metrics_dict,
                    ep_time,
                    is_best,
                    critic_feedback,
                )

            if do_val:
                early_stopping(-mean_dice, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered (Stage 2)")
                    break

            if is_best:
                best_dice = mean_dice
                torch.save(model.state_dict(), output_dir / "best_model_final_dice.pth")
                print(f"  → Best Final Dice: {best_dice:.4f}")

            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    output_dir,
                    epoch,
                    "stage2",
                    model,
                    optimizer,
                    scheduler,
                    best_dice,
                    scaler,
                    critic,
                )

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(
        f"Training Complete | Best Dice: {best_dice:.4f} | "
        f"Time: {total_time / 3600:.2f}h"
    )
    print(f"{'=' * 70}")

    # Save final summary
    summary = {
        "version": "V5_VLM_CRITIC_MGPL",
        "best_dice": float(best_dice),
        "total_time_hours": round(total_time / 3600, 2),
        "total_params": sum(p.numel() for p in model.parameters()),
        "mgpl_enabled": config["use_mgpl"],
        "vlm_critic_enabled": config["use_vlm_critic"],
        "config": {k: str(v) for k, v in config.items()},
    }
    if critic:
        summary["critic_history"] = critic.error_history

    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return model


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedDINO-VISTA3D V5: VLM-as-Critic + Multi-Granularity Prompts"
    )
    parser.add_argument(
        "--mode", type=str, default="full", choices=["quick_test", "full"]
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--no-vlm-critic",
        action="store_true",
        help="Ablation: disable VLM critic (MGPL only)",
    )
    parser.add_argument(
        "--no-mgpl",
        action="store_true",
        help="Ablation: disable MGPL (VLM critic only)",
    )
    parser.add_argument(
        "--vlm-backend",
        type=str,
        default=None,
        choices=["ollama", "transformers", "openai"],
        help="VLM backend override",
    )
    parser.add_argument(
        "--vlm-model", type=str, default=None, help="VLM model name override"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    config = get_config(args.mode)

    if args.no_vlm_critic:
        config["use_vlm_critic"] = False
    if args.no_mgpl:
        config["use_mgpl"] = False
    if args.vlm_backend:
        config["vlm_backend"] = args.vlm_backend
    if args.vlm_model:
        config["vlm_model"] = args.vlm_model
    if args.seed is not None:
        config["seed"] = args.seed

    train(config, resume=args.resume)
