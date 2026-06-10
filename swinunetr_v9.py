"""
SwinUNETR V9 — LLM-Enhanced KiTS23 Kidney Segmentation
=======================================================
Based on MedVisionLlama (ICCV 2025 Workshop, Kumar et al.) approach:
  Insert a frozen LLM transformer block at the SwinUNETR bottleneck,
  acting as a "residual attention booster" that refines features.

Architecture:
  CT Volume  →  SwinViT Encoder (pretrained, 5 scales)
                      ↓
              Bottleneck features (B, 768, D/32, H/32, W/32)
                      ↓
              Proj_in (768 → LLM_dim)  ──  trainable
                      ↓
              Frozen LLM Block + LoRA  ──  attention refinement
                      ↓
              Proj_out (LLM_dim → 768) ──  trainable
                      ↓
              + residual + LayerNorm
                      ↓
              SwinUNETR Decoder (skip connections) → 4-class output

Why this works (and why V6b's CLIP approach failed):
  V6b: injected frozen CLIP 2D natural-image TEXT EMBEDDINGS into every
       decoder scale → noise injection at every scale, no spatial grounding
  V9:  uses LLM transformer WEIGHTS as attention refinement at bottleneck
       → LLM's learned attention patterns help focus on relevant structures
       → only at bottleneck (not every scale), residual means it can't hurt
       → no text prompts needed — just the pretrained weights

Key results from MedVisionLlama paper:
  - Dice 0.74 → 0.87 on MSD (10 tasks, CT and MRI)
  - Beats nnUNet (0.81), Swin-UNet (0.85), MissFormer (0.84)
  - General LLMs (Llama) work as well as medical LLMs (BioGPT, ClinicalBERT)
  - LoRA rank 4 is optimal; rank 8 slightly worse
  - Ablation: gains from LLM weights, NOT from added parameters

References:
  [1] MedVisionLlama: Kumar et al., ICCV 2025 Workshop (CVAMD)
      "Leveraging Pre-Trained LLM Layers to Enhance Medical Image Segmentation"
  [2] Pang et al., 2023 "Frozen Transformers in Language Models are
      Effective Visual Encoder Layers" — the theoretical foundation
  [3] Med-VLM: Zhao et al., ICCV 2025 Workshop (PHAROS-AFE-AIMI)
  [4] VILA-M3: Nath et al., CVPR 2025 (NVIDIA)

Usage:
    python swinunetr_v9.py --mode quick_test          # smoke test, ~30-60 min
    python swinunetr_v9.py --quick-test-baseline      # smoke test baseline, no LLM

    # ── Showcase LLM vs baseline (paired comparison in ~3-4 h total) ──────────
    python swinunetr_v9.py --mode showcase              # run 1: WITH LLM  (~1.5 h)
    python swinunetr_v9.py --mode showcase --no-llm \   # run 2: WITHOUT   (~1.5 h)
        --output-dir ./output/v9_showcase_baseline
    # then compare: cat output/v9_showcase/summary.json output/v9_showcase_baseline/summary.json
    # ─────────────────────────────────────────────────────────────────────────

    python swinunetr_v9.py --mode medium               # real Dice fast, ~8-16 h
    python swinunetr_v9.py --mode full                 # full training, ~3-5 days
    python swinunetr_v9.py --mode full --llm-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python swinunetr_v9.py --mode evaluate --checkpoint ./output/v9/best_final.pth
"""

import os
import gc
import json
import time
import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]


# ============================================================================
# CONFIGURATION
# ============================================================================


def get_config(mode: str) -> dict:
    base = {
        # Paths
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/v9",
        # Model
        "num_classes": 4,
        "feature_size": 48,
        # LLM bottleneck block
        "use_llm": True,
        "llm_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "llm_layer_idx": -1,       # last layer (deepest representations)
        "lora_rank": 4,            # LoRA rank — 4 is optimal per MedVisionLlama
        # Data
        "patch_size": (128, 128, 128),
        "batch_size": 2,
        "num_samples_per_volume": 2,
        "val_split": 0.20,
        "test_cases": 40,
        "seed": 42,
        # Stage 1: frozen SwinViT encoder, decoder + LLM adapters train
        "num_epochs_stage1": 50,
        "lr_stage1": 1e-3,
        "patience_stage1": 15,
        "val_every_s1": 5,
        # Stage 2: full model fine-tuning (LLM block stays frozen, LoRA trains)
        "num_epochs_stage2": 250,
        "lr_stage2": 2e-4,
        "patience_stage2": 30,
        "val_every_s2": 5,
        "weight_decay": 1e-5,
        # Sliding window validation
        "sw_val_cases": 20,
        "sw_overlap": 0.25,
        # Post-processing
        "min_kidney_voxels": 5000,
        "min_tumor_voxels": 50,
        "min_cyst_voxels": 50,
    }

    if mode == "quick_test":
        # Smoke test — verifies code runs, gives noisy/low Dice. ~30-60 min on RTX 5090.
        base.update({
            "num_epochs_stage1": 2,
            "num_epochs_stage2": 2,
            "patience_stage1": 3,
            "patience_stage2": 3,
            "val_every_s1": 5,
            "val_every_s2": 5,
            "sw_val_cases": 3,
            "sw_overlap": 0.0,
            "quick_cases": 30,
            "output_dir": "./output/v9_quick",
        })
    elif mode == "showcase":
        # Paired LLM-vs-baseline comparison run. ~3-4 h per run on RTX 5090.
        # Run TWICE: once normally, once with --no-llm --output-dir ./output/v9_showcase_baseline
        # 150 cases → ~120 train / ~30 val — good data coverage.
        # num_samples=4 → 480 patches/epoch → more training signal per epoch.
        # overlap=0.0 → fastest valid sliding-window (no redundant patches).
        base.update({
            "num_epochs_stage1": 15,
            "num_epochs_stage2": 60,
            "patience_stage1": 6,
            "patience_stage2": 15,
            "val_every_s1": 5,
            "val_every_s2": 5,
            "sw_val_cases": 8,
            "sw_overlap": 0.0,
            "num_samples_per_volume": 4,  # 4x patches/vol → denser training
            "quick_cases": 150,           # 150 cases → ~120 train / ~30 val
            "output_dir": "./output/v9_showcase",
        })
    elif mode == "medium":
        # Fast meaningful result — full data, shorter training. ~8-16 h on RTX 5090.
        # Good enough Dice to show progress to supervisor / justify full training.
        base.update({
            "num_epochs_stage1": 30,
            "num_epochs_stage2": 80,
            "patience_stage1": 10,
            "patience_stage2": 20,
            "val_every_s1": 5,
            "val_every_s2": 5,
            "sw_val_cases": 12,
            "sw_overlap": 0.25,
            "quick_cases": None,                  # full dataset
            "output_dir": "./output/v9_medium",
        })
    else:
        base["quick_cases"] = None

    return base


# ============================================================================
# LLM COMPONENTS — Frozen LLM Block with LoRA (MedVisionLlama approach)
# ============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used by Llama models)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for a frozen linear layer.
    Per Hu et al. (2022) and MedVisionLlama ablation:
      rank=4 → best accuracy/efficiency tradeoff
      alpha=rank → scaling=1.0

    The base linear stays frozen. Only lora_A and lora_B are trained.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 4):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base_linear.in_features
        out_f = base_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A)
        self.scaling = 1.0  # alpha / rank = rank / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base.weight, self.base.bias)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling


class LLMTransformerBlock(nn.Module):
    """
    Clean transformer block with SwiGLU MLP, bidirectional attention, no RoPE.
    Weights can be initialized from a pre-trained LLM (Llama architecture).

    Architecture (pre-norm, Llama-style):
      x → RMSNorm → MultiHeadAttn (GQA) → + residual
        → RMSNorm → SwiGLU MLP           → + residual

    No rotary embeddings: positional information comes from the 3D spatial
    structure of the bottleneck features, not from sequential text ordering.
    No causal mask: bidirectional attention over all spatial tokens.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.kv_dim = num_kv_heads * self.head_dim

        # Pre-norm layers (RMSNorm as in Llama)
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

        # Attention projections (supports GQA: num_kv_heads < num_heads)
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        # SwiGLU MLP
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat K/V heads to match Q heads for GQA."""
        if self.num_kv_heads == self.num_heads:
            return x
        n_rep = self.num_heads // self.num_kv_heads
        B, H, N, D = x.shape
        return (
            x.unsqueeze(2)
            .expand(B, H, n_rep, N, D)
            .reshape(B, H * n_rep, N, D)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, hidden_dim)

        # Pre-norm attention
        h = self.norm1(x)
        B, N, _ = h.shape

        q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K/V for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Bidirectional attention (no causal mask, no RoPE)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        attn_out = self.o_proj(attn_out)
        x = x + attn_out

        # Pre-norm SwiGLU MLP
        h = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

        return x


class LLMBottleneckBlock(nn.Module):
    """
    Processes SwinUNETR bottleneck features through a frozen LLM transformer
    block with LoRA adapters. Based on MedVisionLlama (ICCV 2025 Workshop).

    Flow:
      (B, C, D, H, W) → flatten → (B, N, C)
         → proj_in (C → llm_dim)
         → LLM block (frozen + LoRA)
         → proj_out (llm_dim → C)
         → + residual → LayerNorm
         → reshape → (B, C, D, H, W)

    The frozen LLM block uses pretrained attention weights that act as
    "residual attention boosters" — refining feature representations with
    rich structural priors learned from large-scale text pretraining.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        llm_layer_idx: int = -1,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.vision_dim = vision_dim

        # Get LLM architecture config
        llm_cfg = self._get_llm_config(llm_model_name)
        llm_dim = llm_cfg["hidden_dim"]
        self.llm_dim = llm_dim

        # Build clean transformer block
        self.llm_block = LLMTransformerBlock(
            hidden_dim=llm_dim,
            num_heads=llm_cfg["num_heads"],
            num_kv_heads=llm_cfg["num_kv_heads"],
            ffn_dim=llm_cfg["ffn_dim"],
        )

        # Try to load pretrained weights
        self._load_llm_weights(llm_model_name, llm_layer_idx)

        # Freeze all LLM block base weights
        for p in self.llm_block.parameters():
            p.requires_grad = False

        # Apply LoRA to attention projections
        self._apply_lora(lora_rank)

        # Trainable dimension mapping
        self.proj_in = nn.Linear(vision_dim, llm_dim)
        self.proj_out = nn.Linear(llm_dim, vision_dim)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

        # Output normalization for residual
        self.norm = nn.LayerNorm(vision_dim)

        # Count params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"  LLM block: {total:,} total | {trainable:,} trainable | {frozen:,} frozen")

    def _get_llm_config(self, model_name: str) -> dict:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            cfg = {
                "hidden_dim": config.hidden_size,
                "num_heads": config.num_attention_heads,
                "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
                "ffn_dim": config.intermediate_size,
                "num_layers": config.num_hidden_layers,
            }
            print(f"  LLM config: {model_name} → dim={cfg['hidden_dim']}, "
                  f"heads={cfg['num_heads']}/{cfg['num_kv_heads']}, "
                  f"ffn={cfg['ffn_dim']}, layers={cfg['num_layers']}")
            return cfg
        except Exception as e:
            print(f"  ⚠ Cannot load LLM config ({e}), using TinyLlama defaults")
            return {
                "hidden_dim": 2048, "num_heads": 32,
                "num_kv_heads": 4, "ffn_dim": 5632, "num_layers": 22,
            }

    def _load_llm_weights(self, model_name: str, layer_idx: int):
        # Check for cached extracted layer
        cache_dir = Path.home() / ".cache" / "llm_layers"
        safe_name = model_name.replace("/", "_")
        cache_path = cache_dir / f"{safe_name}_layer{layer_idx}.pth"

        if cache_path.exists():
            print(f"  Loading cached LLM layer: {cache_path}")
            cached_state = torch.load(str(cache_path), map_location="cpu", weights_only=True)
            loaded = self._load_mapped_weights(cached_state)
            print(f"  ✓ Loaded {loaded} weights from cache")
            return

        try:
            from transformers import AutoModelForCausalLM
            print(f"  Downloading LLM: {model_name} ...")
            llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

            layers = llm.model.layers
            idx = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
            layer = layers[idx]
            layer_state = {k: v.clone() for k, v in layer.state_dict().items()}

            # Cache for next time
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(layer_state, str(cache_path))

            loaded = self._load_mapped_weights(layer_state)
            print(f"  ✓ Loaded {loaded} LLM weights from layer {idx} "
                  f"(cached → {cache_path})")

            del llm, layers, layer, layer_state
            gc.collect()

        except Exception as e:
            print(f"  ⚠ Cannot load LLM weights: {e}")
            print(f"  → Using randomly initialized block (still effective per ablation)")

    def _load_mapped_weights(self, layer_state: dict) -> int:
        """Map HuggingFace Llama layer keys to our clean block keys."""
        key_map = {
            "norm1.weight": "input_layernorm.weight",
            "norm2.weight": "post_attention_layernorm.weight",
            "q_proj.weight": "self_attn.q_proj.weight",
            "k_proj.weight": "self_attn.k_proj.weight",
            "v_proj.weight": "self_attn.v_proj.weight",
            "o_proj.weight": "self_attn.o_proj.weight",
            "gate_proj.weight": "mlp.gate_proj.weight",
            "up_proj.weight": "mlp.up_proj.weight",
            "down_proj.weight": "mlp.down_proj.weight",
        }
        our_sd = self.llm_block.state_dict()
        mapped = {}
        for our_key, hf_key in key_map.items():
            if hf_key in layer_state and our_key in our_sd:
                if our_sd[our_key].shape == layer_state[hf_key].shape:
                    mapped[our_key] = layer_state[hf_key]
        self.llm_block.load_state_dict(mapped, strict=False)
        return len(mapped)

    def _apply_lora(self, rank: int):
        """Wrap attention projections with LoRA adapters."""
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            original = getattr(self.llm_block, name)
            setattr(self.llm_block, name, LoRALinear(original, rank))
        lora_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"  LoRA applied (rank={rank}): {lora_params:,} trainable adapter params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        residual = x

        # Flatten spatial dims → sequence: (B, D*H*W, C)
        x_seq = x.flatten(2).permute(0, 2, 1)

        # Project to LLM dim → process → project back
        h = self.proj_in(x_seq)
        h = self.llm_block(h)
        h = self.proj_out(h)

        # Residual + norm
        out = self.norm(x_seq + h)

        # Reshape to 3D
        return out.permute(0, 2, 1).view(B, C, D, H, W)


# ============================================================================
# MODEL — SwinUNETR with LLM bottleneck
# ============================================================================


class SwinUNETR_LLM(nn.Module):
    """
    MONAI SwinUNETR with optional frozen LLM transformer block at bottleneck.

    Encoder:  SwinViT — pretrained on 5050 CT/MRI volumes.
    Bottleneck: Frozen LLM block + LoRA — refines features via attention.
    Decoder:  MONAI's native UnetrBasicBlock/UnetrUpBlock with skip connections.

    Why NOT insert at every scale (like V6b)?
      MedVisionLlama inserts LLM between encoder and decoder — only at
      the bottleneck. This is sufficient because:
      1) Bottleneck has the most abstract/semantic features
      2) Skip connections carry spatial detail from encoder to decoder
      3) Multi-scale injection adds noise (V6b showed this empirically)
    """

    def __init__(
        self,
        num_classes: int = 4,
        feature_size: int = 48,
        use_llm: bool = True,
        llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        llm_layer_idx: int = -1,
        lora_rank: int = 4,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.net = SwinUNETR(
            in_channels=1,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=True,
            spatial_dims=3,
        )
        self._load_pretrained_encoder()

        # LLM bottleneck block
        self.use_llm = use_llm
        bottleneck_dim = feature_size * 16  # 768 for feature_size=48
        if use_llm:
            self.llm_block = LLMBottleneckBlock(
                vision_dim=bottleneck_dim,
                llm_model_name=llm_model_name,
                llm_layer_idx=llm_layer_idx,
                lora_rank=lora_rank,
            )
        else:
            self.llm_block = None
            print("  LLM block disabled — running as pure SwinUNETR baseline")

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"SwinUNETR V9 {'+ LLM' if use_llm else '(baseline)'}")
        print(f"Total: {total:,} | Trainable: {trainable:,}")
        print(f"{'='*60}\n")

    def _load_pretrained_encoder(self):
        cached = Path.home() / ".torch" / "models" / "model_swinvit.pt"
        local = Path("./swin_unetr_pretrained.pth")

        if cached.exists():
            wpath = str(cached)
            print(f"  ✓ Using cached SwinUNETR weights: {wpath}")
        elif local.exists():
            wpath = str(local)
            print(f"  ✓ Using local SwinUNETR weights: {wpath}")
        else:
            print("  ⚠ No pretrained encoder found — training from scratch")
            return

        try:
            ckpt = torch.load(wpath, map_location="cpu", weights_only=False)
            weights = ckpt.get("state_dict", ckpt)
            model_dict = self.net.state_dict()
            matched = {}
            for k, v in weights.items():
                mk = k
                if mk.startswith("module."):
                    mk = mk[len("module."):]
                if mk in model_dict and model_dict[mk].shape == v.shape:
                    matched[mk] = v
                else:
                    mk2 = "swinViT." + mk
                    if mk2 in model_dict and model_dict[mk2].shape == v.shape:
                        matched[mk2] = v
            model_dict.update(matched)
            self.net.load_state_dict(model_dict, strict=False)
            print(f"  ✓ Loaded {len(matched)}/{len(model_dict)} pretrained weights")
        except Exception as e:
            print(f"  ⚠ Could not load pretrained weights: {e}")

    def freeze_encoder(self):
        """Freeze SwinViT encoder — decoder + LLM adapters train in Stage 1."""
        for p in self.net.swinViT.parameters():
            p.requires_grad = False
        # LLM base weights are ALWAYS frozen (only LoRA adapters train)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_enc = sum(p.numel() for p in self.net.swinViT.parameters())
        print(f"  Encoder frozen ({frozen_enc:,}) | Trainable ({trainable:,})")

    def unfreeze_encoder(self):
        """Unfreeze SwinViT for Stage 2. LLM base weights stay frozen."""
        for p in self.net.swinViT.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Encoder unfrozen | Trainable: {trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Custom forward that intercepts SwinUNETR internals to insert the
        LLM block at the bottleneck, between encoder10 and decoder5.
        """
        # SwinViT encoder — extract multi-scale features
        hidden_states_out = self.net.swinViT(x, self.net.normalize)

        # Encoder blocks (process features at each scale)
        enc0 = self.net.encoder1(x)
        enc1 = self.net.encoder2(hidden_states_out[0])
        enc2 = self.net.encoder3(hidden_states_out[1])
        enc3 = self.net.encoder4(hidden_states_out[2])

        # Bottleneck
        dec4 = self.net.encoder10(hidden_states_out[4])

        # ── LLM BOTTLENECK ENHANCEMENT ──
        if self.use_llm and self.llm_block is not None:
            dec4 = self.llm_block(dec4)

        # Decoder (upsampling with skip connections)
        dec3 = self.net.decoder5(dec4, hidden_states_out[3])
        dec2 = self.net.decoder4(dec3, enc3)
        dec1 = self.net.decoder3(dec2, enc2)
        dec0 = self.net.decoder2(dec1, enc1)
        out = self.net.decoder1(dec0, enc0)

        return self.net.out(out)


# ============================================================================
# LOSS
# ============================================================================


class DiceCELoss(nn.Module):
    """
    Soft Dice (foreground) + weighted Cross-Entropy.
    CE weights: bg=0.1, kidney=1.0, tumor=8.0, cyst=4.0
    """

    def __init__(self, num_classes: int = 4, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.register_buffer("ce_weight", torch.tensor([0.1, 1.0, 8.0, 4.0]))

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, sample_weight: float = 1.0
    ) -> torch.Tensor:
        tgt = target.squeeze(1).long()
        pred_soft = F.softmax(logits, dim=1)
        tgt_oh = F.one_hot(tgt, self.num_classes).permute(0, 4, 1, 2, 3).float()

        tp = (pred_soft * tgt_oh).sum(dim=(2, 3, 4))
        fp = pred_soft.sum(dim=(2, 3, 4)) - tp
        fn = tgt_oh.sum(dim=(2, 3, 4)) - tp
        dice_per_cls = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        loss_dice = (1 - dice_per_cls[:, 1:]).mean()

        loss_ce = F.cross_entropy(logits, tgt, weight=self.ce_weight)

        return (0.5 * loss_dice + 0.5 * loss_ce) * sample_weight


# ============================================================================
# AUGMENTATION — nnUNet-style
# ============================================================================


def augment_patch(
    img: np.ndarray, lbl: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # 1. Random flip
    for ax in range(3):
        if np.random.random() < 0.5:
            img = np.flip(img, axis=ax)
            lbl = np.flip(lbl, axis=ax)

    # 2. Random 90° rotation
    if np.random.random() < 0.40:
        k = np.random.randint(1, 4)
        axes = tuple(np.random.choice(3, size=2, replace=False).tolist())
        img = np.rot90(img, k=k, axes=axes)
        lbl = np.rot90(lbl, k=k, axes=axes)

    # 3. Scale augmentation
    if np.random.random() < 0.15:
        scale = float(np.random.uniform(0.88, 1.12))
        pd, ph, pw = img.shape
        img_s = ndimage.zoom(img.astype(np.float32), scale, order=1)
        lbl_s = ndimage.zoom(lbl.astype(np.float32), scale, order=0)
        img_out = np.zeros((pd, ph, pw), dtype=np.float32)
        lbl_out = np.zeros((pd, ph, pw), dtype=lbl.dtype)
        sd, sh, sw = img_s.shape
        s0 = max(0, (sd - pd) // 2)
        s1 = max(0, (sh - ph) // 2)
        s2 = max(0, (sw - pw) // 2)
        d0 = max(0, (pd - sd) // 2)
        d1 = max(0, (ph - sh) // 2)
        d2 = max(0, (pw - sw) // 2)
        c0 = min(sd - s0, pd - d0)
        c1 = min(sh - s1, ph - d1)
        c2 = min(sw - s2, pw - d2)
        if c0 > 0 and c1 > 0 and c2 > 0:
            img_out[d0:d0+c0, d1:d1+c1, d2:d2+c2] = img_s[s0:s0+c0, s1:s1+c1, s2:s2+c2]
            lbl_out[d0:d0+c0, d1:d1+c1, d2:d2+c2] = lbl_s[s0:s0+c0, s1:s1+c1, s2:s2+c2]
            img, lbl = img_out, lbl_out

    # 4. Gaussian noise
    if np.random.random() < 0.25:
        std = float(np.random.uniform(0.01, 0.08))
        img = np.clip(
            img + np.random.randn(*img.shape).astype(np.float32) * std, 0.0, 1.0
        )

    # 5. Intensity shift
    if np.random.random() < 0.20:
        img = np.clip(img + float(np.random.uniform(-0.10, 0.10)), 0.0, 1.0)

    # 6. Gamma augmentation
    if np.random.random() < 0.20:
        gamma = float(np.random.uniform(0.70, 1.50))
        img = np.power(np.clip(img, 0.0, 1.0), gamma)

    # 7. Gaussian blur
    if np.random.random() < 0.15:
        sigma = float(np.random.uniform(0.5, 1.5))
        img = ndimage.gaussian_filter(img.astype(np.float32), sigma=sigma)

    return img.copy(), lbl.copy()


# ============================================================================
# DATASET
# ============================================================================

_SAMPLE_BIAS = {"tumor": 0.60, "cyst": 0.10, "kidney": 0.20, "random": 0.10}


class KiTS23Dataset(Dataset):
    def __init__(
        self,
        data_dicts: List[dict],
        patch_size: Tuple[int, int, int],
        num_samples: int = 2,
        is_train: bool = True,
    ):
        self.patch_size = patch_size
        self.is_train = is_train
        self.items = [d for d in data_dicts for _ in range(num_samples)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        d = self.items[idx]
        img = nib.load(d["image"]).get_fdata().astype(np.float32)
        lbl = nib.load(d["label"]).get_fdata().astype(np.int64)
        img = (np.clip(img, -175.0, 250.0) + 175.0) / 425.0

        if self.is_train:
            img, lbl = self._sample_patch_train(img, lbl)
            img, lbl = augment_patch(img, lbl)
        else:
            img, lbl = self._center_crop(img, lbl)

        return {
            "image": torch.from_numpy(img).float().unsqueeze(0),
            "label": torch.from_numpy(lbl.copy()).long().unsqueeze(0),
            "case_id": d.get("case_id", "unknown"),
        }

    def _pad_if_needed(self, img, lbl):
        pd, ph, pw = self.patch_size
        d, h, w = img.shape
        if d < pd or h < ph or w < pw:
            pad = ((0, max(0, pd-d)), (0, max(0, ph-h)), (0, max(0, pw-w)))
            img = np.pad(img, pad, mode="constant", constant_values=0.0)
            lbl = np.pad(lbl, pad, mode="constant", constant_values=0)
        return img, lbl

    def _sample_patch_train(self, img, lbl):
        img, lbl = self._pad_if_needed(img, lbl)
        d, h, w = img.shape
        pd, ph, pw = self.patch_size

        r = np.random.random()
        b = _SAMPLE_BIAS
        if r < b["tumor"]:
            target_cls = 2
        elif r < b["tumor"] + b["cyst"]:
            target_cls = 3
        elif r < b["tumor"] + b["cyst"] + b["kidney"]:
            target_cls = 1
        else:
            target_cls = None

        ds, hs, ws = self._find_center(lbl, d, h, w, pd, ph, pw, target_cls)
        return img[ds:ds+pd, hs:hs+ph, ws:ws+pw].copy(), lbl[ds:ds+pd, hs:hs+ph, ws:ws+pw].copy()

    def _center_crop(self, img, lbl):
        img, lbl = self._pad_if_needed(img, lbl)
        d, h, w = img.shape
        pd, ph, pw = self.patch_size
        ds = (d - pd) // 2
        hs = (h - ph) // 2
        ws = (w - pw) // 2
        return img[ds:ds+pd, hs:hs+ph, ws:ws+pw], lbl[ds:ds+pd, hs:hs+ph, ws:ws+pw]

    @staticmethod
    def _find_center(lbl, d, h, w, pd, ph, pw, target_cls):
        if target_cls is not None:
            idx = np.argwhere(lbl == target_cls)
            if len(idx) == 0:
                idx = np.argwhere(lbl > 0)
            if len(idx) > 0:
                c = idx[np.random.randint(len(idx))]
                return (
                    int(np.clip(c[0] - pd // 2, 0, d - pd)),
                    int(np.clip(c[1] - ph // 2, 0, h - ph)),
                    int(np.clip(c[2] - pw // 2, 0, w - pw)),
                )
        return (
            np.random.randint(0, max(1, d - pd + 1)),
            np.random.randint(0, max(1, h - ph + 1)),
            np.random.randint(0, max(1, w - pw + 1)),
        )


def get_dataloaders(config):
    data_dir = Path(config["kits23_dir"])
    all_cases = sorted(
        [c for c in data_dir.iterdir() if c.is_dir() and c.name.startswith("case_")]
    )

    n_test = config.get("test_cases", 40)
    all_cases = all_cases[:-n_test]

    if config.get("quick_cases"):
        all_cases = all_cases[: config["quick_cases"]]

    data_dicts = [
        {
            "image": str(c / "imaging.nii.gz"),
            "label": str(c / "segmentation.nii.gz"),
            "case_id": c.name,
        }
        for c in all_cases
        if (c / "imaging.nii.gz").exists() and (c / "segmentation.nii.gz").exists()
    ]

    n_val = int(len(data_dicts) * config["val_split"])
    val_dicts = data_dicts[:n_val]
    train_dicts = data_dicts[n_val:]
    print(f"  Split: {len(train_dicts)} train | {len(val_dicts)} val | {n_test} test")

    train_ds = KiTS23Dataset(
        train_dicts, config["patch_size"],
        num_samples=config["num_samples_per_volume"], is_train=True,
    )
    val_ds = KiTS23Dataset(
        val_dicts, config["patch_size"], num_samples=1, is_train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=8, pin_memory=False, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=False, persistent_workers=True,
    )
    return train_loader, val_loader, train_dicts, val_dicts


# ============================================================================
# POST-PROCESSING
# ============================================================================


def postprocess(pred: np.ndarray, config: dict) -> np.ndarray:
    out = pred.copy()
    for cls_id, min_size in [
        (1, config["min_kidney_voxels"]),
        (2, config["min_tumor_voxels"]),
        (3, config["min_cyst_voxels"]),
    ]:
        mask = out == cls_id
        if not mask.any():
            continue
        labeled, n_components = ndimage.label(mask)
        if n_components == 0:
            continue
        comp_sizes = np.bincount(labeled.ravel())
        lut = comp_sizes < min_size
        lut[0] = False
        out[lut[labeled]] = 0
    return out


# ============================================================================
# SLIDING WINDOW INFERENCE — PyTorch 2.9 compatible
# ============================================================================


def _sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    predictor,
    overlap: float = 0.25,
    mode: str = "constant",
) -> torch.Tensor:
    assert inputs.shape[0] == 1
    device = inputs.device
    pd, ph, pw = roi_size

    _, _, d_orig, h_orig, w_orig = inputs.shape
    pad_d = max(0, pd - d_orig)
    pad_h = max(0, ph - h_orig)
    pad_w = max(0, pw - w_orig)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
    _, _, D, H, W = inputs.shape

    def _starts(size, patch, stride):
        pts = list(range(0, size - patch + 1, stride))
        if not pts or pts[-1] + patch < size:
            pts.append(size - patch)
        return pts

    stride_d = max(1, int(pd * (1.0 - overlap)))
    stride_h = max(1, int(ph * (1.0 - overlap)))
    stride_w = max(1, int(pw * (1.0 - overlap)))

    patch_coords = [
        (sd, sh, sw_s)
        for sd in _starts(D, pd, stride_d)
        for sh in _starts(H, ph, stride_h)
        for sw_s in _starts(W, pw, stride_w)
    ]

    if mode == "gaussian":
        def _gauss1d(n, sigma):
            idx = torch.arange(n).float() - n / 2.0
            return torch.exp(-idx ** 2 / (2.0 * sigma ** 2))
        importance = (
            _gauss1d(pd, pd / 8.0)[:, None, None]
            * _gauss1d(ph, ph / 8.0)[None, :, None]
            * _gauss1d(pw, pw / 8.0)[None, None, :]
        ).unsqueeze(0).unsqueeze(0)
    else:
        importance = torch.ones(1, 1, pd, ph, pw)

    sd0, sh0, sw0 = patch_coords[0]
    first_sl = tuple([
        slice(None), slice(None),
        slice(sd0, sd0 + pd), slice(sh0, sh0 + ph), slice(sw0, sw0 + pw),
    ])
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            first_out = predictor(inputs[first_sl])
    n_cls = first_out.shape[1]

    output = torch.zeros(1, n_cls, D, H, W, dtype=torch.float32)
    count = torch.zeros(1, 1, D, H, W, dtype=torch.float32)

    output[first_sl] += first_out.cpu() * importance
    count[first_sl] += importance
    del first_out

    remaining = patch_coords[1:]
    i = 0
    while i < len(remaining):
        batch_coords = remaining[i : i + sw_batch_size]
        batch_in = torch.cat(
            [
                inputs[tuple([
                    slice(None), slice(None),
                    slice(sd, sd + pd), slice(sh, sh + ph), slice(sw_s, sw_s + pw),
                ])]
                for sd, sh, sw_s in batch_coords
            ],
            dim=0,
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                batch_out = predictor(batch_in)

        batch_out_cpu = batch_out.cpu()
        del batch_in, batch_out

        for j, (sd, sh, sw_s) in enumerate(batch_coords):
            sl = tuple([
                slice(None), slice(None),
                slice(sd, sd + pd), slice(sh, sh + ph), slice(sw_s, sw_s + pw),
            ])
            output[sl] += batch_out_cpu[j : j + 1] * importance
            count[sl] += importance

        del batch_out_cpu
        i += sw_batch_size

    output = output / count.clamp(min=1e-8)
    crop_sl = tuple([
        slice(None), slice(None),
        slice(0, d_orig), slice(0, h_orig), slice(0, w_orig),
    ])
    return output[crop_sl]


# ============================================================================
# SLIDING WINDOW VALIDATION
# ============================================================================


def validate_sliding_window(
    model, val_dicts, config, device, n_cases=None, logger=None,
):
    model.eval()
    patch_size = config["patch_size"]
    overlap = config.get("sw_overlap", 0.25)

    if n_cases and n_cases < len(val_dicts):
        indices = np.random.choice(len(val_dicts), size=n_cases, replace=False)
        cases = [val_dicts[i] for i in indices]
    else:
        cases = val_dicts

    class_tp = np.zeros(4, dtype=np.float64)
    class_fp = np.zeros(4, dtype=np.float64)
    class_fn = np.zeros(4, dtype=np.float64)
    log = logger.info if logger else print

    for d in tqdm(cases, desc="SW-Val", leave=False):
        try:
            img = nib.load(d["image"]).get_fdata().astype(np.float32)
            lbl_gt = nib.load(d["label"]).get_fdata().astype(np.int64)
            img = (np.clip(img, -175.0, 250.0) + 175.0) / 425.0
            img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

            pred_logits = _sliding_window_inference(
                img_t, roi_size=patch_size, sw_batch_size=2,
                predictor=model, overlap=overlap, mode="constant",
            )
            pred = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()
            pred = postprocess(pred, config)

            for c in range(4):
                tp = int(((pred == c) & (lbl_gt == c)).sum())
                fp = int(((pred == c) & (lbl_gt != c)).sum())
                fn = int(((pred != c) & (lbl_gt == c)).sum())
                class_tp[c] += tp
                class_fp[c] += fp
                class_fn[c] += fn

            del img_t, pred_logits
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"  SW-Val error on {d.get('case_id', '?')}: {e}")

    dice = 2 * class_tp / (2 * class_tp + class_fp + class_fn + 1e-8)
    mean_fg = float(dice[1:].mean())
    log(
        f"  SW-Val ({len(cases)} cases): "
        f"Kidney={dice[1]:.4f}  Tumor={dice[2]:.4f}  Cyst={dice[3]:.4f}  "
        f"MeanFG={mean_fg:.4f}"
    )
    return mean_fg, {
        "kidney": float(dice[1]), "tumor": float(dice[2]),
        "cyst": float(dice[3]), "mean_fg": mean_fg,
    }


# ============================================================================
# TRAINING UTILITIES
# ============================================================================


class EarlyStopping:
    def __init__(self, patience, ckpt_path):
        self.patience = patience
        self.ckpt_path = ckpt_path
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best is None or score > self.best:
            self.best = score
            self.counter = 0
            torch.save(model.state_dict(), self.ckpt_path)
            return True
        self.counter += 1
        print(f"  EarlyStopping: no improvement {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def _fmt_val(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "N/A"


def save_history(output_dir, stage, epoch, train_loss, val_loss, sw_metrics, epoch_time_s, is_best):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage, "epoch": epoch,
        "epoch_time_min": round(epoch_time_s / 60, 2),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "sw_metrics": sw_metrics, "is_best": is_best,
    }
    hist_file = Path(output_dir) / "history.json"
    hist = json.loads(hist_file.read_text()) if hist_file.exists() else {"epochs": []}
    hist["epochs"].append(entry)
    hist_file.write_text(json.dumps(hist, indent=2))


# ============================================================================
# TRAIN / QUICK-VAL PER EPOCH
# ============================================================================


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    torch.cuda.empty_cache()
    model.train()
    total_loss, count = 0.0, 0

    for batch in tqdm(loader, desc="Train", leave=False):
        try:
            img = batch["image"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = loss_fn(out, lbl)

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
            print(f"  ⚠ Train batch error: {e}")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    return total_loss / max(count, 1)


def validate_patch_loss(model, loader, loss_fn, device):
    torch.cuda.empty_cache()
    model.eval()
    total_loss, count = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="QuickVal", leave=False):
            try:
                img = batch["image"].to(device, non_blocking=True)
                lbl = batch["label"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    out = model(img)
                    loss = loss_fn(out, lbl)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    count += 1
                del img, lbl, out, loss
            except RuntimeError:
                torch.cuda.empty_cache()

    return total_loss / max(count, 1)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================


def _run_stage(
    stage_name, model, train_loader, val_loader, val_dicts, config,
    optimizer, scheduler, loss_fn, scaler, device,
    n_epochs, val_every, patience, ckpt_path, output_dir, logger,
    start_epoch=1, resume_es_best=None, resume_es_counter=0,
):
    es = EarlyStopping(patience, ckpt_path)
    if resume_es_best is not None:
        es.best = resume_es_best
    es.counter = resume_es_counter
    best_dice = resume_es_best if resume_es_best is not None else 0.0
    resume_save_path = str(Path(output_dir) / "resume_state.pth")

    for ep in range(start_epoch, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        scheduler.step()
        elapsed = time.time() - t0

        do_val = (ep % val_every == 0) or (ep == n_epochs)
        val_loss = None
        sw_metrics = None
        improved = False

        if do_val:
            val_loss = validate_patch_loss(model, val_loader, loss_fn, device)
            sw_dice, sw_metrics = validate_sliding_window(
                model, val_dicts, config, device,
                n_cases=config["sw_val_cases"], logger=logger,
            )
            improved = es(sw_dice, model)
            if improved:
                best_dice = sw_dice
                logger.info(f"  ★ New best MeanFG Dice = {best_dice:.4f}")

        logger.info(
            f"[{stage_name} E{ep:03d}/{n_epochs}] "
            f"train={train_loss:.4f}  val={_fmt_val(val_loss)}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed/60:.1f}min"
        )
        save_history(output_dir, stage_name, ep, train_loss, val_loss, sw_metrics, elapsed, improved)

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "stage": stage_name, "epoch": ep,
            "es_best": es.best, "es_counter": es.counter,
        }, resume_save_path)

        if es.early_stop:
            logger.info(f"  Early stopping at epoch {ep} (patience={patience})")
            break

    return es.best or best_dice


def _resume_from_history(output_dir, device):
    hist_file = output_dir / "history.json"
    if not hist_file.exists():
        return None
    all_eps = json.loads(hist_file.read_text()).get("epochs", [])
    if not all_eps:
        return None

    last = all_eps[-1]
    stage = last["stage"]
    epoch = last["epoch"]

    ckpt_name = "best_final.pth" if stage == "S2" else "best_stage1.pth"
    ckpt_path = output_dir / ckpt_name
    if not ckpt_path.exists():
        return None

    stage_val = [e for e in all_eps if e["stage"] == stage and e.get("sw_metrics")]
    if not stage_val:
        return None

    es_best = max(e["sw_metrics"]["mean_fg"] for e in stage_val)
    es_counter = 0
    for e in reversed(stage_val):
        if e["sw_metrics"]["mean_fg"] >= es_best - 1e-8:
            break
        es_counter += 1

    weights = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    return {
        "model_state_dict": weights,
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
        "scaler_state_dict": None,
        "stage": stage, "epoch": epoch,
        "es_best": es_best, "es_counter": es_counter,
        "_fallback": True,
    }


def train(config, resume_path=None):
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "training.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger("v9")
    logger.info("=" * 65)
    logger.info("SwinUNETR V9 — LLM-Enhanced KiTS23 Kidney Segmentation")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Output:  {output_dir}")
    if config["use_llm"]:
        logger.info(f"LLM:     {config['llm_model_name']} (layer {config['llm_layer_idx']}, "
                     f"LoRA rank {config['lora_rank']})")
    logger.info("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = SwinUNETR_LLM(
        num_classes=config["num_classes"],
        feature_size=config["feature_size"],
        use_llm=config["use_llm"],
        llm_model_name=config["llm_model_name"],
        llm_layer_idx=config["llm_layer_idx"],
        lora_rank=config["lora_rank"],
    ).to(device)

    train_loader, val_loader, _, val_dicts = get_dataloaders(config)
    loss_fn = DiceCELoss(config["num_classes"]).to(device)
    scaler = torch.amp.GradScaler("cuda")

    # ── Resume detection ─────────────────────────────────────────────────
    resume_state = None
    auto_resume = output_dir / "resume_state.pth"
    if resume_path and Path(resume_path).exists():
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        logger.info(f"  ✓ Resuming from: {resume_path}")
    elif auto_resume.exists():
        resume_state = torch.load(str(auto_resume), map_location=device, weights_only=False)
        logger.info(f"  ✓ Auto-resuming from: {auto_resume}")
    else:
        resume_state = _resume_from_history(output_dir, device)
        if resume_state:
            logger.info("  ✓ Fallback resume from history.json")

    if resume_state:
        logger.info(
            f"    Stage={resume_state['stage']}  Epoch={resume_state['epoch']}  "
            f"ES_best={resume_state['es_best']:.4f}"
        )

    in_s2 = resume_state is not None and resume_state["stage"] == "S2"

    # ── Stage 1: frozen encoder ──────────────────────────────────────────
    if not in_s2:
        logger.info("\n── Stage 1: Frozen encoder, decoder + LLM adapters ──────")
        model.freeze_encoder()

        opt1 = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["lr_stage1"], weight_decay=config["weight_decay"],
        )
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=config["num_epochs_stage1"],
            eta_min=config["lr_stage1"] * 0.1,
        )

        s1_start, s1_es_best, s1_es_counter = 1, None, 0
        if resume_state and resume_state["stage"] == "S1":
            s1_start = resume_state["epoch"] + 1
            s1_es_best = resume_state["es_best"]
            s1_es_counter = resume_state["es_counter"]
            model.load_state_dict(resume_state["model_state_dict"])
            if resume_state.get("optimizer_state_dict"):
                opt1.load_state_dict(resume_state["optimizer_state_dict"])
                sched1.load_state_dict(resume_state["scheduler_state_dict"])
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            else:
                for _ in range(resume_state["epoch"]):
                    sched1.step()
            logger.info(f"  ✓ Restored S1 — continuing from epoch {s1_start}")

        best_s1 = _run_stage(
            "S1", model, train_loader, val_loader, val_dicts, config,
            opt1, sched1, loss_fn, scaler, device,
            n_epochs=config["num_epochs_stage1"],
            val_every=config["val_every_s1"],
            patience=config["patience_stage1"],
            ckpt_path=str(output_dir / "best_stage1.pth"),
            output_dir=str(output_dir), logger=logger,
            start_epoch=s1_start, resume_es_best=s1_es_best,
            resume_es_counter=s1_es_counter,
        )

        s1_ckpt = output_dir / "best_stage1.pth"
        if s1_ckpt.exists():
            model.load_state_dict(torch.load(str(s1_ckpt), map_location=device))
            logger.info(f"  ✓ Restored Stage 1 best (MeanFG={best_s1:.4f})")
    else:
        logger.info("\n── Stage 1: Skipped (resuming Stage 2) ─────────────────")
        best_s1 = 0.0

    # ── Stage 2: full model fine-tuning ──────────────────────────────────
    logger.info("\n── Stage 2: Full model fine-tuning (LLM base stays frozen) ─")
    model.unfreeze_encoder()

    opt2 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr_stage2"], weight_decay=config["weight_decay"],
    )
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=config["num_epochs_stage2"],
        eta_min=config["lr_stage2"] * 0.01,
    )

    s2_start, s2_es_best, s2_es_counter = 1, None, 0
    if resume_state and resume_state["stage"] == "S2":
        s2_start = resume_state["epoch"] + 1
        s2_es_best = resume_state["es_best"]
        s2_es_counter = resume_state["es_counter"]
        model.load_state_dict(resume_state["model_state_dict"])
        if resume_state.get("optimizer_state_dict"):
            opt2.load_state_dict(resume_state["optimizer_state_dict"])
            sched2.load_state_dict(resume_state["scheduler_state_dict"])
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        else:
            for _ in range(resume_state["epoch"]):
                sched2.step()
        logger.info(f"  ✓ Restored S2 — continuing from epoch {s2_start}")

    best_s2 = _run_stage(
        "S2", model, train_loader, val_loader, val_dicts, config,
        opt2, sched2, loss_fn, scaler, device,
        n_epochs=config["num_epochs_stage2"],
        val_every=config["val_every_s2"],
        patience=config["patience_stage2"],
        ckpt_path=str(output_dir / "best_final.pth"),
        output_dir=str(output_dir), logger=logger,
        start_epoch=s2_start, resume_es_best=s2_es_best,
        resume_es_counter=s2_es_counter,
    )

    best_dice = max(best_s1, best_s2)
    logger.info(f"\n✓ Training complete. Best MeanFG Dice: {best_dice:.4f}")

    history = json.loads((output_dir / "history.json").read_text())
    total_min = sum(e.get("epoch_time_min", 0) for e in history["epochs"])
    summary = {
        "version": "SwinUNETR_V9_LLM",
        "llm_model": config["llm_model_name"] if config["use_llm"] else "none",
        "lora_rank": config["lora_rank"],
        "best_dice": best_dice,
        "total_hours": round(total_min / 60, 2),
        "config": {k: str(v) for k, v in config.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary → {output_dir / 'summary.json'}")


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate(config, checkpoint):
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)], force=True,
    )
    logger = logging.getLogger("eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR_LLM(
        num_classes=config["num_classes"],
        feature_size=config["feature_size"],
        use_llm=config["use_llm"],
        llm_model_name=config["llm_model_name"],
        llm_layer_idx=config["llm_layer_idx"],
        lora_rank=config["lora_rank"],
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    logger.info(f"✓ Loaded: {checkpoint}")

    data_dir = Path(config["kits23_dir"])
    test_cases = sorted(
        [c for c in data_dir.iterdir() if c.is_dir() and c.name.startswith("case_")]
    )[-config["test_cases"]:]

    model.eval()
    class_tp = np.zeros(4, dtype=np.float64)
    class_fp = np.zeros(4, dtype=np.float64)
    class_fn = np.zeros(4, dtype=np.float64)
    per_case = []

    logger.info(f"\nEvaluating {len(test_cases)} test cases ...")

    with torch.no_grad():
        for case_dir in test_cases:
            img_path = case_dir / "imaging.nii.gz"
            lbl_path = case_dir / "segmentation.nii.gz"
            if not img_path.exists() or not lbl_path.exists():
                continue

            t0 = time.time()
            img = nib.load(str(img_path)).get_fdata().astype(np.float32)
            lbl_gt = nib.load(str(lbl_path)).get_fdata().astype(np.int64)
            img = (np.clip(img, -175.0, 250.0) + 175.0) / 425.0

            img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
            pred_logits = _sliding_window_inference(
                img_t, roi_size=config["patch_size"], sw_batch_size=2,
                predictor=model, overlap=0.5, mode="gaussian",
            )
            pred = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()
            pred = postprocess(pred, config)

            dice_pc = []
            for c in range(4):
                tp = int(((pred == c) & (lbl_gt == c)).sum())
                fp = int(((pred == c) & (lbl_gt != c)).sum())
                fn = int(((pred != c) & (lbl_gt == c)).sum())
                dice_pc.append(2 * tp / (2 * tp + fp + fn + 1e-8))
                class_tp[c] += tp
                class_fp[c] += fp
                class_fn[c] += fn

            elapsed = time.time() - t0
            per_case.append({"case_id": case_dir.name, "dice": [float(x) for x in dice_pc]})
            logger.info(
                f"{case_dir.name} ... K={dice_pc[1]:.3f} T={dice_pc[2]:.3f} "
                f"C={dice_pc[3]:.3f} ({elapsed:.0f}s)"
            )
            del img_t, pred_logits

    dice = 2 * class_tp / (2 * class_tp + class_fp + class_fn + 1e-8)
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY (V9)")
    logger.info("=" * 60)
    logger.info(f"Cases:  {len(per_case)}")
    logger.info(f"Kidney  Dice: {dice[1]:.4f}")
    logger.info(f"Tumor   Dice: {dice[2]:.4f}")
    logger.info(f"Cyst    Dice: {dice[3]:.4f}")
    logger.info(f"MeanFG  Dice: {dice[1:].mean():.4f}")
    logger.info("=" * 60)

    out = {
        "global_dice": {
            "kidney": float(dice[1]), "tumor": float(dice[2]),
            "cyst": float(dice[3]), "mean_fg": float(dice[1:].mean()),
        },
        "per_case": per_case,
    }
    out_path = Path(config["output_dir"]) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Results → {out_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="SwinUNETR V9 — LLM-Enhanced KiTS23")
    parser.add_argument("--mode", choices=["full", "medium", "showcase", "quick_test", "evaluate"], default="full")
    parser.add_argument(
        "--quick-test-baseline",
        action="store_true",
        help="Shortcut for --mode quick_test --no-llm with default output ./output/v9_quick_baseline",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--kits23-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patch-size", default=None, help="D,H,W e.g. 128,128,128")
    parser.add_argument("--sw-overlap", type=float, default=None)
    parser.add_argument("--llm-model", default=None, help="HuggingFace model name for LLM block")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM block (baseline)")
    parser.add_argument("--resume", default=None, metavar="PATH")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    selected_mode = "quick_test" if args.quick_test_baseline else args.mode
    cfg = get_config(selected_mode if selected_mode not in ("evaluate",) else "full")
    if args.kits23_dir:
        cfg["kits23_dir"] = args.kits23_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    elif args.quick_test_baseline:
        cfg["output_dir"] = "./output/v9_quick_baseline"
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.patch_size:
        cfg["patch_size"] = tuple(int(x) for x in args.patch_size.split(","))
    if args.sw_overlap is not None:
        cfg["sw_overlap"] = args.sw_overlap
    if args.llm_model:
        cfg["llm_model_name"] = args.llm_model
    if args.lora_rank is not None:
        cfg["lora_rank"] = args.lora_rank
    if args.no_llm or args.quick_test_baseline:
        cfg["use_llm"] = False

    if selected_mode == "evaluate":
        if not args.checkpoint:
            parser.error("--checkpoint required for evaluate mode")
        evaluate(cfg, args.checkpoint)
    else:
        if args.no_resume:
            resume_f = Path(cfg["output_dir"]) / "resume_state.pth"
            if resume_f.exists():
                resume_f.unlink()
        train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
