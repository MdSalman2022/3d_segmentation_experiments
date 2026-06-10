"""
SwinUNETR V10 -- LLM-Enhanced KiTS23 Kidney Segmentation  [PRODUCTION READY]
=====================================================================
Based on MedVisionLlama (ICCV 2025 Workshop, Kumar et al.)

Speed and Accuracy Optimizations in V10:
  [1]  Auto-download Pretrained Weights -- downloads model_swinvit.pt from MONAI zoo if missing.
  [2]  Isotropic Spacing Normalization -- resamples all volumes to 1.5mm isotropic in build_cache.
  [3]  Stable MONAI DiceCELoss -- batch=True to prevent instability of absent classes.
  [4]  Full Fine-tuning in Stage 2 -- unfreezes SwinViT encoder fully by default for maximum accuracy.
  [5]  BF16 AMP -- stable dynamic range on high-end GPUs.
  [6]  Disk Cache -- loads mmap'd float16 resampled .npy files.
  [7]  Gradient accumulation -- effective batch = batch_size * grad_accum_steps.
"""

import copy
import csv
import os
import gc
import json
import time
import argparse
import sys
import logging
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Suppress TF and CUDA deprecation noise before any imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"
os.environ["TF_CPP_MIN_LOG_LEVEL"]     = "3"
os.environ["CUDA_MODULE_LOADING"]      = "LAZY"

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from tqdm import tqdm

CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]
_DTYPE      = torch.bfloat16


# ============================================================================
# DOWNLOAD PROGRESS
# ============================================================================

def _download_progress(block_num, block_size, total_size):
    """Callback hook to print download progress of large weight files."""
    if total_size > 0:
        percent = (block_num * block_size * 100.0) / total_size
        percent = min(100.0, percent)
        step = int(percent / 10)
        if not hasattr(_download_progress, 'last_step') or _download_progress.last_step != step:
            _download_progress.last_step = step
            logger = logging.getLogger("v10")
            logger.info(f"    -> Download progress: {percent:.1f}% "
                        f"({(block_num * block_size) / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")


# ============================================================================
# CONFIGURATION
# ============================================================================


def get_config(mode: str) -> dict:
    n_cpu = min(multiprocessing.cpu_count(), 12)

    base = {
        "kits23_dir":             "./kits23/dataset",
        "output_dir":             "./output/swin_v10",
        "num_classes":            4,
        "feature_size":           48,
        # LLM bottleneck
        "use_llm":                True,
        "llm_model_name":         "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "llm_layer_idx":          -1,
        "lora_rank":              4,    # LoRA rank for LLM block
        # Stage 2 tuning strategy
        "lora_s2":                False, # Default: fully unfreeze encoder (SwinViT) in S2 for best accuracy
        "lora_s2_rank":           4,     # rank if lora_s2 is enabled
        # Data
        "patch_size":             (128, 128, 128),
        "batch_size":             2,
        "grad_accum_steps":       2,    # effective batch = batch_size * grad_accum_steps
        "num_samples_per_volume": 2,
        "val_split":              0.20,
        "val_cases":              40,   # fixed val count (overrides val_split when set)
        "test_cases":             40,
        "seed":                   42,
        # DataLoader
        "num_workers":            n_cpu,
        "prefetch_factor":        4,
        # Stage 1
        "num_epochs_stage1":      50,
        "lr_stage1":              1e-3,
        "warmup_epochs_s1":       3,
        "patience_stage1":        15,
        "val_every_s1":           5,
        # Stage 2
        "num_epochs_stage2":      250,
        "lr_stage2":              2e-4,
        "warmup_epochs_s2":       5,
        "patience_stage2":        30,
        "val_every_s2":           5,
        "weight_decay":           1e-5,
        # Sliding window
        "sw_val_cases":           20,
        "sw_overlap":             0.25,
        "sw_batch_size":          4,
        # Flags
        "use_checkpoint":         True,
        "use_compile":            False,
        # Post-processing
        "min_kidney_voxels":      5000,
        "min_tumor_voxels":       50,
        "min_cyst_voxels":        50,
    }

    if mode == "quick_test":
        base.update({
            "num_epochs_stage1":  2,
            "num_epochs_stage2":  2,
            "warmup_epochs_s1":   1,
            "warmup_epochs_s2":   1,
            "patience_stage1":    3,
            "patience_stage2":    3,
            "val_every_s1":       5,
            "val_every_s2":       5,
            "sw_val_cases":       3,
            "sw_overlap":         0.0,
            "quick_cases":        30,
            "val_cases":          5,
            "output_dir":         "./output/swin_v10_quick",
            "use_compile":        False,
            "grad_accum_steps":   1,
        })
    elif mode == "showcase":
        base.update({
            "num_epochs_stage1":       15,
            "num_epochs_stage2":       60,
            "warmup_epochs_s1":        2,
            "warmup_epochs_s2":        3,
            "patience_stage1":         6,
            "patience_stage2":         15,
            "val_every_s1":            5,
            "val_every_s2":            5,
            "sw_val_cases":            8,
            "sw_overlap":              0.0,
            "num_samples_per_volume":  4,
            "quick_cases":             150,
            "output_dir":              "./output/swin_v10_showcase",
            "grad_accum_steps":        2,
        })
    elif mode == "medium":
        base.update({
            "num_epochs_stage1":  30,
            "num_epochs_stage2":  80,
            "warmup_epochs_s1":   4,
            "warmup_epochs_s2":   6,
            "patience_stage1":    10,
            "patience_stage2":    20,
            "val_every_s1":       5,
            "val_every_s2":       5,
            "sw_val_cases":       12,
            "sw_overlap":         0.25,
            "quick_cases":        None,
            "output_dir":         "./output/swin_v10_medium",
            "grad_accum_steps":   2,
        })
    else:
        base["quick_cases"] = None

    return base


# ============================================================================
# DISK CACHE
# ============================================================================


def _cache_dir_for(kits23_dir: str) -> Path:
    return Path(kits23_dir).parent / "kits23_cache_bf16_isotropic_1.5"


def build_cache(kits23_dir: str, cache_dir: Path):
    """
    One-time preprocessing: resample to 1.5mm isotropic spacing -> clip -> norm -> save .npy.
    Subsequent epochs load via mmap -- near-zero I/O cost.
    """
    logger = logging.getLogger("v10")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted([c for c in Path(kits23_dir).iterdir()
                    if c.is_dir() and c.name.startswith("case_")])
    logger.info(f"Building cache (1.5mm isotropic) for {len(cases)} cases -> {cache_dir}")
    from scipy.ndimage import zoom
    skipped = 0
    for case in tqdm(cases, desc="Caching"):
        img_c = cache_dir / f"{case.name}_img.npy"
        lbl_c = cache_dir / f"{case.name}_lbl.npy"
        if img_c.exists() and lbl_c.exists():
            skipped += 1; continue
        img_path = case / "imaging.nii.gz"
        lbl_path = case / "segmentation.nii.gz"
        if not img_path.exists() or not lbl_path.exists():
            logger.warning(f"  [Cache] Missing imaging or label file for {case.name}, skipping.")
            continue
        
        try:
            img_nii = nib.load(str(img_path))
            lbl_nii = nib.load(str(lbl_path))
            
            img = img_nii.get_fdata().astype(np.float32)
            lbl = lbl_nii.get_fdata().astype(np.uint8)
            
            # Spacing Resampling
            zooms = img_nii.header.get_zooms()
            target_spacing = (1.5, 1.5, 1.5)
            zoom_factors = [z / t for z, t in zip(zooms, target_spacing)]
            
            img_res = zoom(img, zoom_factors, order=1)
            lbl_res = zoom(lbl, zoom_factors, order=0)
            
            # Clip & Normalize
            img_res = (np.clip(img_res, -175.0, 250.0) + 175.0) / 425.0
            
            np.save(str(img_c), img_res.astype(np.float16))
            np.save(str(lbl_c), lbl_res)
        except Exception as e:
            logger.error(f"  [Cache] Failed to process {case.name}: {e}", exc_info=True)
            
    logger.info(f"  Cache ready: {cache_dir}  ({skipped} already cached)")


def _load_volume(d: dict, cache_dir: Optional[Path]):
    """Load from disk cache (fast, mmap'd) or fall back to .nii.gz with online resampling."""
    case_id = d.get("case_id", "")
    if cache_dir:
        img_c = cache_dir / f"{case_id}_img.npy"
        lbl_c = cache_dir / f"{case_id}_lbl.npy"
        if img_c.exists() and lbl_c.exists():
            img = np.array(np.load(str(img_c), mmap_mode="r"), dtype=np.float32)
            lbl = np.array(np.load(str(lbl_c), mmap_mode="r"), dtype=np.int64)
            return img, lbl
            
    # Fallback to online loading and resampling
    img_nii = nib.load(d["image"])
    lbl_nii = nib.load(d["label"])
    img = img_nii.get_fdata().astype(np.float32)
    lbl = lbl_nii.get_fdata().astype(np.int64)
    
    zooms = img_nii.header.get_zooms()
    target_spacing = (1.5, 1.5, 1.5)
    zoom_factors = [z / t for z, t in zip(zooms, target_spacing)]
    
    from scipy.ndimage import zoom
    img = zoom(img, zoom_factors, order=1)
    lbl = zoom(lbl, zoom_factors, order=0)
    
    img = (np.clip(img, -175.0, 250.0) + 175.0) / 425.0
    return img, lbl


# ============================================================================
# LORA COMPONENTS
# ============================================================================


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation of a frozen linear layer.
    Base weights are frozen; only lora_A and lora_B train.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 4):
        super().__init__()
        self.base    = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f  = base_linear.in_features
        out_f = base_linear.out_features
        self.lora_A  = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B  = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A)
        self.scaling = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.linear(x, self.base.weight, self.base.bias)
                + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling)


def apply_lora_to_linears(
    module: nn.Module,
    rank: int,
    target_names: Tuple[str, ...] = ("qkv", "proj", "fc1", "fc2", "linear1", "linear2"),
):
    """Recursively wrap Linear layers whose attribute name contains target_name with LoRALinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, LoRALinear(child, rank=rank))
        else:
            apply_lora_to_linears(child, rank, target_names)


def count_lora_params(module: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# LLM COMPONENTS
# ============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class LLMTransformerBlock(nn.Module):
    """Llama-style transformer block for 3D vision features."""
    def __init__(self, hidden_dim, num_heads, num_kv_heads, ffn_dim):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = hidden_dim // num_heads
        self.kv_dim       = num_kv_heads * self.head_dim
        self.norm1     = RMSNorm(hidden_dim)
        self.norm2     = RMSNorm(hidden_dim)
        self.q_proj    = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj    = nn.Linear(hidden_dim, self.kv_dim, bias=False)
        self.v_proj    = nn.Linear(hidden_dim, self.kv_dim, bias=False)
        self.o_proj    = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj  = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def _repeat_kv(self, x):
        if self.num_kv_heads == self.num_heads: return x
        n_rep = self.num_heads // self.num_kv_heads
        B, H, N, D = x.shape
        return x.unsqueeze(2).expand(B, H, n_rep, N, D).reshape(B, H * n_rep, N, D)

    def forward(self, x):
        h = self.norm1(x); B, N, _ = h.shape
        q = self.q_proj(h).view(B, N, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = self._repeat_kv(k); v = self._repeat_kv(v)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.o_proj(attn.transpose(1, 2).contiguous().view(B, N, -1))
        h = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class LLMBottleneckBlock(nn.Module):
    def __init__(self, vision_dim, llm_model_name, llm_layer_idx=-1, lora_rank=4):
        super().__init__()
        self.vision_dim = vision_dim
        logger = logging.getLogger("v10")
        cfg     = self._get_llm_config(llm_model_name)
        llm_dim = cfg["hidden_dim"]
        self.llm_dim = llm_dim
        self.llm_block = LLMTransformerBlock(
            hidden_dim=llm_dim, num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"], ffn_dim=cfg["ffn_dim"])
        self._load_llm_weights(llm_model_name, llm_layer_idx)
        for p in self.llm_block.parameters():
            p.requires_grad = False
        self._apply_lora(lora_rank)
        self.proj_in  = nn.Linear(vision_dim, llm_dim)
        self.proj_out = nn.Linear(llm_dim, vision_dim)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)
        self.norm = nn.LayerNorm(vision_dim)
        total, trainable = count_lora_params(self)
        logger.info(f"  LLM block: {total:,} total | {trainable:,} trainable | {total-trainable:,} frozen")

    def _get_llm_config(self, model_name):
        logger = logging.getLogger("v10")
        try:
            from transformers import AutoConfig
            c = AutoConfig.from_pretrained(model_name)
            cfg = {"hidden_dim": c.hidden_size, "num_heads": c.num_attention_heads,
                   "num_kv_heads": getattr(c, "num_key_value_heads", c.num_attention_heads),
                   "ffn_dim": c.intermediate_size, "num_layers": c.num_hidden_layers}
            logger.info(f"  LLM config: dim={cfg['hidden_dim']}, heads={cfg['num_heads']}/{cfg['num_kv_heads']}, ffn={cfg['ffn_dim']}")
            return cfg
        except Exception as e:
            logger.warning(f"  LLM config fallback: TinyLlama defaults ({e})")
            return {"hidden_dim": 2048, "num_heads": 32, "num_kv_heads": 4, "ffn_dim": 5632, "num_layers": 22}

    def _load_llm_weights(self, model_name, layer_idx):
        logger = logging.getLogger("v10")
        cache_dir  = Path.home() / ".cache" / "llm_layers"
        cache_path = cache_dir / f"{model_name.replace('/','_')}_layer{layer_idx}.pth"
        if cache_path.exists():
            try:
                state = torch.load(str(cache_path), map_location="cpu", weights_only=True)
                logger.info(f"  [LLM] Loading weights from local cache: {cache_path}")
                mapped_count = self._map_weights(state)
                logger.info(f"  [LLM] ✓ Loaded {mapped_count} LLM weights from cache")
                return
            except Exception as e:
                logger.warning(f"  [LLM] Cache load failed ({e}). Re-fetching from huggingface...")
        try:
            logger.info(f"  [LLM] Fetching model '{model_name}' from Hugging Face for layer {layer_idx}...")
            from transformers import AutoModelForCausalLM
            llm    = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            layers = llm.model.layers
            idx    = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
            state  = {k: v.clone() for k, v in layers[idx].state_dict().items()}
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, str(cache_path))
            mapped_count = self._map_weights(state)
            logger.info(f"  [LLM] ✓ Loaded {mapped_count} LLM weights (layer saved to cache at {cache_path})")
            del llm, layers, state; gc.collect()
        except Exception as e:
            logger.error(f"  [LLM] ⚠ Hugging Face weight load failed ({e}) -- using random initialization", exc_info=True)

    def _map_weights(self, layer_state):
        key_map = {
            "norm1.weight": "input_layernorm.weight", "norm2.weight": "post_attention_layernorm.weight",
            "q_proj.weight": "self_attn.q_proj.weight", "k_proj.weight": "self_attn.k_proj.weight",
            "v_proj.weight": "self_attn.v_proj.weight", "o_proj.weight": "self_attn.o_proj.weight",
            "gate_proj.weight": "mlp.gate_proj.weight", "up_proj.weight": "mlp.up_proj.weight",
            "down_proj.weight": "mlp.down_proj.weight",
        }
        our_sd = self.llm_block.state_dict()
        mapped = {ok: layer_state[hk] for ok, hk in key_map.items()
                  if hk in layer_state and ok in our_sd and our_sd[ok].shape == layer_state[hk].shape}
        self.llm_block.load_state_dict(mapped, strict=False)
        return len(mapped)

    def _apply_lora(self, rank):
        logger = logging.getLogger("v10")
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(self.llm_block, name, LoRALinear(getattr(self.llm_block, name), rank))
        lp = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  LLM LoRA rank={rank}: {lp:,} trainable adapter params")

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_seq = x.flatten(2).permute(0, 2, 1)
        h = self.proj_out(self.llm_block(self.proj_in(x_seq)))
        return self.norm(x_seq + h).permute(0, 2, 1).view(B, C, D, H, W)


# ============================================================================
# MODEL
# ============================================================================


class SwinUNETR_LLM(nn.Module):
    """MONAI SwinUNETR + frozen LLM bottleneck block."""
    def __init__(self, num_classes=4, feature_size=48, use_llm=True,
                 llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 llm_layer_idx=-1, lora_rank=4, use_checkpoint=True):
        super().__init__()
        logger = logging.getLogger("v10")
        from monai.networks.nets import SwinUNETR
        self.net = SwinUNETR(
            in_channels=1, out_channels=num_classes,
            feature_size=feature_size, use_checkpoint=use_checkpoint, spatial_dims=3)
        self._load_pretrained_encoder()
        self.use_llm = use_llm
        if use_llm:
            self.llm_block = LLMBottleneckBlock(
                vision_dim=feature_size * 16, llm_model_name=llm_model_name,
                llm_layer_idx=llm_layer_idx, lora_rank=lora_rank)
        else:
            self.llm_block = None
            logger.info("  LLM block disabled -- pure SwinUNETR baseline")
        total, trainable = count_lora_params(self)
        logger.info(f"\n{'='*60}\nSwinUNETR V10 {'+ LLM' if use_llm else '(baseline)'}")
        logger.info(f"Total: {total:,} | Trainable: {trainable:,}\n{'='*60}\n")

    def _load_pretrained_encoder(self):
        logger = logging.getLogger("v10")
        target_dir = Path.home() / ".torch" / "models"
        target_dir.mkdir(parents=True, exist_ok=True)
        wpath = target_dir / "model_swinvit.pt"
        local_path = Path("./swin_unetr_pretrained.pth")
        
        if local_path.exists():
            wpath = local_path
            logger.info(f"  [Init] Found local pretrained weights path at: {wpath}")

        # Download automatically if missing
        if not wpath.exists():
            logger.info(f"  [Init] Pretrained encoder weights not found at: {wpath}")
            logger.info("  [Init] Downloading model_swinvit.pt from MONAI GitHub release (approx 411MB)...")
            url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
            try:
                import urllib.request
                _download_progress.last_step = -1
                urllib.request.urlretrieve(url, str(wpath), reporthook=_download_progress)
                logger.info(f"  [Init] ✓ Download completed successfully. Saved to: {wpath}")
            except Exception as e:
                logger.error(f"  [Init] ⚠ Failed to download pretrained weights: {e}", exc_info=True)
                logger.warning("  [Init] Model will train from scratch without pretrained weights.")
                return
        try:
            logger.info(f"  [Init] Loading weights from checkpoint file: {wpath}")
            ckpt = torch.load(str(wpath), map_location="cpu", weights_only=False)
            weights = ckpt.get("state_dict", ckpt)
            model_dict = self.net.state_dict()
            matched = {}
            mismatched_keys = []
            
            for k, v in weights.items():
                mk = k.removeprefix("module.")
                if mk in model_dict:
                    if model_dict[mk].shape == v.shape:
                        matched[mk] = v
                    else:
                        mismatched_keys.append((mk, f"Shape mismatch: expected {model_dict[mk].shape}, got {v.shape}"))
                else:
                    mk2 = "swinViT." + mk
                    if mk2 in model_dict:
                        if model_dict[mk2].shape == v.shape:
                            matched[mk2] = v
                        else:
                            mismatched_keys.append((mk2, f"Shape mismatch: expected {model_dict[mk2].shape}, got {v.shape}"))
            
            if mismatched_keys:
                logger.warning(f"  [Init] ⚠ {len(mismatched_keys)} keys had shape mismatches and were skipped:")
                for mk, reason in mismatched_keys[:5]:
                    logger.warning(f"    - {mk}: {reason}")
                if len(mismatched_keys) > 5:
                    logger.warning(f"    - ... and {len(mismatched_keys) - 5} more mismatched keys.")

            model_dict.update(matched)
            self.net.load_state_dict(model_dict, strict=False)
            logger.info(f"  [Init] ✓ Successfully loaded {len(matched)}/{len(model_dict)} pretrained SwinViT encoder weights.")
            
            # Check for keys in model that were NOT loaded
            missing_in_weights = [k for k in model_dict.keys() if k not in matched and k.startswith("swinViT")]
            if missing_in_weights:
                logger.info(f"  [Init] Note: {len(missing_in_weights)} swinViT encoder keys were not in pretrained file (normal for newly initialized layers):")
                logger.info(f"    - Missing keys snippet: {missing_in_weights[:5]}")
        except Exception as e:
            logger.error(f"  [Init] ⚠ Could not load pretrained weights: {e}", exc_info=True)

    def freeze_encoder(self):
        """Stage 1: freeze entire SwinViT encoder."""
        logger = logging.getLogger("v10")
        for p in self.net.swinViT.parameters():
            p.requires_grad = False
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        f = sum(p.numel() for p in self.net.swinViT.parameters())
        logger.info(f"  Encoder frozen ({f:,}) | Trainable ({t:,})")

    def unfreeze_encoder_with_lora(self, lora_rank: int):
        """Stage 2: keep SwinViT BASE frozen, apply LoRA adapters on attention/FFN."""
        logger = logging.getLogger("v10")
        for p in self.net.swinViT.parameters():
            p.requires_grad = False
        apply_lora_to_linears(
            self.net.swinViT, rank=lora_rank,
            target_names=("qkv", "proj", "fc1", "fc2"),
        )
        total, trainable = count_lora_params(self)
        swin_total, swin_train = count_lora_params(self.net.swinViT)
        logger.info(f"  SwinViT LoRA applied (rank={lora_rank}): "
                    f"{swin_train:,} encoder trainable / {trainable:,} total trainable")

    def unfreeze_encoder_full(self):
        """Stage 2: fully unfreeze SwinViT encoder for maximum accuracy."""
        logger = logging.getLogger("v10")
        for p in self.net.swinViT.parameters():
            p.requires_grad = True
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  Encoder fully unfrozen | Trainable: {t:,}")

    def forward(self, x):
        hs   = self.net.swinViT(x, self.net.normalize)
        enc0 = self.net.encoder1(x)
        enc1 = self.net.encoder2(hs[0])
        enc2 = self.net.encoder3(hs[1])
        enc3 = self.net.encoder4(hs[2])
        dec4 = self.net.encoder10(hs[4])
        if self.use_llm and self.llm_block is not None:
            dec4 = self.llm_block(dec4)
        dec3 = self.net.decoder5(dec4, hs[3])
        dec2 = self.net.decoder4(dec3, enc3)
        dec1 = self.net.decoder3(dec2, enc2)
        dec0 = self.net.decoder2(dec1, enc1)
        out  = self.net.decoder1(dec0, enc0)
        return self.net.out(out)


# ============================================================================
# LOSS
# ============================================================================


class DiceCELoss(nn.Module):
    """Stable Wrapper around MONAI's robust DiceCELoss with batch-level division."""
    def __init__(self, num_classes=4, smooth=1e-6):
        super().__init__()
        from monai.losses import DiceCELoss as MONAIDiceCELoss
        self.monai_loss = MONAIDiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            batch=True,
            weight=torch.tensor([0.1, 1.0, 8.0, 4.0])
        )

    def forward(self, logits, target, sample_weight=1.0):
        # Dynamically map loss parameters to the device/dtype of the logits
        weight = getattr(self.monai_loss, "cross_entropy", None)
        if weight is not None:
            weight = weight.weight
        if weight is not None and (weight.device != logits.device or weight.dtype != logits.dtype):
            self.monai_loss.cross_entropy.weight = weight.to(device=logits.device, dtype=logits.dtype)
            
        loss = self.monai_loss(logits, target)
        return loss * sample_weight


# ============================================================================
# METRICS
# ============================================================================


def compute_seg_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> dict:
    dice_l, iou_l, prec_l, rec_l = [], [], [], []
    for c in range(num_classes):
        p = pred == c; g = gt == c
        tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
        dice_l.append(2*tp / (2*tp + fp + fn + 1e-8))
        iou_l.append(    tp / (tp + fp + fn + 1e-8))
        prec_l.append(   tp / (tp + fp + 1e-8))
        rec_l.append(    tp / (tp + fn + 1e-8))
    return {"dice": dice_l, "iou": iou_l, "precision": prec_l, "recall": rec_l}


def compute_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    if not pred_mask.any() or not gt_mask.any():
        return float("nan")
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
        pb = pred_mask ^ binary_erosion(pred_mask)
        gb = gt_mask   ^ binary_erosion(gt_mask)
        all_d = np.concatenate([distance_transform_edt(~gt_mask)[pb],
                                 distance_transform_edt(~pred_mask)[gb]])
        return float(np.percentile(all_d, 95))
    except Exception:
        return float("nan")


def aggregate_metrics(per_case_metrics: list, num_classes: int = 4) -> dict:
    tp = np.zeros(num_classes); fp = np.zeros(num_classes); fn = np.zeros(num_classes)
    for m in per_case_metrics:
        rc = m.get("raw_counts", {})
        if rc:
            tp += np.array(rc["tp"]); fp += np.array(rc["fp"]); fn += np.array(rc["fn"])
    eps  = 1e-8
    dice = (2*tp / (2*tp + fp + fn + eps)).tolist()
    iou  = (tp / (tp + fp + fn + eps)).tolist()
    prec = (tp / (tp + fp + eps)).tolist()
    rec  = (tp / (tp + fn + eps)).tolist()
    hd95_arr  = np.array([m.get("hd95", [float("nan")]*num_classes) for m in per_case_metrics])
    hd95_mean = np.nanmean(hd95_arr, axis=0).tolist()
    return {
        "dice": dice, "iou": iou, "precision": prec, "recall": rec, "hd95_mean": hd95_mean,
        "mean_fg_dice":      float(np.mean(dice[1:])),
        "mean_fg_iou":       float(np.mean(iou[1:])),
        "mean_fg_precision": float(np.mean(prec[1:])),
        "mean_fg_recall":    float(np.mean(rec[1:])),
        "mean_fg_hd95":      float(np.nanmean(hd95_mean[1:])),
    }


# ============================================================================
# AUGMENTATION
# ============================================================================


def augment_patch(img, lbl):
    for ax in range(3):
        if np.random.random() < 0.5: img = np.flip(img, axis=ax); lbl = np.flip(lbl, axis=ax)
    if np.random.random() < 0.40:
        k = np.random.randint(1, 4); axes = tuple(np.random.choice(3, 2, replace=False).tolist())
        img = np.rot90(img, k=k, axes=axes); lbl = np.rot90(lbl, k=k, axes=axes)
    if np.random.random() < 0.15:
        scale = float(np.random.uniform(0.88, 1.12)); pd, ph, pw = img.shape
        img_s = ndimage.zoom(img.astype(np.float32), scale, order=1)
        lbl_s = ndimage.zoom(lbl.astype(np.float32), scale, order=0)
        img_o = np.zeros((pd,ph,pw), np.float32); lbl_o = np.zeros((pd,ph,pw), lbl.dtype)
        sd,sh,sw = img_s.shape
        s0=max(0,(sd-pd)//2); s1=max(0,(sh-ph)//2); s2=max(0,(sw-pw)//2)
        d0=max(0,(pd-sd)//2); d1=max(0,(ph-sh)//2); d2=max(0,(pw-sw)//2)
        c0=min(sd-s0,pd-d0);  c1=min(sh-s1,ph-d1);  c2=min(sw-s2,pw-d2)
        if c0>0 and c1>0 and c2>0:
            img_o[d0:d0+c0,d1:d1+c1,d2:d2+c2]=img_s[s0:s0+c0,s1:s1+c1,s2:s2+c2]
            lbl_o[d0:d0+c0,d1:d1+c1,d2:d2+c2]=lbl_s[s0:s0+c0,s1:s1+c1,s2:s2+c2]
            img, lbl = img_o, lbl_o
    if np.random.random() < 0.25:
        img = np.clip(img + np.random.randn(*img.shape).astype(np.float32) * float(np.random.uniform(0.01,0.08)), 0., 1.)
    if np.random.random() < 0.20: img = np.clip(img + float(np.random.uniform(-0.1,0.1)), 0., 1.)
    if np.random.random() < 0.20: img = np.power(np.clip(img, 0., 1.), float(np.random.uniform(0.7, 1.5)))
    if np.random.random() < 0.15: img = ndimage.gaussian_filter(img.astype(np.float32), float(np.random.uniform(0.5,1.5)))
    return img.copy(), lbl.copy()


# ============================================================================
# DATASET
# ============================================================================


_SAMPLE_BIAS = {"tumor": 0.60, "cyst": 0.10, "kidney": 0.20, "random": 0.10}


class KiTS23Dataset(Dataset):
    def __init__(self, data_dicts, patch_size, num_samples=2, is_train=True, cache_dir=None):
        self.patch_size = patch_size; self.is_train = is_train; self.cache_dir = cache_dir
        self.items = [d for d in data_dicts for _ in range(num_samples)]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]; img, lbl = _load_volume(d, self.cache_dir)
        if self.is_train:
            img, lbl = self._sample_patch_train(img, lbl); img, lbl = augment_patch(img, lbl)
        else:
            img, lbl = self._center_crop(img, lbl)
        return {"image": torch.from_numpy(img).float().unsqueeze(0),
                "label": torch.from_numpy(lbl.copy()).long().unsqueeze(0),
                "case_id": d.get("case_id", "unknown")}

    def _pad_if_needed(self, img, lbl):
        pd,ph,pw=self.patch_size; d,h,w=img.shape
        if d<pd or h<ph or w<pw:
            pad=((0,max(0,pd-d)),(0,max(0,ph-h)),(0,max(0,pw-w)))
            img=np.pad(img,pad,mode="constant",constant_values=0.)
            lbl=np.pad(lbl,pad,mode="constant",constant_values=0)
        return img, lbl

    def _sample_patch_train(self, img, lbl):
        img,lbl=self._pad_if_needed(img,lbl); d,h,w=img.shape; pd,ph,pw=self.patch_size
        r=np.random.random(); b=_SAMPLE_BIAS
        if   r < b["tumor"]:                          tc=2
        elif r < b["tumor"]+b["cyst"]:               tc=3
        elif r < b["tumor"]+b["cyst"]+b["kidney"]:   tc=1
        else:                                          tc=None
        ds,hs,ws=self._find_center(lbl,d,h,w,pd,ph,pw,tc)
        return img[ds:ds+pd,hs:hs+ph,ws:ws+pw].copy(), lbl[ds:ds+pd,hs:hs+ph,ws:ws+pw].copy()

    def _center_crop(self, img, lbl):
        img,lbl=self._pad_if_needed(img,lbl); d,h,w=img.shape; pd,ph,pw=self.patch_size
        return (img[(d-pd)//2:(d-pd)//2+pd,(h-ph)//2:(h-ph)//2+ph,(w-pw)//2:(w-pw)//2+pw],
                lbl[(d-pd)//2:(d-pd)//2+pd,(h-ph)//2:(h-ph)//2+ph,(w-pw)//2:(w-pw)//2+pw])

    @staticmethod
    def _find_center(lbl,d,h,w,pd,ph,pw,tc):
        if tc is not None:
            idx=np.argwhere(lbl==tc)
            if len(idx)==0: idx=np.argwhere(lbl>0)
            if len(idx)>0:
                c=idx[np.random.randint(len(idx))]
                return (int(np.clip(c[0]-pd//2,0,d-pd)),
                        int(np.clip(c[1]-ph//2,0,h-ph)),
                        int(np.clip(c[2]-pw//2,0,w-pw)))
        return (np.random.randint(0,max(1,d-pd+1)),
                np.random.randint(0,max(1,h-ph+1)),
                np.random.randint(0,max(1,w-pw+1)))


def get_dataloaders(config):
    logger = logging.getLogger("v10")
    data_dir  = Path(config["kits23_dir"])
    cache_dir = _cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists():
        logger.warning(f"  *** Cache not found at {cache_dir}.")
        logger.warning(f"  *** Run: python swinunetr_v10.py --mode build_cache")
        logger.warning(f"  *** Training will be 2-3x slower without cache.")
        cache_dir = None
    all_cases = sorted([c for c in data_dir.iterdir()
                        if c.is_dir() and c.name.startswith("case_")])
    all_cases = all_cases[:-config.get("test_cases", 40)]
    if config.get("quick_cases"):
        all_cases = all_cases[:config["quick_cases"]]
    data_dicts = [{"image":str(c/"imaging.nii.gz"),"label":str(c/"segmentation.nii.gz"),"case_id":c.name}
                  for c in all_cases if (c/"imaging.nii.gz").exists() and (c/"segmentation.nii.gz").exists()]
    n_val       = config["val_cases"] if config.get("val_cases") else int(len(data_dicts) * config["val_split"])
    val_dicts   = data_dicts[:n_val]
    train_dicts = data_dicts[n_val:]
    logger.info(f"  Split: {len(train_dicts)} train | {len(val_dicts)} val | {config.get('test_cases',40)} test")
    nw=config.get("num_workers",8); pf=config.get("prefetch_factor",4)
    train_loader = DataLoader(
        KiTS23Dataset(train_dicts, config["patch_size"],
                      num_samples=config["num_samples_per_volume"], is_train=True, cache_dir=cache_dir),
        batch_size=config["batch_size"], shuffle=True, num_workers=nw,
        pin_memory=True, persistent_workers=True, prefetch_factor=pf, drop_last=True)
    val_loader = DataLoader(
        KiTS23Dataset(val_dicts, config["patch_size"], num_samples=1, is_train=False, cache_dir=cache_dir),
        batch_size=1, shuffle=False, num_workers=max(nw//2,2),
        pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return train_loader, val_loader, train_dicts, val_dicts


# ============================================================================
# POST-PROCESSING
# ============================================================================


def postprocess(pred: np.ndarray, config: dict) -> np.ndarray:
    out = pred.copy()
    for cls_id, min_size in [(1,config["min_kidney_voxels"]),(2,config["min_tumor_voxels"]),(3,config["min_cyst_voxels"])]:
        mask=out==cls_id
        if not mask.any(): continue
        labeled,_=ndimage.label(mask); cs=np.bincount(labeled.ravel())
        lut=cs<min_size; lut[0]=False; out[lut[labeled]]=0
    return out


# ============================================================================
# SLIDING WINDOW INFERENCE
# ============================================================================


def _sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap=0.25, mode="constant"):
    assert inputs.shape[0]==1
    device=inputs.device; pd,ph,pw=roi_size
    _,_,d_orig,h_orig,w_orig=inputs.shape
    pad_d=max(0,pd-d_orig); pad_h=max(0,ph-h_orig); pad_w=max(0,pw-w_orig)
    if pad_d or pad_h or pad_w:
        inputs=F.pad(inputs,(0,pad_w,0,pad_h,0,pad_d),value=0.)
    _,_,D,H,W=inputs.shape
    def _starts(sz,p,s):
        pts=list(range(0,sz-p+1,s))
        if not pts or pts[-1]+p<sz: pts.append(sz-p)
        return pts
    sd=max(1,int(pd*(1-overlap))); sh=max(1,int(ph*(1-overlap))); sw=max(1,int(pw*(1-overlap)))
    coords=[(a,b,c) for a in _starts(D,pd,sd) for b in _starts(H,ph,sh) for c in _starts(W,pw,sw)]
    if mode=="gaussian":
        def _g(n,s): idx=torch.arange(n).float()-n/2; return torch.exp(-idx**2/(2*s**2))
        importance=(_g(pd,pd/8)[:,None,None]*_g(ph,ph/8)[None,:,None]*_g(pw,pw/8)[None,None,:]).unsqueeze(0).unsqueeze(0)
    else:
        importance=torch.ones(1,1,pd,ph,pw)
    a0,b0,c0=coords[0]
    sl0=(slice(None),slice(None),slice(a0,a0+pd),slice(b0,b0+ph),slice(c0,c0+pw))
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=_DTYPE):
            first_out=predictor(inputs[sl0])
    n_cls=first_out.shape[1]
    output=torch.zeros(1,n_cls,D,H,W); count=torch.zeros(1,1,D,H,W)
    output[sl0]+=first_out.cpu().float()*importance; count[sl0]+=importance; del first_out
    i=1
    while i<len(coords):
        bc=coords[i:i+sw_batch_size]
        batch_in=torch.cat([inputs[(slice(None),slice(None),slice(a,a+pd),slice(b,b+ph),slice(c,c+pw))] for a,b,c in bc],0)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=_DTYPE):
                bout=predictor(batch_in)
        boc=bout.cpu().float(); del batch_in,bout
        for j,(a,b,c) in enumerate(bc):
            sl=(slice(None),slice(None),slice(a,a+pd),slice(b,b+ph),slice(c,c+pw))
            output[sl]+=boc[j:j+1]*importance; count[sl]+=importance
        del boc; i+=sw_batch_size
    output=output/count.clamp(min=1e-8)
    return output[(slice(None),slice(None),slice(0,d_orig),slice(0,h_orig),slice(0,w_orig))]


# ============================================================================
# SLIDING WINDOW VALIDATION
# ============================================================================


def validate_sliding_window(model, val_dicts, config, device, n_cases=None, logger=None):
    model.eval()
    cache_dir=_cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists(): cache_dir=None
    cases=val_dicts
    if n_cases and n_cases<len(val_dicts):
        idx=np.random.choice(len(val_dicts),size=n_cases,replace=False); cases=[val_dicts[i] for i in idx]
    tp_acc=np.zeros(4,dtype=np.float64); fp_acc=np.zeros(4,dtype=np.float64); fn_acc=np.zeros(4,dtype=np.float64)
    log=logger.info if logger else print
    err_log = logger.error if logger else print
    for d in tqdm(cases, desc="SW-Val", leave=False):
        try:
            img,lbl_gt=_load_volume(d,cache_dir)
            img_t=torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
            pred_logits=_sliding_window_inference(img_t,roi_size=config["patch_size"],
                sw_batch_size=config.get("sw_batch_size",4),predictor=model,
                overlap=config.get("sw_overlap",0.25),mode="constant")
            pred=torch.argmax(pred_logits,dim=1).squeeze(0).cpu().numpy()
            pred=postprocess(pred,config)
            for c in range(4):
                tp_acc[c]+=int(((pred==c)&(lbl_gt==c)).sum())
                fp_acc[c]+=int(((pred==c)&(lbl_gt!=c)).sum())
                fn_acc[c]+=int(((pred!=c)&(lbl_gt==c)).sum())
            del img_t,pred_logits; torch.cuda.empty_cache()
        except Exception as e:
            err_log(f"  [Val] SW-Val error on case {d.get('case_id','?')}: {e}", exc_info=True)
    eps=1e-8
    dice=(2*tp_acc/(2*tp_acc+fp_acc+fn_acc+eps)); iou=(tp_acc/(tp_acc+fp_acc+fn_acc+eps))
    prec=(tp_acc/(tp_acc+fp_acc+eps)); rec=(tp_acc/(tp_acc+fn_acc+eps))
    mean_fg=float(dice[1:].mean())
    log(f"  SW-Val ({len(cases)} cases): K={dice[1]:.4f} T={dice[2]:.4f} C={dice[3]:.4f} "
        f"MeanFG={mean_fg:.4f} | mIoU={iou[1:].mean():.4f} Prec={prec[1:].mean():.4f} Rec={rec[1:].mean():.4f}")
    return mean_fg, {"dice":dice.tolist(),"iou":iou.tolist(),"precision":prec.tolist(),"recall":rec.tolist(),
                     "mean_fg_dice":mean_fg,"mean_fg_iou":float(iou[1:].mean()),
                     "mean_fg_prec":float(prec[1:].mean()),"mean_fg_rec":float(rec[1:].mean())}


# ============================================================================
# LR SCHEDULER + SAFE RESUME
# ============================================================================


def _make_scheduler(optimizer, n_epochs, warmup_epochs):
    def _lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch+1)/float(max(1,warmup_epochs))
        progress=float(epoch-warmup_epochs)/float(max(1,n_epochs-warmup_epochs))
        return 0.01+0.5*0.99*(1.+np.cos(np.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _strip_compiled_prefix(state_dict: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def _safe_load_scheduler(scheduler, state_dict, n_done):
    logger = logging.getLogger("v10")
    try:
        scheduler.load_state_dict(state_dict)
    except (KeyError, ValueError) as e:
        logger.warning(f"  Scheduler state incompatible ({e}) -- fast-forwarding {n_done} steps")
        for _ in range(n_done): scheduler.step()


# ============================================================================
# TRAINING UTILITIES
# ============================================================================


class EarlyStopping:
    def __init__(self, patience, ckpt_path):
        self.patience=patience; self.ckpt_path=ckpt_path
        self.counter=0; self.best=None; self.early_stop=False

    def __call__(self, score, model):
        logger = logging.getLogger("v10")
        if self.best is None or score>self.best:
            self.best=score; self.counter=0; torch.save(_strip_compiled_prefix(model.state_dict()), self.ckpt_path); return True
        self.counter+=1
        logger.info(f"  EarlyStopping: no improvement {self.counter}/{self.patience}")
        if self.counter>=self.patience: self.early_stop=True
        return False


def _fmt(v, fmt=".4f"): return f"{v:{fmt}}" if v is not None else "N/A"


def save_history(output_dir, stage, epoch, train_loss, val_loss, sw_metrics, elapsed_s, is_best):
    entry={"timestamp":datetime.now().isoformat(),"stage":stage,"epoch":epoch,
           "epoch_time_min":round(elapsed_s/60,2),"train_loss":float(train_loss),
           "val_loss":float(val_loss) if val_loss is not None else None,
           "sw_metrics":sw_metrics,"is_best":is_best}
    hist_file=Path(output_dir)/"history.json"
    hist=json.loads(hist_file.read_text()) if hist_file.exists() else {"epochs":[]}
    hist["epochs"].append(entry)
    hist_file.write_text(json.dumps(hist,indent=2))


# ============================================================================
# TRAIN / VAL
# ============================================================================


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_accum_steps=1):
    torch.cuda.empty_cache(); model.train()
    total_loss=0.; count=0
    optimizer.zero_grad(set_to_none=True)
    logger = logging.getLogger("v10")
    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        try:
            img=batch["image"].to(device, non_blocking=True)
            lbl=batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=_DTYPE):
                out=model(img)
                loss=loss_fn(out,lbl) / grad_accum_steps
            if not torch.isfinite(loss):
                logger.warning(f"  [Train] Infinite loss on batch {step}, skipping.")
                del img,lbl,out,loss; continue
            loss.backward()
            total_loss+=loss.item()*grad_accum_steps; count+=1
            if (step+1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            del img,lbl,out,loss
        except RuntimeError as e:
            logger.error(f"  [Train] RuntimeError on batch {step}: {e}", exc_info=True)
            optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"  [Train] Unexpected exception on batch {step}: {e}", exc_info=True)
            optimizer.zero_grad(set_to_none=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return total_loss/max(count,1)


def validate_patch_loss(model, loader, loss_fn, device):
    torch.cuda.empty_cache(); model.eval()
    total_loss=0.; count=0
    logger = logging.getLogger("v10")
    with torch.no_grad():
        for batch in tqdm(loader, desc="QuickVal", leave=False):
            try:
                img=batch["image"].to(device,non_blocking=True)
                lbl=batch["label"].to(device,non_blocking=True)
                with torch.amp.autocast("cuda",dtype=_DTYPE):
                    out=model(img); loss=loss_fn(out,lbl)
                if torch.isfinite(loss): total_loss+=loss.item(); count+=1
                del img,lbl,out,loss
            except RuntimeError as e:
                logger.error(f"  [Val] Patch loss runtime error: {e}", exc_info=True)
                torch.cuda.empty_cache()
    return total_loss/max(count,1)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================


def _run_stage(stage_name, model, train_loader, val_loader, val_dicts, config,
               optimizer, scheduler, loss_fn, device,
               n_epochs, val_every, patience, ckpt_path, output_dir, logger,
               start_epoch=1, resume_es_best=None, resume_es_counter=0):
    es=EarlyStopping(patience,ckpt_path)
    if resume_es_best is not None: es.best=resume_es_best
    es.counter=resume_es_counter; best_dice=resume_es_best or 0.
    resume_save=str(Path(output_dir)/"resume_state.pth")
    accum=config.get("grad_accum_steps",1)

    for ep in range(start_epoch, n_epochs+1):
        t0=time.time()
        train_loss=train_one_epoch(model,train_loader,optimizer,loss_fn,device,grad_accum_steps=accum)
        scheduler.step(); elapsed=time.time()-t0
        do_val=(ep%val_every==0) or (ep==n_epochs)
        val_loss=None; sw_metrics=None; improved=False
        if do_val:
            val_loss=validate_patch_loss(model,val_loader,loss_fn,device)
            sw_dice,sw_metrics=validate_sliding_window(model,val_dicts,config,device,
                n_cases=config["sw_val_cases"],logger=logger)
            improved=es(sw_dice,model)
            if improved: best_dice=sw_dice; logger.info(f"  New best MeanFG Dice = {best_dice:.4f}")
        logger.info(f"[{stage_name} E{ep:03d}/{n_epochs}] train={train_loss:.4f}  "
                    f"val={_fmt(val_loss)}  lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"time={elapsed/60:.1f}min  accum={accum}")
        save_history(output_dir,stage_name,ep,train_loss,val_loss,sw_metrics,elapsed,improved)
        _resume_tmp=resume_save+".tmp"
        torch.save({"model_state_dict":_strip_compiled_prefix(model.state_dict()),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "scheduler_state_dict":scheduler.state_dict(),
                    "stage":stage_name,"epoch":ep,"es_best":es.best,"es_counter":es.counter},_resume_tmp)
        os.replace(_resume_tmp, resume_save)
        if es.early_stop: logger.info(f"  Early stopping at epoch {ep}"); break
    return es.best or best_dice


def _resume_from_history(output_dir, device):
    logger = logging.getLogger("v10")
    hist_file=output_dir/"history.json"
    if not hist_file.exists(): return None
    all_eps=json.loads(hist_file.read_text()).get("epochs",[])
    if not all_eps: return None
    last=all_eps[-1]; stage=last["stage"]; epoch=last["epoch"]
    ckpt_path=output_dir/("best_final.pth" if stage=="S2" else "best_stage1.pth")
    if not ckpt_path.exists():
        logger.warning(f"  [Resume] Found history.json but checkpoint file {ckpt_path} is missing!")
        return None
    stage_val=[e for e in all_eps if e["stage"]==stage and e.get("sw_metrics")]
    if not stage_val: return None
    es_best=max(e["sw_metrics"]["mean_fg_dice"] for e in stage_val)
    try:
        weights=torch.load(str(ckpt_path),map_location=device,weights_only=False)
        return {"model_state_dict":weights,"optimizer_state_dict":None,"scheduler_state_dict":None,
                "stage":stage,"epoch":epoch,"es_best":es_best,"es_counter":0,"_fallback":True}
    except Exception as e:
        logger.error(f"  [Resume] Failed to load checkpoint {ckpt_path} during history resume: {e}", exc_info=True)
        return None


def train(config, resume_path=None, skip_history=False):
    seed=config.get("seed",42)
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True

    output_dir=Path(config["output_dir"]); output_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(output_dir/"training.log"),
                  logging.StreamHandler(sys.stdout)],force=True)
    logger=logging.getLogger("v10")
    accum=config.get("grad_accum_steps",1)
    eff_batch=config["batch_size"]*accum

    logger.info("="*65)
    logger.info("SwinUNETR V10 OPTIMIZED -- Isotropic Resampling & Stable Loss")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Output:  {output_dir}  |  Dtype: BF16  |  Eff.Batch: {eff_batch} ({config['batch_size']}x{accum})")
    logger.info(f"S2 Tuning: {'LoRA (rank='+str(config['lora_s2_rank'])+')' if config.get('lora_s2',False) else 'Full Encoder Fine-tuning'}")
    if config["use_llm"]:
        logger.info(f"LLM: {config['llm_model_name']} (layer {config['llm_layer_idx']}, rank {config['lora_rank']})")
    logger.info("="*65)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    try:
        model=SwinUNETR_LLM(
            num_classes=config["num_classes"], feature_size=config["feature_size"],
            use_llm=config["use_llm"], llm_model_name=config["llm_model_name"],
            llm_layer_idx=config["llm_layer_idx"], lora_rank=config["lora_rank"],
            use_checkpoint=config.get("use_checkpoint",True)).to(device)
    except Exception as e:
        logger.error(f"  [Model] Failed to instantiate SwinUNETR_LLM: {e}", exc_info=True)
        sys.exit("Model initialization failed. Check traceback above.")

    if config.get("use_compile",False) and hasattr(torch,"compile"):
        logger.info("  Compiling model (reduce-overhead) ...")
        model=torch.compile(model,mode="reduce-overhead",dynamic=False)

    train_loader,val_loader,_,val_dicts=get_dataloaders(config)
    loss_fn=DiceCELoss(config["num_classes"]).to(device)

    # Resume
    resume_state=None; auto_resume=output_dir/"resume_state.pth"
    if resume_path and Path(resume_path).exists():
        try:
            resume_state=torch.load(resume_path,map_location=device,weights_only=False)
            logger.info(f"  Resuming from: {resume_path}")
        except Exception as e:
            logger.error(f"  [Resume] Failed to load explicit resume path {resume_path}: {e}", exc_info=True)
    elif auto_resume.exists():
        try:
            resume_state=torch.load(str(auto_resume),map_location=device,weights_only=False)
            logger.info(f"  Auto-resuming from: {auto_resume}")
        except Exception as e:
            logger.error(f"  [Resume] Failed to load auto-resume file {auto_resume}: {e}", exc_info=True)
    elif not skip_history:
        resume_state=_resume_from_history(output_dir,device)
        if resume_state: logger.info("  Fallback resume from history.json")
        
    if resume_state:
        es_best_str = f"{resume_state['es_best']:.4f}" if resume_state['es_best'] is not None else "N/A"
        logger.info(f"    Stage={resume_state['stage']}  Epoch={resume_state['epoch']}  ES_best={es_best_str}")

    in_s2=resume_state is not None and resume_state["stage"]=="S2"
    s1_ckpt=output_dir/"best_stage1.pth"

    # Stage 1
    if in_s2 or s1_ckpt.exists():
        best_s1=0.
        if in_s2:
            logger.info("\n-- Stage 1: Skipped (resuming Stage 2) --")
        else:
            logger.info("\n-- Stage 1: Skipped (best_stage1.pth already exists) --")
            try:
                model.load_state_dict(_strip_compiled_prefix(
                    torch.load(str(s1_ckpt),map_location=device,weights_only=False)))
                logger.info(f"  Loaded best_stage1.pth for fresh S2 start")
            except Exception as e:
                logger.error(f"  [Init] Failed to load {s1_ckpt}: {e}", exc_info=True)
                logger.warning("  Starting Stage 1 training instead.")
                s1_ckpt.unlink(missing_ok=True)
                resume_state=None
                
    if not (in_s2 or s1_ckpt.exists()):
        logger.info("\n-- Stage 1: Frozen encoder, decoder + LLM-LoRA adapters train --")
        model.freeze_encoder()
        opt1=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=config["lr_stage1"],weight_decay=config["weight_decay"])
        sched1=_make_scheduler(opt1,config["num_epochs_stage1"],config.get("warmup_epochs_s1",3))
        s1_start,s1_es_best,s1_es_counter=1,None,0
        if resume_state and resume_state["stage"]=="S1":
            s1_start=resume_state["epoch"]+1; s1_es_best=resume_state["es_best"]; s1_es_counter=resume_state["es_counter"]
            try:
                model.load_state_dict(_strip_compiled_prefix(resume_state["model_state_dict"]))
                if resume_state.get("optimizer_state_dict"):
                    opt1.load_state_dict(resume_state["optimizer_state_dict"])
                    _safe_load_scheduler(sched1,resume_state["scheduler_state_dict"],resume_state["epoch"])
                else:
                    for _ in range(resume_state["epoch"]): sched1.step()
                logger.info(f"  Restored S1 -- continuing from epoch {s1_start}")
            except Exception as e:
                logger.error(f"  [Resume] Failed to restore S1 state: {e}", exc_info=True)
                logger.warning("  Restarting Stage 1 from scratch.")
                s1_start,s1_es_best,s1_es_counter=1,None,0
                
        best_s1=_run_stage("S1",model,train_loader,val_loader,val_dicts,config,
                            opt1,sched1,loss_fn,device,
                            n_epochs=config["num_epochs_stage1"],val_every=config["val_every_s1"],
                            patience=config["patience_stage1"],ckpt_path=str(output_dir/"best_stage1.pth"),
                            output_dir=str(output_dir),logger=logger,
                            start_epoch=s1_start,resume_es_best=s1_es_best,resume_es_counter=s1_es_counter)
        if s1_ckpt.exists():
            try:
                model.load_state_dict(_strip_compiled_prefix(torch.load(str(s1_ckpt),map_location=device,weights_only=False)))
                logger.info(f"  Restored S1 best (MeanFG={best_s1:.4f})")
            except Exception as e:
                logger.error(f"  [Init] Failed to load best S1 checkpoint: {e}", exc_info=True)

    s2_start,s2_es_best,s2_es_counter=1,None,0
    _s2_state=None
    if resume_state and resume_state["stage"]=="S2":
        s2_start=resume_state["epoch"]+1; s2_es_best=resume_state["es_best"]; s2_es_counter=resume_state["es_counter"]
        _s2_state=_strip_compiled_prefix(resume_state["model_state_dict"])

    _ckpt_has_lora=_s2_state is not None and any("swinViT" in k and "lora_A" in k for k in _s2_state)
    if _ckpt_has_lora and config.get("lora_s2", False):
        _sample_lora_key=next(k for k in _s2_state if "swinViT" in k and "lora_A" in k)
        _ckpt_rank=_s2_state[_sample_lora_key].shape[0]
        if _ckpt_rank != config["lora_s2_rank"]:
            logger.warning(f"  S2 checkpoint LoRA rank mismatch. Discarding S2 state.")
            model.load_state_dict(_strip_compiled_prefix(
                torch.load(str(s1_ckpt), map_location=device, weights_only=False)))
            _s2_state=None; _ckpt_has_lora=False
            s2_start,s2_es_best,s2_es_counter=1,None,0

    if _ckpt_has_lora:
        logger.info(f"  S2 checkpoint has encoder LoRA keys — applying LoRA before load ...")
        if config.get("lora_s2", False):
            model.unfreeze_encoder_with_lora(lora_rank=config["lora_s2_rank"])
        else:
            model.unfreeze_encoder_full()
        model.to(device)
        model.load_state_dict(_s2_state)
    else:
        if _s2_state is not None:
            model.load_state_dict(_s2_state)
        if config.get("lora_s2", False):
            model.unfreeze_encoder_with_lora(lora_rank=config["lora_s2_rank"])
        else:
            model.unfreeze_encoder_full()
        model.to(device)

    opt2=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=config["lr_stage2"],weight_decay=config["weight_decay"])
    sched2=_make_scheduler(opt2,config["num_epochs_stage2"],config.get("warmup_epochs_s2",5))
    if resume_state and resume_state["stage"]=="S2":
        if resume_state.get("optimizer_state_dict"):
            try:
                opt2.load_state_dict(resume_state["optimizer_state_dict"])
                _safe_load_scheduler(sched2,resume_state["scheduler_state_dict"],resume_state["epoch"])
                logger.info(f"  Restored S2 -- continuing from epoch {s2_start}")
            except Exception as e:
                logger.error(f"  [Resume] Failed to restore S2 optimizer state: {e}", exc_info=True)
                logger.warning("  Continuing Stage 2 with a clean optimizer.")
        else:
            for _ in range(resume_state["epoch"]): sched2.step()
            logger.info(f"  Restored S2 (weights only) -- continuing from epoch {s2_start}")
            
    best_s2=_run_stage("S2",model,train_loader,val_loader,val_dicts,config,
                        opt2,sched2,loss_fn,device,
                        n_epochs=config["num_epochs_stage2"],val_every=config["val_every_s2"],
                        patience=config["patience_stage2"],ckpt_path=str(output_dir/"best_final.pth"),
                        output_dir=str(output_dir),logger=logger,
                        start_epoch=s2_start,resume_es_best=s2_es_best,resume_es_counter=s2_es_counter)

    best_dice=max(best_s1,best_s2); logger.info(f"\nTraining complete. Best MeanFG Dice: {best_dice:.4f}")
    
    try:
        history=json.loads((output_dir/"history.json").read_text())
        total_min=sum(e.get("epoch_time_min",0) for e in history["epochs"])
        (output_dir/"summary.json").write_text(json.dumps({
            "version":"SwinUNETR_V10_LLM","llm_model":config["llm_model_name"] if config["use_llm"] else "none",
            "lora_rank":config["lora_rank"],"lora_s2":config.get("lora_s2",False),"lora_s2_rank":config.get("lora_s2_rank",4),
            "dtype":"bfloat16","compiled":config.get("use_compile",False),"grad_accum":accum,"eff_batch":eff_batch,
            "best_dice":best_dice,"total_hours":round(total_min/60,2),
            "config":{k:str(v) for k,v in config.items()}},indent=2))
        logger.info(f"Summary -> {output_dir/'summary.json'}")
    except Exception as e:
        logger.error(f"  Failed to save summary.json: {e}", exc_info=True)


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate(config, checkpoint):
    logging.basicConfig(level=logging.INFO,format="%(message)s",handlers=[logging.StreamHandler(sys.stdout)],force=True)
    logger=logging.getLogger("eval")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SwinUNETR_LLM(num_classes=config["num_classes"],feature_size=config["feature_size"],
                         use_llm=config["use_llm"],llm_model_name=config["llm_model_name"],
                         llm_layer_idx=config["llm_layer_idx"],lora_rank=config["lora_rank"]).to(device)
    ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    logger.info(f"Loaded: {checkpoint}")
    
    cache_dir=_cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists(): cache_dir=None
    
    data_dir=Path(config["kits23_dir"])
    test_cases=sorted([c for c in data_dir.iterdir()
                       if c.is_dir() and c.name.startswith("case_")])[-config["test_cases"]:]
    model.eval(); per_case=[]; conf_matrix=np.zeros((4,4),dtype=np.int64)
    sw_bs=config.get("sw_batch_size",4)
    logger.info(f"\nEvaluating {len(test_cases)} test cases with isotropic back-resampling...")

    from scipy.ndimage import zoom

    with torch.no_grad():
        for case_dir in test_cases:
            d={"image":str(case_dir/"imaging.nii.gz"),"label":str(case_dir/"segmentation.nii.gz"),"case_id":case_dir.name}
            if not Path(d["image"]).exists() or not Path(d["label"]).exists(): continue
            t0=time.time()
            
            # Load original NIfTI properties
            img_nii = nib.load(d["image"])
            lbl_nii = nib.load(d["label"])
            lbl_gt_orig = lbl_nii.get_fdata().astype(np.int64)
            
            # Load resampled volume for inference
            img, _ = _load_volume(d, cache_dir)
            img_t=torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Inference in 1.5mm space
            pred_logits=_sliding_window_inference(img_t,roi_size=config["patch_size"],sw_batch_size=sw_bs,
                predictor=model,overlap=0.5,mode="gaussian")
            pred_res=torch.argmax(pred_logits,dim=1).squeeze(0).cpu().numpy()
            
            # Resample predicted mask back to original shape
            zoom_factors_back = [o / p for o, p in zip(lbl_gt_orig.shape, pred_res.shape)]
            pred_orig = zoom(pred_res, zoom_factors_back, order=0).astype(np.int32)
            
            # Post-process at original scale
            pred_orig = postprocess(pred_orig, config)
            
            # Metrics evaluated at original scale
            m=compute_seg_metrics(pred_orig,lbl_gt_orig,num_classes=4)
            hd95=[float("nan")]+[compute_hd95(pred_orig==c,lbl_gt_orig==c) for c in range(1,4)]
            
            for gc_ in range(4):
                for pc_ in range(4): conf_matrix[gc_,pc_]+=int(((lbl_gt_orig==gc_)&(pred_orig==pc_)).sum())
                
            tp_raw=[int(((pred_orig==c)&(lbl_gt_orig==c)).sum()) for c in range(4)]
            fp_raw=[int(((pred_orig==c)&(lbl_gt_orig!=c)).sum()) for c in range(4)]
            fn_raw=[int(((pred_orig!=c)&(lbl_gt_orig==c)).sum()) for c in range(4)]
            
            elapsed=time.time()-t0
            per_case.append({"case_id":case_dir.name,"dice":m["dice"],"iou":m["iou"],
                             "precision":m["precision"],"recall":m["recall"],"hd95":hd95,
                             "raw_counts":{"tp":tp_raw,"fp":fp_raw,"fn":fn_raw},"elapsed_s":round(elapsed,1)})
            logger.info(f"{case_dir.name} | K={m['dice'][1]:.3f} T={m['dice'][2]:.3f} C={m['dice'][3]:.3f} | "
                        f"K_hd95={hd95[1]:.1f} T_hd95={hd95[2]:.1f} ({elapsed:.0f}s)")
            del img_t,pred_logits

    agg=aggregate_metrics(per_case,num_classes=4)
    logger.info("\n"+"="*72+"\nEVALUATION SUMMARY\n"+"="*72)
    logger.info(f"{'Class':<12} {'Dice':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'HD95':>8}")
    logger.info("-"*72)
    for i,cls in enumerate(CLASS_NAMES):
        logger.info(f"{cls:<12} {agg['dice'][i]:>8.4f} {agg['iou'][i]:>8.4f} "
                    f"{agg['precision'][i]:>10.4f} {agg['recall'][i]:>8.4f} {agg['hd95_mean'][i]:>8.2f}")
    logger.info("-"*72)
    logger.info(f"{'MeanFG':<12} {agg['mean_fg_dice']:>8.4f} {agg['mean_fg_iou']:>8.4f} "
                f"{agg['mean_fg_precision']:>10.4f} {agg['mean_fg_recall']:>8.4f} {agg['mean_fg_hd95']:>8.2f}")
    logger.info("="*72)

    out_dir=Path(config["output_dir"]); out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"eval_results.json").write_text(json.dumps({"global_metrics":agg,"per_case":per_case},indent=2))
    logger.info(f"Results -> {out_dir/'eval_results.json'}")

    csv_path=out_dir/"eval_per_case.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["case_id","dice_bg","dice_kidney","dice_tumor","dice_cyst",
                    "iou_bg","iou_kidney","iou_tumor","iou_cyst",
                    "prec_bg","prec_kidney","prec_tumor","prec_cyst",
                    "rec_bg","rec_kidney","rec_tumor","rec_cyst",
                    "hd95_kidney","hd95_tumor","hd95_cyst","elapsed_s"])
        for pc in per_case:
            w.writerow([pc["case_id"],*pc["dice"],*pc["iou"],*pc["precision"],*pc["recall"],
                        pc["hd95"][1],pc["hd95"][2],pc["hd95"][3],pc["elapsed_s"]])
    logger.info(f"Per-case CSV -> {csv_path}")
    (out_dir/"confusion_matrix.json").write_text(json.dumps(
        {"class_names":CLASS_NAMES,"matrix_rows_GT_cols_pred":conf_matrix.tolist(),
          "note":"Voxel-level 4x4 confusion matrix. Row=GT, Col=Predicted."},indent=2))


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    parser=argparse.ArgumentParser(description="SwinUNETR V10 OPTIMIZED")
    parser.add_argument("--mode",choices=["full","medium","showcase","quick_test","evaluate","build_cache"],default="full")
    parser.add_argument("--quick-test-baseline",action="store_true")
    parser.add_argument("--checkpoint",default=None)
    parser.add_argument("--kits23-dir",default=None)
    parser.add_argument("--output-dir",default=None)
    parser.add_argument("--batch-size",type=int,default=None)
    parser.add_argument("--grad-accum",type=int,default=None)
    parser.add_argument("--patch-size",default=None)
    parser.add_argument("--sw-overlap",type=float,default=None)
    parser.add_argument("--sw-batch-size",type=int,default=None)
    parser.add_argument("--llm-model",default=None)
    parser.add_argument("--lora-rank",type=int,default=None)
    parser.add_argument("--lora-s2-rank",type=int,default=None)
    parser.add_argument("--lora-s2",action="store_true",help="Enable LoRA on SwinViT encoder in S2 instead of full unfreeze")
    parser.add_argument("--no-llm",action="store_true")
    parser.add_argument("--no-compile",action="store_true")
    parser.add_argument("--no-checkpoint",action="store_true")
    parser.add_argument("--resume",default=None,metavar="PATH")
    parser.add_argument("--no-resume",action="store_true")
    args=parser.parse_args()

    selected_mode="quick_test" if args.quick_test_baseline else args.mode
    cfg=get_config(selected_mode if selected_mode not in ("evaluate","build_cache") else "full")

    if args.kits23_dir:            cfg["kits23_dir"]     = args.kits23_dir
    if args.output_dir:            cfg["output_dir"]     = args.output_dir
    elif args.quick_test_baseline: cfg["output_dir"]     = "./output/swin_v10_quick_baseline"
    if args.batch_size:            cfg["batch_size"]     = args.batch_size
    if args.grad_accum:            cfg["grad_accum_steps"] = args.grad_accum
    if args.patch_size:            cfg["patch_size"]     = tuple(int(x) for x in args.patch_size.split(","))
    if args.sw_overlap is not None: cfg["sw_overlap"]    = args.sw_overlap
    if args.sw_batch_size:         cfg["sw_batch_size"]  = args.sw_batch_size
    if args.llm_model:             cfg["llm_model_name"] = args.llm_model
    if args.lora_rank is not None: cfg["lora_rank"]      = args.lora_rank
    if args.lora_s2_rank is not None: cfg["lora_s2_rank"] = args.lora_s2_rank
    if args.lora_s2:               cfg["lora_s2"]        = True
    if args.no_llm or args.quick_test_baseline: cfg["use_llm"] = False
    if args.no_compile:            cfg["use_compile"]    = False
    if args.no_checkpoint:         cfg["use_checkpoint"] = False

    # Setup basic logging to allow logger usage even in non-train/non-eval commands (e.g. build_cache)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)

    if selected_mode == "build_cache":
        build_cache(cfg["kits23_dir"], _cache_dir_for(cfg["kits23_dir"]))
    elif selected_mode == "evaluate":
        if not args.checkpoint: parser.error("--checkpoint required for evaluate mode")
        evaluate(cfg, args.checkpoint)
    else:
        if args.no_resume:
            rp=Path(cfg["output_dir"])/"resume_state.pth"
            if rp.exists(): rp.unlink()
        train(cfg, resume_path=args.resume, skip_history=args.no_resume)


if __name__ == "__main__":
    main()
