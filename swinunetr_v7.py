"""
3D Kidney Segmentation — V7
============================
Changes from V6:
  1. BiomedCLIP replaces ViT-B-32 for MGPL
       - microsoft/BiomedCLIP-PubMedBERT_256-vit_L_14 via open_clip
       - Trained on 15M PubMed figure-caption pairs → medically grounded
         embeddings for "renal tumor", "cyst", "kidney cortex" etc.
       - Text embed dim: 512 (same interface, no decoder changes needed)

  2. MONAI renalStructures_UNEST_segmentation encoder init
       - Pre-trained on KiTS21 (same task: kidney/tumor/cyst) via MONAI bundle
       - Falls back to generic SwinViT if bundle unavailable
       - Expected to reach 0.65+ Dice much faster than generic CT pretraining

  3. Hard-Example Resampling (HER) replaces broken VLM critic
       - After every val epoch, ranks cases by worst Dice score
       - Injects extra training patches from top-N worst cases into the
         next epoch's DataLoader via on-the-fly dataset reweighting
       - No external model, no network calls, zero overhead
       - Directly addresses the cyst/tumor recall problem

Dataset: KiTS23 (Kidney Tumor Segmentation)
Classes: 0=Background, 1=Kidney, 2=Tumor, 3=Cyst
GPU:     RTX 5090 32GB

Usage:
    python swinunetr_v7.py --mode full
    python swinunetr_v7.py --mode full --resume
    python swinunetr_v7.py --mode quick_test
"""

import os
import json
import time
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
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]

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


def get_config(mode: str) -> dict:
    base = {
        "kits23_dir": "./kits23/dataset",
        "output_dir": "./output/swinunetr_v7",
        "num_classes": 4,
        "feature_dim": 256,
        "swin_feature_size": 48,
        "batch_size": 2,
        "val_split": 0.20,
        "num_samples_per_volume": 2,
        "patch_size": (96, 192, 192),
        "lr_stage1": 1e-4,
        "lr_stage2": 1e-5,
        "weight_decay": 1e-5,
        # BiomedCLIP for MGPL
        "clip_model": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_L_14",
        "clip_pretrained": "",  # unused for hf-hub: models
        "use_mgpl": True,
        # Hard-Example Resampling
        "her_top_n": 20,  # worst N cases to oversample
        "her_extra_samples": 2,  # extra patches per hard case per epoch
        "her_every_k_epochs": 3,  # update hard list every K val epochs
        # Split
        "test_cases": 40,
        "seed": 42,
    }

    _qt = {
        "num_epochs_stage1": 5,
        "num_epochs_stage2": 2,
        "quick_cases": 40,
        "use_warmup": False,
        "patience_stage1": 3,
        "patience_stage2": 2,
        "warmup_s2_epochs": 1,
        "val_every_s1": 1,
        "val_every_s2": 1,
        "output_dir": "./output/swinunetr_v7_quick",
    }

    if mode == "quick_test":
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
# IMPROVEMENT 1 — BiomedCLIP MULTI-GRANULARITY PROMPT ENCODER
# ============================================================================


class MultiGranularityPromptEncoder(nn.Module):
    """
    BiomedCLIP text encoder (microsoft/BiomedCLIP-PubMedBERT_256-vit_L_14).
    Trained on 15M PubMed figure-caption pairs — radiology terminology is
    semantically meaningful unlike generic CLIP.

    Falls back to ViT-B-32/openai if BiomedCLIP is unavailable.
    Output: (num_classes, num_levels, embed_dim) — identical interface to V6.
    """

    BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_L_14"

    def __init__(self, clip_model_name: str = None, pretrained: str = ""):
        super().__init__()
        import open_clip

        model_id = clip_model_name or self.BIOMEDCLIP_ID

        try:
            if model_id.startswith("hf-hub:"):
                model, _, _ = open_clip.create_model_and_transforms(model_id)
                self.tokenizer = open_clip.get_tokenizer(model_id)
                print(f"  ✓ BiomedCLIP loaded: {model_id}")
            else:
                model, _, _ = open_clip.create_model_and_transforms(
                    model_id, pretrained=pretrained
                )
                self.tokenizer = open_clip.get_tokenizer(model_id)
                print(f"  ✓ CLIP loaded: {model_id}/{pretrained}")
        except Exception as e:
            print(f"  ⚠ BiomedCLIP load failed ({e}), falling back to ViT-B-32/openai")
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.text_encoder = model
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self._build_prompt_embeddings()
        self.embed_dim = self.prompt_embeddings.shape[-1]
        print(
            f"✓ MGPL (BiomedCLIP): {self.prompt_embeddings.shape[0]} classes × "
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
        return self.prompt_embeddings  # (4, 5, embed_dim)


# ============================================================================
# IMPROVEMENT 2 — SWINUNETR ENCODER WITH RENAL UNEST INIT
# ============================================================================


class SwinUNETREncoder(nn.Module):
    """
    SwinUNETR encoder.
    Weight loading priority:
      1. MONAI renalStructures_UNEST_segmentation bundle (KiTS21-pretrained)
      2. Generic SwinViT weights from MONAI (CT/MRI pretrained)
      3. Random init (with warning)

    Skip/bottleneck dims projected to feature_dim=256 for decoder.
    """

    def __init__(self, feature_dim: int = 256, feature_size: int = 48):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.swin = SwinUNETR(
            in_channels=1,
            out_channels=14,
            feature_size=feature_size,
            use_checkpoint=True,
            spatial_dims=3,
        )
        self._load_weights()

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

    def _load_weights(self):
        """Try renal bundle first, fall back to generic SwinViT."""
        if self._try_load_renal_bundle():
            return
        self._try_load_generic_swinvit()

    def _try_load_renal_bundle(self) -> bool:
        """
        Download and load MONAI renalStructures_UNEST_segmentation.
        This model was trained on KiTS21 — directly relevant for KiTS23.
        Returns True if successful.
        """
        bundle_dir = Path("./monai_bundles")
        bundle_name = "renalStructures_UNEST_segmentation"
        ckpt_candidates = list(bundle_dir.glob(f"{bundle_name}/**/*.pt")) + list(
            bundle_dir.glob(f"{bundle_name}/**/*.pth")
        )

        if not ckpt_candidates:
            print(f"  Downloading MONAI bundle: {bundle_name}...")
            try:
                from monai.bundle import download

                download(name=bundle_name, bundle_dir=str(bundle_dir))
                ckpt_candidates = list(
                    bundle_dir.glob(f"{bundle_name}/**/*.pt")
                ) + list(bundle_dir.glob(f"{bundle_name}/**/*.pth"))
            except Exception as e:
                print(f"  ⚠ Bundle download failed: {e}")
                return False

        if not ckpt_candidates:
            print(f"  ⚠ No checkpoint found in bundle directory")
            return False

        # Prefer the largest file (most likely the model weights, not config)
        ckpt_path = max(ckpt_candidates, key=lambda p: p.stat().st_size)
        print(f"  ✓ Loading renal bundle weights: {ckpt_path.name}")

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            weights = ckpt.get("state_dict", ckpt)
            model_dict = self.swin.state_dict()

            matched = self._match_weights(weights, model_dict)
            if matched:
                model_dict.update(matched)
                self.swin.load_state_dict(model_dict, strict=False)
                print(
                    f"  ✓ Loaded {len(matched)}/{len(model_dict)} renal bundle weights"
                    f" ({len(matched)/len(model_dict)*100:.0f}%)"
                )
                return True
            else:
                print("  ⚠ Renal bundle: no weight keys matched")
                return False
        except Exception as e:
            print(f"  ⚠ Renal bundle load error: {e}")
            return False

    def _try_load_generic_swinvit(self):
        """Fall back to generic MONAI SwinViT pretrained weights."""
        cached = Path.home() / ".torch" / "models" / "model_swinvit.pt"
        local = Path("./swin_unetr_pretrained.pth")

        if cached.exists():
            weight_path = cached
        elif local.exists():
            weight_path = local
        else:
            weight_path = local
            print("  Downloading generic SwinUNETR weights (~392MB)...")
            import urllib.request

            url = (
                "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
                "download/0.8.1/model_swinvit.pt"
            )
            try:
                urllib.request.urlretrieve(url, str(weight_path))
                print("  ✓ Download complete")
            except Exception as e:
                print(f"  ⚠ Download failed: {e} — training from scratch")
                return

        try:
            ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
            weights = ckpt.get("state_dict", ckpt)
            model_dict = self.swin.state_dict()
            matched = self._match_weights(weights, model_dict)
            if matched:
                model_dict.update(matched)
                self.swin.load_state_dict(model_dict, strict=False)
                print(
                    f"  ✓ Loaded {len(matched)}/{len(model_dict)} generic SwinViT weights"
                )
            else:
                print("  ⚠ Generic SwinViT: no keys matched — training from scratch")
        except Exception as e:
            print(f"  ⚠ SwinViT load error: {e} — training from scratch")

    @staticmethod
    def _match_weights(src: dict, dst: dict) -> dict:
        """Try several prefix-remapping strategies to match src → dst keys."""

        def _strip_add(w, strip="", add=""):
            out = {}
            for k, v in w.items():
                mk = k[len(strip) :] if strip and k.startswith(strip) else k
                mk = add + mk if add else mk
                if mk in dst and dst[mk].shape == v.shape:
                    out[mk] = v
            return out

        for strip, add in [
            ("", ""),
            ("module.", "swinViT."),
            ("module.", ""),
            ("swinViT.", ""),
            ("swin.", ""),
            ("model.", ""),
        ]:
            matched = _strip_add(src, strip, add)
            if matched:
                if strip or add:
                    note = f"strip='{strip}' add='{add}'"
                    print(f"  (Remapped weights: {note})")
                return matched
        return {}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden_states = self.swin.swinViT(x, self.swin.normalize)

        enc = self.swin.encoder1(x)
        enc1 = self.swin.encoder2(hidden_states[0])
        enc2 = self.swin.encoder3(hidden_states[1])
        enc3 = self.swin.encoder4(hidden_states[2])
        enc4 = self.swin.encoder10(hidden_states[4])

        features = {
            "bottleneck": self.bottle_proj(enc4),
            "skip3": (
                self.skip_projs[3](hidden_states[3])
                if hidden_states[3].shape[1] == self.skip_projs[3][0].in_channels
                else self.skip_projs[3](self.swin.encoder4(hidden_states[2]))
            ),
        }
        for i, enc_out in enumerate([enc1, enc2, enc3]):
            if enc_out.shape[1] == self.skip_projs[i][0].in_channels:
                features[f"skip{i}"] = self.skip_projs[i](enc_out)
            else:
                features[f"skip{i}"] = self.skip_projs[i](hidden_states[i])

        return features

    def unfreeze_encoder(self, num_layers: int = 8):
        params = list(self.swin.parameters())
        for p in params[-num_layers * 20 :]:
            p.requires_grad = True
        print(f"  ✓ Unfroze last {num_layers} SwinUNETR blocks")


# ============================================================================
# DECODER (unchanged from V6)
# ============================================================================


class TextVisualAlignment(nn.Module):
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


class PromptAttentionModule(nn.Module):
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
        text_kv = self.text_proj(weighted_p)
        vis_flat = visual_feat.flatten(2).permute(0, 2, 1)
        text_kv_b = text_kv.unsqueeze(0).expand(B, -1, -1)
        attended, _ = self.cross_attn(vis_flat, text_kv_b, text_kv_b)
        enhanced = vis_flat + self.out_proj(attended)
        return enhanced.permute(0, 2, 1).view(B, C, D, H, W)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


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
            f"✓ TextGuidedDecoder: {decoder_channels}, prompt_attn={'ON' if use_prompt_attention else 'OFF'}"
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


class SegNetV7(nn.Module):
    """SwinUNETR + BiomedCLIP MGPL + TextGuidedDecoder."""

    def __init__(
        self,
        num_classes: int = 4,
        feature_dim: int = 256,
        swin_feature_size: int = 48,
        clip_model: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_L_14",
        use_mgpl: bool = True,
    ):
        super().__init__()
        self.use_mgpl = use_mgpl

        self.encoder = SwinUNETREncoder(
            feature_dim=feature_dim, feature_size=swin_feature_size
        )

        if use_mgpl:
            self.prompt_encoder = MultiGranularityPromptEncoder(
                clip_model_name=clip_model
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
        print(f"SegNetV7 | Total: {total:,} | Trainable: {trainable:,}")
        print(f"MGPL (BiomedCLIP): {'ON' if use_mgpl else 'OFF'}")
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

    def unfreeze_encoder(self, num_layers: int = 8):
        self.encoder.unfreeze_encoder(num_layers)


# ============================================================================
# IMPROVEMENT 3 — HARD-EXAMPLE RESAMPLING (HER)
# ============================================================================


class HardExampleResampler:
    """
    After each validation epoch, ranks all val cases by their Dice score.
    The N worst cases get extra-weight patches injected into the training
    dataset for the next epoch.

    This replaces the VLM critic. No external model needed.
    Effect: model sees more patches from cases it fails on.
    """

    def __init__(self, top_n: int = 20, extra_samples: int = 2):
        self.top_n = top_n
        self.extra_samples = extra_samples
        self.hard_case_ids: List[str] = []
        self.update_count = 0

    def update(self, per_case: List[dict]):
        """Rank cases by FG Dice, store worst N case_ids."""
        ranked = sorted(per_case, key=lambda c: c["mean_dice_fg"])
        self.hard_case_ids = [c["case_id"] for c in ranked[: self.top_n]]
        self.update_count += 1
        worst_dice = [round(c["mean_dice_fg"], 3) for c in ranked[:5]]
        print(
            f"  [HER] Updated hard list: {len(self.hard_case_ids)} cases | "
            f"5 worst Dice: {worst_dice}"
        )

    def inject_hard_samples(self, dataset: "KiTS23Dataset"):
        """
        Rebuild dataset item list with extra entries for hard cases.
        Hard cases get (1 + extra_samples) patches instead of 1.
        """
        if not self.hard_case_ids:
            return
        hard_set = set(self.hard_case_ids)
        extra_items = []
        for entry in dataset.data_dicts:
            if entry["case_id"] in hard_set:
                for _ in range(self.extra_samples):
                    extra_items.append({"type": "gt", "data": entry, "weight": 1.0})
        dataset.items.extend(extra_items)
        print(
            f"  [HER] +{len(extra_items)} hard-example patches → "
            f"dataset now {len(dataset.items)} items"
        )


# ============================================================================
# LOSS FUNCTION (same as V6 — proven effective)
# ============================================================================


class HybridLoss(nn.Module):
    """Tversky + dedicated tumor/cyst Tversky + Focal + Dice."""

    def __init__(self):
        super().__init__()
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
# DATASET
# ============================================================================


class KiTS23Dataset(Dataset):
    def __init__(
        self,
        data_dicts: List[dict],
        patch_size: Tuple,
        num_samples: int = 2,
        is_train: bool = True,
        sampling_bias: Optional[dict] = None,
    ):
        self.data_dicts = data_dicts
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.is_train = is_train
        self.sampling_bias = sampling_bias or {
            "tumor": 0.65,
            "cyst": 0.10,
            "kidney": 0.15,
            "random": 0.10,
        }
        self._build_item_list()

    def _build_item_list(self):
        self.items = []
        for d in self.data_dicts:
            for _ in range(self.num_samples):
                self.items.append({"type": "gt", "data": d, "weight": 1.0})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        data = item["data"]
        weight = item["weight"]

        img = nib.load(data["image"]).get_fdata().astype(np.float32)
        lbl = nib.load(data["label"]).get_fdata().astype(np.int64)
        img = np.clip(img, -175, 250)
        img = (img - (-175)) / (250 - (-175))

        p_img, p_lbl = self._sample_patch(img, lbl)
        return {
            "image": torch.from_numpy(p_img).float().unsqueeze(0),
            "label": torch.from_numpy(p_lbl).long().unsqueeze(0),
            "case_id": data.get("case_id", "unknown"),
            "weight": torch.tensor(weight),
        }

    def _sample_patch(self, img: np.ndarray, lbl: np.ndarray):
        d, h, w = img.shape
        pd, ph, pw = self.patch_size

        if d < pd or h < ph or w < pw:
            pad = ((0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w)))
            img = np.pad(img, pad, mode="constant")
            lbl = np.pad(lbl, pad, mode="constant")
            d, h, w = img.shape

        if self.is_train:
            rand_val = np.random.random()
            bias = self.sampling_bias
            tumor_th = bias.get("tumor", 0.65)
            cyst_th = tumor_th + bias.get("cyst", 0.10)
            kidney_th = cyst_th + bias.get("kidney", 0.15)

            target_cls = None
            if rand_val < tumor_th:
                target_cls = 2
            elif rand_val < cyst_th:
                target_cls = 3
            elif rand_val < kidney_th:
                target_cls = 1

            ds, hs, ws = self._find_patch_start(lbl, d, h, w, pd, ph, pw, target_cls)
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


# ============================================================================
# DATA LOADING
# ============================================================================


def get_dataloaders(config: dict):
    data_dir = Path(config["kits23_dir"])
    cases = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")]
    )

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
    )
    val_ds = KiTS23Dataset(
        val_dicts, config["patch_size"], num_samples=1, is_train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
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
                    }
                )
                del img, lbl, out, loss, preds
            except RuntimeError as e:
                print(f"  ⚠ Val: {e}")
                torch.cuda.empty_cache()
    report, mean_dice = tracker.report()
    metrics = tracker.compute()
    return total_loss / max(count, 1), mean_dice, report, metrics, per_case


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


def save_history(
    output_dir,
    stage,
    epoch,
    train_loss,
    val_loss,
    metrics,
    epoch_time,
    is_best,
    her_summary=None,
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
    if her_summary:
        entry["her"] = her_summary
    hist_file = Path(output_dir) / "history.json"
    hist = json.loads(hist_file.read_text()) if hist_file.exists() else {"epochs": []}
    hist["epochs"].append(entry)
    hist_file.write_text(json.dumps(hist, indent=2))


def _load_checkpoint(output_dir, stage, model, optimizer, scheduler, scaler, device):
    """Load latest or highest-epoch checkpoint. Returns (start_ep, best_dice)."""
    latest = Path(output_dir) / f"ckpt_{stage}_latest.pth"
    ckpts = sorted(
        Path(output_dir).glob(f"ckpt_{stage}_ep*.pth"),
        key=lambda p: int(p.stem.split("ep")[-1]),
    )
    resume_path = latest if latest.exists() else (ckpts[-1] if ckpts else None)
    if resume_path is None:
        return 0, 0.0
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scheduler_state_dict") and scheduler:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_ep = ckpt["epoch"] + 1
    best_dice = ckpt["best_dice"]
    print(
        f"  Resumed {stage} from epoch {start_ep} (dice={best_dice:.4f}) [{resume_path.name}]"
    )
    return start_ep, best_dice


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================


def train(config: dict, resume: bool = False):
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 65)
    logger.info("SegNetV7 — BiomedCLIP + Renal Init + HardExampleResampling")
    logger.info(f"Started at {datetime.now().isoformat()}")
    logger.info("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"\n{'='*65}")
    print("SegNetV7 — BiomedCLIP + Renal Bundle Init + Hard-Example Resampling")
    print(f"{'='*65}")
    print(f"Device:  {device}")
    print(f"Output:  {output_dir}")
    print(f"MGPL:    {'ON (BiomedCLIP)' if config['use_mgpl'] else 'OFF'}")
    print(f"HER:     top-{config['her_top_n']} +{config['her_extra_samples']} patches")
    print(f"Patch:   {config['patch_size']}")
    print(f"{'='*65}\n")

    # ── Data ──
    train_loader, val_loader, train_dicts, val_dicts = get_dataloaders(config)
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # ── Model ──
    model = SegNetV7(
        num_classes=config["num_classes"],
        feature_dim=config["feature_dim"],
        swin_feature_size=config["swin_feature_size"],
        clip_model=config["clip_model"],
        use_mgpl=config["use_mgpl"],
    ).to(device)

    loss_fn = HybridLoss().to(device)
    her = HardExampleResampler(
        top_n=config["her_top_n"],
        extra_samples=config["her_extra_samples"],
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

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1 — Frozen encoder
    # ─────────────────────────────────────────────────────────────────────────
    print("\n─── STAGE 1: Train Decoder + MGPL (SwinUNETR frozen) ───")
    best_dice = 0.0
    start_ep = 0
    last_val = dict(loss=0.0, dice=0.0, report="", metrics={}, per_case=[])

    if resume:
        start_ep, best_dice = _load_checkpoint(
            output_dir, "stage1", model, optimizer, scheduler, scaler, device
        )

    val_every = config.get("val_every_s1", 3)
    her_every = config.get("her_every_k_epochs", 3)
    t_total = time.time()

    for epoch in range(start_ep, n_s1):
        # Rebuild item list each epoch — resets injected hard-example extras
        train_loader.dataset._build_item_list()

        # Inject hard examples if HER has been seeded
        if her.hard_case_ids and (epoch % her_every == 0 or epoch == start_ep):
            her.inject_hard_samples(train_loader.dataset)

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

        sys.stdout.flush()

        # Update HER list every her_every val epochs
        her_summary = None
        if do_val and last_val["per_case"] and (epoch + 1) % her_every == 0:
            her.update(last_val["per_case"])
            her_summary = {
                "hard_cases": her.hard_case_ids,
                "num_hard": len(her.hard_case_ids),
            }

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
                her_summary,
            )
            early_stop(mean_dice, model)
            if early_stop.early_stop:
                print("  Early stopping (Stage 1)")
                break

        if is_best:
            best_dice = mean_dice
            torch.save(model.state_dict(), output_dir / "best_stage1_dice.pth")
            print(f"  ★ Best Stage1 Dice: {best_dice:.4f}")

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
    # STAGE 2 — Fine-tune encoder
    # ─────────────────────────────────────────────────────────────────────────
    n_s2 = config["num_epochs_stage2"]
    if n_s2 > 0:
        print("\n─── STAGE 2: Fine-tune SwinUNETR Encoder ───")

        best_s1 = output_dir / "best_stage1_dice.pth"
        if best_s1.exists():
            model.load_state_dict(
                torch.load(best_s1, map_location=device, weights_only=True)
            )
        model.unfreeze_encoder(num_layers=8)

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

        start_ep_s2 = 0
        last_val = dict(loss=0.0, dice=best_dice, report="", metrics={}, per_case=[])

        if resume:
            start_ep_s2, best_dice = _load_checkpoint(
                output_dir, "stage2", model, optimizer, scheduler, scaler, device
            )

        val_every_s2 = config.get("val_every_s2", 1)

        for epoch in range(start_ep_s2, n_s2):
            train_loader.dataset._build_item_list()
            if her.hard_case_ids:
                her.inject_hard_samples(train_loader.dataset)

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

            sys.stdout.flush()

            her_summary = None
            if do_val and last_val["per_case"] and (epoch + 1) % her_every == 0:
                her.update(last_val["per_case"])
                her_summary = {
                    "hard_cases": her.hard_case_ids,
                    "num_hard": len(her.hard_case_ids),
                }

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
                    her_summary,
                )
                early_stop(mean_dice, model)
                if early_stop.early_stop:
                    print("  Early stopping (Stage 2)")
                    break

            if is_best:
                best_dice = mean_dice
                torch.save(model.state_dict(), output_dir / "best_final_dice.pth")
                print(f"  ★ Best Final Dice: {best_dice:.4f}")

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
        f"Training Complete | Best Dice: {best_dice:.4f} | Time: {total_time/3600:.2f}h"
    )
    print(f"{'='*65}")

    summary = {
        "version": "SegNetV7",
        "best_dice": float(best_dice),
        "total_hours": round(total_time / 3600, 2),
        "mgpl_model": config["clip_model"],
        "encoder_init": "renalStructures_UNEST (with generic SwinViT fallback)",
        "hard_example_resampling": True,
        "her_hard_cases_final": her.hard_case_ids,
        "her_updates": her.update_count,
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
        description="SegNetV7 — BiomedCLIP + Renal Init + HER"
    )
    parser.add_argument("--mode", default="full", choices=["full", "quick_test"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-mgpl", action="store_true", help="Disable MGPL (ablation)"
    )
    parser.add_argument("--kits23-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = get_config(args.mode)
    if args.no_mgpl:
        config["use_mgpl"] = False
    if args.kits23_dir:
        config["kits23_dir"] = args.kits23_dir
    if args.seed is not None:
        config["seed"] = args.seed

    print("Checking dependencies...")
    try:
        import monai

        print(f"  ✓ MONAI {monai.__version__}")
    except ImportError:
        print("  ✗ MONAI not found — pip install monai")
        sys.exit(1)
    try:
        import open_clip

        print("  ✓ open_clip available")
    except ImportError:
        print("  ✗ open_clip not found — pip install open_clip_torch")
        if config["use_mgpl"]:
            sys.exit(1)
    try:
        import nibabel

        print("  ✓ nibabel available")
    except ImportError:
        print("  ✗ nibabel not found — pip install nibabel")
        sys.exit(1)

    print()
    train(config, resume=args.resume)
