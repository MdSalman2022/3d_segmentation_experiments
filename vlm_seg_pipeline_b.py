"""
Pipeline B: VLM Reasoning Chain + SAM2 — Multi-Step 3D Segmentation
=====================================================================

Architecture:
    CT Volume → VLM Anatomy Assessment → VLM Grounded Localization
    → SAM2 Segmentation → VLM Self-Verification → Final Mask

This pipeline uses a strong VLM (InternVL2.5 / Qwen2.5-VL / Molmo / Phi-4)
for multi-step reasoning about each CT volume, then SAM2 for precise masks.

Key difference from Pipeline A:
    - Pipeline A: Florence-2 is a specialist detector (fast, small, no reasoning)
    - Pipeline B: Large VLM does multi-step clinical reasoning, catches edge cases

Supported VLM Backends:
    1. Ollama (local server — recommended, easiest setup)
    2. HuggingFace Transformers (direct GPU loading)
    3. OpenAI-compatible API (cloud or local vLLM/TGI server)

Supported VLMs (any with vision + text capabilities):
    - InternVL2.5-8B (strong reasoning + grounding)
    - Qwen2.5-VL-7B (native bbox grounding)
    - Molmo-7B (point grounding — best for SAM2 point prompts)
    - Phi-4-Multimodal (efficient, high quality)
    - LLaVA-OneVision-7B
    - DeepSeek-VL2
    - Any Ollama vision model

Usage:
    # Using Ollama (recommended)
    python vlm_seg_pipeline_b.py --mode test --backend ollama --model llava:13b
    python vlm_seg_pipeline_b.py --mode full --backend ollama --model internvl2.5:8b

    # Using HuggingFace Transformers
    python vlm_seg_pipeline_b.py --mode test --backend transformers \
        --model "OpenGVLab/InternVL2_5-8B"

    # Using Qwen2.5-VL via Ollama
    python vlm_seg_pipeline_b.py --mode full --backend ollama --model qwen2.5-vl:7b

    # Quick test with 5 cases
    python vlm_seg_pipeline_b.py --mode test --max-cases 5
"""

import os
import io
import re
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Paths
    "kits23_dir": "./kits23/dataset",
    "output_dir": "./output/vlm_pipeline_b_reasoning_chain",
    # VLM
    "vlm_backend": "ollama",  # "ollama", "transformers", "openai"
    "vlm_model": "internvl2.5:8b",  # Model name for the chosen backend
    "vlm_api_url": "http://localhost:11434",
    "vlm_temperature": 0.1,
    "vlm_max_tokens": 2048,
    # SAM2
    "sam2_checkpoint": "facebook/sam2-hiera-large",
    "sam2_device": "cuda",
    # Pipeline
    "num_assessment_slices": 5,  # Slices for initial anatomy assessment
    "num_localization_slices": 10,  # Slices for detailed localization
    "enable_self_verification": True,
    "max_verification_rounds": 2,
    # CT windowing
    "hu_min": -175,
    "hu_max": 250,
}


# ============================================================================
# VLM CLIENT — Unified interface for all backends
# ============================================================================


class VLMClient:
    """
    Unified client for calling Vision-Language Models.
    Supports Ollama, HuggingFace Transformers, and OpenAI-compatible APIs.
    """

    def __init__(
        self,
        backend: str,
        model: str,
        api_url: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.backend = backend
        self.model = model
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._hf_model = None
        self._hf_processor = None

        print(f"✓ VLM Client: {backend}:{model}")

    def query(
        self,
        prompt: str,
        images: Optional[List[np.ndarray]] = None,
        images_b64: Optional[List[str]] = None,
    ) -> str:
        """
        Send a query to the VLM with optional images.

        Args:
            prompt: Text prompt
            images: List of numpy arrays (H, W, 3) uint8
            images_b64: List of base64-encoded PNG strings (alternative)

        Returns:
            VLM text response
        """
        # Convert images to base64 if needed
        if images is not None and images_b64 is None:
            images_b64 = [self._numpy_to_b64(img) for img in images]

        if self.backend == "ollama":
            return self._query_ollama(prompt, images_b64)
        elif self.backend == "transformers":
            return self._query_transformers(prompt, images)
        elif self.backend == "openai":
            return self._query_openai(prompt, images_b64)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _query_ollama(self, prompt: str, images_b64: Optional[List[str]]) -> str:
        """Query Ollama API."""
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if images_b64:
            payload["images"] = images_b64

        try:
            resp = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            print(f"  ⚠ Ollama error: {e}")
            return ""

    def _query_transformers(
        self, prompt: str, images: Optional[List[np.ndarray]]
    ) -> str:
        """Query HuggingFace Transformers model (loaded locally)."""
        if self._hf_model is None:
            self._load_hf_model()

        from PIL import Image as PILImage

        pil_images = []
        if images is not None:
            for img in images:
                pil_images.append(PILImage.fromarray(img))

        try:
            # Build input (model-specific formatting)
            if "internvl" in self.model.lower():
                return self._query_internvl(prompt, pil_images)
            elif "qwen" in self.model.lower():
                return self._query_qwen_vl(prompt, pil_images)
            else:
                return self._query_generic_hf(prompt, pil_images)
        except Exception as e:
            print(f"  ⚠ Transformers error: {e}")
            return ""

    def _load_hf_model(self):
        """Load HuggingFace model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

        print(f"  Loading HF model: {self.model}...")

        try:
            self._hf_processor = AutoProcessor.from_pretrained(
                self.model, trust_remote_code=True
            )
        except Exception:
            self._hf_processor = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_8bit=True,  # Quantize to fit in memory
        )
        self._hf_model.eval()
        print(f"  ✓ Model loaded (8-bit quantized)")

    def _query_internvl(self, prompt: str, images: List) -> str:
        """InternVL2.5-specific inference."""
        if images:
            # InternVL expects <image> token in prompt
            img_tokens = "".join(["<image>\n"] * len(images))
            full_prompt = img_tokens + prompt
        else:
            full_prompt = prompt

        inputs = self._hf_processor(
            text=full_prompt, images=images if images else None, return_tensors="pt"
        ).to(self._hf_model.device)

        with torch.no_grad():
            out = self._hf_model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=False,
            )
        return self._hf_processor.decode(out[0], skip_special_tokens=True)

    def _query_qwen_vl(self, prompt: str, images: List) -> str:
        """Qwen2.5-VL-specific inference."""
        messages = [{"role": "user", "content": []}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": prompt})

        text = self._hf_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._hf_processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self._hf_model.device)

        with torch.no_grad():
            out = self._hf_model.generate(**inputs, max_new_tokens=self.max_tokens)
        return self._hf_processor.decode(out[0], skip_special_tokens=True)

    def _query_generic_hf(self, prompt: str, images: List) -> str:
        """Generic HuggingFace VLM inference."""
        inputs = self._hf_processor(
            text=prompt, images=images[0] if images else None, return_tensors="pt"
        ).to(self._hf_model.device)

        with torch.no_grad():
            out = self._hf_model.generate(**inputs, max_new_tokens=self.max_tokens)
        return self._hf_processor.decode(out[0], skip_special_tokens=True)

    def _query_openai(self, prompt: str, images_b64: Optional[List[str]]) -> str:
        """Query OpenAI-compatible API."""
        import requests

        content = [{"type": "text", "text": prompt}]
        if images_b64:
            for b64 in images_b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )

        try:
            resp = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  ⚠ OpenAI API error: {e}")
            return ""

    @staticmethod
    def _numpy_to_b64(image: np.ndarray) -> str:
        """Convert numpy image to base64 PNG string."""
        from PIL import Image as PILImage

        pil = PILImage.fromarray(image)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================================
# SAM2 SEGMENTER (shared with Pipeline A)
# ============================================================================


class SAM2Segmenter:
    """SAM2 segmentation with box and point prompts."""

    def __init__(
        self, checkpoint: str = "facebook/sam2-hiera-large", device: str = "cuda"
    ):
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self._predictor = None
        print(f"✓ SAM2 configured: {checkpoint}")

    def _load(self):
        if self._predictor is not None:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(
                "sam2_hiera_l.yaml", self.checkpoint, device=str(self.device)
            )
            self._predictor = SAM2ImagePredictor(model)
            print(f"  ✓ SAM2 loaded")
        except ImportError:
            try:
                from segment_anything import sam_model_registry, SamPredictor

                sam = sam_model_registry["vit_l"](checkpoint=self.checkpoint)
                sam.to(self.device)
                self._predictor = SamPredictor(sam)
                print(f"  ✓ SAM1 fallback loaded")
            except ImportError:
                print("  ⚠ No SAM available — using box-fill fallback")
                self._predictor = "fallback"

    @torch.no_grad()
    def segment_box(self, image_rgb: np.ndarray, box: List[float]) -> np.ndarray:
        """Segment with a bounding box prompt. Returns (H, W) bool mask."""
        self._load()
        if self._predictor == "fallback":
            return self._box_fill(image_rgb, box)

        self._predictor.set_image(image_rgb)
        masks, scores, _ = self._predictor.predict(
            box=np.array(box, dtype=np.float32),
            multimask_output=True,
        )
        return masks[np.argmax(scores)].astype(bool)

    @torch.no_grad()
    def segment_points(
        self, image_rgb: np.ndarray, points: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Segment with point prompts. Returns (H, W) bool mask."""
        self._load()
        if self._predictor == "fallback":
            return np.zeros(image_rgb.shape[:2], dtype=bool)

        self._predictor.set_image(image_rgb)
        masks, scores, _ = self._predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        return masks[np.argmax(scores)].astype(bool)

    @staticmethod
    def _box_fill(image: np.ndarray, box: List[float]) -> np.ndarray:
        """Fallback: fill bounding box."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = [int(b) for b in box]
        mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True
        return mask


# ============================================================================
# REASONING CHAIN PIPELINE
# ============================================================================


class ReasoningChainPipeline:
    """
    Multi-step VLM reasoning pipeline for 3D medical segmentation.

    Step 1: ANATOMY ASSESSMENT
        → VLM analyzes representative slices to identify structures

    Step 2: GROUNDED LOCALIZATION
        → VLM provides bounding box coordinates for each structure per slice

    Step 3: SAM2 SEGMENTATION
        → SAM2 produces precise masks from VLM box/point prompts

    Step 4: SELF-VERIFICATION (optional)
        → VLM reviews the segmentation overlay and suggests corrections
    """

    CLASS_NAMES = {0: "background", 1: "kidney", 2: "tumor", 3: "cyst"}
    CLASS_LABEL_MAP = {
        "kidney": 1,
        "renal": 1,
        "kidney parenchyma": 1,
        "tumor": 2,
        "mass": 2,
        "renal cell carcinoma": 2,
        "rcc": 2,
        "neoplasm": 2,
        "malignant": 2,
        "cyst": 3,
        "renal cyst": 3,
        "simple cyst": 3,
    }

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # VLM and SAM2
        self.vlm = VLMClient(
            backend=config["vlm_backend"],
            model=config["vlm_model"],
            api_url=config.get("vlm_api_url", ""),
            temperature=config["vlm_temperature"],
            max_tokens=config["vlm_max_tokens"],
        )

        try:
            self.sam2 = SAM2Segmenter(
                checkpoint=config["sam2_checkpoint"],
                device=config["sam2_device"],
            )
        except Exception:
            self.sam2 = SAM2Segmenter.__new__(SAM2Segmenter)
            self.sam2._predictor = "fallback"

    def process_volume(
        self,
        image_path: str,
        label_path: Optional[str] = None,
        case_id: str = "unknown",
    ) -> Dict:
        """Full pipeline for one CT volume."""
        print(f"\n{'━' * 60}")
        print(f"PIPELINE B — {case_id}")
        print(f"{'━' * 60}")

        # ── Load volume ──
        img_nii = nib.load(image_path)
        img_data = img_nii.get_fdata().astype(np.float32)
        gt_data = None
        if label_path and Path(label_path).exists():
            gt_data = nib.load(label_path).get_fdata().astype(np.int64)

        # CT windowing
        hu_min, hu_max = self.config["hu_min"], self.config["hu_max"]
        img_norm = np.clip(img_data, hu_min, hu_max)
        img_norm = (img_norm - hu_min) / (hu_max - hu_min)
        D, H, W = img_norm.shape
        print(f"  Volume: {D}×{H}×{W}")

        t_start = time.time()

        # ══════════════════════════════════════════════════
        # STEP 1: ANATOMY ASSESSMENT
        # ══════════════════════════════════════════════════
        print("\n  ── Step 1: Anatomy Assessment ──")
        assessment = self._step1_anatomy_assessment(img_norm)
        print(f"  Assessment: {json.dumps(assessment, indent=2)[:300]}...")

        # ══════════════════════════════════════════════════
        # STEP 2: GROUNDED LOCALIZATION
        # ══════════════════════════════════════════════════
        print("\n  ── Step 2: Grounded Localization ──")
        localizations = self._step2_localization(img_norm, assessment)
        n_locs = sum(len(v) for v in localizations.values())
        print(f"  Got {n_locs} localizations across {len(localizations)} slices")

        # ══════════════════════════════════════════════════
        # STEP 3: SAM2 SEGMENTATION
        # ══════════════════════════════════════════════════
        print("\n  ── Step 3: SAM2 Segmentation ──")
        segmentation = self._step3_sam2_segmentation(img_norm, localizations)
        print(
            f"  Segmented voxels: "
            f"kidney={int((segmentation == 1).sum()):,}, "
            f"tumor={int((segmentation == 2).sum()):,}, "
            f"cyst={int((segmentation == 3).sum()):,}"
        )

        # ══════════════════════════════════════════════════
        # STEP 4: SELF-VERIFICATION
        # ══════════════════════════════════════════════════
        if self.config["enable_self_verification"]:
            print("\n  ── Step 4: Self-Verification ──")
            segmentation = self._step4_self_verification(
                img_norm, segmentation, assessment
            )

        # ── Post-process ──
        segmentation = self._postprocess(segmentation)

        t_total = time.time() - t_start
        print(f"\n  Total time: {t_total:.1f}s")

        # ── Evaluate ──
        results = {
            "case_id": case_id,
            "volume_shape": [D, H, W],
            "total_time_s": round(t_total, 2),
            "assessment": assessment,
            "num_localizations": n_locs,
        }

        if gt_data is not None:
            metrics = self._compute_metrics(segmentation, gt_data)
            results["metrics"] = metrics
            print(
                f"  Dice → Kidney: {metrics['dice'][1]:.4f} | "
                f"Tumor: {metrics['dice'][2]:.4f} | "
                f"Cyst: {metrics['dice'][3]:.4f}"
            )

        # Save prediction
        pred_nii = nib.Nifti1Image(segmentation.astype(np.int16), img_nii.affine)
        nib.save(pred_nii, self.output_dir / f"{case_id}_pred.nii.gz")

        # Save reasoning log
        with open(self.output_dir / f"{case_id}_reasoning.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: ANATOMY ASSESSMENT
    # ──────────────────────────────────────────────────────────────────────

    def _step1_anatomy_assessment(self, volume: np.ndarray) -> Dict:
        """
        Show representative slices to VLM for anatomy overview.
        The VLM identifies what structures are present.
        """
        D = volume.shape[0]
        n_slices = self.config["num_assessment_slices"]

        # Pick evenly-spaced slices (middle portion where kidneys are)
        start = D // 4
        end = 3 * D // 4
        slice_indices = np.linspace(start, end, n_slices, dtype=int)

        # Render slices
        images = [self._slice_to_rgb(volume[z]) for z in slice_indices]

        prompt = f"""You are an expert radiologist analyzing a kidney CT scan.
I am showing you {n_slices} representative axial slices from this volume.
The slices are from positions {slice_indices.tolist()} out of {D} total slices.

Analyze what you see and respond ONLY with valid JSON:
{{
  "kidneys_visible": true/false,
  "num_kidneys": 1 or 2,
  "kidney_sides": ["left", "right"] or ["left"] or ["right"],
  "has_tumor": true/false,
  "tumor_description": "brief description if present, 'none' otherwise",
  "tumor_approximate_size": "small (<2cm) / medium (2-5cm) / large (>5cm) / none",
  "has_cyst": true/false,
  "cyst_description": "brief description if present, 'none' otherwise",
  "slice_range_with_pathology": [start_slice, end_slice],
  "confidence": "low/medium/high"
}}"""

        response = self.vlm.query(prompt, images=images)
        return self._parse_json(
            response,
            default={
                "kidneys_visible": True,
                "num_kidneys": 2,
                "has_tumor": True,
                "has_cyst": False,
                "slice_range_with_pathology": [D // 4, 3 * D // 4],
                "confidence": "low",
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: GROUNDED LOCALIZATION
    # ──────────────────────────────────────────────────────────────────────

    def _step2_localization(
        self, volume: np.ndarray, assessment: Dict
    ) -> Dict[int, List[Dict]]:
        """
        For each slice in the pathology range, ask VLM for bounding boxes.
        Returns: {slice_idx: [{"class": int, "bbox": [x1,y1,x2,y2]}]}
        """
        D, H, W = volume.shape
        n_loc_slices = self.config["num_localization_slices"]

        # Determine slice range from assessment
        path_range = assessment.get("slice_range_with_pathology", [D // 4, 3 * D // 4])
        if not isinstance(path_range, list) or len(path_range) != 2:
            path_range = [D // 4, 3 * D // 4]

        start_z = max(0, int(path_range[0]))
        end_z = min(D - 1, int(path_range[1]))

        # Sample slices within the pathology range
        slice_indices = np.linspace(start_z, end_z, n_loc_slices, dtype=int)

        localizations = {}

        for z in tqdm(slice_indices, desc="  Localization", leave=False):
            image = self._slice_to_rgb(volume[z])

            # Structures to localize
            structures = ["kidney"]
            if assessment.get("has_tumor"):
                structures.append("kidney tumor")
            if assessment.get("has_cyst"):
                structures.append("kidney cyst")

            prompt = f"""This is axial CT slice #{z} of a kidney scan.
Image dimensions: {W} pixels wide, {H} pixels tall.
Pixel coordinates: (0,0) is top-left, ({W},{H}) is bottom-right.

Locate each visible structure and provide bounding box coordinates.
Respond ONLY with valid JSON:
{{
  "structures": [
    {{"name": "kidney", "bbox": [x1, y1, x2, y2], "side": "left/right", "confidence": 0.0-1.0}},
    {{"name": "kidney tumor", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}},
    {{"name": "kidney cyst", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}}
  ]
}}

Only include structures you can actually see in this slice.
Structures to look for: {structures}
Bounding box format: [x_left, y_top, x_right, y_bottom] in pixel coordinates."""

            response = self.vlm.query(prompt, images=[image])
            parsed = self._parse_json(response, default={"structures": []})

            # Convert to internal format
            slice_locs = []
            for s in parsed.get("structures", []):
                name = s.get("name", "").lower()
                bbox = s.get("bbox", [])
                conf = s.get("confidence", 0.5)

                # Map name to class ID
                cls_id = None
                for key, cid in self.CLASS_LABEL_MAP.items():
                    if key in name:
                        cls_id = cid
                        break

                if cls_id is not None and len(bbox) == 4 and conf > 0.3:
                    # Validate bbox coordinates
                    bbox = [
                        max(0, min(W, float(bbox[0]))),
                        max(0, min(H, float(bbox[1]))),
                        max(0, min(W, float(bbox[2]))),
                        max(0, min(H, float(bbox[3]))),
                    ]
                    # Ensure valid box
                    if bbox[2] > bbox[0] + 5 and bbox[3] > bbox[1] + 5:
                        slice_locs.append(
                            {
                                "class": cls_id,
                                "bbox": bbox,
                                "confidence": conf,
                                "name": name,
                            }
                        )

            if slice_locs:
                localizations[z] = slice_locs

        return localizations

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: SAM2 SEGMENTATION
    # ──────────────────────────────────────────────────────────────────────

    def _step3_sam2_segmentation(
        self,
        volume: np.ndarray,
        localizations: Dict[int, List[Dict]],
    ) -> np.ndarray:
        """
        Use SAM2 to produce precise masks from VLM bounding boxes.
        Propagates detections to neighboring slices.
        """
        D, H, W = volume.shape
        segmentation = np.zeros((D, H, W), dtype=np.int64)

        if not localizations:
            print("  ⚠ No localizations to segment")
            return segmentation

        # For each slice within the detected range
        loc_slices = sorted(localizations.keys())
        min_z, max_z = loc_slices[0], loc_slices[-1]

        for z in tqdm(
            range(max(0, min_z - 5), min(D, max_z + 5)), desc="  SAM2", leave=False
        ):
            # Find nearest localization
            nearest_z = min(loc_slices, key=lambda lz: abs(lz - z))

            # Only propagate within a reasonable distance
            if abs(z - nearest_z) > 10:
                continue

            locs = localizations[nearest_z]
            image_rgb = self._slice_to_rgb(volume[z])

            for loc in locs:
                cls_id = loc["class"]
                bbox = loc["bbox"]

                # Adjust box slightly for non-detected slices
                # (farther from detected slice → slightly larger box)
                dist = abs(z - nearest_z)
                if dist > 0:
                    expand = dist * 2  # Expand by 2 pixels per slice distance
                    bbox = [
                        max(0, bbox[0] - expand),
                        max(0, bbox[1] - expand),
                        min(W, bbox[2] + expand),
                        min(H, bbox[3] + expand),
                    ]

                    # Don't propagate small structures too far
                    box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if box_area < 100 and dist > 3:
                        continue

                # SAM2 segmentation
                mask = self.sam2.segment_box(image_rgb, bbox)

                # Write to segmentation (higher class overwrites)
                if mask is not None:
                    segmentation[z][mask] = cls_id

        return segmentation

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: SELF-VERIFICATION
    # ──────────────────────────────────────────────────────────────────────

    def _step4_self_verification(
        self,
        volume: np.ndarray,
        segmentation: np.ndarray,
        assessment: Dict,
    ) -> np.ndarray:
        """
        Show the segmentation result to VLM for self-verification.
        VLM checks for obvious errors and suggests corrections.
        """
        D, H, W = volume.shape

        for verification_round in range(self.config["max_verification_rounds"]):
            # Pick representative slices that have segmentation
            seg_slices = [z for z in range(D) if segmentation[z].max() > 0]
            if not seg_slices:
                print("    No segmentation to verify")
                break

            # Pick 3 slices: start, middle, end of segmented region
            check_indices = [
                seg_slices[0],
                seg_slices[len(seg_slices) // 2],
                seg_slices[-1],
            ]

            # Render overlays
            overlay_images = []
            for z in check_indices:
                overlay = self._render_overlay(volume[z], segmentation[z])
                overlay_images.append(overlay)

            prompt = f"""You are verifying a kidney CT segmentation result.
I'm showing you 3 slices with segmentation overlays:
  Green = Kidney, Red = Tumor, Blue = Cyst

Assessment from initial analysis:
  Kidneys: {assessment.get('num_kidneys', '?')}, 
  Tumor: {assessment.get('has_tumor', '?')},
  Cyst: {assessment.get('has_cyst', '?')}

Current segmentation statistics:
  Kidney voxels: {int((segmentation == 1).sum()):,}
  Tumor voxels: {int((segmentation == 2).sum()):,}
  Cyst voxels: {int((segmentation == 3).sum()):,}

Check for these errors and respond ONLY with valid JSON:
{{
  "is_correct": true/false,
  "errors": [
    {{
      "type": "missed_structure/over_segmented/wrong_class/mislocated",
      "structure": "kidney/tumor/cyst",
      "slice": <slice_number>,
      "description": "brief error description",
      "correction_bbox": [x1, y1, x2, y2] or null
    }}
  ],
  "overall_quality": "good/acceptable/poor",
  "needs_refinement": true/false
}}"""

            response = self.vlm.query(prompt, images=overlay_images)
            verification = self._parse_json(
                response,
                default={
                    "is_correct": True,
                    "errors": [],
                    "overall_quality": "acceptable",
                    "needs_refinement": False,
                },
            )

            print(
                f"    Round {verification_round + 1}: "
                f"quality={verification.get('overall_quality', '?')}, "
                f"errors={len(verification.get('errors', []))}"
            )

            if verification.get("is_correct") or not verification.get(
                "needs_refinement"
            ):
                break

            # Apply corrections
            for error in verification.get("errors", []):
                correction_bbox = error.get("correction_bbox")
                err_type = error.get("type", "")
                structure = error.get("structure", "")

                if correction_bbox and len(correction_bbox) == 4:
                    cls_id = self.CLASS_LABEL_MAP.get(structure)
                    if cls_id and err_type == "missed_structure":
                        # Try to segment the missed region
                        err_slice = error.get("slice")
                        if isinstance(err_slice, int) and 0 <= err_slice < D:
                            image_rgb = self._slice_to_rgb(volume[err_slice])
                            mask = self.sam2.segment_box(image_rgb, correction_bbox)
                            if mask is not None:
                                segmentation[err_slice][mask] = cls_id
                                print(
                                    f"    → Fixed: added {structure} at slice {err_slice}"
                                )

        return segmentation

    # ──────────────────────────────────────────────────────────────────────
    # POST-PROCESSING
    # ──────────────────────────────────────────────────────────────────────

    def _postprocess(self, seg: np.ndarray) -> np.ndarray:
        """Remove small components, enforce spatial consistency."""
        from scipy import ndimage

        for cls_id in [1, 2, 3]:
            cls_mask = seg == cls_id
            if cls_mask.sum() == 0:
                continue

            labeled, n_comp = ndimage.label(cls_mask)
            min_sizes = {1: 5000, 2: 200, 3: 200}
            for comp_id in range(1, n_comp + 1):
                if (labeled == comp_id).sum() < min_sizes.get(cls_id, 200):
                    seg[labeled == comp_id] = 0

        # Ensure tumor/cyst inside dilated kidney
        kidney_mask = seg == 1
        if kidney_mask.sum() > 0:
            struct = ndimage.generate_binary_structure(3, 2)
            kidney_dilated = ndimage.binary_dilation(kidney_mask, struct, iterations=5)
            for cls_id in [2, 3]:
                outside = (seg == cls_id) & ~kidney_dilated
                seg[outside] = 0

        return seg

    # ──────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _slice_to_rgb(ct_slice: np.ndarray) -> np.ndarray:
        """Convert normalized CT slice to (H, W, 3) uint8 RGB."""
        s = (ct_slice * 255).clip(0, 255).astype(np.uint8)
        return np.stack([s, s, s], axis=-1)

    def _render_overlay(
        self, ct_slice: np.ndarray, seg_slice: np.ndarray
    ) -> np.ndarray:
        """Render segmentation overlay on CT slice."""
        rgb = self._slice_to_rgb(ct_slice)
        colors = {1: [0, 255, 0], 2: [255, 0, 0], 3: [0, 0, 255]}

        for cls_id, color in colors.items():
            mask = seg_slice == cls_id
            if mask.any():
                alpha = 0.4
                for c in range(3):
                    rgb[:, :, c][mask] = (
                        rgb[:, :, c][mask] * (1 - alpha) + color[c] * alpha
                    ).astype(np.uint8)
        return rgb

    @staticmethod
    def _parse_json(text: str, default: dict) -> dict:
        """Parse JSON from VLM response, with fallback."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        return default

    def _compute_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict:
        """Dice, IoU, precision, recall per class."""
        metrics = {"dice": {}, "iou": {}, "precision": {}, "recall": {}}
        for c in range(4):
            p, g = (pred == c), (gt == c)
            tp = (p & g).sum()
            fp = (p & ~g).sum()
            fn = (~p & g).sum()
            metrics["dice"][c] = round(float(2 * tp / (2 * tp + fp + fn + 1e-8)), 5)
            metrics["iou"][c] = round(float(tp / (tp + fp + fn + 1e-8)), 5)
            metrics["precision"][c] = round(float(tp / (tp + fp + 1e-8)), 5)
            metrics["recall"][c] = round(float(tp / (tp + fn + 1e-8)), 5)
        return metrics

    # ──────────────────────────────────────────────────────────────────────
    # MAIN EVALUATION
    # ──────────────────────────────────────────────────────────────────────

    def run_evaluation(self, max_cases: Optional[int] = None):
        """Run pipeline on KiTS23 and compute aggregate metrics."""
        data_dir = Path(self.config["kits23_dir"])
        cases = sorted(
            [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")]
        )

        if max_cases:
            cases = cases[:max_cases]

        # Same val split as V4/V5
        split = int(len(cases) * 0.2)
        test_cases = cases[:split]
        print(f"\nEvaluating {len(test_cases)} cases")

        all_results = []
        for case_dir in test_cases:
            img_path = str(case_dir / "imaging.nii.gz")
            lbl_path = str(case_dir / "segmentation.nii.gz")

            if not Path(img_path).exists():
                continue

            result = self.process_volume(img_path, lbl_path, case_dir.name)
            all_results.append(result)

        # Aggregate
        if all_results and "metrics" in all_results[0]:
            all_dice = {c: [] for c in range(4)}
            for r in all_results:
                if "metrics" not in r:
                    continue
                for c in range(4):
                    all_dice[c].append(r["metrics"]["dice"].get(c, 0))

            mean_dice = {c: float(np.mean(v)) for c, v in all_dice.items() if v}
            mean_fg = float(np.mean([mean_dice.get(c, 0) for c in [1, 2, 3]]))

            print(f"\n{'═' * 60}")
            print(f"PIPELINE B RESULTS — VLM Reasoning Chain + SAM2")
            print(f"VLM: {self.config['vlm_backend']}:{self.config['vlm_model']}")
            print(f"{'═' * 60}")
            for c, name in self.CLASS_NAMES.items():
                if c == 0:
                    continue
                print(f"  {name:<10} Dice: {mean_dice.get(c, 0):.4f}")
            print(f"  {'Mean FG':<10} Dice: {mean_fg:.4f}")
            print(f"{'═' * 60}")

            summary = {
                "pipeline": f"VLM Reasoning Chain ({self.config['vlm_model']}) + SAM2",
                "num_cases": len(all_results),
                "aggregate": {"mean_dice": mean_dice, "mean_fg_dice": mean_fg},
                "per_case": all_results,
                "config": {k: str(v) for k, v in self.config.items()},
            }
            with open(self.output_dir / "results.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

        return all_results


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline B: VLM Reasoning Chain + SAM2"
    )
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama", "transformers", "openai"],
    )
    parser.add_argument(
        "--model", type=str, default="internvl2.5:8b", help="VLM model name"
    )
    parser.add_argument("--api-url", type=str, default="http://localhost:11434")
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument(
        "--no-verify", action="store_true", help="Disable self-verification step"
    )
    parser.add_argument(
        "--loc-slices", type=int, default=None, help="Number of slices for localization"
    )
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["vlm_backend"] = args.backend
    config["vlm_model"] = args.model
    config["vlm_api_url"] = args.api_url

    if args.sam_checkpoint:
        config["sam2_checkpoint"] = args.sam_checkpoint
    if args.no_verify:
        config["enable_self_verification"] = False
    if args.loc_slices:
        config["num_localization_slices"] = args.loc_slices

    max_cases = args.max_cases
    if max_cases is None:
        max_cases = 5 if args.mode == "test" else None

    pipeline = ReasoningChainPipeline(config)
    pipeline.run_evaluation(max_cases=max_cases)


if __name__ == "__main__":
    main()
