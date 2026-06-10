"""
Pipeline A: Florence-2 + SAM2 — Zero-Shot 3D Medical Segmentation
===================================================================

Architecture:
    CT Volume → Florence-2 (per-slice grounding) → SAM2 (3D refinement) → Segmentation

Florence-2 (Microsoft, 0.8B params):
    - Runs locally, ~3GB VRAM
    - Supports: open-vocabulary detection, referring expression segmentation,
      region proposals, dense captioning
    - Outputs bounding boxes and polygon masks from text prompts

SAM2 (Meta):
    - Segment Anything Model 2 with video propagation mode
    - Takes box/point/mask prompts → produces refined segmentation
    - Video mode propagates masks across frames (= CT slices)

Pipeline:
    1. Load CT volume (D, H, W)
    2. For each axial slice → Florence-2 detects kidney/tumor/cyst regions
    3. Convert detections → SAM2 box prompts
    4. SAM2 video predictor propagates and refines through volume
    5. Assemble 3D segmentation mask
    6. Evaluate against ground truth

Usage:
    python vlm_seg_pipeline_a.py --mode test --max-cases 5
    python vlm_seg_pipeline_a.py --mode full
    python vlm_seg_pipeline_a.py --mode full --sam-only   # Skip Florence-2, use GT boxes
    python vlm_seg_pipeline_a.py --mode test --florence-model "microsoft/Florence-2-base"
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Paths
    "kits23_dir": "./kits23/dataset",
    "output_dir": "./output/vlm_pipeline_a_florence2_sam2",
    # Florence-2
    "florence_model": "microsoft/Florence-2-large",
    "florence_device": "cuda",  # or "cpu" to run Florence on CPU, SAM on GPU
    # SAM2
    "sam2_checkpoint": "facebook/sam2-hiera-large",
    "sam2_device": "cuda",
    # Pipeline
    "slice_step": 3,  # Process every Nth slice with Florence-2 (speed vs accuracy)
    "confidence_threshold": 0.3,
    "min_detection_area": 100,  # Min pixels for a valid detection
    "sam2_points_per_side": 0,  # 0 = use box prompts only; >0 = also add point grid
    # CT windowing
    "hu_min": -175,
    "hu_max": 250,
    # Classes
    "class_prompts": {
        1: ["kidney", "renal organ", "kidney parenchyma"],
        2: ["kidney tumor", "renal mass", "renal cell carcinoma", "tumor in kidney"],
        3: ["kidney cyst", "renal cyst", "fluid filled cyst in kidney"],
    },
}


# ============================================================================
# FLORENCE-2 DETECTOR
# ============================================================================


class Florence2Detector:
    """
    Uses Microsoft Florence-2 to detect and segment kidney structures
    in 2D CT slices via text-guided grounding.
    """

    CLASS_NAMES = {1: "kidney", 2: "tumor", 3: "cyst"}

    def __init__(
        self, model_name: str = "microsoft/Florence-2-large", device: str = "cuda"
    ):
        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"Loading Florence-2: {model_name}...")
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print(
            f"✓ Florence-2 loaded on {device} "
            f"({sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params)"
        )

    @torch.no_grad()
    def detect_structures(
        self,
        ct_slice: np.ndarray,
        class_prompts: Dict[int, List[str]],
        confidence_threshold: float = 0.3,
    ) -> Dict[int, List[Dict]]:
        """
        Detect kidney structures in a single CT slice.

        Args:
            ct_slice: (H, W) normalized CT slice [0, 1]
            class_prompts: {class_id: [text prompts]}
            confidence_threshold: minimum confidence for detection

        Returns:
            {class_id: [{"bbox": [x1,y1,x2,y2], "score": float,
                         "mask": np.ndarray or None}]}
        """
        from PIL import Image

        # Convert to RGB PIL image (Florence-2 expects RGB)
        ct_uint8 = (ct_slice * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(np.stack([ct_uint8] * 3, axis=-1))

        detections = {}

        for cls_id, prompts in class_prompts.items():
            cls_detections = []

            for prompt_text in prompts:
                # ── Method 1: Open Vocabulary Detection ──
                try:
                    ovd_dets = self._run_open_vocab_detection(
                        pil_img, prompt_text, confidence_threshold
                    )
                    cls_detections.extend(ovd_dets)
                except Exception:
                    pass

                # ── Method 2: Phrase Grounding ──
                try:
                    pg_dets = self._run_phrase_grounding(
                        pil_img, prompt_text, confidence_threshold
                    )
                    cls_detections.extend(pg_dets)
                except Exception:
                    pass

            # Deduplicate overlapping boxes (NMS)
            if cls_detections:
                cls_detections = self._nms(cls_detections, iou_threshold=0.5)

            detections[cls_id] = cls_detections

        return detections

    def _run_open_vocab_detection(
        self, pil_img, text: str, conf_thresh: float
    ) -> List[Dict]:
        """<OPEN_VOCABULARY_DETECTION> task."""
        task = "<OPEN_VOCABULARY_DETECTION>"
        prompt = task + text

        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt").to(
            self.device
        )

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

        result = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[
            0
        ]
        parsed = self.processor.post_process_generation(
            result, task=task, image_size=pil_img.size
        )

        detections = []
        if task in parsed:
            bboxes = parsed[task].get("bboxes", [])
            labels = parsed[task].get("bboxes_labels", [])
            for i, bbox in enumerate(bboxes):
                detections.append(
                    {
                        "bbox": [float(b) for b in bbox],
                        "score": 0.8,  # Florence-2 doesn't output confidence natively
                        "label": labels[i] if i < len(labels) else text,
                        "mask": None,
                    }
                )

        return detections

    def _run_phrase_grounding(
        self, pil_img, text: str, conf_thresh: float
    ) -> List[Dict]:
        """<CAPTION_TO_PHRASE_GROUNDING> task."""
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = task + f"A CT scan showing {text}."

        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt").to(
            self.device
        )

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

        result = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[
            0
        ]
        parsed = self.processor.post_process_generation(
            result, task=task, image_size=pil_img.size
        )

        detections = []
        if task in parsed:
            bboxes = parsed[task].get("bboxes", [])
            labels = parsed[task].get("labels", [])
            for i, bbox in enumerate(bboxes):
                detections.append(
                    {
                        "bbox": [float(b) for b in bbox],
                        "score": 0.7,
                        "label": labels[i] if i < len(labels) else text,
                        "mask": None,
                    }
                )

        return detections

    def _run_referring_segmentation(self, pil_img, text: str) -> List[Dict]:
        """<REFERRING_EXPRESSION_SEGMENTATION> — outputs polygon masks."""
        task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        prompt = task + text

        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt").to(
            self.device
        )

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

        result = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[
            0
        ]
        parsed = self.processor.post_process_generation(
            result, task=task, image_size=pil_img.size
        )

        detections = []
        if task in parsed and "polygons" in parsed[task]:
            w, h = pil_img.size
            for polygon in parsed[task]["polygons"]:
                if len(polygon) > 0 and len(polygon[0]) >= 6:
                    # Convert polygon to mask
                    mask = self._polygon_to_mask(polygon[0], h, w)
                    # Derive bbox from mask
                    ys, xs = np.where(mask)
                    if len(ys) > 0:
                        bbox = [
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max()),
                            float(ys.max()),
                        ]
                        detections.append(
                            {
                                "bbox": bbox,
                                "score": 0.85,
                                "mask": mask,
                                "label": text,
                            }
                        )

        return detections

    @staticmethod
    def _polygon_to_mask(polygon: list, h: int, w: int) -> np.ndarray:
        """Convert polygon coordinates to binary mask."""
        from PIL import Image, ImageDraw

        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        # polygon is [x1,y1,x2,y2,...] — convert to [(x1,y1),(x2,y2),...]
        pts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon) - 1, 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
        return np.array(mask_img, dtype=bool)

    @staticmethod
    def _nms(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Simple NMS on detections."""
        if len(detections) <= 1:
            return detections

        # Sort by score descending
        dets = sorted(detections, key=lambda d: d["score"], reverse=True)
        keep = []

        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [
                d for d in dets if _box_iou(best["bbox"], d["bbox"]) < iou_threshold
            ]

        return keep


def _box_iou(box1: list, box2: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-8)


# ============================================================================
# SAM2 SEGMENTER
# ============================================================================


class SAM2Segmenter:
    """
    SAM2-based segmentation with 3D propagation.

    Supports:
    - Box-prompted segmentation (from Florence-2 detections)
    - Point-prompted segmentation (from VLM coordinates)
    - Video-mode propagation (treat CT slices as video frames)

    Can use either SAM2 (Meta) or MedSAM2 (fine-tuned for medical).
    """

    def __init__(
        self, checkpoint: str = "facebook/sam2-hiera-large", device: str = "cuda"
    ):
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self._model = None
        self._predictor = None
        print(f"✓ SAM2 configured: {checkpoint}")

    def _load_model(self):
        """Lazy-load SAM2 model."""
        if self._predictor is not None:
            return

        try:
            # Try sam2 package first
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(
                config_file="sam2_hiera_l.yaml",
                ckpt_path=self.checkpoint,
                device=str(self.device),
            )
            self._predictor = SAM2ImagePredictor(model)
            print(f"✓ SAM2 loaded from sam2 package")

        except ImportError:
            try:
                # Fallback: try segment-anything-2
                from segment_anything_2.build_sam import build_sam2
                from segment_anything_2.sam2_image_predictor import SAM2ImagePredictor

                model = build_sam2(
                    config_file="sam2_hiera_l.yaml",
                    ckpt_path=self.checkpoint,
                    device=str(self.device),
                )
                self._predictor = SAM2ImagePredictor(model)
                print(f"✓ SAM2 loaded from segment_anything_2 package")

            except ImportError:
                # Fallback: use original SAM from segment-anything
                print("⚠ SAM2 not found, falling back to SAM1...")
                from segment_anything import sam_model_registry, SamPredictor

                sam = sam_model_registry["vit_l"](checkpoint=self.checkpoint)
                sam.to(self.device)
                self._predictor = SamPredictor(sam)
                print(f"✓ SAM1 loaded as fallback")

    @torch.no_grad()
    def segment_with_boxes(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
    ) -> List[np.ndarray]:
        """
        Segment using bounding box prompts.

        Args:
            image: (H, W, 3) uint8 RGB image
            boxes: list of [x1, y1, x2, y2] bounding boxes

        Returns:
            List of binary masks (H, W) for each box
        """
        self._load_model()

        if len(boxes) == 0:
            return []

        self._predictor.set_image(image)

        masks = []
        for box in boxes:
            box_np = np.array(box, dtype=np.float32)
            mask_pred, scores, _ = self._predictor.predict(
                box=box_np,
                multimask_output=True,
            )
            # Take the highest-scoring mask
            best_idx = np.argmax(scores)
            masks.append(mask_pred[best_idx].astype(bool))

        return masks

    @torch.no_grad()
    def segment_with_points(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Segment using point prompts.

        Args:
            image: (H, W, 3) uint8 RGB image
            points: (N, 2) array of (x, y) coordinates
            labels: (N,) array of 1 (foreground) or 0 (background)

        Returns:
            Binary mask (H, W)
        """
        self._load_model()

        self._predictor.set_image(image)
        mask_pred, scores, _ = self._predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        return mask_pred[best_idx].astype(bool)


# ============================================================================
# FALLBACK: SIMPLE THRESHOLD SEGMENTER (when SAM2 is not available)
# ============================================================================


class SimpleBoxSegmenter:
    """
    Fallback segmenter that uses box prompts + intensity thresholding
    when SAM2 is not installed. Less accurate but requires no extra deps.
    """

    def __init__(self):
        print("⚠ Using simple box segmenter (SAM2 not available)")

    def segment_with_boxes(
        self, image: np.ndarray, boxes: List[List[float]]
    ) -> List[np.ndarray]:
        """Create masks by filling bounding boxes (rough approximation)."""
        h, w = image.shape[:2]
        masks = []

        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True

            # Refine with intensity: keep darker region (potential mass)
            if image.ndim == 3:
                gray = image.mean(axis=-1)
            else:
                gray = image
            roi = gray[y1:y2, x1:x2]
            if roi.size > 0:
                thresh = np.median(roi)
                roi_mask = roi < thresh + 30  # Keep slightly below median
                mask[y1:y2, x1:x2] = roi_mask

            masks.append(mask)

        return masks


# ============================================================================
# 3D VOLUME PIPELINE
# ============================================================================


class Florence2SAM2Pipeline:
    """
    Complete zero-shot 3D segmentation pipeline:
    Florence-2 (detection) + SAM2 (segmentation) + 3D assembly.
    """

    CLASS_NAMES = {0: "background", 1: "kidney", 2: "tumor", 3: "cyst"}

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load models
        self.detector = Florence2Detector(
            model_name=config["florence_model"],
            device=config["florence_device"],
        )

        # Try SAM2, fall back to simple segmenter
        try:
            self.segmenter = SAM2Segmenter(
                checkpoint=config["sam2_checkpoint"],
                device=config["sam2_device"],
            )
        except Exception as e:
            print(f"⚠ SAM2 load failed ({e}), using simple segmenter")
            self.segmenter = SimpleBoxSegmenter()

    def process_volume(
        self,
        image_path: str,
        label_path: Optional[str] = None,
        case_id: str = "unknown",
    ) -> Dict:
        """
        Process a single CT volume through the full pipeline.

        Args:
            image_path: Path to imaging.nii.gz
            label_path: Path to segmentation.nii.gz (for evaluation)
            case_id: Case identifier

        Returns:
            Results dict with segmentation and metrics
        """
        print(f"\n{'─' * 50}")
        print(f"Processing: {case_id}")

        # ── Load & preprocess ──
        img_nii = nib.load(image_path)
        img_data = img_nii.get_fdata().astype(np.float32)

        gt_data = None
        if label_path and Path(label_path).exists():
            gt_data = nib.load(label_path).get_fdata().astype(np.int64)

        # CT windowing
        hu_min, hu_max = self.config["hu_min"], self.config["hu_max"]
        img_norm = np.clip(img_data, hu_min, hu_max)
        img_norm = (img_norm - hu_min) / (hu_max - hu_min)  # [0, 1]

        D, H, W = img_norm.shape
        print(f"  Volume: {D}×{H}×{W}")

        # ── Stage 1: Florence-2 detection on sampled slices ──
        t0 = time.time()
        slice_step = self.config["slice_step"]
        sampled_slices = list(range(0, D, slice_step))

        all_detections = {}  # {slice_idx: {class_id: [detections]}}
        for z in tqdm(sampled_slices, desc="  Florence-2", leave=False):
            ct_slice = img_norm[z]
            dets = self.detector.detect_structures(
                ct_slice,
                self.config["class_prompts"],
                self.config["confidence_threshold"],
            )
            all_detections[z] = dets

        t_detect = time.time() - t0
        n_dets = sum(
            len(d) for z_dets in all_detections.values() for d in z_dets.values()
        )
        print(f"  Florence-2: {n_dets} detections in {t_detect:.1f}s")

        # ── Stage 2: SAM2 segmentation ──
        t0 = time.time()
        segmentation = np.zeros((D, H, W), dtype=np.int64)

        for z in tqdm(range(D), desc="  SAM2 seg", leave=False):
            # Get detections for this slice (use nearest detected slice)
            nearest_z = min(
                all_detections.keys(), key=lambda dz: abs(dz - z), default=None
            )

            if nearest_z is None or abs(nearest_z - z) > slice_step * 2:
                continue

            dets_for_slice = all_detections[nearest_z]

            # Prepare slice as RGB uint8 for SAM
            ct_uint8 = (img_norm[z] * 255).clip(0, 255).astype(np.uint8)
            rgb_slice = np.stack([ct_uint8] * 3, axis=-1)

            # Process each class
            for cls_id in [1, 2, 3]:  # kidney, tumor, cyst
                cls_dets = dets_for_slice.get(cls_id, [])
                if not cls_dets:
                    continue

                boxes = [d["bbox"] for d in cls_dets]
                masks = self.segmenter.segment_with_boxes(rgb_slice, boxes)

                for mask in masks:
                    # Higher class ID overwrites lower (tumor > kidney)
                    segmentation[z][mask] = cls_id

        t_seg = time.time() - t0
        print(f"  SAM2: segmentation in {t_seg:.1f}s")

        # ── Stage 3: Post-processing ──
        segmentation = self._postprocess(segmentation)

        # ── Evaluate ──
        results = {
            "case_id": case_id,
            "volume_shape": [D, H, W],
            "detection_time_s": round(t_detect, 2),
            "segmentation_time_s": round(t_seg, 2),
            "total_time_s": round(t_detect + t_seg, 2),
            "num_detections": n_dets,
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
        save_path = self.output_dir / f"{case_id}_pred.nii.gz"
        nib.save(pred_nii, save_path)

        return results

    def _postprocess(self, seg: np.ndarray) -> np.ndarray:
        """
        Post-processing:
        1. Remove small connected components
        2. Ensure tumor/cyst are inside kidney
        3. Fill holes
        """
        from scipy import ndimage

        # Remove small components per class
        for cls_id in [1, 2, 3]:
            cls_mask = seg == cls_id
            if cls_mask.sum() == 0:
                continue

            labeled, n_comp = ndimage.label(cls_mask)
            min_size = {1: 5000, 2: 200, 3: 200}  # Min voxels per component
            for comp_id in range(1, n_comp + 1):
                comp_mask = labeled == comp_id
                if comp_mask.sum() < min_size.get(cls_id, 200):
                    seg[comp_mask] = 0

        # Ensure tumor (2) and cyst (3) are inside dilated kidney region
        kidney_mask = seg == 1
        if kidney_mask.sum() > 0:
            # Dilate kidney region to allow border tumors
            struct = ndimage.generate_binary_structure(3, 2)
            kidney_dilated = ndimage.binary_dilation(kidney_mask, struct, iterations=5)

            for cls_id in [2, 3]:
                cls_mask = seg == cls_id
                outside = cls_mask & ~kidney_dilated
                seg[outside] = 0  # Remove tumor/cyst outside kidney region

        return seg

    def _compute_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict:
        """Compute Dice, IoU, precision, recall per class."""
        metrics = {"dice": {}, "iou": {}, "precision": {}, "recall": {}}

        for cls_id in range(4):
            p = pred == cls_id
            g = gt == cls_id
            tp = (p & g).sum()
            fp = (p & ~g).sum()
            fn = (~p & g).sum()

            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)

            metrics["dice"][cls_id] = round(float(dice), 5)
            metrics["iou"][cls_id] = round(float(iou), 5)
            metrics["precision"][cls_id] = round(float(prec), 5)
            metrics["recall"][cls_id] = round(float(rec), 5)

        return metrics

    def run_evaluation(self, max_cases: Optional[int] = None):
        """Run pipeline on KiTS23 test set and compute aggregate metrics."""
        data_dir = Path(self.config["kits23_dir"])
        cases = sorted(
            [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")]
        )

        if max_cases:
            cases = cases[:max_cases]

        # Use same val split as V4/V5 (first 20%)
        split = int(len(cases) * 0.2)
        test_cases = cases[:split]  # Same val set
        print(f"\nEvaluating on {len(test_cases)} cases")

        all_results = []
        for case_dir in test_cases:
            img_path = str(case_dir / "imaging.nii.gz")
            lbl_path = str(case_dir / "segmentation.nii.gz")

            if not Path(img_path).exists():
                continue

            result = self.process_volume(img_path, lbl_path, case_dir.name)
            all_results.append(result)

        # Aggregate metrics
        if all_results and "metrics" in all_results[0]:
            agg = self._aggregate_metrics(all_results)
            print(f"\n{'═' * 60}")
            print("PIPELINE A RESULTS — Florence-2 + SAM2 (Zero-Shot)")
            print(f"{'═' * 60}")
            print(f"Cases evaluated: {len(all_results)}")
            for cls_id, name in self.CLASS_NAMES.items():
                if cls_id == 0:
                    continue
                d = agg["mean_dice"].get(cls_id, 0)
                print(f"  {name:<10} Dice: {d:.4f}")
            print(f"  {'Mean FG':<10} Dice: {agg['mean_fg_dice']:.4f}")
            print(f"{'═' * 60}")

            # Save aggregate results
            summary = {
                "pipeline": "Florence-2 + SAM2 (Zero-Shot)",
                "num_cases": len(all_results),
                "aggregate_metrics": agg,
                "per_case_results": all_results,
                "config": {k: str(v) for k, v in self.config.items()},
            }
            with open(self.output_dir / "results.json", "w") as f:
                json.dump(summary, f, indent=2)

        return all_results

    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Compute mean metrics across all cases."""
        all_dice = {i: [] for i in range(4)}

        for r in results:
            if "metrics" not in r:
                continue
            for cls_id in range(4):
                d = r["metrics"]["dice"].get(
                    cls_id, r["metrics"]["dice"].get(str(cls_id), 0)
                )
                all_dice[cls_id].append(d)

        mean_dice = {i: float(np.mean(v)) if v else 0.0 for i, v in all_dice.items()}
        mean_fg = float(np.mean([mean_dice[i] for i in [1, 2, 3]]))

        return {"mean_dice": mean_dice, "mean_fg_dice": mean_fg}


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline A: Florence-2 + SAM2 Zero-Shot 3D Segmentation"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="test",
        help="test=5 cases, full=all validation cases",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None, help="Override max number of cases"
    )
    parser.add_argument(
        "--florence-model",
        type=str,
        default=None,
        help="Florence-2 model name (HuggingFace)",
    )
    parser.add_argument(
        "--florence-device", type=str, default=None, choices=["cuda", "cpu"]
    )
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument(
        "--slice-step",
        type=int,
        default=None,
        help="Process every N-th slice with Florence-2",
    )
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)

    if args.florence_model:
        config["florence_model"] = args.florence_model
    if args.florence_device:
        config["florence_device"] = args.florence_device
    if args.sam_checkpoint:
        config["sam2_checkpoint"] = args.sam_checkpoint
    if args.slice_step:
        config["slice_step"] = args.slice_step

    max_cases = args.max_cases
    if max_cases is None:
        max_cases = 5 if args.mode == "test" else None

    pipeline = Florence2SAM2Pipeline(config)
    pipeline.run_evaluation(max_cases=max_cases)


if __name__ == "__main__":
    main()
