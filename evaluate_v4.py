"""
Evaluation Script for MedDINO-VISTA3D V4
========================================

Evaluates the trained V4 model on the test set and generates:
1. Per-case segmentation predictions (NIfTI files)
2. Detailed metrics JSON (Dice, IoU, Precision, Recall per case)
3. Aggregate statistics and visualizations

Usage:
    python evaluate_v4.py --checkpoint best_model_final_dice.pth
    python evaluate_v4.py --checkpoint best_model_stage1_dice.pth --num_cases 10
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import time

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Import model from V4 script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from meddino_vista3d_v4_tumor_focused import MedDINOVISTA3DEnhanced


def sliding_window_inference(model, image, roi_size=(140, 224, 224), overlap=0.5, device='cuda'):
    """
    Perform sliding window inference on a full 3D volume.
    
    Args:
        model: The segmentation model
        image: Full 3D volume (D, H, W)
        roi_size: Size of the sliding window
        overlap: Overlap ratio between windows (0.5 = 50% overlap)
        device: Device to run inference on
    """
    model.eval()
    d, h, w = image.shape
    pd, ph, pw = roi_size
    
    # Calculate stride based on overlap
    stride_d = int(pd * (1 - overlap))
    stride_h = int(ph * (1 - overlap))
    stride_w = int(pw * (1 - overlap))
    
    # Pad image if needed
    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        d, h, w = image.shape
    
    # Initialize output
    output = np.zeros((4, d, h, w), dtype=np.float32)  # 4 classes
    count = np.zeros((d, h, w), dtype=np.float32)
    
    # Generate sliding window positions
    positions = []
    for ds in range(0, d - pd + 1, stride_d):
        for hs in range(0, h - ph + 1, stride_h):
            for ws in range(0, w - pw + 1, stride_w):
                positions.append((ds, hs, ws))
    
    # Add final positions to cover edges
    if d > pd:
        for hs in range(0, h - ph + 1, stride_h):
            for ws in range(0, w - pw + 1, stride_w):
                positions.append((d - pd, hs, ws))
    if h > ph:
        for ds in range(0, d - pd + 1, stride_d):
            for ws in range(0, w - pw + 1, stride_w):
                positions.append((ds, h - ph, ws))
    if w > pw:
        for ds in range(0, d - pd + 1, stride_d):
            for hs in range(0, h - ph + 1, stride_h):
                positions.append((ds, hs, w - pw))
    
    # Remove duplicates
    positions = list(set(positions))
    
    print(f"  Processing {len(positions)} windows...")
    
    with torch.no_grad():
        for ds, hs, ws in tqdm(positions, desc="  Inference", leave=False):
            # Extract patch
            patch = image[ds:ds+pd, hs:hs+ph, ws:ws+pw]
            
            # Normalize
            patch = (patch - patch.mean()) / (patch.std() + 1e-8)
            
            # To tensor
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(patch_tensor)
            
            # To probabilities
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Accumulate
            output[:, ds:ds+pd, hs:hs+ph, ws:ws+pw] += probs
            count[ds:ds+pd, hs:hs+ph, ws:ws+pw] += 1
    
    # Average overlapping predictions
    count[count == 0] = 1
    output = output / count[None, :, :, :]
    
    # Get final prediction
    prediction = np.argmax(output, axis=0)
    
    # Remove padding
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        prediction = prediction[:d-pad_d, :h-pad_h, :w-pad_w]
    
    return prediction


def compute_metrics(pred, target, num_classes=4):
    """Compute Dice, IoU, Precision, Recall per class"""
    metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': []
    }
    
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        tp = np.sum(pred_c & target_c)
        fp = np.sum(pred_c & ~target_c)
        fn = np.sum(~pred_c & target_c)
        
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        metrics['dice'].append(float(dice))
        metrics['iou'].append(float(iou))
        metrics['precision'].append(float(precision))
        metrics['recall'].append(float(recall))
    
    return metrics


def evaluate_model(config):
    """Main evaluation function"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config['output_dir'])
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"MedDINO-VISTA3D V4 Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {config['checkpoint']}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    model = MedDINOVISTA3DEnhanced(
        num_classes=4,
        freeze_encoder=False,  # Not needed for inference
        dinov2_backbone="dinov2_vitb14"
    ).to(device)
    
    checkpoint_path = output_dir / config['checkpoint']
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✓ Loaded checkpoint: {checkpoint_path.name}\n")
    
    # Get test cases
    data_dir = Path(config['kits23_dir'])
    all_cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    
    # Use validation split (first 20%)
    split = int(len(all_cases) * 0.2)
    test_cases = all_cases[:split]
    
    if config.get('num_cases'):
        test_cases = test_cases[:config['num_cases']]
    
    print(f"Evaluating on {len(test_cases)} cases...\n")
    
    # Evaluate each case
    results = []
    total_start = time.time()
    
    for case_dir in test_cases:
        case_name = case_dir.name
        print(f"Processing {case_name}...")
        
        # Load image and label
        img_path = case_dir / "imaging.nii.gz"
        lbl_path = case_dir / "segmentation.nii.gz"
        
        if not img_path.exists():
            print(f"  ⚠ Skipping (no imaging file)")
            continue
        
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()
        
        lbl = None
        if lbl_path.exists():
            lbl = nib.load(lbl_path).get_fdata()
        
        case_start = time.time()
        
        # Run inference
        pred = sliding_window_inference(
            model, img, 
            roi_size=config['patch_size'],
            overlap=config.get('overlap', 0.5),
            device=device
        )
        
        case_time = time.time() - case_start
        
        # Save prediction
        pred_nii = nib.Nifti1Image(pred.astype(np.uint8), img_nii.affine, img_nii.header)
        pred_path = predictions_dir / f"{case_name}_prediction.nii.gz"
        nib.save(pred_nii, pred_path)
        
        # Compute metrics if ground truth available
        case_result = {
            'case': case_name,
            'shape': list(img.shape),
            'inference_time_seconds': round(case_time, 2)
        }
        
        if lbl is not None:
            metrics = compute_metrics(pred, lbl, num_classes=4)
            case_result['metrics'] = metrics
            
            print(f"  ✓ Dice: K={metrics['dice'][1]:.3f} T={metrics['dice'][2]:.3f} C={metrics['dice'][3]:.3f} | Time: {case_time:.1f}s")
        else:
            print(f"  ✓ Prediction saved (no ground truth) | Time: {case_time:.1f}s")
        
        results.append(case_result)
    
    total_time = time.time() - total_start
    
    # Compute aggregate metrics
    if any('metrics' in r for r in results):
        all_metrics = [r['metrics'] for r in results if 'metrics' in r]
        
        aggregate = {}
        for class_idx, class_name in enumerate(['background', 'kidney', 'tumor', 'cyst']):
            class_metrics = {
                'mean_dice': float(np.mean([m['dice'][class_idx] for m in all_metrics])),
                'std_dice': float(np.std([m['dice'][class_idx] for m in all_metrics])),
                'median_dice': float(np.median([m['dice'][class_idx] for m in all_metrics])),
                'min_dice': float(np.min([m['dice'][class_idx] for m in all_metrics])),
                'max_dice': float(np.max([m['dice'][class_idx] for m in all_metrics])),
            }
            aggregate[class_name] = class_metrics
        
        # Mean foreground dice
        aggregate['mean_foreground_dice'] = float(np.mean([
            np.mean(m['dice'][1:]) for m in all_metrics
        ]))
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total cases: {len(results)}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"\nAggregate Metrics (Mean ± Std):")
        print(f"  Kidney: {aggregate['kidney']['mean_dice']:.4f} ± {aggregate['kidney']['std_dice']:.4f}")
        print(f"  Tumor:  {aggregate['tumor']['mean_dice']:.4f} ± {aggregate['tumor']['std_dice']:.4f}")
        print(f"  Cyst:   {aggregate['cyst']['mean_dice']:.4f} ± {aggregate['cyst']['std_dice']:.4f}")
        print(f"  Mean FG: {aggregate['mean_foreground_dice']:.4f}")
        print(f"{'='*60}\n")
    
    # Save results
    output_json = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': config['checkpoint'],
        'num_test_cases': len(results),
        'per_case_results': results
    }
    
    if any('metrics' in r for r in results):
        output_json['aggregate_metrics'] = aggregate
    
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Predictions saved to: {predictions_dir}\n")
    
    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_model_final_dice.pth",
                        help="Checkpoint filename (in output directory)")
    parser.add_argument("--num_cases", type=int, default=None,
                        help="Number of cases to evaluate (default: all validation cases)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Sliding window overlap ratio (default: 0.5)")
    args = parser.parse_args()
    
    config = {
        'kits23_dir': './kits23/dataset',
        'output_dir': './output/meddino_vista3d_v4_tumor_focused',
        'checkpoint': args.checkpoint,
        'num_cases': args.num_cases,
        'patch_size': (140, 224, 224),
        'overlap': args.overlap,
    }
    
    evaluate_model(config)
