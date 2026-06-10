"""
MedDINO-VISTA3D V3 - Comprehensive Testing Script
=================================================

Tests the trained model on the validation/test set and generates:
- Per-case metrics
- Aggregate statistics
- JSON results
- Optional visualizations

Usage:
    python test_model_v3.py --checkpoint ./output/meddino_vista3d_v3_quality/best_model_final_dice.pth
    python test_model_v3.py --checkpoint ./output/meddino_vista3d_v3_quality/best_model_final_dice.pth --save-viz
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Import model from training script
sys.path.insert(0, str(Path(__file__).parent))
from meddino_vista3d_v3_quality import MedDINOVISTA3DEnhanced, get_config


def calculate_metrics(pred, target, num_classes=4):
    """Calculate Dice, IoU, Precision, Recall per class"""
    metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': []
    }
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum()
        pred_sum = pred_mask.sum()
        target_sum = target_mask.sum()
        union = pred_sum + target_sum - intersection
        
        # Dice
        dice = (2 * intersection) / (pred_sum + target_sum + 1e-8)
        metrics['dice'].append(float(dice))
        
        # IoU
        iou = intersection / (union + 1e-8)
        metrics['iou'].append(float(iou))
        
        # Precision
        precision = intersection / (pred_sum + 1e-8)
        metrics['precision'].append(float(precision))
        
        # Recall
        recall = intersection / (target_sum + 1e-8)
        metrics['recall'].append(float(recall))
    
    return metrics


def sliding_window_inference(model, image, patch_size=(140, 224, 224), overlap=0.5, device='cuda'):
    """
    Perform sliding window inference on a full 3D volume
    """
    model.eval()
    D, H, W = image.shape
    pd, ph, pw = patch_size
    
    # Calculate stride
    stride_d = int(pd * (1 - overlap))
    stride_h = int(ph * (1 - overlap))
    stride_w = int(pw * (1 - overlap))
    
    # Initialize output
    num_classes = 4
    output_sum = torch.zeros((num_classes, D, H, W), device=device)
    count_map = torch.zeros((D, H, W), device=device)
    
    # Pad image if needed
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        D, H, W = image.shape
        output_sum = torch.zeros((num_classes, D, H, W), device=device)
        count_map = torch.zeros((D, H, W), device=device)
    
    # Sliding window
    with torch.no_grad():
        for d in range(0, D - pd + 1, stride_d):
            for h in range(0, H - ph + 1, stride_h):
                for w in range(0, W - pw + 1, stride_w):
                    # Extract patch
                    patch = image[d:d+pd, h:h+ph, w:w+pw]
                    
                    # Normalize
                    patch = (patch - patch.mean()) / (patch.std() + 1e-8)
                    
                    # Convert to tensor
                    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Predict
                    with torch.amp.autocast("cuda"):
                        logits = model(patch_tensor)
                    
                    # Accumulate
                    output_sum[:, d:d+pd, h:h+ph, w:w+pw] += logits.squeeze(0)
                    count_map[d:d+pd, h:h+ph, w:w+pw] += 1
    
    # Average overlapping predictions
    output = output_sum / count_map.unsqueeze(0).clamp(min=1)
    prediction = torch.argmax(output, dim=0).cpu().numpy()
    
    # Remove padding
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        D_orig = D - pad_d
        H_orig = H - pad_h
        W_orig = W - pad_w
        prediction = prediction[:D_orig, :H_orig, :W_orig]
    
    return prediction


def test_model(checkpoint_path, output_dir, config, save_visualizations=False):
    """
    Test the model on validation set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = MedDINOVISTA3DEnhanced(
        num_classes=config["num_classes"],
        freeze_encoder=True,
        dropout=0.1,
        dinov2_backbone=config["dinov2_backbone"],
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Get validation cases
    data_dir = Path(config["kits23_dir"])
    all_cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    
    # Use same split as training
    split_idx = int(len(all_cases) * config["val_split"])
    val_cases = all_cases[:split_idx]
    
    print(f"\nTesting on {len(val_cases)} validation cases...")
    
    # Test each case
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "num_test_cases": len(val_cases),
        "per_case_results": [],
        "aggregate_metrics": {}
    }
    
    all_dice = {0: [], 1: [], 2: [], 3: []}  # BG, Kidney, Tumor, Cyst
    
    for case_dir in tqdm(val_cases, desc="Testing"):
        case_name = case_dir.name
        
        try:
            # Load data
            img_path = case_dir / "imaging.nii.gz"
            lbl_path = case_dir / "segmentation.nii.gz"
            
            if not img_path.exists() or not lbl_path.exists():
                print(f"⚠ Skipping {case_name}: Missing files")
                continue
            
            image = nib.load(str(img_path)).get_fdata()
            label = nib.load(str(lbl_path)).get_fdata()
            
            # Perform sliding window inference
            prediction = sliding_window_inference(model, image, config["patch_size"], device=device)
            
            # Calculate metrics
            metrics = calculate_metrics(prediction, label)
            
            # Store per-case results
            case_result = {
                "case": case_name,
                "metrics": metrics,
                "shape": list(image.shape)
            }
            results["per_case_results"].append(case_result)
            
            # Accumulate for aggregate
            for cls in range(4):
                all_dice[cls].append(metrics['dice'][cls])
            
            # Save visualization if requested
            if save_visualizations:
                save_dir = output_dir / "visualizations" / case_name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save prediction
                pred_nii = nib.Nifti1Image(prediction.astype(np.uint8), nib.load(str(img_path)).affine)
                nib.save(pred_nii, str(save_dir / "prediction.nii.gz"))
        
        except Exception as e:
            print(f"\n❌ Error processing {case_name}: {e}")
            continue
    
    # Calculate aggregate statistics
    class_names = ["background", "kidney", "tumor", "cyst"]
    aggregate = {}
    
    for cls, name in enumerate(class_names):
        if all_dice[cls]:
            aggregate[name] = {
                "mean_dice": float(np.mean(all_dice[cls])),
                "std_dice": float(np.std(all_dice[cls])),
                "median_dice": float(np.median(all_dice[cls])),
                "min_dice": float(np.min(all_dice[cls])),
                "max_dice": float(np.max(all_dice[cls]))
            }
    
    # Mean foreground Dice
    fg_dice = [all_dice[1], all_dice[2], all_dice[3]]
    fg_dice_flat = [d for sublist in fg_dice for d in sublist]
    aggregate["mean_foreground_dice"] = float(np.mean(fg_dice_flat))
    
    results["aggregate_metrics"] = aggregate
    
    # Save results
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Testing complete!")
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    for name, stats in aggregate.items():
        if isinstance(stats, dict):
            print(f"\n{name.upper()}:")
            print(f"  Mean Dice: {stats['mean_dice']:.4f} ± {stats['std_dice']:.4f}")
            print(f"  Median: {stats['median_dice']:.4f}")
            print(f"  Range: [{stats['min_dice']:.4f}, {stats['max_dice']:.4f}]")
        else:
            print(f"\n{name.upper()}: {stats:.4f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth file)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: same as checkpoint)")
    parser.add_argument("--save-viz", action="store_true", help="Save prediction visualizations")
    parser.add_argument("--mode", type=str, default="full", choices=["quick_test", "full"])
    args = parser.parse_args()
    
    # Get config
    config = get_config(args.mode)
    
    # Set output directory
    if args.output_dir is None:
        checkpoint_path = Path(args.checkpoint)
        output_dir = checkpoint_path.parent
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run testing
    test_model(args.checkpoint, output_dir, config, args.save_viz)
