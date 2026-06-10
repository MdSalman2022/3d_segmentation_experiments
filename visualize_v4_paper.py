"""
V4 Paper-Quality Visualization Generator (Enhanced)
===================================================

Generates high-quality figures for the MedDINO-VISTA3D V4 paper.

Updates:
- Auto-selects BEST cases for Kidney, Tumor, and Cyst
- Generates 3D Volumetric Renders (Meshes)
- Creates Tumor-focused Zoom comparisons

Usage:
    python visualize_v4_paper.py
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# ============================================================================
# CONFIGURATION
# ============================================================================

COLORS = {
    0: (0, 0, 0),           # Background - black
    1: (0.2, 0.8, 0.2),     # Kidney - green
    2: (0.9, 0.2, 0.2),     # Tumor - red
    3: (0.2, 0.5, 0.9),     # Cyst - blue
}

CLASS_NAMES = {
    0: "Background",
    1: "Kidney",
    2: "Tumor",  
    3: "Cyst"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_dice(pred, gt, class_id):
    """Calculate Dice score for a specific class"""
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)
    intersection = (pred_mask & gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 0.0 if intersection == 0 else 1.0
    return (2.0 * intersection) / total

def mask_to_rgb(mask, alpha=1.0):
    """Convert segmentation mask to RGB image"""
    rgb = np.zeros((*mask.shape, 3))
    for class_id, color in COLORS.items():
        if class_id == 0: continue
        class_mask = mask == class_id
        for c in range(3):
            rgb[class_mask, c] = color[c]
    return rgb

def overlay_mask_on_image(image, mask, alpha=0.5):
    """Overlay colored mask on grayscale image"""
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image_rgb = np.stack([image_norm] * 3, axis=-1)
    mask_rgb = mask_to_rgb(mask)
    
    alpha_mask = np.zeros_like(mask, dtype=float)
    alpha_mask[mask > 0] = alpha
    alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
    
    overlay = image_rgb * (1 - alpha_mask) + mask_rgb * alpha_mask
    return overlay

def find_best_slices(gt, pred, class_id=2, num_slices=3):
    """Find slices that show the specific class best"""
    scores = []
    for i in range(gt.shape[0]):
        gt_pixels = (gt[i] == class_id).sum()
        pred_pixels = (pred[i] == class_id).sum()
        if gt_pixels > 0 or pred_pixels > 0:
            score = gt_pixels + pred_pixels
            scores.append((i, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores: return []
        
    top_slices = [x[0] for x in scores[:15]] 
    selected = []
    if top_slices: selected.append(top_slices[0])
        
    for s in top_slices[1:]:
        if len(selected) >= num_slices: break
        if all(abs(s - exist) > 10 for exist in selected): # Spread out
            selected.append(s)
            
    return sorted(selected)

def get_best_cases(json_path: Path) -> Dict[str, str]:
    """Find best cases for each class from results JSON"""
    if not json_path.exists():
        print(f"Warning: Results file {json_path} not found.")
        return {}
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cases = data.get('cases', data.get('per_case_results', [])) # Handle both formats
    
    best_cases = {}
    
    # Check data structure format
    # structure 1: fair_results: cases[i]['metrics']['dice'][class_idx]
    # structure 2: test_results: cases[i]['metrics']['dice'][class_idx]
    
    for cls_name, cls_idx in [("Kidney", 1), ("Tumor", 2), ("Cyst", 3)]:
        valid_cases = []
        for c in cases:
            metrics = c['metrics']
            dice = metrics['dice'][cls_idx]
            # Ensure class actually exists (vol_gt > 0 if available, or just check if dice > 0)
            has_content = True
            if 'vol_gt' in metrics:
                has_content = metrics['vol_gt'][cls_idx] > 100 # Min pixels
            
            if has_content:
                valid_cases.append((c['case'], dice))
        
        valid_cases.sort(key=lambda x: x[1], reverse=True)
        if valid_cases:
            best_cases[cls_name] = valid_cases[0] # (name, dice)
            
    return best_cases

# ============================================================================
# 3D VISUALIZATION
# ============================================================================

def create_3d_render(gt, pred, output_path, title="3D Visualization"):
    """
    Create a 3D surface rendering of the segmentation using Marching Cubes
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Ground Truth
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"{title} - Ground Truth", fontsize=14)
    
    has_plot1 = False
    for cls_idx, color in [(1, 'green'), (2, 'red'), (3, 'blue')]:
        if (gt == cls_idx).sum() < 100: continue
        
        # Marching cubes
        try:
            verts, faces, _, _ = measure.marching_cubes(gt == cls_idx, level=0.5, step_size=2)
            mesh = Poly3DCollection(verts[faces], alpha=0.3 if cls_idx==1 else 0.8) # Kidney transparent
            mesh.set_facecolor(color)
            ax1.add_collection3d(mesh)
            has_plot1 = True
        except Exception as e:
            print(f"  GT Mesh error class {cls_idx}: {e}")
            
    if has_plot1:
        # Scale fix
        shape = gt.shape
        ax1.set_xlim(0, shape[0]); ax1.set_ylim(0, shape[1]); ax1.set_zlim(0, shape[2])
        ax1.view_init(elev=30, azim=45)
    else:
        ax1.text(0.5, 0.5, 0.5, "Empty Volume", ha='center')

    # 2. Prediction
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"{title} - Prediction", fontsize=14)
    
    has_plot2 = False
    for cls_idx, color in [(1, 'green'), (2, 'red'), (3, 'blue')]:
        if (pred == cls_idx).sum() < 100: continue
        
        try:
            verts, faces, _, _ = measure.marching_cubes(pred == cls_idx, level=0.5, step_size=2)
            mesh = Poly3DCollection(verts[faces], alpha=0.3 if cls_idx==1 else 0.8)
            mesh.set_facecolor(color)
            ax2.add_collection3d(mesh)
            has_plot2 = True
        except:
             pass

    if has_plot2:
        ax2.set_xlim(0, shape[0]); ax2.set_ylim(0, shape[1]); ax2.set_zlim(0, shape[2])
        ax2.view_init(elev=30, azim=45)
        
    # Legend
    legend_elements = [
        mpatches.Patch(color='green', label='Kidney'),
        mpatches.Patch(color='red', label='Tumor'),
        mpatches.Patch(color='blue', label='Cyst')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved 3D render: {output_path}")

# ============================================================================
# 2D VISUALIZATION (Comparison)
# ============================================================================

def create_full_comparison(image_path, gt_path, pred_path, output_path, focus_class=2):
    """Create a panel comparison focused on a specific class"""
    image = nib.load(image_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    pred = nib.load(pred_path).get_fdata()
    
    # Find interesting slices
    slices = find_best_slices(gt, pred, class_id=focus_class, num_slices=4)
    
    # If focus class is missing, try others
    if not slices:
        slices = find_best_slices(gt, pred, class_id=1, num_slices=4)
        
    if not slices: return # Skip if empty

    num_rows = len(slices)
    fig = plt.figure(figsize=(20, num_rows * 4))
    gs = GridSpec(num_rows, 5, figure=fig, hspace=0.1, wspace=0.1)
    
    case_name = Path(image_path).parent.name
    fig.suptitle(f"Case {case_name} (Focus: {CLASS_NAMES[focus_class]})", fontsize=16, fontweight='bold', y=0.95)
    
    for idx, slice_idx in enumerate(slices):
        img_s = np.rot90(image[slice_idx])
        gt_s = np.rot90(gt[slice_idx])
        pred_s = np.rot90(pred[slice_idx])
        
        dice_t = calculate_dice(pred_s, gt_s, 2)
        
        # 1. CT
        ax = fig.add_subplot(gs[idx, 0]); ax.imshow(img_s, cmap='gray'); ax.axis('off')
        ax.text(5, 10, f"z={slice_idx}", color='white')
        if idx==0: ax.set_title("CT", fontweight='bold')
        
        # 2. GT
        ax = fig.add_subplot(gs[idx, 1]); ax.imshow(overlay_mask_on_image(img_s, gt_s)); ax.axis('off')
        if idx==0: ax.set_title("Ground Truth", fontweight='bold')
        
        # 3. Pred
        ax = fig.add_subplot(gs[idx, 2]); ax.imshow(overlay_mask_on_image(img_s, pred_s)); ax.axis('off')
        if idx==0: ax.set_title("Prediction", fontweight='bold')
        
        # 4. Error
        ax = fig.add_subplot(gs[idx, 3])
        fp_mask = (pred_s > 0) & (gt_s == 0)
        fn_mask = (pred_s == 0) & (gt_s > 0)
        err_rgb = np.zeros((*gt_s.shape, 3))
        err_rgb[fp_mask] = [1, 0, 0] 
        err_rgb[fn_mask] = [0, 0, 1] 
        img_back = np.stack([(img_s - img_s.min()) / (img_s.max() - img_s.min())]*3, axis=-1) * 0.5
        final_err = np.where(np.stack([fp_mask | fn_mask]*3, axis=-1), err_rgb, img_back)
        ax.imshow(final_err); ax.axis('off')
        if idx==0: ax.set_title("Errors (R=FP, B=FN)", fontweight='bold')
        
        # 5. Zoom (Crop to ROI)
        ax = fig.add_subplot(gs[idx, 4])
        # Find bbox of foreground
        rows = np.any(gt_s + pred_s, axis=1)
        cols = np.any(gt_s + pred_s, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Pad
            p = 20
            rmin = max(0, rmin-p); rmax = min(img_s.shape[0], rmax+p)
            cmin = max(0, cmin-p); cmax = min(img_s.shape[1], cmax+p)
            
            ax.imshow(overlay_mask_on_image(img_s[rmin:rmax, cmin:cmax], pred_s[rmin:rmax, cmin:cmax]))
        else:
            ax.imshow(overlay_mask_on_image(img_s, pred_s))
        ax.axis('off')
        if idx==0: ax.set_title("Zoom", fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved 2D comparison: {output_path}")

# ============================================================================
# MAIN
# ============================================================================

def run_visualization():
    base_dir = Path("./")
    kits_dir = base_dir / "kits23/dataset"
    output_base = base_dir / "output/meddino_vista3d_v4_tumor_focused"
    pred_dir = output_base / "predictions"
    viz_dir = output_base / "figures"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Try to load Fair Results first, then Test Results
    json_path = output_base / "fair_results.json"
    if not json_path.exists():
        json_path = output_base / "test_results.json"
        
    print(f"\n1. Finding Best Cases from: {json_path}")
    best_cases = get_best_cases(json_path)
    
    if not best_cases:
        print(" ! Could not find best cases (JSON missing or empty). Generating for random cases instead.")
        # Fallback to listing files
        files = sorted(list(pred_dir.glob("*_prediction.nii.gz")))[:5]
        target_cases = [f.name.replace("_prediction.nii.gz", "") for f in files]
        best_cases = { "Random1": (target_cases[0], 0.0) } if target_cases else {}
    else:
        print("  Found Best Cases:")
        for k, v in best_cases.items():
            print(f"   - {k}: {v[0]} (Dice: {v[1]:.4f})")

    # 2. Process Specific Best Cases
    print(f"\n2. Generating Visualizations...")
    
    for cls_name, (case_name, dice) in best_cases.items():
        print(f"\n--- Processing BEST {cls_name.upper()}: {case_name} ---")
        
        img_path = kits_dir / case_name / "imaging.nii.gz"
        gt_path = kits_dir / case_name / "segmentation.nii.gz"
        pred_path = pred_dir / f"{case_name}_prediction.nii.gz"
        
        if not pred_path.exists(): 
            print(f"Missing prediction file for {case_name}")
            continue

        # Focus class ID
        focus_id = 2 # Default Tumor
        if cls_name == "Kidney": focus_id = 1
        elif cls_name == "Cyst": focus_id = 3
        
        # A. 2D Comparison
        create_full_comparison(
            img_path, gt_path, pred_path, 
            viz_dir / f"Best_{cls_name}_{case_name}_2D.png",
            focus_class=focus_id
        )
        
        # B. 3D Render
        try:
            gt = nib.load(gt_path).get_fdata()
            pred = nib.load(pred_path).get_fdata()
            create_3d_render(
                gt, pred, 
                viz_dir / f"Best_{cls_name}_{case_name}_3D.png",
                title=f"Best {cls_name} ({case_name})"
            )
        except Exception as e:
            print(f"Failed 3D render: {e}")

if __name__ == "__main__":
    run_visualization()
