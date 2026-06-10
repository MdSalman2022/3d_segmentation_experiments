"""
Publication-Quality Visualization for 3D Kidney Segmentation
===========================================================

Creates medical imaging paper-style visualizations:
- Multi-panel slice comparisons
- Overlay visualizations with transparency
- Dice score annotations
- Side-by-side GT vs Prediction
- Class-specific visualizations (Kidney, Tumor, Cyst)

Usage:
    # Generate visualizations from test results
    python visualize_results.py --results-dir ./output/meddino_vista3d_v3_quality/visualizations
    
    # Generate comparison figure for specific case
    python visualize_results.py --case case_00123 --output comparison.png
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import ndimage

# Color scheme for classes (RGB normalized)
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
        class_mask = mask == class_id
        for c in range(3):
            rgb[class_mask, c] = color[c]
    
    return rgb


def overlay_mask_on_image(image, mask, alpha=0.5):
    """Overlay colored mask on grayscale image"""
    # Normalize image to [0, 1]
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Convert to RGB
    image_rgb = np.stack([image_norm] * 3, axis=-1)
    
    # Get mask RGB
    mask_rgb = mask_to_rgb(mask)
    
    # Blend
    overlay = image_rgb * (1 - alpha) + mask_rgb * alpha
    
    return overlay


def find_informative_slices(volume, mask, num_slices=5):
    """Find slices with the most interesting content"""
    scores = []
    
    for i in range(volume.shape[0]):
        # Score based on presence of foreground classes
        fg_pixels = (mask[i] > 0).sum()
        num_classes = len(np.unique(mask[i]))
        
        # Prefer slices with foreground and multiple classes
        score = fg_pixels * num_classes
        scores.append(score)
    
    # Get top slices
    top_indices = np.argsort(scores)[-num_slices:]
    return sorted(top_indices)


def create_comparison_figure(
    image_path: str,
    gt_path: str,
    pred_path: str,
    output_path: str,
    num_slices: int = 5,
    class_focus: Optional[int] = None
):
    """
    Create publication-style comparison figure
    
    Layout:
    - Column 1: Original image
    - Column 2: GT overlay
    - Column 3: Prediction overlay  
    - Column 4: GT only
    - Column 5: Pred only
    """
    # Load data
    image = nib.load(image_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    pred = nib.load(pred_path).get_fdata()
    
    # Find informative slices
    slice_indices = find_informative_slices(image, gt, num_slices)
    
    # Create figure
    fig = plt.figure(figsize=(20, num_slices * 4))
    gs = GridSpec(num_slices, 5, figure=fig, hspace=0.02, wspace=0.02)
    
    case_name = Path(image_path).parent.name
    
    for row, slice_idx in enumerate(slice_indices):
        img_slice = image[slice_idx]
        gt_slice = gt[slice_idx]
        pred_slice = pred[slice_idx]
        
        # Calculate Dice for this slice
        dice_scores = {}
        for class_id in [1, 2, 3]:  # Skip background
            dice_scores[class_id] = calculate_dice(pred_slice, gt_slice, class_id)
        
        # Column 1: Original Image
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(img_slice, cmap='gray')
        ax1.axis('off')
        if row == 0:
            ax1.set_title(f'{case_name}\nSlice {slice_idx}', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.95, f'z={slice_idx}', transform=ax1.transAxes,
                    ha='center', va='top', fontsize=10, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Column 2: GT Overlay
        ax2 = fig.add_subplot(gs[row, 1])
        overlay_gt = overlay_mask_on_image(img_slice, gt_slice, alpha=0.4)
        ax2.imshow(overlay_gt)
        ax2.axis('off')
        if row == 0:
            ax2.set_title('Ground Truth\nOverlay', fontsize=12, fontweight='bold')
        
        # Column 3: Prediction Overlay
        ax3 = fig.add_subplot(gs[row, 2])
        overlay_pred = overlay_mask_on_image(img_slice, pred_slice, alpha=0.4)
        ax3.imshow(overlay_pred)
        ax3.axis('off')
        if row == 0:
            ax3.set_title('Prediction\nOverlay', fontsize=12, fontweight='bold')
        
        # Column 4: GT Mask Only
        ax4 = fig.add_subplot(gs[row, 3])
        gt_rgb = mask_to_rgb(gt_slice)
        ax4.imshow(gt_rgb)
        ax4.axis('off')
        if row == 0:
            ax4.set_title('GT Mask', fontsize=12, fontweight='bold')
        
        # Column 5: Prediction Mask Only with Dice
        ax5 = fig.add_subplot(gs[row, 4])
        pred_rgb = mask_to_rgb(pred_slice)
        ax5.imshow(pred_rgb)
        ax5.axis('off')
        if row == 0:
            ax5.set_title('Pred Mask', fontsize=12, fontweight='bold')
        
        # Add Dice scores to prediction column
        dice_text = f"K: {dice_scores[1]:.3f}\nT: {dice_scores[2]:.3f}\nC: {dice_scores[3]:.3f}"
        ax5.text(0.98, 0.02, dice_text, transform=ax5.transAxes,
                ha='right', va='bottom', fontsize=9, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS[1], label='Kidney'),
        mpatches.Patch(facecolor=COLORS[2], label='Tumor'),
        mpatches.Patch(facecolor=COLORS[3], label='Cyst')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              frameon=True, fontsize=12, bbox_to_anchor=(0.5, -0.02))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved comparison figure: {output_path}")


def create_class_specific_figure(
    image_path: str,
    gt_path: str,
    pred_path: str,
    output_path: str,
    class_id: int = 2,  # Default: Tumor
    num_slices: int = 3
):
    """
    Create figure focused on a specific class (e.g., Tumor only)
    
    Layout: Image | GT Class | Pred Class (with Dice)
    """
    image = nib.load(image_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    pred = nib.load(pred_path).get_fdata()
    
    # Find slices with this class
    class_slices = []
    for i in range(gt.shape[0]):
        if (gt[i] == class_id).sum() > 100:  # At least 100 pixels
            class_slices.append(i)
    
    if not class_slices:
        print(f"⚠ No slices found with class {class_id}")
        return
    
    # Sample evenly
    step = max(1, len(class_slices) // num_slices)
    slice_indices = class_slices[::step][:num_slices]
    
    # Create figure
    fig, axes = plt.subplots(num_slices, 3, figsize=(12, num_slices * 4))
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    case_name = Path(image_path).parent.name
    class_name = CLASS_NAMES[class_id]
    
    fig.suptitle(f'{case_name} - {class_name} Segmentation', 
                fontsize=16, fontweight='bold')
    
    for row, slice_idx in enumerate(slice_indices):
        img_slice = image[slice_idx]
        gt_slice = gt[slice_idx]
        pred_slice = pred[slice_idx]
        
        # Binary masks for this class
        gt_binary = (gt_slice == class_id).astype(float)
        pred_binary = (pred_slice == class_id).astype(float)
        
        # Calculate Dice
        dice = calculate_dice(pred_slice, gt_slice, class_id)
        
        # Column 1: Image
        axes[row, 0].imshow(img_slice, cmap='gray')
        axes[row, 0].set_title(f'Slice {slice_idx}', fontsize=10)
        axes[row, 0].axis('off')
        
        # Column 2: GT
        axes[row, 1].imshow(img_slice, cmap='gray')
        axes[row, 1].contour(gt_binary, colors=[COLORS[class_id]], linewidths=2)
        if gt_binary.sum() > 0:
            axes[row, 1].imshow(np.ma.masked_where(gt_binary == 0, gt_binary),
                               cmap='Greens', alpha=0.3)
        axes[row, 1].set_title('Ground Truth', fontsize=10)
        axes[row, 1].axis('off')
        
        # Column 3: Prediction with Dice
        axes[row, 2].imshow(img_slice, cmap='gray')
        axes[row, 2].contour(pred_binary, colors=[COLORS[class_id]], linewidths=2)
        if pred_binary.sum() > 0:
            axes[row, 2].imshow(np.ma.masked_where(pred_binary == 0, pred_binary),
                               cmap='Reds', alpha=0.3)
        axes[row, 2].set_title(f'Prediction (Dice: {dice:.3f})', fontsize=10)
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved {class_name} figure: {output_path}")


def create_summary_figure(results_json: str, output_path: str):
    """
    Create box plot summary of Dice scores across all test cases
    """
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    # Extract Dice scores per class
    dice_data = {1: [], 2: [], 3: []}  # Kidney, Tumor, Cyst
    
    for case in results['per_case_results']:
        metrics = case['metrics']
        dice_data[1].append(metrics['dice'][1])
        dice_data[2].append(metrics['dice'][2])
        dice_data[3].append(metrics['dice'][3])
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    box_data = [dice_data[1], dice_data[2], dice_data[3]]
    positions = [1, 2, 3]
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    # Color boxes
    colors = [COLORS[1], COLORS[2], COLORS[3]]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Set labels
    ax.set_xticks(positions)
    ax.set_xticklabels(['Kidney', 'Tumor', 'Cyst'], fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Dice Score Distribution Across Test Cases', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Add mean values as text
    for i, (pos, data) in enumerate(zip(positions, box_data)):
        mean_val = np.mean(data)
        ax.text(pos, 0.95, f'μ={mean_val:.3f}', ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved summary figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D segmentation results')
    parser.add_argument('--results-dir', type=str, 
                       default='./output/meddino_vista3d_v3_quality/visualizations',
                       help='Directory with test results')
    parser.add_argument('--data-dir', type=str,
                       default='./kits23/dataset',
                       help='KiTS23 dataset directory')
    parser.add_argument('--output-dir', type=str,
                       default='./output/meddino_vista3d_v3_quality/figures',
                       help='Output directory for figures')
    parser.add_argument('--case', type=str, default=None,
                       help='Specific case to visualize')
    parser.add_argument('--num-cases', type=int, default=5,
                       help='Number of cases to visualize if --case not specified')
    parser.add_argument('--class-focus', type=int, default=None,
                       help='Focus on specific class (1=Kidney, 2=Tumor, 3=Cyst)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Publication-Quality Visualization Generator")
    print("=" * 70)
    
    if args.case:
        # Visualize specific case
        case_dir = data_dir / args.case
        pred_dir = results_dir / args.case
        
        if not case_dir.exists() or not pred_dir.exists():
            print(f"❌ Case {args.case} not found")
            return
        
        print(f"\nGenerating visualizations for {args.case}...")
        
        # Comparison figure
        create_comparison_figure(
            str(case_dir / "imaging.nii.gz"),
            str(case_dir / "segmentation.nii.gz"),
            str(pred_dir / "prediction.nii.gz"),
            str(output_dir / f"{args.case}_comparison.png")
        )
        
        # Class-specific figures
        if args.class_focus:
            create_class_specific_figure(
                str(case_dir / "imaging.nii.gz"),
                str(case_dir / "segmentation.nii.gz"),
                str(pred_dir / "prediction.nii.gz"),
                str(output_dir / f"{args.case}_class{args.class_focus}.png"),
                class_id=args.class_focus
            )
        else:
            # Generate for all classes
            for class_id in [2, 3]:  # Tumor and Cyst
                create_class_specific_figure(
                    str(case_dir / "imaging.nii.gz"),
                    str(case_dir / "segmentation.nii.gz"),
                    str(pred_dir / "prediction.nii.gz"),
                    str(output_dir / f"{args.case}_{CLASS_NAMES[class_id].lower()}.png"),
                    class_id=class_id
                )
    
    else:
        # Visualize multiple cases
        pred_cases = sorted([d for d in results_dir.iterdir() if d.is_dir()])
        
        if not pred_cases:
            print(f"❌ No predictions found in {results_dir}")
            print("   Run test_model_v3.py with --save-viz first!")
            return
        
        # Select cases (best, worst, median)
        print(f"\nFound {len(pred_cases)} cases with predictions")
        print(f"Generating visualizations for {min(args.num_cases, len(pred_cases))} cases...\n")
        
        for case_dir in pred_cases[:args.num_cases]:
            case_name = case_dir.name
            data_case = data_dir / case_name
            
            if not data_case.exists():
                continue
            
            print(f"Processing {case_name}...")
            
            create_comparison_figure(
                str(data_case / "imaging.nii.gz"),
                str(data_case / "segmentation.nii.gz"),
                str(case_dir / "prediction.nii.gz"),
                str(output_dir / f"{case_name}_comparison.png")
            )
    
    # Generate summary if test_results.json exists
    results_json = Path(args.results_dir).parent / "test_results.json"
    if results_json.exists():
        print("\nGenerating summary statistics...")
        create_summary_figure(
            str(results_json),
            str(output_dir / "dice_boxplot_summary.png")
        )
    
    print("\n" + "=" * 70)
    print(f"✅ All visualizations saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
