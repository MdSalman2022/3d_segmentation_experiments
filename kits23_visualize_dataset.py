"""
KiTS23 Comprehensive Dataset Analysis
Generates detailed figures including class-specific visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.gridspec as gridspec

# Configuration
DATA_DIR = Path("./kits23/dataset")
OUTPUT_DIR = Path("./output/kits23_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'kidney': '#2ecc71',  # Green
    'tumor': '#e74c3c',   # Red
    'cyst': '#3498db',    # Blue
    'background': '#95a5a6'  # Gray
}

def analyze_all_cases():
    """Comprehensive analysis of the entire dataset."""
    cases = sorted(list(DATA_DIR.glob("case_*")))
    print(f"Found {len(cases)} total cases")
    
    # Detailed statistics
    case_info = []
    
    for case_path in tqdm(cases, desc="Analyzing all cases"):
        seg_path = case_path / "segmentation.nii.gz"
        if not seg_path.exists():
            continue
            
        seg_nii = nib.load(seg_path)
        seg = seg_nii.get_fdata()
        spacing = seg_nii.header.get_zooms()
        voxel_vol = np.prod(spacing) / 1000  # mL
        
        info = {
            'case_id': case_path.name,
            'path': case_path,
            'shape': seg.shape,
            'spacing': spacing,
            'has_kidney': (seg == 1).any(),
            'has_tumor': (seg == 2).any(),
            'has_cyst': (seg == 3).any(),
            'kidney_vol': (seg == 1).sum() * voxel_vol,
            'tumor_vol': (seg == 2).sum() * voxel_vol,
            'cyst_vol': (seg == 3).sum() * voxel_vol,
            'total_kidney_region': ((seg == 1) | (seg == 2) | (seg == 3)).sum() * voxel_vol
        }
        case_info.append(info)
    
    return case_info, cases

def plot_comprehensive_overview(case_info):
    """Create a comprehensive 6-panel overview figure."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("KiTS23 Dataset - Comprehensive Analysis", fontsize=18, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract data
    n_total = len(case_info)
    n_kidney = sum(1 for c in case_info if c['has_kidney'])
    n_tumor = sum(1 for c in case_info if c['has_tumor'])
    n_cyst = sum(1 for c in case_info if c['has_cyst'])
    n_tumor_only = sum(1 for c in case_info if c['has_tumor'] and not c['has_cyst'])
    n_cyst_only = sum(1 for c in case_info if c['has_cyst'] and not c['has_tumor'])
    n_both = sum(1 for c in case_info if c['has_tumor'] and c['has_cyst'])
    n_neither = sum(1 for c in case_info if not c['has_tumor'] and not c['has_cyst'])
    
    # 1. Case distribution pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Tumor Only', 'Cyst Only', 'Both', 'Kidney Only']
    sizes = [n_tumor_only, n_cyst_only, n_both, n_neither]
    colors_pie = [COLORS['tumor'], COLORS['cyst'], '#9b59b6', COLORS['kidney']]
    explode = (0.05, 0.05, 0.1, 0)
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title(f'Case Distribution (n={n_total})', fontsize=12, fontweight='bold')
    
    # 2. Class prevalence bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    classes = ['Kidney\n(any)', 'Tumor', 'Cyst']
    counts = [n_kidney, n_tumor, n_cyst]
    percentages = [c/n_total*100 for c in counts]
    bars = ax2.bar(classes, counts, color=[COLORS['kidney'], COLORS['tumor'], COLORS['cyst']], 
                   edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Number of Cases', fontsize=12)
    ax2.set_title('Class Prevalence', fontsize=12, fontweight='bold')
    for bar, count, pct in zip(bars, counts, percentages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, max(counts) * 1.2)
    
    # 3. Volume distribution boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    kidney_vols = [c['kidney_vol'] for c in case_info if c['kidney_vol'] > 0]
    tumor_vols = [c['tumor_vol'] for c in case_info if c['tumor_vol'] > 0]
    cyst_vols = [c['cyst_vol'] for c in case_info if c['cyst_vol'] > 0]
    
    bp = ax3.boxplot([kidney_vols, tumor_vols, cyst_vols], labels=['Kidney', 'Tumor', 'Cyst'],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], [COLORS['kidney'], COLORS['tumor'], COLORS['cyst']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Volume (mL)', fontsize=12)
    ax3.set_title('Volume Distribution by Class', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    
    # 4. Volume statistics table
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    
    stats_data = [
        ['Metric', 'Kidney', 'Tumor', 'Cyst'],
        ['Cases with', f'{n_kidney}', f'{n_tumor}', f'{n_cyst}'],
        ['Prevalence', f'{n_kidney/n_total*100:.1f}%', f'{n_tumor/n_total*100:.1f}%', f'{n_cyst/n_total*100:.1f}%'],
        ['Mean Vol (mL)', f'{np.mean(kidney_vols):.1f}', f'{np.mean(tumor_vols):.1f}' if tumor_vols else 'N/A', 
         f'{np.mean(cyst_vols):.1f}' if cyst_vols else 'N/A'],
        ['Median Vol', f'{np.median(kidney_vols):.1f}', f'{np.median(tumor_vols):.1f}' if tumor_vols else 'N/A',
         f'{np.median(cyst_vols):.1f}' if cyst_vols else 'N/A'],
        ['Max Vol', f'{max(kidney_vols):.1f}', f'{max(tumor_vols):.1f}' if tumor_vols else 'N/A',
         f'{max(cyst_vols):.1f}' if cyst_vols else 'N/A'],
        ['Min Vol', f'{min(kidney_vols):.1f}', f'{min(tumor_vols):.1f}' if tumor_vols else 'N/A',
         f'{min(cyst_vols):.1f}' if cyst_vols else 'N/A'],
    ]
    
    table = ax4.table(cellText=stats_data, loc='center', cellLoc='center',
                      colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    ax4.set_title('Volume Statistics Summary', fontsize=12, fontweight='bold', pad=20)
    
    # 5. Image dimension histogram
    ax5 = fig.add_subplot(gs[1, 1])
    depths = [c['shape'][0] for c in case_info]
    ax5.hist(depths, bins=30, color='#8e44ad', edgecolor='black', alpha=0.7)
    ax5.axvline(np.median(depths), color='red', linestyle='--', label=f'Median: {np.median(depths):.0f}')
    ax5.set_xlabel('Number of Slices (Depth)', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('CT Scan Depth Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    
    # 6. Spacing distribution
    ax6 = fig.add_subplot(gs[1, 2])
    z_spacings = [c['spacing'][0] for c in case_info]
    xy_spacings = [c['spacing'][1] for c in case_info]
    ax6.scatter(xy_spacings, z_spacings, alpha=0.5, c='#16a085', s=30)
    ax6.set_xlabel('X-Y Spacing (mm)', fontsize=12)
    ax6.set_ylabel('Z Spacing (mm)', fontsize=12)
    ax6.set_title('Voxel Spacing Distribution', fontsize=12, fontweight='bold')
    ax6.axhline(np.median(z_spacings), color='red', linestyle='--', alpha=0.7)
    ax6.axvline(np.median(xy_spacings), color='red', linestyle='--', alpha=0.7)
    
    plt.savefig(OUTPUT_DIR / "comprehensive_overview.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'comprehensive_overview.png'}")
    plt.close()

def plot_class_specific_samples(case_info):
    """Generate separate figures for kidney-only, tumor, and cyst cases."""
    
    # Find cases for each category
    kidney_only_cases = [c for c in case_info if c['has_kidney'] and not c['has_tumor'] and not c['has_cyst']]
    tumor_cases = sorted([c for c in case_info if c['has_tumor']], key=lambda x: x['tumor_vol'], reverse=True)
    cyst_cases = sorted([c for c in case_info if c['has_cyst']], key=lambda x: x['cyst_vol'], reverse=True)
    
    print(f"\nFound {len(kidney_only_cases)} kidney-only cases")
    print(f"Found {len(tumor_cases)} tumor cases")
    print(f"Found {len(cyst_cases)} cyst cases")
    
    # 1. Kidney-only samples
    plot_category_samples(kidney_only_cases[:4], "Kidney-Only Cases (No Tumor/Cyst)", 
                         "kidney_samples.png", highlight_class=1)
    
    # 2. Tumor samples (largest tumors)
    plot_category_samples(tumor_cases[:4], "Tumor Cases (Largest by Volume)", 
                         "tumor_samples.png", highlight_class=2)
    
    # 3. Cyst samples (largest cysts)
    plot_category_samples(cyst_cases[:4], "Cyst Cases (Largest by Volume)", 
                         "cyst_samples.png", highlight_class=3)

def plot_category_samples(cases, title, filename, highlight_class):
    """Plot sample slices for a specific category."""
    if len(cases) == 0:
        print(f"No cases found for {title}")
        return
        
    n_samples = min(4, len(cases))
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    fig.suptitle(f"KiTS23 - {title}", fontsize=16, fontweight='bold')
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    class_names = {1: 'Kidney', 2: 'Tumor', 3: 'Cyst'}
    class_colors = {1: [0, 1, 0], 2: [1, 0, 0], 3: [0, 0, 1]}
    
    for row, case in enumerate(cases[:n_samples]):
        case_path = case['path']
        seg_path = case_path / "segmentation.nii.gz"
        img_path = case_path / "imaging.nii.gz"
        
        seg = nib.load(seg_path).get_fdata()
        
        if img_path.exists():
            img = nib.load(img_path).get_fdata()
        else:
            img = np.zeros_like(seg)
        
        # Find best slice for the highlighted class
        class_per_slice = (seg == highlight_class).sum(axis=(1, 2))
        if class_per_slice.max() > 0:
            best_slice = np.argmax(class_per_slice)
        else:
            # Fallback to any foreground
            best_slice = np.argmax((seg > 0).sum(axis=(1, 2)))
        
        # Column 1: CT only
        ax = axes[row, 0]
        ax.imshow(img[best_slice], cmap='gray', vmin=-200, vmax=300)
        ax.set_title(f'{case["case_id"]} - CT (z={best_slice})')
        ax.axis('off')
        
        # Column 2: CT with all overlays
        ax = axes[row, 1]
        ax.imshow(img[best_slice], cmap='gray', vmin=-200, vmax=300)
        overlay = np.zeros((*seg[best_slice].shape, 4))
        overlay[seg[best_slice] == 1] = [0, 1, 0, 0.4]  # Kidney
        overlay[seg[best_slice] == 2] = [1, 0, 0, 0.7]  # Tumor
        overlay[seg[best_slice] == 3] = [0, 0, 1, 0.6]  # Cyst
        ax.imshow(overlay)
        ax.set_title('All Labels Overlay')
        ax.axis('off')
        
        # Column 3: Highlighted class only
        ax = axes[row, 2]
        ax.imshow(img[best_slice], cmap='gray', vmin=-200, vmax=300)
        highlight_overlay = np.zeros((*seg[best_slice].shape, 4))
        highlight_overlay[seg[best_slice] == highlight_class] = [*class_colors[highlight_class], 0.7]
        ax.imshow(highlight_overlay)
        ax.set_title(f'{class_names[highlight_class]} Only (Vol: {case[class_names[highlight_class].lower()+"_vol"]:.1f} mL)')
        ax.axis('off')
        
        # Column 4: Segmentation mask
        ax = axes[row, 3]
        seg_colored = np.zeros((*seg[best_slice].shape, 3))
        seg_colored[seg[best_slice] == 1] = [0.2, 0.8, 0.2]
        seg_colored[seg[best_slice] == 2] = [0.9, 0.2, 0.2]
        seg_colored[seg[best_slice] == 3] = [0.2, 0.2, 0.9]
        ax.imshow(seg_colored)
        ax.set_title('Segmentation Mask')
        ax.axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Kidney'),
        Patch(facecolor='red', alpha=0.7, label='Tumor'),
        Patch(facecolor='blue', alpha=0.7, label='Cyst')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / filename}")
    plt.close()

def plot_tumor_size_analysis(case_info):
    """Detailed analysis of tumor sizes."""
    tumor_cases = [c for c in case_info if c['has_tumor']]
    
    if len(tumor_cases) == 0:
        print("No tumor cases found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("KiTS23 - Tumor Size Analysis", fontsize=16, fontweight='bold')
    
    tumor_vols = [c['tumor_vol'] for c in tumor_cases]
    kidney_vols = [c['kidney_vol'] for c in tumor_cases]
    tumor_ratios = [t/(k+t+0.01)*100 for t, k in zip(tumor_vols, kidney_vols)]
    
    # 1. Tumor volume histogram
    ax = axes[0]
    ax.hist(tumor_vols, bins=30, color=COLORS['tumor'], edgecolor='black', alpha=0.7)
    ax.axvline(np.median(tumor_vols), color='black', linestyle='--', linewidth=2,
               label=f'Median: {np.median(tumor_vols):.1f} mL')
    ax.set_xlabel('Tumor Volume (mL)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Tumor Volume Distribution', fontsize=12)
    ax.legend()
    
    # 2. Tumor as % of kidney region
    ax = axes[1]
    ax.hist(tumor_ratios, bins=30, color='#e67e22', edgecolor='black', alpha=0.7)
    ax.axvline(np.median(tumor_ratios), color='black', linestyle='--', linewidth=2,
               label=f'Median: {np.median(tumor_ratios):.1f}%')
    ax.set_xlabel('Tumor as % of Kidney Region', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Relative Tumor Size', fontsize=12)
    ax.legend()
    
    # 3. Tumor vs Kidney volume scatter
    ax = axes[2]
    ax.scatter(kidney_vols, tumor_vols, alpha=0.6, c=COLORS['tumor'], s=50, edgecolor='black')
    ax.set_xlabel('Kidney Volume (mL)', fontsize=12)
    ax.set_ylabel('Tumor Volume (mL)', fontsize=12)
    ax.set_title('Tumor vs Kidney Volume', fontsize=12)
    
    # Add reference line
    max_val = max(max(kidney_vols), max(tumor_vols))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal volumes')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tumor_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'tumor_analysis.png'}")
    plt.close()

def plot_cyst_analysis(case_info):
    """Detailed analysis of cyst cases."""
    cyst_cases = [c for c in case_info if c['has_cyst']]
    
    if len(cyst_cases) == 0:
        print("No cyst cases found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("KiTS23 - Cyst Analysis", fontsize=16, fontweight='bold')
    
    cyst_vols = [c['cyst_vol'] for c in cyst_cases]
    
    # 1. Cyst volume histogram
    ax = axes[0]
    ax.hist(cyst_vols, bins=20, color=COLORS['cyst'], edgecolor='black', alpha=0.7)
    ax.axvline(np.median(cyst_vols), color='black', linestyle='--', linewidth=2,
               label=f'Median: {np.median(cyst_vols):.1f} mL')
    ax.set_xlabel('Cyst Volume (mL)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Cyst Volume Distribution', fontsize=12)
    ax.legend()
    
    # 2. Co-occurrence with tumor
    ax = axes[1]
    cyst_with_tumor = sum(1 for c in cyst_cases if c['has_tumor'])
    cyst_without_tumor = len(cyst_cases) - cyst_with_tumor
    ax.pie([cyst_with_tumor, cyst_without_tumor], 
           labels=[f'With Tumor\n({cyst_with_tumor})', f'Without Tumor\n({cyst_without_tumor})'],
           colors=[COLORS['tumor'], COLORS['cyst']], autopct='%1.1f%%',
           explode=(0.05, 0), shadow=True)
    ax.set_title('Cyst Cases: Tumor Co-occurrence', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cyst_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'cyst_analysis.png'}")
    plt.close()

def create_detailed_summary(case_info):
    """Create detailed text summary."""
    n_total = len(case_info)
    n_kidney = sum(1 for c in case_info if c['has_kidney'])
    n_tumor = sum(1 for c in case_info if c['has_tumor'])
    n_cyst = sum(1 for c in case_info if c['has_cyst'])
    
    kidney_vols = [c['kidney_vol'] for c in case_info if c['kidney_vol'] > 0]
    tumor_vols = [c['tumor_vol'] for c in case_info if c['tumor_vol'] > 0]
    cyst_vols = [c['cyst_vol'] for c in case_info if c['cyst_vol'] > 0]
    
    summary = f"""
================================================================================
              KiTS23 COMPREHENSIVE DATASET ANALYSIS REPORT
================================================================================

DATASET OVERVIEW
----------------
Total CT Scans: {n_total}
Image Modality: Contrast-enhanced CT
Task: Multi-class Semantic Segmentation

CLASS DISTRIBUTION
------------------
                    Count       Prevalence
Kidney (any):       {n_kidney:>5}       {n_kidney/n_total*100:>6.1f}%
Tumor:              {n_tumor:>5}       {n_tumor/n_total*100:>6.1f}%
Cyst:               {n_cyst:>5}       {n_cyst/n_total*100:>6.1f}%

CASE BREAKDOWN
--------------
Kidney only (no lesion):  {sum(1 for c in case_info if c['has_kidney'] and not c['has_tumor'] and not c['has_cyst'])} cases
Tumor only:               {sum(1 for c in case_info if c['has_tumor'] and not c['has_cyst'])} cases
Cyst only:                {sum(1 for c in case_info if c['has_cyst'] and not c['has_tumor'])} cases
Both tumor and cyst:      {sum(1 for c in case_info if c['has_tumor'] and c['has_cyst'])} cases

VOLUME STATISTICS (mL)
----------------------
Structure   Mean      Median    Min       Max       Std
Kidney      {np.mean(kidney_vols):>8.1f}  {np.median(kidney_vols):>8.1f}  {min(kidney_vols):>8.1f}  {max(kidney_vols):>8.1f}  {np.std(kidney_vols):>8.1f}
Tumor       {np.mean(tumor_vols):>8.1f}  {np.median(tumor_vols):>8.1f}  {min(tumor_vols):>8.1f}  {max(tumor_vols):>8.1f}  {np.std(tumor_vols):>8.1f}
Cyst        {np.mean(cyst_vols):>8.1f}  {np.median(cyst_vols):>8.1f}  {min(cyst_vols):>8.1f}  {max(cyst_vols):>8.1f}  {np.std(cyst_vols):>8.1f}

IMAGE PROPERTIES
----------------
Depth (slices): Mean={np.mean([c['shape'][0] for c in case_info]):.0f}, Range=[{min(c['shape'][0] for c in case_info)}-{max(c['shape'][0] for c in case_info)}]
In-plane size:  Typically 512×512 voxels
Z-spacing:      Mean={np.mean([c['spacing'][0] for c in case_info]):.2f}mm, Range=[{min(c['spacing'][0] for c in case_info):.2f}-{max(c['spacing'][0] for c in case_info):.2f}]mm
XY-spacing:     Mean={np.mean([c['spacing'][1] for c in case_info]):.2f}mm

KEY CHALLENGES
--------------
1. CLASS IMBALANCE: Tumor is {np.mean(kidney_vols)/np.mean(tumor_vols):.1f}x smaller than kidney on average
2. RARE CYSTS: Only {n_cyst/n_total*100:.1f}% of cases contain cysts
3. VARIABLE SPACING: Z-spacing varies significantly, requiring careful resampling
4. SMALL STRUCTURES: Median tumor volume is only {np.median(tumor_vols):.1f} mL

MODEL CONSIDERATIONS
--------------------
- Use class-weighted loss functions
- Oversample minority classes (tumor, cyst)
- Apply heavy data augmentation
- Consider multi-scale architectures
- Post-processing for connectivity

================================================================================
Generated: KiTS23 Dataset Analysis Tool
================================================================================
"""
    
    with open(OUTPUT_DIR / "detailed_analysis_report.txt", 'w') as f:
        f.write(summary)
    print(f"Saved: {OUTPUT_DIR / 'detailed_analysis_report.txt'}")

def main():
    print("=" * 60)
    print("KiTS23 Comprehensive Dataset Analysis")
    print("=" * 60)
    
    # Analyze all cases
    case_info, cases = analyze_all_cases()
    
    print("\nGenerating comprehensive figures...")
    
    # Generate all figures
    plot_comprehensive_overview(case_info)
    plot_class_specific_samples(case_info)
    plot_tumor_size_analysis(case_info)
    plot_cyst_analysis(case_info)
    create_detailed_summary(case_info)
    
    print(f"\n{'='*60}")
    print(f"✅ All figures saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. comprehensive_overview.png  - 6-panel dataset overview")
    print("  2. kidney_samples.png          - Kidney-only case examples")
    print("  3. tumor_samples.png           - Tumor case examples (largest)")
    print("  4. cyst_samples.png            - Cyst case examples (largest)")
    print("  5. tumor_analysis.png          - Tumor size analysis")
    print("  6. cyst_analysis.png           - Cyst analysis")
    print("  7. detailed_analysis_report.txt - Complete text report")

if __name__ == "__main__":
    main()
