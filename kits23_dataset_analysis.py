"""
KiTS23 Dataset Analysis Script
Analyzes tumor and cyst sizes to determine optimal patch size and resolution.

Usage:
    python kits23_dataset_analysis.py
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
import json

# Configuration
KITS23_DIR = "./kits23/dataset"

def get_connected_components_stats(mask, spacing):
    """Get statistics for each connected component in a mask."""
    if mask.sum() == 0:
        return []
    
    labeled, num_components = ndimage.label(mask)
    stats = []
    
    for i in range(1, num_components + 1):
        component = (labeled == i)
        
        # Get voxel count and volume
        voxel_count = component.sum()
        volume_mm3 = voxel_count * np.prod(spacing)
        
        # Get bounding box dimensions
        coords = np.where(component)
        if len(coords[0]) == 0:
            continue
            
        dims_voxels = [
            coords[0].max() - coords[0].min() + 1,
            coords[1].max() - coords[1].min() + 1,
            coords[2].max() - coords[2].min() + 1
        ]
        dims_mm = [d * s for d, s in zip(dims_voxels, spacing)]
        
        # Equivalent diameter (sphere of same volume)
        equiv_diameter_mm = 2 * ((3 * volume_mm3) / (4 * np.pi)) ** (1/3)
        
        stats.append({
            "voxel_count": int(voxel_count),
            "volume_mm3": float(volume_mm3),
            "dims_mm": dims_mm,
            "max_dim_mm": max(dims_mm),
            "min_dim_mm": min(dims_mm),
            "equiv_diameter_mm": equiv_diameter_mm
        })
    
    return stats


def analyze_dataset(data_dir):
    """Analyze all cases in the dataset."""
    data_dir = Path(data_dir)
    
    # Find all cases
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    
    print(f"Found {len(cases)} cases")
    print("=" * 70)
    
    # Storage for all statistics
    all_stats = {
        "kidney": [],
        "tumor": [],
        "cyst": []
    }
    
    case_count = {
        "with_tumor": 0,
        "with_cyst": 0,
        "with_kidney": 0
    }
    
    # Analyze each case
    for case_dir in tqdm(cases, desc="Analyzing cases"):
        seg_path = case_dir / "segmentation.nii.gz"
        if not seg_path.exists():
            continue
        
        # Load segmentation
        seg_nii = nib.load(seg_path)
        seg = seg_nii.get_fdata().astype(np.int32)
        spacing = seg_nii.header.get_zooms()[:3]
        
        # Analyze each class
        # Label 1: Kidney, Label 2: Tumor, Label 3: Cyst
        
        # Kidney (label >= 1, includes tumor and cyst inside)
        kidney_mask = (seg >= 1).astype(np.uint8)
        kidney_stats = get_connected_components_stats(kidney_mask, spacing)
        if kidney_stats:
            all_stats["kidney"].extend(kidney_stats)
            case_count["with_kidney"] += 1
        
        # Tumor (label 2)
        tumor_mask = (seg == 2).astype(np.uint8)
        tumor_stats = get_connected_components_stats(tumor_mask, spacing)
        if tumor_stats:
            all_stats["tumor"].extend(tumor_stats)
            case_count["with_tumor"] += 1
        
        # Cyst (label 3)
        cyst_mask = (seg == 3).astype(np.uint8)
        cyst_stats = get_connected_components_stats(cyst_mask, spacing)
        if cyst_stats:
            all_stats["cyst"].extend(cyst_stats)
            case_count["with_cyst"] += 1
    
    return all_stats, case_count, len(cases)


def compute_summary(stats_list):
    """Compute summary statistics."""
    if not stats_list:
        return None
    
    volumes = [s["volume_mm3"] for s in stats_list]
    equiv_diams = [s["equiv_diameter_mm"] for s in stats_list]
    max_dims = [s["max_dim_mm"] for s in stats_list]
    min_dims = [s["min_dim_mm"] for s in stats_list]
    
    return {
        "count": len(stats_list),
        "volume_mm3": {
            "min": float(np.min(volumes)),
            "p5": float(np.percentile(volumes, 5)),
            "p25": float(np.percentile(volumes, 25)),
            "median": float(np.median(volumes)),
            "p75": float(np.percentile(volumes, 75)),
            "p95": float(np.percentile(volumes, 95)),
            "max": float(np.max(volumes)),
        },
        "equiv_diameter_mm": {
            "min": float(np.min(equiv_diams)),
            "p5": float(np.percentile(equiv_diams, 5)),
            "p25": float(np.percentile(equiv_diams, 25)),
            "median": float(np.median(equiv_diams)),
            "p75": float(np.percentile(equiv_diams, 75)),
            "p95": float(np.percentile(equiv_diams, 95)),
            "max": float(np.max(equiv_diams)),
        },
        "max_dimension_mm": {
            "min": float(np.min(max_dims)),
            "p5": float(np.percentile(max_dims, 5)),
            "p25": float(np.percentile(max_dims, 25)),
            "median": float(np.median(max_dims)),
            "p75": float(np.percentile(max_dims, 75)),
            "p95": float(np.percentile(max_dims, 95)),
            "max": float(np.max(max_dims)),
        },
        "min_dimension_mm": {
            "min": float(np.min(min_dims)),
            "p5": float(np.percentile(min_dims, 5)),
            "median": float(np.median(min_dims)),
        }
    }


def recommend_parameters(summaries):
    """Recommend optimal spacing and patch size based on analysis."""
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)
    
    # For tumors and cysts
    tumor_min = summaries["tumor"]["min_dimension_mm"]["p5"] if summaries["tumor"] else 10
    cyst_min = summaries["cyst"]["min_dimension_mm"]["p5"] if summaries["cyst"] else 10
    
    smallest_lesion = min(tumor_min, cyst_min)
    
    # Rule: smallest lesion should span at least 2-3 voxels
    recommended_spacing = smallest_lesion / 3
    # Round to nice number
    if recommended_spacing < 0.5:
        recommended_spacing = 0.5
    elif recommended_spacing < 0.8:
        recommended_spacing = 0.75
    elif recommended_spacing < 1.2:
        recommended_spacing = 1.0
    else:
        recommended_spacing = 1.5
    
    # For patch size: should contain median tumor + context
    tumor_median = summaries["tumor"]["max_dimension_mm"]["median"] if summaries["tumor"] else 50
    recommended_patch_xy = int((tumor_median * 3) / recommended_spacing)
    # Round to multiple of 32
    recommended_patch_xy = max(96, min(256, ((recommended_patch_xy + 31) // 32) * 32))
    
    # Z dimension can be smaller due to anisotropic data
    recommended_patch_z = max(64, recommended_patch_xy // 2)
    # Round to multiple of 16
    recommended_patch_z = ((recommended_patch_z + 15) // 16) * 16
    
    print(f"\n📐 Smallest lesion (5th percentile): {smallest_lesion:.1f} mm")
    print(f"   → Need at least 2-3 voxels to capture")
    
    print(f"\n🎯 RECOMMENDED STAGE 2 PARAMETERS:")
    print(f"   spacing: ({recommended_spacing:.1f}, {recommended_spacing:.1f}, {recommended_spacing * 1.5:.1f})")
    print(f"   patch_size: ({recommended_patch_xy}, {recommended_patch_xy}, {recommended_patch_z})")
    
    # Stage 1 recommendations (for kidney - much larger)
    kidney_median = summaries["kidney"]["max_dimension_mm"]["median"] if summaries["kidney"] else 150
    stage1_spacing = 2.0  # Low res is fine for kidney
    stage1_patch = 160  # Larger patches for kidney
    
    print(f"\n🔍 RECOMMENDED STAGE 1 PARAMETERS:")
    print(f"   spacing: (2.0, 2.0, 2.0)")
    print(f"   patch_size: ({stage1_patch}, {stage1_patch}, {stage1_patch // 2})")
    
    return {
        "stage1": {
            "spacing": (2.0, 2.0, 2.0),
            "patch_size": (stage1_patch, stage1_patch, stage1_patch // 2)
        },
        "stage2": {
            "spacing": (recommended_spacing, recommended_spacing, recommended_spacing * 1.5),
            "patch_size": (recommended_patch_xy, recommended_patch_xy, recommended_patch_z)
        }
    }


def main():
    print("\n" + "=" * 70)
    print("  KiTS23 DATASET ANALYSIS")
    print("=" * 70)
    
    # Analyze dataset
    all_stats, case_count, total_cases = analyze_dataset(KITS23_DIR)
    
    # Compute summaries
    summaries = {}
    for label_name in ["kidney", "tumor", "cyst"]:
        summaries[label_name] = compute_summary(all_stats[label_name])
    
    # Print results
    print("\n" + "=" * 70)
    print("  DATASET SUMMARY")
    print("=" * 70)
    print(f"\nTotal cases: {total_cases}")
    print(f"Cases with kidney: {case_count['with_kidney']}")
    print(f"Cases with tumor: {case_count['with_tumor']}")
    print(f"Cases with cyst: {case_count['with_cyst']}")
    
    for label_name in ["kidney", "tumor", "cyst"]:
        summary = summaries[label_name]
        if not summary:
            print(f"\n📊 {label_name.upper()}: No instances found")
            continue
            
        print(f"\n📊 {label_name.upper()} ({summary['count']} instances):")
        print(f"   Volume (mm³):")
        print(f"      Min: {summary['volume_mm3']['min']:.1f}")
        print(f"      5th percentile: {summary['volume_mm3']['p5']:.1f}")
        print(f"      Median: {summary['volume_mm3']['median']:.1f}")
        print(f"      95th percentile: {summary['volume_mm3']['p95']:.1f}")
        print(f"      Max: {summary['volume_mm3']['max']:.1f}")
        
        print(f"   Equivalent Diameter (mm):")
        print(f"      Min: {summary['equiv_diameter_mm']['min']:.1f}")
        print(f"      5th percentile: {summary['equiv_diameter_mm']['p5']:.1f}")
        print(f"      Median: {summary['equiv_diameter_mm']['median']:.1f}")
        print(f"      95th percentile: {summary['equiv_diameter_mm']['p95']:.1f}")
        print(f"      Max: {summary['equiv_diameter_mm']['max']:.1f}")
        
        print(f"   Max Dimension (mm):")
        print(f"      Min: {summary['max_dimension_mm']['min']:.1f}")
        print(f"      Median: {summary['max_dimension_mm']['median']:.1f}")
        print(f"      Max: {summary['max_dimension_mm']['max']:.1f}")
    
    # Make recommendations
    recommendations = recommend_parameters(summaries)
    
    # Save results
    output = {
        "case_count": case_count,
        "total_cases": total_cases,
        "summaries": summaries,
        "recommendations": recommendations
    }
    
    output_path = Path("kits23_dataset_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Full results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
