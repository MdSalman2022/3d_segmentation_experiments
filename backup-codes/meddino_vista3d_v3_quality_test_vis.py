"""
Auto-Visualize: Automatic Best/Worst Case Selection
===================================================

Automatically selects and visualizes representative cases:
- Top 3 best performers
- 2 median performers  
- Bottom 2 challenging cases

Generates comprehensive publication-quality figures.

Usage:
    python auto_visualize.py
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from visualize_results import (
    create_comparison_figure, create_class_specific_figure,
    create_summary_figure
)
import numpy as np


def select_representative_cases(results_json_path, num_best=3, num_worst=2, num_median=2):
    """Auto-select best/worst/median cases"""
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    case_scores = []
    for case in results['per_case_results']:
        dice = case['metrics']['dice']
        mean_fg = np.mean([dice[1], dice[2], dice[3]])
        case_scores.append({
            'case': case['case'],
            'mean_dice': mean_fg,
            'kidney': dice[1],
            'tumor': dice[2],
            'cyst': dice[3]
        })
    
    case_scores.sort(key=lambda x: x['mean_dice'], reverse=True)
    
    mid = len(case_scores) // 2
    
    return {
        'best': case_scores[:num_best],
        'worst': case_scores[-num_worst:],
        'median': case_scores[mid - num_median//2 : mid + num_median//2 + 1]
    }


def main():
    results_json = Path('./output/meddino_vista3d_v3_quality/test_results.json')
    viz_dir = Path('./output/meddino_vista3d_v3_quality/visualizations')
    data_dir = Path('./kits23/dataset')
    output_dir = Path('./output/meddino_vista3d_v3_quality/publication_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AUTO-VISUALIZATION: Smart Case Selection")
    print("=" * 70)
    
    if not results_json.exists():
        print(f"\n❌ Run test_model_v3.py --save-viz first!")
        return
    
    print("\n📊 Analyzing results...")
    selected = select_representative_cases(results_json)
    
    print("\n🏆 TOP 3 PERFORMERS:")
    for c in selected['best']:
        print(f"  {c['case']}: {c['mean_dice']:.3f} (K:{c['kidney']:.2f} T:{c['tumor']:.2f} C:{c['cyst']:.2f})")
    
    print("\n📊 MEDIAN:")
    for c in selected['median']:
        print(f"  {c['case']}: {c['mean_dice']:.3f}")
    
    print("\n⚠️  CHALLENGING:")
    for c in selected['worst']:
        print(f"  {c['case']}: {c['mean_dice']:.3f}")
    
    print(f"\n{'='*70}")
    print("Generating figures...")
    print(f"{'='*70}\n")
    
    all_cases = selected['best'] + selected['median'] + selected['worst']
    
    for i, case_info in enumerate(all_cases, 1):
        name = case_info['case']
        category = 'best' if case_info in selected['best'] else ('worst' if case_info in selected['worst'] else 'median')
        
        case_dir = data_dir / name
        pred_dir = viz_dir / name
        
        if not pred_dir.exists():
            continue
        
        print(f"[{i}/{len(all_cases)}] {name} ({category}) - Dice: {case_info['mean_dice']:.3f}")
        
        # Comparison
        create_comparison_figure(
            str(case_dir / "imaging.nii.gz"),
            str(case_dir / "segmentation.nii.gz"),
            str(pred_dir / "prediction.nii.gz"),
            str(output_dir / f"{category}_{name}_comparison.png")
        )
        
        # Tumor
        if case_info['tumor'] > 0.1:
            create_class_specific_figure(
                str(case_dir / "imaging.nii.gz"),
                str(case_dir / "segmentation.nii.gz"),
                str(pred_dir / "prediction.nii.gz"),
                str(output_dir / f"{category}_{name}_tumor.png"),
                class_id=2
            )
        
        # Cyst
        if case_info['cyst'] > 0.05:
            create_class_specific_figure(
                str(case_dir / "imaging.nii.gz"),
                str(case_dir / "segmentation.nii.gz"),
                str(pred_dir / "prediction.nii.gz"),
                str(output_dir / f"{category}_{name}_cyst.png"),
                class_id=3
            )
    
    # Summary
    print("\n📈 Creating summary...")
    create_summary_figure(str(results_json), str(output_dir / "00_summary.png"))
    
    # Table
    with open(output_dir / "00_performance_table.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MEDDIN O-VISTA3D V3 PERFORMANCE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Category':<12} {'Case':<15} {'Mean':<8} {'Kidney':<8} {'Tumor':<8} {'Cyst':<8}\n")
        f.write("-" * 80 + "\n")
        
        for cat, cases in [('BEST', selected['best']), ('MEDIAN', selected['median']), ('WORST', selected['worst'])]:
            for c in cases:
                f.write(f"{cat:<12} {c['case']:<15} {c['mean_dice']:.4f}   {c['kidney']:.4f}   {c['tumor']:.4f}   {c['cyst']:.4f}\n")
            f.write("\n")
    
    print(f"\n{'='*70}")
    print(f"✅ DONE! Output: {output_dir}")
    print(f"{'='*70}")
    print("\n📂 Generated:")
    print("  - 00_summary.png (boxplot)")
    print("  - 00_performance_table.txt")
    print("  - best_* (top performers)")
    print("  - median_* (representative)")
    print("  - worst_* (challenging)")


if __name__ == "__main__":
    main()
