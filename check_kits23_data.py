"""
KiTS23 Dataset Integrity Checker
================================
Checks if all required files (imaging + segmentation) are present.
Run this on your research lab machine to verify dataset completeness.
"""

import os
from pathlib import Path
from collections import defaultdict

# ===== CONFIGURATION =====
KITS23_DIR = Path("./kits23")  # Change this to your dataset path

def check_dataset_integrity(kits23_dir):
    """Check KiTS23 dataset for completeness"""
    kits23_dir = Path(kits23_dir)
    
    if not kits23_dir.exists():
        print(f"❌ Directory not found: {kits23_dir}")
        return
    
    # Find all case folders
    case_dirs = sorted([d for d in kits23_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('case_')])
    
    print("="*60)
    print("KiTS23 DATASET INTEGRITY CHECK")
    print("="*60)
    print(f"Dataset path: {kits23_dir.absolute()}")
    print(f"Total case folders found: {len(case_dirs)}")
    print("="*60)
    
    # Track status
    stats = {
        'complete': [],      # Both imaging and segmentation
        'imaging_only': [],  # Only imaging
        'seg_only': [],      # Only segmentation
        'empty': []          # Neither
    }
    
    file_counts = defaultdict(int)
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        
        has_imaging = (case_dir / "imaging.nii.gz").exists()
        has_seg = (case_dir / "segmentation.nii.gz").exists()
        
        if has_imaging:
            file_counts['imaging'] += 1
        if has_seg:
            file_counts['segmentation'] += 1
        
        if has_imaging and has_seg:
            stats['complete'].append(case_id)
        elif has_imaging:
            stats['imaging_only'].append(case_id)
        elif has_seg:
            stats['seg_only'].append(case_id)
        else:
            stats['empty'].append(case_id)
    
    # Print summary
    print("\n📊 SUMMARY")
    print("-"*40)
    print(f"✅ Complete (imaging + segmentation): {len(stats['complete'])}")
    print(f"🖼️  Imaging only:                      {len(stats['imaging_only'])}")
    print(f"🏷️  Segmentation only:                 {len(stats['seg_only'])}")
    print(f"❌ Empty (missing both):              {len(stats['empty'])}")
    print("-"*40)
    print(f"Total imaging.nii.gz files:      {file_counts['imaging']}")
    print(f"Total segmentation.nii.gz files: {file_counts['segmentation']}")
    
    # Calculate readiness
    ready_for_training = len(stats['complete'])
    total_cases = len(case_dirs)
    completeness = (ready_for_training / total_cases * 100) if total_cases > 0 else 0
    
    print("\n" + "="*60)
    print(f"📈 DATASET COMPLETENESS: {completeness:.1f}%")
    print(f"🚀 READY FOR nnU-Net TRAINING: {ready_for_training} cases")
    print("="*60)
    
    # Show first few problematic cases if any
    if stats['seg_only'] and len(stats['seg_only']) <= 10:
        print(f"\n⚠️  Cases with segmentation only (missing imaging):")
        for c in stats['seg_only'][:10]:
            print(f"    - {c}")
    
    if stats['imaging_only'] and len(stats['imaging_only']) <= 10:
        print(f"\n⚠️  Cases with imaging only (missing segmentation):")
        for c in stats['imaging_only'][:10]:
            print(f"    - {c}")
    
    # Return detailed stats
    return {
        'total_cases': total_cases,
        'complete': len(stats['complete']),
        'imaging_only': len(stats['imaging_only']),
        'seg_only': len(stats['seg_only']),
        'empty': len(stats['empty']),
        'completeness_pct': completeness,
        'complete_cases': stats['complete'],
        'incomplete_cases': stats['imaging_only'] + stats['seg_only'] + stats['empty']
    }

if __name__ == "__main__":
    # Run the check
    result = check_dataset_integrity(KITS23_DIR)
    
    if result and result['completeness_pct'] == 100:
        print("\n✅ Dataset is COMPLETE! Ready for nnU-Net training.")
    elif result and result['complete'] > 0:
        print(f"\n⚠️  Partial dataset: {result['complete']} cases ready for training.")
    else:
        print("\n❌ No complete cases found. Need to download imaging data.")
