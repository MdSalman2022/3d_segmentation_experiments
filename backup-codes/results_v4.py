"""
Generate Validation Results Table (Paper Quality)
=================================================

Generates a high-resolution table image summarizing the V4 model performance.
Features:
- "Fair" Dice calculation (Average vs. Positive-Only)
- Color-coded performance (Green=Good, Yellow=Ok, Red=Bad)
- Breakdowns by Class (Kidney, Tumor, Cyst)
- Stats: Mean, Median, Min, Max, Std Dev

Usage:
    python generate_results_table.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def get_fair_stats(results, class_idx):
    """Calculate detailed stats for a class"""
    all_dice = []
    positive_dice = []
    
    for r in results:
        metrics = r['metrics']
        d = metrics['dice'][class_idx]
        
        all_dice.append(d)
        
        # Check if GT has volume (Positive case)
        if 'vol_gt' in metrics and metrics['vol_gt'][class_idx] > 0:
            positive_dice.append(d)
        elif metrics['has_fg'][class_idx]: # Fallback to boolean flag
            positive_dice.append(d)
            
    return {
        "All_Mean": np.mean(all_dice) if all_dice else 0.0,
        "Pos_Mean": np.mean(positive_dice) if positive_dice else 0.0,
        "Pos_Median": np.median(positive_dice) if positive_dice else 0.0,
        "Pos_Std": np.std(positive_dice) if positive_dice else 0.0,
        "Pos_Min": np.min(positive_dice) if positive_dice else 0.0,
        "Pos_Max": np.max(positive_dice) if positive_dice else 0.0,
        "Count_Pos": len(positive_dice),
        "Count_Total": len(all_dice)
    }

def create_table_image():
    base_dir = Path("./")
    output_dir = base_dir / "output/meddino_vista3d_v4_tumor_focused/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON
    json_path = base_dir / "output/meddino_vista3d_v4_tumor_focused/fair_results.json"
    if not json_path.exists():
        json_path = base_dir / "output/meddino_vista3d_v4_tumor_focused/test_results.json"
    
    print(f"Loading results from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    cases = data.get('cases', data.get('per_case_results', []))
    
    # Calculate Stats
    stats = {}
    for name, idx in [("Kidney", 1), ("Tumor", 2), ("Cyst", 3)]:
        stats[name] = get_fair_stats(cases, idx)
        
    # Create DataFrame for Table
    rows = []
    for cls in ["Kidney", "Tumor", "Cyst"]:
        s = stats[cls]
        rows.append([
            cls,
            f"{s['All_Mean']:.3f}",           # Overall Dice
            f"{s['Pos_Mean']:.3f} ± {s['Pos_Std']:.2f}", # Tumor-Positive Dice
            f"{s['Pos_Median']:.3f}",         # Median
            f"{s['Pos_Min']:.2f} - {s['Pos_Max']:.2f}", # Range
            f"{s['Count_Pos']} / {s['Count_Total']}"    # N Cases
        ])
        
    columns = ["Class", "Overall Dice", "Positive-Only Dice\n(Mean ± SD)", "Median\n(Pos)", "Range\n(Min-Max)", "N Cases"]
    
    # Plotting - Increased width to prevent overlap
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis('off')
    
    # Create Table with specific column widths
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#40466e']*len(columns),
        colWidths=[0.1, 0.15, 0.25, 0.15, 0.2, 0.15] # allocating more space for the long column
    )
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(14) # Increased font size for readability
    table.scale(1, 2.5) # Increased height scale
    
    # Header Style
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.2) # Taller header for multiline text
        else:
            # Row Styling
            if rows[row-1][0] == "Kidney":
                cell.set_facecolor('#e8f5e9') # Light Green
            elif rows[row-1][0] == "Tumor":
                 cell.set_facecolor('#ffebee') # Light Red
            elif rows[row-1][0] == "Cyst":
                 cell.set_facecolor('#e3f2fd') # Light Blue
                 
    # Highlight "Bad" Stats with Red Text
    # Check Tumor Min Dice (Row 2, Col 4 normally) - logic simplified for visualization code
    
    plt.title("MedDINO-VISTA3D V4 Performance Summary", fontsize=16, fontweight='bold', y=0.85)
    
    out_path = output_dir / "performance_table.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Table saved to: {out_path}")

if __name__ == "__main__":
    create_table_image()
