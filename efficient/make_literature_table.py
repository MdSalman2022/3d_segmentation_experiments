"""
Render a KiTS23 literature-comparison table (PNG + PDF) with WaLM-Net included.

    python make_literature_table.py --out literature_table

>>> IMPORTANT — these methods use DIFFERENT evaluation setups (splits, k-fold,
metric definitions). They are a LANDSCAPE, not a fair head-to-head. The
"Eval Setup" column is kept on purpose. The only row directly comparable to
WaLM-Net is Mobile U-ViT (same 80/20 public split, per-class Dice). Numbers for
other rows are AS REPORTED in their papers — verify before citing.
"""
from __future__ import annotations

import argparse
import textwrap

# (num, method, year, setup, dice_reported, iou_reported, is_ours)
ROWS = [
    (1, "Two-Stage NN Pipeline (Kidney+Tumor)", "2025", "390 tr / 49 val / 49 test",
     "Kidney+Mass 0.9582; Tumor 0.6261", "Kidney+Mass 0.9216; Tumor 0.5148", False),
    (2, "Dual-Stage AI Model (CT)", "2025", "389 tr / 100 val",
     "Kidney 0.97; Tumor(S) 0.84; (M) 0.89; (L) 0.91", "-", False),
    (3, "Self-Optimizing nnU-NetV2", "2025", "320 tr / 80 val / 89 test",
     "K+T+C 0.8334; T+C 0.6678; Tumor 0.6009; FG 0.7007",
     "K+T+C 0.7705; T+C 0.5621; Tumor 0.5078; FG 0.6135", False),
    (4, "Rel-UNet", "2025", "400 tr/val + 89 test, 5-fold",
     "Overall 0.800; 5-fold 0.811; 3D 0.733; Tumor 0.641", "-", False),
    (5, "Mobile U-ViT  [comparable]", "2025", "80/20 public split",
     "Mean 70.93; Kidney 93.16; Tumor 74.80; Cyst 44.84", "-", False),
    (6, "WaveFormer", "2025", "5-fold CV, 80/20 per fold",
     "Mean 80.91", "-", False),
    (7, "Submanifold Sparse ConvNets", "2025/26", "5-fold CV",
     "Kidney+Masses 95.8; Tumor+Cyst 85.7; Tumor 80.3", "-", False),
    (8, "RBK SSL (Random Block Recon.)", "2026", "10% labeled fine-tune",
     "Avg 0.8027; Kidney 0.9498; Masses 0.7529; Tumor 0.7054", "-", False),
    (9, "ResEnc-Net", "2025", "KiTS23 (details n/a)",
     "Dice 88.87", "-", False),
    ("*", "WaLM-Net (Ours)", "2025", "80/20 split, 97 val (held-out)",
     "Mean 75.05; Kidney 91.78; Tumor 74.97; Cyst 58.39",
     "Mean 64.97; Kidney 85.24; Tumor 63.07; Cyst 46.58", True),
]

COLS = ["#", "Method", "Year", "Eval Setup", "Dice reported (%)", "IoU reported (%)"]
WRAP = {3: 18, 4: 34, 5: 26, 1: 24}   # per-column wrap widths


def _wrap(text, width):
    return "\n".join(textwrap.wrap(text, width)) if text else "-"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="literature_table")
    ap.add_argument("--title", default="KiTS23 literature landscape (reported results)")
    args = ap.parse_args()

    cell_text, ours_idx = [], None
    for i, (num, method, year, setup, dice, iou, ours) in enumerate(ROWS):
        if ours:
            ours_idx = i
        cell_text.append([
            str(num), _wrap(method, WRAP[1]), year,
            _wrap(setup, WRAP[3]), _wrap(dice, WRAP[4]), _wrap(iou, WRAP[5]),
        ])

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=COLS, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    widths = [0.03, 0.22, 0.06, 0.18, 0.31, 0.20]
    for c, w in enumerate(widths):
        for r in range(len(ROWS) + 1):
            tbl[r, c].set_width(w)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_height(0.085)
        cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor("#1f3b57"); cell.set_text_props(color="white", weight="bold")
        elif ours_idx is not None and r == ours_idx + 1:
            cell.set_facecolor("#ffe9a8"); cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f7fa")

    ax.set_title(args.title, fontsize=14, fontweight="bold", pad=14)
    fig.text(0.5, 0.02,
             "Setups differ (splits / k-fold / metric definitions) — NOT a fair head-to-head. "
             "Only Mobile U-ViT shares WaLM-Net's 80/20 per-class setup. Numbers as reported in source papers.",
             ha="center", fontsize=8.5, style="italic", color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=300)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
