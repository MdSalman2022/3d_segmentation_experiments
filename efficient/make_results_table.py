"""
Render a publication results table (PNG + PDF) from a WaLM-Net summary.json.

    python make_results_table.py --summary output/walmnet_medium/summary.json
    python make_results_table.py --summary summary.json --out results_table \
                                 --title "WaLM-Net on KiTS23 (validation, 97 cases)"

Reads `full_val_metrics` (dice / iou / hd95 / surface_dice per class) and renders
a clean per-class table: Kidney / Tumor / Cyst / Mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASSES = ["Kidney", "Tumor", "Cyst"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="output/walmnet_medium/summary.json")
    ap.add_argument("--out", default="results_table")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    data = json.loads(Path(args.summary).read_text())
    m = data.get("full_val_metrics") or data.get("metrics") or data
    dice = m["dice"]; iou = m["iou"]; hd95 = m["hd95"]
    sdice = m.get("surface_dice", [float("nan")] * 3)
    n = data.get("n_val_cases_full") or data.get("n_cases") or "?"
    params = data.get("params_millions", "?")

    title = args.title or (f"WaLM-Net — KiTS23 validation ({n} cases, {params}M params)")

    # rows: per class + mean.  cols: Dice / IoU / Surface Dice (%) + HD95 (mm)
    def pct(v): return f"{v*100:.2f}"
    def mm(v): return f"{v:.2f}"
    rows = []
    for i, c in enumerate(CLASSES):
        rows.append([c, pct(dice[i]), pct(iou[i]), pct(sdice[i]), mm(hd95[i])])
    mean_row = ["Mean",
                pct(sum(dice) / 3), pct(sum(iou) / 3),
                pct(sum(sdice) / 3), mm(sum(hd95) / 3)]
    rows.append(mean_row)
    col_labels = ["Class", "Dice (%)", "IoU (%)", "Surface Dice (%)", "HD95 (mm)"]

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 1.6)

    ncol = len(col_labels)
    for (r, cidx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:                                   # header
            cell.set_facecolor("#1f3b57"); cell.set_text_props(color="white", weight="bold")
        elif r == len(rows):                         # mean row (last)
            cell.set_facecolor("#dbe6f0"); cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f7fa")
        if cidx == 0 and r > 0:
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=300)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
