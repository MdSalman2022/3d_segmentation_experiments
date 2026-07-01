"""
FLOPs-vs-Dice efficiency plot: WaLM-Net vs other 3D segmentation models.

    python make_sota_plot.py --out sota_flops_dice

================================ READ THIS =================================
The competitor numbers below are ESTIMATES, not verified KiTS23 results:
  * params / FLOPs  -> taken from the UNETR++ paper (Synapse, 96^3 input).
                       FLOPs depend on input size, so they are only indicative
                       at our 128^3 patch. Params are architecture-fixed.
  * Dice (%)        -> GUESSED KiTS23 mean-foreground Dice (these models are
                       NOT on the KiTS23 leaderboard). Replace with real numbers
                       before any submission.
WaLM-Net params (14.26M) and Dice (75.05% meanFG) are REAL. Its FLOPs is an
estimate -> run `python walmnet_flops.py --mode medium` and paste the value.
The figure is drawn with an "ESTIMATED" watermark so it cannot be mistaken for
verified data. Edit METHODS, then remove the watermark via --final.
===========================================================================
"""
from __future__ import annotations

import argparse

# name, dice(%), flops(G), params(M), marker, color, estimated
METHODS = [
    {"name": "TransUNet",       "dice": 70.0, "flops": 88.0,  "params": 96.07, "marker": "p", "color": "#e8743b", "est": True},
    {"name": "UNETR",           "dice": 71.0, "flops": 82.0,  "params": 92.49, "marker": "D", "color": "#7b2d8e", "est": True},
    {"name": "CoTr",            "dice": 72.0, "flops": 668.0, "params": 41.90, "marker": "<", "color": "#4f81bd", "est": True},
    {"name": "nnFormer",        "dice": 73.0, "flops": 213.0, "params": 150.5, "marker": "s", "color": "#b5b21e", "est": True},
    {"name": "Swin-UNETR",      "dice": 74.0, "flops": 384.0, "params": 62.83, "marker": "X", "color": "#d6299a", "est": True},
    {"name": "nnU-Net",         "dice": 76.0, "flops": 350.0, "params": 30.00, "marker": "^", "color": "#2ca02c", "est": True},
    # ---- WaLM-Net: params & Dice REAL; flops = ESTIMATE (run walmnet_flops.py) ----
    {"name": "WaLM-Net (Ours)", "dice": 75.05, "flops": 140.0, "params": 14.26, "marker": "*", "color": "#8b1a1a", "est": False},
]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sota_flops_dice")
    ap.add_argument("--title", default="KiTS23: accuracy vs. compute (estimated)")
    ap.add_argument("--final", action="store_true",
                    help="remove the ESTIMATED watermark (only once numbers are verified)")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "serif", "axes.grid": True,
                         "grid.alpha": 0.35})
    fig, ax = plt.subplots(figsize=(7.6, 6))

    handles = []
    for m in METHODS:
        ours = m["marker"] == "*"
        ms = 380 if ours else 150
        face = m["color"] if ours else "none"
        ax.scatter(m["flops"], m["dice"], s=ms, marker=m["marker"],
                   facecolor=face, edgecolor=m["color"], linewidths=2.0, zorder=3)
        ax.annotate(f"{m['params']}M", (m["flops"], m["dice"]),
                    textcoords="offset points", xytext=(9, 7), fontsize=10,
                    fontweight="bold" if ours else "normal", color="#222")
        lbl = m["name"] + ("" if not m["est"] else " *")
        handles.append(Line2D([0], [0], marker=m["marker"], color="w",
                              markerfacecolor=face, markeredgecolor=m["color"],
                              markersize=13, label=lbl))

    ax.set_xlabel("FLOPs (G)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Dice Score (%)", fontsize=14, fontweight="bold")
    ax.set_title(args.title, fontsize=13)
    leg = ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=10,
                    title="*  = estimated values")
    leg.get_title().set_fontsize(9)

    if not args.final:
        ax.text(0.5, 0.5, "ESTIMATED", transform=ax.transAxes, fontsize=52,
                color="gray", alpha=0.13, ha="center", va="center", rotation=25,
                fontweight="bold", zorder=0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=300)
    print(f"wrote {args.out}.png / .pdf")
    print("NOTE: competitor Dice are GUESSES; params/FLOPs from UNETR++ (Synapse). "
          "Verify before publishing. Replace WaLM-Net flops with walmnet_flops.py output.")


if __name__ == "__main__":
    main()
