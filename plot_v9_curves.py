"""
Training curves for SwinUNETR V9 showcase runs.
Run: python plot_v9_curves.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("./output/v9_curves")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────

run1 = {
    "label": "Run 1 (35 cases, 47 epochs)",
    "s1_n": 12,
    "train_loss": [
        0.9503, 0.8353, 0.7999, 0.7696, 0.7883, 0.7564, 0.7463, 0.7464,
        0.7467, 0.7440, 0.7151, 0.7303,
        0.7359, 0.7256, 0.7227, 0.7177, 0.6871, 0.6750, 0.6769, 0.6836,
        0.6539, 0.6780, 0.6637, 0.6486, 0.6337, 0.6584, 0.6628, 0.6043,
        0.6029, 0.5904, 0.6074, 0.5824, 0.6127, 0.5827, 0.5840, 0.5701,
        0.5697, 0.5663, 0.5352, 0.5980, 0.5347, 0.5735, 0.5805, 0.5631,
        0.5612, 0.5361, 0.5749,
    ],
    # (global_epoch, kidney, tumor, cyst, mean_fg)
    "val": [
        (4,  0.0679, 0.0007, 0.0000, 0.0228),
        (8,  0.1031, 0.0008, 0.0000, 0.0346),
        (12, 0.1050, 0.0009, 0.0000, 0.0353),
        (17, 0.1625, 0.0016, 0.0000, 0.0547),
        (22, 0.1903, 0.0007, 0.0000, 0.0637),
        (27, 0.2044, 0.0008, 0.0303, 0.0785),
        (32, 0.2497, 0.0020, 0.0020, 0.0846),
        (37, 0.4142, 0.0015, 0.0552, 0.1570),
        (42, 0.3056, 0.0017, 0.0000, 0.1024),
        (47, 0.3009, 0.0029, 0.0790, 0.1276),
    ],
    "best": 0.1570,
}

run2 = {
    "label": "Run 2 (100 cases, 38 epochs)",
    "s1_n": 8,
    "train_loss": [
        0.8907, 0.8390, 0.8029, 0.7982, 0.7818, 0.7619, 0.7425, 0.7356,
        0.7530, 0.7377, 0.7318, 0.7216, 0.7032, 0.6936, 0.6991, 0.6769,
        0.6810, 0.6578, 0.6675, 0.6478, 0.6249, 0.6455, 0.6373, 0.6126,
        0.6328, 0.6065, 0.5879, 0.5907, 0.6028, 0.5744, 0.5954, 0.5855,
        0.5957, 0.5558, 0.5682, 0.5581, 0.5632, 0.5618,
    ],
    "val": [
        (4,  0.1538, 0.0072, 0.0000, 0.0537),
        (8,  0.0690, 0.0002, 0.0000, 0.0230),
        (13, 0.1250, 0.0017, 0.0089, 0.0452),
        (18, 0.1796, 0.0204, 0.0000, 0.0667),
        (23, 0.2086, 0.0023, 0.0019, 0.0710),
        (28, 0.3195, 0.0012, 0.0000, 0.1069),
        (33, 0.5272, 0.0257, 0.0000, 0.1843),
        (38, 0.3538, 0.0012, 0.0000, 0.1184),
    ],
    "best": 0.1843,
}


# ── plot 1: train loss ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("SwinUNETR V9 (LLM bottleneck) — Training Loss", fontsize=12)

for ax, run, col in zip(axes, [run1, run2], ["steelblue", "seagreen"]):
    eps = list(range(1, len(run["train_loss"]) + 1))
    ax.plot(eps, run["train_loss"], color=col, linewidth=1.6)
    ax.axvline(run["s1_n"] + 0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(run["s1_n"] / 2, 0.98, "Stage 1", ha="center", va="top",
            fontsize=8, color="gray", transform=ax.get_xaxis_transform())
    ax.text(run["s1_n"] + (len(eps) - run["s1_n"]) / 2, 0.98, "Stage 2",
            ha="center", va="top", fontsize=8, color="gray",
            transform=ax.get_xaxis_transform())
    ax.set_title(run["label"], fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(0.45, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "01_train_loss.png", dpi=130, bbox_inches="tight")
plt.close()


# ── plot 2: meanfg dice ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("MeanFG Dice progress — increasing cases helps significantly", fontsize=11)

for run, col, marker in [(run1, "steelblue", "o"), (run2, "seagreen", "s")]:
    ve = [v[0] for v in run["val"]]
    vd = [v[4] for v in run["val"]]
    ax.plot(ve, vd, color=col, marker=marker, markersize=5,
            linewidth=1.6, label=run["label"])
    best_i = int(np.argmax(vd))
    ax.annotate(f'best={vd[best_i]:.4f}',
                xy=(ve[best_i], vd[best_i]),
                xytext=(ve[best_i] - 6, vd[best_i] + 0.012),
                fontsize=8, color=col,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.7))

ax.set_xlabel("Epoch")
ax.set_ylabel("MeanFG Dice")
ax.set_ylim(-0.005, 0.25)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)

ax.text(0.98, 0.05,
        "Showcase runs only (~5% of full training steps)",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom", color="gray")

plt.tight_layout()
plt.savefig(OUT_DIR / "02_meanfg_dice.png", dpi=130, bbox_inches="tight")
plt.close()


# ── plot 3: per-class dice, run 2 only ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Per-class Dice — Run 2 (100 cases, 38 epochs)", fontsize=11)

ve = [v[0] for v in run2["val"]]
for cls, col, mk, idx in [
    ("Kidney",  "steelblue",    "o", 1),
    ("Tumor",   "tomato",       "^", 2),
    ("Cyst",    "seagreen",     "s", 3),
    ("MeanFG",  "mediumpurple", "D", 4),
]:
    vals = [v[idx] for v in run2["val"]]
    lw = 2.0 if cls == "MeanFG" else 1.4
    ls = "--" if cls == "MeanFG" else "-"
    ax.plot(ve, vals, color=col, marker=mk, markersize=5,
            linewidth=lw, linestyle=ls, label=cls)

ax.axvline(run2["s1_n"] + 0.5, color="gray", linestyle=":", linewidth=1.0)
ax.text(run2["s1_n"] + 0.7, 0.005, "S1→S2", fontsize=8, color="gray")
ax.set_xlabel("Epoch")
ax.set_ylabel("Dice")
ax.set_ylim(-0.01, 0.60)
ax.legend(fontsize=9, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(OUT_DIR / "03_per_class_dice.png", dpi=130, bbox_inches="tight")
plt.close()


# ── plot 4: bar summary ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_title("Best Dice achieved — showcase vs full training target", fontsize=11)

r1b = max(run1["val"], key=lambda x: x[4])
r2b = max(run2["val"], key=lambda x: x[4])

classes = ["Kidney", "Tumor", "Cyst", "MeanFG"]
r1v = [r1b[1], r1b[2], r1b[3], r1b[4]]
r2v = [r2b[1], r2b[2], r2b[3], r2b[4]]

x = np.arange(len(classes))
w = 0.3
ax.bar(x - w/2, r1v, w, label="Run 1 (35 cases)", color="steelblue", alpha=0.75)
ax.bar(x + w/2, r2v, w, label="Run 2 (100 cases)", color="seagreen", alpha=0.75)

for bars, vals in [(x - w/2, r1v), (x + w/2, r2v)]:
    for xi, v in zip(bars, vals):
        if v > 0.005:
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

ax.axhline(0.70, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
ax.text(3.3, 0.71, "target (full training)", fontsize=8, color="red", ha="right")

ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("Dice Score")
ax.set_ylim(0, 0.75)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "04_summary_bar.png", dpi=130, bbox_inches="tight")
plt.close()

print(f"Done. Plots saved to {OUT_DIR}/")
print(f"  Run 1 best MeanFG: {run1['best']:.4f}")
print(f"  Run 2 best MeanFG: {run2['best']:.4f}  (+{(run2['best']-run1['best'])/run1['best']*100:.0f}% from more data)")
