"""Generate training curves for WaLM-Net KiTS23 medium run from history.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
hist = json.loads((HERE / "history.json").read_text())["epochs"]

CLASSES = ["Kidney", "Tumor", "Cyst"]
COLORS = ["#2563eb", "#dc2626", "#16a34a"]

ep = [e["epoch"] for e in hist]
loss = [e["train_loss"] for e in hist]

# validation points only (score not null)
vep = [e["epoch"] for e in hist if e["score"] is not None]
dice = {c: [e["metrics"]["dice"][i] for e in hist if e["score"] is not None]
        for i, c in enumerate(CLASSES)}
hd95 = {c: [e["metrics"]["hd95"][i] for e in hist if e["score"] is not None]
        for i, c in enumerate(CLASSES)}
mean_fg = [e["metrics"]["mean_fg_dice"] for e in hist if e["score"] is not None]
rare = [e["metrics"]["rare_mean"] for e in hist if e["score"] is not None]

plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 130})

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("WaLM-Net  KiTS23  (medium run, A100, 44/120 epochs before credit cutoff)",
             fontsize=14, fontweight="bold")

# 1. Train loss
ax[0, 0].plot(ep, loss, color="#7c3aed", lw=2)
ax[0, 0].set(title="Training Loss", xlabel="Epoch", ylabel="Loss")

# 2. Per-class Dice
for c, col in zip(CLASSES, COLORS):
    ax[0, 1].plot(vep, dice[c], marker="o", ms=4, color=col, label=c)
ax[0, 1].set(title="Validation Dice per Class", xlabel="Epoch",
             ylabel="Dice", ylim=(0, 1))
ax[0, 1].legend()

# 3. Aggregate scores
ax[1, 0].plot(vep, mean_fg, marker="s", ms=4, color="#0891b2", label="Mean FG Dice")
ax[1, 0].plot(vep, rare, marker="^", ms=4, color="#ea580c", label="Rare (Tumor+Cyst)")
best_i = max(range(len(rare)), key=lambda i: rare[i])
ax[1, 0].annotate(f"best rare={rare[best_i]:.3f}\n@E{vep[best_i]}",
                  (vep[best_i], rare[best_i]), textcoords="offset points",
                  xytext=(-10, 18), fontsize=9,
                  arrowprops=dict(arrowstyle="->", color="#ea580c"))
ax[1, 0].set(title="Aggregate Dice Scores", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
ax[1, 0].legend()

# 4. HD95 (lower = better)
for c, col in zip(CLASSES, COLORS):
    ax[1, 1].plot(vep, hd95[c], marker="o", ms=4, color=col, label=c)
ax[1, 1].set(title="HD95 per Class (lower better)", xlabel="Epoch",
             ylabel="HD95 (mm)")
ax[1, 1].legend()

plt.tight_layout(rect=(0, 0, 1, 0.97))
out = HERE / "training_curves.png"
plt.savefig(out, bbox_inches="tight")
print(f"saved {out}")

# Standalone hero Dice plot
fig2, a2 = plt.subplots(figsize=(9, 5.5))
for c, col in zip(CLASSES, COLORS):
    a2.plot(vep, dice[c], marker="o", ms=5, lw=2, color=col, label=c)
a2.plot(vep, mean_fg, marker="s", ms=5, lw=2, ls="--", color="#111827",
        label="Mean FG")
a2.set(title="WaLM-Net KiTS23 — Validation Dice (medium, 44 epochs)",
       xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
a2.grid(alpha=0.3)
a2.legend(loc="lower right")
out2 = HERE / "dice_curve.png"
fig2.savefig(out2, bbox_inches="tight", dpi=140)
print(f"saved {out2}")

# print final numbers
print("\nLast val @E", vep[-1])
for c in CLASSES:
    print(f"  {c}: dice={dice[c][-1]:.4f}")
print(f"  mean_fg={mean_fg[-1]:.4f} rare={rare[-1]:.4f}")
print(f"BEST rare={rare[best_i]:.4f} @E{vep[best_i]}")
