"""
Generate paper/presentation tables and graphs from training + evaluation outputs.

This script does not run inference. It only reads existing artifacts such as:
  - history.json
  - summary.json
  - showcase_test/eval_results.json or eval_results.json
  - showcase_test/eval_per_case.csv
  - showcase_test/confusion_matrix.json

Examples:
  python generate_training_eval_report.py --output-dir output/swin_v12_dinov3_tumor
  python generate_training_eval_report.py --output-dir output/swin_v12_dinov3_tumor --eval-dir output/swin_v12_dinov3_tumor/showcase_val
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]
FG_CLASSES = ["kidney", "tumor", "cyst"]
COLORS = {
    "kidney": "#00A6D6",
    "tumor": "#E63946",
    "cyst": "#F4A261",
    "mean_fg": "#2A9D8F",
    "train": "#264653",
    "val": "#8D99AE",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: Any, digits: int = 4) -> str:
    value = _safe_float(value)
    if math.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _write_md_table(path: Path, rows: list[dict[str, Any]], columns: list[str], title_map: dict[str, str] | None = None) -> None:
    title_map = title_map or {}
    headers = [title_map.get(col, col) for col in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def _save_table_png(path: Path, rows: list[dict[str, Any]], columns: list[str], title: str, max_rows: int = 18) -> None:
    rows = rows[:max_rows]
    fig_h = max(2.6, 0.45 * (len(rows) + 2))
    fig_w = max(8.5, 1.35 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=12)
    cell_text = [[str(row.get(col, "")) for col in columns] for row in rows]
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F3F4F6")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _history_rows(history: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in history.get("epochs", []):
        sw = item.get("sw_metrics") or {}
        dice = sw.get("dice") or [None, None, None, None]
        rows.append({
            "epoch": int(item.get("epoch", 0)),
            "train_loss": _safe_float(item.get("train_loss")),
            "val_loss": _safe_float(item.get("val_loss")),
            "kidney_dice": _safe_float(dice[1] if len(dice) > 1 else None),
            "tumor_dice": _safe_float(dice[2] if len(dice) > 2 else None),
            "cyst_dice": _safe_float(dice[3] if len(dice) > 3 else None),
            "rare_mean": _safe_float(sw.get("rare_mean")),
            "mean_fg_dice": _safe_float(sw.get("mean_fg_dice")),
            "selection_metric": _safe_float(sw.get("selection_metric")),
            "epoch_time_min": _safe_float(item.get("epoch_time_min")),
            "is_best": bool(item.get("is_best", False)),
        })
    return rows


def _plot_training_curves(rows: list[dict[str, Any]], report_dir: Path) -> list[str]:
    written = []
    if not rows:
        return written
    epochs = np.array([r["epoch"] for r in rows], dtype=float)
    train_loss = np.array([r["train_loss"] for r in rows], dtype=float)
    val_loss = np.array([r["val_loss"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
    ax.plot(epochs, train_loss, color=COLORS["train"], linewidth=2.2, label="Train loss")
    valid = ~np.isnan(val_loss)
    if valid.any():
        ax.plot(epochs[valid], val_loss[valid], "o-", color=COLORS["val"], linewidth=2.2, label="Validation loss")
    ax.set_title("Training and Validation Loss", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = report_dir / "training_loss_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    written.append(str(out))

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
    for key, label in [
        ("kidney_dice", "Kidney"),
        ("tumor_dice", "Tumor"),
        ("cyst_dice", "Cyst"),
        ("mean_fg_dice", "Mean FG"),
    ]:
        values = np.array([r[key] for r in rows], dtype=float)
        valid = ~np.isnan(values)
        if valid.any():
            color = COLORS.get(label.lower().replace(" ", "_"), None)
            ax.plot(epochs[valid], values[valid], "o-", linewidth=2.2, label=label, color=color)
    ax.set_title("Sliding-Window Validation Dice Progress", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper left")
    fig.tight_layout()
    out = report_dir / "validation_dice_progress.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    written.append(str(out))

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
    tumor = np.array([r["tumor_dice"] for r in rows], dtype=float)
    valid = ~np.isnan(tumor)
    if valid.any():
        ax.plot(epochs[valid], tumor[valid], "o-", linewidth=2.5, color=COLORS["tumor"], label="Tumor Dice")
        best_idx = np.nanargmax(tumor)
        ax.scatter([epochs[best_idx]], [tumor[best_idx]], s=110, color="#111827", zorder=5, label="Best")
        ax.annotate(
            f"Best {tumor[best_idx]:.3f} @ E{int(epochs[best_idx])}",
            xy=(epochs[best_idx], tumor[best_idx]),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=10,
            arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 1},
        )
    ax.set_title("Tumor Dice Improvement", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tumor Dice")
    ax.set_ylim(0, max(0.35, min(1.0, np.nanmax(tumor) + 0.1 if valid.any() else 1.0)))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = report_dir / "tumor_dice_progress.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    written.append(str(out))

    epoch_time = np.array([r["epoch_time_min"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    ax.bar(epochs, epoch_time, color="#457B9D", alpha=0.85)
    ax.set_title("Epoch Runtime", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Minutes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = report_dir / "epoch_runtime.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    written.append(str(out))
    return written


def _format_history_tables(rows: list[dict[str, Any]], report_dir: Path) -> list[str]:
    written = []
    if not rows:
        return written
    columns = [
        "epoch", "train_loss", "val_loss", "kidney_dice", "tumor_dice", "cyst_dice",
        "mean_fg_dice", "selection_metric", "epoch_time_min", "is_best",
    ]
    raw_rows = []
    pretty_rows = []
    for r in rows:
        raw_rows.append({k: r[k] for k in columns})
        pretty_rows.append({
            "epoch": r["epoch"],
            "train_loss": _fmt(r["train_loss"]),
            "val_loss": _fmt(r["val_loss"]),
            "kidney_dice": _fmt(r["kidney_dice"]),
            "tumor_dice": _fmt(r["tumor_dice"]),
            "cyst_dice": _fmt(r["cyst_dice"]),
            "mean_fg_dice": _fmt(r["mean_fg_dice"]),
            "selection_metric": _fmt(r["selection_metric"]),
            "epoch_time_min": _fmt(r["epoch_time_min"], 2),
            "is_best": "yes" if r["is_best"] else "",
        })
    _write_csv(report_dir / "training_history_table.csv", raw_rows, columns)
    _write_md_table(report_dir / "training_history_table.md", pretty_rows, columns)
    val_rows = [r for r in pretty_rows if r["val_loss"] != "N/A"]
    _save_table_png(report_dir / "validation_progress_table.png", val_rows, columns[:8], "Validation Progress", max_rows=24)
    written.extend([
        str(report_dir / "training_history_table.csv"),
        str(report_dir / "training_history_table.md"),
        str(report_dir / "validation_progress_table.png"),
    ])
    return written


def _normalise_per_case(eval_data: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in eval_data.get("per_case", []):
        metrics = item.get("metrics", item)
        out.append({"case_id": item.get("case_id", ""), "metrics": metrics})
    return out


def _plot_eval_graphs(eval_data: dict[str, Any], confusion: dict[str, Any] | None, report_dir: Path) -> list[str]:
    written = []
    agg = eval_data.get("global_metrics") or {}
    per_case = _normalise_per_case(eval_data)

    if agg:
        dice = agg.get("dice", [np.nan] * 4)
        iou = agg.get("iou", [np.nan] * 4)
        precision = agg.get("precision", [np.nan] * 4)
        recall = agg.get("recall", [np.nan] * 4)
        x = np.arange(3)
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
        ax.bar(x - 1.5 * width, dice[1:], width, label="Dice", color="#1D4ED8")
        ax.bar(x - 0.5 * width, iou[1:], width, label="IoU", color="#0891B2")
        ax.bar(x + 0.5 * width, precision[1:], width, label="Precision", color="#059669")
        ax.bar(x + 1.5 * width, recall[1:], width, label="Recall", color="#D97706")
        ax.set_title("Evaluation Metrics by Class", loc="left", fontsize=15, weight="bold")
        ax.set_xticks(x, [c.title() for c in FG_CLASSES])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=4, loc="upper left")
        fig.tight_layout()
        out = report_dir / "eval_metrics_by_class.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(str(out))

    if per_case:
        rows = []
        for item in per_case:
            metrics = item["metrics"]
            dice = metrics.get("dice", [np.nan] * 4)
            rows.append({
                "case_id": item["case_id"],
                "kidney_dice": _safe_float(dice[1] if len(dice) > 1 else None),
                "tumor_dice": _safe_float(dice[2] if len(dice) > 2 else None),
                "cyst_dice": _safe_float(dice[3] if len(dice) > 3 else None),
            })
        rows_sorted = sorted(rows, key=lambda r: (math.isnan(r["tumor_dice"]), -r["tumor_dice"] if not math.isnan(r["tumor_dice"]) else 0))

        fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
        data = [
            [r["kidney_dice"] for r in rows if not math.isnan(r["kidney_dice"])],
            [r["tumor_dice"] for r in rows if not math.isnan(r["tumor_dice"])],
            [r["cyst_dice"] for r in rows if not math.isnan(r["cyst_dice"])],
        ]
        bp = ax.boxplot(data, labels=[c.title() for c in FG_CLASSES], patch_artist=True)
        for patch, color in zip(bp["boxes"], [COLORS["kidney"], COLORS["tumor"], COLORS["cyst"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title("Per-Case Dice Distribution", loc="left", fontsize=15, weight="bold")
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = report_dir / "per_case_dice_distribution.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(str(out))

        tumor_rows = [r for r in rows_sorted if not math.isnan(r["tumor_dice"])]
        fig_h = max(5.8, min(12.0, 0.28 * len(tumor_rows) + 2))
        fig, ax = plt.subplots(figsize=(10, fig_h), dpi=180)
        y = np.arange(len(tumor_rows))
        ax.barh(y, [r["tumor_dice"] for r in tumor_rows], color=COLORS["tumor"], alpha=0.85)
        ax.set_yticks(y, [r["case_id"] for r in tumor_rows], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title("Tumor Dice by Case", loc="left", fontsize=15, weight="bold")
        ax.set_xlabel("Tumor Dice")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        out = report_dir / "tumor_dice_by_case.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(str(out))

    if confusion:
        matrix = np.array(confusion.get("matrix_rows_GT_cols_pred", []), dtype=float)
        if matrix.size:
            norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=180)
            im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_title("Normalized Confusion Matrix", loc="left", fontsize=15, weight="bold")
            ax.set_xticks(np.arange(4), [c.title() for c in CLASS_NAMES], rotation=35, ha="right")
            ax.set_yticks(np.arange(4), [c.title() for c in CLASS_NAMES])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Ground Truth")
            for i in range(norm.shape[0]):
                for j in range(norm.shape[1]):
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            out = report_dir / "confusion_matrix_normalized.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            written.append(str(out))
    return written


def _format_eval_tables(eval_data: dict[str, Any], report_dir: Path) -> list[str]:
    written = []
    agg = eval_data.get("global_metrics") or {}
    if agg:
        rows = []
        for i, name in enumerate(CLASS_NAMES[1:], start=1):
            rows.append({
                "class": name.title(),
                "dice": _fmt((agg.get("dice") or [None] * 4)[i]),
                "iou": _fmt((agg.get("iou") or [None] * 4)[i]),
                "precision": _fmt((agg.get("precision") or [None] * 4)[i]),
                "recall": _fmt((agg.get("recall") or [None] * 4)[i]),
                "hd95": _fmt((agg.get("hd95_mean") or [None] * 4)[i], 2),
            })
        rows.append({
            "class": "Mean FG",
            "dice": _fmt(agg.get("mean_fg_dice")),
            "iou": _fmt(agg.get("mean_fg_iou")),
            "precision": _fmt(agg.get("mean_fg_precision")),
            "recall": _fmt(agg.get("mean_fg_recall")),
            "hd95": _fmt(agg.get("mean_fg_hd95"), 2),
        })
        cols = ["class", "dice", "iou", "precision", "recall", "hd95"]
        _write_csv(report_dir / "eval_global_metrics_table.csv", rows, cols)
        _write_md_table(report_dir / "eval_global_metrics_table.md", rows, cols)
        _save_table_png(report_dir / "eval_global_metrics_table.png", rows, cols, "Evaluation Summary", max_rows=8)
        written.extend([
            str(report_dir / "eval_global_metrics_table.csv"),
            str(report_dir / "eval_global_metrics_table.md"),
            str(report_dir / "eval_global_metrics_table.png"),
        ])

    per_case = _normalise_per_case(eval_data)
    if per_case:
        rows = []
        for item in per_case:
            m = item["metrics"]
            dice = m.get("dice", [None] * 4)
            iou = m.get("iou", [None] * 4)
            rows.append({
                "case_id": item["case_id"],
                "dice_kidney": _fmt(dice[1]),
                "dice_tumor": _fmt(dice[2]),
                "dice_cyst": _fmt(dice[3]),
                "iou_kidney": _fmt(iou[1]),
                "iou_tumor": _fmt(iou[2]),
                "iou_cyst": _fmt(iou[3]),
                "elapsed_s": _fmt(m.get("elapsed_s"), 1),
            })
        raw_sorted = sorted(rows, key=lambda r: _safe_float(r["dice_tumor"]), reverse=True)
        cols = ["case_id", "dice_kidney", "dice_tumor", "dice_cyst", "iou_kidney", "iou_tumor", "iou_cyst", "elapsed_s"]
        _write_csv(report_dir / "eval_per_case_ranked_table.csv", raw_sorted, cols)
        _write_md_table(report_dir / "eval_per_case_ranked_table.md", raw_sorted, cols)
        _save_table_png(report_dir / "eval_top_cases_table.png", raw_sorted, cols, "Top Cases by Tumor Dice", max_rows=15)
        _save_table_png(report_dir / "eval_bottom_cases_table.png", list(reversed(raw_sorted)), cols, "Lowest Cases by Tumor Dice", max_rows=15)
        written.extend([
            str(report_dir / "eval_per_case_ranked_table.csv"),
            str(report_dir / "eval_per_case_ranked_table.md"),
            str(report_dir / "eval_top_cases_table.png"),
            str(report_dir / "eval_bottom_cases_table.png"),
        ])
    return written


def _auto_eval_dir(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "showcase_test",
        output_dir / "showcase_val",
        output_dir,
    ]
    for path in candidates:
        if (path / "eval_results.json").exists():
            return path
    return None


def _write_index(report_dir: Path, written: list[str], output_dir: Path, eval_dir: Path | None) -> None:
    rel = []
    for item in written:
        p = Path(item)
        try:
            rel.append(p.relative_to(report_dir))
        except ValueError:
            rel.append(p)
    lines = [
        "# Training and Evaluation Report",
        "",
        f"- Source output dir: `{output_dir}`",
        f"- Evaluation dir: `{eval_dir if eval_dir else 'not found'}`",
        "",
        "## Generated Artifacts",
        "",
    ]
    for item in rel:
        lines.append(f"- `{item}`")
    lines.append("")
    (report_dir / "report_index.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate V12/V13 training + eval tables and graphs")
    parser.add_argument("--output-dir", default="output/swin_v12_dinov3_tumor")
    parser.add_argument("--eval-dir", default=None, help="Directory containing eval_results.json; auto-detected if omitted")
    parser.add_argument("--report-dir", default=None, help="Where to write generated tables/graphs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    eval_dir = Path(args.eval_dir) if args.eval_dir else _auto_eval_dir(output_dir)
    report_dir = _ensure_dir(Path(args.report_dir) if args.report_dir else output_dir / "report")
    written: list[str] = []

    history = _read_json(output_dir / "history.json")
    if history:
        hist_rows = _history_rows(history)
        written.extend(_format_history_tables(hist_rows, report_dir))
        written.extend(_plot_training_curves(hist_rows, report_dir))
    else:
        print(f"[warn] Missing history.json in {output_dir}")

    eval_data = _read_json(eval_dir / "eval_results.json") if eval_dir else None
    confusion = _read_json(eval_dir / "confusion_matrix.json") if eval_dir else None
    if eval_data:
        written.extend(_format_eval_tables(eval_data, report_dir))
        written.extend(_plot_eval_graphs(eval_data, confusion, report_dir))
    else:
        print("[warn] Missing eval_results.json; run eval_showcase_v12.py first for evaluation graphs/tables.")

    _write_index(report_dir, written, output_dir, eval_dir)
    print(f"Report written to: {report_dir}")
    if written:
        print("Key files:")
        for path in written[:12]:
            print(f"  {path}")
        if len(written) > 12:
            print(f"  ... {len(written) - 12} more")


if __name__ == "__main__":
    main()
