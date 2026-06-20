"""
Evaluate V12 and export paper/presentation-grade result artifacts.

Outputs:
  - eval_results.json
  - eval_per_case.csv
  - metrics_table.md
  - metrics_table.png
  - figures/*_panel.png with CT, GT, prediction, and error overlays

Examples:
  python eval_showcase_v12.py --checkpoint output/swin_v12_dinov3_tumor/best_final.pth --split test
  python eval_showcase_v12.py --checkpoint output/swin_v12_dinov3_tumor/best_final.pth --split val --num-cases 5 --no-tta
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import swinunetr_v12 as v12

base = v12.base

CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]
MASK_COLORS = {
    1: np.array([0, 180, 255], dtype=np.float32),   # kidney: cyan-blue
    2: np.array([255, 48, 64], dtype=np.float32),   # tumor: red
    3: np.array([255, 205, 48], dtype=np.float32),  # cyst: amber
}


def _default_font(size: int = 22):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _case_dict_from_dir(case_dir: Path) -> dict:
    return {
        "image": str(case_dir / "imaging.nii.gz"),
        "label": str(case_dir / "segmentation.nii.gz"),
        "case_id": case_dir.name,
    }


def _select_cases(config: dict, split: str, num_cases: int | None) -> list[dict]:
    train_dicts, val_dicts, test_case_dirs = base.get_data_dicts(config)
    if split == "train":
        cases = train_dicts
    elif split == "val":
        cases = val_dicts
    else:
        cases = [_case_dict_from_dir(Path(p)) for p in test_case_dirs]
    cases = [c for c in cases if Path(c["image"]).exists() and Path(c["label"]).exists()]
    return cases[:num_cases] if num_cases else cases


def _load_model(config: dict, checkpoint: str, device: torch.device) -> torch.nn.Module:
    v12._patch_base_for_v12()
    model = v12.build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(base._strip_state(state))
    model.eval()
    return model


def _metrics_for_case(pred: np.ndarray, label: np.ndarray, elapsed_s: float) -> dict:
    metrics = base.compute_seg_metrics(pred, label, num_classes=4)
    hd95 = [float("nan")] + [base.compute_hd95(pred == c, label == c) for c in range(1, 4)]
    tp = [int(((pred == c) & (label == c)).sum()) for c in range(4)]
    fp = [int(((pred == c) & (label != c)).sum()) for c in range(4)]
    fn = [int(((pred != c) & (label == c)).sum()) for c in range(4)]
    return {
        "dice": metrics["dice"],
        "iou": metrics["iou"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "hd95": hd95,
        "raw_counts": {"tp": tp, "fp": fp, "fn": fn},
        "elapsed_s": round(elapsed_s, 1),
    }


def _window_ct(slice2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice2d, [1, 99])
    if hi <= lo:
        lo, hi = float(slice2d.min()), float(slice2d.max() + 1e-6)
    x = np.clip((slice2d - lo) / (hi - lo + 1e-6), 0, 1)
    return (x * 255).astype(np.uint8)


def _overlay_mask(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.48) -> Image.Image:
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    for class_id, color in MASK_COLORS.items():
        m = mask == class_id
        rgb[m] = (1.0 - alpha) * rgb[m] + alpha * color
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _error_overlay(gray: np.ndarray, pred: np.ndarray, label: np.ndarray, alpha: float = 0.58) -> Image.Image:
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    gt_fg = label > 0
    pr_fg = pred > 0
    tp = gt_fg & pr_fg & (pred == label)
    fp = pr_fg & ~tp
    fn = gt_fg & ~tp
    colors = [
        (tp, np.array([52, 211, 153], dtype=np.float32)),   # correct foreground
        (fp, np.array([236, 72, 153], dtype=np.float32)),   # false positive
        (fn, np.array([250, 204, 21], dtype=np.float32)),   # false negative
    ]
    for mask, color in colors:
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _resize_panel(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (10, 14, 20))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def _add_title(img: Image.Image, title: str, font) -> Image.Image:
    bar_h = 42
    out = Image.new("RGB", (img.width, img.height + bar_h), (8, 12, 18))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    draw.text((14, 9), title, fill=(245, 248, 252), font=font)
    return out


def _best_slice_indices(label: np.ndarray, pred: np.ndarray, n: int) -> list[int]:
    tumor_score = ((label == 2).sum(axis=(1, 2)) * 2) + (pred == 2).sum(axis=(1, 2))
    cyst_score = ((label == 3).sum(axis=(1, 2)) * 2) + (pred == 3).sum(axis=(1, 2))
    kidney_score = (label == 1).sum(axis=(1, 2)) + (pred == 1).sum(axis=(1, 2))
    score = tumor_score * 4 + cyst_score * 2 + kidney_score * 0.2
    order = np.argsort(score)[::-1]
    selected = []
    min_gap = max(1, label.shape[0] // max(8, n * 4))
    for idx in order:
        idx = int(idx)
        if score[idx] <= 0 and selected:
            break
        if all(abs(idx - prev) >= min_gap for prev in selected):
            selected.append(idx)
        if len(selected) >= n:
            break
    if not selected:
        selected = [label.shape[0] // 2]
    return sorted(selected)


def _save_case_panels(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    case_id: str,
    case_metrics: dict,
    figures_dir: Path,
    slices_per_case: int,
    panel_size: int,
) -> list[str]:
    font_title = _default_font(22)
    font_footer = _default_font(20)
    written = []
    for z in _best_slice_indices(label, pred, slices_per_case):
        gray = _window_ct(image[z])
        panels = [
            _add_title(_resize_panel(Image.fromarray(gray).convert("RGB"), panel_size), "CT", font_title),
            _add_title(_resize_panel(_overlay_mask(gray, label[z]), panel_size), "Ground truth", font_title),
            _add_title(_resize_panel(_overlay_mask(gray, pred[z]), panel_size), "Prediction", font_title),
            _add_title(_resize_panel(_error_overlay(gray, pred[z], label[z]), panel_size), "Error map", font_title),
        ]
        gap = 14
        footer_h = 72
        width = sum(p.width for p in panels) + gap * (len(panels) - 1)
        height = max(p.height for p in panels) + footer_h
        canvas = Image.new("RGB", (width, height), (5, 8, 13))
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width + gap
        draw = ImageDraw.Draw(canvas)
        dice = case_metrics["dice"]
        footer = (
            f"{case_id} | slice {z} | "
            f"Dice K={dice[1]:.3f}  T={dice[2]:.3f}  C={dice[3]:.3f}"
        )
        legend = "Colors: kidney cyan, tumor red, cyst amber | Error: TP green, FP magenta, FN yellow"
        draw.text((18, height - 62), footer, fill=(245, 248, 252), font=font_footer)
        draw.text((18, height - 32), legend, fill=(178, 188, 204), font=_default_font(16))
        out_path = figures_dir / f"{case_id}_z{z:03d}_panel.png"
        canvas.save(out_path, quality=95)
        written.append(str(out_path))
    return written


def _write_csv(path: Path, per_case: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id",
            "dice_kidney", "dice_tumor", "dice_cyst",
            "iou_kidney", "iou_tumor", "iou_cyst",
            "precision_kidney", "precision_tumor", "precision_cyst",
            "recall_kidney", "recall_tumor", "recall_cyst",
            "hd95_kidney", "hd95_tumor", "hd95_cyst",
            "elapsed_s",
        ])
        for item in per_case:
            m = item["metrics"]
            writer.writerow([
                item["case_id"],
                *m["dice"][1:],
                *m["iou"][1:],
                *m["precision"][1:],
                *m["recall"][1:],
                *m["hd95"][1:],
                m["elapsed_s"],
            ])


def _write_markdown_table(path: Path, agg: dict) -> None:
    lines = [
        "| Class | Dice | IoU | Precision | Recall | HD95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i, name in enumerate(CLASS_NAMES[1:], start=1):
        lines.append(
            f"| {name.title()} | {agg['dice'][i]:.4f} | {agg['iou'][i]:.4f} | "
            f"{agg['precision'][i]:.4f} | {agg['recall'][i]:.4f} | {agg['hd95_mean'][i]:.2f} |"
        )
    lines.append(
        f"| Mean FG | {agg['mean_fg_dice']:.4f} | {agg['mean_fg_iou']:.4f} | "
        f"{agg['mean_fg_precision']:.4f} | {agg['mean_fg_recall']:.4f} | {agg['mean_fg_hd95']:.2f} |"
    )
    path.write_text("\n".join(lines) + "\n")


def _write_table_png(path: Path, agg: dict) -> None:
    font_h = _default_font(24)
    font = _default_font(22)
    rows = [["Class", "Dice", "IoU", "Precision", "Recall", "HD95"]]
    for i, name in enumerate(CLASS_NAMES[1:], start=1):
        rows.append([
            name.title(),
            f"{agg['dice'][i]:.4f}",
            f"{agg['iou'][i]:.4f}",
            f"{agg['precision'][i]:.4f}",
            f"{agg['recall'][i]:.4f}",
            f"{agg['hd95_mean'][i]:.2f}",
        ])
    rows.append([
        "Mean FG",
        f"{agg['mean_fg_dice']:.4f}",
        f"{agg['mean_fg_iou']:.4f}",
        f"{agg['mean_fg_precision']:.4f}",
        f"{agg['mean_fg_recall']:.4f}",
        f"{agg['mean_fg_hd95']:.2f}",
    ])
    col_w = [180, 130, 130, 170, 140, 130]
    row_h = 54
    pad = 28
    width = sum(col_w) + pad * 2
    height = row_h * len(rows) + pad * 2 + 40
    img = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.text((pad, 14), "V12 Segmentation Evaluation", fill=(15, 23, 42), font=font_h)
    y = pad + 34
    for r, row in enumerate(rows):
        x = pad
        bg = (15, 23, 42) if r == 0 else ((241, 245, 249) if r % 2 else (255, 255, 255))
        fg = (255, 255, 255) if r == 0 else (15, 23, 42)
        draw.rectangle((pad, y, width - pad, y + row_h), fill=bg)
        for c, text in enumerate(row):
            draw.text((x + 12, y + 14), text, fill=fg, font=font)
            x += col_w[c]
        y += row_h
    img.save(path, quality=95)


def main():
    parser = argparse.ArgumentParser(description="V12 evaluation + paper-style visual showcase")
    parser.add_argument("--checkpoint", default="output/swin_v12_dinov3_tumor/best_final.pth")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--kits23-dir", default=None)
    parser.add_argument("--num-cases", type=int, default=None)
    parser.add_argument("--fig-cases", type=int, default=6)
    parser.add_argument("--slices-per-case", type=int, default=2)
    parser.add_argument("--panel-size", type=int, default=420)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--sw-batch-size", type=int, default=None)
    parser.add_argument("--patch-size", default=None)
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--local-dinov3", action="store_true")
    args = parser.parse_args()

    config = v12.get_config("full")
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.kits23_dir:
        config["kits23_dir"] = args.kits23_dir
    if args.sw_batch_size:
        config["sw_batch_size"] = args.sw_batch_size
    if args.patch_size:
        config["patch_size"] = tuple(int(x) for x in args.patch_size.split(","))
    if args.hf_model:
        config["dinov3_hf_model"] = args.hf_model
    if args.local_dinov3:
        config["dinov3_source"] = "local"
    config["use_tta"] = not args.no_tta
    config["val_tta"] = not args.no_tta

    out_dir = Path(config["output_dir"]) / f"showcase_{args.split}"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(config, args.checkpoint, device)
    cache_dir = base._cache_dir_for(config["kits23_dir"])
    if not cache_dir.exists():
        cache_dir = None
    cases = _select_cases(config, args.split, args.num_cases)

    per_case = []
    figure_paths = []
    conf_matrix = np.zeros((4, 4), dtype=np.int64)
    print(f"Evaluating {len(cases)} {args.split} cases | TTA={not args.no_tta} | checkpoint={args.checkpoint}")
    with torch.no_grad():
        for idx, case in enumerate(tqdm(cases, desc="Eval/showcase")):
            t0 = time.time()
            image, label = base._load_volume(case, cache_dir)
            pred = base.predict_volume(model, image, config, device, tta=not args.no_tta)
            metrics = _metrics_for_case(pred, label, time.time() - t0)
            item = {"case_id": case["case_id"], "metrics": metrics}
            per_case.append(item)
            for gt_class in range(4):
                for pred_class in range(4):
                    conf_matrix[gt_class, pred_class] += int(((label == gt_class) & (pred == pred_class)).sum())
            if idx < args.fig_cases:
                figure_paths.extend(
                    _save_case_panels(
                        image=image,
                        label=label,
                        pred=pred,
                        case_id=case["case_id"],
                        case_metrics=metrics,
                        figures_dir=figures_dir,
                        slices_per_case=args.slices_per_case,
                        panel_size=args.panel_size,
                    )
                )
            dice = metrics["dice"]
            print(f"{case['case_id']} | K={dice[1]:.3f} T={dice[2]:.3f} C={dice[3]:.3f} | {metrics['elapsed_s']:.0f}s")
            del image, label, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    agg = base.aggregate_metrics([x["metrics"] for x in per_case], num_classes=4)
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_cases": len(per_case),
        "tta": not args.no_tta,
        "global_metrics": agg,
        "class_names": CLASS_NAMES,
        "per_case": per_case,
        "figures": figure_paths,
    }
    (out_dir / "eval_results.json").write_text(json.dumps(result, indent=2))
    (out_dir / "confusion_matrix.json").write_text(json.dumps({
        "class_names": CLASS_NAMES,
        "matrix_rows_GT_cols_pred": conf_matrix.tolist(),
    }, indent=2))
    _write_csv(out_dir / "eval_per_case.csv", per_case)
    _write_markdown_table(out_dir / "metrics_table.md", agg)
    _write_table_png(out_dir / "metrics_table.png", agg)

    print("\nSummary")
    print(f"Kidney Dice: {agg['dice'][1]:.4f}")
    print(f"Tumor Dice : {agg['dice'][2]:.4f}")
    print(f"Cyst Dice  : {agg['dice'][3]:.4f}")
    print(f"Mean FG    : {agg['mean_fg_dice']:.4f}")
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
