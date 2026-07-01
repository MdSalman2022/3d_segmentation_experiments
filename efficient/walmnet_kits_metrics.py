"""
KiTS23 official-format metrics (Hierarchical Evaluation Classes) for WaLM-Net.

The KiTS23 challenge ranks methods on THREE HECs, not on raw kidney/tumor/cyst:
    Kidney + Masses = labels {1,2,3}
    Masses          = labels {2,3}   (tumor + cyst)
    Tumor           = label  {2}
and reports Dice + Surface Dice (NSD) for each, plus their average ("Dice").

This script loads a checkpoint, runs the full validation split, and reports the
metrics in that same format so they can be placed next to the leaderboard.

    python walmnet_kits_metrics.py --checkpoint output/walmnet_medium/best.pth --mode medium

>>> CAVEATS (state these when comparing) <<<
 * Evaluated on the local held-out validation split of the *training* set
   (~97 cases), NOT the official hidden KiTS23 test set. Indicative only.
 * Surface Dice tolerance here (--nsd-tol, default 1.0 mm) may differ from the
   challenge's official tolerance/implementation; treat NSD as approximate.
   Plain Dice is directly comparable in definition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HECS = ["Kidney+Masses", "Masses", "Tumor"]
HEC_LABELS = [(1, 2, 3), (2, 3), (2,)]   # label sets per HEC


def _import_W():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import walmnet_kits23
    return walmnet_kits23


def _dice(p, g):
    denom = p.sum() + g.sum()
    return float("nan") if denom == 0 else float(2.0 * (p & g).sum() / denom)


def main():
    import numpy as np
    import torch
    from monai.data import Dataset, DataLoader
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, EnsureTyped,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="output/walmnet_medium/best.pth")
    ap.add_argument("--mode", default="medium", choices=["full", "medium", "quick_test"])
    ap.add_argument("--data-dir", default="./kits23/dataset")
    ap.add_argument("--num-cases", type=int, default=0, help="0 = all val cases")
    ap.add_argument("--nsd-tol", type=float, default=1.0, help="surface-dice tolerance (mm)")
    ap.add_argument("--out", default="kits_metrics")
    args = ap.parse_args()

    W = _import_W()
    cfg = W.get_config(args.mode); cfg["kits23_dir"] = args.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = W.build_model(cfg).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck.get("ema_state_dict") or ck.get("model_state_dict") or ck)
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ck.get('epoch')})", flush=True)

    items = W._list_cases(cfg["kits23_dir"])
    n_val = max(1, int(len(items) * cfg["val_split"]))
    val = items[:n_val]
    if args.num_cases > 0:
        val = val[:args.num_cases]
    print(f"Evaluating {len(val)} held-out cases", flush=True)

    a_min, a_max = cfg["ct_window"]
    tf = Compose([
        LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    loader = DataLoader(Dataset(val, tf), batch_size=1, num_workers=2)

    # optional surface dice
    try:
        from monai.metrics import SurfaceDiceMetric
        sdm = SurfaceDiceMetric(class_thresholds=[args.nsd_tol], include_background=False,
                                reduction="mean")
        have_nsd = True
    except Exception:
        have_nsd = False

    per_case = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            case = Path(val[i]["image"]).parent.name
            img = b["image"].to(device)
            gt = b["label"][0, 0].cpu().numpy().astype(np.int16)
            lg = sliding_window_inference(img, cfg["patch_size"], cfg["sw_batch_size"],
                                          model, overlap=cfg["sw_overlap"])
            pred = torch.argmax(lg[0], 0).cpu().numpy().astype(np.int16)

            rec = {"case": case}
            for name, labs in zip(HECS, HEC_LABELS):
                pm = np.isin(pred, labs); gm = np.isin(gt, labs)
                rec[f"{name}_dice"] = _dice(pm, gm)
                if have_nsd:
                    pt = torch.from_numpy(pm[None, None].astype(np.float32))
                    gtt = torch.from_numpy(gm[None, None].astype(np.float32))
                    try:
                        rec[f"{name}_nsd"] = float(sdm(pt, gtt).item()); sdm.reset()
                    except Exception:
                        rec[f"{name}_nsd"] = float("nan")
            per_case.append(rec)
            print(f"  {case}: " + " ".join(f"{h}={rec[h+'_dice']:.3f}" for h in HECS), flush=True)

    def _mean(key):
        vals = [r[key] for r in per_case if r.get(key) == r.get(key)]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {h: {"dice": _mean(f"{h}_dice")} for h in HECS}
    if have_nsd:
        for h in HECS:
            summary[h]["nsd"] = _mean(f"{h}_nsd")
    avg_dice = sum(summary[h]["dice"] for h in HECS) / 3.0

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = {"checkpoint": args.checkpoint, "n_cases": len(per_case),
           "avg_dice": avg_dice, "hec": summary, "per_case": per_case,
           "note": "local val split of training set; NSD tolerance=%.1fmm (approx)" % args.nsd_tol}
    (Path(args.out) / "kits_metrics.json").write_text(json.dumps(out, indent=2))

    print("\n=== KiTS23 HEC format (local val, %d cases) ===" % len(per_case))
    print(f"  Average Dice = {avg_dice:.4f}")
    for h in HECS:
        line = f"  {h:<14} Dice={summary[h]['dice']:.4f}"
        if have_nsd:
            line += f"  SD={summary[h]['nsd']:.4f}"
        print(line)
    print(f"\nSaved -> {Path(args.out)/'kits_metrics.json'}")


if __name__ == "__main__":
    main()
