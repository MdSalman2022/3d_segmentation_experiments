"""
Measure WaLM-Net parameter count and FLOPs (for the efficiency plot / table).

    python walmnet_flops.py --mode medium                 # patch from the mode (128^3)
    python walmnet_flops.py --mode medium --patch 128,128,128
    python walmnet_flops.py --mode medium --device cpu

Reports params (M) and GFLOPs for ONE forward pass at the given patch size — the
standard "per-patch" convention for sliding-window 3D segmentation. State the
patch size whenever you quote the number. Tries fvcore, then ptflops, then thop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _import_W():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import walmnet_kits23
    return walmnet_kits23


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="medium", choices=["full", "medium", "quick_test"])
    ap.add_argument("--patch", default="", help="e.g. 128,128,128 (default: mode patch)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    W = _import_W()
    cfg = W.get_config(args.mode)
    patch = tuple(int(v) for v in args.patch.split(",")) if args.patch else cfg["patch_size"]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = W.build_model(cfg).to(device).eval()
    total, trainable = W.count_parameters(model)
    x = torch.randn(1, cfg["in_channels"], *patch, device=device)

    print(f"Model: WaLM-Net | mixer={'mamba' if W._HAS_MAMBA else 'lite'} | patch={patch}")
    print(f"Params: {total/1e6:.3f} M (trainable {trainable/1e6:.3f} M)")

    gflops = None
    backend = None
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            gflops = FlopCountAnalysis(model, x).total() / 1e9
        backend = "fvcore (MACs->FLOPs not doubled; fvcore counts MACs as flops)"
    except Exception:
        try:
            from ptflops import get_model_complexity_info
            macs, _ = get_model_complexity_info(
                model, (cfg["in_channels"], *patch), as_strings=False,
                print_per_layer_stat=False, verbose=False)
            gflops = 2 * macs / 1e9
            backend = "ptflops (2*MACs)"
        except Exception:
            try:
                from thop import profile
                macs, _ = profile(model, inputs=(x,), verbose=False)
                gflops = 2 * macs / 1e9
                backend = "thop (2*MACs)"
            except Exception as exc:  # noqa: BLE001
                print(f"\nNo FLOPs backend available ({exc}).")
                print("Install one:  pip install fvcore   # or ptflops / thop")

    if gflops is not None:
        print(f"FLOPs: {gflops:.2f} G   [backend: {backend}]")
        print(f"\nFor the plot:  flops={gflops:.1f}, params={total/1e6:.2f}  (patch {patch})")


if __name__ == "__main__":
    main()
