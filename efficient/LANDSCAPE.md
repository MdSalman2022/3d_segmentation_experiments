# Efficient 3D Medical Segmentation — Landscape & Novelty Plan

> Goal: a fast, lightweight 3D segmentation network that reaches near-SOTA Dice
> WITHOUT LLM/VLM, ensembles, or external pretraining. Primary contribution =
> the architecture / efficiency method itself. Benchmark target: KiTS23 (local)
> + one public set (BraTS2023 or AMOS) for cross-dataset claims.

## Why this direction
- Hardware-limited setting makes "efficient yet SOTA" a genuine, fundable niche.
- nnU-Net still strong but heavy/slow; clear room for a lean competitor.
- Avoiding LLM/VLM sidesteps the compute arms race; efficiency rewards design.

## SOTA efficient models (2024–2026)

| Model | Venue/Year | Size | Core idea | Weakness / gap |
|---|---|---|---|---|
| WaveFormer | MICCAI 2025 | 7.7M | Discrete Wavelet Transform attention; wavelet up/downsample; -93.75% attn compute | Fixed Haar wavelet (not learnable); transformer attention still core; brain+abd only |
| LHU-Net | MICCAI 2025 | 10.5M / 51.75 GFLOP | spatial-then-channel hybrid attention U-Net | Hand-tuned block order; no SSM; no high-freq path |
| EffiDec3D | CVPR 2025 | decoder -96% params / -93% FLOPs | channel reduction + drop low-value high-res decoder layers | Decoder-only bolt-on; encoder untouched |
| SegMamba / SegMamba-V2 | MICCAI 2024 / IEEE 2025 | linear cost | Mamba SSM, tri-directional scan, long-range | Scan-order sensitivity; weak fine boundary & high-freq detail |
| LightM-UNet | 2024 | ~1M | pure Mamba, residual visual Mamba layers | Accuracy drops on small/rare structures (e.g. tumor) |
| Slim UNETR | 2024 | low | slimmed hybrid transformer | Superseded by above |
| EfficientMedNeXt | MICCAI 2025 | conv | multi-receptive dilated convolutions | No long-range global modeling |

### Sources
- WaveFormer: https://arxiv.org/abs/2503.23764
- LHU-Net: https://arxiv.org/abs/2404.05102 | code https://github.com/xmindflow/lhunet
- EffiDec3D: https://github.com/SLDGroup/EffiDec3D (CVPR 2025)
- SegMamba: https://link.springer.com/chapter/10.1007/978-3-031-72111-3_54
- Mamba survey: https://github.com/xmindflow/Awesome_Mamba

## White space (our novelty)
Two gaps no one has filled TOGETHER:

1. **High-frequency detail + linear cost.** Mamba is cheap but blurs fine
   boundaries and rare classes; wavelet keeps high-freq but its backbone is
   still quadratic-ish attention. No work fuses a wavelet high-frequency path
   with a Mamba linear backbone.
   -> **Wavelet–Mamba hybrid:** Mamba on low-freq sub-band (global context,
      linear); lightweight conv/attention on high-freq sub-bands (boundary +
      small-structure detail). Linear overall, detail preserved.

2. **Learnable wavelet basis.** WaveFormer uses fixed Haar. A learnable
   lifting-scheme wavelet adapts the frequency split per organ for a tiny
   parameter cost = clean, defensible novelty.

## Standout axis (beyond raw speed)
Efficiency alone is crowded. Pair it with one of:
- **Rare/small-structure robustness** (directly maps to kidney tumor Dice — our
  existing pain point and a measurable differentiator).
- **Boundary fidelity** via the high-freq path (report surface Dice / HD95, not
  just volumetric Dice).

## Next steps
- [ ] Read WaveFormer code (efficient/models/WaveFormer) — reuse DWT modules.
- [ ] Clone/read LHU-Net + EffiDec3D + SegMamba for block-level ideas.
- [ ] Prototype Wavelet–Mamba block; measure params/FLOPs/latency vs WaveFormer.
- [ ] Fix benchmark protocol: same patches, same GPU, KiTS23 + 1 public set.
- [ ] Decide standout axis (rare-class vs boundary) before writing.
