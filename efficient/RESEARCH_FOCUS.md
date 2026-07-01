# Proposed Model — Research Focus, Gaps & Architecture Plan

> **Purpose.** A decision document that synthesizes our local corpus
> (`efficient/papers/` — 43 PDFs, `efficient/PAPERS_3D_SEG.md` — tiered survey,
> `efficient/LANDSCAPE.md` — novelty plan, `efficient/MODELS_COMPENDIUM.md` —
> code-level map of 27 repos) **with our own KiTS23 results** to answer four
> questions:
> 1. What should our proposed model focus on?
> 2. Which paper(s) should we build on / beat?
> 3. What are the open gaps?
> 4. Which architecture solves the issues of current best works?
>
> Scope guardrail (from `LANDSCAPE.md`): the contribution is the **architecture /
> efficiency method itself** — no LLM/VLM core, no ensembles, no external
> pretraining dependency. Target = **KiTS23** (local) + one public set
> (BraTS2023 or AMOS) for cross-dataset claims.

---

## 0. TL;DR — the recommendation

**Build a single-branch, linear-cost, lightweight 3D U-shaped network whose
*headline* contribution is rare/small-structure robustness (kidney *tumor* +
*cyst*) at a fraction of nnU-Net compute, by fusing a Mamba/SSD global backbone
(linear cost) with a *learnable lifting-wavelet high-frequency detail path* and a
*rare-class-aware detail head*, and report boundary metrics (surface-Dice / HD95),
not just volumetric Dice.**

- **Focus axis:** not "efficiency" alone (saturated) — **efficiency × small/rare-class
  tumor+cyst robustness × boundary fidelity.** This is *our* documented pain point
  and the field's universal weak spot, so it is both fundable and measurable.
- **Primary papers to build on:** **WaveFormer** (wavelet primitives, MICCAI'25) +
  **SegMamba** (linear Mamba backbone, MICCAI'24). **Primary baseline to beat:**
  **nnU-Net** + **MedNeXt** + **HybridMamba** (closest, KiTS-relevant).
- **The defensible twist** (because Wavelet+Mamba is now crowded): a **learnable
  lifting-scheme** wavelet (not fixed Haar, not generic FFT/DCT filters), feeding a
  **dedicated rare-class high-frequency head**, in a **single light branch** (not
  HybridMamba's dual branch), validated on **CT kidney tumor+cyst** with **boundary
  metrics** — a combination unoccupied as of mid-2026.
- **Pivot away from** the DINOv3 / VISTA3D / SwinUNETR / CLIP / VLM-critic pipeline:
  on our own runs it is **heavy (93M–266M params), slow (up to 33 h), unstable
  (best Dice 0.42–0.65), and cyst-blind** — and it violates the no-VLM guardrail.

---

## 1. Where we are now (the evidence base)

Our own KiTS23 runs (kidney / tumor / cyst, 4-class) from `output/`:

| Run | Approach | Params | Train time | Best result (key numbers) |
|---|---|---|---|---|
| v3 quality | DINOv2-ViT-B + VISTA3D | 93.2M | 33.25 h | overall Dice **0.755**; per-case cyst often **0.000** (median & worst cases) |
| v4 tumor-focused | DINO + VISTA3D, tumor-weighted | ~93M | — | Kidney **0.920**, Tumor **0.650**, Cyst **0.257** (all) / 0.470 (positive only) |
| v5 VLM-critic + MGPL | DINOv2 + CLIP + Qwen3-VL critic | 266.6M | 1.25 h | best Dice **0.427** (collapsed) |
| v6b VLM-loop | SwinUNETR + ViT-L + VLM + pseudo-labels | 223.1M | 2.93 h | best Dice **0.537** |

**Read-out:**
- The heavy foundation/VLM direction **plateaus around the tumor and is essentially
  cyst-blind** (cyst = 0.000 in the median case for v3; 0.257 mean for v4).
- More parameters did **not** help — v5 (266M) and v6b (223M) are *worse* than v4 (~93M).
- The VLM critic loop is unstable and adds infra burden (Ollama, CLIP) for no Dice gain.
- **What is genuinely worth keeping** from this work: the small-structure machinery —
  Tversky/region loss, class re-weighting, tumor/cyst-biased sampling, deep
  supervision, two-stage coarse→fine. Port these onto a *light* backbone.

**Conclusion:** our comparative advantage is **domain insight into the tumor+cyst
failure mode**, not scale. The model should be built around that.

---

## 2. The target and the bar (KiTS23)

KiTS23 uses hierarchical regions: **Kidney+Masses**, **Masses (tumor+cyst)**,
**Tumor**, scored by Dice **and Surface Dice (boundary)**.

| Reference | Kidney | Masses (tumor+cyst) | Tumor | Notes |
|---|---|---|---|---|
| KiTS23 winner (avg) | ~0.93 | ~0.73 (cyst) | ~0.57 | avg Dice ~0.835; heavy ensemble, slow |
| KiTS23 2nd place | 0.948 | 0.776 | 0.738 | SD: 0.899 / 0.635 / 0.602 |
| **The open problem** | (solved) | **cyst ~0.73** | **tumor ~0.57–0.74** | tumor & cyst are where everyone loses points |

`analysis.txt` (our 489-case dataset profile) explains *why*: tumor median volume
4054 mm³ but **5th-percentile diameter ~1.1 mm**; cyst median **only 401 mm³**,
5th-percentile ~1.4 mm. Many lesions occupy **2–3 voxels** and are low-contrast.
**Small + low-contrast + rare = the entire battle.** This is exactly what a
high-frequency detail path plus rare-class loss is designed to fix.

**Our quantified goal:** match/beat the winner's **tumor (~0.57) and cyst (~0.73)
Dice + their surface-Dice**, at **<15M params and a fraction of the FLOPs/latency**,
single model (no ensemble), no external pretraining.

---

## 3. Gaps in the current best works

### 3a. Per-model issue → what it leaves open (from `MODELS_COMPENDIUM.md` + papers)

| Best work | Core strength | Issue / what it leaves open |
|---|---|---|
| **nnU-Net** (the bar) | robust auto-config, top accuracy | heavy & slow; tumor/cyst still weak; no explicit detail/freq path |
| **MedNeXt** (ICLR'23/MICCAI'23) | ConvNeXt scaling, strong baseline | large-kernel cost; **no long-range global mixing**; heavy variants slow |
| **SegMamba / -V2** (MICCAI'24/TMI'25) | linear-cost long-range Mamba | **blurs fine boundaries**, **scan-order bias**, **under-segments small/rare classes** |
| **WaveFormer** (MICCAI'25, 7.7M) | wavelet attention, −93.75% attn compute | **fixed Haar (not learnable)**; attention still core; **never validated on CT tumor/KiTS** |
| **HybridMamba** (MICCAI'25) | dual-domain (spatial+freq) Mamba | **two full branches = overhead**; fusion heuristic; tumor still the weak link |
| **EM-Net** (MICCAI'24) | Mamba + frequency learning, ~½ params | frequency is **channel-wise**, not a **spatial wavelet sub-band**; rare class not targeted |
| **LHU-Net** (MICCAI'25, 10.5M) | lean hybrid attention U-Net | hand-tuned block order; **no SSM, no high-freq path** |
| **EffiDec3D** (CVPR'25) | −96% decoder params / −93% FLOPs | **decoder-only bolt-on; encoder untouched** |
| **LightM-UNet** (~1M) | extreme lightness | **accuracy drops on small/rare structures** (the tumor problem) |
| **VeloxSeg** (2025, JL-lemma) | 11×/48× throughput | random projection may **hurt fine detail**; no rare-class focus |
| **Submanifold Sparse ConvNet** (KiTS) | compute only on occupied voxels | **no long-range / no frequency**; small-tumor Dice still hard |

### 3b. Cross-cutting white-space (the gaps we can own)

1. **Rare/small-structure (tumor+cyst) robustness under a tiny budget.** Every
   lightweight net (LightM-UNet, UltraLight, Slim) *admits* accuracy loss on small/rare
   classes. **No efficient model makes rare-class robustness its headline.**
2. **Boundary / high-frequency fidelity with linear cost.** Mamba nets are cheap but
   blur edges; most report only volumetric Dice. **Owning surface-Dice/HD95 is open.**
3. **Learnable, adaptive frequency basis.** WaveFormer = fixed Haar; EM-Net =
   channel-wise freq; HybridMamba = learnable *frequency filters* (not a spatial
   *lifting wavelet*). A **learnable lifting-scheme spatial wavelet with guaranteed
   perfect reconstruction** is unoccupied for 3D medical seg.
4. **Encoder-side efficiency fused with decoder pruning.** EffiDec3D only fixed the
   decoder. An efficient encoder **+** EffiDec3D-style decoder is uncombined.
5. **Single-branch detail-preserving design.** HybridMamba needs two branches; a
   *single* light branch that still keeps high-freq detail is open.

### 3c. Reality check — the space is hot (be precise or get scooped)

Wavelet/frequency + Mamba is **partly taken** and moving fast (2025–2026):
Topology-Aware Wavelet Mamba (airway), HybridMamba (dual-domain), EM-Net (freq+Mamba),
plus brand-new **SSFMamba** (symmetry-driven spatial-frequency), **SegResMamba**
(efficient), **SwiM-UNet** (Nature Sci. Rep. 2026, Mamba–Transformer on-device), and a
**Hybrid Mamba-SAM with Multi-Frequency Gated Convolution** (DCT, Feb 2026, Dice 0.906).
**Implication:** "efficient Mamba" or "Mamba + frequency" *alone* is no longer novel.
Our edge must be the **specific** combination in §0/§5 — and the **CT tumor+cyst +
boundary-metric** validation that none of these target.

---

## 4. Which paper(s) to focus on

**Build on (reuse code/primitives):**
- **WaveFormer** ([github](https://github.com/mahfuzalhasan/WaveFormer)) — reuse DWT/IDWT
  modules (`wave_helper.py`, `idwt_upsample.py`); **replace its fixed Haar with our
  learnable lifting wavelet** and its attention core with Mamba.
- **SegMamba** ([github](https://github.com/ge-xing/SegMamba)) — reuse `MambaLayer`,
  `GSC` gated spatial conv, hierarchical `MambaEncoder` as the **linear-cost backbone**.
- **EffiDec3D** ([github](https://github.com/SLDGroup/EffiDec3D)) — reuse the decoder
  channel-reduction + resolution-factor logic for **encoder-matched decoder efficiency**.

**Beat (primary comparison baselines):**
- **nnU-Net** (mandatory bar), **MedNeXt** (efficient conv), **SegMamba** (Mamba SOTA),
  **WaveFormer** (wavelet SOTA), **LHU-Net** (lean hybrid), **EffiDec3D** (efficiency),
  **HybridMamba** (closest competitor, KiTS-relevant).

**Methodological references for the rare-class/boundary story:**
- **HFF-Net** (adaptive-Laplacian high-freq, +4.48% tumor-subregion Dice) — motivation
  for the high-freq detail branch.
- **Detail-Aware Net** (lifting wavelets, brain MRI, 2D-leaning) — proves lifting
  wavelets help tumor detail; we move it to **3D + CT + rare-class + linear backbone**.
- **MPSTrans** (small-tumor focus, −56% to −71% FLOPs) — small-structure precedent.
- **Submanifold Sparse ConvNet** (KiTS kidney+tumor) — KiTS efficiency precedent.

---

## 5. Proposed architecture — **WaLM-Net** (Wavelet-Lifting Mamba Net)

> Working name. Single light U-shaped backbone (target **8–14M params**, **linear**
> attention/SSM cost), one high-frequency detail path, rare-class-aware supervision.

```
                 input volume (1ch CT patch)
                          │
              ┌───────── Stem (conv) ─────────┐
              │                               │
   ╭──────────▼──────────╮        Learnable Lifting Wavelet (per stage)
   │  Mamba/SSD Encoder   │   split → [LL low-freq]  +  [LH/HL/HH high-freq]
   │  (linear, hierarch.) │           │                     │
   │  GSC local conv +    │           │                     ▼
   │  multi-dir scan      │      (global path)     High-Freq Detail Branch
   ╰──────────┬──────────╯      MambaLayer on LL    lightweight conv/attn
              │                        │            (boundary + small struct)
              │                        └─────── fuse (gated) ───────┐
              │                                                     │
   EffiDec3D-style efficient decoder  ◄── skip (att-gated) ────────┘
              │
   ┌──────────┼─────────────────────────────────────────────┐
   │  Main head (4-class)   +   Rare-class detail head (tumor/cyst)  +  Deep sup. │
   └──────────┴─────────────────────────────────────────────┘
   Loss = Dice + CE + class-weighted Tversky(α<β, boosts FN recall on tumor/cyst)
          + boundary loss (surface/HD-aware) on the detail head
```

**The four pieces and the issue each fixes:**

| Component | What it is | Issue it solves |
|---|---|---|
| **Learnable lifting wavelet** (predict/update operators learned; perfect reconstruction by construction) | replaces fixed Haar split/merge for down/upsampling | WaveFormer's **fixed basis**; adapts the freq split per organ at tiny param cost → defensible novelty |
| **Mamba/SSD backbone on the low-freq (LL) sub-band** | SegMamba-style `MambaLayer` + `GSC`, multi-directional scan | nnU-Net/MedNeXt **cost**; SegMamba **scan-order bias** (multi-dir averaging) |
| **High-frequency detail branch** (LH/HL/HH → light conv/attn) fused by a learned gate | a *single* extra light path, not a second full branch | SegMamba/Mamba **boundary blur** & **small-class loss**; HybridMamba **dual-branch overhead** |
| **Rare-class detail head + boundary loss + Tversky** | extra head supervised on tumor/cyst with surface-aware loss | the universal **tumor/cyst weak spot**; reports **surface-Dice/HD95** rivals omit |
| **EffiDec3D-style decoder** | channel-reduced, resolution-factored decoder | **encoder+decoder both efficient** (EffiDec3D fixed only decoder) |

**Why single-branch + lifting is the honest novelty:** HybridMamba already does
dual-domain Mamba with *learnable frequency filters*; WaveFormer already does wavelet
attention with *fixed Haar*. Neither does a **learnable lifting-scheme spatial wavelet**
whose **high-freq sub-bands drive a rare-class head**, in **one light branch**, **on CT
tumor+cyst with boundary metrics.** That intersection is the contribution.

---

## 6. Risks & fallbacks (kill criteria)

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Wavelet+Mamba gets too crowded / a 2026 paper lands the same combo | Medium-High | Lead with **rare-class tumor+cyst + boundary metrics on KiTS** (none target this); keep lifting-wavelet as the primitive, swap backbone if needed |
| Mamba kernels (causal-conv1d) hard to build / slow on our GPUs | Medium | **Plan B (no Mamba):** MedNeXt/conv backbone + learnable-lifting high-freq branch + EffiDec3D decoder + rare-class head. Avoids the crowded Mamba space entirely |
| Learnable lifting unstable to train | Medium | Initialize at Haar (recovers WaveFormer), regularize toward perfect reconstruction, learn a residual correction only |
| Single-branch underperforms dual-branch on tumor | Low-Med | Ablate; if needed add a *thin* second branch but keep param budget <15M |
| Gains don't beat nnU-Net on tumor | Med | Pivot the claim to **efficiency at iso-accuracy** (same Dice, 5–10× less compute) — still publishable |

**Decision rule:** if after the prototype the learnable wavelet gives **<0.5% tumor
Dice** over fixed-Haar **and** the Mamba space looks saturated, switch to **Plan B**
(conv backbone, frequency detail branch, efficiency-first story).

---

## 7. Experiment protocol & success criteria

- **Datasets:** KiTS23 (primary, our 489 cases) + BraTS2023 *or* AMOS (cross-dataset).
- **Fixed protocol:** identical patches, spacing, splits, and **one GPU** for *all*
  baselines (re-run, don't quote papers). Report **params, GFLOPs, latency/volume,
  peak VRAM** alongside accuracy.
- **Metrics:** per-class **Dice + Surface-Dice + HD95** (especially tumor & cyst);
  KiTS hierarchical regions (Kidney+Masses, Masses, Tumor).
- **Ablations:** (1) fixed Haar vs learnable lifting; (2) ± high-freq branch;
  (3) ± rare-class head/boundary loss; (4) single- vs dual-branch; (5) full vs
  EffiDec3D decoder; (6) scan-order sensitivity.
- **Success =** beat SegMamba/WaveFormer/LHU-Net on **tumor+cyst Dice and surface-Dice**
  at **≤ their params/FLOPs**, and reach **≥90% of nnU-Net tumor/cyst Dice at <20% of
  its compute**, single model.

---

## 8. Next steps

- [ ] Stand up the **fixed benchmark harness** (KiTS23 + 1 public, one GPU, logs
      params/FLOPs/latency) — reuse our existing MONAI training/eval scaffolding.
- [ ] Re-run baselines: nnU-Net, MedNeXt, SegMamba, WaveFormer, LHU-Net, EffiDec3D,
      HybridMamba — get *our-protocol* numbers (the honest bar).
- [ ] Prototype the **learnable lifting-wavelet** block; sanity-check perfect
      reconstruction; ablate vs Haar on one organ.
- [ ] Assemble **WaLM-Net v0** (SegMamba backbone + lifting wavelet + high-freq branch
      + rare-class head + EffiDec3D decoder); measure params/FLOPs/latency vs WaveFormer.
- [ ] Port our **rare-class machinery** (Tversky weights, tumor/cyst-biased sampling,
      deep supervision, coarse→fine) from the v3/v4 pipeline onto WaLM-Net.
- [ ] Add **boundary loss** + surface-Dice/HD95 reporting.
- [ ] Run ablations (§7); decide primary vs Plan B at the kill-criteria checkpoint.

---

## Sources

**Local:** `efficient/LANDSCAPE.md`, `efficient/PAPERS_3D_SEG.md`,
`efficient/MODELS_COMPENDIUM.md`, `efficient/papers/*.pdf`, `analysis.txt`,
`output/meddino_vista3d_v3_quality/`, `output/meddino_vista3d_v4_tumor_focused/`,
`output/v5/`, `output/vlm_loop_v6b/`.

**Papers / web (verified June 2026):**
- KiTS23 challenge — https://kits-challenge.org/kits23/ ; winner solution
  https://arxiv.org/pdf/2310.04110 ; 2nd-place
  https://link.springer.com/chapter/10.1007/978-3-031-54806-2_20
- WaveFormer — https://arxiv.org/abs/2503.23764 · https://github.com/mahfuzalhasan/WaveFormer
- SegMamba — https://arxiv.org/abs/2401.13560 · https://github.com/ge-xing/SegMamba
- HybridMamba — https://papers.miccai.org/miccai-2025/0426-Paper2815.html · https://arxiv.org/pdf/2509.14609
- EM-Net — https://arxiv.org/pdf/2409.17675
- MedNeXt — https://link.springer.com/chapter/10.1007/978-3-031-43901-8_39
- LHU-Net — https://arxiv.org/abs/2404.05102 · https://github.com/xmindflow/lhunet
- EffiDec3D — https://github.com/SLDGroup/EffiDec3D
- LightM-UNet — https://arxiv.org/pdf/2403.05246
- VeloxSeg (JL-lemma) — https://arxiv.org/pdf/2509.22307
- Topology-Aware Wavelet Mamba — https://arxiv.org/pdf/2502.14363
- SSFMamba (spatial-frequency) — https://arxiv.org/pdf/2508.03069
- SegResMamba (efficient) — https://arxiv.org/pdf/2503.07766
- SwiM-UNet (Nature Sci. Rep. 2026) — https://www.nature.com/articles/s41598-026-35771-4
- Hybrid Mamba-SAM + Multi-Frequency Gated Conv — https://arxiv.org/abs/2602.00650
- Comprehensive analysis of Mamba for 3D seg — https://arxiv.org/pdf/2503.19308
- Submanifold Sparse ConvNet (kidney+tumor) — https://arxiv.org/pdf/2511.04334
- HFF-Net (freq-domain brain tumor) — https://pubmed.ncbi.nlm.nih.gov/40504718/
- Detail-Aware lifting-wavelet tumor — https://www.sciencedirect.com/science/article/abs/pii/S1568494625012827
</content>
</invoke>
