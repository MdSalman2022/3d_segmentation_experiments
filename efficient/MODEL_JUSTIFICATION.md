# WaLM-Net — Justification Against the Surveyed Research

> **Purpose.** Defend the `walmnet_kits23.py` implementation choice-by-choice
> against the 26-model corpus in `PAPERS_3D_SEG.md` / `MODELS_COMPENDIUM.md`,
> the novelty plan in `LANDSCAPE.md`, and the decision document
> `RESEARCH_FOCUS.md`. The question being answered: *given that wavelet+Mamba is
> already crowded (2025–2026), why is THIS specific combination defensible, and
> how does each component respond to a documented weakness in the field?*

---

## 0. The thesis in one paragraph

WaLM-Net is **not** "another efficient Mamba net" and **not** "Mamba + frequency
done again." It is the **first single-branch, linear-cost, true-volumetric-3D U-Net
whose headline contribution is rare/small-structure (kidney tumor + cyst)
robustness at a fraction of nnU-Net compute**, achieved by fusing (i) a **learnable
lifting-scheme wavelet** with guaranteed perfect reconstruction, (ii) a
**bidirectional Mamba/SSM token mixer** on the low-frequency sub-band, (iii) a
**single lightweight high-frequency detail branch** gated back into the main path,
and (iv) a **dedicated rare-class head with
Tversky + boundary loss**, validated on **CT kidney tumor+cyst** with **surface-Dice
and HD95** — a four-way intersection no surveyed work occupies. Every component
below is traced to a *specific* gap in a *specific* paper.

---

## 1. Component-by-component justification

### 1.1 Learnable lifting wavelet (`LearnableLiftingWavelet3D`, `_Lifting1D`)

**What it is.** A separable 3D lifting-scheme DWT/IDWT where the predict (`P`) and
update (`U`) operators are depthwise conv3ds, **initialised to zero (lazy wavelet)**,
so at init `DWT→IDWT` is exactly invertible. The lifting structure guarantees
**perfect reconstruction for any learned P, U** — not approximately, by construction.

**Why this is the defensible novelty (vs every surveyed wavelet work):**

| Surveyed work | Its wavelet choice | What WaLM-Net does differently |
|---|---|---|
| **WaveFormer** (MICCAI'25, 7.7M) | **Fixed Haar** DWT/IDWT (`wave_helper.py`) | Learnable lifting — the basis *adapts per organ* at ~0 extra params; Haar is the init, not the ceiling |
| **HybridMamba** (MICCAI'25) | Learnable *frequency filters* (channel-wise), **not a spatial lifting wavelet** | A true **spatial** lifting wavelet with sub-band structure (LL + 7 highs), not channel-wise spectral mixing |
| **EM-Net** (MICCAI'24) | Frequency layer is **channel-wise** spectral learning | Spatial sub-band decomposition that preserves locality and feeds a *spatial* detail branch |
| **Topology-Aware Wavelet Mamba** (airway, 2025) | Wavelet+Mamba **but fixed basis + tubular topology prior** | Learnable basis, no topology prior (tumor is blobby, not tubular — the prior wouldn't transfer) |
| **WMREN** (IJCAI'25) | Wavelet multi-scale, **2D-leaning**, region-enhanced | Native 3D separable lifting, no hand-tuned region module |
| **Detail-Aware Net** (brain MRI, 2D-leaning) | Lifting wavelets **in 2D** for tumor detail | Lifted to **3D + CT + rare-class + linear Mamba backbone** — the unoccupied 3D extension |

**The honest edge.** "Wavelet + Mamba" alone is taken (§3c of `RESEARCH_FOCUS.md`
lists SSFMamba, SegResMamba, SwiM-UNet, Hybrid Mamba-SAM/DCT). What is **not** taken
is a **learnable lifting-scheme spatial wavelet with guaranteed perfect
reconstruction, feeding a rare-class head, on CT tumor+cyst, with boundary metrics.**
The lifting guarantee is the technical moat: WaveFormer cannot adapt its basis
without breaking invertibility; HybridMamba's filters are channel-wise; a generic
learnable filter bank has no reconstruction guarantee. Lifting gives both
adaptivity *and* invertibility by construction — this is the one defensible primitive.

**Why lazy-wavelet init matters.** It means WaLM-Net *starts* as WaveFormer (Haar =
lazy lifting with P=U=0) and can only improve from there. This is also the kill-switch
in `RESEARCH_FOCUS.md` §6: if learnable lifting gives <0.5% tumor Dice over fixed Haar,
the contribution collapses to "WaveFormer + Mamba + rare head" and we pivot to Plan B.
The init makes that ablation clean.

---

### 1.2 Bidirectional Mamba/SSM token mixer (`TokenMixer`, `WaMambaBlock`, `_SSMLite`)

**What it is.** A `GSC` (SegMamba-style gated spatial conv, local) + a token mixer
that runs the SSM **forward and backward and averages** (scan-order debiasing), with
a dependency-free `_SSMLite` fallback (depthwise conv + broadcast global mean + SiLU
gate) so the model runs anywhere even without `mamba_ssm`/`causal-conv1d`.

**Why this, vs every surveyed backbone:**

| Surveyed backbone | Its global-mixing choice | WaLM-Net's response |
|---|---|---|
| **nnU-Net** (the bar) | Pure conv — **no long-range global mixing** | Linear-cost global SSM on the LL sub-band gives long-range at O(N) |
| **MedNeXt** (ICLR'23) | Large-kernel ConvNeXt convs — **costly at scale, no global token mixing** | SSM is linear in voxels; no large-kernel memory blowup |
| **SegMamba / -V2** (MICCAI'24/TMI'25) | Tri-orientated Mamba — **scan-order bias, blurs boundaries** | Bidirectional averaging explicitly attacks scan-order bias; the HF branch attacks the blur |
| **U-Mamba** (2024) | Hybrid CNN-SSM, accuracy-first, **no frequency/detail path** | Same hybrid idea but with the wavelet detail path on top |
| **LightM-UNet** (~1M) | Pure Mamba residual — **accuracy drops on small/rare structures** | Rare-class head + HF branch + Tversky directly target that failure |
| **nnMamba** (2024) | Mamba-in-conv block — **block-level, modest efficiency edge** | Full encoder-decoder with wavelet downsampling, not a block swap |
| **Swin-UMamba** (2024) | ImageNet-pretrained Swin + Mamba — **pretraining dependency** | No external pretraining (guardrail) |
| **EfficientViM** (CVPR'25) | HSM-SSD for vision — **general-purpose, not medical-tuned, no boundary design** | Medical-tuned, rare-class-aware, boundary-loss-supervised |
| **UltraLight VM-UNet** (0.049M) | PVM Mamba — **2D/skin, under-capacity for tumor** | 3D volumetric, capacity sized for tumor+cyst |

**Why bidirectional averaging is non-trivial.** SegMamba's documented weakness
(`PAPERS_3D_SEG.md` §1, §10) is scan-order sensitivity. Running the SSM on the
flattened voxel sequence forward and backward and averaging is the cheapest possible
debiasing that stays linear. It is not a full tri-orientation scan (SegMamba-V2's
approach) — that is a deliberate efficiency trade: we spend the saved compute on the
HF detail branch instead, which is where tumor/cyst actually lose points.

**Why the `_SSMLite` fallback exists.** `RESEARCH_FOCUS.md` §6 lists "Mamba kernels
hard to build / slow on our GPUs" as a Medium-likelihood risk. The fallback makes the
model runnable on any PyTorch install (Plan B survivability) without changing the
architecture contract — the rest of the net is identical. This is engineering
honesty, not a novelty claim.

---

### 1.3 High-frequency detail branch (`HFBranch`, gated fusion in `WaLMEncoderStage`)

**What it is.** The 7 high-frequency sub-bands (LH/HL/HH per axis) are concatenated,
run through a **depthwise-separable** conv (grouped depthwise + 1×1 pointwise), and
**gated** back into the low-frequency main path via a learned sigmoid gate.

**Why this, vs every surveyed detail/frequency approach:**

| Surveyed work | Its detail/frequency handling | WaLM-Net's response |
|---|---|---|
| **HybridMamba** (MICCAI'25) | **Two full branches** (spatial + frequency) = compute overhead, heuristic fusion | **Single** light branch, gated fusion — half the branch cost, principled gate |
| **SegMamba / Mamba nets** | **No high-freq path** — blurs fine boundaries, under-segments small/rare classes | Explicit HF sub-band path preserves what Mamba blurs |
| **WaveFormer** | Wavelet is the **attention mechanism itself** (still quadratic-ish attention core) | Wavelet is the **downsampler + detail source**; the backbone is linear SSM |
| **EM-Net** | Frequency is **channel-wise**, not spatial sub-band | Spatial sub-bands preserve locality → better for blobby tumor boundaries |
| **HFF-Net** (adaptive-Laplacian, +4.48% tumor-subregion Dice) | Adaptive Laplacian high-freq — **motivation, not the method** | We adopt the *idea* (high-freq helps tumor subregions) via a different, cheaper primitive |
| **EffiDec3D** (CVPR'25) | **Decoder-only** — encoder untouched, no freq path | HF branch is encoder-side; combined with EffiDec3D-style decoder = both ends efficient |

**Why single-branch, not dual.** `RESEARCH_FOCUS.md` §3b gap #5: "HybridMamba needs
two branches; a *single* light branch that still keeps high-freq detail is open." The
gated fusion is the key — it lets one cheap depthwise-separable branch carry the
boundary/small-structure signal without a second full encoder. If ablation (§7 of
`RESEARCH_FOCUS.md`) shows single-branch underperforms, the kill-criteria allows a
*thin* second branch under the 15M param budget. The architecture is designed to
degrade gracefully.

**Why depthwise-separable.** The 7 sub-bands × channels is a wide tensor; grouped
depthwise keeps it cheap (the EffiDec3D philosophy applied to the detail path). This
is the same logic that lets WaveFormer hit 7.7M params — we reuse it, then add the
learnable basis on top.

---

### 1.4 Rare-class head + Tversky + boundary loss (`WaLMLoss`, `rare_head`)

**What it is.** A **dedicated 2-channel head** (tumor, cyst) supervised with BCE,
plus the main 4-class head supervised with **Dice + weighted CE + Tversky
(α=0.3 < β=0.7, boosting FN recall on tumor/cyst) + Laplacian boundary loss**, with
**deep supervision** at 1/2 and 1/4 resolution.

**Why this, vs every surveyed loss/training approach:**

| Surveyed work | Its loss/training | WaLM-Net's response |
|---|---|---|
| **All Mamba/efficient nets** (SegMamba, LightM-UNet, LHU-Net, etc.) | Dice + CE, **rare-class not targeted** | Tversky with α<β + rare_boost explicitly penalizes tumor/cyst false negatives |
| **nnU-Net / MedNeXt** | Dice + CE, deep supervision — **no boundary term** | Laplacian boundary loss on the detail path → surface-Dice/HD95 gains |
| **HybridMamba** (kidney 97.54% Dice) | Strong on kidney, **tumor still the weak link** | Rare head + Tversky directly attack the tumor failure mode |
| **KiTS23 winner/2nd** | Heavy ensembles, tumor ~0.57 / cyst ~0.73 | Single model, same target, fraction of compute |
| **MPSTrans** (small-tumor, −56–71% FLOPs) | Small-tumor focus — **precedent, not competitor** | We adopt the small-structure framing via a different mechanism (HF + rare head) |
| **Pancreatic-Tumor-SEG / PFNet** | Positioning-focus cascade — **two-stage, organ-specific** | Single-stage, organ-agnostic rare head (portable to BraTS/AMOS) |

**Why this is the actual headline, not the wavelet.** `RESEARCH_FOCUS.md` §0 is
explicit: the headline is **rare/small-structure robustness under a tiny budget**,
not the wavelet primitive. Every lightweight net in the survey (LightM-UNet,
UltraLight, Slim UNETR/++, VeloxSeg) *admits* accuracy loss on small/rare classes.
**No efficient model makes rare-class robustness its headline.** The rare head +
Tversky + boundary loss + surface-Dice reporting is what makes WaLM-Net's claim
measurable and different — the wavelet is the *mechanism*, the rare-class result is
the *contribution*.

**Why boundary metrics matter.** `PAPERS_3D_SEG.md` and `RESEARCH_FOCUS.md` §3b gap
#2: Mamba nets are cheap but blur edges, and **most report only volumetric Dice**.
KiTS23 scores on **Surface Dice** too. Owning surface-Dice/HD95 is open white-space.
The Laplacian boundary loss + MONAI surface-Dice/HD95 reporting is how we claim it.

---

### 1.5 EffiDec3D-style efficient decoder (`UpBlock`, `AttentionGate`, reduced `decoder_channels`)

**What it is.** Channel-reduced decoder widths `(128, 64, 32, 32)` vs encoder
`(32, 64, 128, 256)`, with attention-gated skips and deep supervision heads at
reduced resolutions.

**Why this, vs every surveyed decoder:**

| Surveyed work | Its decoder | WaLM-Net's response |
|---|---|---|
| **EffiDec3D** (CVPR'25) | Channel reduction + drop low-value high-res layers — **decoder-only bolt-on, encoder untouched** | We pair it with an efficient wavelet encoder = **both ends efficient** (gap #4) |
| **3D UX-Net / SwinUNETR** | Heavy decoders (what EffiDec3D prunes) | Inherit the pruning logic |
| **SegFormer3D** | All-MLP lightweight decoder — **workshop-tier, loses detail** | Attention-gated skips preserve detail; rare head adds it back |
| **STU-Net** | Scales 14M→1.4B — **big models slow** | Fixed small budget (<15M target) |

**Why attention-gated skips.** Standard U-Net skips pass encoder noise to the
decoder; attention gates (the AttentionGate module) filter skips with the decoder
gating signal. This is a well-validated detail-preserving trick (Attention U-Net
lineage) that costs almost nothing and complements the HF branch — the HF branch
*adds* boundary signal, the attention gate *removes* irrelevant skip signal.

---

## 2. The "crowded space" reality check (§3c of RESEARCH_FOCUS.md)

`RESEARCH_FOCUS.md` §3c is blunt: wavelet/frequency + Mamba is **partly taken** and
moving fast — Topology-Aware Wavelet Mamba, HybridMamba, EM-Net, SSFMamba, SegResMamba,
SwiM-UNet, Hybrid Mamba-SAM/DCT (Dice 0.906). The justification is **not** "we also do
wavelet+Mamba." It is the **specific four-way intersection** none of them occupy:

| Axis | WaveFormer | SegMamba | HybridMamba | EM-Net | LHU-Net | EffiDec3D | LightM-UNet | **WaLM-Net** |
|---|---|---|---|---|---|---|---|---|
| Learnable lifting wavelet (PR guaranteed) | ✗ (fixed Haar) | ✗ | ✗ (channel filters) | ✗ (channel freq) | ✗ | ✗ | ✗ | **✓** |
| Linear-cost global mixer | ✗ (attention) | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | **✓** |
| Single-branch HF detail path | ✗ | ✗ | ✗ (dual branch) | ✗ | ✗ | ✗ | ✗ | **✓** |
| Dedicated rare-class (tumor+cyst) head | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Boundary loss + surface-Dice/HD95 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| CT kidney tumor+cyst validation | ✗ | ✗ | partial (kidney only) | ✗ | ✗ | ✗ | ✗ | **✓** |
| No external pretraining | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |

No column has all checkmarks except WaLM-Net. **That is the defense.** Each individual
piece exists somewhere; the combination does not, and the combination is what the
rare-class tumor+cyst problem requires.

---

## 3. Why NOT the alternatives (the negative justification)

### 3a. Why not just use nnU-Net / MedNeXt?
Heavy, slow, no long-range global mixing (MedNeXt), no explicit detail/frequency
path, tumor/cyst still weak. WaLM-Net targets the same tumor/cyst Dice at <20% of
nnU-Net compute (`RESEARCH_FOCUS.md` §7 success criterion).

### 3b. Why not just use SegMamba?
Scan-order bias, blurs fine boundaries, under-segments small/rare classes
(`PAPERS_3D_SEG.md` §1, §10). WaLM-Net keeps the linear Mamba backbone but adds the
HF branch (fixes blur) + rare head (fixes rare-class under-segmentation) + bidirectional
scan (fixes scan-order bias).

### 3c. Why not just use WaveFormer?
Fixed Haar (cannot adapt per organ), attention still core (not linear), never
validated on CT tumor/KiTS. WaLM-Net makes the basis learnable (with PR guarantee),
swaps attention for linear SSM, and targets CT tumor+cyst.

### 3d. Why not just use HybridMamba?
Two full branches = compute overhead, heuristic fusion, tumor still the weak link
even with kidney 97.54% Dice. WaLM-Net uses a single gated HF branch (half the cost)
and a dedicated rare head (attacks the tumor weak link directly).

### 3e. Why not just use EffiDec3D?
Decoder-only bolt-on; encoder untouched. WaLM-Net pairs EffiDec3D-style decoder
pruning with an efficient wavelet encoder — both ends efficient (gap #4).

### 3f. Why not just use LightM-UNet / UltraLight?
Extreme lightness but **accuracy drops on small/rare structures** — exactly the
tumor problem. WaLM-Net trades a few params for the HF branch + rare head that
directly fix that failure.

### 3g. Why not the DINOv3/VISTA3D/SwinUNETR/VLM-critic pipeline we already ran?
Our own evidence (`RESEARCH_FOCUS.md` §1): 93M–266M params, up to 33h, unstable
(Dice 0.42–0.65), cyst-blind (cyst 0.000 median in v3), and it **violates the no-VLM
guardrail**. More params made it worse (v5 266M < v4 93M). The comparative advantage is
domain insight into the tumor+cyst failure mode, not scale.

### 3h. Why not VeloxSeg / JL-lemma random projection?
Very new, unproven adoption; JL random projection **may hurt fine detail** — exactly
what tumor/cyst segmentation cannot afford. No rare-class focus.

### 3i. Why not SAM-derived (SAM-Med3D, FastSAM3D, HER-Seg, TP-Mamba, VISTA3D)?
All depend on a heavy SAM/VISTA backbone (guardrail violation) and are
adapter/prompt-tuning, not from-scratch efficient nets. VISTA3D is what v3/v4/v5
already ran and cyst-blinded.

---

## 4. Honest limitations & kill criteria (from RESEARCH_FOCUS.md §6)

This justification is **conditional**, not absolute. The model is defensible *if*
the ablations in §7 hold:

1. **Learnable lifting must beat fixed Haar by ≥0.5% tumor Dice.** If not, the
   wavelet novelty collapses and we pivot to Plan B (conv backbone + frequency detail
   branch + efficiency-first story). The lazy-wavelet init makes this ablation clean.
2. **Mamba space saturation.** If a 2026 paper lands the same combo, lead with the
   rare-class tumor+cyst + boundary-metrics-on-KiTS story (none target this) and keep
   lifting as the primitive.
3. **Single-branch must not underperform dual-branch on tumor.** If it does, add a
   thin second branch but keep <15M params.
4. **Must beat nnU-Net on tumor.** If not, pivot the claim to efficiency-at-iso-accuracy
   (same Dice, 5–10× less compute) — still publishable.

The architecture is designed so each of these failures has a graceful fallback
(Plan B, dual-branch, efficiency-first claim) without a full rewrite. That
optionality is itself part of the justification — the bet is hedged.

---

## 5. What is genuinely novel (the contribution claim)

This is the part reviewers will probe hardest, so it is stated with maximum
honesty. There are **three layers** of novelty, ranked from strongest to weakest.
Lead with layer 1; layer 3 is supporting evidence, not a headline claim.

### Layer 1 — The strong, defensible claims (what we present as novel)

**N1. Learnable lifting-scheme 3D wavelet with guaranteed perfect reconstruction,
used as the encoder downsampler.**
- *What's new:* The predict/update operators (`_Lifting1D.P`, `.U`) are learned
  depthwise conv3ds, initialised to zero (lazy wavelet = Haar). The lifting
  algebra (`detail = odd − P(even)`, `approx = even + U(detail)`, with the exact
  inverse in `_Lifting1D.inverse`) means **perfect reconstruction holds for any
  learned P, U — by construction, not by regularization.**
- *Why no one else has this:* WaveFormer uses **fixed Haar** (`wave_helper.py`);
  HybridMamba uses **learnable channel-wise frequency filters** (not a spatial
  lifting wavelet); EM-Net uses **channel-wise spectral learning**; WMREN is
  2D-leaning. A *spatial* lifting wavelet that is *learnable* AND *invertible by
  construction* is unoccupied for 3D medical seg as of mid-2026.
- *The moat:* A generic learnable filter bank has no reconstruction guarantee —
  it can distort the signal. Lifting gives adaptivity **and** invertibility
  simultaneously. This is the one primitive no surveyed work can match without
  changing their wavelet core.

**N2. Rare-class (tumor+cyst) robustness as the *headline* contribution of an
efficient net, via a dedicated rare-class head + Tversky(α<β) + boundary loss.**
- *What's new:* Every lightweight net in the survey (LightM-UNet, UltraLight,
  Slim UNETR/++, VeloxSeg) **admits** accuracy loss on small/rare classes as the
  price of efficiency. **No efficient model makes rare-class robustness its
  headline.** WaLM-Net's `rare_head` (2-channel tumor/cyst BCE) + Tversky with
  `tversky_rare_boost=1.5` + Laplacian boundary loss + surface-Dice/HD95 reporting
  inverts that framing: efficiency is the *enabler*, rare-class Dice is the *claim*.
- *Why this is novel, not just engineering:* The framing is the contribution.
  "Efficient Mamba" is saturated; "efficient Mamba that is *specifically* robust to
  the tumor+cyst failure mode, measured by boundary metrics, on CT" is not.
- *Evidence it's needed:* Our own v3–v6b runs (`RESEARCH_FOCUS.md` §1) are
  cyst-blind (cyst 0.000 median in v3) despite 93–266M params. The field's universal
  weak spot is tumor (~0.57–0.74) and cyst (~0.73) Dice. This is a documented,
  measurable, fundable gap.

**N3. Single-branch high-frequency detail path with gated fusion (not dual-branch).**
- *What's new:* HybridMamba needs **two full branches** (spatial + frequency) with
  heuristic fusion. WaLM-Net uses **one** depthwise-separable HF branch
  (`HFBranch`) gated into the main path via a learned sigmoid (`WaLMEncoderStage`).
  Half the branch cost, principled gate, same detail-preservation goal.
- *Why it matters:* This is `RESEARCH_FOCUS.md` §3b gap #5 — "a single light branch
  that still keeps high-freq detail is open." The gate is the trick: it lets one
  cheap branch carry the boundary/small-structure signal without a second encoder.

### Layer 2 — The combination claim (the intersection no one occupies)

**N4. The four-way intersection itself.** No single surveyed work has all of:
learnable-PR wavelet + linear SSM + single-branch HF detail + rare-class head +
boundary metrics + CT tumor+cyst. The §2 matrix proves this. The novelty is **the
combination targeted at the tumor+cyst problem**, not any one piece. This is the
honest framing: each piece exists somewhere; the combination, validated on CT
kidney tumor+cyst with surface-Dice, does not.

### Layer 3 — Supporting contributions (not headline claims)

**N5. Bidirectional SSM averaging for scan-order debiasing.** Running the mixer
forward and backward and averaging (`TokenMixer.forward`) is the cheapest linear
fix for SegMamba's documented scan-order bias. Not novel as an idea (bidirectional
scans exist), but novel *in this efficient-CT-tumor context*.

**N6. Dependency-free `_SSMLite` fallback.** Lets the architecture run without
`mamba_ssm`/`causal-conv1d` — engineering robustness, not a novelty claim. It exists
so Plan B (`RESEARCH_FOCUS.md` §6) is survivable.

**N7. EffiDec3D-style decoder paired with an efficient wavelet encoder.** EffiDec3D
fixed only the decoder; we pair it with an efficient encoder (gap #4). Incremental,
but closes the "both ends efficient" gap.

### What we do NOT claim as novel (to stay honest)

- ❌ "Mamba for 3D segmentation" — SegMamba/U-Mamba/nnMamba did it first.
- ❌ "Wavelet + Mamba" — Topology-Aware Wavelet Mamba, HybridMamba, EM-Net exist.
- ❌ "Efficient 3D U-Net" — WaveFormer, LHU-Net, LightM-UNet exist.
- ❌ "Tversky loss for rare classes" — well-established.
- ❌ "Attention gates / deep supervision" — standard U-Net machinery.
- ❌ "Boundary loss" — exists (HFF-Net, Boundary Loss papers).

**The contribution is the *specific intersection* (N1+N2+N3+N4) targeted at the
CT tumor+cyst problem with boundary metrics — not any isolated piece.**

---

## 6. Is WaLM-Net true 3D or 2D slices? (the volumetric-ness audit)

**Answer: true volumetric 3D. Not 2D slices, not 2.5D, not slice-stacked.**

This matters because several "3D" works in the survey are actually 2D-leaning
(WMREN is 2D-leaning; UltraLight VM-UNet is 2D/skin; Swin-UMamba is mostly 2D;
SegFormer3D workshop-tier). WaLM-Net is not. The audit against the code:

### 6.1 Every conv is `nn.Conv3d`, every norm is `nn.InstanceNorm3d`

Verified by grep across `walmnet_kits23.py` — **zero** `Conv2d` / `InstanceNorm2d` /
`ConvTranspose2d` calls exist in the file. Every spatial operator is volumetric:

| Module | Operator | Evidence (line) |
|---|---|---|
| `_Lifting1D.P` / `.U` | `nn.Conv3d(..., (1,1,kernel))` | lines 177, 179 |
| `GSC.conv1/conv2/gate` | `nn.Conv3d(..., 3, padding=1)` | lines 307, 309, 310 |
| `GSC.norm` | `nn.InstanceNorm3d` | line 306 |
| `WaMambaBlock.mlp` | `nn.Conv3d(..., 1)` | lines 326, 328 |
| `HFBranch.dw/pw` | `nn.Conv3d(..., groups=...)` | lines 348, 349 |
| `ResBlock.conv1/conv2/skip` | `nn.Conv3d(..., 3, padding=1)` | lines 361, 363, 366 |
| `AttentionGate.w_g/w_x/psi` | `nn.Conv3d(..., 1)` | (AttentionGate class) |
| `UpBlock.up` | `nn.ConvTranspose3d(..., 2, stride=2)` | (UpBlock class) |
| `WaLMNet.head` / `rare_head` | `nn.Conv3d(..., 1)` | (WaLMNet class) |

### 6.2 Patches are volumetric cubes, not slices

- `patch_size: (128, 128, 128)` for full training (line 80)
- `patch_size: (96, 96, 96)` for quick test (line 137)
- These are **isotropic 3D crops**, not `(H, W, 1)` slice patches.

### 6.3 The token mixer operates on the full 5D voxel tensor

`TokenMixer.forward` (lines 291–298):
```python
b, c, d, h, w = x.shape              # 5D: batch, channel, depth, height, width
t = x.flatten(2).transpose(1, 2)      # (B, D*H*W, C) — flattens the WHOLE volume
...
return t.transpose(1, 2).reshape(b, c, d, h, w)   # back to 5D
```
The SSM sequence length is `D × H × W` (e.g. `128³ = 2,097,152` tokens at full
res, or `64³ = 262k` at the bottleneck) — **the entire 3D volume is one sequence**,
not per-slice. This is what gives true volumetric long-range mixing: a voxel in the
top-back of the kidney can attend to a voxel in the bottom-front, which 2D-slice
models fundamentally cannot do.

### 6.4 The wavelet is a separable 3D lifting transform, not a 2D wavelet applied slice-wise

`LearnableLiftingWavelet3D.forward` applies `_Lifting1D` along **three distinct
spatial axes** (W=4, H=3, D=2 — the `dim` argument to `_Lifting1D.forward`), producing
the full 8-sub-band 3D decomposition (1 LL + 7 highs: `aad, ada, add_, daa, dad,
dda, ddd`). A 2D wavelet would produce only 4 sub-bands (LL, LH, HL, HH) per slice.
The 7-high sub-band structure is the signature of true 3D wavelet decomposition.

### 6.5 Sliding-window inference is volumetric

`validate()` uses MONAI `SlidingWindowInferer` with `sw_overlap=0.5` over the full
3D volume — not per-slice inference stitched together.

### 6.6 Why this matters for the novelty claim

True 3D is **not** itself novel (nnU-Net, MedNeXt, SegMamba are all true 3D). But
it is a **prerequisite** for the rare-class claim: tumor and cyst are small 3D blobs
(2–3 voxels, `analysis.txt`), and a 2D-slice model sees them as even smaller 2D
cross-sections (often 1 pixel or absent in a given slice). Volumetric context is
necessary for the rare-class head to work. So the 3D-ness is not the contribution —
it is the **substrate** that makes N2 (rare-class robustness) possible.

### 6.7 Comparison: which surveyed works are actually 3D vs 2D-leaning

| Work | True 3D? | Note |
|---|---|---|
| nnU-Net, MedNeXt, SegMamba, U-Mamba, nnMamba | ✓ true 3D | Volumetric convs/SSM |
| 3D UX-Net, SwinUNETR, STU-Net | ✓ true 3D | Volumetric |
| **WaLM-Net** | **✓ true 3D** | Volumetric convs + 3D lifting wavelet + volumetric SSM |
| WaveFormer | ✓ 3D | But fixed Haar |
| HybridMamba, EM-Net | ✓ 3D | But dual-branch / channel-freq |
| LHU-Net | ✓ 3D | But no SSM/HF path |
| LightM-UNet | ✓ 3D | But rare-class blind |
| **WMREN** | ⚠ 2D-leaning | IJCAI'25, multi-organ but 2D-leaning |
| **UltraLight VM-UNet** | ⚠ 2D/skin | 0.049M, ISIC skin lesion |
| **Swin-UMamba** | ⚠ mostly 2D | ImageNet-pretrained Swin, abdominal/endoscopy |
| **SegFormer3D** | ✓ 3D but workshop-tier | Loses detail |
| SAM-Med3D, FastSAM3D | ✓ 3D | But SAM-dependent (guardrail violation) |

WaLM-Net sits firmly in the true-3D column, alongside the heavyweights it competes
with — not in the 2D-leaning column where several "efficient" works live.

---

## 7. Summary — the one-sentence defense

> WaLM-Net is justified because it is the **only true-volumetric-3D architecture in
> the surveyed corpus that simultaneously** (a) makes the wavelet basis **learnable
> with guaranteed perfect reconstruction** (vs WaveFormer's fixed Haar, HybridMamba's
> channel filters), (b) keeps global mixing **linear** over the full D×H×W volume
> (vs nnU-Net/MedNeXt's conv cost, WaveFormer's attention), (c) preserves
> high-frequency detail in a **single light branch** (vs HybridMamba's dual-branch
> overhead, Mamba's boundary blur), (d) makes **rare-class tumor+cyst robustness its
> headline** with a dedicated head + Tversky + boundary loss (vs every efficient net
> that admits rare-class accuracy loss), and (e) validates on **CT kidney tumor+cyst
> with surface-Dice/HD95** that no surveyed work reports — and it does all this under
> a **<15M-param, no-pretraining, no-ensemble, no-VLM** guardrail that the field's
> heavyweights violate.
