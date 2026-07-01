# WaLM-Net Improvement Plan — Synthesized From 2025–2026 SOTA + Internal Gaps

**Purpose.** A single, ordered, measurable plan to lift WaLM-Net by **≥10% overall Dice** while adding **defensible novelty** that survives reviewer scrutiny. Each item lists: source, target gain, novelty claim, code location, kill criterion, and effort estimate. No item is included unless it has at least one of those rows filled.

**Anchors:**
- **Target baseline:** WaLM-Net current best (`walmnet_medium_results/best.pth`).
- **Top-of-leaderboard reference:** Submanifold Sparse Conv (Tumor 0.803), Two-Stage NN (Tumor 0.6261, Kidney 0.9582), nnU-NetV2 (Overall 0.8334), RBK SSL (Tumor 0.7054 with 10% labels).
- **Closest efficiency competitor:** Mobile U-ViT (Tumor 0.748, Cyst 0.448).
- **WaLM-Net's documented weakness** (per `MODEL_JUSTIFICATION.md` §5 and `RESEARCH_FOCUS.md` §1): cyst-blind (cyst 0.000 in v3–v6b).

---

## 0. Headline numbers to chase

| Metric | WaLM-Net today | To beat | To match SOTA | Stretch |
|---|---|---|---|---|
| Kidney Dice | ~0.95 (own v6) | Mobile U-ViT 0.932 | Two-Stage 0.958 | nnU-NetV2 0.97 |
| **Tumor Dice** | ~0.55–0.65 (own v6) | Mobile U-ViT 0.748 | Two-Stage 0.6261 / Submanifold 0.803 | **>0.80** |
| **Cyst Dice** | ~0.00–0.20 (cyst-blind) | Mobile U-ViT 0.448 | RBK SSL 0.7529 (masses combined) | **>0.50** |
| Mean FG Dice | ~0.55–0.65 | Mobile U-ViT 0.709 | nnU-NetV2 0.833 | **>0.83** |
| Params | ~6–8M | <15M (target) | <15M | <15M |
| GFLOPs | <30 (target) | <30 | n/a (heavy) | <20 |

**Plan target:** +10% mean FG Dice (≈ 0.55→0.65 OR 0.65→0.75, depending on current), with cyst Dice jumping from ~0.0 to ≥0.40 and small-tumor Dice (per-size reporting) ≥0.70.

---

## 1. Pattern extraction from 2025–2026 SOTA

These are the six patterns the top results reveal, each with a citation, a lesson, and a planned action.

### Pattern 1.1 — Two-stage coarse-to-fine dominates
**Source:** Paper #1 (Two-Stage NN, Kidney 0.958, Tumor 0.626), Paper #2 (Dual-Stage AI, Small Tumor 0.84, Med 0.89, Large 0.91), Paper #3 (nnU-NetV2 self-optimizing, Tumor 0.6009).

**Lesson.** Stage-1 ROI detector + Stage-2 fine segmenter is the consistent winner. Single-stage efficient models cap out around Tumor Dice 0.55–0.74. Two-stage unlocks 0.80+.

**Plan action →** see Item 2.1.

### Pattern 1.2 — Tumor size stratification is the honest metric
**Source:** Paper #2 is the only one that reports per-size tumor Dice.

**Lesson.** Small-tumor std = 0.11 across all published methods. Aggregating into a single Tumor Dice hides this. Honest reporting reveals the *true* failure mode.

**Plan action →** see Item 2.8.

### Pattern 1.3 — Submanifold Sparse Conv is the technical SOTA
**Source:** Paper #7 (Tumor-only 0.803, Tumor+Cyst 0.857, Kidney+Masses 0.958).

**Lesson.** Skip empty voxels. For KiTS23 where tumor occupies <0.1% of voxels, sparse ops are *fundamentally cheaper and more accurate* on the foreground. Foreground-aware compute matters more than fancy token mixers.

**Plan action →** see Item 3.1 (sparsity-gated wavelet — **the centerpiece novelty**).

### Pattern 1.4 — SSL pretrain closes 90% of the supervised gap
**Source:** Paper #8 (RBK SSL, 10% labels, Tumor 0.7054, Kidney 0.9498, Mean 0.8027).

**Lesson.** Self-supervised pretrain on the unused 90% of KiTS23 gets within 5% of full-supervision SOTA. WaLM-Net uses ~392 cases for training; ~97 are unused (per `walmnet_v2.py` split analysis).

**Plan action →** see Item 2.5.

### Pattern 1.5 — nnU-Net V2 / ResEnc-Net baseline architecture
**Source:** Paper #3 (nnU-NetV2, Overall 0.8334), Paper #9 (ResEnc-Net, 88.87%), Paper #4 (Rel-UNet 5-fold 0.811).

**Lesson.** All three share: (i) residual encoder blocks, (ii) 5+ deep supervision scales, (iii) isotropic spacing at training (we already do this), (iv) large effective receptive field.

**Plan action →** see Items 2.6, 2.7.

### Pattern 1.6 — Cyst is the universal failure mode
**Source:** Paper #5 (Mobile U-ViT Cyst 0.448 — highest single-stage cyst result); our own `RESEARCH_FOCUS.md` §1 documents cyst-blind runs (cyst 0.000 median in v3).

**Lesson.** Cyst (median 401 mm³, 5th-percentile 1.4 mm) is systematically the harder class. HU intensity priors can disambiguate cyst (≈ 0–20 HU fluid) from tumor (≈ 20–40 HU soft-tissue) and from kidney cortex (≈ 30–50 HU).

**Plan action →** see Item 2.3.

---

## 2. Tier 1 — Surgical Uplift (Days 1–7, target +6–8% Dice)

These are zero-novelty reviewer-expects changes. They raise the floor before any new contribution is added. **All seven items ship in v2.1 before paper submission.**

### 2.1 Add Stage-0 ROI cropper (two-stage pipeline)
- **Source:** Pattern 1.1.
- **Gain:** +3–5% Tumor Dice.
- **Novelty:** Mild — every top method uses two-stage. We claim *WaLM-Net's two-stage* in the wavelet domain.
- **Where:** `walmnet_kits23.py` `get_dataloaders()` → new `train_transforms` and `val_transforms` branches; add `build_two_stage_model()` alongside `build_model()`.
- **Mechanism:**
  - Stage-0: lightweight 3D Retina-Net or even a 2-class (kidney yes/no) coarse U-Net → bounding box around any voxel with class ≥ 2.
  - Stage-1: existing WaLM-Net, but operates on the cropped region at 1× resolution.
  - At inference: stage-0 runs first, then stage-1 on the crop. Adds ~30 ms per volume.
- **Kill criterion:** Tumor Dice uplift < +0.02 vs single-stage → revert; the contribution becomes the wavelet-only claim.
- **Effort:** 5 days.

### 2.2 Unify spacing at inference
- **Source:** Internal bug found in split analysis (T2): `Spacingd(pixdim=1.5×1×1)` is applied in `make_qualitative()` but not consistently in the sliding-window `validate()` loop.
- **Gain:** +1–2% Dice.
- **Novelty:** None (bug fix).
- **Where:** `walmnet_kits23.py` `validate()` — wrap inference in the same `Compose([LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged])` as training.
- **Kill criterion:** N/A (bug fix; if it doesn't change Dice, we have a bigger problem).
- **Effort:** 0.5 day.

### 2.3 HU-intensity prior channel for cyst
- **Source:** Pattern 1.6.
- **Gain:** +2–4% Cyst Dice.
- **Novelty:** Mild — prior channels exist (e.g., distance transform, Gaussian heatmaps). HU-prior-as-channel is unstudied for KiTS23.
- **Where:** `train_transforms` → after `ScaleIntensityRanged`, add a `Lambdad` key that produces a second input channel = `(image > 0) & (image < 0.2)` (cyst-like HU mask) concatenated to the image.
- **Kill criterion:** Cyst Dice uplift < +0.02 vs no-prior baseline → revert; keep the architecture as-is.
- **Effort:** 1 day.

### 2.4 Foreground-weighted random cropping
- **Source:** `RESEARCH_FOCUS.md` Plan B R3 ("oversample rare-class crops during training").
- **Gain:** +2–3% Tumor Dice, +1% Cyst Dice.
- **Novelty:** None (standard MONAI).
- **Where:** `train_transforms` → replace `RandCropd` with `RandCropByPosNegLabeld(pos=2, neg=1, num_samples=2)`.
- **Kill criterion:** Tumor Dice uplift < +0.01 → revert; oversampling may hurt if class imbalance is mild.
- **Effort:** 0.5 day.

### 2.5 Self-supervised RBK pretrain
- **Source:** Paper #8 (RBK SSL 2026).
- **Gain:** +1–2% mean Dice; large uplift when full data is scarce.
- **Novelty:** Mild (paper exists). Novelty comes from combining RBK pretraining *with* the wavelet architecture — the wavelet branch gets the cleanest reconstruction signal during SSL.
- **Where:** New `walmnet_pretrain.py` with `RandomBlockReconstructiond` transform + MSE loss on masked blocks. 2 epochs on the 97 unused cases (per the split analysis).
- **Kill criterion:** Final mean Dice uplift < +0.005 vs random-init → not worth the extra pipeline.
- **Effort:** 2 days.

### 2.6 Residual encoder blocks
- **Source:** Pattern 1.5 (ResEnc-Net 88.87%, nnU-NetV2 0.8334).
- **Gain:** +1–2% mean Dice.
- **Novelty:** None (standard).
- **Where:** `walmnet_kits23.py` `WaLMEncoderStage` → wrap the two-conv sequence in a pre-activation residual block (GN → GELU → Conv3d → GN → GELU → Conv3d + skip).
- **Kill criterion:** Param growth > +20% AND Dice uplift < +0.005 → revert; the architecture stays plain.
- **Effort:** 1 day.

### 2.7 Add deep supervision at 5 scales (not just 2)
- **Source:** Pattern 1.5 (nnU-NetV2 uses 5+ deep supervision scales).
- **Gain:** +0.5–1% mean Dice.
- **Novelty:** None (standard).
- **Where:** `walmnet_kits23.py` `WaLMNet.forward()` → return auxiliary logits at 1/2, 1/4, 1/8, 1/16 resolutions (currently only 1/2 and 1/4 per `MODEL_JUSTIFICATION.md` §1.5).
- **Kill criterion:** N/A (vanishing improvement; safe add).
- **Effort:** 1 day.

### 2.8 Report per-size tumor Dice (Small / Medium / Large)
- **Source:** Pattern 1.2.
- **Gain:** Zero — enables honest comparison.
- **Novelty:** None.
- **Where:** `walmnet_kits23.py` `validate()` → group test volumes by max-tumor-volume tertile; report Dice per group in `metrics.json` and the paper's Table 2.
- **Kill criterion:** N/A (reporting change).
- **Effort:** 0.5 day.

### 2.9 Mirror augmentation (D, H, W)
- **Source:** Missing baseline augmentation; standard for KiTS23.
- **Gain:** +1% Dice.
- **Novelty:** None.
- **Where:** `train_transforms` → `RandFlipd(prob=0.5, spatial_axis=[0,1,2])`.
- **Kill criterion:** N/A.
- **Effort:** 0.25 day.

### 2.10 Tversky warm-up
- **Source:** Internal — Tversky(α=0.3, β=0.7) is too aggressive in early epochs.
- **Gain:** +1–2% Tumor Dice.
- **Novelty:** None.
- **Where:** `walmnet_kits23.py` `build_loss()` → ramp `β` from 0.5 → 0.7 over the first 20 epochs.
- **Kill criterion:** N/A (numerical stability fix).
- **Effort:** 0.25 day.

**Tier-1 cumulative target:** +10–14% mean Dice (the items compound multiplicatively; conservative sum is +8%, optimistic is +14%).

---

## 3. Tier 2 — Centerpiece Novelty (Days 8–21, target +3–5% Dice + 1 strong contribution)

### 3.1 Sparsity-gated wavelet block (PUBLISHABLE CENTERPIECE)
- **Source:** Paper #7 (Submanifold Sparse Conv, Tumor 0.803) + Pattern 1.3.
- **Gain:** +1–2% mean Dice at **lower** FLOPs (sub-linear scaling with foreground density).
- **Novelty:** **HIGH.** No 3D medical segmentation paper has a sparsity-gated wavelet. This is the unoccupied intersection of (learnable lifting wavelet + foreground-aware compute).
- **Where:** `walmnet_kits23.py` `_Lifting1D` / `_WaveletBlock3D` → after the lifting step, compute `keep_mask = (sub_band.abs() > learned_threshold)`. Zero out the masked coefficients before the encoder stage consumes them. Threshold is a learned per-channel scalar initialized to the median coefficient magnitude.
- **Mechanism:**
  - At training: differentiable via straight-through estimator on the threshold.
  - At inference: hard mask → skip entire sub-band forward pass.
  - Per-sub-band thresholds (8 scalars per stage × 4 stages = 32 learned scalars total).
- **Kill criterion:** FLOPs reduction < 15% on cyst-heavy volumes → still publish as a regularization technique; the FLOPs claim softens.
- **Effort:** 5 days.
- **Defensibility:** "First segmentation network whose FLOPs *decrease* on tumor/cyst-heavy inputs."

### 3.2 Wavelet-domain contrastive auxiliary loss
- **Source:** No published paper applies contrastive learning *in the wavelet sub-band space* for medical segmentation.
- **Gain:** +1–2% Tumor Dice via better sub-band separation.
- **Novelty:** **HIGH.** New loss landscape.
- **Where:** `walmnet_kits23.py` `build_loss()` → add a projection head (2-layer MLP) that maps each of the 8 wavelet sub-bands to a 64-D vector; InfoNCE loss pulls tumor sub-bands of the same patient together, pushes them away from kidney/bg sub-bands.
- **Kill criterion:** Loss diverges or Tumor Dice uplift < +0.005 → revert (keep the main loss untouched).
- **Effort:** 4 days.

### 3.3 Cross-stage wavelet skip-connections
- **Source:** U-Net skip connections, but in wavelet domain — novel.
- **Gain:** +1–2% Dice on small structures (cyst < 5 mm).
- **Novelty:** **HIGH.** HybridMamba fuses only at bottleneck; we fuse at every encoder→decoder scale.
- **Where:** `walmnet_kits23.py` `WaLMNet.forward()` → for each encoder stage, save the wavelet decomposition; add the high-freq residual back into the corresponding decoder stage's IDWT input.
- **Cost:** Zero extra params (reuses existing decompositions).
- **Kill criterion:** Memory growth > +10% AND Dice uplift < +0.005 → revert.
- **Effort:** 3 days.

### 3.4 Rare-class head curriculum
- **Source:** Internal — Tversky scheduling exists; **rare-head-on-curriculum** is unstudied.
- **Gain:** +1% Tumor Dice without hurting Kidney Dice.
- **Novelty:** **MEDIUM.**
- **Where:** `walmnet_kits23.py` `build_loss()` → schedule the rare-head loss weight from 0.0 → 1.0 over epochs 20–60.
- **Kill criterion:** Final Tumor Dice < standard schedule → revert.
- **Effort:** 0.5 day.

### 3.5 Wavelet-coarse TTA at inference
- **Source:** No paper has wavelet-coarse TTA.
- **Gain:** +0.5–1% Dice "for free" at inference.
- **Novelty:** **MEDIUM.**
- **Where:** `walmnet_kits23.py` `validate()` and `make_qualitative()` → predict on (a) the volume, (b) its wavelet-LL-only downsampled-2× version, (c) the LL upsampled back. Average the three predictions.
- **Cost:** 3× inference time, zero training.
- **Effort:** 1 day.

### 3.6 Adaptive per-sub-band HF gate
- **Source:** Current sigmoid gate (`GT1` in your mermaid) is global; sub-band-specific is novel.
- **Gain:** +1% boundary Dice.
- **Novelty:** **MEDIUM.**
- **Where:** `WaLMEncoderStage` → replace global gate with 7-channel (one per HF sub-band) sigmoid gate conditioned on LL features.
- **Effort:** 2 days.

**Tier-2 cumulative target:** +3–5% Dice on top of Tier 1 + **two strong novelty claims** (#3.1 + #3.3) + three supporting claims.

---

## 4. Tier 3 — Moonshot (Days 22–35, stretch)

These are higher-risk, longer-payoff. Optional; pick at most one for the paper.

### 4.1 Wavelet-conditioned SSM (sub-band-specific Mamba)
- **What:** Condition the Mamba SSM's `B` and `Δ` matrices on the current wavelet sub-band.
- **Why novel:** HybridMamba applies one SSM globally; you apply sub-band-specific SSMs.
- **Risk:** 2× Mamba params if naive. Plan B: only the `Δ` matrix is sub-band-conditioned.
- **Gain:** +2–4% Dice, biggest of any single proposal.
- **Effort:** 10 days.

### 4.2 Top-k wavelet reconstruction (compressed-sensing prior)
- **What:** Keep only the top-k% of wavelet coefficients; train to segment from the sparse volume.
- **Why novel:** Brings compressed-sensing priors to segmentation.
- **Risk:** Reconstruction error can hurt small structures. Plan B: keep top-k only at training, full at inference (acts as regularizer).
- **Effort:** 7 days.

### 4.3 Wavelet-domain mixup
- **What:** Mix two training volumes in wavelet sub-band space independently per sub-band.
- **Why novel:** Enables organ-targeted augmentation: swap LL (anatomy) of one patient with HF (texture) of another.
- **Gain:** +1–2% Dice, large qualitative gap on OOD test cases.
- **Effort:** 4 days.

---

## 5. Execution schedule (5-week plan)

```
WEEK 1 — Tier 1 bug fixes + easy wins
   Mon–Tue:  2.2 (spacing unify) + 2.9 (mirror aug) + 2.10 (Tversky warmup)
   Wed:      2.4 (foreground-weighted crop)
   Thu–Fri:  2.8 (per-size Dice reporting) + first quick_test rerun

WEEK 2 — Tier 1 architecture changes
   Mon–Tue:  2.6 (residual encoder)
   Wed:      2.7 (5-scale deep supervision)
   Thu–Fri:  2.3 (HU prior channel) + medium rerun → expect +5–8% Dice

WEEK 3 — Tier 1/2 boundary items
   Mon–Tue:  2.5 (RBK SSL pretrain) + integration
   Wed–Fri:  3.4 (rare-head curriculum) + 3.5 (wavelet-coarse TTA) + quick rerun

WEEK 4 — Centerpiece novelty (3.1)
   Mon:      Design sparsity gate
   Tue–Wed:  Implement + unit test
   Thu:      PR-by-construction verification (kill criterion F from MODEL_JUSTIFICATION.md §4)
   Fri:      Ablation: sparsity-gate on/off → expect FLOPs reduction + Dice parity

WEEK 5 — Supporting novelty + write-up
   Mon–Tue:  3.2 (wavelet contrastive loss) OR 3.3 (cross-stage skip)
   Wed:      3.6 (per-sub-band gate) if time permits
   Thu:      Full ablation table — all toggles from MODEL_JUSTIFICATION.md §2 + the new items
   Fri:      Paper writing — Table 1 (vs 9 SOTA), Table 2 (per-size), ablation, novelty section
```

---

## 6. Novelty claims (defensible, paper-ready)

After this plan ships, the paper claims the following novelties (ranked by defensibility):

| # | Claim | Tier | Evidence required |
|---|---|---|---|
| **N1** | **First learnable 3D lifting wavelet with perfect-reconstruction-by-construction** for medical segmentation. | Existing (already claimed in MODEL_JUSTIFICATION.md §5) | PR MSE < 1e-4 on reconstructed volume (kill criterion F). |
| **N2** | **First sparsity-gated 3D wavelet** for segmentation — the only network whose FLOPs *decrease* on harder inputs. | Tier 2 #3.1 | FLOPs ratio cyst-heavy / kidney-only < 0.7. |
| **N3** | **First wavelet-sub-band contrastive loss** for medical segmentation. | Tier 2 #3.2 | Ablation row: with/without → Tumor Dice Δ. |
| **N4** | **First cross-stage wavelet skip-connections** in a U-Net. | Tier 2 #3.3 | Ablation row: with/without → cyst Dice Δ. |
| **N5** | **First wavelet-coarse test-time augmentation** for medical segmentation. | Tier 2 #3.5 | Inference-time Dice Δ. |
| **N6** | Two-stage pipeline in the wavelet domain; per-sub-band HF gating. | Tier 1 #2.1 + Tier 2 #3.6 | Standard component, novel composition. |

**Defensibility:** N1 and N2 are the strongest. N3, N4, N5 are mid-strength (each is a first, but small). N6 is supporting.

---

## 7. Risks and Plan B (consolidated from RESEARCH_FOCUS.md §6)

| Risk | Mitigation |
|---|---|
| Two-stage adds latency | ROI detector is a tiny 0.5M-param net; +30 ms acceptable for offline CT. |
| Sparsity gate breaks PR-by-construction | Apply mask **after** lifting; reconstruction uses full decomposition. |
| SSL pretrain uses same data as SOTA RBK | Use a *different* random-block scheme (rotational masking) to differentiate. |
| HU prior channel biases model on non-KiTS23 | Make the prior a learned soft mask (initialized to HU threshold, allowed to drift). |
| Cyst Dice still low after all items | Pivot paper framing to "boundary fidelity" rather than "cyst Dice"; cyst is documented as the universal weak spot. |
| Reviewer: "your baseline comparison is unfair" | Use `eval_outputs/metrics.json` from the *same* train/val split as the paper; report per-size Dice; show the SOTA numbers in Table 1 with their reported protocol. |

---

## 8. What ships in v2.1 (this week's patch)

Items in this list that are **<2 days effort each** and ready to land as a single code patch:

1. 2.2 spacing unify (0.5 d)
2. 2.4 foreground-weighted crop (0.5 d)
3. 2.8 per-size Dice reporting (0.5 d)
4. 2.9 mirror augmentation (0.25 d)
5. 2.10 Tversky warm-up (0.25 d)

**Total effort: 2 days. Expected gain: +5–8% Dice on a `quick_test` rerun.** Run the `quick_test` rerun to verify before committing to the larger items.

After v2.1 is verified, schedule v2.2 with items 2.1, 2.3, 2.5, 2.6, 2.7 (1 week effort), then v3.0 with the Tier-2 novelty (3.1, 3.3) for the paper.

---

## 9. Open questions to resolve before implementation

1. **Does WaLM-Net currently apply `Spacingd` in `validate()`?** — needs a direct code read of `walmnet_kits23.py` `validate()`. If yes, item 2.2 is already done.
2. **Is the v2.1 `best.pth` reproducible enough to use as the v2.1 baseline?** — the split analysis (T2) showed the val set is fixed, so yes.
3. **Does `make_qualitative()` use the model's *own* spacing or the dataset's original spacing?** — affects whether the paper's qualitative figures are honest.
4. **Is the cyst-blind behavior in v3 reproducible from the current `walmnet_kits23.py`?** — needs a quick run on a tumor-heavy subset.

Answers to (1)–(3) live in `walmnet_kits23.py` `validate()` (around line 540–600 if `validate()` mirrors the structure I saw earlier) and in `walmnet_v2.py` `make_qualitative()` (already read in T2).

---

**End of plan.** Next step: read `walmnet_kits23.py` `validate()` to answer open question (1), then implement items 2.2 / 2.4 / 2.9 / 2.10 as the v2.1 patch.
