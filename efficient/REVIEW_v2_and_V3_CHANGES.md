# WaLM-Net v2 review + what v3 fixes

Verified by a multi-agent review (5 investigators → adversarial verification of
every finding). Evidence is `file:line`. **v3 = `walmnet_v3.py`** (driver around
the unchanged `walmnet_kits23.py` model). Some items are repo/doc fixes that live
outside v3 and are still open — see §3.

## 0. What actually holds up
- **No train/val leakage.** Split is deterministic and disjoint: `val=items[:n_val]`,
  `train=items[n_val:]` on a sorted case list (`walmnet_kits23.py:632-633`), identical
  in training and every eval script.
- **v2 is a training/eval driver, not a new network.** It reuses `W.build_model /
  get_dataloaders / build_loss / validate` unchanged (`walmnet_v2.py:449-460`).
- **Metrics files are mutually consistent** (same checkpoint; overlapping per-case
  numbers match). The 97-case KiTS-HEC row reproduces exactly:
  avg **0.818**, Kidney+Masses **0.941**, Masses **0.763**, Tumor **0.750**.
- The model is ~14.26M params; wavelet reconstruction is exact.

## 1. Correctness defects (must fix before quoting numbers)

| # | Defect | Evidence | v3 status |
|---|---|---|---|
| P0-1 | **Dice/NSD scored at 1.5 mm resampled resolution, not native** — `Spacingd((1.5,1.5,1.5))` applied to *both* image and label, no inverse resample anywhere. Not KiTS-comparable; NN-downsampling can erase sub-1.5 mm lesions. | `walmnet_kits_metrics.py:86`, `walmnet_eval.py:169,295`, `walmnet_kits23.py:707` | **Fixed** — `evaluate_native` resamples the prediction back to the native label grid and scores in native space. |
| P0-2 | **NSD tolerance is really 1.5 mm, not 1.0 mm** — `SurfaceDiceMetric` called with no `spacing`, so thresholds are in voxels; also mis-used on a single-channel map with `include_background=False`. Training uses a *different* tol (1.5) than the report. | `walmnet_kits_metrics.py:95,119` | **Fixed** — v3 passes native `spacing` + a documented tolerance and a proper 2-channel one-hot. |
| P0-3 | **Cherry-picked "targeted"/8-case metrics** (cases chosen by largest structure) reported as if representative; cyst means 1.6–2.5× the honest number. `size` selection mislabeled "unbiased". | `walmnet_eval.py:265,271-273,283`; `IMPROVEMENT_PLAN.md:310` | **Partly** — v3 reports **only the full held-out set**. The stale targeted JSONs, the `walmnet_eval.py:265` comment, and the `IMPROVEMENT_PLAN.md:310` instruction still need editing (§3). |
| P0-4 | **Hardcoded, untraceable per-class table/plot numbers** — `WaLM-Net` row (Kidney 91.78 / Tumor 74.97 / Cyst 58.39 / Mean 75.05 + IoU) exists **only** in the generator; no committed JSON produces per-class Kidney/Cyst or any IoU. Plot uses *guessed* competitor Dice + placeholder FLOPs=140. | `make_literature_table.py:38-40`, `make_sota_plot.py:11-13,33` | **Enabled** — v3's native eval now emits **real per-class Dice + IoU + recall/precision**; the generators must be re-pointed at that JSON and the guessed plot numbers removed (§3). |

## 2. Completeness / selection / robustness

| # | Issue | Evidence | v3 status |
|---|---|---|---|
| P1-5 | **Run truncated at 44/120 epochs**; every validated epoch was `*BEST*` with strictly rising score → not converged, no early stop. `best.pth` = epoch 40. Numbers are from an unconverged model. | `history.json` (44 epochs), `walmnet_kits23.py:150` | **Enabled** — v3 trains longer (full=250) and records val score every `val_every` (fixes v2's `score:null`). Needs a real re-run. |
| P1-6 | **Checkpoint selected on the first 8 sorted cases** (cyst=0/NaN on 6/8) — high-variance signal for the rare metric. | `walmnet_kits23.py:152,711` | **Improved** — v3 uses more val cases + full-val at the end. Stratifying to guarantee cyst cases is a further win. |
| P2-8 | **Odd-dimension wavelet crash** — `F.pad(mode="replicate")` on a 5D tensor raises; any `--patch` that halves to an odd dim (e.g. 120³→…→15) hard-crashes. | `walmnet_kits23.py:186-188` | **Fixed in the shared model** — replaced with manual edge replication (works for any ndim; verified 15³ + a 24³ forward). Backward-compatible. |
| P2-7 | **`mean_fg` divides by fixed 3.0** even when a class is absent from a subset → silent deflation. | `walmnet_eval.py:339` | v3 doesn't use that path; the old script still needs the one-line fix (§3). |
| P2-9 | **FLOPs label inconsistent** (fvcore MACs reported as FLOPs, ~2× low vs the ptflops/thop paths) and **the checkpoint used the SSM-lite fallback, not real Mamba** (`mamba_ssm` absent) — not recorded anywhere. | `walmnet_flops.py:53`, `training.log:4` | **Partly** — v3 records `mixer` in its metrics JSON; the FLOPs-2× fix is a `walmnet_flops.py` edit (§3). |
| P1-10 | **Docs overclaim** ("the first…", "the only true 3D…") with no protocol caveat; `IMPROVEMENT_PLAN.md:2.2` alleges a `Spacingd` bug in `validate()` that **does not exist** (line 707 already applies it) and credits a phantom +1–2%. | `MODEL_JUSTIFICATION.md`, `IMPROVEMENT_PLAN.md:2.2` | Open (§3). |

## 3. Still open — repo/doc edits outside v3 (recommended)
1. **Re-run the 97-case eval with v3** (`walmnet_v3.py --evaluate --mode medium`) and treat the **native** numbers as the headline; re-caveat everything else as "1.5 mm-resampled".
2. **Re-point `make_literature_table.py` / `make_results_table.py`** at v3's `kits_metrics_native.json` (now has real per-class Dice/IoU); delete guessed competitor Dice + placeholder FLOPs in `make_sota_plot.py` (or mark "ESTIMATED").
3. Flag the targeted/8-case JSONs `do_not_cite`; fix the `walmnet_eval.py:265` "unbiased" comment; drop the `IMPROVEMENT_PLAN.md:310` instruction to cite the 8-case file.
4. Fix `walmnet_eval.py:339` `mean_fg` denominator; multiply fvcore MACs by 2 in `walmnet_flops.py`.
5. Delete `IMPROVEMENT_PLAN.md` item 2.2 (phantom fix); add a caveat block (resolution, single seed, mixer=lite, undertrained) wherever headline numbers/novelty claims appear.
6. **Re-train to convergence** (resume to ≥120 epochs) before publishing any number.

## 4. What v3 adds (mapping to the fixes above)
- **Native-resolution KiTS HEC evaluation** (P0-1) + proper NSD with spacing/tolerance (P0-2).
- **Real per-class Dice / IoU / recall / precision** in `kits_metrics_native.json` (P0-4) — cyst recall vs precision are now separable (the FP problem is visible).
- **Recall-safe post-processing** actually wired in (v2 left `postprocess_tumor_in_kidney` with zero call sites): keep 2 largest kidney components, drop tumor/cyst far from kidney, remove only *tiny* T/C debris (thresholds kept low to protect small real cysts).
- **Flip TTA** + Gaussian blending + overlap 0.6 (boundary/NSD).
- **Validation records the KiTS metric and selects `best.pth` on it** (fixes `score:null`).
- **Opt-in nnU-Net-style clip+z-score CT normalization** (`--ct-norm znorm`) for a fresh train.
- **Odd-dimension wavelet crash fixed** in the shared model.

## 5. Honest headline (use this framing)
> On a **local 80/20 held-out split** of the KiTS23 training set, WaLM-Net (~14.3M
> params, single model, no ensemble/pretraining, **SSM-lite mixer**) reaches KiTS-HEC
> avg Dice ~0.82 / Tumor ~0.75 — **competitive with the 2nd-place tier** — *pending*
> (a) native-resolution re-scoring, (b) training to convergence, and (c) cross-dataset
> (BraTS/AMOS) validation. Cyst remains the weak class (recall- and FP-limited).

*Note: one verifier agent failed its structured-output retries; 46/47 agents succeeded.*
