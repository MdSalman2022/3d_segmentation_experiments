# Efficient & SOTA 3D Medical Segmentation — Paper Survey (2024–2026)

> Scope: efficient yet near-SOTA 3D volumetric segmentation. No LLM/VLM core.
> Sorted **most novel / impactful → least**. "Novel" = changed how 3D seg is done
> or introduced a reusable architectural primitive. Use the **Gap** column to pick
> what our network attacks. Datasets noted where reported.

Legend — Venue tier: A* conf (MICCAI/CVPR/ICLR/ECCV/IJCAI), IEEE/Q1 journal (TMI, MedIA, JBHI, Patterns).

---

## Tier 1 — Architecture-defining / high novelty

### 1. SegMamba — Long-Range Sequential Modeling Mamba for 3D Seg
- **Year / Venue:** 2024 · MICCAI 2024 (also arXiv 2401.13560)
- **Method:** First pure-Mamba 3D seg net. Tri-orientated Mamba (ToM) + gated spatial conv; whole-volume long-range modeling at 64³ in linear cost.
- **Dataset:** BraTS2023, AIIB2023, CRC-500 (own colorectal).
- **Gap:** Scan-order sensitivity; weak high-freq/boundary detail; rare/small class under-segmented.

### 2. U-Mamba — CNN + SSM Hybrid for Biomedical Seg
- **Year / Venue:** 2024 · arXiv (heavily cited, nnU-Net-style self-config)
- **Method:** Hybrid CNN-SSM block in U-Net; local conv + Mamba long-range; self-configuring.
- **Dataset:** Abdomen CT/MRI, endoscopy, microscopy.
- **Gap:** Generic block; no frequency/detail path; efficiency not the headline (accuracy-first).

### 3. MedNeXt — Transformer-Driven Scaling of ConvNets
- **Year / Venue:** MICCAI 2023 (the 2024-25 efficient baseline everyone cites)
- **Method:** Fully ConvNeXt 3D enc-dec; residual up/down ConvNeXt; UpKern iterative kernel growth; compound scaling.
- **Dataset:** BTCV, AMOS, KiTS19, BraTS21.
- **Gap:** Large-kernel convs costly at scale; no long-range global token mixing; heavy variants slow.

### 4. WaveFormer — Wavelet-Driven 3D Transformer
- **Year / Venue:** 2025 · MICCAI 2025 (arXiv 2503.23764)
- **Method:** Multi-scale DWT for attention; wavelet summarize/reconstruct replaces heavy up/downsample; −93.75% attention compute, 7.7M params.
- **Dataset:** BraTS2023 (91.37% mean Dice), abdominal CT (synapse/BTCV-type).
- **Gap:** **Fixed Haar wavelet (not learnable);** attention still core; not validated on tumor/rare-class CT (KiTS).

### 5. EffiDec3D — Optimized 3D Decoder
- **Year / Venue:** 2025 · CVPR 2025
- **Method:** Channel-reduction across all decoder stages + drop low-value high-res decoder layers. −96% decoder params, −93% FLOPs vs UX-Net decoder.
- **Dataset:** multiple (BTCV, BraTS, etc. across backbones).
- **Gap:** **Decoder-only bolt-on;** encoder untouched; no new representation — pure pruning logic.

### 6. LHU-Net — Lean Hybrid U-Net
- **Year / Venue:** 2025 · MICCAI 2025 (arXiv 2404.05102)
- **Method:** Spatial-attention-first then channel-attention hybrid; 10.52M params, 51.75 GFLOPs; 4× fewer params / 20% fewer FLOPs than rivals, no pretraining.
- **Dataset:** Synapse, ACDC, BraTS, multi-organ.
- **Gap:** Hand-designed block ordering; no SSM/linear long-range; no high-freq path.

### 7. EM-Net — Efficient Channel + Frequency Learning with Mamba
- **Year / Venue:** 2024 · MICCAI 2024 (arXiv 2409.17675)
- **Method:** Frequency-domain learning layer + channel squeeze-reinforce Mamba + Mamba-infused decoder. Faster train, ~half params of peers.
- **Dataset:** BraTS, abdominal multi-organ.
- **Gap:** Frequency layer is channel-wise (not spatial wavelet sub-band); rare-class handling not targeted.

### 8. HybridMamba — Dual-Domain Mamba for 3D Seg
- **Year / Venue:** 2025 · MICCAI 2025 (arXiv 2509.14609)
- **Method:** Dual-domain (spatial + frequency) Mamba branches fused.
- **Dataset:** LiTS2017 (92.52 Dice), BraTS2023 (WT 94.10), **kidney 97.54% Dice** — direct KiTS-relevant baseline.
- **Gap:** Two full branches = compute overhead; fusion heuristic; tumor Dice still the weak link.

### 9. VeloxSeg — Johnson-Lindenstrauss-Guided Efficient Seg
- **Year / Venue:** 2025 · arXiv 2509.22307
- **Method:** Dual-stream CNN-Transformer; Paired Window Attention + JL-lemma-guided convolution (random-projection dim reduction). +26% Dice, **11× GPU / 48× CPU throughput**.
- **Dataset:** multimodal 3D benchmarks.
- **Gap:** Very new, unproven adoption; JL projection may hurt fine detail; no rare-class focus.

### 10. SegMamba-V2 — General 3D Seg Mamba
- **Year / Venue:** 2025 · IEEE TMI (Q1)
- **Method:** Improved tri-directional scan + scaling for general organs; long-range linear.
- **Dataset:** BraTS, abdominal, airway.
- **Gap:** Inherits scan-order limits; boundary detail still secondary.

---

## Tier 2 — Strong efficiency, incremental novelty

### 11. LightM-UNet — Mamba-Assisted Lightweight UNet
- **2024 · arXiv 2403.05246.** Pure Mamba residual layers, **~1M params.** Dataset: LiTS, lung CT.
- **Gap:** Accuracy drops on small/rare structures (tumor) — efficiency over robustness.

### 12. UltraLight VM-UNet — Parallel Vision Mamba
- **2024 · Patterns (Cell, Q1).** PVM layer, **0.049M params, 0.06 GFLOPs.** Dataset: skin lesion (ISIC, 2D-ish).
- **Gap:** Tiny but 2D/skin focus; not validated volumetric multi-organ; under-capacity for tumor.

### 13. Slim UNETR — Scaling Hybrid Transformers
- **2024 · IEEE JBHI (Q1).** Slim feature-gen + cross-attention; low resource. Dataset: BraTS, BTCV.
- **Gap:** Superseded accuracy-wise; transformer cost remains; no frequency path.

### 14. Slim UNETR++ — Lightweight 3D Seg Net
- **2025 · Med Biol Eng Comput (Q1).** Improved Slim UNETR. Dataset: BraTS, abdominal.
- **Gap:** Incremental; no long-range linear core.

### 15. 3D UX-Net — Large-Kernel Volumetric ConvNet
- **2023 · ICLR 2023 (key baseline).** Depthwise 7³ LK convs modernizing Swin. Beats SwinUNETR (0.929→0.938 FLARE). Dataset: FLARE2021, FeTA, AMOS.
- **Gap:** Large-kernel memory cost; no global token mixing; decoder heavy (what EffiDec3D prunes).

### 16. STU-Net — Scalable & Transferable U-Net
- **2023-24 · arXiv (widely used).** nnU-Net-based, scales 14M→1.4B params, supervised pretrain on TotalSegmentator. Dataset: TotalSeg, many.
- **Gap:** Big models slow; efficiency only at small scale; pretraining dependency.

### 17. SegFormer3D — Efficient Transformer for 3D Seg
- **2024 · CVPR Workshops 2024.** Hierarchical efficient attention, lightweight all-MLP decoder. Dataset: Synapse, ACDC, BraTS.
- **Gap:** Workshop-tier; efficient-attention approximations lose detail; no rare-class focus.

### 18. EfficientMedNeXt — Multi-Receptive Dilated Convs
- **2025 · MICCAI 2025.** Dilated multi-receptive conv blocks, lighter MedNeXt. Dataset: CT/MRI multi-organ.
- **Gap:** Pure conv — no long-range global; dilation gridding artifacts on thin structures.

### 19. D-Net — Dynamic Large Kernel + Dynamic Fusion
- **2024 · arXiv 2403.10674.** Dynamic large-kernel conv + dynamic feature fusion. Dataset: abdominal, brain.
- **Gap:** Dynamic kernels add runtime overhead; efficiency claim modest.

### 20. Tri-Plane Mamba — Adapting SAM for 3D Medical
- **2024 · MICCAI 2024.** Tri-plane decomposition + Mamba adapters on frozen SAM; parameter-efficient adaptation.
- **Gap:** Depends on SAM backbone (heavy); adapter only; not a from-scratch efficient net.

### 21. HER-Seg — Holistically Efficient Seg for High-Res
- **2025 · arXiv 2504.06205.** Efficient pipeline for high-resolution medical volumes.
- **Gap:** Pipeline/system-level efficiency, less a new representation; early-stage.

### 22. Topology-Aware Wavelet Mamba — Airway Seg
- **2025 · arXiv 2502.14363.** **Wavelet + Mamba already combined** + topology loss for tubular airway. Dataset: nasopharyngeal-CA CT airway.
- **Gap:** Narrow (airway/tubular only); topology prior not transferable to blobby tumor; **shows wavelet-Mamba space is partly taken → we need a different twist.**

### 23. WMREN — Wavelet Multi-scale Region-Enhanced Net
- **2025 · IJCAI 2025.** Wavelet multi-scale + region enhancement enc-dec. Dataset: multi-organ.
- **Gap:** 2D-leaning; region module hand-tuned; efficiency not headline.

### 24. EfficientViM — Hidden-State-Mixer SSD Vision Mamba
- **2025 · CVPR 2025 (general vision, portable).** HSM-SSD for linear global context + channel mixing.
- **Gap:** General-purpose (not medical-tuned); needs 3D adaptation; no boundary/rare-class design.

---

## Tier 3 — Lightweight / niche / foundation-efficiency

### 25. nnMamba — Mamba-In-Convolution
- **2024 · arXiv.** Mamba-InConv module + multi-channel spatial coupling. Dataset: seg + landmark + classification.
- **Gap:** Block-level contribution; modest efficiency edge.

### 26. Swin-UMamba — Swin + Mamba
- **2024 · arXiv.** ImageNet-pretrained Swin encoder + Mamba. Dataset: abdominal, endoscopy, microscopy.
- **Gap:** Pretraining dependency; mostly 2D; heavier than pure-Mamba.

### 27. Mamba Goes HoME — Hierarchical Soft MoE 3D Seg
- **2025 · arXiv 2507.06363.** Hierarchical soft mixture-of-experts over Mamba. Dataset: multi-organ 3D.
- **Gap:** MoE routing adds params/complexity; efficiency vs capacity trade unclear.

### 28. Mobile U-ViT — Large-Kernel U-shaped ViT
- **2025 · arXiv 2508.01064.** Mobile-grade large-kernel ViT for efficient seg. Dataset: 2D/3D medical.
- **Gap:** Mobile/2D emphasis; limited 3D volumetric validation.

### 29. 3D-EffiViTCaps — Efficient ViT + Capsules
- **2024 · arXiv 2403.16350.** ViT + capsule routing for 3D seg. Dataset: iSeg, Hippocampus, Cardiac.
- **Gap:** Capsule routing costly; scaling to large volumes hard.

### 30. FastSAM3D — Efficient Interactive 3D SAM
- **2024 · MICCAI 2024.** Distill 12-layer ViT-B → 6-layer ViT-Tiny; fast interactive 3D. Dataset: multi-organ prompt seg.
- **Gap:** Interactive/prompt-based (needs clicks); not fully automatic; SAM-derived.

### 31. SAM-Med3D — General Volumetric SAM
- **2024 · ECCV 2024 Workshops.** Native 3D SAM, 247 categories, few-point prompts. Dataset: huge SA-Med3D corpus.
- **Gap:** Large model; prompt-dependent; automatic-mode accuracy lower.

### 32. VISTA3D — Unified 3D Segmentation Foundation Model
- **2024-25 · arXiv / NVIDIA.** Native-3D, 127 classes, interactive correction. Dataset: large multi-org.
- **Gap:** Foundation-scale (heavy); efficiency not the goal; overkill for single-organ.

### 33. ES-UNet — Enhanced Skip-Connection 3D UNet
- **2025 · Q1 journal.** Enhanced skips; **178.5 ms inference, 9.01M params.** Dataset: 3D organ CT.
- **Gap:** Incremental UNet tweak; no long-range/frequency novelty.

### 34. ReCo-KD — Region/Context-Aware Distillation
- **2025-26 · arXiv 2601.08301.** KD transferring region+context to lightweight student. Dataset: 3D CT.
- **Gap:** Needs a heavy teacher; student capacity caps rare-class gains.

### 35. RMIS-Net — MLP-Based Fast Seg
- **2025 · Q1 (PMC).** Multilayer-perceptron fast seg. Dataset: 2D/3D medical.
- **Gap:** MLP weak on fine spatial detail; mostly 2D.

---

## Tier S — Efficient methods strong on SMALL / hard-to-see structures
> Directly relevant: cyst, small tumor, low-contrast lesion. Our core battle.

### S1. MPSTrans — Mixed Parallel Shunted Transformer (small-tumor focus)
- **2024-25 · journal (PMC12381746).** 3D-MPST blocks in U-form; parallel multiscale Transformer–CNN aggregation. **Beats peers on small tumor areas with −56.4% FLOPs (BCV), −71.1% (MSD), −69.3% (ACDC), −56.7% (colon).**
- **Dataset:** BCV, MSD, ACDC, colon-cancer CT.
- **Gap:** Still transformer-heavy attention; no frequency path; rare-class handled by scale, not by explicit detail branch.

### S2. HFF-Net — Harmonized Frequency Fusion (high-freq detail)
- **2025 · IEEE (PubMed 40504718).** Frequency-Domain Decomposition (low-freq split) + **Adaptive Laplacian Conv** with dynamic kernels emphasizing critical high-freq details. **+4.48% mean Dice** on tumor subregions.
- **Dataset:** BraTS (brain tumor subregions).
- **Gap:** Brain-MRI only; adaptive-Laplacian, not learnable-wavelet; not efficiency-headlined → room to make it light + CT-tumor.

### S3. Detail-Aware Net w/ Multi-Freq Directional Filtering + Lifting Wavelets
- **2025 · Eng. Appl. AI (S1568494625012827).** Multi-frequency directional filtering + **lifting wavelets** for low/high-freq detail separation on tumor MRI.
- **Dataset:** brain tumor MRI.
- **Gap:** **Lifting-wavelet for tumor detail already exists (2D-leaning, brain).** Confirms our learnable-lifting idea is plausible BUT must move to *3D + CT + rare-class + efficiency* to be novel.

### S4. TVPNet — Prompt-Guided Small 3D Object Seg
- **2025 · Biomed. Sig. Proc. Control (Q1).** Hybrid image encoder + text-vision prompt to focus on small organ targets.
- **Dataset:** small-organ CT.
- **Gap:** Prompt-dependent (uses text-vision = edges toward VLM you want to avoid); not fully automatic.

### S5. Submanifold Sparse ConvNet — Kidney + Tumor 3D Seg
- **2025 · arXiv 2511.04334.** Submanifold sparse convolutions → compute only on occupied voxels; efficient volumetric kidney+tumor.
- **Dataset:** KiTS-style CT (kidney + tumor).
- **Gap:** Sparse-conv efficiency, but no long-range/frequency modeling; small-tumor Dice still the hard part.

### S6. Lightweight Multiscale Attention Net — 3D PET Tumor
- **2025 · Sci. Reports (Q1, s41598-025-25092-3).** Lightweight multiscale attention for small tumor in low-res noisy PET.
- **Dataset:** PET tumor.
- **Gap:** PET-specific; attention-only; modest 3D validation.

### S7. N-Shaped Lightweight Net (FPN + Hybrid Attention)
- **2024 · Sensors/PMC10888052.** Multi-pyramid + depthwise-separable conv + hybrid attention; lightweight brain tumor.
- **Dataset:** BraTS 2019/2021, UCSF-PDGM, MSD Task01.
- **Gap:** 2D-pyramid heritage; no frequency/SSM; small-region gain from attention placement only.

### S8. Coarse-to-Fine Cascade for Small Pancreatic Tumor
- **2024 · PMC11335681.** 64³ coarse → 32³/16³ fine cascade to recover missed small low-contrast pancreatic tumors.
- **Dataset:** pancreatic CT.
- **Gap:** Multi-pass cascade = slow inference; efficiency sacrificed for recall; engineering, not new primitive.

### S9. Auto3DSeg / nnU-Net KiTS23 winners (reference bar)
- **2023-24 · MICCAI KiTS23.** Top solution: avg Dice 0.835, surface Dice 0.723; per-class **kidney ≈0.93, tumor ≈0.57, cyst ≈0.73.**
- **Gap (the whole point):** **tumor 0.57 and cyst 0.73 are the open problem.** Heavy ensemble, slow. Our target = match/beat these on tumor+cyst at a fraction of compute.

---

## Cross-cutting gaps WE can target (synthesis)

1. **Rare / small-structure (tumor) robustness under a tiny budget.** Every
   lightweight model (LightM-UNet, UltraLight, Slim) admits accuracy drop on
   small/rare classes. KiTS *tumor* Dice is the universal weak spot → strong,
   measurable contribution.
2. **Boundary / high-frequency fidelity with linear cost.** Mamba nets are cheap
   but blur boundaries; report surface-Dice/HD95 where rivals only show volume Dice.
3. **Learnable / adaptive frequency basis.** WaveFormer = fixed Haar; EM-Net =
   channel-wise freq. A *learnable lifting-scheme* spatial wavelet is open.
4. **Encoder-side efficiency.** EffiDec3D only fixed the decoder — an efficient
   encoder + their decoder logic is uncombined.
5. **Scan-order-invariant linear modeling.** SegMamba's directional-scan bias is
   unsolved; a permutation-robust token mixer would be novel.

## Reality check on our earlier idea
"Wavelet + Mamba" is **partly taken** (Topology-Aware Wavelet Mamba #22, HybridMamba #8
dual-domain, EM-Net #7 freq+Mamba). To stand out we must add a *distinct* twist:
- **Learnable lifting-wavelet** sub-band split (not fixed Haar), AND
- **Rare-class-aware high-freq branch** (boundary/tumor focus), AND
- single-branch (not dual-branch like HybridMamba) to keep it genuinely light.

## Sources
- SegMamba: https://arxiv.org/abs/2401.13560 · MICCAI https://dl.acm.org/doi/abs/10.1007/978-3-031-72111-3_54
- WaveFormer: https://arxiv.org/abs/2503.23764
- LHU-Net: https://arxiv.org/abs/2404.05102 · code https://github.com/xmindflow/lhunet
- EffiDec3D: https://github.com/SLDGroup/EffiDec3D
- EM-Net: https://arxiv.org/pdf/2409.17675
- HybridMamba: https://arxiv.org/pdf/2509.14609 · https://papers.miccai.org/miccai-2025/0426-Paper2815.html
- VeloxSeg (JL-lemma): https://arxiv.org/pdf/2509.22307
- SegMamba-V2: https://ieeexplore.ieee.org/document/11084842/
- MedNeXt: https://link.springer.com/chapter/10.1007/978-3-031-43901-8_39
- 3D UX-Net: https://arxiv.org/abs/2209.15076
- LightM-UNet: https://arxiv.org/pdf/2403.05246
- UltraLight VM-UNet: https://github.com/wurenkai/UltraLight-VM-UNet
- Slim UNETR++: https://link.springer.com/article/10.1007/s11517-025-03390-2
- SegFormer3D: (CVPRW 2024)
- EfficientMedNeXt: https://link.springer.com/chapter/10.1007/978-3-032-04965-0_19
- D-Net: https://arxiv.org/pdf/2403.10674
- Tri-Plane Mamba / FastSAM3D / SAM-Med3D: MICCAI/ECCV 2024
- HER-Seg: https://arxiv.org/pdf/2504.06205
- Topology-Aware Wavelet Mamba: https://arxiv.org/pdf/2502.14363
- WMREN: https://www.ijcai.org/proceedings/2025/0187.pdf
- EfficientViM: CVPR 2025
- Mamba HoME: https://arxiv.org/pdf/2507.06363
- Mobile U-ViT: https://arxiv.org/pdf/2508.01064
- 3D-EffiViTCaps: https://arxiv.org/pdf/2403.16350
- VISTA3D: https://www.researchgate.net/publication/394660295
- ES-UNet: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12403535/
- ReCo-KD: https://arxiv.org/pdf/2601.08301
- RMIS-Net: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12192725/
- Awesome-Mamba (survey): https://github.com/xmindflow/Awesome_Mamba
- MPSTrans (parallel multiscale Transformer-CNN): https://pmc.ncbi.nlm.nih.gov/articles/PMC12381746/
- HFF-Net (freq-domain brain tumor): https://pubmed.ncbi.nlm.nih.gov/40504718/
- Detail-Aware lifting-wavelet tumor: https://www.sciencedirect.com/science/article/abs/pii/S1568494625012827
- TVPNet (small 3D object): https://www.sciencedirect.com/science/article/abs/pii/S1746809425007505
- Submanifold Sparse ConvNet (kidney+tumor): https://arxiv.org/pdf/2511.04334
- Lightweight multiscale attn (PET tumor): https://www.nature.com/articles/s41598-025-25092-3
- N-Shaped lightweight net: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10888052/
- Cascade small pancreatic tumor: https://pmc.ncbi.nlm.nih.gov/articles/PMC11335681/
- KiTS23 winner solution: https://arxiv.org/pdf/2310.04110
