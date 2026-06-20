# Efficient 3D Segmentation Model Repository Compendium

Generated from the local checkout at `efficient/models` on 2026-06-20. GitHub URLs come from each folder's local `origin` remote and were spot-verified against GitHub on 2026-06-20.

This document is a source-grounded, end-to-end map of every repository currently present under `efficient/models`. It focuses on the models each repository creates or exposes, the major architecture components, the critical functions/classes, and how training/inference are wired. Some folders contain full third-party frameworks or comparison baselines; those are documented as dependencies/baselines rather than being treated as original contributions of that repo.

## Reading Scope

- Root folders covered: `3D-EffiViTCaps`, `3DUX-Net`, `EfficientMedNeXt`, `EfficientViM`, `EffiDec3D`, `EM-Net`, `FastSAM3D`, `HER-Seg`, `LHUNet`, `LightM-UNet`, `MedNeXt`, `Mobile-U-ViT`, `MONAI`, `nnMamba`, `Pancreatic-Tumor-SEG`, `SAM-Med3D`, `SegFormer3D`, `SegMamba`, `Slim-UNETR`, `STU-Net`, `Swin-UMamba`, `TP-Mamba`, `UltraLight-VM-UNet`, `VeloxSeg`, `VISTA`, `WaveFormer`, and `WMREN`.
- Primary source types read: root READMEs, custom model files, model factory files, training entrypoints, decoders, attention/state-space blocks, wavelet modules, prompt/SAM modules, and loss or evaluation hooks where they affect the model algorithm.
- Dependency handling: copied `nnU-Net`, `MONAI`, `mamba`, `causal-conv1d`, and other framework code is noted at the system level. I only expand those internals when the repo's own model directly depends on them or modifies them.
- Important caveat: `STU-Net` in this checkout is README/assets only, and `WMREN` contains only a minimal README. Their architectures are summarized from the local README/files, but there is no local implementation to audit for those two.

## Repository Matrix

| Repo | GitHub | Main model or role | Core family | Primary local source |
|---|---|---|---|---|
| `3D-EffiViTCaps` | [HidNeuron/3D-EffiViTCaps](https://github.com/HidNeuron/3D-EffiViTCaps) | 3D EfficientViT plus capsule segmentation | Efficient ViT + capsules | `module/effiViTcaps.py`, `main_block/efficientViT3D.py`, `main_block/capsule_layers.py` |
| `3DUX-Net` | [MASILab/3DUX-Net](https://github.com/MASILab/3DUX-Net) | 3D UX-Net | large-kernel ConvNet U-Net | `networks/UXNet_3D/network_backbone.py`, `networks/UXNet_3D/uxnet_encoder.py` |
| `EfficientMedNeXt` | [SLDGroup/EfficientMedNeXt](https://github.com/SLDGroup/EfficientMedNeXt) | EfficientMedNeXt | multi-dilation MedNeXt | `networks/MedNeXt/mednextv1/EfficientMedNext.py`, `efficient_mednext_blocks.py` |
| `EfficientViM` | [mlvlab/EfficientViM](https://github.com/mlvlab/EfficientViM) | EfficientViM | vision Mamba/SSD | `classification/models/efficientvim.py` |
| `EffiDec3D` | [SLDGroup/EffiDec3D](https://github.com/SLDGroup/EffiDec3D) | Efficient decoder variants for 3D UX-Net, SwinUNETR, MedNeXt | decoder optimization | `networks/UXNet_3D/network_backbone.py`, `networks/swin_unetr_effidec3d.py`, `create_mednextv1_effidec3d.py` |
| `EM-Net` | [zang0902/EM-Net](https://github.com/zang0902/EM-Net) | EM-Net | Mamba + frequency/spectral learning | `models/em_net_model.py` |
| `FastSAM3D` | [arcadelab/FastSAM3D](https://github.com/arcadelab/FastSAM3D) | FastSAM3D | efficient 3D SAM distillation | `segment_anything/modeling/*3D*.py`, `distillation.py`, `train.py` |
| `HER-Seg` | [xq141839/HER-Seg](https://github.com/xq141839/HER-Seg) | HER-Seg | SAM/SAM2-based high-resolution segmentation | `model.py`, `SAM/*`, `sam2/*`, `unets/unet.py` |
| `LHUNet` | [xmindflow/LHUNet](https://github.com/xmindflow/LHUNet) | LHU-Net | lean hybrid U-Net | `src/lhunet/models/lhunet.py`, `src/lhunet/blocks/*.py` |
| `LightM-UNet` | [MrBlankness/LightM-UNet](https://github.com/MrBlankness/LightM-UNet) | LightM-UNet, U-Mamba variants | lightweight Mamba U-Net | `lightm-unet/nnunetv2/nets/LightMUNet.py`, `UMambaEnc.py`, `UMambaBot.py` |
| `MedNeXt` | [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) | MedNeXt | ConvNeXt-style medical U-Net | `nnunet_mednext/network_architecture/mednextv1/*` |
| `Mobile-U-ViT` | [FengheTan9/Mobile-U-ViT](https://github.com/FengheTan9/Mobile-U-ViT) | Mobile U-ViT | mobile large-kernel U-shaped ViT | `network/MobileUViT.py`, `network/MobileUViT_3D.py` |
| `MONAI` | [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI) | MONAI framework | medical AI framework/model zoo | `monai/networks`, `monai/transforms`, `monai/losses`, README |
| `nnMamba` | [lhaof/nnMamba](https://github.com/lhaof/nnMamba) | nnMamba segmentation/classification | residual CNN + Mamba SSM | `nnMamba.py`, `nnMamba4cls.py`, `classification/networks/ssm_nnMamba.py` |
| `Pancreatic-Tumor-SEG` | [HeyJGJu/Pancreatic-Tumor-SEG](https://github.com/HeyJGJu/Pancreatic-Tumor-SEG) | PFNet variants | positioning-focus cascade | `PFNet.py`, `PFNet_stan.py`, `PFNet_unet.py` |
| `SAM-Med3D` | [uni-medical/SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) | SAM-Med3D | 3D adaptation of SAM | `segment_anything/modeling/image_encoder3D.py`, `mask_decoder3D.py`, `prompt_encoder3D.py` |
| `SegFormer3D` | [OSUPCVLab/SegFormer3D](https://github.com/OSUPCVLab/SegFormer3D) | SegFormer3D | 3D MixVision Transformer | `architectures/segformer3d.py` |
| `SegMamba` | [ge-xing/SegMamba](https://github.com/ge-xing/SegMamba) | SegMamba | 3D Mamba encoder-decoder | `model_segmamba/segmamba.py` |
| `Slim-UNETR` | [deepang-ai/Slim-UNETR](https://github.com/deepang-ai/Slim-UNETR) | Slim UNETR | compact UNETR with sparse transformer blocks | `src/SlimUNETR/SlimUNETR.py`, `Slim_UNETR_Block.py` |
| `STU-Net` | [openmedlab/STU-Net](https://github.com/openmedlab/STU-Net) | STU-Net | scalable nnU-Net pretraining | README/assets only in local checkout |
| `Swin-UMamba` | [openmedlab/Swin-UMamba](https://github.com/openmedlab/Swin-UMamba) | Swin-UMamba / SwinUMambaD | VMamba encoder + U-Net decoder | `swin_umamba/nnunetv2/nets/SwinUMamba.py`, `SwinUMambaD.py` |
| `TP-Mamba` | [xmed-lab/TP-Mamba](https://github.com/xmed-lab/TP-Mamba) | SAM-Mamba | SAM adapter + Mamba-like prompt adaptation | `networks/sam/samunet_mamba.py` |
| `UltraLight-VM-UNet` | [wurenkai/UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet) | UltraLight VM-UNet | PVM Mamba U-Net | `models/UltraLight_VM_UNet.py` |
| `VeloxSeg` | [JinPLu/VeloxSeg](https://github.com/JinPLu/VeloxSeg) | VeloxSeg | paired-window attention + light conv encoder-decoder | `model/VeloxSeg.py`, `model/components/PWA.py` |
| `VISTA` | [Project-MONAI/VISTA](https://github.com/Project-MONAI/VISTA) | VISTA3D | universal 3D segmentation foundation model | `vista3d/vista3d/modeling/vista3d.py`, `point_head.py`, `class_head.py`, `segresnetds.py` |
| `WaveFormer` | [mahfuzalhasan/WaveFormer](https://github.com/mahfuzalhasan/WaveFormer) | WaveFormer | wavelet-driven 3D transformer U-Net | `network_models/network_backbone.py`, `waveformer.py`, `wave_helper.py`, `idwt_upsample.py` |
| `WMREN` | [C101812/WMREN](https://github.com/C101812/WMREN) | WMREN reference placeholder | README only | `README.md` |

## Cross-Repository Architectural Patterns

Most repositories implement one of six patterns:

1. Encoder-decoder 3D U-Net derivatives: `3DUX-Net`, `MedNeXt`, `EfficientMedNeXt`, `EffiDec3D`, `LHUNet`, `LightM-UNet`, `Mobile-U-ViT`, `Slim-UNETR`, `VeloxSeg`, and `WaveFormer`.
2. State-space/Mamba segmentation: `EM-Net`, `nnMamba`, `SegMamba`, `Swin-UMamba`, `LightM-UNet`, `UltraLight-VM-UNet`, and `TP-Mamba`.
3. SAM-derived promptable segmentation: `SAM-Med3D`, `FastSAM3D`, `HER-Seg`, `TP-Mamba`, and `VISTA`.
4. Efficient transformer encoders: `3D-EffiViTCaps`, `EfficientViM`, `SegFormer3D`, `Slim-UNETR`, `Mobile-U-ViT`, and `WaveFormer`.
5. Framework-scale training systems: `MONAI`, `MedNeXt`, `LightM-UNet`, `Swin-UMamba`, `SegMamba`, and `nnMamba` include or fork nnU-Net/MONAI-style preprocessing, planning, trainers, sliding-window inference, and losses.
6. README/checkpoint reference folders: `STU-Net` and `WMREN` are not complete source checkouts here.

Common training/inference algorithms appearing across repos:

- Sliding-window inference for 3D volumes, usually through MONAI or nnU-Net style helpers.
- Dice + cross entropy losses, often with deep supervision.
- Patch-based volume training with random crops, intensity normalization, foreground-aware sampling, and test-time overlap blending.
- Skip-connected multiscale decoders.
- Optional deep supervision heads at multiple decoder resolutions.
- Model factories that generate size variants by channel width, depth, kernel size, or checkpoint/pretraining options.

## 1. `3D-EffiViTCaps`

### Purpose

`3D-EffiViTCaps` implements "3D Efficient Vision Transformer with Capsule for Medical Image Segmentation." It combines efficient transformer blocks, patch merging/expansion, capsule routing, reconstruction regularization, and a Lightning training module.

### Architecture Flow

The central model is `EffiViTCaps3D` in `module/effiViTcaps.py`.

Input flow:

1. `feature_extractor` converts input volume to the first feature map.
2. `PatchMerging3D` reduces spatial resolution and raises channels.
3. `EfficientViTBlock3D` processes encoder stages with local-window/cascaded group attention.
4. Features are reshaped into capsule form.
5. `ConvSlimCapsule3D` applies 3D capsule convolution and dynamic routing.
6. Bottleneck features are flattened back to channels and passed through another `EfficientViTBlock3D`.
7. Decoder upsamples, concatenates skip features, applies convolution and EfficientViT blocks, then outputs class logits.
8. During training, a reconstruction branch predicts the masked input region and adds reconstruction loss.

### Important Source Components

- `EffiViTCaps3D`: full Lightning module. Important methods:
  - `forward`: inference path from feature extractor through patch merging, EfficientViT encoder blocks, capsule bottleneck, decoder, and final logits.
  - `training_step`: repeats the full forward path and computes segmentation/reconstruction losses.
  - `validation_step`, `validation_epoch_end`, `predict_step`: validation and sliding-window-style prediction hooks.
  - `losses`: combines capsule margin classification loss, segmentation loss, and reconstruction loss.
  - `_build_encoder`: creates `ConvSlimCapsule3D` layers.
  - `_build_decoder`: creates upsampling, skip-fusion, convolution, and output projection.
  - `_build_reconstruct_branch`: 1x1x1 conv reconstruction head.
  - `_build_3D_EfficientViTBlock`: creates encoder, decoder, and bottleneck EfficientViT blocks.

- `main_block/efficientViT3D.py`:
  - `Conv3d_BN`: conv + batch norm block with `fuse` helper for deployment fusion.
  - `Residual`: residual wrapper with optional stochastic drop behavior.
  - `FFN`: efficient feed-forward network.
  - `CascadedGroupAttention3D`: multi-head attention where channel groups cascade through depthwise/query-key-value projections.
  - `LocalWindowAttention3D`: partitions a 3D feature map into local windows when resolution is larger than the configured window.
  - `EfficientViTBlock3D`: combines depthwise residual conv, FFN, attention, and post-FFN.
  - `PatchMerging3D`: 3D downsampling by interleaving eight voxel subgrids and projecting channels.
  - `PatchExpand3D`: inverse-style channel/spatial expansion for decoding.

- `main_block/capsule_layers.py`:
  - `_squash`: capsule vector squashing nonlinearity.
  - `_update_routing`: dynamic routing loop.
  - `DepthwiseConv3d`, `DepthwiseConv4d`, `DepthwiseDeconv3d`, `DepthwiseDeconv4d`: capsule-specific depthwise convolution/deconvolution.
  - `ConvSlimCapsule2D`, `ConvSlimCapsule3D`, `DeconvSlimCapsule2D`, `DeconvSlimCapsule3D`: slim capsule layers.
  - `MarginLoss`: capsule margin loss.

### Data and Training

The repo has `datamodule` classes for iSeg-2017, Cardiac, Hippocampus, and LUNA16-like datasets. Training uses PyTorch Lightning, MONAI transforms/metrics, and scripts under `scripts/`. Evaluation writes masks and metrics from `evaluate.py` and `evaluate_iseg.py`.

## 2. `3DUX-Net`

### Purpose

`3DUX-Net` implements "3D UX-Net: A Large Kernel Volumetric ConvNet Modernizing Hierarchical Transformer for Medical Image Segmentation." It replaces transformer self-attention with a pure 3D ConvNeXt-like encoder using large kernels, then decodes with UNETR-style skip blocks.

### Architecture Flow

Main model: `UXNET` in `networks/UXNet_3D/network_backbone.py`.

1. `uxnet_conv` is the encoder.
2. The encoder uses a 3D stem and four stages.
3. Each stage uses `ux_block`: large-kernel depthwise convolution, `LayerNorm`, pointwise MLP-like linear projections, GELU, and layer-scale residual.
4. Encoder outputs are projected into decoder-ready feature maps.
5. Decoder uses `UnetrBasicBlock`, `UnetrUpBlock`, and `UnetOutBlock` from MONAI-style components.
6. Final logits are produced from the full-resolution decoder output.

### Important Source Components

- `uxnet_encoder.py`:
  - `LayerNorm`: channels-first/channels-last layer normalization.
  - `ux_block`: ConvNeXt-like block with depthwise convolution and residual layer scaling.
  - `uxnet_conv`: hierarchical encoder with downsampling stages and output indices.

- `network_backbone.py`:
  - `ProjectionHead`: optional projection head for representation output.
  - `UXNET`: segmentation model using `uxnet_conv`, MONAI UNETR decoder blocks, skip connections, and output head.
  - `proj_feat`: reshapes hidden features from token-like layout to channels-first volume layout.
  - `forward`: gets four encoder outputs, applies decoder ladder, outputs segmentation logits.

### Baselines Included

The repo also includes `nnFormer`, `TransBTS`, and related comparison networks. Those are useful for reproduction and benchmarking, but the repo's own contribution is the `UXNet_3D` implementation.

## 3. `EfficientMedNeXt`

### Purpose

`EfficientMedNeXt` implements a MICCAI 2025 model that modifies MedNeXt with multi-receptive dilated depthwise convolutions and a more efficient decoder configuration.

### Architecture Flow

The main custom implementation lives in `networks/MedNeXt/mednextv1/`.

1. `EfficientMedNeXt` and `EfficientMedNeXt_L` follow a U-shaped encoder-decoder.
2. The stem maps input channels to a base width.
3. Encoder stages use `EfficientMedNeXtBlock`.
4. Downsampling uses `EfficientMedNeXtDownBlock`.
5. Bottleneck and cross-resolution refinement blocks (`crrb*` in the source) process the deepest and skip-connected features.
6. Decoder stages use `EfficientMedNeXtUpBlock`.
7. `OutBlock` emits full-resolution and optional deep-supervision logits.

### Important Source Components

- `efficient_mednext_blocks.py`:
  - `MultiDilationDepthwiseConv3D`: parallel depthwise branches with kernel choices like 1, 3, 5 converted into 1x1 or 3x3 with dilation. Branch outputs are concatenated along channel dimension.
  - `EfficientMedNeXtBlock`: multi-dilation depthwise convolution, group/layer norm, GELU, optional global response normalization, 1x1 compression, and residual addition.
  - `EfficientMedNeXtDownBlock`: same block with downsampling strides and optional residual resample.
  - `EfficientMedNeXtUpBlock`: transposed-conv variant for decoder upsampling with padding correction.
  - `OutBlock`: 1x1 transposed conv output head.
  - `LayerNorm`: channels-first capable normalization.

- `EfficientMedNext.py`:
  - `EfficientMedNeXt`: configurable model with `kernel_sizes`, `strides`, `uniform_dec_channels`, `deep_supervision`, residual options, checkpointing, and mode.
  - `iterative_checkpoint`: layer-by-layer activation checkpointing.
  - `forward`: complete U-Net path with deep supervision and skip fusion.

- `EfficientMedNext_Full.py`:
  - `EfficientMedNeXt_L`: large/full variant with the same algorithmic pattern.

- `create_efficient_mednext.py`:
  - `create_efficient_mednext_tiny`, `small`, `medium`, `large`, and dispatch function.

### Distinction From Baseline MedNeXt

Baseline `MedNeXt` uses a single depthwise convolution kernel in `MedNeXtBlock`. `EfficientMedNeXt` uses multiple dilated depthwise branches per block to create multi-scale receptive fields without simply increasing dense convolution cost.

## 4. `EfficientViM`

### Purpose

`EfficientViM` implements "Efficient Vision Mamba with Hidden State Mixer-based State Space Duality." It is primarily an efficient 2D vision backbone for classification, detection, and segmentation transfer rather than a dedicated 3D medical segmentation model.

### Architecture Flow

Main file: `classification/models/efficientvim.py`.

1. Image is patch-embedded through convolutional stages.
2. Each stage stacks `EfficientViMBlock`.
3. The block uses `HSMSSD`, a hidden-state mixer based on state-space duality.
4. Local convolution and feed-forward layers complement the state-space operation.
5. The final classifier can optionally use a distillation head.

### Important Source Components

- `HSMSSD`: hidden state mixer / SSD block. It parameterizes a compact state-space recurrence and applies it over vision tokens.
- `EfficientViMBlock`: wraps hidden state mixer with MLP/normalization/residual structure.
- `EfficientViMStage`: downsampling stage plus repeated blocks.
- `EfficientViM`: full model with initialization, FLOP counting, and forward path.
- Factory functions: `EfficientViM_M1`, `M2`, `M3`, `M4`.

### Notes

The repo also contains `detection` and `segmentation` folders with MMDetection/MMSeg-style configs and tools. Those adapt the backbone to downstream dense prediction.

## 5. `EffiDec3D`

### Purpose

`EffiDec3D` is an optimized decoder strategy for 3D medical segmentation backbones. The idea is to reduce redundant high-resolution decoder computation by using a uniform or reduced decoder channel budget and starting decoding at a configurable resolution.

### Architecture Flow

The repo wires EffiDec3D into several backbones:

- `UXNET_EffiDec3D` for 3D UX-Net.
- `SwinUNETR_EffiDec3D` for SwinUNETR/SwinUNETRv2.
- `create_mednextv1_effidec3d` for MedNeXt.

The key idea:

1. Keep the strong encoder/backbone mostly intact.
2. Project selected skip features to a uniform decoder width, controlled by `n_decoder_channels`.
3. Use `resolution_factor` to decide how much of the full decoder ladder to instantiate. For example, larger factors skip more high-resolution decoder blocks.
4. Fuse skip features by concatenation or addition using `skip_aggregation`.
5. Output logits from the highest decoder resolution actually used.

### Important Source Components

- `networks/UXNet_3D/network_backbone.py`:
  - `ModifiedUnetrUpBlock`: transposed convolution, skip fusion by concatenation or addition, then `UnetResBlock`/`UnetBasicBlock`.
  - `UXNET_EffiDec3D`: decoder-optimized UX-Net with `n_decoder_channels`, `resolution_factor`, and `skip_aggregation`.
  - `forward`: conditionally builds and executes encoder/decoder stages depending on `resolution_factor`.

- `networks/swin_unetr_effidec3d.py`:
  - `SwinUNETR_EffiDec3D`: decoder-optimized SwinUNETR.
  - `ModifiedUnetrUpBlock`: same algorithmic decoder building block.
  - Swin internals: `WindowAttention`, `SwinTransformerBlock`, `PatchMerging`, `BasicLayer`, and `SwinTransformer`.

- `networks/MedNeXt/mednextv1/create_mednextv1_effidec3d.py`:
  - `create_mednextv1_effidec3d_small/base/medium/large`.
  - Dispatches decoder-optimized MedNeXt variants.

### Training

`main_train_BTCV_TU.py` and `main_train_MSD_Task01_10.py` choose the network by `--network`, configure `--n_decoder_channels`, `--resolution_factor`, `--skip_aggregation`, and run MONAI-style training/evaluation.

## 6. `EM-Net`

### Purpose

`EM-Net` implements "Efficient Channel and Frequency Learning with Mamba for 3D Medical Image Segmentation." It combines Mamba state-space layers with spectral/frequency parsing modules and a UNETR-like decoder.

### Architecture Flow

Main file: `models/em_net_model.py`.

1. Input volume enters `MambaEncoder`, `MambaEncoder_Bot`, or `MambaEncoder_Conv_Enc` depending on variant.
2. Each encoder stage mixes local convolutions and `MambaLayer`.
3. Frequency features are processed by `FFParser_n` and `Spectral_Layer`.
4. Decoder uses `EMNetUpBlock`, which upsamples, fuses skip features, and optionally applies Mamba/frequency-aware processing.
5. Main variants `EMNet`, `EMNet_Bot`, and `EMNet_Conv_Enc` differ in where Mamba/frequency blocks are placed.

### Important Source Components

- `LayerNorm`: channels-first layer normalization.
- `MambaLayer`: flattens spatial dimensions to token sequence, applies Mamba SSM, then restores volume shape.
- `MlpChannel`: channel MLP.
- `FFParser_n`: frequency parser.
- `Spectral_Layer`: spectral-domain processing block.
- `EMNetUpBlock`: upsampling block combining decoder input and skip features.
- `EMNet`: full encoder-decoder.
- `MambaEncoder`: hierarchical Mamba encoder.
- `EMNet_Bot`, `MambaEncoder_Bot`: bottleneck-focused variant.
- `EMNet_Conv_Enc`, `MambaEncoder_Conv_Enc`, `MambaLayer_Conv`: convolution-enhanced encoder variant.

### Training/Evaluation

`main.py`, `trainer.py`, `test.py`, `cal_metrics.py`, and configs under `configs/` provide training, validation, and parameter/FLOP calculation. The repo vendors `mamba` and `causal-conv1d` as dependencies.

## 7. `FastSAM3D`

### Purpose

`FastSAM3D` is a faster 3D Segment Anything style model. It keeps the SAM prompt/decoder design but adapts encoders and attention for volumetric input and uses distillation/student training.

### Architecture Flow

1. 3D input is patch-embedded by `ImageEncoderViT3D`.
2. Encoder blocks use either standard 3D window attention or dilated attention.
3. Prompt embeddings are produced by a 3D prompt encoder.
4. `MaskDecoder3D` uses `TwoWayTransformer3D` to exchange information between image tokens and prompt tokens.
5. Mask tokens are passed through hypernetwork MLPs to generate mask logits.
6. Training scripts support SAM-style training, student validation, and distillation.

### Important Source Components

- `segment_anything/modeling/image_encoder3D.py`:
  - `ImageEncoderViT3D`, `Block3D`, `Attention`, `PatchEmbed3D`.
  - `window_partition3D`, `window_unpartition3D`, `get_rel_pos`, `add_decomposed_rel_pos`.

- `segment_anything/modeling/image_encoder3D_dilatedattention.py`:
  - Dilated attention variant of `Block3D` and `Attention`.

- `segment_anything/modeling/mask_decoder3D.py`:
  - `TwoWayTransformer3D`, `TwoWayAttentionBlock3D`, `Attention`, `MaskDecoder3D`, `MLP`.

- Training files:
  - `distillation.py`: teacher/student distillation path.
  - `train.py`, `train_unfreeze.py`: main training paths.
  - `validation.py`, `validation_student.py`, `val_2d.py`: evaluation variants.

## 8. `HER-Seg`

### Purpose

`HER-Seg` targets high-resolution medical image segmentation with SAM/SAM2-derived prompting, prompt-free distillation, high-resolution feature handling, and U-Net fallback components.

### Architecture Flow

Main file: `model.py`.

1. `MHRMedSeg` uses a SAM image encoder, prompt encoder, and mask decoder.
2. `PointGenerator` derives prompt points from masks when ground truth is available.
3. `PFD` and `PFDTeacher` implement prompt-free distillation between a student and a SAM-style teacher.
4. `MHRMedSegSAM2` wraps SAM2-style image encoding, prompt preparation, mask prediction, and high-resolution feature fusion.
5. `FpnNeck` merges hierarchical backbone features.

### Important Source Components

- `PointGenerator`: converts masks into positive/negative prompt points.
- `MHRMedSeg`: SAM-style segmentation model with `ImageEncoderViT`, `PromptEncoder`, `MaskDecoder`, and `TwoWayTransformer`.
- `PFD`: prompt-free distillation student.
- `PFDTeacher`: teacher encoder for distillation.
- `MHRMedSegSAM2`: SAM2 based predictor with `_prep_prompts`, `_predict`, and `forward`.
- `FpnNeck`: top-down feature pyramid neck.
- `SAM/*`: local SAM implementation, including `image_encoder.py`, `ecm_decoder.py`, `prompt_encoder.py`, `transformer.py`, and `LGViT.py`.
- `sam2/*`: local SAM2 implementation with memory attention, memory encoder, HieraDet image encoder, SAM mask decoder, and prompt encoder.
- `loss.py`: Dice and segmentation losses.
- `unets/unet.py`: compact U-Net helper.

## 9. `LHUNet`

### Purpose

`LHUNet` implements "Lean Hybrid U-Net" for cost-efficient high-performance volumetric segmentation. It mixes CNN blocks and lightweight transformer-style hybrid blocks in the encoder and decoder.

### Architecture Flow

Main model: `src/lhunet/models/lhunet.py`.

1. A CNN encoder extracts early local features.
2. A hybrid encoder combines convolutional and transformer/attention blocks.
3. A hybrid decoder upsamples deep features with skip fusion.
4. A CNN decoder restores high-resolution details.
5. Output blocks produce final segmentation logits, optionally with deep supervision.

### Important Source Components

- `LHUNet`: top-level configurable model with many parameters controlling CNN kernels, hybrid feature widths, transformer projection sizes, attention heads, dropout, residual modes, skip modes, and deep supervision.
- `blocks/base.py`:
  - `BaseBlock`, `get_conv_layer`, `LayerNorm`, padding helpers.
- `blocks/cnn.py`:
  - `DCNNBlock`, `UnetResBlock`, `UnetBasicBlock`, `UnetUpBlock`, `UnetOutBlock`.
  - `CNNEncoder`, `CNNDecoder`.
- `blocks/hyb.py`:
  - `HybridEncoder`, `HybridDecoder`, `BaseHybridBlock.combine`.
- `dcn/`: deformable convolution implementation.
- `metrics/`: evaluation metrics.

## 10. `LightM-UNet`

### Purpose

`LightM-UNet` adapts Mamba/state-space layers into a lightweight nnU-Net v2 style segmentation framework.

### Architecture Flow

Key files live in `lightm-unet/nnunetv2/nets/`.

- `LightMUNet.py`: standalone lightweight Mamba U-Net.
- `UMambaEnc.py`: Mamba layers inserted throughout the encoder.
- `UMambaBot.py`: Mamba layer placed at bottleneck.

### Important Source Components

- `LightMUNet.py`:
  - `MambaLayer`: SSM layer for flattened spatial tokens.
  - `ResMambaBlock`: residual block around Mamba.
  - `ResUpBlock`: decoder upsampling block.
  - `LightMUNet`: full encode/decode path with `_make_down_layers`, `_make_up_layers`, `encode`, `decode`, and `forward`.

- `UMambaEnc.py`:
  - `ResidualMambaEncoder`: residual encoder with Mamba layers.
  - `UNetResDecoder`: decoder with deep supervision.
  - `UMambaEnc`: full segmentation network.
  - `get_umamba_enc_from_plans`: nnU-Net plans integration.

- `UMambaBot.py`:
  - `UMambaBot`: encoder-decoder where the bottleneck is Mamba-enhanced.
  - `get_umamba_bot_from_plans`: nnU-Net plans integration.

### Framework Integration

The repo contains a full `nnunetv2` tree for preprocessing, planning, inference, trainers, and utilities. The custom algorithmic parts are the Mamba nets above.

## 11. `MedNeXt`

### Purpose

`MedNeXt` is a ConvNeXt-inspired medical segmentation architecture integrated into an nnU-Net framework. It uses large depthwise kernels, inverted bottlenecks, residual connections, and optional deep supervision.

### Architecture Flow

Main model: `nnunet_mednext/network_architecture/mednextv1/MedNextV1.py`.

1. `stem`: 1x1 conv from input channels to base channels.
2. Four encoder levels: `enc_block_0..3`, each made of repeated `MedNeXtBlock`.
3. Four downsample blocks: `MedNeXtDownBlock`.
4. Bottleneck: repeated `MedNeXtBlock`.
5. Four upsample stages: `MedNeXtUpBlock`.
6. Decoder blocks use skip addition (`x_res_i + x_up_i`), not concatenation.
7. `OutBlock` creates logits; optional `out_1..out_4` for deep supervision.

### Important Source Components

- `MedNeXtBlock`: depthwise convolution, group/layer norm, 1x1 expansion, GELU, 1x1 compression, optional residual, optional global response normalization.
- `MedNeXtDownBlock`: stride-2 depthwise downsampling with optional residual projection.
- `MedNeXtUpBlock`: transposed-conv upsampling with asymmetric padding correction.
- `OutBlock`: 1x1 output projection.
- `LayerNorm`: channels-first capable layer norm.
- `iterative_checkpoint`: activation checkpointing workaround.

### Framework

The repo includes `nnunet_mednext`, an nnU-Net v1 style framework fork: dataset conversion, experiment planning, preprocessing, inference, trainers, losses, optimizers, and model selection.

## 12. `Mobile-U-ViT`

### Purpose

`Mobile-U-ViT` implements a lightweight U-shaped vision transformer with local/global blocks, including both 2D and 3D variants.

### Architecture Flow

Files:

- `network/MobileUViT.py`: 2D implementation.
- `network/MobileUViT_3D.py`: 3D implementation.

The core idea is to combine:

1. Mobile convolutional embedding (`ConvUtr`, `Embeddings`).
2. Local aggregation (`LocalAgg`) using depthwise/pointwise operations.
3. Global sparse attention (`GlobalSparseAttn`) with spatial reduction.
4. `LGLBlock` / `LKLGLBlock`: local-global-local processing block.
5. U-Net decoder with upsampling and convolution blocks.

### Important Source Components

- `Residual`: residual wrapper.
- `ConvUtr`: repeated convolutional unit.
- `Embeddings`: stage embedding.
- `Mlp`, `CMlp`: token/channel MLPs.
- `GlobalSparseAttn`: sparse attention over reduced spatial tokens.
- `LocalAgg`: local convolutional aggregation.
- `SelfAttn`: attention + MLP block.
- `LGLBlock` / `LKLGLBlock`: core local/global composition.
- `MobileUViT`: full U-shaped segmentation model.
- Factories: `mobileuvit`, `mobileuvit_l`.

## 13. `MONAI`

### Purpose

`MONAI` is not a single model repo; it is the Medical Open Network for AI framework. It provides reusable transforms, losses, metrics, inferers, data abstractions, and model architectures used by several other repos in this folder.

### Relevant Model Families

Within `monai/networks/nets`, the framework includes architectures such as:

- `UNet`, `BasicUNet`, `BasicUNetPlusPlus`
- `DynUNet`
- `SegResNet`, `SegResNetDS`
- `SwinUNETR`
- `UNETR`
- `VNet`
- `AttentionUnet`
- `FlexibleUNet`
- `HighResNet`
- `RegUNet`, `VoxelMorph`, and others

### Important Framework Algorithms

- Data transforms for intensity, spacing, cropping/padding, spatial augmentation, lazy transforms, and dictionary transforms.
- Inferers for sliding window and patch-based inference.
- Losses for Dice, generalized Dice, focal, Tversky, DiceCE, and deep supervision workflows.
- Metrics and postprocessing utilities.
- Bundle/model-zoo support.

### How It Is Used Here

Many repos import MONAI blocks such as `UnetrBasicBlock`, `UnetrUpBlock`, `UnetOutBlock`, `Convolution`, `UpSample`, and sliding-window inference helpers. MONAI is a dependency foundation, not an original model created by this `efficient/models` collection.

## 14. `nnMamba`

### Purpose

`nnMamba` uses Mamba state-space layers in a residual CNN architecture for 3D biomedical image segmentation, classification, and landmark-like tasks.

### Architecture Flow

Main segmentation file: `nnMamba.py`.

1. Initial convolutional stem.
2. Residual stages built by `make_res_layer`.
3. Selected residual stages inject `MambaLayer`.
4. Decoder uses `DoubleConv`, `SingleConv`, and attention-style channel/spatial recalibration.
5. Output is a segmentation map over `number_classes`.

### Important Source Components

- `conv3x3`, `conv1x1`: ResNet-style conv helpers.
- `BasicBlock`: residual block with optional downsample and optional Mamba layer.
- `make_res_layer`: stage builder.
- `MambaLayer`: flattens 3D volume into a token sequence, applies Mamba, reshapes back.
- `DoubleConv`, `SingleConv`: decoder/convolution helpers.
- `Attentionlayer`: lightweight attention/calibration layer.
- `nnMambaSeg`: full segmentation network.
- `nnMamba4cls.py`: classification encoder variant.
- `classification/networks/ssm_nnMamba.py`: classification-specific state-space model.

### Framework

The repo also includes a forked `nnunet` tree for preprocessing, training, experiment planning, and inference.

## 15. `Pancreatic-Tumor-SEG`

### Purpose

`Pancreatic-Tumor-SEG` implements PFNet-style pancreatic tumor segmentation, including standard, U-Net, and ResNet-backed variants.

### Architecture Flow

Main files: `PFNet.py`, `PFNet_stan.py`, `PFNet_unet.py`.

1. Backbone extracts multiscale features.
2. `Context_Exploration_Block` expands and refines context at each feature level.
3. `Positioning` predicts a coarse location map.
4. `Focus` refines boundaries/regions by combining lower and higher features with the coarse map.
5. Channel and spatial attention blocks calibrate features.
6. Output maps are upsampled to segmentation resolution.

### Important Source Components

- `CA_Block`: channel attention.
- `SA_Block`: spatial attention.
- `Context_Exploration_Block`: multi-branch context extraction.
- `Positioning`: coarse location prediction.
- `Focus`: guided refinement using feature maps and intermediate mask.
- `PFNet`: primary model.
- `PFNet_unet`: U-Net-backed variant.
- `PFNet_stan`: standard variant.

### Training/Inference

`train.py`, `train_res.py`, `infer.py`, `infer_pfunet.py`, `datasets.py`, `joint_transforms.py`, and `eval/` provide training and evaluation scaffolding.

## 16. `SAM-Med3D`

### Purpose

`SAM-Med3D` adapts Segment Anything to volumetric medical images by replacing 2D image/prompt/mask modules with 3D analogs.

### Architecture Flow

1. `ImageEncoderViT3D` embeds a 3D volume into patch tokens.
2. `Block3D` applies local/global 3D self-attention with decomposed relative position embeddings.
3. `PromptEncoder3D` embeds 3D points, boxes, and masks.
4. `MaskDecoder3D` uses `TwoWayTransformer3D` to exchange prompt and image information.
5. Mask tokens are transformed by hypernetwork MLPs into masks.
6. `Sam3D` wires encoder, prompt encoder, and decoder.

### Important Source Components

- `image_encoder3D.py`:
  - `ImageEncoderViT3D`, `Block3D`, `Attention`, `PatchEmbed3D`.
  - `window_partition3D`, `window_unpartition3D`.
  - `get_rel_pos`, `add_decomposed_rel_pos`.

- `prompt_encoder3D.py`:
  - `PromptEncoder3D`.
  - `_embed_points`, `_embed_boxes`, `_embed_masks`.
  - `PositionEmbeddingRandom3D`.

- `mask_decoder3D.py`:
  - `TwoWayTransformer3D`.
  - `TwoWayAttentionBlock3D`.
  - `Attention`.
  - `MaskDecoder3D`.
  - `predict_masks`.

- `build_sam3D.py`: model construction.
- `train.py`: training loop and prompt/mask handling.

## 17. `SegFormer3D`

### Purpose

`SegFormer3D` ports SegFormer to volumetric medical segmentation. It uses a hierarchical 3D MixVision Transformer encoder and lightweight MLP decoder head.

### Architecture Flow

Main file: `architectures/segformer3d.py`.

1. `PatchEmbedding` converts a 3D volume into hierarchical token maps.
2. `MixVisionTransformer` creates four feature stages.
3. Each stage contains `TransformerBlock` modules.
4. `SelfAttention` uses optional spatial reduction ratio (`sr_ratio`) to reduce keys/values.
5. `DWConv` inside MLP preserves local inductive bias.
6. `SegFormerDecoderHead` projects all four feature stages to a common embedding dimension, upsamples them, fuses channels, and predicts segmentation logits.

### Important Source Components

- `build_segformer3d_model`: config-based model factory.
- `SegFormer3D`: top-level model.
- `PatchEmbedding`: 3D overlapping patch projection.
- `SelfAttention`: multi-head attention with spatial reduction.
- `TransformerBlock`: norm, attention, residual, MLP.
- `MixVisionTransformer`: hierarchical encoder.
- `_MLP`, `MLP_`, `DWConv`: decoder and transformer MLP helpers.
- `SegFormerDecoderHead`: multiscale decoder.

## 18. `SegMamba`

### Purpose

`SegMamba` implements long-range sequential modeling with Mamba for 3D medical segmentation.

### Architecture Flow

Main file: `model_segmamba/segmamba.py`.

1. Input enters a convolutional stem and `GSC` local feature blocks.
2. `MambaEncoder` creates hierarchical features.
3. Each stage repeats `MambaLayer` according to configured `depths`.
4. `MambaLayer` flattens spatial dimensions and uses Mamba SSM with `num_slices` settings.
5. Decoder uses MONAI-style `UnetrUpBlock`/`UnetrBasicBlock` style structure.
6. `SegMamba.forward` returns full segmentation logits.

### Important Source Components

- `LayerNorm`: channels-first/channels-last normalization.
- `MambaLayer`: Mamba SSM token mixer.
- `MlpChannel`: channel MLP.
- `GSC`: gated spatial convolution/local feature block.
- `MambaEncoder`: hierarchical encoder.
- `SegMamba`: full segmentation network with projection and decoder.

### Framework

The repo includes `light_training` for data preprocessing, training, prediction, losses, metrics, and plan handling, plus vendored `mamba`, `causal-conv1d`, and `monai`.

## 19. `Slim-UNETR`

### Purpose

`Slim-UNETR` is a compact UNETR-style architecture designed to reduce transformer cost while preserving local and global context.

### Architecture Flow

Main files: `src/SlimUNETR/SlimUNETR.py` and `Slim_UNETR_Block.py`.

1. Input is embedded and passed through encoder stages.
2. Blocks combine local representation and global sparse transformer behavior.
3. Decoder reconstructs segmentation through skip-connected upsampling.
4. Output head emits segmentation classes.

### Important Source Components

- `SlimUNETR`: full model.
- `PatchPartition`: partitions features into local patches.
- `LineConv`: line-shaped convolutional local mixer.
- `LocalRepresentationsCongregation`: local aggregation block.
- `GlobalSparseTransformer`: sparse global attention block.
- `LocalReverseDiffusion`: local reverse diffusion/refinement block.
- `Block`: combined Slim-UNETR block.

### Training

`main.py`, `inference.py`, `weight_test.py`, `config.yml`, and `train.sh` provide configuration-driven training and inference.

## 20. `STU-Net`

### Purpose

The local `STU-Net` folder is a README/assets checkout describing "Scalable and Transferable U-Net." No implementation files are present locally.

### Architecture From Local README

- Built on nnU-Net v1.
- Defines scalable variants: `STU-Net-S`, `STU-Net-B`, `STU-Net-L`, and `STU-Net-H`.
- Parameter scale ranges from about 14.6M to 1.457B.
- Pretrained on TotalSegmentator with more than 100k annotations.
- Designed around joint scaling of width and depth.
- Supports direct inference and fine-tuning on downstream medical segmentation tasks.

### Local Source Status

Only `README.md`, `LICENSE`, and assets are present. The README references files such as `network_training`, `network_architecture`, and `run_finetuning.py`, but those files are not in this local checkout.

## 21. `Swin-UMamba`

### Purpose

`Swin-UMamba` combines a VMamba/VSSM-style state-space encoder with U-Net/nnU-Net decoding. It has encoder-only and decoder-inclusive variants.

### Architecture Flow

Main files:

- `swin_umamba/nnunetv2/nets/SwinUMamba.py`
- `swin_umamba/nnunetv2/nets/SwinUMambaD.py`

1. `PatchEmbed2D` creates 2D patch embeddings.
2. `VSSMEncoder` applies hierarchical VSS layers.
3. `SS2D` performs selective scan in multiple spatial directions.
4. `VSSBlock` wraps state-space mixing with normalization/residual.
5. Decoder uses MONAI/nnU-Net style up blocks and deep supervision.
6. `SwinUMambaD` adds explicit patch expansion and `UNetResDecoder`.

### Important Source Components

- `SS2D`: core 2D selective scan with `dt_init`, `A_log_init`, `D_init`, `forward_core`, and `forward`.
- `VSSBlock`: visual state-space block.
- `VSSLayer`: stacks VSS blocks and optional downsample.
- `VSSMEncoder`: full hierarchical encoder.
- `SwinUMamba`: encoder plus segmentation decoder.
- `SwinUMambaD`: VSS encoder plus residual U-Net decoder.
- `load_pretrained_ckpt`: pretrained VMamba weight loading.
- `get_swin_umamba_from_plans`, `get_swin_umamba_d_from_plans`: nnU-Net v2 integration.

## 22. `TP-Mamba`

### Purpose

`TP-Mamba` provides a SAM-based U-Net/Mamba adapter model. The local implementation is compact and centered on `networks/sam/samunet_mamba.py`.

### Architecture Flow

1. Start from a SAM encoder.
2. Add LoRA-style low-rank adapters to QKV projections.
3. Convert sequences back to spatial feature maps and forward through adapter blocks.
4. Use U-Net upsampling blocks to decode multiscale SAM features.
5. Output segmentation logits.

### Important Source Components

- `window_partition`, `window_unpartition`: local window reshaping helpers.
- Weight init helpers: normal, Xavier, Kaiming, orthogonal.
- `block_forward`: runs a SAM transformer block with adapters.
- `UnetConv3UP`, `UnetUpBlock`: decoder blocks.
- `_LoRA_qkv_timm`: LoRA wrapper for QKV projection.
- `adapter3`: spatial adapter with sequence/desequence helpers.
- `SAM_Mamba`: top-level model with `forward_encoder`, `forward_decoder`, and `forward`.

## 23. `UltraLight-VM-UNet`

### Purpose

`UltraLight-VM-UNet` is a parameter-efficient U-Net variant using parallel vision Mamba/PVM layers and optional channel/spatial attention bridges.

### Architecture Flow

Main file: `models/UltraLight_VM_UNet.py`.

1. Input passes through a small convolutional encoder.
2. `PVMLayer` splits/reshapes features and applies Mamba-style state-space mixing.
3. Multi-resolution features can be connected by channel and/or spatial attention bridges.
4. Decoder upsamples and fuses features.
5. Final output predicts the segmentation mask/classes.

### Important Source Components

- `PVMLayer`: core parallel vision Mamba layer.
- `Channel_Att_Bridge`: channel attention across five feature levels.
- `Spatial_Att_Bridge`: spatial attention across five feature levels.
- `SC_Att_Bridge`: combined spatial-channel bridge.
- `UltraLight_VM_UNet`: full architecture with `_init_weights` and `forward`.

### Training

`train.py`, `engine.py`, `loader.py`, `test.py`, and `utils.py` implement training loops, losses, metrics, and dataset loading.

## 24. `VeloxSeg`

### Purpose

`VeloxSeg` is an efficient segmentation model combining convolutional local encoding with paired-window transformer attention for both 2D and 3D settings.

### Architecture Flow

Main model: `model/VeloxSeg.py`.

1. `Encoder` combines a convolutional encoder and transformer encoder.
2. The convolution branch uses grouped/lightweight conv blocks.
3. The transformer branch uses paired-window attention to model local/global interactions.
4. `Seg_Decoder` or `RC_Decoder` reconstructs segmentation features.
5. `scale_prediction` handles deep-supervision output scaling.

### Important Source Components

- `VeloxSeg`: top-level network, `init_weights`, `scale_prediction`, `forward`.
- `Encoder.py`:
  - `Conv_Encoder`.
  - `Transformer_Encoder`.
  - `Encoder`.
- `Decoder.py`:
  - `RC_Decoder`.
  - `Seg_Decoder`.
- `components/PWA.py`:
  - `Paired_Windows_Attention`: gathers big/small windows, applies attention, scatters output back.
  - `MultiModal_Paired_Windows_Attention`.
  - `Paired_Windows_TransformerBlock`.
  - `Transformer_BasicLayer`.
  - `Cross_Channel_Attention`.
- `components/conv_blocks.py`:
  - `DownConv`, `UpConv`, `JLC`, `JLCLayer`.
- `components/attention_utils.py`:
  - `LayerNorm`, `FFN`, `PositionalEmbedding`, `PatchMerging`.
- `components/common_function.py`: helper functions for conv/norm/activation, channel splitting, gcd/lcm, input checks.

### Baselines

`compared_model/` contains comparison models such as MedNeXt, SlimUNETR, UNETR++, NestedFormer, and others. These are not VeloxSeg's own architecture.

## 25. `VISTA`

### Purpose

`VISTA` contains VISTA2D and VISTA3D foundation-model code. The VISTA3D portion supports class-prompted and point-prompted 3D segmentation for many anatomical labels.

### Architecture Flow

Main files under `vista3d/vista3d/modeling/`.

1. `SegResNetDS2` image encoder produces dense multiscale features and deep supervision outputs.
2. `Class_Mapping_Classify` maps class vectors to class-specific logits/features.
3. `Point_Mapping_SAM` maps point prompts into segmentation corrections using SAM-style two-way transformer blocks.
4. `VISTA3D2` combines automatic class logits and point-guided logits.
5. Postprocessing combines point and class predictions using connected components or Gaussian local fusion.

### Important Source Components

- `vista3d.py`:
  - `VISTA3D2`.
  - `precompute_embedding`, `clear_cache`.
  - `get_bs`.
  - `update_point_to_patch`.
  - `connected_components_combine`.
  - `gaussian_combine`.
  - `set_auto_grad`.
  - `forward`.

- `point_head.py`:
  - `Point_Mapping_SAM`: prompt-point head with SAM-style transformer.

- `class_head.py`:
  - `Class_Mapping_Classify`: maps class vectors/features to class-conditioned predictions.

- `segresnetds.py`:
  - `scales_for_resolution`, `aniso_kernel`.
  - `SegResBlock`.
  - `SegResEncoder`.
  - `SegResNetDS2`.

- `sam_blocks.py`:
  - `TwoWayTransformer`, `TwoWayAttentionBlock`, `PositionEmbeddingRandom`, `MLP`.

### Training/Inference

The repo includes configs for finetune, train, zero-shot evaluation, and supported evaluation. Scripts under `vista3d/scripts` manage validation and inference workflows.

## 26. `WaveFormer`

### Purpose

`WaveFormer` implements a wavelet-driven 3D transformer U-Net. It uses discrete wavelet transform features in the encoder and inverse wavelet reconstruction/refinement in the decoder.

### Architecture Flow

Main files: `network_models/network_backbone.py`, `waveformer.py`, `wave_helper.py`, `idwt_upsample.py`, `attention.py`.

1. `MultiscaleTransformer` embeds the input volume into hierarchical features.
2. `Block` can run single-scale attention or multi-scale wavelet attention.
3. `WaveletTransform3D` computes DWT coefficients via `ptwt.wavedec3`.
4. Encoder returns hidden states plus high-frequency wavelet coefficients.
5. `Waveformer` uses residual convolutional encoder/decoder support blocks.
6. Decoder uses `UnetrIDWTBlock` to combine decoder input, skip features, and high-frequency coefficients.
7. `HFRefinementRes` refines high-frequency coefficients before inverse DWT.
8. Output head predicts the segmentation mask/classes.

### Important Source Components

- `network_backbone.py`:
  - `ProjectionHead`.
  - `ChannelCalibration`: squeeze/excitation-style channel calibration.
  - `Waveformer`: full segmentation network.
  - `_init_waveformer_encoder`, `_init_residual_blocks`, `_init_decoder_blocks`.
  - `create_waveformer`.

- `waveformer.py`:
  - `MultiscaleTransformer`.
  - `forward_features`: returns multiscale transformer features and wavelet outputs.
  - `load_dualpath_model`, `init_weights`, `flops`.

- `wave_helper.py`:
  - `ProjectionUpsample`.
  - `DWConv`.
  - `PatchMergingV2`, `PatchMerging`.
  - `CCF_FFN`, `Mlp`.
  - `WaveletTransform3D`.
  - `Block`: multi-scale or single-scale attention block.
  - `OverlapPatchEmbed`, `PatchEmbed`, `PosCNN`.

- `idwt_upsample.py`:
  - `HFRefinementRes`.
  - `UnetrIDWTBlock`: refines high-frequency coefficients and reconstructs with `ptwt.waverec3`.

- `attention.py`:
  - `Attention`: multi-head attention with FLOP accounting.

### Framework

The repo includes `light_training`, copied MONAI-style utilities, self-supervised helpers, and an old `lib` framework. The custom WaveFormer architecture is in `network_models/`.

## 27. `WMREN`

### Purpose and Local Status

The local `WMREN` folder contains only a seven-byte `README.md` with the title `# WMREN`. There is no implementation, paper detail, architecture diagram, training script, or model file in this checkout.

### What Can Be Safely Said

Based on the folder name and the renamed paper in `efficient/papers`, this likely refers to Wavelet Multi-scale Region-Enhanced Network, but the local repo does not contain source. Do not treat this folder as an auditable implementation until the actual code is added.

## Expanded Implementation Notes

This section adds lower-level implementation details beyond the summary sections above. GitHub links are repeated here so each model can be read independently.

### 1. `3D-EffiViTCaps`

GitHub: [HidNeuron/3D-EffiViTCaps](https://github.com/HidNeuron/3D-EffiViTCaps)

- `EffiViTCaps3D` is a full PyTorch Lightning module, so the repo's model definition includes forward inference, optimizer setup, training loss computation, validation aggregation, prediction, and architecture construction in one class.
- The capsule branch represents features as capsule types and capsule atoms rather than plain channels. `DepthwiseConv4d` and `ConvSlimCapsule3D` operate on 3D capsule grids by generating capsule votes, applying routing logits, squashing vectors, and updating agreement between votes and output capsules.
- Dynamic routing is handled by `_update_routing`: it softmaxes routing logits over output capsules, computes weighted votes, adds bias, applies `_squash`, and iteratively updates logits through agreement. This is one of the main non-U-Net algorithms in the repo.
- `EfficientViTBlock3D` keeps the computation efficient by combining local depthwise residual convolution, channel FFN, and `LocalWindowAttention3D`. When the feature map is larger than the window, it pads and partitions the 3D tensor, applies attention per window, and restores the original shape.
- `CascadedGroupAttention3D` splits channels across attention heads/groups, processes group features with depthwise query/key/value projections, and cascades group outputs so later groups receive information from earlier groups.
- The reconstruction branch regularizes segmentation by reconstructing masked image content. The final loss can combine segmentation loss, capsule margin loss, and foreground/background masked reconstruction MSE through `rec_loss_weight` and `margin_loss_weight`.
- Validation and prediction use MONAI `sliding_window_inference`, which matters for full CT/MRI volumes that cannot fit into GPU memory as one tensor.

### 2. `3DUX-Net`

GitHub: [MASILab/3DUX-Net](https://github.com/MASILab/3DUX-Net)

- `uxnet_conv` is a four-stage hierarchical ConvNeXt-style 3D encoder. The stem performs the first downsample/projection, and later stages use norm plus stride-2 convolution to reduce resolution while increasing channels.
- `ux_block` is the core large-kernel block: it applies depthwise 3D convolution, switches to channels-last layout for `LayerNorm` and linear projections, expands channels with an MLP-like pointwise operation, applies GELU, compresses back, and adds a layer-scaled residual.
- Drop path values are assigned across all blocks using a progressive schedule, so deeper blocks receive stronger stochastic depth regularization.
- `UXNET.forward` produces four encoder scales, then combines them with shallow MONAI encoder blocks and UNETR-style up blocks. The decoder uses explicit full-resolution and intermediate-resolution skip features rather than only transformer hidden states.
- The repo includes `ProjectionHead` for representation/projection experiments, but segmentation output is produced by `UnetOutBlock`.
- Included `nnFormer` and `TransBTS` folders are comparison baselines. They are useful for reproduction, but the custom contribution is the UX-Net large-kernel ConvNet encoder with MONAI/UNETR decoding.

### 3. `EfficientMedNeXt`

GitHub: [SLDGroup/EfficientMedNeXt](https://github.com/SLDGroup/EfficientMedNeXt)

- `MultiDilationDepthwiseConv3D` is the key replacement for a single MedNeXt depthwise kernel. It builds multiple depthwise branches, uses dilation to simulate larger receptive fields efficiently, concatenates branch outputs, and lets later pointwise layers mix them.
- `EfficientMedNeXtBlock` keeps the ConvNeXt/MedNeXt inverted bottleneck idea: depthwise spatial mixing, normalization, channel expansion, GELU, optional global response normalization, channel compression, and residual addition.
- `EfficientMedNeXtDownBlock` and `EfficientMedNeXtUpBlock` reuse the same efficient block idea while changing resolution. The up block uses transposed convolution and includes shape/padding correction so skip additions align spatially.
- The implementation supports `kernel_sizes`, `strides`, `uniform_dec_channels`, `deep_supervision`, `do_res`, `do_res_up_down`, `checkpoint_style`, `block_counts`, `norm_type`, `dim`, and `grn`; the factories choose practical tiny/small/medium/large variants.
- Skip fusion follows the MedNeXt style of addition rather than concatenation. This lowers decoder channel growth and keeps memory pressure lower than a classic U-Net decoder.
- `iterative_checkpoint` activation-checkpoints sequential blocks one by one. This is important for 3D patches because the model is often memory-bound before it is compute-bound.

### 4. `EfficientViM`

GitHub: [mlvlab/EfficientViM](https://github.com/mlvlab/EfficientViM)

- This repo is primarily a 2D vision backbone, but it is relevant as an efficient Mamba/SSD architecture that can be transferred into detection or segmentation frameworks.
- `HSMSSD` projects input tokens into `B`, `C`, and `dt` state terms, applies a depthwise 2D convolution over those terms, builds a softmax-normalized state transition using learnable `A`, and computes hidden states with state-space duality style matrix products.
- `EfficientViMBlock` wraps the hidden-state mixer with two depthwise convolution residual paths and an FFN. A learnable four-part `alpha` gate controls how strongly each operation modifies the residual stream.
- `EfficientViMStage` stacks blocks and optionally downsamples through `PatchMerging`. Each stage returns the current feature, the pre-downsample feature, and the hidden state `h`.
- `EfficientViM.forward` uses multi-stage hidden-state fusion: hidden states from each stage and the final feature map are normalized, pooled, classified by separate heads, and combined by softmax-normalized learnable weights.
- Distillation mode adds parallel distillation heads and averages main/distillation predictions at evaluation time.

### 5. `EffiDec3D`

GitHub: [SLDGroup/EffiDec3D](https://github.com/SLDGroup/EffiDec3D)

- EffiDec3D is a decoder strategy, not one new encoder. It keeps strong encoders such as UX-Net, SwinUNETR, and MedNeXt, then reduces decoder computation by controlling width and which high-resolution stages are instantiated.
- `resolution_factor` gates both encoder output collection and decoder block creation. Lower values keep more decoder stages; higher values skip expensive high-resolution decoding. The UX-Net implementation checks that the value is no larger than 16.
- `n_decoder_channels` projects skip and decoder features to a smaller uniform width, preventing the decoder from inheriting the full encoder channel pyramid.
- `ModifiedUnetrUpBlock` supports `skip_aggregation='concatenation'` or `'addition'`. Concatenation is more expressive but grows channels; addition is cheaper and forces skip/decoder channels to match.
- The SwinUNETR variant keeps window attention, shifted windows, patch merging, optional v2 residual conv blocks, and pretrained weight loading, while replacing the decoder ladder with the efficient decoder configuration.
- The MedNeXt factory layer exposes EffiDec3D variants through `create_mednextv1_effidec3d_*`, making decoder efficiency a selectable mode rather than a separate training stack.

### 6. `EM-Net`

GitHub: [zang0902/EM-Net](https://github.com/zang0902/EM-Net)

- `MambaLayer` applies Mamba to flattened volumetric tokens and also applies a channel MLP path over the spatial volume. The two streams are normalized, combined with a small learnable `gamma`, reshaped back to 3D, then refined by a residual 3D convolution.
- Positional embeddings are stage-dependent. Their token count is computed from `in_shape` and the current downsample stage, so changing patch size or crop size requires matching configuration.
- `FFParser_n` performs a real FFT over the 3D spatial dimensions, multiplies frequency coefficients by a learnable complex weight, then returns to the spatial domain with inverse FFT.
- `Spectral_Layer` wraps the FFT parser with layer normalization and a channel MLP, adding frequency-filtered spatial features back to the input.
- `EMNetUpBlock` upsamples with a transposed convolution, adds the skip tensor, and then either uses a normal UNet block or a stack of Mamba layers, depending on `res_block`.
- The three main model variants move Mamba/frequency computation to different parts of the network: full encoder-decoder, bottleneck-focused, or convolution-enhanced encoder.

### 7. `FastSAM3D`

GitHub: [arcadelab/FastSAM3D](https://github.com/arcadelab/FastSAM3D)

- FastSAM3D adapts SAM-style image encoding, prompt encoding, and mask decoding to 3D volumes while emphasizing faster training/inference and student distillation.
- `ImageEncoderViT3D` uses 3D patch embedding and transformer blocks. The standard encoder supports local 3D window attention with decomposed relative position embeddings; a separate file implements dilated attention to enlarge context without full global attention cost.
- `window_partition3D` and `window_unpartition3D` pad depth, height, and width so arbitrary volume sizes can be partitioned into cubic windows and restored after attention.
- `PromptEncoder3D` creates sparse and dense prompt embeddings for volumetric points, boxes, and masks. This preserves SAM's separation between image features and user/task prompts.
- `MaskDecoder3D` uses `TwoWayTransformer3D` so prompt tokens attend to image tokens and image tokens attend back to prompts. Mask tokens feed hypernetwork MLPs that generate mask-specific projection weights.
- `distillation.py` is the key training algorithm beyond ordinary SAM fine-tuning: it trains a faster student against a teacher path, while `validation_student.py` evaluates the distilled model separately.

### 8. `HER-Seg`

GitHub: [xq141839/HER-Seg](https://github.com/xq141839/HER-Seg)

- HER-Seg is organized around high-resolution 2D medical segmentation rather than volumetric segmentation, but it is relevant because it explores efficient SAM/SAM2 adaptation.
- `PointGenerator` can derive prompt points from masks, letting the model train with prompt-like supervision even when prompts are not manually supplied.
- `MHRMedSeg` wires a SAM image encoder, prompt encoder, mask decoder, and two-way transformer into a medical segmentation model.
- `PFD` and `PFDTeacher` implement prompt-free distillation: the teacher can use stronger prompt/SAM features while the student learns to segment without depending on manual prompts at inference.
- `MHRMedSegSAM2` uses SAM2-style components and exposes prompt preparation and prediction routines such as `_prep_prompts` and `_predict`.
- `FpnNeck` aggregates hierarchical features in a top-down pathway, which is important for high-resolution inputs where shallow boundary detail and deep context must both survive.

### 9. `LHUNet`

GitHub: [xmindflow/LHUNet](https://github.com/xmindflow/LHUNet)

- `LHUNet` is highly configurable: it lets the caller control encoder/decoder widths, CNN depths, hybrid depths, attention heads, projection sizes, dropout, residual behavior, skip mode, and deep supervision.
- The architecture divides work between CNN blocks for local detail and hybrid blocks for broader context. Early and late stages remain convolution-heavy; middle stages use hybrid encoder/decoder processing.
- `blocks/cnn.py` provides standard U-Net components such as `UnetResBlock`, `UnetBasicBlock`, `UnetUpBlock`, `CNNEncoder`, and `CNNDecoder`.
- `blocks/hyb.py` defines `HybridEncoder`, `HybridDecoder`, and `BaseHybridBlock.combine`, which handle the transition between convolutional features and hybrid/attention-like representations.
- The `dcn/` folder adds deformable convolution support, giving the model an optional way to adapt receptive fields to organ shape variation.
- Deep supervision can produce auxiliary outputs at decoder scales, which is useful for small target structures and stabilizing 3D training.

### 10. `LightM-UNet`

GitHub: [MrBlankness/LightM-UNet](https://github.com/MrBlankness/LightM-UNet)

- `LightMUNet` is a MONAI-style standalone network that supports both 2D and 3D by passing `spatial_dims`.
- `MambaLayer` flattens all spatial positions into a token sequence, normalizes with `LayerNorm`, applies Mamba, adds a learnable scaled skip connection, projects to the requested output channels, and reshapes back.
- `ResMambaBlock` replaces the two convolution operations of a residual block with two Mamba layers. It keeps pre-activation normalization/activation and adds the original identity at the end.
- Decoder blocks (`ResUpBlock`) intentionally use depthwise separable convolution rather than Mamba, keeping upsampling cheaper while the encoder carries global sequence mixing.
- The nnU-Net v2 integrations (`UMambaEnc.py`, `UMambaBot.py`) expose Mamba either throughout the encoder or only at the bottleneck through `get_umamba_*_from_plans`, so the architecture can be selected inside nnU-Net's planner/trainer ecosystem.
- The repo carries a full `nnunetv2` tree, meaning preprocessing, plans, trainers, inference, and deep supervision can follow nnU-Net conventions.

### 11. `MedNeXt`

GitHub: [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt)

- MedNeXt adapts ConvNeXt ideas to 3D medical segmentation: large depthwise kernels provide spatial mixing, while 1x1 projections implement an inverted bottleneck channel MLP.
- `MedNeXtBlock` accepts `exp_r`, `kernel_size`, `norm_type`, `grn`, and residual flags. `exp_r` controls the hidden channel expansion and is one of the main capacity knobs.
- `MedNeXtDownBlock` and `MedNeXtUpBlock` are residual resampling blocks. The up block handles transposed-convolution padding asymmetry so feature shapes align for skip addition.
- Decoder skip fusion is additive. Compared with concatenation, this lowers channel count and encourages encoder/decoder features to share a common width at each level.
- Deep supervision heads (`out_1..out_4`) can be attached to decoder stages, and outputs are returned at multiple resolutions for training.
- The framework fork includes nnU-Net v1 style planning, preprocessing, prediction, postprocessing, and trainers; the architecture can also be imported into an external training pipeline.

### 12. `Mobile-U-ViT`

GitHub: [FengheTan9/Mobile-U-ViT](https://github.com/FengheTan9/Mobile-U-ViT)

- The 3D implementation starts with `Embeddings`, a compact convolutional stem plus `ConvUtr` layers. `ConvUtr` uses residual depthwise convolution and residual 1x1 expansion/compression.
- `LocalAgg` adds local positional/context modulation with large depthwise convolution, pointwise convolution, sigmoid gating, and a convolutional MLP.
- `GlobalSparseAttn` reduces the spatial token grid with average pooling when `sr_ratio > 1`, applies attention on the reduced sequence, then uses grouped transposed convolution (`LocalProp`) to propagate the attended features back to full resolution.
- `SelfAttn` combines depthwise positional embedding, sparse global attention, and MLP residual paths. It temporarily flattens `(D,H,W)` to tokens and then reshapes back.
- `LKLGLBlock` composes local aggregation and self-attention, giving the model a local-global-local flavor without full dense attention everywhere.
- The decoder is a U-Net style path using trilinear upsampling, convolution blocks, and skip concatenation from the embedding stages.

### 13. `MONAI`

GitHub: [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

- MONAI is the framework substrate for many repos here. It provides model blocks, inferers, transforms, data abstractions, losses, metrics, and bundle support rather than one specific paper model.
- Network families in the local tree include UNet variants, DynUNet, SegResNet/SegResNetDS, SwinUNETR, UNETR, VNet, AttentionUnet, FlexibleUNet, HighResNet, RegUNet, and VoxelMorph.
- `monai.inferers.sliding_window_inference` is a recurring algorithm across this collection. It crops overlapping windows, runs the model on each patch, blends outputs, and reconstructs a full-volume prediction.
- Loss utilities such as Dice, DiceCE, GeneralizedDice, Focal, Tversky, and deep supervision wrappers are reused by several repos.
- Transform utilities cover spacing/orientation normalization, intensity normalization, foreground sampling, random spatial crops, lazy transforms, and dictionary-based pipelines.
- For this compendium, MONAI should be treated as dependency infrastructure and a reference implementation library, not as a single competing model.

### 14. `nnMamba`

GitHub: [lhaof/nnMamba](https://github.com/lhaof/nnMamba)

- `nnMambaSeg` is a 3D residual encoder-decoder. It starts with a stride-2 double convolution, then uses three residual downsampling stages.
- `BasicBlock` is a ResNet-style 3D block. When a `MambaLayer` is supplied, the block adds global Mamba features into the convolutional residual branch before adding the identity.
- `MambaLayer` performs four directional/axis-flipped Mamba passes: original flattened tokens, length-flipped tokens, channel-flipped tokens, and both flipped. It reverses the flips and averages the four outputs, giving a simple bidirectional/multi-view SSM approximation.
- Skip features are channel-reweighted by `Attentionlayer`, which uses global average pooling followed by a two-layer MLP and sigmoid.
- The decoder upsamples with trilinear interpolation, concatenates reweighted encoder skips, and refines with `DoubleConv`.
- The repo also contains classification-oriented Mamba models and an nnU-Net fork for dataset planning/training/inference.

### 15. `Pancreatic-Tumor-SEG`

GitHub: [HeyJGJu/Pancreatic-Tumor-SEG](https://github.com/HeyJGJu/Pancreatic-Tumor-SEG)

- PFNet separates localization from refinement. `Positioning` first predicts a coarse target map, then `Focus` modules use that map to guide progressive feature refinement.
- `Context_Exploration_Block` uses multiple branches to broaden context around pancreas/tumor features before the focusing stages.
- `CA_Block` and `SA_Block` provide channel and spatial recalibration. They are used to emphasize informative channels/locations before final mask prediction.
- `PFNet`, `PFNet_stan`, and `PFNet_unet` share the same positioning/focusing concept but vary the backbone and U-Net integration.
- The inference scripts include separate paths for the PFNet and PF-UNet variants, so evaluation should match the selected architecture.
- This repo is more task-specific than most others in the folder: it targets pancreatic tumor segmentation, so the architecture is tuned around coarse localization plus boundary refinement.

### 16. `SAM-Med3D`

GitHub: [uni-medical/SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)

- `Sam3D` mirrors original SAM's three-part structure: 3D image encoder, 3D prompt encoder, and 3D mask decoder.
- `ImageEncoderViT3D` embeds volumes with `PatchEmbed3D`, then applies transformer `Block3D` layers. Blocks can switch between full attention and windowed attention depending on `window_size`.
- Decomposed relative position embeddings are extended to depth, height, and width. `add_decomposed_rel_pos` injects axis-specific positional terms into attention.
- `PromptEncoder3D` supports point, box, and mask prompts in volumetric coordinates. It creates sparse embeddings for points/boxes and dense embeddings for masks.
- `MaskDecoder3D.predict_masks` concatenates learned output tokens with sparse prompt tokens, runs the two-way transformer, upsamples image embeddings, and uses per-mask hypernetwork MLPs to produce mask logits.
- `postprocess_masks` resizes masks back toward the requested original size, which is important because SAM-style encoders operate at fixed internal image/volume sizes.

### 17. `SegFormer3D`

GitHub: [OSUPCVLab/SegFormer3D](https://github.com/OSUPCVLab/SegFormer3D)

- `SegFormer3D` follows the SegFormer design: a hierarchical transformer encoder with no heavy decoder, plus a lightweight all-MLP segmentation head.
- `PatchEmbedding` is overlapping and 3D, so each stage has local inductive bias before attention.
- `SelfAttention` supports `sr_ratio`; when larger than 1 it spatially reduces keys and values before attention, lowering token complexity at high resolutions.
- The transformer's MLP includes `DWConv`, a depthwise 3D convolution between linear layers, preserving local spatial information that a pure token MLP would lose.
- `MixVisionTransformer` emits four feature scales. The decoder head linearly projects each scale to the same embedding dimension, upsamples them to the highest resolution among the four, concatenates, fuses, and predicts classes.
- `build_segformer3d_model` is config-driven, so experiments are usually controlled through JSON/YAML settings rather than direct constructor edits.

### 18. `SegMamba`

GitHub: [ge-xing/SegMamba](https://github.com/ge-xing/SegMamba)

- `MambaEncoder` uses a 7x7x7 stride-2 stem followed by three stride-2 downsample layers, producing four feature scales.
- Before each Mamba stage, `GSC` applies gated/local convolutional processing with 3x3 and 1x1 paths, then adds the residual input. This preserves local texture before long-range sequence mixing.
- `MambaLayer` flattens each 3D feature map into tokens and calls Mamba with `bimamba_type="v3"` and stage-specific `num_slices` values `[64, 32, 16, 8]`.
- Each stage output is instance-normalized and passed through `MlpChannel`, a 1x1 channel MLP implemented with 3D convolutions.
- The decoder is UNETR-like: shallow convolutional encoders create skip features from the input and intermediate encoder outputs, while `UnetrUpBlock` reconstructs the segmentation map.
- The repo vendors `light_training`, `mamba`, `causal-conv1d`, and `monai`, so installation/dependency isolation matters before running experiments.

### 19. `Slim-UNETR`

GitHub: [deepang-ai/Slim-UNETR](https://github.com/deepang-ai/Slim-UNETR)

- The core `Block` is a sequence of residual submodules: positional depthwise convolution, local representation congregation, line/channel convolution, another positional block, sparse global transformer, local reverse diffusion, and another line convolution.
- `PatchPartition` is not a token partitioner in the transformer sense; it is a depthwise 3D positional convolution that injects local position information.
- `LocalRepresentationsCongregation` uses batch norm, 1x1 projection, depthwise 3x3 convolution, another norm, and 1x1 projection for cheap local mixing.
- `GlobalSparseTransformer` average-pools by factor `r`, computes q/k/v with 1x1 convolution, performs attention over the reduced 3D token grid, and returns a compact global-context tensor.
- `LocalReverseDiffusion` uses grouped transposed convolution with stride `r` to diffuse the sparse global representation back to local resolution, then applies group norm and pointwise convolution.
- The encoder/decoder files wrap these blocks into a compact UNETR-like segmentation model, trading full global attention for sparse attention plus local recovery.

### 20. `STU-Net`

GitHub: [openmedlab/STU-Net](https://github.com/openmedlab/STU-Net)

- The local checkout has README/assets only, so implementation-level claims cannot be audited from local source.
- The README describes scalable variants `STU-Net-S`, `STU-Net-B`, `STU-Net-L`, and `STU-Net-H`, ranging from small nnU-Net-like scale to a very large 1.4B-parameter model.
- The architecture is described as nnU-Net based, with joint scaling of depth and width rather than changing only channel count.
- The method is framed around transfer: pretrain on a large TotalSegmentator-style label corpus, then fine-tune or infer on downstream medical segmentation tasks.
- Because source files such as `network_architecture`, `network_training`, and `run_finetuning.py` are absent locally, the compendium should treat STU-Net as a documented reference/checkpoint entry until code is added.

### 21. `Swin-UMamba`

GitHub: [openmedlab/Swin-UMamba](https://github.com/openmedlab/Swin-UMamba)

- The local implementation is 2D/nnU-Net-v2 oriented, but it is important for efficient medical segmentation because it adapts VMamba/VSSM blocks with ImageNet pretraining.
- `SS2D` is the central selective-scan module. It initializes time-step parameters, state matrix logs, and skip terms, then scans image features in multiple spatial directions.
- `VSSBlock` wraps selective scan with normalization, residual connections, MLP/FFN behavior, and stochastic depth.
- `VSSLayer` stacks VSS blocks and optionally downsamples, while `VSSMEncoder` builds the hierarchical encoder and returns multi-scale features.
- `SwinUMamba` and `SwinUMambaD` differ in decoder strategy. `SwinUMambaD` includes an explicit residual U-Net decoder and patch expansion path.
- `load_pretrained_ckpt` maps pretrained VMamba weights into the segmentation encoder, which is the practical reason this repo can exploit natural-image pretraining.

### 22. `TP-Mamba`

GitHub: [xmed-lab/TP-Mamba](https://github.com/xmed-lab/TP-Mamba)

- The local code centers on `SAM_Mamba`, which adapts SAM features with LoRA-style QKV modification and U-Net decoding.
- `_LoRA_qkv_timm` wraps a QKV projection with low-rank trainable updates. This reduces trainable parameter count while adapting the SAM transformer.
- `adapter3` provides sequence-to-spatial and spatial-to-sequence conversions around adapter convolution/attention operations, allowing SAM token features to be reused in a segmentation decoder.
- `block_forward` customizes SAM block execution so adapter features can be inserted without rewriting the whole SAM encoder.
- `UnetConv3UP` and `UnetUpBlock` build the decoder that upsamples SAM-derived multiscale features.
- Despite the paper name "Tri-Plane Mamba", the local source snapshot is compact and mostly exposes the SAM adapter/U-Net implementation surface; deeper tri-plane details would need the full repo/paper context.

### 23. `UltraLight-VM-UNet`

GitHub: [wurenkai/UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet)

- This repo is a 2D skin-lesion segmentation model, not a 3D volumetric model, but the PVM block is relevant for lightweight Mamba-style segmentation.
- `PVMLayer` splits normalized channel tokens into four chunks, runs the same Mamba module on each chunk, adds a learnable skip scale, concatenates chunks, normalizes, projects, and reshapes back to feature maps.
- The encoder uses plain convolution for the first three stages and `PVMLayer` for deeper stages. The decoder mirrors this by using PVM layers in deeper upsampling stages and convolution in shallow recovery.
- `SC_Att_Bridge` combines `Spatial_Att_Bridge` and `Channel_Att_Bridge` across five feature levels, letting skip features share global channel context and local spatial saliency before decoding.
- Skip fusion uses addition rather than concatenation, which helps keep the model very small.
- The final output is bilinearly upsampled to input size and passed through sigmoid, so the local code is set up for binary/multilabel-style mask prediction.

### 24. `VeloxSeg`

GitHub: [JinPLu/VeloxSeg](https://github.com/JinPLu/VeloxSeg)

- `VeloxSeg` is designed for multimodal segmentation. `in_ch` is a sequence, so the model can treat modalities separately before fusion.
- The encoder has two complementary paths: a modal-fusion convolution branch using JLC-style local blocks and a modal-cooperative transformer branch using paired-window attention.
- `Paired_Windows_Attention` uses both big and small windows, gathering local regions, computing attention, and scattering attended results back. This targets the efficiency/robustness tradeoff in 3D volumes.
- During training, `forward` returns segmentation predictions, reconstruction predictions, and parameter/feature summaries for knowledge transfer. During inference, it returns only the segmentation prediction.
- Each modality has an `RC_Decoder` reconstruction teacher. Those teachers reconstruct the input modalities and produce features/Gram matrices used for spatially decoupled knowledge transfer.
- `scale_prediction` upsamples deep-supervision outputs to the input size using trilinear or bilinear interpolation depending on `spatial_dim`.
- The `compared_model/` folder is large and useful, but it should be separated from VeloxSeg itself when you compare model complexity.

### 25. `VISTA`

GitHub: [Project-MONAI/VISTA](https://github.com/Project-MONAI/VISTA)

- `VISTA3D2` combines automatic class-conditioned segmentation with point-prompted correction, making it closer to a universal/foundation segmentation system than a fixed-label U-Net.
- `SegResNetDS2` provides the dense image encoder and deep supervision outputs. It includes anisotropic kernel/scale utilities for medical volumes with nonuniform spacing.
- `Class_Mapping_Classify` maps class codes or label prompts into class-specific prediction behavior, supporting many anatomical categories.
- `Point_Mapping_SAM` is a SAM-style point head that maps click prompts and image embeddings into local corrections using two-way transformer blocks.
- `precompute_embedding` and `clear_cache` support repeated prompt interaction over the same image volume without recomputing the full encoder every time.
- `connected_components_combine` and `gaussian_combine` merge automatic and point-guided predictions. This is an algorithmic postprocessing/fusion layer, not merely a display utility.
- The scripts/configs support train, finetune, zero-shot evaluation, and supported evaluation workflows.

### 26. `WaveFormer`

GitHub: [mahfuzalhasan/WaveFormer](https://github.com/mahfuzalhasan/WaveFormer)

- `WaveFormer` is built around wavelet-aware transformer encoding and inverse-wavelet decoder reconstruction.
- `WaveletTransform3D` uses `ptwt.wavedec3` to produce low-frequency and high-frequency wavelet coefficients from 3D feature maps.
- `Block` can run a single-scale attention path or a multiscale wavelet path. In multiscale mode, wavelet features are partitioned into windows, attended, and reassembled with size-aware logic.
- `CCF_FFN` and `ChannelCalibration` add channel/frequency calibration around transformer features, helping preserve useful high-frequency structure.
- The decoder uses `UnetrIDWTBlock`, which refines high-frequency coefficients with `HFRefinementRes`, performs inverse DWT reconstruction through `ptwt.waverec3`, and fuses decoder/skip information.
- `Waveformer.forward_features` returns both multiscale hidden states and wavelet outputs. The segmentation backbone depends on both; removing the wavelet outputs changes the decoder algorithm.
- The repo includes older/shared training utilities, but the custom architecture surface is concentrated in `network_models/`.

### 27. `WMREN`

GitHub: [C101812/WMREN](https://github.com/C101812/WMREN)

- The local folder contains only `README.md` with the title `# WMREN`.
- There are no model classes, training scripts, configs, datasets, losses, or inference functions to audit locally.
- Based on the surrounding paper collection this may correspond to a wavelet/multiscale region-enhancement network, but that cannot be confirmed from local source.
- Treat this as a placeholder repo entry. If the full code is added later, this section should be regenerated from the actual architecture files.

## Source Inventory Appendix

This appendix lists the highest-value source files and their critical classes/functions. It is not a full dump of every vendored framework file, but it covers the custom algorithm surfaces used by the sections above.

### `3D-EffiViTCaps`

- `module/effiViTcaps.py`: `EffiViTCaps3D`, `forward`, `training_step`, `validation_step`, `predict_step`, `configure_optimizers`, `losses`, `_build_encoder`, `_build_decoder`, `_build_reconstruct_branch`, `_build_3D_EfficientViTBlock`.
- `main_block/efficientViT3D.py`: `Conv3d_BN`, `BN_Linear`, `Residual`, `FFN`, `CascadedGroupAttention3D`, `LocalWindowAttention3D`, `EfficientViTBlock3D`, `PatchMerging3D`, `PatchExpand3D`.
- `main_block/capsule_layers.py`: `_squash`, `_update_routing`, `DepthwiseConv3d`, `ConvSlimCapsule2D`, `DepthwiseDeconv3d`, `DeconvSlimCapsule2D`, `DepthwiseConv4d`, `ConvSlimCapsule3D`, `DepthwiseDeconv4d`, `DeconvSlimCapsule3D`, `MarginLoss`.

### `3DUX-Net`

- `networks/UXNet_3D/uxnet_encoder.py`: `LayerNorm`, `ux_block`, `uxnet_conv`.
- `networks/UXNet_3D/network_backbone.py`: `ProjectionHead`, `UXNET`, `proj_feat`, `forward`.

### `EfficientMedNeXt`

- `networks/MedNeXt/mednextv1/efficient_mednext_blocks.py`: `MultiDilationDepthwiseConv3D`, `EfficientMedNeXtBlock`, `EfficientMedNeXtDownBlock`, `EfficientMedNeXtUpBlock`, `OutBlock`, `LayerNorm`.
- `networks/MedNeXt/mednextv1/EfficientMedNext.py`: `EfficientMedNeXt`, `iterative_checkpoint`, `forward`.
- `networks/MedNeXt/mednextv1/EfficientMedNext_Full.py`: `EfficientMedNeXt_L`, `iterative_checkpoint`, `forward`.
- `networks/MedNeXt/mednextv1/create_efficient_mednext.py`: tiny/small/medium/large factory functions and dispatcher.

### `EfficientViM`

- `classification/models/efficientvim.py`: `HSMSSD`, `EfficientViMBlock`, `EfficientViMStage`, `EfficientViM`, `EfficientViM_M1`, `EfficientViM_M2`, `EfficientViM_M3`, `EfficientViM_M4`.

### `EffiDec3D`

- `networks/UXNet_3D/network_backbone.py`: `ModifiedUnetrUpBlock`, `UXNET`, `UXNET_EffiDec3D`.
- `networks/swin_unetr_effidec3d.py`: `ModifiedUnetrUpBlock`, `SwinUNETR`, `SwinUNETR_EffiDec3D`, `WindowAttention`, `SwinTransformerBlock`, `PatchMergingV2`, `PatchMerging`, `BasicLayer`, `SwinTransformer`, `compute_mask`, `window_partition`, `window_reverse`.
- `networks/MedNeXt/mednextv1/create_mednextv1_effidec3d.py`: EffiDec3D MedNeXt factories.

### `EM-Net`

- `models/em_net_model.py`: `LayerNorm`, `MambaLayer`, `MlpChannel`, `FFParser_n`, `Spectral_Layer`, `EMNetUpBlock`, `EMNet`, `MambaEncoder`, `EMNet_Bot`, `MambaEncoder_Bot`, `EMNet_Conv_Enc`, `MambaEncoder_Conv_Enc`, `MambaLayer_Conv`.

### `FastSAM3D`

- `segment_anything/modeling/image_encoder3D.py`: `ImageEncoderViT3D`, `Block3D`, `Block3D_woatt`, `Attention`, `PatchEmbed3D`, 3D window partition helpers, decomposed relative position helpers.
- `segment_anything/modeling/image_encoder3D_dilatedattention.py`: dilated 3D attention variant.
- `segment_anything/modeling/mask_decoder3D.py`: `TwoWayTransformer3D`, `TwoWayAttentionBlock3D`, `Attention`, `MaskDecoder3D`, `MLP`.
- `segment_anything/modeling/prompt_encoder3D.py`: `PromptEncoder3D`, `PositionEmbeddingRandom3D`.
- `distillation.py`, `train.py`, `train_unfreeze.py`, `validation.py`, `validation_student.py`: training/evaluation algorithms.

### `HER-Seg`

- `model.py`: `PointGenerator`, `MHRMedSeg`, `PFD`, `PFDTeacher`, `MHRMedSegSAM2`, `FpnNeck`.
- `SAM/image_encoder.py`: `ImageEncoderViT`.
- `SAM/ecm_decoder.py`: `MaskDecoder`.
- `SAM/prompt_encoder.py`: `PromptEncoder`.
- `SAM/transformer.py`: `TwoWayTransformer`, `TwoWayAttentionBlock`.
- `sam2/modeling/*`: SAM2 base, memory encoder, memory attention, HieraDet image encoder, SAM mask decoder, prompt encoder.
- `unets/unet.py`: U-Net blocks.
- `loss.py`: Dice-style losses.

### `LHUNet`

- `src/lhunet/models/lhunet.py`: `LHUNet`.
- `src/lhunet/blocks/base.py`: `BaseBlock`, `get_conv_layer`, `LayerNorm`, `get_padding`, `get_output_padding`.
- `src/lhunet/blocks/cnn.py`: `DCNNBlock`, `UnetResBlock`, `UnetBasicBlock`, `UnetUpBlock`, `UnetOutBlock`, `CNNEncoder`, `CNNDecoder`.
- `src/lhunet/blocks/hyb.py`: `BaseHybridBlock`, `HybridEncoder`, `HybridDecoder`.

### `LightM-UNet`

- `lightm-unet/nnunetv2/nets/LightMUNet.py`: `MambaLayer`, `ResMambaBlock`, `ResUpBlock`, `LightMUNet`.
- `lightm-unet/nnunetv2/nets/UMambaEnc.py`: `MambaLayer`, `ResidualMambaEncoder`, `UNetResDecoder`, `UMambaEnc`, `get_umamba_enc_from_plans`.
- `lightm-unet/nnunetv2/nets/UMambaBot.py`: `UNetResDecoder`, `UMambaBot`, `get_umamba_bot_from_plans`.

### `MedNeXt`

- `nnunet_mednext/network_architecture/mednextv1/MedNextV1.py`: `MedNeXt`, `iterative_checkpoint`, `forward`.
- `nnunet_mednext/network_architecture/mednextv1/blocks.py`: `MedNeXtBlock`, `MedNeXtDownBlock`, `MedNeXtUpBlock`, `OutBlock`, `LayerNorm`.
- `nnunet_mednext/network_architecture/mednextv1/create_mednext_v1.py`: model-size factories.

### `Mobile-U-ViT`

- `network/MobileUViT.py`: `Residual`, `ConvUtr`, `Embeddings`, `Mlp`, `CMlp`, `GlobalSparseAttn`, `LocalAgg`, `SelfAttn`, `LGLBlock`, `up_conv`, `conv_block`, `MobileUViT`, `mobileuvit`, `mobileuvit_l`.
- `network/MobileUViT_3D.py`: same structure adapted to 3D, with `LKLGLBlock`.

### `nnMamba`

- `nnMamba.py`: `conv3x3`, `conv1x1`, `BasicBlock`, `make_res_layer`, `MambaLayer`, `DoubleConv`, `SingleConv`, `Attentionlayer`, `nnMambaSeg`.
- `nnMamba4cls.py`: `MambaLayer`, `nnMambaEncoder`, classification path.
- `classification/networks/ssm_nnMamba.py`: classification SSM variant.

### `Pancreatic-Tumor-SEG`

- `PFNet.py`: `CA_Block`, `SA_Block`, `Context_Exploration_Block`, `Positioning`, `Focus`, `PFNet`.
- `PFNet_stan.py`: same components with `PFNet_unet` standard variant.
- `PFNet_unet.py`: U-Net-backed PFNet variant.

### `SAM-Med3D`

- `segment_anything/modeling/image_encoder3D.py`: `ImageEncoderViT3D`, `Block3D`, `Attention`, `PatchEmbed3D`, 3D window helpers.
- `segment_anything/modeling/prompt_encoder3D.py`: `PromptEncoder3D`, `PositionEmbeddingRandom3D`.
- `segment_anything/modeling/mask_decoder3D.py`: `TwoWayTransformer3D`, `TwoWayAttentionBlock3D`, `Attention`, `MaskDecoder3D`, `MLP`.
- `segment_anything/modeling/sam3D.py`: `Sam3D`.
- `segment_anything/build_sam3D.py`: model factory.

### `SegFormer3D`

- `architectures/segformer3d.py`: `build_segformer3d_model`, `SegFormer3D`, `PatchEmbedding`, `SelfAttention`, `TransformerBlock`, `MixVisionTransformer`, `_MLP`, `DWConv`, `cube_root`, `MLP_`, `SegFormerDecoderHead`.

### `SegMamba`

- `model_segmamba/segmamba.py`: `LayerNorm`, `MambaLayer`, `MlpChannel`, `GSC`, `MambaEncoder`, `SegMamba`.

### `Slim-UNETR`

- `src/SlimUNETR/SlimUNETR.py`: `SlimUNETR`.
- `src/SlimUNETR/Slim_UNETR_Block.py`: `PatchPartition`, `LineConv`, `LocalRepresentationsCongregation`, `GlobalSparseTransformer`, `LocalReverseDiffusion`, `Block`.

### `Swin-UMamba`

- `swin_umamba/nnunetv2/nets/SwinUMamba.py`: `PatchEmbed2D`, `PatchMerging2D`, `SS2D`, `VSSBlock`, `VSSLayer`, `VSSMEncoder`, `SwinUMamba`, `load_pretrained_ckpt`, `get_swin_umamba_from_plans`.
- `swin_umamba/nnunetv2/nets/SwinUMambaD.py`: `PatchExpand`, `FinalPatchExpand_X4`, `UNetResDecoder`, `SwinUMambaD`, `get_swin_umamba_d_from_plans`.

### `TP-Mamba`

- `networks/sam/samunet_mamba.py`: `window_partition`, `window_unpartition`, initialization helpers, `block_forward`, `UnetConv3UP`, `UnetUpBlock`, `_LoRA_qkv_timm`, `adapter3`, `SAM_Mamba`.

### `UltraLight-VM-UNet`

- `models/UltraLight_VM_UNet.py`: `PVMLayer`, `Channel_Att_Bridge`, `Spatial_Att_Bridge`, `SC_Att_Bridge`, `UltraLight_VM_UNet`.

### `VeloxSeg`

- `model/VeloxSeg.py`: `VeloxSeg`.
- `model/Encoder.py`: `Conv_Encoder`, `Transformer_Encoder`, `Encoder`.
- `model/Decoder.py`: `RC_Decoder`, `Seg_Decoder`.
- `model/components/PWA.py`: `Paired_Windows_Attention`, `MultiModal_Paired_Windows_Attention`, `Paired_Windows_TransformerBlock`, `Transformer_BasicLayer`, `Cross_Channel_Attention`.
- `model/components/conv_blocks.py`: `DownConv`, `UpConv`, `JLC`, `JLCLayer`.
- `model/components/attention_utils.py`: `LayerNorm`, `FFN`, `PositionalEmbedding`, `PatchMerging`.

### `VISTA`

- `vista3d/vista3d/modeling/vista3d.py`: `VISTA3D2`, `precompute_embedding`, `clear_cache`, `get_bs`, `update_point_to_patch`, `connected_components_combine`, `gaussian_combine`, `set_auto_grad`, `forward`.
- `vista3d/vista3d/modeling/point_head.py`: `Point_Mapping_SAM`.
- `vista3d/vista3d/modeling/class_head.py`: `Class_Mapping_Classify`.
- `vista3d/vista3d/modeling/segresnetds.py`: `scales_for_resolution`, `aniso_kernel`, `SegResBlock`, `SegResEncoder`, `SegResNetDS2`.
- `vista3d/vista3d/modeling/sam_blocks.py`: SAM-style transformer/prompt blocks.

### `WaveFormer`

- `network_models/network_backbone.py`: `ProjectionHead`, `ChannelCalibration`, `Waveformer`, `create_waveformer`.
- `network_models/waveformer.py`: `MultiscaleTransformer`.
- `network_models/wave_helper.py`: `ProjectionUpsample`, `DWConv`, `PatchMergingV2`, `PatchMerging`, `CCF_FFN`, `Mlp`, `WaveletTransform3D`, `Block`, `OverlapPatchEmbed`, `PatchEmbed`, `PosCNN`.
- `network_models/idwt_upsample.py`: `HFRefinementRes`, `UnetrIDWTBlock`.
- `network_models/attention.py`: `Attention`.

### `STU-Net` and `WMREN`

- `STU-Net`: README/assets only in local checkout; no local model classes.
- `WMREN`: title-only README; no local model classes.

## Practical Use For Your Kidney Segmentation Work

If your target is an efficient 3D kidney/tumor segmentation model, the most immediately reusable repos are:

1. `EfficientMedNeXt`: strong candidate if you want a convolutional model with multi-scale receptive fields and efficient decoder.
2. `EffiDec3D`: useful if you already like a backbone such as 3D UX-Net, SwinUNETR, or MedNeXt but want a leaner decoder.
3. `SegMamba` or `EM-Net`: useful for long-range dependencies in 3D volumes with state-space layers.
4. `WaveFormer`: useful if wavelet frequency decomposition is interesting for boundary/detail recovery.
5. `VISTA` or `SAM-Med3D`: useful for promptable or foundation-model-style segmentation, but heavier to adapt.
6. `VeloxSeg`: useful if paired-window attention and efficiency are both priorities.

For a clean experiment stack, prefer one of:

- nnU-Net style stack: `MedNeXt`, `EfficientMedNeXt`, `LightM-UNet`, `Swin-UMamba`.
- MONAI style stack: `3DUX-Net`, `EffiDec3D`, `SAM-Med3D`, `VISTA`.
- Custom lightweight stack: `Slim-UNETR`, `Mobile-U-ViT`, `UltraLight-VM-UNet`, `VeloxSeg`.
