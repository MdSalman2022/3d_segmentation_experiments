"""
Stripped-down 3D multi-class segmentation methodology.

Label meaning in this anonymized version:
  0 -> background
  1 -> larger object region in each 3D image
  2 -> smaller object region in each 3D image
  3 -> smallest object region in each 3D image

Minimum kept connected-component size after inference:
  class 1 -> 5000 voxels
  class 2 -> 50 voxels
  class 3 -> 50 voxels

Results copied from the existing output files:
  - Best validation mean foreground Dice: 0.3670
  - Total training time: 21.24 hours
  - Test volumes evaluated: 40
  - Test mean foreground Dice: 0.3011
  - Test mean foreground IoU: 0.2286
  - Test Dice by foreground class:
      class 1 (larger object region in each 3D image): 0.6923 +- 0.1737
      class 2 (smaller object region in each 3D image): 0.1390 +- 0.1923
      class 3 (smallest object region in each 3D image): 0.0719 +- 0.1383
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


DTYPE = torch.bfloat16


@dataclass
class MethodConfig:
    num_classes: int = 4
    feature_size: int = 48
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 2
    grad_accum_steps: int = 2
    lr_stage1: float = 1e-3
    lr_stage2: float = 2e-4
    num_epochs_stage1: int = 50
    num_epochs_stage2: int = 250
    lora_rank_llm: int = 4
    lora_rank_encoder: int = 2
    use_llm: bool = True
    class_sampling_bias: tuple[float, float, float, float] = (0.20, 0.60, 0.10, 0.10)
    class_keep_min_voxels: tuple[int, int, int, int] = (0, 5000, 50, 50)
    ce_weight: tuple[float, float, float, float] = (0.1, 1.0, 8.0, 4.0)
    sw_batch_size: int = 4
    sw_overlap: float = 0.25
    grad_clip_norm: float = 1.0


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 4):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A)
        self.scaling = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.linear(x, self.base.weight, self.base.bias)
            + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        )


def apply_lora_to_linears(
    module: nn.Module,
    rank: int,
    target_names: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2", "linear1", "linear2"),
) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and any(token in name for token in target_names):
            setattr(module, name, LoRALinear(child, rank=rank))
        else:
            apply_lora_to_linears(child, rank=rank, target_names=target_names)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class TransformerPriorBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int, ffn_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        n_rep = self.num_heads // self.num_kv_heads
        bsz, heads, seq_len, dim = x.shape
        return x.unsqueeze(2).expand(bsz, heads, n_rep, seq_len, dim).reshape(
            bsz, heads * n_rep, seq_len, dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        bsz, seq_len, _ = h.shape
        q = self.q_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.o_proj(attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1))
        h = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class LLMBottleneckBlock(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        hidden_dim: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int = 4,
        ffn_dim: int = 5632,
        lora_rank: int = 4,
        pretrained_layer_state: Optional[dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.llm_block = TransformerPriorBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            ffn_dim=ffn_dim,
        )
        if pretrained_layer_state is not None:
            self._load_pretrained_layer(pretrained_layer_state)
        for p in self.llm_block.parameters():
            p.requires_grad = False
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.llm_block, name, LoRALinear(getattr(self.llm_block, name), rank=lora_rank))
        self.proj_in = nn.Linear(vision_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, vision_dim)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)
        self.norm = nn.LayerNorm(vision_dim)

    def _load_pretrained_layer(self, layer_state: dict[str, torch.Tensor]) -> None:
        key_map = {
            "norm1.weight": "input_layernorm.weight",
            "norm2.weight": "post_attention_layernorm.weight",
            "q_proj.weight": "self_attn.q_proj.weight",
            "k_proj.weight": "self_attn.k_proj.weight",
            "v_proj.weight": "self_attn.v_proj.weight",
            "o_proj.weight": "self_attn.o_proj.weight",
            "gate_proj.weight": "mlp.gate_proj.weight",
            "up_proj.weight": "mlp.up_proj.weight",
            "down_proj.weight": "mlp.down_proj.weight",
        }
        mapped = {
            own_key: layer_state[src_key]
            for own_key, src_key in key_map.items()
            if src_key in layer_state and own_key in self.llm_block.state_dict()
        }
        self.llm_block.load_state_dict(mapped, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, depth, height, width = x.shape
        x_seq = x.flatten(2).permute(0, 2, 1)
        h = self.proj_out(self.llm_block(self.proj_in(x_seq)))
        return self.norm(x_seq + h).permute(0, 2, 1).view(bsz, channels, depth, height, width)


class SwinUNETRWithPrior(nn.Module):
    def __init__(
        self,
        config: MethodConfig,
        encoder_state_dict: Optional[dict[str, torch.Tensor]] = None,
        llm_layer_state: Optional[dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.net = SwinUNETR(
            in_channels=1,
            out_channels=config.num_classes,
            feature_size=config.feature_size,
            use_checkpoint=True,
            spatial_dims=3,
        )
        if encoder_state_dict is not None:
            self.net.load_state_dict(encoder_state_dict, strict=False)
        self.use_llm = config.use_llm
        self.llm_block = (
            LLMBottleneckBlock(
                vision_dim=config.feature_size * 16,
                lora_rank=config.lora_rank_llm,
                pretrained_layer_state=llm_layer_state,
            )
            if config.use_llm
            else None
        )

    def freeze_encoder(self) -> None:
        for p in self.net.swinViT.parameters():
            p.requires_grad = False

    def enable_encoder_lora(self, rank: int) -> None:
        for p in self.net.swinViT.parameters():
            p.requires_grad = False
        apply_lora_to_linears(
            self.net.swinViT,
            rank=rank,
            target_names=("qkv", "proj", "fc1", "fc2"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = self.net.swinViT(x, self.net.normalize)
        enc0 = self.net.encoder1(x)
        enc1 = self.net.encoder2(hs[0])
        enc2 = self.net.encoder3(hs[1])
        enc3 = self.net.encoder4(hs[2])
        dec4 = self.net.encoder10(hs[4])
        if self.llm_block is not None:
            dec4 = self.llm_block(dec4)
        dec3 = self.net.decoder5(dec4, hs[3])
        dec2 = self.net.decoder4(dec3, enc3)
        dec1 = self.net.decoder3(dec2, enc2)
        dec0 = self.net.decoder2(dec1, enc1)
        out = self.net.decoder1(dec0, enc0)
        return self.net.out(out)


class DiceCELoss(nn.Module):
    def __init__(self, config: MethodConfig, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = config.num_classes
        self.smooth = smooth
        self.register_buffer("ce_weight", torch.tensor(config.ce_weight))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.squeeze(1).long()
        pred_soft = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        tp = (pred_soft * target_oh).sum(dim=(2, 3, 4))
        fp = pred_soft.sum(dim=(2, 3, 4)) - tp
        fn = target_oh.sum(dim=(2, 3, 4)) - tp
        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        loss_dice = (1 - dice[:, 1:]).mean()
        loss_ce = F.cross_entropy(logits, target, weight=self.ce_weight.to(logits.dtype))
        return 0.5 * loss_dice + 0.5 * loss_ce


def augment_patch(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for axis in range(3):
        if np.random.random() < 0.5:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    if np.random.random() < 0.40:
        k = np.random.randint(1, 4)
        axes = tuple(np.random.choice(3, 2, replace=False).tolist())
        image = np.rot90(image, k=k, axes=axes)
        label = np.rot90(label, k=k, axes=axes)
    if np.random.random() < 0.15:
        scale = float(np.random.uniform(0.88, 1.12))
        image = ndimage.zoom(image.astype(np.float32), scale, order=1)
        label = ndimage.zoom(label.astype(np.float32), scale, order=0)
    if np.random.random() < 0.25:
        image = np.clip(
            image + np.random.randn(*image.shape).astype(np.float32) * float(np.random.uniform(0.01, 0.08)),
            0.0,
            1.0,
        )
    if np.random.random() < 0.20:
        image = np.clip(image + float(np.random.uniform(-0.1, 0.1)), 0.0, 1.0)
    if np.random.random() < 0.20:
        image = np.power(np.clip(image, 0.0, 1.0), float(np.random.uniform(0.7, 1.5)))
    if np.random.random() < 0.15:
        image = ndimage.gaussian_filter(image.astype(np.float32), float(np.random.uniform(0.5, 1.5)))
    return image.copy(), label.copy()


def sample_training_patch(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Sequence[int],
    class_sampling_bias: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    pd, ph, pw = patch_size
    depth, height, width = image.shape
    if depth < pd or height < ph or width < pw:
        pad = ((0, max(0, pd - depth)), (0, max(0, ph - height)), (0, max(0, pw - width)))
        image = np.pad(image, pad, mode="constant", constant_values=0.0)
        label = np.pad(label, pad, mode="constant", constant_values=0)
        depth, height, width = image.shape

    r = np.random.random()
    b1, b2, b3, _ = class_sampling_bias
    if r < b2:
        target_class = 2
    elif r < b2 + b3:
        target_class = 3
    elif r < b2 + b3 + b1:
        target_class = 1
    else:
        target_class = None

    if target_class is not None:
        idx = np.argwhere(label == target_class)
        if len(idx) == 0:
            idx = np.argwhere(label > 0)
        if len(idx) > 0:
            center = idx[np.random.randint(len(idx))]
            ds = int(np.clip(center[0] - pd // 2, 0, depth - pd))
            hs = int(np.clip(center[1] - ph // 2, 0, height - ph))
            ws = int(np.clip(center[2] - pw // 2, 0, width - pw))
        else:
            ds = np.random.randint(0, max(1, depth - pd + 1))
            hs = np.random.randint(0, max(1, height - ph + 1))
            ws = np.random.randint(0, max(1, width - pw + 1))
    else:
        ds = np.random.randint(0, max(1, depth - pd + 1))
        hs = np.random.randint(0, max(1, height - ph + 1))
        ws = np.random.randint(0, max(1, width - pw + 1))

    image = image[ds : ds + pd, hs : hs + ph, ws : ws + pw]
    label = label[ds : ds + pd, hs : hs + ph, ws : ws + pw]
    return augment_patch(image, label)


def postprocess_prediction(prediction: np.ndarray, config: MethodConfig) -> np.ndarray:
    output = prediction.copy()
    for class_id, min_voxels in enumerate(config.class_keep_min_voxels):
        if class_id == 0 or min_voxels <= 0:
            continue
        mask = output == class_id
        if not mask.any():
            continue
        labeled, _ = ndimage.label(mask)
        component_sizes = np.bincount(labeled.ravel())
        remove = component_sizes < min_voxels
        remove[0] = False
        output[remove[labeled]] = 0
    return output


def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Sequence[int],
    sw_batch_size: int,
    predictor: nn.Module,
    overlap: float = 0.25,
) -> torch.Tensor:
    assert inputs.shape[0] == 1
    pd, ph, pw = roi_size
    _, _, d_orig, h_orig, w_orig = inputs.shape
    pad_d = max(0, pd - d_orig)
    pad_h = max(0, ph - h_orig)
    pad_w = max(0, pw - w_orig)
    if pad_d or pad_h or pad_w:
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h, 0, pad_d), value=0.0)

    _, _, depth, height, width = inputs.shape
    step_d = max(1, int(pd * (1 - overlap)))
    step_h = max(1, int(ph * (1 - overlap)))
    step_w = max(1, int(pw * (1 - overlap)))

    def starts(size: int, patch: int, step: int) -> list[int]:
        points = list(range(0, size - patch + 1, step))
        if not points or points[-1] + patch < size:
            points.append(size - patch)
        return points

    coords = [
        (d0, h0, w0)
        for d0 in starts(depth, pd, step_d)
        for h0 in starts(height, ph, step_h)
        for w0 in starts(width, pw, step_w)
    ]
    output = None
    count = None

    for start in range(0, len(coords), sw_batch_size):
        batch_coords = coords[start : start + sw_batch_size]
        batch_in = torch.cat(
            [
                inputs[
                    :,
                    :,
                    d0 : d0 + pd,
                    h0 : h0 + ph,
                    w0 : w0 + pw,
                ]
                for d0, h0, w0 in batch_coords
            ],
            dim=0,
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=DTYPE):
                batch_out = predictor(batch_in).float()

        if output is None:
            n_classes = batch_out.shape[1]
            output = torch.zeros(1, n_classes, depth, height, width)
            count = torch.zeros(1, 1, depth, height, width)

        for i, (d0, h0, w0) in enumerate(batch_coords):
            output[:, :, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw] += batch_out[i : i + 1].cpu()
            count[:, :, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw] += 1

    output = output / count.clamp(min=1e-8)
    return output[:, :, :d_orig, :h_orig, :w_orig]


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_accum_steps: int,
    grad_clip_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    count = 0

    for step, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=DTYPE):
            logits = model(image)
            loss = loss_fn(logits, label) / grad_accum_steps
        loss.backward()
        total_loss += loss.item() * grad_accum_steps
        count += 1

        if step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total_loss / max(count, 1)


def validate_sliding_window(
    model: nn.Module,
    volumes: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    config: MethodConfig,
    device: torch.device,
) -> float:
    model.eval()
    tp = np.zeros(config.num_classes, dtype=np.float64)
    fp = np.zeros(config.num_classes, dtype=np.float64)
    fn = np.zeros(config.num_classes, dtype=np.float64)

    for volume, label in zip(volumes, labels):
        image = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(device)
        logits = sliding_window_inference(
            image,
            roi_size=config.patch_size,
            sw_batch_size=config.sw_batch_size,
            predictor=model,
            overlap=config.sw_overlap,
        )
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = postprocess_prediction(pred, config)
        for class_id in range(config.num_classes):
            tp[class_id] += int(((pred == class_id) & (label == class_id)).sum())
            fp[class_id] += int(((pred == class_id) & (label != class_id)).sum())
            fn[class_id] += int(((pred != class_id) & (label == class_id)).sum())

    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return float(dice[1:].mean())


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.01 + 0.5 * 0.99 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_two_stage_methodology(
    model: SwinUNETRWithPrior,
    train_loader: Iterable[dict[str, torch.Tensor]],
    val_volumes: Sequence[np.ndarray],
    val_labels: Sequence[np.ndarray],
    config: MethodConfig,
    device: torch.device,
) -> float:
    loss_fn = DiceCELoss(config).to(device)
    best_mean_fg_dice = 0.0

    model.freeze_encoder()
    optimizer_stage1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr_stage1,
        weight_decay=1e-5,
    )
    scheduler_stage1 = cosine_warmup_scheduler(optimizer_stage1, config.num_epochs_stage1, warmup_epochs=3)
    for _ in range(config.num_epochs_stage1):
        train_one_epoch(
            model,
            train_loader,
            optimizer_stage1,
            loss_fn,
            device,
            grad_accum_steps=config.grad_accum_steps,
            grad_clip_norm=config.grad_clip_norm,
        )
        scheduler_stage1.step()
        best_mean_fg_dice = max(
            best_mean_fg_dice,
            validate_sliding_window(model, val_volumes, val_labels, config, device),
        )

    model.enable_encoder_lora(rank=config.lora_rank_encoder)
    optimizer_stage2 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr_stage2,
        weight_decay=1e-5,
    )
    scheduler_stage2 = cosine_warmup_scheduler(optimizer_stage2, config.num_epochs_stage2, warmup_epochs=5)
    for _ in range(config.num_epochs_stage2):
        train_one_epoch(
            model,
            train_loader,
            optimizer_stage2,
            loss_fn,
            device,
            grad_accum_steps=config.grad_accum_steps,
            grad_clip_norm=config.grad_clip_norm,
        )
        scheduler_stage2.step()
        best_mean_fg_dice = max(
            best_mean_fg_dice,
            validate_sliding_window(model, val_volumes, val_labels, config, device),
        )

    return best_mean_fg_dice
