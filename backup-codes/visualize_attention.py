"""
Attention/Feature Visualization for MedDINO-VISTA3D
Generates cosine similarity maps similar to Figure 4 in the paper.
"""

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from meddino_vista3d_dinov3 import MedDINOv3VISTA3D, get_config


def extract_features(model, image_slice):
    """
    Extract DINOv2 features from a 2D slice.
    Returns: features (patch_grid_h, patch_grid_w, hidden_dim)
    """
    model.eval()
    with torch.no_grad():
        # Prepare slice for ViT
        img_2d = (
            torch.from_numpy(image_slice).float().unsqueeze(0).unsqueeze(0)
        )  # (1, 1, H, W)

        # Resize to be divisible by 14 (DINOv2 patch size)
        h, w = img_2d.shape[2:]
        new_h = ((h + 13) // 14) * 14  # Round up to nearest multiple of 14
        new_w = ((w + 13) // 14) * 14
        img_2d = F.interpolate(
            img_2d, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

        img_rgb = img_2d.repeat(1, 3, 1, 1)  # (1, 3, H, W)

        # Extract features from last layer
        features = model.encoder.vit.get_intermediate_layers(img_rgb, n=[11])[
            0
        ]  # (1, N_patches, 768)

        # Reshape to grid
        n_patches = features.shape[1]
        grid_size = int(np.sqrt(n_patches))
        features = features.reshape(grid_size, grid_size, -1)  # (h, w, C)

        return features.cpu().numpy()


def compute_cosine_similarity(features, reference_patch_idx):
    """
    Compute cosine similarity between reference patch and all other patches.

    Args:
        features: (h, w, C) feature map
        reference_patch_idx: (row, col) index of reference patch

    Returns:
        similarity_map: (h, w) cosine similarity values
    """
    h, w, c = features.shape
    features_flat = features.reshape(-1, c)  # (h*w, C)

    # Normalize features
    features_norm = features_flat / (
        np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8
    )

    # Reference feature
    ref_idx = reference_patch_idx[0] * w + reference_patch_idx[1]
    ref_feature = features_norm[ref_idx : ref_idx + 1]  # (1, C)

    # Compute cosine similarity
    similarities = np.dot(features_norm, ref_feature.T).squeeze()  # (h*w,)
    similarity_map = similarities.reshape(h, w)

    return similarity_map


def visualize_attention_evolution(
    case_dir, checkpoint_paths, output_path, slice_idx=None
):
    """
    Visualize attention map evolution across training stages (like Figure 4).

    Args:
        case_dir: Path to KiTS23 case (e.g., './kits23/dataset/case_00000')
        checkpoint_paths: Dict of {'Stage 1: 50k': 'path/to/ckpt1.pth', ...}
        output_path: Where to save visualization
        slice_idx: Which slice to visualize (None = middle slice)
    """
    # Load image
    img = nib.load(Path(case_dir) / "imaging.nii.gz").get_fdata()

    if slice_idx is None:
        slice_idx = img.shape[0] // 2  # Middle slice

    img_slice = img[slice_idx]

    # Normalize
    img_slice = (img_slice - img_slice.mean()) / (img_slice.std() + 1e-8)

    # Reference patch (e.g., center of kidney/tumor)
    ref_patch = (
        img_slice.shape[0] // 14 // 2,
        img_slice.shape[1] // 14 // 2,
    )  # Approximate center

    # Setup figure
    n_checkpoints = len(checkpoint_paths) + 1  # +1 for original image
    fig, axes = plt.subplots(2, n_checkpoints, figsize=(4 * n_checkpoints, 8))

    # Plot original image
    axes[0, 0].imshow(img_slice, cmap="gray")
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    # Load config and model
    config = get_config("quick_test")

    # Process each checkpoint
    for idx, (stage_name, ckpt_path) in enumerate(checkpoint_paths.items(), start=1):
        model = MedDINOv3VISTA3D(
            num_classes=config["num_classes"], dinov2_backbone=config["dinov2_backbone"]
        )

        # Load checkpoint
        if Path(ckpt_path).exists():
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            print(f"✓ Loaded {stage_name} from {ckpt_path}")
        else:
            print(f"⚠ Checkpoint not found: {ckpt_path}, using random weights")

        # Extract features
        features = extract_features(model, img_slice)

        # Compute similarity
        similarity_map = compute_cosine_similarity(features, ref_patch)

        # Resize to image size for visualization
        similarity_resized = (
            F.interpolate(
                torch.from_numpy(similarity_map).unsqueeze(0).unsqueeze(0),
                size=img_slice.shape,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

        # Plot top row: heatmap overlay
        axes[0, idx].imshow(img_slice, cmap="gray", alpha=0.3)
        im = axes[0, idx].imshow(
            similarity_resized, cmap="jet", alpha=0.7, vmin=0, vmax=1
        )
        axes[0, idx].set_title(stage_name)
        axes[0, idx].axis("off")

        # Plot bottom row: pure heatmap
        axes[1, idx].imshow(similarity_resized, cmap="jet", vmin=0, vmax=1)
        axes[1, idx].axis("off")

    plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    plt.suptitle(f"Cosine Similarity Evolution - Slice {slice_idx}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved visualization to {output_path}")
    plt.close()


def visualize_decoder_attention(model, image, output_path):
    """
    Visualize VISTA3D decoder's cross-attention maps for each class.
    Shows which regions the model focuses on for kidney/tumor/cyst.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Prepare image - ensure dimensions are divisible by 14
        img_tensor = (
            torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        )  # (1, 1, D, H, W)

        # Resize to be divisible by 14
        d, h, w = img_tensor.shape[2:]
        new_d = ((d + 13) // 14) * 14
        new_h = ((h + 13) // 14) * 14
        new_w = ((w + 13) // 14) * 14

        img_resized = F.interpolate(
            img_tensor,
            size=(new_d, new_h, new_w),
            mode="trilinear",
            align_corners=False,
        ).to(device)

        # Forward pass
        logits = model(img_resized)

        # Resize back to original size
        logits_original = F.interpolate(
            logits, size=(d, h, w), mode="trilinear", align_corners=False
        )
        probs = F.softmax(logits_original, dim=1).cpu().numpy()[0]  # (4, D, H, W)

    # Visualize middle slice
    mid_slice = probs.shape[1] // 2
    classes = ["Background", "Kidney", "Tumor", "Cyst"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for idx, (ax, class_name) in enumerate(zip(axes.ravel(), classes)):
        ax.imshow(image[mid_slice], cmap="gray", alpha=0.4)
        im = ax.imshow(probs[idx, mid_slice], cmap="hot", alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f"{class_name} Attention Map")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Class-Specific Attention Maps (VISTA3D Decoder)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved decoder attention to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Example usage
    output_dir = Path("./output/meddino_vista3d_dinov3")

    # Define checkpoints from your training
    checkpoints = {
        "Stage 1: Epoch 1": output_dir / "best_model_stage1.pth",
        "Stage 1: Epoch 3": output_dir / "best_model_stage1_dice.pth",
        "Stage 2: Epoch 2": output_dir / "best_model_final.pth",
    }

    # Visualize evolution
    print("Generating attention evolution visualization...")
    visualize_attention_evolution(
        case_dir="./kits23/dataset/case_00000",
        checkpoint_paths=checkpoints,
        output_path=output_dir / "attention_evolution.png",
        slice_idx=100,  # Adjust based on your data
    )

    # Visualize decoder attention
    print("\nGenerating decoder attention maps...")
    config = get_config("quick_test")
    model = MedDINOv3VISTA3D(
        num_classes=config["num_classes"], dinov2_backbone=config["dinov2_backbone"]
    )
    model.load_state_dict(torch.load(output_dir / "best_model_final.pth"))

    # Load test image
    img = nib.load("./kits23/dataset/case_00000/imaging.nii.gz").get_fdata()
    img_norm = (img - img.mean()) / (img.std() + 1e-8)

    visualize_decoder_attention(
        model=model, image=img_norm, output_path=output_dir / "decoder_attention.png"
    )

    print("\n✅ All visualizations generated!")
