#!/usr/bin/env python3
"""
Quick verification that the fixed model can load properly.
Tests model instantiation without training.
"""

import sys

sys.path.insert(0, ".")

try:
    print("=" * 60)
    print("Testing Fixed MedDINO-VISTA3D Model")
    print("=" * 60)

    import torch

    print(f"✓ PyTorch {torch.__version__} loaded")

    # Import the model
    from meddino_vista3d_dinov3 import MedDINOv3Encoder, MedDINOv3VISTA3D, get_config

    print("✓ Model classes imported successfully")

    # Get config
    config = get_config("quick_test")
    print(f"✓ Config loaded: {config['dinov2_backbone']}")

    # Test encoder initialization
    print("\n--- Testing Encoder ---")
    encoder = MedDINOv3Encoder(
        model_name=config["dinov2_backbone"], freeze_backbone=True
    )
    print(
        f"✓ Encoder created with {sum(p.numel() for p in encoder.parameters()):,} parameters"
    )

    # Test full model
    print("\n--- Testing Full Model ---")
    model = MedDINOv3VISTA3D(
        num_classes=config["num_classes"],
        freeze_encoder=True,
        dropout=0.1,
        dinov2_backbone=config["dinov2_backbone"],
    )
    print(
        f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} total parameters"
    )

    # Test forward pass with dummy data
    print("\n--- Testing Forward Pass ---")
    dummy_input = torch.randn(1, 1, 140, 224, 224)  # Batch=1, Channel=1, D,H,W
    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Check for NaN
    if torch.isnan(output).any():
        print("❌ ERROR: Output contains NaN!")
        sys.exit(1)
    else:
        print("✓ Output is valid (no NaN)")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Model is ready for training!")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
