
import os
import torch
import timm
import sys
import gc

# Force cleanup
gc.collect()
torch.cuda.empty_cache()

print(f"PyTorch Version: {torch.__version__}")
print(f"Timm Version: {timm.__version__}")

model_name = "vit_base_patch16_dinov3.lvd1689m"
print(f"Attempting to load {model_name}...")

try:
    # Try with pretrained=False first to check definition
    print("1. Creating model definition (no weights)...")
    model = timm.create_model(model_name, pretrained=False)
    print("Success! Model definition works.")
    del model
    gc.collect()

    # Try loading weights
    print("2. Loading with weights...")
    model = timm.create_model(model_name, pretrained=True)
    print("Success! Weights loaded.")
    
    print("3. Checking features extraction...")
    features = model.forward_features(torch.randn(1, 3, 224, 224))
    print(f"Features shape: {features.shape}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
