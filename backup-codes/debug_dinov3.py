import os
import sys
import torch
from pathlib import Path

# Add dinov3 to path
dinov3_path = Path("./dinov3").resolve()
if dinov3_path.exists():
    print(f"Adding {dinov3_path} to sys.path")
    sys.path.append(str(dinov3_path))
else:
    print(f"ERROR: {dinov3_path} does not exist!")
    sys.exit(1)

def check_weights(path):
    p = Path(path)
    if not p.exists():
        print(f"Weights file not found: {p}")
        return False
    
    size_mb = p.stat().st_size / (1024 * 1024)
    print(f"Weights file: {p}")
    print(f"Size: {size_mb:.2f} MB")
    
    if size_mb < 1:
        print("WARNING: File is too small! Likely corrupt or HTML error page.")
        return False
    if size_mb > 1000:
        print("Note: File is > 1GB. Ensure you have enough RAM.")
        
    return True

def test_direct_load(weights_path=None):
    print("\n--- Testing Direct Import ---")
    try:
        from dinov3.hub.backbones import dinov3_vitb16
        print("Import successful!")
    except ImportError as e:
        print(f"Import failed: {e}")
        return

    print("Initializing model (CPU)...")
    try:
        # Load directly
        if weights_path:
            # If function accepts weights/pretrained arg
            # Inspecting hubconf suggests it might just return the model
            model = dinov3_vitb16(pretrained=True, weights=weights_path) 
        else:
            model = dinov3_vitb16(pretrained=False) # Helper for random init
            
        print(f"Model created. Params: {sum(p.numel() for p in model.parameters())}")
        
        if torch.cuda.is_available():
            print("Moving to CUDA...")
            model = model.cuda()
            print("Success!")
        else:
            print("CUDA not available, skipping GPU move.")
            
    except Exception as e:
        print(f"CRASH during load: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Check for weights in common locations
    possible_weights = [
        "./dinov3_weights/dinov3_vitb16.pth",
        "./checkpoints/dinov3_vitb16.pth",
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" 
    ]
    
    weights_found = None
    for w in possible_weights:
        if Path(w).exists():
            weights_found = w
            break
            
    if weights_found:
        if check_weights(weights_found):
            test_direct_load(weights_found)
    else:
        print("No local weights found to test. Testing random init...")
        test_direct_load(None)
