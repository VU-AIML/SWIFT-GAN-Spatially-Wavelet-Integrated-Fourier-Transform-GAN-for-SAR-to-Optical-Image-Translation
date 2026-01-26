import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import sys

# --- LOCAL IMPORTS ---
sys.path.append('./src')
from model import SwiftGenerator 

# ==========================================================
# 1. CONFIGURATION
# ==========================================================
VV_PATH = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-1/22588247ff6d53170dfec01c75255b58/S1A_IW_GRDH_1SDV_20240104T043534_20240104T043559_051952_0646F6_FEFD_VV.tif"
VH_PATH = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-1/22588247ff6d53170dfec01c75255b58/S1A_IW_GRDH_1SDV_20240104T043534_20240104T043559_051952_0646F6_FEFD_VH.tif"

CHECKPOINT_PATH = "./checkpoints_swift_gan_v2/netG_AB_ep100.pth"
OUTPUT_DIR = "./output_inference_swift_gan_v2_sliding_window_results"

PATCH_SIZE = 256
STRIDE = 256 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 2. MODEL LOADING
# ==========================================================
def load_model(checkpoint_path, device):
    print(f"[INFO] Loading SWIFT-GAN Generator from {checkpoint_path}...")
    netG = SwiftGenerator(input_nc=3, output_nc=3).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        netG.load_state_dict(checkpoint['model_state_dict'])
    else:
        netG.load_state_dict(checkpoint)
        
    netG.eval() 
    return netG

# ==========================================================
# 3. DATA PREPROCESSING (EXACT MATCH WITH data.py)
# ==========================================================
def _normalize_s1(data):
    """
    Robust percentile normalization for S1.
    Matches your training logic.
    """
    valid_data = data[~np.isnan(data)]
    if valid_data.size == 0 or np.max(valid_data) == 0:
        return np.zeros_like(data)
        
    p2, p98 = np.percentile(valid_data, (2, 98))
    stretched = np.clip(data, p2, p98)
    
    if p98 - p2 == 0: 
        return np.zeros_like(data)
        
    return (stretched - p2) / (p98 - p2)

def preprocess_dual_pol_patch(vv_patch, vh_patch):
    """
    Constructs the 3-channel input exactly as done in training.
    Channels: [VV_norm, VH_norm, Ratio_norm]
    """
    # 1. Normalize VV and VH independently
    vv_norm = _normalize_s1(vv_patch)
    vh_norm = _normalize_s1(vh_patch)
    
    # 2. Calculate Ratio: Normalized VH / Normalized VV
    ratio = np.divide(vh_norm, vv_norm + 1e-6)
    ratio_norm = np.clip(ratio, 0, 1) 
    
    # 3. Stack channels [VV, VH, Ratio]
    # This creates the RGB-like structure (R=VV, G=VH, B=Ratio)
    img_stacked = np.stack([vv_norm, vh_norm, ratio_norm], axis=0)
    
    # 4. Convert to Tensor and Normalize to [-1, 1]
    tensor = torch.from_numpy(img_stacked).float()
    tensor = (tensor - 0.5) / 0.5 
    
    return tensor.unsqueeze(0) 

def denormalize_image(tensor):
    """
    Converts any model tensor (Input or Output) from [-1, 1] to [0, 1] RGB
    """
    img = tensor.cpu().detach().numpy()
    img = (img * 0.5 + 0.5) 
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0)) 
    return img

# ==========================================================
# 4. SLIDING WINDOW LOGIC
# ==========================================================
def run_sliding_window_inference(netG, vv_path, vh_path):
    print(f"[INFO] Opening SAR files...")
    
    with rasterio.open(vv_path) as src_vv, rasterio.open(vh_path) as src_vh:
        width = src_vv.width
        height = src_vv.height
        
        if width != src_vh.width or height != src_vh.height:
            print("[ERROR] VV and VH dimensions do not match!")
            return

        print(f"[INFO] Image Dimensions: {width}x{height}")
        patch_counter = 0
        
        for y in range(0, height, STRIDE):
            for x in range(0, width, STRIDE):
                
                if x + PATCH_SIZE > width or y + PATCH_SIZE > height:
                    continue
                
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                vv_patch = src_vv.read(1, window=window)
                vh_patch = src_vh.read(1, window=window)
                
                if np.max(vv_patch) < 1e-4:
                    continue

                input_tensor = preprocess_dual_pol_patch(vv_patch, vh_patch).to(DEVICE)
                
                with torch.no_grad():
                    fake_optical = netG(input_tensor)
                
                # --- VISUALIZATION FIX ---
                # We visualize the Input Tensor directly as RGB.
                # Since input is [VV, VH, Ratio], it maps to [R, G, B].
                sar_rgb_vis = denormalize_image(input_tensor[0])
                
                # Optical output
                img_opt_vis = denormalize_image(fake_optical[0])
                
                save_patch_result(sar_rgb_vis, img_opt_vis, patch_counter, x, y)
                patch_counter += 1
                
                if patch_counter % 10 == 0:
                    print(f"[INFO] Processed {patch_counter} patches...", end='\r')

def save_patch_result(sar_rgb, fake_opt, idx, x, y):
    plt.figure(figsize=(10, 5))
    
    # Left: Input SAR (RGB Composite)
    plt.subplot(1, 2, 1)
    plt.imshow(sar_rgb) 
    plt.title(f"Input SAR (RGB Composite)\nPos: x={x}, y={y}")
    plt.axis('off')
    
    # Right: Generated Optical
    plt.subplot(1, 2, 2)
    plt.imshow(fake_opt)
    plt.title("Generated Optical")
    plt.axis('off')
    
    plt.tight_layout()
    filename = f"inference_{idx:04d}_x{x}_y{y}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()

# ==========================================================
# 5. MAIN
# ==========================================================
if __name__ == "__main__":
    try:
        generator = load_model(CHECKPOINT_PATH, DEVICE)
        
        if not os.path.exists(VV_PATH) or not os.path.exists(VH_PATH):
            print("[ERROR] Check file paths.")
        else:
            run_sliding_window_inference(generator, VV_PATH, VH_PATH)
            print(f"\n[SUCCESS] Done. Results saved to {OUTPUT_DIR}")
            
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    except Exception as e:
        print(f"\n[ERROR] {e}")