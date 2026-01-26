import os
import torch
import numpy as np
import csv
import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Yerel dosyalarından importlar
from data import SentinelDataset 
from model import SwiftGenerator
from metrics import GANMetrics

# ==========================================================
# 1. PATH CONFIGURATION (Senin Belirttiğin Pathler)
# ==========================================================
S1_ROOT = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-1/22588247ff6d53170dfec01c75255b58/"
S2_ROOT = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-2/22588247ff6d53170dfec01c75255b58/"

# Checkpoint yolları
CHECKPOINT_AB = "./checkpoints_swift_no_wavelet/netG_AB_ep100.pth" # SAR -> Opt
CHECKPOINT_BA = "./checkpoints_swift_no_wavelet/netG_BA_ep100.pth" # Opt -> SAR

OUTPUT_EVAL_DIR = "./final_evaluation_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

# ==========================================================
# 2. EVALUATION ENGINE
# ==========================================================
def run_evaluation():
    print(f"[INFO] Initializing Evaluation on {DEVICE}...")

    # 1. Modelleri Yükle
    netG_AB = SwiftGenerator(input_nc=3, output_nc=3).to(DEVICE)
    netG_BA = SwiftGenerator(input_nc=3, output_nc=3).to(DEVICE)
    
    netG_AB.load_state_dict(torch.load(CHECKPOINT_AB, map_location=DEVICE))
    netG_BA.load_state_dict(torch.load(CHECKPOINT_BA, map_location=DEVICE))
    netG_AB.eval()
    netG_BA.eval()

    # 2. Dataset Hazırla (Paired veri okuması için cache kullanmıyoruz)
    # Senin data.py sınıfın S1 ve S2 için ayrı ayrı patch index oluşturuyor.
    ds_test_s1 = SentinelDataset(root_dir=S1_ROOT, domain='s1', patch_size=256, cache_index=False)
    ds_test_s2 = SentinelDataset(root_dir=S2_ROOT, domain='s2', patch_size=256, cache_index=False)
    
    # Paired test set için DataLoader'lar
    loader_s1 = DataLoader(ds_test_s1, batch_size=1, shuffle=False, num_workers=4)
    loader_s2 = DataLoader(ds_test_s2, batch_size=1, shuffle=False, num_workers=4)

    # 3. Metrik Nesneleri
    metrics_struct = GANMetrics(device=DEVICE)   # SAR -> Opt -> SAR
    metrics_realism = GANMetrics(device=DEVICE)  # Generated Opt vs Real Opt

    # En küçük veri kümesi kadar iterasyon yap (Paired veri garantisi için)
    test_len = min(len(ds_test_s1), len(ds_test_s2))
    print(f"[INFO] Evaluating on {test_len} samples...")

    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(tqdm.tqdm(zip(loader_s1, loader_s2), total=test_len)):
            if i >= test_len: break
            
            real_A = real_A.to(DEVICE) 
            real_B = real_B.to(DEVICE)

            # --- Forward: SAR -> Opt ---
            fake_B = netG_AB(real_A)
            
            # --- Backward: Opt -> SAR ---
            rec_A = netG_BA(fake_B)

            # --- Realism Metrikleri (Optik vs Optik) ---
            metrics_realism.update_pixel_metrics(real_B.detach(), fake_B.detach())
            metrics_realism.update_distribution_metrics(real_B.detach(), fake_B.detach())

            # --- Structural Metrikler (SAR vs Rec-SAR) ---
            metrics_struct.update_pixel_metrics(real_A.detach(), rec_A.detach())
            
            if i % 100 == 0:
                # Görsel: [Input SAR | Fake Opt | Real Opt]
                vis = torch.cat([denormalize(real_A), denormalize(fake_B), denormalize(real_B)], 3)
                save_image(vis, os.path.join(OUTPUT_EVAL_DIR, f"eval_vis_{i}.png"))

    # Sonuç Raporu
    struct_res = metrics_struct.get_pixel_metrics()
    realism_pix = metrics_realism.get_pixel_metrics()
    realism_dist = metrics_realism.compute_distribution_metrics()

    print("\n" + "="*85)
    print(f"{'METRIC':<15} | {'STRUCTURAL (SAR-SAR)':<25} | {'REALISM (OPT-OPT)':<25}")
    print("-" * 85)
    for m in ['PSNR', 'SSIM', 'GSSIM', 'MAE (L1)', 'LPIPS']:
        print(f"{m:<15} | {struct_res[m]:<25.4f} | {realism_pix[m]:<25.4f}")
    print("-" * 85)
    print(f"{'FID':<15} | {'-':<25} | {realism_dist['FID']:<25.4f}")
    print(f"{'KID':<15} | {'-':<25} | {realism_dist['KID']:<25.5f}")
    print(f"{'IS':<15} | {'-':<25} | {realism_dist['IS']:<25.4f}")
    print("="*85)

if __name__ == "__main__":
    run_evaluation()