import os
import time
import itertools
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from huggingface_hub import login
import random

# Local Imports
from data import SentinelDataset
from model import SwiftGenerator, SpectralDiscriminator, init_weights
from metrics import GANMetrics 
from loss import GANLoss, PhaseConsistencyLoss, WaveletLoss, LABColorLoss # Switched to LABColorLoss

# ==========================================================
# CONFIGURATION (SWIFT-GAN SOTA TRAINING - FP32 MODE)
# ==========================================================
S1_ROOT = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-1/22588247ff6d53170dfec01c75255b58/"
S2_ROOT = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-2/22588247ff6d53170dfec01c75255b58/"

BATCH_SIZE = 4           
NUM_EPOCHS = 100        
LR = 0.0002              
CHECKPOINT_DIR = "./checkpoints_swift_gan_v2"
OUTPUT_DIR = "./outputs_swift_gan_v2" 

# --- LOSS WEIGHTS (REBALANCED FOR TEXTURE/FID) ---
LAMBDA_GAN = 2.0         # Increased for better Realism
LAMBDA_CYCLE = 5.0       # Decreased to allow texture hallucination
LAMBDA_ID = 2.0          
LAMBDA_PHASE = 2.0       # Structure Lock
LAMBDA_WAVE = 2.0        # Frequency Match
LAMBDA_STYLE = 0.1       # LAB Loss Weight (Lower than Gram Matrix because values are larger)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# HELPER CLASSES
# ==========================================================

class ImagePool():
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def get_scheduler(optimizer, n_epochs, n_epochs_decay):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - n_epochs) / float(n_epochs_decay + 1)
        return lr_l
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

# ==========================================================
# MAIN TRAINING LOOP
# ==========================================================

def train():
    print(f"[INFO] Initializing SWIFT-GAN on {device} (FP32 + LAB Color Loss)...")
    
    # --- FULL METRIC LOGGING SETUP ---
    log_path = os.path.join(OUTPUT_DIR, "swift_metrics.csv")
    if not os.path.isfile(log_path):
        with open(log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch", "Time_sec", 
                "Loss_G", "Loss_D", "Loss_Phase", "Loss_LAB_Style",
                "Cyc_PSNR", "Cyc_SSIM", "Cyc_GSSIM", "Cyc_L1", "Cyc_LPIPS",  # GSSIM included
                "FID", "KID", "IS" 
            ])
    
    print(f"[INFO] Logging metrics to: {log_path}")

    # 1. Dataset
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    
    ds_s1 = SentinelDataset(root_dir=S1_ROOT, domain='s1', transform=train_transform)
    ds_s2 = SentinelDataset(root_dir=S2_ROOT, domain='s2', transform=train_transform)

    # V100 handles 8 workers easily
    loader_s1 = DataLoader(ds_s1, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    loader_s2 = DataLoader(ds_s2, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    
    # 2. Models
    netG_AB = SwiftGenerator(input_nc=3, output_nc=3).to(device) # SAR -> Opt
    netG_BA = SwiftGenerator(input_nc=3, output_nc=3).to(device) # Opt -> SAR
    netD_A = SpectralDiscriminator(input_nc=3).to(device) 
    netD_B = SpectralDiscriminator(input_nc=3).to(device) 

    init_weights(netG_AB)
    init_weights(netG_BA)
    init_weights(netD_A)
    init_weights(netD_B)

    # 3. Losses
    criterion_GAN = GANLoss().to(device)
    criterion_Cycle = nn.L1Loss()     
    criterion_Identity = nn.L1Loss()
    
    # NOVELTY LOSSES
    criterion_Phase = PhaseConsistencyLoss().to(device)
    criterion_Wave = WaveletLoss(device).to(device)
    criterion_Style = LABColorLoss().to(device) # <-- Replaced StyleLoss with LABColorLoss

    # 4. Optimizers
    optimizer_G = optim.Adam(
        itertools.chain(netG_AB.parameters(), netG_BA.parameters()), 
        lr=LR, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        itertools.chain(netD_A.parameters(), netD_B.parameters()),
        lr=LR, betas=(0.5, 0.999)
    )

    # No Scaler needed for FP32

    lr_scheduler_G = get_scheduler(optimizer_G, NUM_EPOCHS//2, NUM_EPOCHS//2)
    lr_scheduler_D = get_scheduler(optimizer_D, NUM_EPOCHS//2, NUM_EPOCHS//2)

    pool_fake_A = ImagePool(50)
    pool_fake_B = ImagePool(50)
    
    metrics = GANMetrics(device=device)

    print(f"[INFO] Config: Batch={BATCH_SIZE} | LAB_Style_Wt={LAMBDA_STYLE} | Phase_Wt={LAMBDA_PHASE}")

    # 5. EPOCH LOOP
    for epoch in range(NUM_EPOCHS):
        metrics.reset_pixel_metrics()
        metrics.reset_distribution_metrics()
        epoch_start = time.time()
        
        for i, (real_A, real_B) in enumerate(zip(loader_s1, loader_s2)):
            real_A = real_A.to(device) # SAR
            real_B = real_B.to(device) # Optical

            # ==================================================
            #  TRAIN GENERATORS (Standard FP32)
            # ==================================================
            optimizer_G.zero_grad()
            
            # --- Identity Loss ---
            id_A = netG_BA(real_A)
            loss_id_A = criterion_Identity(id_A, real_A) * LAMBDA_ID * 0.5
            id_B = netG_AB(real_B)
            loss_id_B = criterion_Identity(id_B, real_B) * LAMBDA_ID * 0.5
            
            # --- GAN Loss ---
            fake_B = netG_AB(real_A) # Generated Optical
            loss_GAN_AB = criterion_GAN(netD_B(fake_B), True) * LAMBDA_GAN

            fake_A = netG_BA(real_B) # Generated SAR
            loss_GAN_BA = criterion_GAN(netD_A(fake_A), True) * LAMBDA_GAN

            # --- Cycle Consistency (Wavelet Enhanced) ---
            rec_A = netG_BA(fake_B)
            loss_cycle_A = criterion_Cycle(rec_A, real_A) * LAMBDA_CYCLE
            loss_cycle_A_Wave = criterion_Wave(rec_A, real_A) * LAMBDA_WAVE 
            
            rec_B = netG_AB(fake_A)
            loss_cycle_B = criterion_Cycle(rec_B, real_B) * LAMBDA_CYCLE
            loss_cycle_B_Wave = criterion_Wave(rec_B, real_B) * LAMBDA_WAVE 

            # --- Phase Consistency ---
            loss_phase_B = criterion_Phase(real_A, fake_B) * LAMBDA_PHASE

            # --- LAB Style Loss (The Atmosphere Fixer) ---
            # Forces Fake_B to match color statistics (A/B channels) of Real_B
            # Ignores structure (L channel)
            loss_style_B = criterion_Style(fake_B, real_B) * LAMBDA_STYLE

            # TOTAL G LOSS
            loss_G = (loss_GAN_AB + loss_GAN_BA + 
                      loss_cycle_A + loss_cycle_B +
                      loss_cycle_A_Wave + loss_cycle_B_Wave +
                      loss_id_A + loss_id_B + 
                      loss_phase_B + 
                      loss_style_B)
            
            loss_G.backward()
            optimizer_G.step()

            # ==================================================
            #  TRAIN DISCRIMINATORS (Standard FP32)
            # ==================================================
            optimizer_D.zero_grad()

            # D_A (Checks SAR)
            fake_A_pool = pool_fake_A.query(fake_A)
            loss_D_A_real = criterion_GAN(netD_A(real_A), True)
            loss_D_A_fake = criterion_GAN(netD_A(fake_A_pool.detach()), False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            loss_D_A.backward()

            # D_B (Checks Optical)
            fake_B_pool = pool_fake_B.query(fake_B)
            loss_D_B_real = criterion_GAN(netD_B(real_B), True)
            loss_D_B_fake = criterion_GAN(netD_B(fake_B_pool.detach()), False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            loss_D_B.backward()

            optimizer_D.step()

            # --- Metrics Collection ---
            with torch.no_grad():
                # Cycle Reconstruction Metrics (Standard + GSSIM)
                metrics.update_pixel_metrics(real_A.detach(), rec_A.detach())
                
                # Realism Metrics (Distribution)
                metrics.update_distribution_metrics(real_B.detach(), fake_B.detach())

            if i % 100 == 0:
                print(f"[Ep {epoch+1}/{NUM_EPOCHS}] [Batch {i}] Loss_G: {loss_G.item():.4f} | LAB_Style: {loss_style_B.item():.4f}")

        # Step Schedulers
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        
        # --- COMPUTE & LOG METRICS ---
        time_ep = time.time() - epoch_start
        
        # 1. Pixel Metrics
        pix = metrics.get_pixel_metrics()
        
        # 2. Distribution Metrics
        dist = metrics.compute_distribution_metrics()
        
        # Console Report
        print(f"\n" + "="*85)
        print(f"EPOCH {epoch+1} FINISHED | Time: {time_ep:.0f}s")
        print(f"LOSSES    >> G: {loss_G.item():.4f} | Phase: {loss_phase_B.item():.4f} | LAB_Style: {loss_style_B.item():.4f}")
        print("-" * 85)
        print(f"CYCLE (Reconst):")
        print(f"   PSNR : {pix.get('PSNR', 0):.2f}")
        print(f"   SSIM : {pix.get('SSIM', 0):.4f}")
        print(f"   GSSIM: {pix.get('GSSIM', 0):.4f}  <-- Gradient Similarity Included")
        print(f"   MAE  : {pix.get('MAE (L1)', 0):.4f}")
        print(f"   LPIPS: {pix.get('LPIPS', 0):.4f}")
        print("-" * 85)
        print(f"REALISM (Dist):")
        print(f"   FID  : {dist.get('FID', 0):.2f}")
        print(f"   KID  : {dist.get('KID', 0):.5f}")
        print(f"   IS   : {dist.get('IS', 0):.2f}")
        print("="*85 + "\n")

        # CSV Logging
        with open(log_path, "a", newline='') as f:
            csv.writer(f).writerow([
                epoch + 1, int(time_ep), 
                f"{loss_G.item():.4f}", f"{(loss_D_A+loss_D_B).item():.4f}", 
                f"{loss_phase_B.item():.4f}", f"{loss_style_B.item():.4f}",
                f"{pix.get('PSNR', 0):.4f}", 
                f"{pix.get('SSIM', 0):.4f}", 
                f"{pix.get('GSSIM', 0):.4f}", 
                f"{pix.get('MAE (L1)', 0):.4f}", 
                f"{pix.get('LPIPS', 0):.4f}", 
                f"{dist.get('FID', 0):.4f}", 
                f"{dist.get('KID', 0):.5f}", 
                f"{dist.get('IS', 0):.4f}"
            ])

        # Save Visuals
        with torch.no_grad():
            vis = torch.stack([
                denormalize(real_A[0].cpu()), # Input SAR
                denormalize(fake_B[0].cpu()), # Generated Opt
                denormalize(rec_A[0].cpu()),  # Reconstructed SAR
                denormalize(real_B[0].cpu())  # Real Opt (Unpaired Ref)
            ])
            save_image(vis, os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}_swift.png"))

        # Save Checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(netG_AB.state_dict(), os.path.join(CHECKPOINT_DIR, f"netG_AB_ep{epoch+1}.pth"))
            torch.save(netG_BA.state_dict(), os.path.join(CHECKPOINT_DIR, f"netG_BA_ep{epoch+1}.pth"))
            torch.save(netD_A.state_dict(), os.path.join(CHECKPOINT_DIR, f"netD_A_ep{epoch+1}.pth"))
            torch.save(netD_B.state_dict(), os.path.join(CHECKPOINT_DIR, f"netD_B_ep{epoch+1}.pth"))

    print(f"[INFO] SWIFT-GAN Training Finished.")

if __name__ == "__main__":
    train()
