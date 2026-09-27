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
import random

# Local Imports
from data import FolderDataset               # <-- benchmark RGB folder dataset
from model import SwiftGenerator, SpectralDiscriminator, init_weights
from metrics import GANMetrics               # <-- UNCHANGED metrics
from loss import GANLoss, PhaseConsistencyLoss, WaveletLoss, LABColorLoss

# ==========================================================
# CONFIGURATION (read from environment so one script serves all benchmarks)
# ==========================================================
# Example:
#   BENCH_ROOT=.../benchmark_data/horse2zebra RUN_NAME=horse2zebra python train_benchmark.py
#
# Domain A = trainA/testA, Domain B = trainB/testB (standard CycleGAN layout).
BENCH_ROOT = os.environ.get("BENCH_ROOT", "./benchmark_data/horse2zebra")
RUN_NAME   = os.environ.get("RUN_NAME", os.path.basename(BENCH_ROOT.rstrip("/")))

TRAIN_A = os.path.join(BENCH_ROOT, "trainA")
TRAIN_B = os.path.join(BENCH_ROOT, "trainB")

# SWAP_AB=1 ise domain A ve B ters bağlanır. Bu, benchmark'ın klasör yönü
# tablodaki yönle uyuşmadığında gerekir.
#   maps: klasörde trainA=uydu, trainB=harita.  A->B = Satellite->Map olur.
#   Tablo "Map->Satellite" istediği için SWAP_AB=1 ile A=harita, B=uydu yaparız,
#   böylece A->B = Map->Satellite (fake_B = uydu) doğru yönde ölçülür.
# horse2zebra ve summer2winter zaten tablo yönünde, onlarda SWAP_AB kullanma.
SWAP_AB = os.environ.get("SWAP_AB", "0") == "1"
if SWAP_AB:
    TRAIN_A, TRAIN_B = TRAIN_B, TRAIN_A
    print(f"[INFO] SWAP_AB=1 -> domain A/B ters baglandi (A<-trainB, B<-trainA)")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "100"))
LR         = float(os.environ.get("LR", "0.0002"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "256"))

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", f"./checkpoints_{RUN_NAME}")
OUTPUT_DIR     = os.environ.get("OUTPUT_DIR",     f"./outputs_{RUN_NAME}")

# --- LOSS WEIGHTS (same defaults as SAR training; overridable via env) ---
LAMBDA_GAN   = float(os.environ.get("LAMBDA_GAN",   "1.0"))
LAMBDA_CYCLE = float(os.environ.get("LAMBDA_CYCLE", "10.0"))
LAMBDA_ID    = float(os.environ.get("LAMBDA_ID",    "5.0"))
LAMBDA_PHASE = float(os.environ.get("LAMBDA_PHASE", "1.0"))
LAMBDA_WAVE  = float(os.environ.get("LAMBDA_WAVE",  "1.0"))
LAMBDA_STYLE = float(os.environ.get("LAMBDA_STYLE", "0.1"))

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# HELPER CLASSES (identical to train.py)
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
    print(f"[INFO] SWIFT-GAN benchmark run '{RUN_NAME}' on {device}")
    print(f"[INFO] A={TRAIN_A}\n[INFO] B={TRAIN_B}")

    log_path = os.path.join(OUTPUT_DIR, f"metrics_{RUN_NAME}.csv")
    if not os.path.isfile(log_path):
        with open(log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch", "Time_sec",
                "Loss_G", "Loss_D", "Loss_Phase", "Loss_LAB_Style",
                "Cyc_PSNR", "Cyc_SSIM", "Cyc_GSSIM", "Cyc_L1", "Cyc_LPIPS",
                "Dir_PSNR", "Dir_SSIM", "Dir_GSSIM", "Dir_L1", "Dir_LPIPS",
                "FID", "KID", "IS"
            ])
    print(f"[INFO] Logging metrics to: {log_path}")

    # 1. Datasets (one per domain) — tensor-level augmentation, same as SAR run
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    ds_A = FolderDataset(TRAIN_A, patch_size=PATCH_SIZE, transform=train_transform)
    ds_B = FolderDataset(TRAIN_B, patch_size=PATCH_SIZE, transform=train_transform)

    print(f"[INFO] Domain A images: {len(ds_A)} | Domain B images: {len(ds_B)}")
    if len(ds_A) == 0 or len(ds_B) == 0:
        raise RuntimeError(
            f"Benchmark dataset empty (A={len(ds_A)}, B={len(ds_B)}). "
            f"Check trainA/trainB under {BENCH_ROOT}"
        )

    loader_A = DataLoader(ds_A, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    loader_B = DataLoader(ds_B, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # 2. Models (identical architecture, 3->3 channels)
    netG_AB = SwiftGenerator(input_nc=3, output_nc=3).to(device)   # A -> B
    netG_BA = SwiftGenerator(input_nc=3, output_nc=3).to(device)   # B -> A
    netD_A = SpectralDiscriminator(input_nc=3).to(device)
    netD_B = SpectralDiscriminator(input_nc=3).to(device)

    init_weights(netG_AB); init_weights(netG_BA)
    init_weights(netD_A);  init_weights(netD_B)

    # 3. Losses (unchanged)
    criterion_GAN = GANLoss().to(device)
    criterion_Cycle = nn.L1Loss()
    criterion_Identity = nn.L1Loss()
    criterion_Phase = PhaseConsistencyLoss().to(device)
    criterion_Wave = WaveletLoss(device).to(device)
    criterion_Style = LABColorLoss().to(device)

    # 4. Optimizers
    optimizer_G = optim.Adam(
        itertools.chain(netG_AB.parameters(), netG_BA.parameters()),
        lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(
        itertools.chain(netD_A.parameters(), netD_B.parameters()),
        lr=LR, betas=(0.5, 0.999))

    lr_scheduler_G = get_scheduler(optimizer_G, NUM_EPOCHS // 2, NUM_EPOCHS // 2)
    lr_scheduler_D = get_scheduler(optimizer_D, NUM_EPOCHS // 2, NUM_EPOCHS // 2)

    pool_fake_A = ImagePool(50)
    pool_fake_B = ImagePool(50)

    metrics_cycle = GANMetrics(device=device)
    metrics_direct = GANMetrics(device=device)

    best_fid = float("inf")

    # 5. EPOCH LOOP
    for epoch in range(NUM_EPOCHS):
        metrics_cycle.reset_pixel_metrics()
        metrics_direct.reset_pixel_metrics()
        metrics_direct.reset_distribution_metrics()
        epoch_start = time.time()

        for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # ---- Generators ----
            optimizer_G.zero_grad()

            id_A = netG_BA(real_A)
            loss_id_A = criterion_Identity(id_A, real_A) * LAMBDA_ID * 0.5
            id_B = netG_AB(real_B)
            loss_id_B = criterion_Identity(id_B, real_B) * LAMBDA_ID * 0.5

            fake_B = netG_AB(real_A)
            loss_GAN_AB = criterion_GAN(netD_B(fake_B), True) * LAMBDA_GAN
            fake_A = netG_BA(real_B)
            loss_GAN_BA = criterion_GAN(netD_A(fake_A), True) * LAMBDA_GAN

            rec_A = netG_BA(fake_B)
            loss_cycle_A = criterion_Cycle(rec_A, real_A) * LAMBDA_CYCLE
            loss_cycle_A_Wave = criterion_Wave(rec_A, real_A) * LAMBDA_WAVE
            rec_B = netG_AB(fake_A)
            loss_cycle_B = criterion_Cycle(rec_B, real_B) * LAMBDA_CYCLE
            loss_cycle_B_Wave = criterion_Wave(rec_B, real_B) * LAMBDA_WAVE

            loss_phase_B = (criterion_Phase(rec_A, real_A) +
                            criterion_Phase(rec_B, real_B)) * 0.5 * LAMBDA_PHASE

            loss_style_B = criterion_Style(fake_B, real_B) * LAMBDA_STYLE

            loss_G = (loss_GAN_AB + loss_GAN_BA +
                      loss_cycle_A + loss_cycle_B +
                      loss_cycle_A_Wave + loss_cycle_B_Wave +
                      loss_id_A + loss_id_B +
                      loss_phase_B + loss_style_B)

            loss_G.backward()
            optimizer_G.step()

            # ---- Discriminators ----
            optimizer_D.zero_grad()

            fake_A_pool = pool_fake_A.query(fake_A)
            loss_D_A_real = criterion_GAN(netD_A(real_A), True)
            loss_D_A_fake = criterion_GAN(netD_A(fake_A_pool.detach()), False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            loss_D_A.backward()

            fake_B_pool = pool_fake_B.query(fake_B)
            loss_D_B_real = criterion_GAN(netD_B(real_B), True)
            loss_D_B_fake = criterion_GAN(netD_B(fake_B_pool.detach()), False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            loss_D_B.backward()

            optimizer_D.step()

            # ---- Metrics (unchanged calls) ----
            with torch.no_grad():
                metrics_cycle.update_pixel_metrics(real_A.detach(), rec_A.detach())
                metrics_direct.update_pixel_metrics(real_B.detach(), fake_B.detach())
                metrics_direct.update_distribution_metrics(real_B.detach(), fake_B.detach())

            if i % 100 == 0:
                print(f"[{RUN_NAME}][Ep {epoch+1}/{NUM_EPOCHS}][B {i}] "
                      f"Loss_G: {loss_G.item():.4f}")

        lr_scheduler_G.step()
        lr_scheduler_D.step()

        time_ep = time.time() - epoch_start
        pix_cycle = metrics_cycle.get_pixel_metrics()
        pix_direct = metrics_direct.get_pixel_metrics()
        dist = metrics_direct.compute_distribution_metrics()

        print("\n" + "=" * 85)
        print(f"[{RUN_NAME}] EPOCH {epoch+1} | Time {time_ep:.0f}s")
        print(f"DIRECT  SSIM {pix_direct.get('SSIM',0):.4f} | "
              f"GSSIM {pix_direct.get('GSSIM',0):.4f} | "
              f"MAE {pix_direct.get('MAE (L1)',0):.4f} | "
              f"LPIPS {pix_direct.get('LPIPS',0):.4f}")
        print(f"REALISM FID {dist.get('FID',0):.2f} | "
              f"KID {dist.get('KID',0):.5f} | IS {dist.get('IS',0):.2f}")
        print("=" * 85 + "\n")

        with open(log_path, "a", newline='') as f:
            csv.writer(f).writerow([
                epoch + 1, int(time_ep),
                f"{loss_G.item():.4f}", f"{(loss_D_A+loss_D_B).item():.4f}",
                f"{loss_phase_B.item():.4f}", f"{loss_style_B.item():.4f}",
                f"{pix_cycle.get('PSNR',0):.4f}", f"{pix_cycle.get('SSIM',0):.4f}",
                f"{pix_cycle.get('GSSIM',0):.4f}", f"{pix_cycle.get('MAE (L1)',0):.4f}",
                f"{pix_cycle.get('LPIPS',0):.4f}",
                f"{pix_direct.get('PSNR',0):.4f}", f"{pix_direct.get('SSIM',0):.4f}",
                f"{pix_direct.get('GSSIM',0):.4f}", f"{pix_direct.get('MAE (L1)',0):.4f}",
                f"{pix_direct.get('LPIPS',0):.4f}",
                f"{dist.get('FID',0):.4f}", f"{dist.get('KID',0):.5f}",
                f"{dist.get('IS',0):.4f}"
            ])

        # Save visuals (A -> fake_B -> rec_A, and real_B ref)
        with torch.no_grad():
            vis = torch.stack([
                denormalize(real_A[0].cpu()),
                denormalize(fake_B[0].cpu()),
                denormalize(rec_A[0].cpu()),
                denormalize(real_B[0].cpu())
            ])
            save_image(vis, os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}_{RUN_NAME}.png"))

        # Save best-FID checkpoint + periodic
        cur_fid = dist.get('FID', 0)
        if cur_fid > 0 and cur_fid < best_fid:
            best_fid = cur_fid
            torch.save(netG_AB.state_dict(), os.path.join(CHECKPOINT_DIR, "netG_AB_best.pth"))
            torch.save(netG_BA.state_dict(), os.path.join(CHECKPOINT_DIR, "netG_BA_best.pth"))
            print(f"[{RUN_NAME}] New best FID {best_fid:.2f} (epoch {epoch+1}) checkpoint saved.")

        if (epoch + 1) % 10 == 0:
            torch.save(netG_AB.state_dict(), os.path.join(CHECKPOINT_DIR, f"netG_AB_ep{epoch+1}.pth"))
            torch.save(netG_BA.state_dict(), os.path.join(CHECKPOINT_DIR, f"netG_BA_ep{epoch+1}.pth"))

    print(f"[INFO] '{RUN_NAME}' finished. Best FID: {best_fid:.2f}")


if __name__ == "__main__":
    train()