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
from data import SentinelDataset
from model import SwiftGenerator, SpectralDiscriminator, init_weights
from metrics import GANMetrics
from loss import GANLoss, PhaseConsistencyLoss, WaveletLoss, LABColorLoss
from spectral_energy_loss import SpectralEnergyLoss

# ==========================================================
# CONFIG FROM ENVIRONMENT (each SLURM array task sets these)
# ==========================================================
def _f(name, default):
    return float(os.environ.get(name, default))

def _i(name, default):
    return int(os.environ.get(name, default))

def _b(name, default):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")

RUN_NAME = os.environ.get("RUN_NAME", "run")

S1_ROOT = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-1"
S2_ROOT = "/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/data/Sentinel-2"

BATCH_SIZE = _i("BATCH_SIZE", 2)
NUM_EPOCHS = _i("NUM_EPOCHS", 100)
LR = _f("LR", 0.0002)

# Separate dirs per run so parallel jobs never collide
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", f"./checkpoints_{RUN_NAME}")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"./outputs_{RUN_NAME}")

# --- CLASSIC LOSS WEIGHTS (env-overridable for ablations) ---
LAMBDA_GAN = _f("LAMBDA_GAN", 1.0)
LAMBDA_CYCLE = _f("LAMBDA_CYCLE", 10.0)
LAMBDA_ID = _f("LAMBDA_ID", 5.0)
LAMBDA_PHASE = _f("LAMBDA_PHASE", 1.0)     # old PhaseConsistencyLoss
LAMBDA_WAVE = _f("LAMBDA_WAVE", 1.0)
LAMBDA_STYLE = _f("LAMBDA_STYLE", 0.1)

# --- SPECTRAL ENERGY LOSS WEIGHTS (novelty; 0 = off) ---
LAMBDA_SPEC_ENERGY = _f("LAMBDA_SPEC_ENERGY", 0.0)
LAMBDA_PHASE_ENERGY = _f("LAMBDA_PHASE_ENERGY", 0.0)
SPEC_ENERGY_BINS = _i("SPEC_ENERGY_BINS", 64)
SPEC_ENERGY_BUFFER = _i("SPEC_ENERGY_BUFFER", 64)

# --- ARCHITECTURE ABLATION ---
USE_FFT = _b("USE_FFT", True)  # False => "w/o FFT" ablation

# --- EARLY STOPPING (on FID) ---
ES_PATIENCE = _i("ES_PATIENCE", 15)
ES_MIN_EPOCHS = _i("ES_MIN_EPOCHS", 25)
ES_MIN_DELTA = _f("ES_MIN_DELTA", 0.5)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# HELPERS
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
        return torch.cat(return_images, 0)


def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def get_scheduler(optimizer, n_epochs, n_epochs_decay):
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch + 1 - n_epochs) / float(n_epochs_decay + 1)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


# ==========================================================
# MAIN
# ==========================================================
def train():
    print("=" * 70)
    print(f"[RUN] {RUN_NAME}")
    print(f"  GAN={LAMBDA_GAN} CYCLE={LAMBDA_CYCLE} ID={LAMBDA_ID} "
          f"PHASE={LAMBDA_PHASE} WAVE={LAMBDA_WAVE} STYLE={LAMBDA_STYLE}")
    print(f"  SPEC_ENERGY={LAMBDA_SPEC_ENERGY} PHASE_ENERGY={LAMBDA_PHASE_ENERGY} "
          f"USE_FFT={USE_FFT}")
    print(f"  early-stop: patience={ES_PATIENCE} min_epochs={ES_MIN_EPOCHS} "
          f"min_delta={ES_MIN_DELTA} (metric=FID)")
    print(f"  ckpt={CHECKPOINT_DIR} out={OUTPUT_DIR}")
    print("=" * 70, flush=True)

    if LAMBDA_PHASE_ENERGY > 0 and LAMBDA_PHASE > 0:
        print("[WARN] Both LAMBDA_PHASE (old) and LAMBDA_PHASE_ENERGY > 0. "
              "Phase penalized twice; set LAMBDA_PHASE=0 when using energy phase.")

    log_path = os.path.join(OUTPUT_DIR, f"metrics_{RUN_NAME}.csv")
    if not os.path.isfile(log_path):
        with open(log_path, "w", newline='') as f:
            csv.writer(f).writerow([
                "Epoch", "Time_sec", "Loss_G", "Loss_D",
                "Loss_Phase", "Loss_LAB_Style", "Loss_SpecED", "Loss_PhaseEnergy",
                "Cyc_PSNR", "Cyc_SSIM", "Cyc_GSSIM", "Cyc_L1", "Cyc_LPIPS",
                "Dir_PSNR", "Dir_SSIM", "Dir_GSSIM", "Dir_L1", "Dir_LPIPS",
                "FID", "KID", "IS", "Best_FID"
            ])

    # --- Data ---
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    # cache_index=True so the 6 parallel jobs reuse one index scan instead of
    # each re-scanning the whole dataset (they share the same .npy on disk).
    ds_s1 = SentinelDataset(root_dir=S1_ROOT, domain='s1', transform=train_transform, cache_index=True)
    ds_s2 = SentinelDataset(root_dir=S2_ROOT, domain='s2', transform=train_transform, cache_index=True)
    print(f"[INFO] S1 patches: {len(ds_s1)} | S2 patches: {len(ds_s2)}", flush=True)
    if len(ds_s1) == 0 or len(ds_s2) == 0:
        raise RuntimeError(f"Empty dataset (S1={len(ds_s1)}, S2={len(ds_s2)}).")

    loader_s1 = DataLoader(ds_s1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    loader_s2 = DataLoader(ds_s2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    # --- Models ---
    netG_AB = SwiftGenerator(input_nc=3, output_nc=3, use_fft=USE_FFT).to(device)
    netG_BA = SwiftGenerator(input_nc=3, output_nc=3, use_fft=USE_FFT).to(device)
    netD_A = SpectralDiscriminator(input_nc=3).to(device)
    netD_B = SpectralDiscriminator(input_nc=3).to(device)
    for net in [netG_AB, netG_BA, netD_A, netD_B]:
        init_weights(net)

    # --- Losses ---
    criterion_GAN = GANLoss().to(device)
    criterion_Cycle = nn.L1Loss()
    criterion_Identity = nn.L1Loss()
    criterion_Phase = PhaseConsistencyLoss().to(device)
    criterion_Wave = WaveletLoss(device).to(device)
    criterion_Style = LABColorLoss().to(device)
    criterion_SpecEnergy = SpectralEnergyLoss(
        n_bins=SPEC_ENERGY_BINS, buffer_size=SPEC_ENERGY_BUFFER,
        w_phase=LAMBDA_PHASE_ENERGY, w_spec_ed=LAMBDA_SPEC_ENERGY,
    ).to(device)

    # --- Optimizers ---
    optimizer_G = optim.Adam(itertools.chain(netG_AB.parameters(), netG_BA.parameters()),
                             lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()),
                             lr=LR, betas=(0.5, 0.999))
    lr_scheduler_G = get_scheduler(optimizer_G, NUM_EPOCHS // 2, NUM_EPOCHS // 2)
    lr_scheduler_D = get_scheduler(optimizer_D, NUM_EPOCHS // 2, NUM_EPOCHS // 2)

    pool_fake_A = ImagePool(50)
    pool_fake_B = ImagePool(50)
    metrics_cycle = GANMetrics(device=device)
    metrics_direct = GANMetrics(device=device)

    best_fid = float('inf')
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        metrics_cycle.reset_pixel_metrics()
        metrics_direct.reset_pixel_metrics()
        metrics_direct.reset_distribution_metrics()
        epoch_start = time.time()
        run_spec_ed = run_phase_energy = 0.0
        n_spec = 0

        for i, (real_A, real_B) in enumerate(zip(loader_s1, loader_s2)):
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
            rec_B = netG_AB(fake_A)
            loss_cycle_B = criterion_Cycle(rec_B, real_B) * LAMBDA_CYCLE

            if LAMBDA_WAVE > 0:
                loss_cycle_A_Wave = criterion_Wave(rec_A, real_A) * LAMBDA_WAVE
                loss_cycle_B_Wave = criterion_Wave(rec_B, real_B) * LAMBDA_WAVE
            else:
                loss_cycle_A_Wave = fake_B.new_tensor(0.0)
                loss_cycle_B_Wave = fake_B.new_tensor(0.0)

            if LAMBDA_PHASE > 0:
                loss_phase_B = (criterion_Phase(rec_A, real_A) +
                                criterion_Phase(rec_B, real_B)) * 0.5 * LAMBDA_PHASE
            else:
                loss_phase_B = fake_B.new_tensor(0.0)

            if LAMBDA_STYLE > 0:
                loss_style_B = criterion_Style(fake_B, real_B) * LAMBDA_STYLE
            else:
                loss_style_B = fake_B.new_tensor(0.0)

            loss_spec_energy, spec_comps = criterion_SpecEnergy(
                fake=fake_B, real=real_B, rec=rec_A, real_same=real_A)

            loss_G = (loss_GAN_AB + loss_GAN_BA +
                      loss_cycle_A + loss_cycle_B +
                      loss_cycle_A_Wave + loss_cycle_B_Wave +
                      loss_id_A + loss_id_B +
                      loss_phase_B + loss_style_B +
                      loss_spec_energy)
            loss_G.backward()
            optimizer_G.step()

            # ---- Discriminators ----
            optimizer_D.zero_grad()
            fake_A_pool = pool_fake_A.query(fake_A)
            loss_D_A = (criterion_GAN(netD_A(real_A), True) +
                        criterion_GAN(netD_A(fake_A_pool.detach()), False)) * 0.5
            loss_D_A.backward()
            fake_B_pool = pool_fake_B.query(fake_B)
            loss_D_B = (criterion_GAN(netD_B(real_B), True) +
                        criterion_GAN(netD_B(fake_B_pool.detach()), False)) * 0.5
            loss_D_B.backward()
            optimizer_D.step()

            run_spec_ed += float(spec_comps.get('spec_ed', 0.0))
            run_phase_energy += float(spec_comps.get('phase_energy', 0.0))
            n_spec += 1

            with torch.no_grad():
                metrics_cycle.update_pixel_metrics(real_A.detach(), rec_A.detach())
                metrics_direct.update_pixel_metrics(real_B.detach(), fake_B.detach())
                metrics_direct.update_distribution_metrics(real_B.detach(), fake_B.detach())

            if i % 100 == 0:
                print(f"[{RUN_NAME}][Ep {epoch+1}/{NUM_EPOCHS}][B {i}] "
                      f"G:{loss_G.item():.3f} SpecED:{float(spec_comps.get('spec_ed',0)):.3f} "
                      f"PhaseEn:{float(spec_comps.get('phase_energy',0)):.3f}", flush=True)

        lr_scheduler_G.step()
        lr_scheduler_D.step()

        time_ep = time.time() - epoch_start
        pix_cycle = metrics_cycle.get_pixel_metrics()
        pix_direct = metrics_direct.get_pixel_metrics()
        dist = metrics_direct.compute_distribution_metrics()
        avg_spec_ed = run_spec_ed / max(1, n_spec)
        avg_phase_energy = run_phase_energy / max(1, n_spec)
        cur_fid = dist.get('FID', float('inf'))

        # ---- Early stopping bookkeeping (on FID) ----
        improved = cur_fid < (best_fid - ES_MIN_DELTA)
        if improved:
            best_fid = cur_fid
            epochs_no_improve = 0
            torch.save(netG_AB.state_dict(), os.path.join(CHECKPOINT_DIR, "netG_AB_best.pth"))
            torch.save(netG_BA.state_dict(), os.path.join(CHECKPOINT_DIR, "netG_BA_best.pth"))
        else:
            epochs_no_improve += 1

        print(f"\n{'='*80}\n[{RUN_NAME}] EPOCH {epoch+1} | {time_ep:.0f}s | "
              f"FID={cur_fid:.2f} (best={best_fid:.2f}, no_improve={epochs_no_improve})")
        print(f"  DIRECT SSIM:{pix_direct.get('SSIM',0):.4f} GSSIM:{pix_direct.get('GSSIM',0):.4f} "
              f"MAE:{pix_direct.get('MAE (L1)',0):.4f} LPIPS:{pix_direct.get('LPIPS',0):.4f}")
        print(f"  KID={dist.get('KID',0):.5f} IS={dist.get('IS',0):.2f} "
              f"SpecED={avg_spec_ed:.3f} PhaseEn={avg_phase_energy:.3f}\n{'='*80}\n", flush=True)

        with open(log_path, "a", newline='') as f:
            csv.writer(f).writerow([
                epoch + 1, int(time_ep), f"{loss_G.item():.4f}",
                f"{(loss_D_A+loss_D_B).item():.4f}",
                f"{loss_phase_B.item():.4f}", f"{loss_style_B.item():.4f}",
                f"{avg_spec_ed:.4f}", f"{avg_phase_energy:.4f}",
                f"{pix_cycle.get('PSNR',0):.4f}", f"{pix_cycle.get('SSIM',0):.4f}",
                f"{pix_cycle.get('GSSIM',0):.4f}", f"{pix_cycle.get('MAE (L1)',0):.4f}",
                f"{pix_cycle.get('LPIPS',0):.4f}",
                f"{pix_direct.get('PSNR',0):.4f}", f"{pix_direct.get('SSIM',0):.4f}",
                f"{pix_direct.get('GSSIM',0):.4f}", f"{pix_direct.get('MAE (L1)',0):.4f}",
                f"{pix_direct.get('LPIPS',0):.4f}",
                f"{cur_fid:.4f}", f"{dist.get('KID',0):.5f}", f"{dist.get('IS',0):.4f}",
                f"{best_fid:.4f}"
            ])

        with torch.no_grad():
            vis = torch.stack([
                denormalize(real_A[0].cpu()), denormalize(fake_B[0].cpu()),
                denormalize(rec_A[0].cpu()), denormalize(real_B[0].cpu())
            ])
            save_image(vis, os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}.png"))

        if (epoch + 1) % 5 == 0:
            torch.save(netG_AB.state_dict(), os.path.join(CHECKPOINT_DIR, f"netG_AB_ep{epoch+1}.pth"))
            torch.save(netG_BA.state_dict(), os.path.join(CHECKPOINT_DIR, f"netG_BA_ep{epoch+1}.pth"))

        # ---- Early stop check ----
        if epoch + 1 >= ES_MIN_EPOCHS and epochs_no_improve >= ES_PATIENCE:
            print(f"[{RUN_NAME}] EARLY STOP at epoch {epoch+1}: "
                  f"no FID improvement for {ES_PATIENCE} epochs (best FID={best_fid:.2f}).",
                  flush=True)
            break

    print(f"[{RUN_NAME}] DONE. Best FID = {best_fid:.2f}", flush=True)


if __name__ == "__main__":
    train()