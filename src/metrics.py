import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
    print("[INFO] LPIPS library found.")
except ImportError:
    LPIPS_AVAILABLE = False
    print("[WARNING] 'lpips' library not found! Run 'pip install lpips'.")

# Try to import TorchMetrics for FID, KID, IS
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    TORCHMETRICS_AVAILABLE = True
    print("[INFO] TorchMetrics library found (FID, KID, IS available).")
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("[WARNING] 'torchmetrics' not found! Run 'pip install torchmetrics[image]'.")

class GANMetrics:
    """
    A comprehensive metric calculator for GANs (specifically Unpaired/CycleGAN).
    
    This class handles two types of metrics:
    1. Pixel-wise Metrics (L1, MSE, PSNR, SSIM, GSSIM, LPIPS):
       - Use these to measure Cycle Consistency (Original SAR vs Reconstructed SAR).
       
    2. Distribution Metrics (FID, KID, IS):
       - Use these to measure realism of the generated domain (Generated Optical vs Real Optical Dataset).
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.metrics_accumulated = {}
        self.reset_pixel_metrics()
        
        # --- Initialize LPIPS ---
        if LPIPS_AVAILABLE:
            # 'alex' net is faster and widely used for evaluation
            self.lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
        else:
            self.lpips_fn = None

        # --- Initialize Distribution Metrics (FID, KID, IS) ---
        if TORCHMETRICS_AVAILABLE:
            # FID requires uint8 images usually, but torchmetrics handles normalization if configured.
            # feature=2048 is standard for InceptionV3.
            self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            self.kid_metric = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
            self.is_metric = InceptionScore(normalize=True).to(device)
        else:
            self.fid_metric = None
            self.kid_metric = None
            self.is_metric = None

    def reset_pixel_metrics(self):
        """Resets accumulators for pixel-wise metrics (PSNR, SSIM, etc.)."""
        self.pixel_stats = {
            'mae': 0.0,   # L1
            'mse': 0.0,
            'rmse': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'gssim': 0.0,
            'lpips': 0.0,
            'count': 0
        }

    def reset_distribution_metrics(self):
        """Resets accumulators for FID, KID, IS."""
        if TORCHMETRICS_AVAILABLE:
            self.fid_metric.reset()
            self.kid_metric.reset()
            self.is_metric.reset()

    # =========================================================================
    # GROUP 1: PIXEL-WISE METRICS (For Cycle Consistency: Real A vs Rec A)
    # =========================================================================
    
    def update_pixel_metrics(self, real_img, fake_img):
        """
        Calculates and accumulates L1, MSE, PSNR, SSIM, GSSIM, LPIPS.
        
        Args:
            real_img (Tensor): Real/Original images [B, 3, H, W], range [-1, 1]
            fake_img (Tensor): Reconstructed/Generated images [B, 3, H, W], range [-1, 1]
        """
        batch_size = real_img.size(0)
        
        # --- 1. LPIPS (GPU) ---
        current_lpips = 0.0
        if self.lpips_fn is not None:
            with torch.no_grad():
                # LPIPS expects input in [-1, 1]
                lpips_val = self.lpips_fn(fake_img, real_img)
                current_lpips = lpips_val.mean().item()

        # --- 2. CPU Metrics (L1, MSE, PSNR, SSIM) ---
        # Accumulators for this batch
        batch_mae = 0.0
        batch_mse = 0.0
        batch_psnr = 0.0
        batch_ssim = 0.0
        batch_gssim = 0.0

        for i in range(batch_size):
            # Convert tensors to numpy [0, 255]
            r_np = self._tensor_to_img_numpy(real_img[i])
            f_np = self._tensor_to_img_numpy(fake_img[i])
            
            # Float versions for math
            r_f = r_np.astype(np.float32)
            f_f = f_np.astype(np.float32)

            # L1 (MAE)
            batch_mae += np.mean(np.abs(r_f - f_f)) / 255.0
            
            # MSE
            mse_val = np.mean((r_f - f_f) ** 2)
            batch_mse += mse_val / (255.0 ** 2) # Normalize to [0,1] scale logic if needed, or keep raw

            # PSNR
            batch_psnr += psnr_metric(r_np, f_np, data_range=255)
            
            # SSIM
            batch_ssim += ssim_metric(r_np, f_np, data_range=255, channel_axis=2, win_size=7)
            
            # GSSIM (Gradient SSIM)
            batch_gssim += self._calculate_gssim(r_np, f_np)

        # Update global stats
        self.pixel_stats['mae'] += batch_mae / batch_size
        self.pixel_stats['mse'] += batch_mse / batch_size
        self.pixel_stats['rmse'] += np.sqrt(batch_mse / batch_size) # Approximate batch RMSE
        self.pixel_stats['psnr'] += batch_psnr / batch_size
        self.pixel_stats['ssim'] += batch_ssim / batch_size
        self.pixel_stats['gssim'] += batch_gssim / batch_size
        self.pixel_stats['lpips'] += current_lpips
        self.pixel_stats['count'] += 1

    def get_pixel_metrics(self):
        """Returns the average of all pixel-wise metrics processed so far."""
        cnt = self.pixel_stats['count']
        if cnt == 0:
            return {k: 0.0 for k in self.pixel_stats if k != 'count'}
        
        return {
            'MAE (L1)': self.pixel_stats['mae'] / cnt,
            'MSE': self.pixel_stats['mse'] / cnt,
            'RMSE': self.pixel_stats['rmse'] / cnt,
            'PSNR': self.pixel_stats['psnr'] / cnt,
            'SSIM': self.pixel_stats['ssim'] / cnt,
            'GSSIM': self.pixel_stats['gssim'] / cnt,
            'LPIPS': self.pixel_stats['lpips'] / cnt
        }

    # =========================================================================
    # GROUP 2: DISTRIBUTION METRICS (FID, KID, IS)
    # =========================================================================
    
    def update_distribution_metrics(self, real_opt_batch, fake_opt_batch):
        """
        Feeds batch data to FID, KID, and IS calculators.
        NOTE: These metrics are calculated over the whole dataset distribution, 
        not just per-batch averages.
        
        Args:
            real_opt_batch (Tensor): Real Optical images (Target domain) [B, 3, H, W], [-1, 1]
            fake_opt_batch (Tensor): Generated Optical images (Fake domain) [B, 3, H, W], [-1, 1]
        """
        if not TORCHMETRICS_AVAILABLE:
            return

        # TorchMetrics Image expects inputs in [0, 1] for normalize=True
        # Or [0, 255] uint8. We will convert [-1, 1] -> [0, 1]
        real_01 = (real_opt_batch + 1.0) / 2.0
        fake_01 = (fake_opt_batch + 1.0) / 2.0
        
        # Clamp to ensure numerical stability
        real_01 = torch.clamp(real_01, 0.0, 1.0)
        fake_01 = torch.clamp(fake_01, 0.0, 1.0)

        # Update FID and KID
        # real=True for real images, real=False for generated images
        self.fid_metric.update(real_01, real=True)
        self.fid_metric.update(fake_01, real=False)
        
        self.kid_metric.update(real_01, real=True)
        self.kid_metric.update(fake_01, real=False)
        
        # Update IS (Inception Score only needs generated images)
        self.is_metric.update(fake_01)

    def compute_distribution_metrics(self):
        """Computes the final FID, KID, and IS scores from accumulated batches."""
        if not TORCHMETRICS_AVAILABLE:
            return {'FID': -1, 'KID': -1, 'IS': -1}
        
        print("[INFO] Computing distribution metrics (this may take a moment)...")
        fid_score = self.fid_metric.compute().item()
        
        # KID returns (mean, std), we usually care about mean
        kid_score_mean, _ = self.kid_metric.compute()
        kid_score = kid_score_mean.item()
        
        # IS returns (mean, std)
        is_score_mean, _ = self.is_metric.compute()
        is_score = is_score_mean.item()
        
        return {
            'FID': fid_score,
            'KID': kid_score,
            'IS': is_score
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _tensor_to_img_numpy(self, tensor):
        """Helper to convert [-1, 1] tensor to [0, 255] numpy HWC."""
        img = tensor.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0)) # CHW -> HWC
        img = (img + 1.0) / 2.0            # [-1, 1] -> [0, 1]
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img

    def _calculate_gssim(self, img1_np, img2_np):
        """
        Gradient-based SSIM.
        Calculates SSIM on the gradient magnitude of the images.
        """
        # Convert to Gray for gradient calculation
        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

        # Compute Gradients (Sobel)
        g1x = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        g1y = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        mag1 = cv2.magnitude(g1x, g1y)

        g2x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        g2y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        mag2 = cv2.magnitude(g2x, g2y)

        # Normalize magnitudes to 0-255 uint8 for SSIM function
        mag1 = cv2.normalize(mag1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        mag2 = cv2.normalize(mag2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        return ssim_metric(mag1, mag2, data_range=255)

# Example Usage
if __name__ == "__main__":
    print("Testing Metrics Module...")
    
    # 1. Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = GANMetrics(device=device)
    
    # 2. Mock Data (Batch size 4)
    # SAR Images (Real and Reconstructed) -> For Pixel Metrics
    real_sar = torch.randn(4, 3, 256, 256).to(device).clamp(-1, 1)
    rec_sar = torch.randn(4, 3, 256, 256).to(device).clamp(-1, 1)
    
    # Optical Images (Real Dataset and Generated Fake) -> For Dist Metrics (FID/KID)
    real_opt = torch.randn(4, 3, 256, 256).to(device).clamp(-1, 1)
    fake_opt = torch.randn(4, 3, 256, 256).to(device).clamp(-1, 1)
    
    # 3. Training/Validation Loop Simulation
    print("Simulating batch updates...")
    for _ in range(2): # Run 2 batches
        # Update Pixel Metrics (Cycle Consistency)
        metrics.update_pixel_metrics(real_img=real_sar, fake_img=rec_sar)
        
        # Update Distribution Metrics (Realism)
        metrics.update_distribution_metrics(real_opt_batch=real_opt, fake_opt_batch=fake_opt)
        
    # 4. Get Results
    print("\n--- Pixel-wise Metrics Results (Cycle Consistency) ---")
    pixel_res = metrics.get_pixel_metrics()
    for k, v in pixel_res.items():
        print(f"{k}: {v:.4f}")
        
    print("\n--- Distribution Metrics Results (Perceptual Quality) ---")
    # Note: FID/KID usually require thousands of images to be accurate.
    dist_res = metrics.compute_distribution_metrics()
    for k, v in dist_res.items():
        print(f"{k}: {v:.4f}")