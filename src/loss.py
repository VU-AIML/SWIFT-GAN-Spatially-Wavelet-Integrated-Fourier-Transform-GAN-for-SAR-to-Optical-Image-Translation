import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ==========================================================
# 1. PHASE CONSISTENCY LOSS (NOVELTY: STRUCTURE LOCK)
# ==========================================================
class PhaseConsistencyLoss(nn.Module):
    """
    SWIFT-GAN CORE COMPONENT:
    Computes the L1 distance between the Fourier Phase spectra.
    
    Logic:
    - Amplitude holds 'Style' (Texture/Noise).
    - Phase holds 'Structure' (Geometry/Edges).
    
    By minimizing phase difference, we strictly enforce that 
    generated Optical images keep the exact layout of SAR inputs.
    """
    def __init__(self):
        super(PhaseConsistencyLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # 1. FFT (Real -> Complex)
        eps = 1e-6
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # 2. Represent phase as UNIT PHASORS (F / |F|) instead of raw angles.
        #    torch.angle() is discontinuous at the +/-pi wrap boundary and has an
        #    undefined gradient where the amplitude is ~0, which injects noise and
        #    can destabilize training. Comparing unit phasors captures the exact
        #    same phase information (structure) with smooth, bounded gradients and
        #    no NaNs, while still fully preserving the phase-locking objective.
        pred_phasor = pred_fft / (pred_fft.abs() + eps)
        target_phasor = target_fft / (target_fft.abs() + eps)

        # 3. L1 distance between phasors (real + imaginary parts)
        diff = pred_phasor - target_phasor
        return diff.real.abs().mean() + diff.imag.abs().mean()

# ==========================================================
# 2. WAVELET LOSS (NOVELTY: FREQUENCY SEPARATION)
# ==========================================================
class WaveletLoss(nn.Module):
    """
    SWIFT-GAN FREQUENCY COMPONENT:
    Decomposes images into 4 bands (LL, LH, HL, HH) using Haar Wavelets.
    
    Logic:
    - LL (Low-Low): Main Structure -> High Penalty needed.
    - HH (High-High): Noise/Speckle -> Low Penalty needed.
    """
    def __init__(self, device='cuda'):
        super(WaveletLoss, self).__init__()
        # Haar Wavelet Filters (Fixed buffers)
        self.register_buffer('dec_lo', torch.tensor([1, 1], dtype=torch.float32).view(1, 1, 1, 2) / 1.4142)
        self.register_buffer('dec_hi', torch.tensor([-1, 1], dtype=torch.float32).view(1, 1, 1, 2) / 1.4142)

    def dwt_one_level(self, x):
        B, C, H, W = x.shape
        # Expand filters to match channels
        lo = self.dec_lo.expand(C, 1, 1, 2)
        hi = self.dec_hi.expand(C, 1, 1, 2)
        
        # Convolve Rows
        x_lo = F.conv2d(x, lo, stride=(1, 2), groups=C)
        x_hi = F.conv2d(x, hi, stride=(1, 2), groups=C)
        
        # Convolve Cols
        lo_t = lo.transpose(2, 3)
        hi_t = hi.transpose(2, 3)
        
        LL = F.conv2d(x_lo, lo_t, stride=(2, 1), groups=C)
        LH = F.conv2d(x_lo, hi_t, stride=(2, 1), groups=C)
        HL = F.conv2d(x_hi, lo_t, stride=(2, 1), groups=C)
        HH = F.conv2d(x_hi, hi_t, stride=(2, 1), groups=C)
        
        return LL, LH, HL, HH

    def forward(self, pred, target):
        pred_LL, pred_LH, pred_HL, pred_HH = self.dwt_one_level(pred)
        target_LL, target_LH, target_HL, target_HH = self.dwt_one_level(target)
        
        # Strategy: Prioritize Structure (LL), tolerate some Noise (HH)
        loss_LL = F.l1_loss(pred_LL, target_LL)
        loss_LH = F.l1_loss(pred_LH, target_LH)
        loss_HL = F.l1_loss(pred_HL, target_HL)
        loss_HH = F.l1_loss(pred_HH, target_HH)
        
        # Weighted Sum
        total_loss = (2.0 * loss_LL) + (1.0 * loss_LH) + (1.0 * loss_HL) + (0.5 * loss_HH)
        return total_loss

# ==========================================================
# 3. LAB COLOR LOSS (REPLACES STYLE LOSS)
# ==========================================================
class LABColorLoss(nn.Module):
    """
    PLAN C: LAB Color Space Moments Loss.
    
    Why?
    - Standard RGB losses mix Structure (Luminance) and Color (Chrominance).
    - We want to fix the 'Atmosphere' (Green/Blue/Sepia balance) WITHOUT 
      disturbing the building edges (Structure).
      
    Mechanism:
    1. Convert RGB -> LAB (Differentiable).
    2. Discard 'L' channel (Structure is handled by Phase/Wavelet).
    3. Match Mean/Std of 'A' and 'B' channels between Fake and Real.
    """
    def __init__(self):
        super(LABColorLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def rgb_to_xyz(self, image):
        # RGB [-1, 1] -> [0, 1]
        image = (image + 1.0) * 0.5
        
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        # Gamma correction (approximate for sRGB)
        mask = (image > 0.04045).float()
        image = mask * torch.pow((image + 0.055) / 1.055, 2.4) + (1 - mask) * (image / 12.92)
        
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        
        # sRGB to XYZ matrix
        X = 0.412453 * r + 0.357580 * g + 0.180423 * b
        Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        Z = 0.019334 * r + 0.119193 * g + 0.950227 * b
        
        return X, Y, Z

    def xyz_to_lab(self, X, Y, Z):
        # D65 White Point
        X_n, Y_n, Z_n = 0.95047, 1.00000, 1.08883
        
        X = X / X_n
        Y = Y / Y_n
        Z = Z / Z_n

        # Non-linear transformation (f function)
        # Using a small epsilon to prevent NaN gradients near 0
        mask = (Y > 0.008856).float()
        
        def f(t):
            # Safe power for differentiation
            t = torch.clamp(t, min=1e-6) 
            return torch.pow(t, 1/3)

        fX = mask * f(X) + (1 - mask) * (7.787 * X + 16/116)
        fY = mask * f(Y) + (1 - mask) * (7.787 * Y + 16/116)
        fZ = mask * f(Z) + (1 - mask) * (7.787 * Z + 16/116)

        L = 116 * fY - 16
        a = 500 * (fX - fY)
        b = 200 * (fY - fZ)
        
        return L, a, b

    def forward(self, fake, real):
        # 1. Convert to LAB
        fX, fY, fZ = self.rgb_to_xyz(fake)
        fake_L, fake_a, fake_b = self.xyz_to_lab(fX, fY, fZ)
        
        rX, rY, rZ = self.rgb_to_xyz(real)
        real_L, real_a, real_b = self.xyz_to_lab(rX, rY, rZ)

        # 2. Compute Statistics (Moments) for A and B channels only
        # We ignore L channel to protect structure
        
        # Mean Matching (Global Tone)
        mean_loss = self.l1(fake_a.mean(), real_a.mean()) + \
                    self.l1(fake_b.mean(), real_b.mean())
        
        # Std Matching (Contrast/Vibrance)
        std_loss = self.l1(fake_a.std(), real_a.std()) + \
                   self.l1(fake_b.std(), real_b.std())

        return mean_loss + std_loss

# ==========================================================
# 4. STANDARD GAN LOSS (LSGAN)
# ==========================================================
class GANLoss(nn.Module):
    """
    Standard Adversarial Loss (Least Squares GAN).
    More stable than BCE Loss for CycleGAN training.
    """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)