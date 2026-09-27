import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.nn.utils.spectral_norm as spectral_norm

# ==========================================================
# 1. HELPER BLOCKS: WAVELET, FFT, SWIN
# ==========================================================

class WaveletDownsample(nn.Module):
    """
    NOVELTY 1: Lossless Downsampling using Haar Wavelets.
    Splits input into 4 frequency bands (LL, LH, HL, HH).
    Output channels = Input channels * 4.
    """
    def __init__(self, in_channels, out_channels):
        super(WaveletDownsample, self).__init__()
        # Define Haar Wavelet Filters (Fixed, Non-trainable)
        self.register_buffer('dec_lo', torch.tensor([1, 1], dtype=torch.float32).view(1, 1, 1, 2) / 1.4142)
        self.register_buffer('dec_hi', torch.tensor([-1, 1], dtype=torch.float32).view(1, 1, 1, 2) / 1.4142)
        
        # 1x1 Conv to mix the 4 bands into desired output channels
        self.reducer = nn.Conv2d(in_channels * 4, out_channels, 1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # Expand filters to match input channels
        lo = self.dec_lo.expand(C, 1, 1, 2)
        hi = self.dec_hi.expand(C, 1, 1, 2)
        
        # Convolve Rows
        x_lo = F.conv2d(x, lo, stride=(1, 2), groups=C)
        x_hi = F.conv2d(x, hi, stride=(1, 2), groups=C)
        
        # Convolve Cols (Transpose filters)
        lo_t = lo.transpose(2, 3)
        hi_t = hi.transpose(2, 3)
        
        # Extract 4 Bands
        LL = F.conv2d(x_lo, lo_t, stride=(2, 1), groups=C) # Structure
        LH = F.conv2d(x_lo, hi_t, stride=(2, 1), groups=C) # Horizontal Edges
        HL = F.conv2d(x_hi, lo_t, stride=(2, 1), groups=C) # Vertical Edges
        HH = F.conv2d(x_hi, hi_t, stride=(2, 1), groups=C) # Noise / Diagonal
        
        # Concatenate bands: [B, 4*C, H/2, W/2]
        out = torch.cat([LL, LH, HL, HH], dim=1)
        
        # Reduce dimension
        return self.act(self.norm(self.reducer(out)))

class SpectralTransformBlock(nn.Module):
    """
    NOVELTY 2: Phase-Aware Frequency Processing Block (soft phase lock).

    Core idea (unchanged): amplitude carries style/texture, phase carries
    structure/geometry. We modulate amplitude freely and KEEP the phase almost
    fixed to preserve SAR structure.

    Soft phase lock:
    - Fully freezing the phase (new_phase = phase) leaves the generator unable to
      make the small geometric adjustments needed to turn SAR structure into a
      plausible optical image, which pushes it toward a degenerate flat/black
      output (mode collapse). Instead we allow a SMALL, learnable phase correction
      `dphase`, gated by `phase_scale` which starts at 0. So at initialization the
      phase is still exactly frozen (thesis-consistent), and during training the
      network may open up a limited amount of phase adjustment as needed. The bulk
      of the phase (structure) is preserved; only a bounded refinement is learned.
    """
    def __init__(self, channels):
        super(SpectralTransformBlock, self).__init__()
        # Learnable mask for amplitude modulation
        self.amp_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        # Small learnable phase-correction network (operates on real+imag features)
        self.phase_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Tanh()  # bounded correction in [-1, 1] before scaling
        )
        # Normalize the frequency branch output to keep its scale in check
        self.norm = nn.InstanceNorm2d(channels)
        # Residual gate, small positive init so the frequency branch is slightly
        # active from the start (lets gradients reach phase_scale). The amplitude
        # path dominates early; the branch grows only if it helps.
        self.gamma = nn.Parameter(torch.full((1,), 0.1))
        # Phase-correction gate, starts at 0 -> phase fully frozen at init,
        # opens up gradually during training (soft phase lock).
        self.phase_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # 2. Amplitude Modulation (Style Transfer / Denoising)
        amp_mask = self.amp_conv(amp)
        new_amp = amp * amp_mask

        # 3. Soft Phase Lock: phase is preserved plus a small, gated correction.
        #    phase_scale starts at 0, so this is an exact phase freeze at init.
        phase_in = torch.cat([x_fft.real, x_fft.imag], dim=1)
        dphase = self.phase_conv(phase_in) * 3.141592653589793 * self.phase_scale
        new_phase = phase + dphase

        # 4. Inverse FFT
        new_fft = torch.polar(new_amp, new_phase)
        x_spatial = torch.fft.irfft2(new_fft, s=x.shape[2:], norm='ortho')

        # 5. Normalize the frequency branch, then add as a gated residual
        x_spatial = self.norm(x_spatial)

        # Gated Residual Connection (gamma starts at 0 -> identity at init)
        return x + self.gamma * x_spatial

class SwinBlock(nn.Module):
    """
    NOVELTY 3: Simplified Swin Transformer Block for Global Context.
    Captures long-range dependencies (e.g., road continuity).
    """
    def __init__(self, dim, num_heads=4):
        super(SwinBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten: [B, C, H, W] -> [B, H*W, C]
        flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Attention
        res = flat
        flat = self.norm1(flat)
        attn_out, _ = self.attn(flat, flat, flat)
        flat = res + attn_out
        
        # MLP
        res = flat
        flat = self.norm2(flat)
        mlp_out = self.mlp(flat)
        flat = res + mlp_out
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        return flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

# ==========================================================
# 2. GENERATOR: SWIFT ARCHITECTURE
# ==========================================================
class SwiftGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, use_fft=True):
        super(SwiftGenerator, self).__init__()

        # use_fft=False -> ABLATION "w/o FFT": the SpectralTransformBlocks are
        # replaced by Identity, so the bottleneck keeps the exact same Swin layout
        # and channel dims but performs no frequency (FFT) processing. Everything
        # else (wavelet down/up, decoder) is unchanged, so this isolates the
        # contribution of the FFT blocks only.
        self.use_fft = use_fft

        # --- ENCODER (Frequency Aware) ---
        # Input: 256x256 -> Downsample -> 128x128
        self.down1 = WaveletDownsample(input_nc, ngf * 2) 
        # 128x128 -> Downsample -> 64x64
        self.down2 = WaveletDownsample(ngf * 2, ngf * 4)

        # --- BOTTLENECK (Hybrid: Swin + FFT) ---
        # Processing at 64x64 resolution
        spectral_1 = SpectralTransformBlock(ngf * 4) if use_fft else nn.Identity()
        spectral_2 = SpectralTransformBlock(ngf * 4) if use_fft else nn.Identity()
        self.bottleneck = nn.Sequential(
            SwinBlock(ngf * 4),   # Global Context
            spectral_1,           # Frequency Consistency (or Identity if w/o FFT)
            SwinBlock(ngf * 4),   # Global Context
            spectral_2,           # Frequency Consistency (or Identity if w/o FFT)
            SwinBlock(ngf * 4)    # Global Context
        )

        # --- DECODER (Spatial Reconstruction) ---
        # 64x64 -> 128x128
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        
        # 128x128 -> 256x256
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        # Final Layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x) # [B, 128, 128, 128]
        d2 = self.down2(d1) # [B, 256, 64, 64]
        
        # Bottleneck
        neck = self.bottleneck(d2) # [B, 256, 64, 64]
        
        # Decoder (with minimal skip connections logic if needed, 
        # but CycleGAN standard usually implies direct flow for domain shift)
        u1 = self.up1(neck) # [B, 128, 128, 128]
        u2 = self.up2(u1)   # [B, 64, 256, 256]
        
        return self.final(u2)

# ==========================================================
# 3. DISCRIMINATOR: SPECTRAL GATING + PATCHGAN
# ==========================================================

class SpectralGatingBlock(nn.Module):
    """
    NOVELTY 4: Frequency Gating Mechanism for Discriminator.
    Filters 'Fake' frequencies that don't match Real distribution.

    Stability notes:
    - The complex weight is initialized near 1 (identity filter) rather than near 0,
      so the block passes features through unchanged at init instead of zeroing them.
    - Applied as a gated residual (gamma starts at 0) so the spectral filtering ramps
      up gradually and does not destabilize the discriminator early in training.
    """
    def __init__(self, dim):
        super(SpectralGatingBlock, self).__init__()
        # Initialize as an identity filter: real part ~1, imaginary part ~0
        w = torch.zeros(dim, 2, dtype=torch.float32)
        w[:, 0] = 1.0
        w = w + torch.randn(dim, 2) * 0.02
        self.complex_weight = nn.Parameter(w)
        # Learnable residual scale, starts at 0 -> block is identity at init
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        
        # Apply learnable spectral filter
        # Expand weight to match spatial dims (broadcasting)
        x_filt = x_fft * weight.view(1, -1, 1, 1)
        
        # IFFT
        x_spatial = torch.fft.irfft2(x_filt, s=x.shape[2:], norm='ortho')

        # Gated residual connection (gamma starts at 0 -> identity at init)
        return x + self.gamma * x_spatial

class SpectralDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super(SpectralDiscriminator, self).__init__()
        
        def disc_block(in_f, out_f, stride=2, norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Layer 1
        self.layer1 = nn.Sequential(*disc_block(input_nc, ndf, norm=False))
        
        # Layer 2 + Spectral Gating (Novelty)
        self.layer2 = nn.Sequential(*disc_block(ndf, ndf * 2))
        self.gate = SpectralGatingBlock(ndf * 2)
        
        # Layer 3
        self.layer3 = nn.Sequential(*disc_block(ndf * 2, ndf * 4))
        
        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output
        self.final = nn.Conv2d(ndf * 8, 1, 4, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.gate(x) # Spectral Gating applied here
        x = self.layer3(x)
        x = self.layer4(x)
        return self.final(x)

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

if __name__ == "__main__":
    print("Testing SWIFT-GAN Architecture...")
    x = torch.randn(1, 3, 256, 256) # Batch size 1 for test
    
    # Test Generator
    gen = SwiftGenerator(3, 3)
    out_gen = gen(x)
    print(f"Generator Output: {out_gen.shape}") # Should be [1, 3, 256, 256]
    
    # Test Discriminator
    disc = SpectralDiscriminator(3)
    out_disc = disc(x)
    print(f"Discriminator Output: {out_disc.shape}") # Should be [1, 1, 30, 30] approx