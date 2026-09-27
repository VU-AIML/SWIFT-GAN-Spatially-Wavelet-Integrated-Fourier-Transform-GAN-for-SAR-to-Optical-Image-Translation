# SWIFT-GAN: Structure-Preserving SAR-to-Optical Translation with Bounded Fourier Phase-Locking

[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official PyTorch implementation of **SWIFT-GAN** (Spatially-Wavelet Integrated Fourier Transform GAN), an unpaired generator that translates Sentinel-1 SAR images into Sentinel-2-like optical images while preserving the scene layout.

**Authors:** Kürşat Kömürcü, Vytautas Paura, Valentas Gruzauskas, Linas Petkevičius
*Artificial Intelligence Methods Lab, Institute of Computer Science, Vilnius University*

---

## Overview

Unpaired translation models often produce realistic optical texture that is only loosely tied to the SAR geometry. SWIFT-GAN addresses this with **phase-locking**: inside the generator, the Fourier *amplitude* of the features is modulated freely, while their *phase*, which carries most of the spatial layout, may change only by a small, bounded, learned amount.

Main results on 2,000 co-registered Sentinel-1/2 test pairs (acquired at most 3 days apart):

- SWIFT-GAN gives the lowest **MAE (0.134)** and **LPIPS (0.521)** to the aligned optical image among the compared methods, with the third-lowest FID.
- Removing the spectral transform blocks (no phase-locking) raises **FID from 121.7 to 195.8** and lowers edge agreement (GSSIM 0.204 → 0.158).
- The generator has **3.5 M parameters** and runs in a single forward pass (16.7 ms per 256×256 image on a V100).

---

## Method

### Generator (3.54 M parameters)

| Stage | Description |
| :--- | :--- |
| **Wavelet downsampling** (×2) | One-level Haar transform per channel (LL, LH, HL, HH sub-bands), followed by a 1×1 projection. Unlike strided convolution, no information is discarded before the learned projection. |
| **Bottleneck** (64×64, 256 ch.) | Three transformer blocks with global multi-head self-attention, interleaved with two **Spectral Transform Blocks (STB)**. |
| **Decoder** | Two transposed convolutions and a 7×7 output convolution with `tanh`. |

**Spectral Transform Block.** For a feature map `h` with `F(h) = A·exp(iΦ)`:

- amplitude: `A' = A ⊙ σ(M(A))` (learned mask),
- phase: `Φ' = Φ + π·s·tanh(P(Re F, Im F))`, with a learnable gate `s` initialised to 0, so `|Φ' − Φ| ≤ π|s|`,
- output: `h + γ · IN(F⁻¹(A'·exp(iΦ')))`.

In trained models the phase correction stays below about 20° per frequency, and setting it to zero at inference changes the results by less than 0.01. The effective mechanism is therefore amplitude modulation under an (almost) fixed phase.

### Discriminator

PatchGAN (four 4×4 convolutional stages with instance normalisation). The implementation contains an additional residual complex-valued gate. It has one weight per channel and in practice reduces to a per-channel scaling.

### Losses

| Loss | Weight | Description |
| :--- | :---: | :--- |
| Adversarial (LSGAN) | 1 | both directions |
| Cycle consistency (L1) | 10 | SAR→optical→SAR and optical→SAR→optical |
| Identity (L1) | 5 | |
| Phase consistency | 1 | L1 between **unit phasors** `F/|F|` of the input and its cycle reconstruction. This avoids the ±π wrap-around and undefined phase at zero amplitude. |
| Wavelet | 1 | L1 on Haar sub-bands of the cycle reconstruction (weights LL 2, LH 1, HL 1, HH 0.5) |
| LAB colour | 0.1 | matches mean and std of the CIELAB *a*, *b* channels of generated and real optical images |

---

## Dataset

Sentinel-1 and Sentinel-2 imagery for **181 areas of 20×20 km** in Lithuania, Latvia and Estonia, exported through Google Earth Engine (January 2024 – October 2025).

| | |
| :--- | :--- |
| **SAR input** | Sentinel-1 GRD, IW mode. 3-channel composite: **VV**, **VH**, **VH/VV ratio**. VV and VH are clipped to their 2nd–98th percentile and rescaled to [0, 1] per patch. |
| **Optical target** | Sentinel-2 L2A, RGB (B4, B3, B2) / 3000, clipped to [0, 1]. Patches with ≥ 10 % cloud, shadow or saturated pixels (SCL) are removed. |
| **Grid** | Both sensors exported on the same UTM grid at 10 m, so identical pixel windows cover identical ground areas. |
| **Training patches** | 256×256, non-overlapping: 702,242 SAR and 23,888 optical patches. |
| **Test set** | 2,000 co-registered SAR/optical patch pairs (\|Δt\| ≤ 3 days, < 5 % invalid pixels) from 149 areas; the exact list is in `paired_test_set.csv`. |

A subset covering the Vilnius region is available at [Zenodo](https://zenodo.org/records/18373534). The full Baltic dataset and the test-patch list will be released upon publication.

Expected layout, one folder per area, with both sensors on the same grid:

```
data/
├── Sentinel-1/<area_id>/S1A_IW_GRDH_1SDV_<date>_..._VV.tif   (+ _VH.tif)
└── Sentinel-2/<area_id>/<date>_..._<tile>_{B2,B3,B4,SCL}.tif
```

---

## Installation

```bash
git clone https://github.com/kursatkomurcu/SWIFT-GAN.git
cd SWIFT-GAN

conda create -n swiftgan python=3.12
conda activate swiftgan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install rasterio tqdm numpy scipy matplotlib opencv-python-headless scikit-image lpips "torchmetrics[image]" torch-fidelity einops
```

---

## Usage

### 1. Training on Sentinel-1/2

Set `S1_ROOT` and `S2_ROOT` in `train.py`, then:

```bash
RUN_NAME=swift_gan python train.py
```

All settings can be overridden with environment variables. Ablations use the same script:

```bash
USE_FFT=false RUN_NAME=wo_stb       python train.py   # without spectral transform blocks
LAMBDA_WAVE=0 RUN_NAME=wo_wavelet   python train.py   # without wavelet loss
LAMBDA_STYLE=0 RUN_NAME=wo_lab      python train.py   # without LAB colour loss
```

Other variables: `BATCH_SIZE`, `NUM_EPOCHS`, `LR`, `LAMBDA_GAN`, `LAMBDA_CYCLE`, `LAMBDA_ID`, `LAMBDA_PHASE`. Checkpoints are written to `checkpoints_<RUN_NAME>/`, logs to `outputs_<RUN_NAME>/`. On SLURM, `run_ablations.sh` together with `run_configs.sh` trains all ablations as an array job.

### 2. Evaluation on co-registered pairs

```bash
python evaluate_paired.py --max-days 3 --n-total 2000
```

The script pairs every Sentinel-1 acquisition with the Sentinel-2 scene of the same area acquired at most `--max-days` apart, samples cloud-free 256×256 windows and scores every model against the aligned optical image:

- **Fidelity:** SSIM, PSNR, MAE, LPIPS, GSSIM.
- **Realism:** FID, KID.
- **Reference rows:** *Identity* (SAR input as output) and *Random optical* (an unrelated real optical patch).
- **Phase switch-off:** the main model is also evaluated with its learned phase correction set to zero.

Outputs: `paired_test_set.csv` (the test patches) and `results_paired.csv`.

### 3. Standard benchmarks

```bash
bash download_benchmarks.sh                                # horse2zebra, maps, summer2winter, cityscapes, AFHQ
BENCH_ROOT=benchmark_data/horse2zebra RUN_NAME=horse2zebra python train_benchmark.py
BENCH_ROOT=benchmark_data/maps RUN_NAME=maps SWAP_AB=1     python train_benchmark.py   # Map→Satellite
python evaluate_benchmarks_test.py                          # FID / KID / SSIM on the test splits
```

### 4. Inference on full scenes

Sliding-window translation of a Sentinel-1 GeoTIFF (set `VV_PATH`, `VH_PATH` and `CHECKPOINT_PATH` in `inference.py`):

```bash
python inference.py
```

### 5. Figures and timing

```bash
python make_phaselock_figure.py --n-rows 4     # with vs. without phase-locking on test pairs
python make_qualitative_figure.py              # comparison with all baselines
python measure_inference_speed.py              # parameters and ms per image
```

---

## Results

### Sentinel-1 → Sentinel-2 (2,000 co-registered test pairs)

| Method | SSIM ↑ | PSNR ↑ | MAE ↓ | LPIPS ↓ | GSSIM ↑ | FID ↓ | KID ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Identity (SAR input)* | 0.006 | 5.07 | 0.493 | 0.862 | 0.057 | 393.1 | 0.454 |
| *Random optical* | 0.307 | – | 0.144 | 0.568 | 0.177 | 0.0 | 0.000 |
| CycleGAN | 0.270 | 15.34 | 0.153 | 0.558 | **0.261** | **104.6** | **0.071** |
| CycleGAN-CBAM | 0.288 | 15.93 | 0.142 | 0.548 | 0.223 | 111.7 | 0.086 |
| CUT | 0.041 | 7.14 | 0.407 | 0.677 | 0.093 | 255.7 | 0.272 |
| SwinCUT | **0.337** | **16.49** | 0.150 | 0.553 | 0.246 | 207.5 | 0.207 |
| QS-Attn | 0.198 | 15.31 | 0.140 | 0.630 | 0.163 | 284.0 | 0.310 |
| UNSB | 0.029 | 6.83 | 0.402 | 0.814 | 0.069 | 368.8 | 0.410 |
| **SWIFT-GAN** | 0.290 | 16.32 | **0.134** | **0.521** | 0.204 | 121.7 | 0.102 |
| ↳ no phase correction | 0.289 | 16.21 | 0.135 | 0.531 | 0.210 | 122.0 | 0.099 |
| ↳ w/o STB (no phase-locking) | 0.293 | 17.07 | 0.126 | 0.542 | 0.158 | 195.8 | 0.187 |
| ↳ w/o wavelet loss | 0.305 | 16.25 | 0.134 | 0.525 | 0.236 | 107.8 | 0.087 |
| ↳ w/o LAB colour loss | 0.403 | 18.09 | 0.108 | 0.491 | 0.234 | 131.9 | 0.115 |

Rows in italics are reference rows. The FID of *Random optical* is zero by construction, because it re-uses the reference patches.

### Standard benchmarks (test splits, last checkpoint)

| Task | FID ↓ | KID ↓ | SSIM (input, output) ↑ |
| :--- | :---: | :---: | :---: |
| Cityscapes Label → Photo | 65.4 | 0.022 | 0.411 |
| Map → Satellite | 64.0 | 0.023 | 0.169 |
| Summer → Winter | 75.9 | 0.005 | 0.892 |
| Horse → Zebra | 131.5 | 0.072 | 0.858 |
| Cat → Dog (AFHQ) | 146.6 | 0.093 | 0.800 |

Phase-locking works well when the translation keeps the layout (label-to-photo, map-to-satellite, season transfer). It is not suited to tasks that must change object shape (Cat → Dog).

---

## Citation

The paper is currently submitted to *Scientific Reports*. If you use this code or dataset, please cite:

```bibtex
@article{komurcu2026swiftgan,
  title   = {Structure-preserving {SAR}-to-optical image translation with bounded {F}ourier phase-locking},
  author  = {K{\"o}m{\"u}rc{\"u}, K{\"u}r{\c{s}}at and Paura, Vytautas and Gruzauskas, Valentas and Petkevi{\v{c}}ius, Linas},
  journal = {Scientific Reports},
  note    = {Submitted},
  year    = {2026}
}
```

## License

This project is released under the MIT License (see `LICENSE`).
