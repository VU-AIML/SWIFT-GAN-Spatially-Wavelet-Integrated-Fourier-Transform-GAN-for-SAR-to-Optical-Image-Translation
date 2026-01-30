# SWIFT-GAN: Spatially-Wavelet Integrated Fourier Transform GAN for SAR-to-Optical Image Translation

[![Paper](https://img.shields.io/badge/Paper-IJCNN%202026-blue)](https://github.com/username/repo)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Official PyTorch Implementation of "SWIFT-GAN: Spatially-Wavelet Integrated Fourier Transform GAN for SAR to Optical Image Translation"**

**Authors:** Kursat Komurcu, Vytautas Paura, Valentas Gruzauskas, Linas Petkevicius  
*Institute of Computer Science, Artificial Intelligence Methods Lab, Vilnius University*

## 🏗 Architecture & Methodology

SWIFT-GAN departs from standard CycleGAN architectures by introducing a hybrid generator and frequency-aware discriminators.

### 1. Hybrid Generator
* **Lossless Wavelet Downsampling:** Instead of standard pooling (which discards details), we use Discrete Wavelet Transform (DWT) to split input into frequency bands (LL, LH, HL, HH).
* **Spectral Transform Blocks (STB):** Operates in the Fourier domain to modulate amplitude (style) while **strictly preserving phase (structure)**.
* **Swin Transformer Blocks:** Integrated into the bottleneck to capture long-range semantic dependencies.

### 2. Novel Loss Functions
* **Phase Consistency Loss ($L_{phase}$):** Minimizes the $L_1$ distance between the Fourier phase spectra of the input SAR and generated Optical image. Since phase encodes structure, this prevents geometric hallucinations.
* **Wavelet Loss ($L_{wave}$):** Applies weighted supervision on DWT sub-bands to balance structure (LL) and texture (HH).
* **LAB Color Loss ($L_{color}$):** Matches the mean and std of 'A' and 'B' channels in CIELAB space to ensure atmospheric realism without distorting luminance (structure).

### 3. Spectral Discriminator
**Spectral Gating:** A mechanism that learns to filter unrealistic frequency artifacts in the Fourier domain, ensuring generated images match the spectral density of real optical data.

---

## 📂 Dataset: Vilnius Benchmark

The project uses a curated Sentinel-1/2 dataset covering the Vilnius region ($292.95 km^2$).

**SAR Input:** 3-Channel Composite
    * Channel 1: **VV** Polarization
    * Channel 2: **VH** Polarization
    * Channel 3: **VH/VV Ratio** (Physics-informed feature for volume scattering).
    
**Optical Target:** Sentinel-2 RGB (Cloud-free, <30%).

**Preprocessing:** All images are tiled into $256 \times 256$ non-overlapping patches.

The dataset available at: https://zenodo.org/records/18373534

---

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kursatkomurcu/SWIFT-GAN.git](https://github.com/yourusername/SWIFT-GAN.git)
    ```

2.  **Create Environment:**
    ```bash
    conda create -n swiftgan python=3.12
    conda activate swiftgan
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    pip install rasterio, tqdm, numpy, matplotlib, lpips, torchmetrics, scipy
    ```

---

## 🛠 Usage

### 1. Data Preparation
Organize your dataset as follows. Update the paths in `data.py` or the configuration section of `train.py`.

### 2. Training
To train the model from scratch using the SWIFT-GAN configuration:

```bash
# Update S1_ROOT and S2_ROOT in train.py before running
python train.py
```

### 3. Inference (Sliding Window)
To run inference on large SAR GeoTIFFs using a sliding window approach:

```bash
# Update VV_PATH, VH_PATH and CHECKPOINT_PATH in inference.py
python inference.py
```

### 4. Evaluation
To calculate quantitative metrics (PSNR, SSIM, FID, KID) on the test set:

```bash
python evaluation.py
```

| Method | Fidelity (PSNR) | Realism (FID) ↓ | Structure (Cycle-SSIM) ↑ |
| :--- | :---: | :---: | :---: |
| CycleGAN | 13.56 | 166.6 | 0.865 |
| CUT | 11.26 | 238.1 | 0.433 |
| BBDM (Diffusion) | 12.26 | 383.3 | N/A |
| **SWIFT-GAN (Ours)** | **12.40** | **169.6** | **0.908** |


###  🌲 Project Structure
```bash
SWIFT-GAN/
├── data.py           # Sentinel-1/2 Dataset Loader (3-Channel SAR logic)
├── model.py          # SWIFT Generator & Spectral Discriminator
├── loss.py           # Phase, Wavelet, and LAB Color Losses
├── metrics.py        # GAN Metrics (FID, KID, LPIPS, SSIM)
├── train.py          # Main training loop
├── evaluation.py     # Evaluation script for paired/unpaired metrics
├── inference.py      # Sliding window inference for large TIFs
└── checkpoints/      # Saved models
```

### 📜 Citation
If you use this code or dataset in your research, please cite our paper.

```bash
@inproceedings{komurcu2026swift,
  title={SWIFT-GAN: Spatially-Wavelet Integrated Fourier Transform GAN for SAR to Optical Image Translation},
  author={Kursat Komurcu, Vytautas Paura, Valentas Gruzauskas and Linas Petkevicius},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2026},
  organization={IEEE International Joint Conference on Neural Networks (IJCNN)}
}
```
