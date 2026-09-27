#!/bin/bash
#SBATCH --job-name=swift_city
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=city_%j.out
#SBATCH --error=city_%j.err
#SBATCH --account=alloc_09195_paraiska2025

# ============================================================
# Cityscapes Label->Photo:
#   1) (once) format: gtFine/leftImg8bit -> trainA/trainB
#   2) SWIFT-GAN training (FID/KID/SSIM/... = existing metrics)
#   3) post-training semantic score (pixAcc/classAcc/mIoU, DRN)
# Usage:  sbatch run_cityscapes.sh
# ============================================================

echo "[INFO] Activating virtual environment..."
source /scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/venv/bin/activate

echo "[INFO] Job started: $(date) on $(hostname)"
nvidia-smi

SRC="/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/sar_to_optical/swift_gan_src"
cd "$SRC"
DATA="$SRC/benchmark_data"
CITY="$DATA/cityscapes"

# --- 1) Format (only if trainA is missing) ---
if [ ! -d "$CITY/trainA" ]; then
    echo "[INFO] Formatting Cityscapes (label->A, photo->B)..."
    python prepare_cityscapes.py --root "$DATA" --out "$CITY"
else
    echo "[INFO] Cityscapes already formatted, skipping."
fi

# --- 2) Training (Label->Photo; A=label, B=photo => no swap) ---
export BENCH_ROOT="$CITY"
export RUN_NAME="cityscapes"
export SWAP_AB=0
export NUM_EPOCHS=100
export BATCH_SIZE=2
export CHECKPOINT_DIR="./checkpoints_bench_cityscapes"
export OUTPUT_DIR="./outputs_bench_cityscapes"

echo "[INFO] Training starts (Label->Photo)..."
python train_benchmark.py

# --- 3) Post-training semantic score (DRN) ---
echo "[INFO] Computing semantic score (pixAcc/classAcc/mIoU)..."
# If a DRN checkpoint is available, provide it via DRN_CKPT:
# export DRN_CKPT=/path/to/drn_d_22_cityscapes.pth
python eval_cityscapes_semantic.py \
    --ckpt "$CHECKPOINT_DIR/netG_AB_best.pth" \
    --testA "$CITY/testA" \
    --out "$OUTPUT_DIR"

echo "[INFO] Cityscapes done: $(date)"
echo "[INFO] Existing metrics: $OUTPUT_DIR/metrics_cityscapes.csv"
echo "[INFO] Semantic scores:  $OUTPUT_DIR/cityscapes_semantic_scores.txt"