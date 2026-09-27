#!/bin/bash
#SBATCH --job-name=swift_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                 # 1 GPU per array task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2           # matches num_workers=4 in train.py
#SBATCH --time=48:00:00
#SBATCH --array=2-5                 # 6 runs -> 6 GPUs, in parallel
#SBATCH --output=ablation_%A_%a.out  # %A=job id, %a=array task id
#SBATCH --error=ablation_%A_%a.err
#SBATCH --account=alloc_09195_paraiska2025

# ==========================================================
# Runs all 6 ablations at once, one per GPU, via a SLURM array.
# Each array task ($SLURM_ARRAY_TASK_ID) picks its config from run_configs.sh.
#
# NOTE ON THE SHARED INDEX CACHE:
#   train.py uses cache_index=True. To avoid 6 jobs scanning the dataset at the
#   same time (and racing to write the same .npy), we PRE-BUILD the index once
#   below, only on array task 0, and make the others wait briefly. If your index
#   .npy files already exist, this is instant.
# ==========================================================

echo "[INFO] Array task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "[INFO] Activating virtual environment..."
source /scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/venv/bin/activate

cd "$SLURM_SUBMIT_DIR" || exit 1

# --- Load this task's configuration ---
source ./run_configs.sh "${SLURM_ARRAY_TASK_ID}"

echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] Weights: GAN=${LAMBDA_GAN} CYCLE=${LAMBDA_CYCLE} PHASE=${LAMBDA_PHASE} "\
"WAVE=${LAMBDA_WAVE} STYLE=${LAMBDA_STYLE} SPEC_ED=${LAMBDA_SPEC_ENERGY} "\
"PHASE_ENERGY=${LAMBDA_PHASE_ENERGY} USE_FFT=${USE_FFT}"

# --- Shared index cache: let task 0 build it first, others wait a bit ---
if [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then
    echo "[INFO] Task ${SLURM_ARRAY_TASK_ID} waiting 90s for index cache (task 0 builds it)..."
    sleep 90
fi

echo "[INFO] Starting training for ${RUN_NAME}..."
python train.py

echo "[INFO] ${RUN_NAME} finished at $(date)"