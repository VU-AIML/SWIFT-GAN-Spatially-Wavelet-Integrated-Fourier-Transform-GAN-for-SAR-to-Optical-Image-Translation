#!/bin/bash
#SBATCH --job-name=sar2opt_train
#SBATCH --partition=gpu        # Use the GPU partition
#SBATCH --gres=gpu:1           # Request 1 GPU card
#SBATCH --nodes=1              # Use 1 node
#SBATCH --ntasks=1             # Run 1 task
#SBATCH --cpus-per-task=4      # 4 CPU cores for DataLoader (matches num_workers in train.py)
#SBATCH --time=24:00:00        # Max runtime limit (12 hours)
#SBATCH --output=slurm_%j.out  # Standard output log file (%j is the job ID)
#SBATCH --error=slurm_%j.err   # Standard error log file
#SBATCH --account=alloc_09195_paraiska2025

# 1. Activate the Virtual Environment
echo "[INFO] Activating virtual environment..."
source /scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/venv/bin/activate

# 2. Print System Information (for debugging)
echo "[INFO] Job started at: $(date)"
echo "[INFO] Node running the job: $(hostname)"
echo "[INFO] GPU Information:"
nvidia-smi

# 3. Run the Training Script
echo "[INFO] Starting the training process..."
python swift_gan_src/error_analysis.py

echo "[INFO] Training job finished at: $(date)"