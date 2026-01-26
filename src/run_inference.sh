#!/bin/bash
#SBATCH --job-name=sar2opt_inf
#SBATCH --partition=gpu        
#SBATCH --gres=gpu:1           
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2      
#SBATCH --time=00:30:00        
#SBATCH --output=inference_%j.out
#SBATCH --error=inference_%j.err
#SBATCH --account=alloc_3e953_vu74162vult  

echo "[INFO] Activating virtual environment..."
source /scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/venv/bin/activate

echo "[INFO] Starting Inference Process..."
python swift_gan_src/inference.py

echo "[INFO] Inference job finished."