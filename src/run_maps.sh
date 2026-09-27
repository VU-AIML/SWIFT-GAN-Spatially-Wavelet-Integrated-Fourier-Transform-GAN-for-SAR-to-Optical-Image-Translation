#!/bin/bash
#SBATCH --job-name=swift_maps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=maps_%j.out
#SBATCH --error=maps_%j.err
#SBATCH --account=alloc_09195_paraiska2025
 
# ============================================================
# Sadece maps benchmark'ı — Map->Satellite yönü (SWAP_AB=1)
# Kullanım:  sbatch run_maps.sh
# ============================================================
 
echo "[INFO] Activating virtual environment..."
source /scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/venv/bin/activate
 
echo "[INFO] Job started: $(date) on $(hostname)"
nvidia-smi
 
DATA_ROOT="/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/sar_to_optical/swift_gan_src/benchmark_data"
 
export BENCH_ROOT="$DATA_ROOT/maps"
export RUN_NAME="maps"
export SWAP_AB=1                 # <-- Map->Satellite yönü (harita->uydu)
export NUM_EPOCHS=100
export BATCH_SIZE=2
export CHECKPOINT_DIR="./checkpoints_bench_maps"
export OUTPUT_DIR="./outputs_bench_maps"
 
echo "[INFO] BENCH_ROOT=$BENCH_ROOT | SWAP_AB=$SWAP_AB (Map->Satellite)"
python train_benchmark.py
 
echo "[INFO] maps finished at: $(date)"