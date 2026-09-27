#!/bin/bash
#SBATCH --job-name=swift_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=bench_%A_%a.out
#SBATCH --error=bench_%A_%a.err
#SBATCH --account=alloc_09195_paraiska2025
#SBATCH --array=0-2          # 3 benchmarks -> 3 array tasks (one GPU each)

# ============================================================
# SWIFT-GAN benchmark eğitimi (horse2zebra, maps, summer2winter)
# Her array görevi bir benchmark'ı bir GPU'da eğitir.
# Kullanım:  sbatch run_benchmarks.sh
# ============================================================

echo "[INFO] Activating virtual environment..."
source /scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/venv/bin/activate

echo "[INFO] Job started: $(date) on $(hostname)"
nvidia-smi

# --- Benchmark listesi (indirilen 3 veri seti) ---
DATA_ROOT="/scratch/lustre/projects/hpc_project_a1fb2511d81f42fba1f872930bb56828/sar_to_optical/swift_gan_src/benchmark_data"
BENCHES=(horse2zebra maps summer2winter_yosemite)

NAME=${BENCHES[$SLURM_ARRAY_TASK_ID]}
echo "[INFO] Array task $SLURM_ARRAY_TASK_ID -> benchmark: $NAME"

export BENCH_ROOT="$DATA_ROOT/$NAME"
export RUN_NAME="$NAME"
export NUM_EPOCHS=100
export BATCH_SIZE=2
export CHECKPOINT_DIR="./checkpoints_bench_$NAME"
export OUTPUT_DIR="./outputs_bench_$NAME"

# maps: klasör yönü (uydu->harita) tabloyla (Map->Satellite) ters, swap et.
# horse2zebra ve summer2winter zaten tablo yönünde.
if [ "$NAME" = "maps" ]; then
    export SWAP_AB=1
    echo "[INFO] maps -> SWAP_AB=1 (Map->Satellite yönü)"
else
    export SWAP_AB=0
fi

echo "[INFO] BENCH_ROOT=$BENCH_ROOT"
python train_benchmark.py

echo "[INFO] Benchmark '$NAME' finished at: $(date)"