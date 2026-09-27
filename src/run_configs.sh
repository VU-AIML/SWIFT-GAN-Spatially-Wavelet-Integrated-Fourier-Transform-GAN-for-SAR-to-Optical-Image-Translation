#!/bin/bash
# ==========================================================
# Ablation run configurations (6 runs, one per SLURM array task).
# Sourced by run_ablations.sh; sets env vars for train.py based on $CFG_ID.
#
# Runs 0-2 : classic ablations that fill the EMPTY rows in your paper table
#            (w/o Wavelet, w/o LAB, w/o FFT).
# Runs 3-5 : energy-loss ablations (isolate ED, ED+Phase, and energy-only).
#
# Baseline (full SWIFT-GAN, FID 73.39) is ALREADY trained -> not repeated here.
# ==========================================================

CFG_ID=$1

# Shared defaults (train.py also has these as fallbacks)
export BATCH_SIZE=2
export NUM_EPOCHS=100
export LR=0.0002
export ES_PATIENCE=15
export ES_MIN_EPOCHS=25
export ES_MIN_DELTA=0.5

case $CFG_ID in
  0)
    # ---- w/o Wavelet ----  (classic ablation, table row)
    export RUN_NAME="wo_wavelet"
    export LAMBDA_GAN=1.0; export LAMBDA_CYCLE=10.0; export LAMBDA_ID=5.0
    export LAMBDA_PHASE=1.0; export LAMBDA_WAVE=0.0; export LAMBDA_STYLE=0.1
    export LAMBDA_SPEC_ENERGY=0.0; export LAMBDA_PHASE_ENERGY=0.0
    export USE_FFT=true
    ;;
  1)
    # ---- w/o LAB (style) ----  (classic ablation, table row)
    export RUN_NAME="wo_lab"
    export LAMBDA_GAN=1.0; export LAMBDA_CYCLE=10.0; export LAMBDA_ID=5.0
    export LAMBDA_PHASE=1.0; export LAMBDA_WAVE=1.0; export LAMBDA_STYLE=0.0
    export LAMBDA_SPEC_ENERGY=0.0; export LAMBDA_PHASE_ENERGY=0.0
    export USE_FFT=true
    ;;
  2)
    # ---- w/o FFT ----  (classic ablation, table row; architectural)
    export RUN_NAME="wo_fft"
    export LAMBDA_GAN=1.0; export LAMBDA_CYCLE=10.0; export LAMBDA_ID=5.0
    export LAMBDA_PHASE=1.0; export LAMBDA_WAVE=1.0; export LAMBDA_STYLE=0.1
    export LAMBDA_SPEC_ENERGY=0.0; export LAMBDA_PHASE_ENERGY=0.0
    export USE_FFT=false
    ;;
  3)
    # ---- Full + ED ----  (isolate the new energy-distance term)
    export RUN_NAME="energy_ed"
    export LAMBDA_GAN=1.0; export LAMBDA_CYCLE=10.0; export LAMBDA_ID=5.0
    export LAMBDA_PHASE=1.0; export LAMBDA_WAVE=1.0; export LAMBDA_STYLE=0.1
    export LAMBDA_SPEC_ENERGY=0.1; export LAMBDA_PHASE_ENERGY=0.0
    export USE_FFT=true
    ;;
  4)
    # ---- Full + ED + PhaseEnergy ----  (full spectral-energy functional)
    # old PhaseConsistencyLoss OFF so phase isn't penalized twice.
    export RUN_NAME="energy_ed_phase"
    export LAMBDA_GAN=1.0; export LAMBDA_CYCLE=10.0; export LAMBDA_ID=5.0
    export LAMBDA_PHASE=0.0; export LAMBDA_WAVE=1.0; export LAMBDA_STYLE=0.1
    export LAMBDA_SPEC_ENERGY=0.1; export LAMBDA_PHASE_ENERGY=0.5
    export USE_FFT=true
    ;;
  5)
    # ---- Energy-only ----  (replace ALL frequency/style losses with energy;
    #      GAN + cycle infrastructure kept. Tests "energy can replace them".)
    export RUN_NAME="energy_only"
    export LAMBDA_GAN=1.0; export LAMBDA_CYCLE=10.0; export LAMBDA_ID=5.0
    export LAMBDA_PHASE=0.0; export LAMBDA_WAVE=0.0; export LAMBDA_STYLE=0.0
    export LAMBDA_SPEC_ENERGY=0.1; export LAMBDA_PHASE_ENERGY=0.5
    export USE_FFT=true
    ;;
  *)
    echo "[ERROR] Unknown CFG_ID=$CFG_ID (expected 0-5)"; exit 1
    ;;
esac

export CHECKPOINT_DIR="./checkpoints_${RUN_NAME}"
export OUTPUT_DIR="./outputs_${RUN_NAME}"