#!/bin/bash
# Submit a focused LocalWNO sweep, matched baselines, and dependent comparison.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

run_dirs=(
    checkpoints/xhi2d_localwno_lr2e4_e30
    checkpoints/xhi2d_localwno_lr3e4_e30
    checkpoints/xhi2d_localwno_lr3e4_r32
    checkpoints/xhi2d_localwno_lr3e4_w48r24
    checkpoints/xhi2d_localwno_lr3e4_h1e3
    checkpoints/xhi2d_localfno_lr3e4_e30
    checkpoints/xhi2d_ufno_lr3e4_e30
)
for run_dir in "${run_dirs[@]}"; do
    if [ -e "$run_dir" ]; then
        echo "Refusing to reuse existing run directory: $run_dir" >&2
        exit 1
    fi
done

# Pin every environment variable parsed by fno_21cm.py. --export=ALL is kept
# only so the cluster's module/Conda environment reaches the batch shell.
COMMON="CACHE_FILE=$PROJECT_ROOT/trainset.h5,INPUT_FEATURES=density_z_params,N_EPOCHS=30,EVAL_INTERVAL=5,BATCH_SIZE=32,LEARNING_RATE=3e-4,WEIGHT_DECAY=1e-5,RUN_SEED=0,SPLIT_SEED=42,VAL_FRACTION=0.1,TEST_FRACTION=0.1,LOSS_L2_WEIGHT=1.0,LOSS_H1_WEIGHT=0.0,DEVICE=cuda,RESUME_DIR=,N_MODES_X=32,N_MODES_Y=32,HIDDEN_CHANNELS=64,N_LAYERS=4,UFNO_WIDTH=32,UFNO_NORM=batchnorm,LOCALFNO_BASE_WIDTH=32,LOCALFNO_WINDOW_X=16,LOCALFNO_WINDOW_Y=16,LOCALFNO_MODES_X=6,LOCALFNO_MODES_Y=6,LOCALFNO_GLOBAL_MODES_X=16,LOCALFNO_GLOBAL_MODES_Y=16,LOCALFNO_SPECTRAL_RANK=16,LOCALFNO_PATCH_CHUNK_SIZE=32,LOCALWNO_LEVELS=2"

jobs=()

job=$(sbatch --parsable --job-name=x2-wno-lr2 --time=08:00:00 \
    --export="ALL,$COMMON,MODEL_KIND=localwno,LEARNING_RATE=2e-4,CHECKPOINT_DIR=checkpoints/xhi2d_localwno_lr2e4_e30" \
    slurm/train_2d_xhi.sbatch)
jobs+=("$job")
echo "LocalWNO lr=2e-4:             $job"

job=$(sbatch --parsable --job-name=x2-wno-lr3 --time=08:00:00 \
    --export="ALL,$COMMON,MODEL_KIND=localwno,CHECKPOINT_DIR=checkpoints/xhi2d_localwno_lr3e4_e30" \
    slurm/train_2d_xhi.sbatch)
jobs+=("$job")
echo "LocalWNO lr=3e-4:             $job"

job=$(sbatch --parsable --job-name=x2-wno-r32 --time=08:00:00 \
    --export="ALL,$COMMON,MODEL_KIND=localwno,LOCALFNO_SPECTRAL_RANK=32,CHECKPOINT_DIR=checkpoints/xhi2d_localwno_lr3e4_r32" \
    slurm/train_2d_xhi.sbatch)
jobs+=("$job")
echo "LocalWNO rank=32:              $job"

job=$(sbatch --parsable --job-name=x2-wno-w48 --time=12:00:00 \
    --export="ALL,$COMMON,MODEL_KIND=localwno,LOCALFNO_BASE_WIDTH=48,LOCALFNO_SPECTRAL_RANK=24,CHECKPOINT_DIR=checkpoints/xhi2d_localwno_lr3e4_w48r24" \
    slurm/train_2d_xhi.sbatch)
jobs+=("$job")
echo "LocalWNO width=48 rank=24:     $job"

job=$(sbatch --parsable --job-name=x2-wno-h1 --time=08:00:00 \
    --export="ALL,$COMMON,MODEL_KIND=localwno,LOSS_H1_WEIGHT=0.001,CHECKPOINT_DIR=checkpoints/xhi2d_localwno_lr3e4_h1e3" \
    slurm/train_2d_xhi.sbatch)
jobs+=("$job")
echo "LocalWNO H1 weight=0.001:      $job"

job=$(sbatch --parsable --job-name=x2-localfno --time=08:00:00 \
    --export="ALL,$COMMON,MODEL_KIND=localfno,CHECKPOINT_DIR=checkpoints/xhi2d_localfno_lr3e4_e30" \
    slurm/train_2d_xhi.sbatch)
jobs+=("$job")
echo "LocalFNO matched baseline:     $job"

job=$(sbatch --parsable --job-name=x2-ufno --time=08:00:00 \
    --export="ALL,$COMMON,MODEL_KIND=ufno,CHECKPOINT_DIR=checkpoints/xhi2d_ufno_lr3e4_e30" \
    slurm/train_2d_xhi.sbatch)
jobs+=("$job")
echo "UFNO matched baseline:         $job"

dependencies=$(IFS=:; printf '%s' "${jobs[*]}")
viz_job=$(sbatch --parsable --dependency="afterany:$dependencies" \
    slurm/viz_xhi2d_wno_comparison.sbatch)
echo "Comparison viz (after all):    $viz_job"
echo "Training job IDs:              ${jobs[*]}"
