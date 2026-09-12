#!/bin/bash
#SBATCH --job-name=localwno-zre-l2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/localwno-zre-l2-%j.out
#SBATCH --error=logs/localwno-zre-l2-%j.err

set -eo pipefail

export MODEL_KIND="${MODEL_KIND:-localwno}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/zre/localwno/checkpoints_zre_localwno_bw32_l2_lr3e4}"
export LOCALFNO_BASE_WIDTH="${LOCALFNO_BASE_WIDTH:-32}"
export LOCALFNO_WINDOW_X="${LOCALFNO_WINDOW_X:-16}"
export LOCALFNO_WINDOW_Y="${LOCALFNO_WINDOW_Y:-16}"
export LOCALFNO_GLOBAL_MODES_X="${LOCALFNO_GLOBAL_MODES_X:-16}"
export LOCALFNO_GLOBAL_MODES_Y="${LOCALFNO_GLOBAL_MODES_Y:-16}"
export LOCALFNO_SPECTRAL_RANK="${LOCALFNO_SPECTRAL_RANK:-16}"
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-32}"
export LOCALWNO_LEVELS="${LOCALWNO_LEVELS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export N_EPOCHS="${N_EPOCHS:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export LOSS_L2_WEIGHT="${LOSS_L2_WEIGHT:-1.0}"
export LOSS_H1_WEIGHT="${LOSS_H1_WEIGHT:-0.0}"
export LOSS_L1_WEIGHT="${LOSS_L1_WEIGHT:-0.0}"
export LOSS_H2_WEIGHT="${LOSS_H2_WEIGHT:-0.0}"
export LOSS_RELATIVE="${LOSS_RELATIVE:-0}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
exec bash "$PROJECT_ROOT/slurm/train_zre_localsirenfno_a100_l2only.sbatch"
