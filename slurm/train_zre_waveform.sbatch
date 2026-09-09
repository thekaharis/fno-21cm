#!/bin/bash
# Submit from the project root after mkdir -p logs.
# sbatch --export=ALL,N_EPOCHS=200,LEARNING_RATE=1e-4 slurm/train_zre_waveform.sbatch
# Absolute L2 by default, matching the existing z_re L2 ablations.
#SBATCH --job-name=waveform-zre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/waveform-zre-%j.out
#SBATCH --error=logs/waveform-zre-%j.err

set -eo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MODEL_KIND=localop LOCAL_OPERATOR=learned_waveform GLOBAL_OPERATOR=learned_waveform
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/checkpoints_zre_local_lwf_lwf_l2}"
export WAVEFORM_LOCAL_BINS="${WAVEFORM_LOCAL_BINS:-15}"
export WAVEFORM_GLOBAL_BINS="${WAVEFORM_GLOBAL_BINS:-31}"
export WAVEFORM_CONDITION_LIMIT="${WAVEFORM_CONDITION_LIMIT:-10000}"
export WAVEFORM_LR_RATIO="${WAVEFORM_LR_RATIO:-0.1}"
export LOCALFNO_BASE_WIDTH="${LOCALFNO_BASE_WIDTH:-32}"
export LOCALFNO_SPECTRAL_RANK="${LOCALFNO_SPECTRAL_RANK:-16}"
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-32}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export LOSS_L2_WEIGHT="${LOSS_L2_WEIGHT:-1.0}" LOSS_H1_WEIGHT="${LOSS_H1_WEIGHT:-0.0}"
export LOSS_L1_WEIGHT="${LOSS_L1_WEIGHT:-0.0}" LOSS_H2_WEIGHT="${LOSS_H2_WEIGHT:-0.0}"
export LOSS_RELATIVE="${LOSS_RELATIVE:-0}"
exec bash "$PROJECT_ROOT/slurm/train_zre.sbatch"
