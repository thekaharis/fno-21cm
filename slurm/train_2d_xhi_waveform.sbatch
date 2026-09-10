#!/bin/bash
# Submit from the project root after mkdir -p logs.
# sbatch --export=ALL,N_EPOCHS=20,BATCH_SIZE=8 slurm/train_2d_xhi_waveform.sbatch
# Existing 2-D data/environment setup; both operator slots are learned waveforms.
#SBATCH --job-name=waveform-xhi2d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/waveform-xhi2d-%j.out
#SBATCH --error=logs/waveform-xhi2d-%j.err

set -eo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MODEL_KIND=localop LOCAL_OPERATOR=learned_waveform GLOBAL_OPERATOR=learned_waveform
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/checkpoints_2d_xhi_local_lwf_lwf}"
export WAVEFORM_LOCAL_BINS="${WAVEFORM_LOCAL_BINS:-15}"
export WAVEFORM_GLOBAL_BINS="${WAVEFORM_GLOBAL_BINS:-31}"
export WAVEFORM_CONDITION_LIMIT="${WAVEFORM_CONDITION_LIMIT:-10000}"
export WAVEFORM_LR_RATIO="${WAVEFORM_LR_RATIO:-0.1}"
export WAVEFORM_INIT="${WAVEFORM_INIT:-random}"
export WAVEFORM_TRAINING_MODE="${WAVEFORM_TRAINING_MODE:-joint}"
export WAVEFORM_PHASE_EPOCHS="${WAVEFORM_PHASE_EPOCHS:-1}"
export WAVEFORM_KERNEL_EPOCHS="${WAVEFORM_KERNEL_EPOCHS:-5}"
export WAVEFORM_FIRST_PHASE="${WAVEFORM_FIRST_PHASE:-waveform}"
export WAVEFORM_KERNEL_SCOPE="${WAVEFORM_KERNEL_SCOPE:-spectral}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
exec bash "$PROJECT_ROOT/slurm/train_2d_xhi.sbatch"
