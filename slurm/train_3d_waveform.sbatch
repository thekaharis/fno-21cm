#!/bin/bash
# Submit from the project root after mkdir -p logs.
# sbatch --export=ALL,N_EPOCHS=20,LOSS=plain slurm/train_3d_waveform.sbatch
# Uses the existing matrix runner's data staging and loss presets. Resource
# requests follow that runner; waveform GPU throughput/memory need profiling.
#SBATCH --job-name=waveform-3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G
#SBATCH --time=96:00:00
#SBATCH --output=logs/waveform-3d-%j.out
#SBATCH --error=logs/waveform-3d-%j.err

set -eo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ARCH=lwf_lwf
export LOSS="${LOSS:-plain}"
export WAVEFORM_LOCAL_BINS="${WAVEFORM_LOCAL_BINS:-15}"
export WAVEFORM_GLOBAL_BINS="${WAVEFORM_GLOBAL_BINS:-31}"
export WAVEFORM_CONDITION_LIMIT="${WAVEFORM_CONDITION_LIMIT:-10000}"
export WAVEFORM_LR_RATIO="${WAVEFORM_LR_RATIO:-0.1}"
export WAVEFORM_INIT="${WAVEFORM_INIT:-random}"
export WAVEFORM_TRANSFORM="${WAVEFORM_TRANSFORM:-tied}"
if [[ "$WAVEFORM_TRANSFORM" == "separate" ]]; then
  export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/3d_xhi/${ARCH}/checkpoints_3d_${ARCH}_${LOSS}_separate}"
fi
export WAVEFORM_TRAINING_MODE="${WAVEFORM_TRAINING_MODE:-joint}"
export WAVEFORM_ADAPT_EPOCHS="${WAVEFORM_ADAPT_EPOCHS:-25}"
export WAVEFORM_PHASE_EPOCHS="${WAVEFORM_PHASE_EPOCHS:-1}"
export WAVEFORM_KERNEL_EPOCHS="${WAVEFORM_KERNEL_EPOCHS:-5}"
export WAVEFORM_FIRST_PHASE="${WAVEFORM_FIRST_PHASE:-waveform}"
export WAVEFORM_KERNEL_SCOPE="${WAVEFORM_KERNEL_SCOPE:-spectral}"
# The 3-D entry point currently fixes the per-rank batch size at one.
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-32}"
export LOCALFNO_GRAD_CLIP_NORM="${LOCALFNO_GRAD_CLIP_NORM:-1.0}"
exec bash "$PROJECT_ROOT/slurm/train_3d_matrix.sbatch"
