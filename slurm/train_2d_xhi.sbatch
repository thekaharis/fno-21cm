#!/bin/bash
#SBATCH --job-name=localwno-xhi2d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/localwno-xhi2d-%j.out
#SBATCH --error=logs/localwno-xhi2d-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

# Datasets live outside the repo; see dataset/paths.py
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/../data/compressed}"
mkdir -p logs

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

export CACHE_FILE="${CACHE_FILE:-$DATA_DIR/trainset.h5}"
export INPUT_FEATURES="${INPUT_FEATURES:-density_z_params}"
# Same operator-slot interface as the 3-D matrix: MODEL_KIND=localop pairs the
# local and global slots freely, and the named kinds are shorthand for a fixed
# pair. The trainer derives the checkpoint directory from the resolved pair, so
# a localop run lands in ..._local_<local>_<global> rather than ..._localop.
export MODEL_KIND="${MODEL_KIND:-localwno}"
if [ "$MODEL_KIND" = "localop" ]; then
    export LOCAL_OPERATOR="${LOCAL_OPERATOR:?set LOCAL_OPERATOR for MODEL_KIND=localop}"
    export GLOBAL_OPERATOR="${GLOBAL_OPERATOR:?set GLOBAL_OPERATOR for MODEL_KIND=localop}"
else
    unset LOCAL_OPERATOR GLOBAL_OPERATOR
fi
export LOCALFNO_BASE_WIDTH="${LOCALFNO_BASE_WIDTH:-32}"
export LOCALFNO_WINDOW_X="${LOCALFNO_WINDOW_X:-16}"
export LOCALFNO_WINDOW_Y="${LOCALFNO_WINDOW_Y:-16}"
export LOCALFNO_GLOBAL_MODES_X="${LOCALFNO_GLOBAL_MODES_X:-16}"
export LOCALFNO_GLOBAL_MODES_Y="${LOCALFNO_GLOBAL_MODES_Y:-16}"
export LOCALFNO_SPECTRAL_RANK="${LOCALFNO_SPECTRAL_RANK:-16}"
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-32}"
export LOCALWNO_LEVELS="${LOCALWNO_LEVELS:-2}"
# 50 epochs is the standing 2-D budget (set 2026-09-14). Across 53 runs that
# logged >=60 epochs, 83% reached their best val_l2 by epoch 50 and the median
# cost of stopping there is 0.00000, mean +0.00010 -- inside the 0.0002-0.0009
# seed-replicate floor. Note T_max=N_EPOCHS, so the cosine schedule anneals to
# zero at the budget: a 50-epoch run is NOT the first 50 epochs of a 100-epoch
# run, and is generally slightly better. Alternating waveform schedules peak
# latest and are the ones worth running longer.
export N_EPOCHS="${N_EPOCHS:-50}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export LOSS_L2_WEIGHT="${LOSS_L2_WEIGHT:-1.0}"
export LOSS_H1_WEIGHT="${LOSS_H1_WEIGHT:-0.0}"

echo "================ 2-D x_HI training ================"
echo "Host:             $(hostname)"
echo "Job ID:           ${SLURM_JOB_ID:-N/A}"
echo "CACHE_FILE:       $CACHE_FILE"
echo "CHECKPOINT_DIR:   $CHECKPOINT_DIR"
echo "Model:            $MODEL_KIND${LOCAL_OPERATOR:+ ($LOCAL_OPERATOR local / $GLOBAL_OPERATOR global)}"
echo "Input features:   $INPUT_FEATURES"
echo "Epochs / batch:   $N_EPOCHS / $BATCH_SIZE"
echo "Loss:             L2=$LOSS_L2_WEIGHT H1=$LOSS_H1_WEIGHT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || echo "(nvidia-smi unavailable)"
echo "===================================================="

python fno_xhi2d.py
