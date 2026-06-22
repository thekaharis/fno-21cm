#!/bin/bash
# 4-GPU H200 DDP training for the windowed Local-FNO U-Net.
# Submit: sbatch slurm/train_localfno_h200_4gpu.sbatch

#SBATCH --job-name=localfno-3d-4gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/localfno3d-4gpu-%j.out
#SBATCH --error=logs/localfno3d-4gpu-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

export MODEL_KIND=localfno
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_3d_localfno}"
export N_EPOCHS="${N_EPOCHS:-70}"

# Whole-volume bottleneck modes at 35x35x64.
export N_MODES_X="${N_MODES_X:-16}"
export N_MODES_Y="${N_MODES_Y:-16}"
export N_MODES_Z="${N_MODES_Z:-16}"

# Local Fourier U-Net defaults.
export LOCALFNO_WINDOW_X="${LOCALFNO_WINDOW_X:-16}"
export LOCALFNO_WINDOW_Y="${LOCALFNO_WINDOW_Y:-16}"
export LOCALFNO_WINDOW_Z="${LOCALFNO_WINDOW_Z:-32}"
export LOCALFNO_MODES_X="${LOCALFNO_MODES_X:-6}"
export LOCALFNO_MODES_Y="${LOCALFNO_MODES_Y:-6}"
export LOCALFNO_MODES_Z="${LOCALFNO_MODES_Z:-12}"
export LOCALFNO_BASE_WIDTH="${LOCALFNO_BASE_WIDTH:-16}"
export LOCALFNO_SPECTRAL_RANK="${LOCALFNO_SPECTRAL_RANK:-16}"
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-128}"

export LOCALFNO_LEARNING_RATE="${LOCALFNO_LEARNING_RATE:-1e-4}"
export LOCALFNO_H1_WARMUP_EPOCHS="${LOCALFNO_H1_WARMUP_EPOCHS:-5}"
export LOCALFNO_GRAD_CLIP_NORM="${LOCALFNO_GRAD_CLIP_NORM:-1.0}"
export LOSS_L2_WEIGHT="${LOSS_L2_WEIGHT:-0.5}"
export LOSS_H1_WEIGHT="${LOSS_H1_WEIGHT:-0.5}"
export LOSS_BCE_WEIGHT="${LOSS_BCE_WEIGHT:-0.0}"

export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
PROJECT_CUBES="${PROJECT_CUBES:-/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5}"
LOCAL_SCRATCH="${LOCAL_SCRATCH:-yes}"
if [ "$LOCAL_SCRATCH" = "yes" ]; then
    SCRATCH_BASE=""
    for candidate in "${TMPDIR:-}" "/scratch/${SLURM_JOB_ID}" "/local/${SLURM_JOB_ID}"; do
        if [ -n "$candidate" ] && [ -d "$candidate" ] && [ -w "$candidate" ]; then
            SCRATCH_BASE="$candidate"
            break
        fi
    done
    if [ -n "$SCRATCH_BASE" ]; then
        LOCAL_CUBES="$SCRATCH_BASE/cubes_3d.h5"
        cp --reflink=auto "$PROJECT_CUBES" "$LOCAL_CUBES" || cp "$PROJECT_CUBES" "$LOCAL_CUBES"
        export CUBES_CACHE="$LOCAL_CUBES"
    else
        export CUBES_CACHE="$PROJECT_CUBES"
    fi
else
    export CUBES_CACHE="${CUBES_CACHE:-$PROJECT_CUBES}"
fi

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "================ Local-FNO training ==============="
echo "MODEL_KIND:         ${MODEL_KIND}"
echo "CHECKPOINT_DIR:     ${CHECKPOINT_DIR}"
echo "GLOBAL_MODES:       (${N_MODES_X}, ${N_MODES_Y}, ${N_MODES_Z})"
echo "WINDOW:             (${LOCALFNO_WINDOW_X}, ${LOCALFNO_WINDOW_Y}, ${LOCALFNO_WINDOW_Z})"
echo "LOCAL_MODES:        (${LOCALFNO_MODES_X}, ${LOCALFNO_MODES_Y}, ${LOCALFNO_MODES_Z})"
echo "BASE_WIDTH:         ${LOCALFNO_BASE_WIDTH}"
echo "SPECTRAL_RANK:      ${LOCALFNO_SPECTRAL_RANK}"
echo "PATCH_CHUNK_SIZE:   ${LOCALFNO_PATCH_CHUNK_SIZE}"
echo "N_EPOCHS:           ${N_EPOCHS}"
echo "CUBES_CACHE:        ${CUBES_CACHE}"
nvidia-smi -L || true
echo "===================================================="

srun python fno_21cm_3d.py
