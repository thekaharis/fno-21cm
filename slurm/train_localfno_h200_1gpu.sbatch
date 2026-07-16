#!/bin/bash
# Single-GPU H200 training for the windowed Local-FNO U-Net (72 h walltime).
# Submit: sbatch slurm/train_localfno_h200_1gpu.sbatch

#SBATCH --job-name=localfno-3d-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/localfno3d-1gpu-%j.out
#SBATCH --error=logs/localfno3d-1gpu-%j.err

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

# 8 workers keep a single GPU fed from local scratch; 32 (the 4-GPU sizing)
# made the prefetch queue alone ~72 GB at batch size 4 and OOM-killed the job.
export NUM_WORKERS="${NUM_WORKERS:-8}"

export MODEL_KIND=localfno
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/checkpoints_3d_localfno_1gpu}"
# ~19.6 min/epoch on 4x H200 -> ~78 min/epoch on one GPU; 50 epochs keeps a
# few hours of margin inside the 72 h walltime.
export N_EPOCHS="${N_EPOCHS:-50}"

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

# Base LR; fno_21cm_3d.py applies the sqrt(world_size) rule, which is a
# no-op for a single task.
export LOCALFNO_LEARNING_RATE="${LOCALFNO_LEARNING_RATE:-1e-4}"
export LOCALFNO_H1_WARMUP_EPOCHS="${LOCALFNO_H1_WARMUP_EPOCHS:-5}"
export LOCALFNO_GRAD_CLIP_NORM="${LOCALFNO_GRAD_CLIP_NORM:-1.0}"
# "absolute" (historical) or "relative" (dimensionless; makes the weights
# below directly comparable across terms).
export LOSS_L2_MODE="${LOSS_L2_MODE:-absolute}"
export LOSS_H1_MODE="${LOSS_H1_MODE:-absolute}"
export LOSS_L2_WEIGHT="${LOSS_L2_WEIGHT:-0.5}"
export LOSS_H1_WEIGHT="${LOSS_H1_WEIGHT:-0.5}"
export LOSS_BCE_WEIGHT="${LOSS_BCE_WEIGHT:-0.0}"
export LOSS_IONIZED_WALL_WEIGHT="${LOSS_IONIZED_WALL_WEIGHT:-0.0}"
export IONIZED_WALL_KERNEL_SIZE="${IONIZED_WALL_KERNEL_SIZE:-7}"
export IONIZED_WALL_THRESHOLD="${IONIZED_WALL_THRESHOLD:-0.5}"

export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
PROJECT_CUBES="${PROJECT_CUBES:-/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5}"
LOCAL_SCRATCH="${LOCAL_SCRATCH:-yes}"
if [ "$LOCAL_SCRATCH" = "yes" ]; then
    LOCAL_CUBES="$TMPDIR/cubes_3d.h5"
    echo "Staging cube cache to local scratch: $LOCAL_CUBES"
    t0=$(date +%s)
    cp "$PROJECT_CUBES" "$LOCAL_CUBES"
    echo "  staged ($(du -h "$LOCAL_CUBES" | cut -f1)) in $(( $(date +%s) - t0 ))s"
    export CUBES_CACHE="$LOCAL_CUBES"
else
    export CUBES_CACHE="${CUBES_CACHE:-$PROJECT_CUBES}"
fi

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "================ Local-FNO training (1 GPU) ========"
echo "MODEL_KIND:         ${MODEL_KIND}"
echo "CHECKPOINT_DIR:     ${CHECKPOINT_DIR}"
echo "GLOBAL_MODES:       (${N_MODES_X}, ${N_MODES_Y}, ${N_MODES_Z})"
echo "WINDOW:             (${LOCALFNO_WINDOW_X}, ${LOCALFNO_WINDOW_Y}, ${LOCALFNO_WINDOW_Z})"
echo "LOCAL_MODES:        (${LOCALFNO_MODES_X}, ${LOCALFNO_MODES_Y}, ${LOCALFNO_MODES_Z})"
echo "BASE_WIDTH:         ${LOCALFNO_BASE_WIDTH}"
echo "SPECTRAL_RANK:      ${LOCALFNO_SPECTRAL_RANK}"
echo "PATCH_CHUNK_SIZE:   ${LOCALFNO_PATCH_CHUNK_SIZE}"
echo "N_EPOCHS:           ${N_EPOCHS}"
echo "LOSS MODES:         L2=${LOSS_L2_MODE} H1=${LOSS_H1_MODE}"
echo "LOSS WEIGHTS:       L2=${LOSS_L2_WEIGHT} H1=${LOSS_H1_WEIGHT} BCE=${LOSS_BCE_WEIGHT} WALL=${LOSS_IONIZED_WALL_WEIGHT}"
echo "CUBES_CACHE:        ${CUBES_CACHE}"
nvidia-smi -L || true
echo "===================================================="

srun python fno_21cm_3d.py
