#!/bin/bash
# =============================================================================
# SLURM batch script -- 4-GPU H200 DDP training of the **U-FNO** variant.
#
# Submit from the project root:
#
#     sbatch slurm/train_ufno_h200_4gpu.sbatch
#
# Differences vs slurm/train_h200_4gpu.sbatch (the plain-FNO variant):
#   * Exports MODEL_KIND=ufno so fno_21cm_3d.py constructs Wen et al.'s
#     U-FNO (3 FNO blocks + 3 U-Fourier blocks with a mini 3-D U-Net path
#     inside each) instead of the neuralop FNO.
#   * N_EPOCHS=30 -- this is the first U-FNO experiment, keep it short to
#     decide whether the architecture breaks the parameter-conditioning
#     floor (val_l2 ~= 0.058, val_h1 ~= 11.5) seen with the plain FNO.
#   * Walltime padded to 18 h to absorb the ~2x per-epoch slowdown from the
#     6-block + U-Net body.  Expected: ~7-10 min/epoch * 30 epochs ~= 3-5 h
#     train + 1 h cube-cache staging = comfortably inside 18 h.
#   * Distinct job name, log paths, and CHECKPOINT_DIR (./checkpoints_3d_ufno
#     auto-selected by fno_21cm_3d.py because MODEL_KIND=ufno) so the FNO
#     baseline is never overwritten.
#
# Everything else (NVMe staging, MASTER_ADDR/PORT, srun launch, conda env)
# mirrors the plain-FNO sbatch -- changes there should propagate here.
# =============================================================================

#SBATCH --job-name=ufno-3d-4gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=18:00:00
#SBATCH --output=logs/ufno3d-4gpu-%j.out
#SBATCH --error=logs/ufno3d-4gpu-%j.err

set -eo pipefail

# --- Run from the project root ----------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

# --- Conda env --------------------------------------------------------------
module load devel/miniforge
ENV_NAME="fno-env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- Thread / I/O hygiene ---------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# --- Model selection --------------------------------------------------------
# The single switch that distinguishes this sbatch from the FNO one.
# fno_21cm_3d.py reads MODEL_KIND at import time and (a) picks the model
# constructor and (b) selects CHECKPOINT_DIR="./checkpoints_3d_ufno" so
# this run never collides with the plain-FNO baseline.
export MODEL_KIND=ufno

# 30 epochs is the first-pass experimental budget for U-FNO.  If the
# trajectory looks promising at the end, extend by submitting again with
# N_EPOCHS=70 (the previous CosineAnnealingLR T_max=100 isn't quite
# meaningful when we stop short, but fine for a first cut).
export N_EPOCHS=30

# --- Data paths -------------------------------------------------------------
export LIGHTCONE_DIR=/pfs/10/work/hd_id260-fno_training/data/data
PROJECT_CUBES=/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5

# Stage cube cache on local NVMe (mirrors the FNO sbatch -- see that file
# for the rationale).  ~5-10 min one-time copy of 180 GB; reads run at
# ~3 GB/s after.
LOCAL_SCRATCH="${LOCAL_SCRATCH:-yes}"
if [ "$LOCAL_SCRATCH" = "yes" ]; then
    SCRATCH_BASE=""
    for candidate in "${TMPDIR:-}" "/scratch/${SLURM_JOB_ID}" "/local/${SLURM_JOB_ID}"; do
        if [ -n "$candidate" ] && [ -d "$candidate" ] && [ -w "$candidate" ]; then
            SCRATCH_BASE="$candidate"; break
        fi
    done
    if [ -n "$SCRATCH_BASE" ]; then
        LOCAL_CUBES="$SCRATCH_BASE/cubes_3d.h5"
        echo "Staging cube cache to local NVMe: $LOCAL_CUBES"
        t0=$(date +%s)
        cp --reflink=auto "$PROJECT_CUBES" "$LOCAL_CUBES" || cp "$PROJECT_CUBES" "$LOCAL_CUBES"
        echo "  staged ($(du -h "$LOCAL_CUBES" | cut -f1)) in $(( $(date +%s) - t0 ))s"
        export CUBES_CACHE="$LOCAL_CUBES"
    else
        echo "No local scratch detected (TMPDIR/scratch unset); reading cubes from project FS."
        export CUBES_CACHE="$PROJECT_CUBES"
    fi
else
    echo "LOCAL_SCRATCH=no: reading cubes from project FS."
    export CUBES_CACHE="$PROJECT_CUBES"
fi

# --- Distributed master coordinates -----------------------------------------
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- Diagnostics ------------------------------------------------------------
echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "Submit dir:         ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Tasks (ranks):      ${SLURM_NTASKS:-1}"
echo "Tasks/node:         ${SLURM_NTASKS_PER_NODE:-1}"
echo "CPUs/task:          ${SLURM_CPUS_PER_TASK:-1}"
echo "MASTER_ADDR:        ${MASTER_ADDR}"
echo "MASTER_PORT:        ${MASTER_PORT}"
echo "MODEL_KIND:         ${MODEL_KIND}"
echo "N_EPOCHS:           ${N_EPOCHS}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "================================================="

# --- Launch -----------------------------------------------------------------
srun python fno_21cm_3d.py
