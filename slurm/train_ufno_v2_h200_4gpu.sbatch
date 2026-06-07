#!/bin/bash
# =============================================================================
# SLURM batch script -- 4-GPU H200 DDP training of the **U-FNO v2** variant.
#
# Submit from the project root:
#
#     sbatch slurm/train_ufno_v2_h200_4gpu.sbatch
#
# v1 vs v2 (the "A+B+C bundle" -- three independent improvements all
# bundled into one run because they're cheap, low-risk, and each touches
# a different part of the system):
#
#   A. Asymmetric Z modes: N_MODES_Z=32 (vs 16) -- LOS direction has more
#      step-function content from the reionization transition; bumping
#      modes targets that specifically.  FFT cost is unchanged (it's over
#      the whole grid regardless of how many modes we keep).
#
#   B. GroupNorm in the U-Net path (UFNO_NORM=groupnorm).  Replaces every
#      BatchNorm3d in models_ufno's U-Net with a matching GroupNorm at
#      construction time (post-construction patch; ufno.py untouched).
#      GroupNorm is batch-independent -- sidesteps the small-batch + DDP
#      foot-gun that motivated wrapping the v1 model in SyncBatchNorm.
#      SyncBN call is still applied downstream; it's a no-op on GroupNorm
#      modules so it costs nothing.
#
#   C. H1-weighted loss: 0.3*L2 + 0.7*H1 (vs 0.5/0.5).  Pushes the
#      optimiser harder on gradient fidelity, which is where v1 U-FNO
#      already shines (val_h1 = 8.27 vs FNO floor 11.36).  Should sharpen
#      bubble walls further without trading off val_l2 significantly.
#
# Distinct CHECKPOINT_DIR (./checkpoints_3d_ufno_v2) so v1 and v2 don't
# overwrite each other.  Reuses the v1 sbatch's NVMe staging, master-addr
# wiring, and resource sizing; only env-var overrides + the job name and
# log paths differ.
# =============================================================================

#SBATCH --job-name=ufno-v2-3d-4gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=18:00:00
#SBATCH --output=logs/ufno-v2-3d-4gpu-%j.out
#SBATCH --error=logs/ufno-v2-3d-4gpu-%j.err

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
export MODEL_KIND=ufno
export N_EPOCHS=30

# --- v2 A+B+C bundle overrides ---------------------------------------------
# A: asymmetric Z modes (16, 16, 32) vs v1's (16, 16, 16)
export N_MODES_Z=32

# B: GroupNorm instead of BatchNorm inside the U-Net path
export UFNO_NORM=groupnorm

# C: H1-weighted loss recipe
export LOSS_L2_WEIGHT=0.3
export LOSS_H1_WEIGHT=0.7
export LOSS_BCE_WEIGHT=0.0

# bf16 mixed precision -- ~1.5-2x speedup on H200 tensor cores, no
# GradScaler needed (bf16 keeps fp32 exponent range).  Numerically robust
# for FFTs.  Disable by setting AUTOCAST=off here if you want pure fp32.
export AUTOCAST=bf16

# Separate checkpoint dir so v1 and v2 don't collide.  fno_21cm_3d.py
# honors CHECKPOINT_DIR if set; otherwise it picks ./checkpoints_3d_ufno
# from MODEL_KIND=ufno.
export CHECKPOINT_DIR=./checkpoints_3d_ufno_v2

# --- Data paths -------------------------------------------------------------
export LIGHTCONE_DIR=/pfs/10/work/hd_id260-fno_training/data/data
PROJECT_CUBES=/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5

# Stage cube cache on local NVMe (28 TB on the H200 node).
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
echo "N_MODES_Z:          ${N_MODES_Z}"
echo "UFNO_NORM:          ${UFNO_NORM}"
echo "LOSS L2 / H1 / BCE: ${LOSS_L2_WEIGHT} / ${LOSS_H1_WEIGHT} / ${LOSS_BCE_WEIGHT}"
echo "AUTOCAST:           ${AUTOCAST:-off}"
echo "CHECKPOINT_DIR:     ${CHECKPOINT_DIR}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "================================================="

# --- Launch -----------------------------------------------------------------
srun python fno_21cm_3d.py
