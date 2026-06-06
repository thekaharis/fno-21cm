#!/bin/bash
# =============================================================================
# SLURM batch script -- 4-GPU H200 DDP training of U-FNO with the Tier 2
# **global-pooling residual** addition (option E).
#
# Submit from the project root:
#
#     sbatch slurm/train_ufno_v3_globalres_h200_4gpu.sbatch
#
# What this changes vs v2:
#   * UFNO_GLOBAL_RESIDUAL=true -- wraps each U-Net path with a
#     GlobalResidualWrapper that computes per-channel global mean of the
#     input, projects through a small MLP, broadcasts back, and adds to
#     the wrapped path's input.  Gives the local-feature path cone-level
#     context (e.g. "this cone is heavily reionized") that its small
#     receptive field can't see from local patches alone.
#   * UFNO_UNET_VARIANT=default -- composed with the upstream U_net.
#     If you want global residual on top of an alternative U-Net variant,
#     copy this sbatch and change UFNO_UNET_VARIANT.
#
# Inherits from v2: GroupNorm in U-Net path, H1-weighted loss, asymmetric
# Z modes.
#
# Distinct CHECKPOINT_DIR (./checkpoints_3d_ufno_v3_globalres).
# =============================================================================

#SBATCH --job-name=ufno-v3-globalres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=18:00:00
#SBATCH --output=logs/ufno-v3-globalres-%j.out
#SBATCH --error=logs/ufno-v3-globalres-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

module load devel/miniforge
ENV_NAME="fno-env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# --- Model selection --------------------------------------------------------
export MODEL_KIND=ufno
export N_EPOCHS=30

# --- v3 (Tier 2) Option E: global-pooling residual -------------------------
export UFNO_UNET_VARIANT=default
export UFNO_GLOBAL_RESIDUAL=true

# --- v2 baseline overrides (carried forward) -------------------------------
export N_MODES_Z=32
export UFNO_NORM=groupnorm
export LOSS_L2_WEIGHT=0.3
export LOSS_H1_WEIGHT=0.7
export LOSS_BCE_WEIGHT=0.0

export CHECKPOINT_DIR=./checkpoints_3d_ufno_v3_globalres

# --- Data paths + NVMe staging ---------------------------------------------
export LIGHTCONE_DIR=/pfs/10/work/hd_id260-fno_training/data/data
PROJECT_CUBES=/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5

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
        echo "No local scratch detected; reading cubes from project FS."
        export CUBES_CACHE="$PROJECT_CUBES"
    fi
else
    echo "LOCAL_SCRATCH=no: reading cubes from project FS."
    export CUBES_CACHE="$PROJECT_CUBES"
fi

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "MODEL_KIND:         ${MODEL_KIND}"
echo "N_EPOCHS:           ${N_EPOCHS}"
echo "N_MODES_Z:          ${N_MODES_Z}"
echo "UFNO_NORM:          ${UFNO_NORM}"
echo "UFNO_UNET_VARIANT:  ${UFNO_UNET_VARIANT}"
echo "UFNO_GLOBAL_RES:    ${UFNO_GLOBAL_RESIDUAL}"
echo "LOSS L2 / H1 / BCE: ${LOSS_L2_WEIGHT} / ${LOSS_H1_WEIGHT} / ${LOSS_BCE_WEIGHT}"
echo "CHECKPOINT_DIR:     ${CHECKPOINT_DIR}"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "================================================="

srun python fno_21cm_3d.py
