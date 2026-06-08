#!/bin/bash
# =============================================================================
# SLURM batch script -- render figures from the v3 U-FNO **global-pooling
# residual** (option E) checkpoint.
#
# Submit from the project root:
#
#     sbatch slurm/viz_ufno_v3_globalres.sbatch
#
# Differences vs slurm/viz_ufno.sbatch (the v1 U-FNO viz):
#   * UFNO_UNET_VARIANT=default (the upstream U_net), wrapped per-block in
#     GlobalResidualWrapper via UFNO_GLOBAL_RESIDUAL=true.  Both flags must
#     match training time or load_model() fails the shape-match check.
#   * Inherits v2 baseline overrides (groupnorm + N_MODES_Z=32).
#   * CHECKPOINT points at ./checkpoints_3d_ufno_v3_globalres/.
#   * VIZ_TAG=ufno-v3-globalres so figures land in a clearly-labelled folder.
#
# Pairs with slurm/viz_ufno_v3_globalres_detailed.sbatch.
# =============================================================================

#SBATCH --job-name=viz-ufno-v3-globalres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz-ufno-v3-globalres-%j.out
#SBATCH --error=logs/viz-ufno-v3-globalres-%j.err

set -eo pipefail

# --- Run from the project root ----------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs figures

# --- Conda env --------------------------------------------------------------
module load devel/miniforge
ENV_NAME="fno-env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- Thread / I/O hygiene ---------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# --- Model selection (must match training time exactly) ---------------------
export MODEL_KIND=ufno

# v3 (Tier 2) Option E: global-pooling residual on top of the default U-Net.
# Composed via two flags -- variant + residual wrapper -- both required.
export UFNO_UNET_VARIANT=default
export UFNO_GLOBAL_RESIDUAL=true

# v2 baseline overrides carried forward.
export N_MODES_Z=32
export UFNO_NORM=groupnorm

# --- Checkpoint + figure-tag wiring -----------------------------------------
export CHECKPOINT_DIR=./checkpoints_3d_ufno_v3_globalres
export CHECKPOINT="${CHECKPOINT_DIR}/best_model_state_dict.pt"
export VIZ_TAG=ufno-v3-globalres

# --- Data paths -------------------------------------------------------------
export LIGHTCONE_DIR=/pfs/10/work/hd_id260-fno_training/data/data
export CUBES_CACHE=/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5

# --- Diagnostics ------------------------------------------------------------
echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "Submit dir:         ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "MODEL_KIND:         ${MODEL_KIND}"
echo "UFNO_UNET_VARIANT:  ${UFNO_UNET_VARIANT}"
echo "UFNO_GLOBAL_RES:    ${UFNO_GLOBAL_RESIDUAL}"
echo "UFNO_NORM:          ${UFNO_NORM}"
echo "N_MODES_Z:          ${N_MODES_Z}"
echo "CHECKPOINT:         ${CHECKPOINT}"
echo "VIZ_TAG:            ${VIZ_TAG}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "================================================="

# --- Render -----------------------------------------------------------------
python visualize_3d.py
