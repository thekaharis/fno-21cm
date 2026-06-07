#!/bin/bash
# =============================================================================
# SLURM batch script -- render figures from the v3 U-FNO **1-D LOS conv
# path** (option F) checkpoint.
#
# Submit from the project root:
#
#     sbatch slurm/viz_ufno_v3_los1d.sbatch
#
# Differences vs slurm/viz_ufno.sbatch (the v1 U-FNO viz):
#   * UFNO_UNET_VARIANT=los1d -- replaces the upstream U_net with the
#     LOSConv1DPath stack of 1-D Z-only Conv3d layers (kernel=(1,1,7),
#     n_layers=4).  Must match training time or load_model() fails.
#   * Inherits v2 baseline overrides (groupnorm + N_MODES_Z=32).
#   * CHECKPOINT points at ./checkpoints_3d_ufno_v3_los1d/.
#   * VIZ_TAG=ufno-v3-los1d so the figure folder name encodes the variant.
#
# Pairs with slurm/viz_ufno_v3_los1d_detailed.sbatch.
# =============================================================================

#SBATCH --job-name=viz-ufno-v3-los1d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz-ufno-v3-los1d-%j.out
#SBATCH --error=logs/viz-ufno-v3-los1d-%j.err

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

# v3 (Tier 2) Option F: 1-D LOS conv path.  Much fewer parameters than the
# 3-D U-Net (~28K vs ~250K per path); load_model() expects this exact
# topology when reading the checkpoint.
export UFNO_UNET_VARIANT=los1d

# v2 baseline overrides carried forward.
export N_MODES_Z=32
export UFNO_NORM=groupnorm

# --- Checkpoint + figure-tag wiring -----------------------------------------
export CHECKPOINT_DIR=./checkpoints_3d_ufno_v3_los1d
export CHECKPOINT="${CHECKPOINT_DIR}/model_state_dict.pt"
export VIZ_TAG=ufno-v3-los1d

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
