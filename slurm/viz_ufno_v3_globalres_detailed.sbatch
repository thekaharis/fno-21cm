#!/bin/bash
# =============================================================================
# SLURM batch script -- detailed viz (16 cones, active-z slice picker) for
# the v3 U-FNO **global-pooling residual** (option E) checkpoint.
#
# Submit from the project root:
#
#     sbatch slurm/viz_ufno_v3_globalres_detailed.sbatch
#
# Differences vs slurm/viz_ufno_v3_globalres.sbatch:
#   * Runs visualize_3d_detailed.py instead of visualize_3d.py.
#   * 16 cones per split; per-cone active-z slice picker.
#   * Walltime bumped to 60 min.
#   * VIZ_TAG=ufno-v3-globalres; detailed script appends "-detailed".
# =============================================================================

#SBATCH --job-name=viz-ufno-v3-globalres-detailed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz-ufno-v3-globalres-detailed-%j.out
#SBATCH --error=logs/viz-ufno-v3-globalres-detailed-%j.err

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
export UFNO_UNET_VARIANT=default
export UFNO_GLOBAL_RESIDUAL=true
export N_MODES_Z=32
export UFNO_NORM=groupnorm

# --- Checkpoint + figure-tag wiring -----------------------------------------
export CHECKPOINT_DIR=./checkpoints_3d_ufno_v3_globalres
export CHECKPOINT="${CHECKPOINT_DIR}/model_state_dict.pt"
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
python visualize_3d_detailed.py
