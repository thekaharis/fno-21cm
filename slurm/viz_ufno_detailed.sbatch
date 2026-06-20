#!/bin/bash
# =============================================================================
# SLURM batch script -- detailed viz (16 cones, active-z slices) for the
# U-FNO checkpoint in ./checkpoints_3d_ufno/.
#
# Submit from the project root:
#
#     sbatch slurm/viz_ufno_detailed.sbatch
#
# Differences vs slurm/viz_ufno.sbatch:
#   * Runs visualize_3d_detailed.py instead of visualize_3d.py
#   * 16 cones per split (vs 4); active-z slice picker per cone.
#   * Output goes to a per-run subfolder under figures/ (e.g.
#     figures/ufno-detailed_20260606-143022_job12345/).
#   * Walltime bumped to 60 min.
#
# Differences vs slurm/viz_detailed.sbatch:
#   * MODEL_KIND=ufno -> loads checkpoints_3d_ufno/best_model_state_dict.pt and
#     reconstructs the U-FNO architecture in load_model().
#
# This is one of two viz workflows -- the 4-cone summary is via
# slurm/viz_ufno.sbatch.
# =============================================================================

#SBATCH --job-name=viz-ufno-3d-detailed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz-ufno-detailed-%j.out
#SBATCH --error=logs/viz-ufno-detailed-%j.err

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

# --- Model selection --------------------------------------------------------
# Must be exported before python imports fno_21cm_3d (visualize_3d_detailed
# imports visualize_3d which imports MODEL_KIND at import time).
export MODEL_KIND=ufno

# --- Data paths -------------------------------------------------------------
export LIGHTCONE_DIR=/pfs/10/work/hd_id260-fno_training/data/data
export CUBES_CACHE=/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5

# --- Diagnostics ------------------------------------------------------------
echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "Submit dir:         ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "MODEL_KIND:         ${MODEL_KIND}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "================================================="

# --- Render -----------------------------------------------------------------
python visualize_3d_detailed.py
