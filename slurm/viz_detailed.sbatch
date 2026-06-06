#!/bin/bash
# =============================================================================
# SLURM batch script -- detailed viz (16 cones, active-z slices) for the
# plain-FNO checkpoint in ./checkpoints_3d/.
#
# Submit from the project root:
#
#     sbatch slurm/viz_detailed.sbatch
#
# Differences vs slurm/viz.sbatch:
#   * Runs visualize_3d_detailed.py instead of visualize_3d.py
#     -> 16 cones per split (vs 4) at evenly spaced percentiles of
#        per-cone mean ionization
#     -> per-cone active-z slice picker (skips fully-neutral / fully-
#        ionized LOS stretches that carry no info about model fidelity)
#   * Output goes to a per-run subfolder under figures/ (e.g.
#     figures/fno-detailed_20260606-143022_job12345/) so successive viz
#     runs never overwrite each other
#   * Walltime bumped from 30 min -> 60 min: 16 forward passes per split +
#     4 PNGs per cone + summary grid is ~30-40 min on an A30
#
# For the U-FNO checkpoint, use slurm/viz_ufno_detailed.sbatch.
# =============================================================================

#SBATCH --job-name=viz-3d-detailed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz-detailed-%j.out
#SBATCH --error=logs/viz-detailed-%j.err

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

# --- Data paths -------------------------------------------------------------
export LIGHTCONE_DIR=/pfs/10/work/hd_id260-fno_training/data/data
export CUBES_CACHE=/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5

# --- Diagnostics ------------------------------------------------------------
echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "Submit dir:         ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "================================================="

# --- Render -----------------------------------------------------------------
python visualize_3d_detailed.py
