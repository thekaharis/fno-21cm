#!/bin/bash
# =============================================================================
# SLURM batch script -- render figures from the latest 3-D FNO checkpoint.
#
# Submit from the project root:
#
#     sbatch slurm/viz.sbatch
#
# Renders for each held-out split (val + test):
#   * figures/comparison_3d_<split>.png  -- N z-slices side-by-side
#   * figures/lightcone_3d_<split>.png   -- xz strip across the full LOS
#   * figures/scatter_3d_<split>.png     -- voxel scatter, R^2 + RMSE
#
# Resource sizing: a single A30 is plenty -- viz predicts one cube per split
# (1, 13, 140, 140, 256) with no backward pass, finishes in well under a minute
# of GPU time.  Most of the wall time is matplotlib + the cube load.
#
# If you bumped HIDDEN_CHANNELS or N_MODES in fno_21cm_3d.py, viz now reads
# those constants directly, so no edit is needed here as long as the checkpoint
# matches the current values.
# =============================================================================

#SBATCH --job-name=viz-3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz-%j.out
#SBATCH --error=logs/viz-%j.err

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
# Same paths as train.sbatch / train_h200_4gpu.sbatch so the deterministic
# val/test split (driven by len(dataset) + SPLIT_SEED) reproduces the same
# held-out cones used during training.
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
python visualize_3d.py
