#!/bin/bash
# =============================================================================
# SLURM batch script -- render figures from the latest U-FNO checkpoint.
#
# Submit from the project root:
#
#     sbatch slurm/viz_ufno.sbatch
#
# Differences vs slurm/viz.sbatch (the plain-FNO variant):
#   * Exports MODEL_KIND=ufno so visualize_3d.py constructs the U-FNO
#     architecture in load_model() and looks for the checkpoint in
#     ./checkpoints_3d_ufno/ instead of ./checkpoints_3d/.
#   * Distinct job-name and log-output paths so the FNO and U-FNO viz jobs
#     stay separable in `squeue -u $USER -l` and `ls logs/`.
#
# Why this needs to be a separate sbatch and not a `MODEL_KIND=ufno sbatch
# slurm/viz.sbatch` invocation: sbatch only propagates env vars that are
# explicitly listed in the script (or via --export=ALL,KEY=VAL).  Setting
# MODEL_KIND in the submitting shell does NOT carry into the SLURM job.
#
# Everything else (resource sizing, conda env, data paths) mirrors viz.sbatch
# -- changes there should propagate here.
# =============================================================================

#SBATCH --job-name=viz-ufno-3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz-ufno-%j.out
#SBATCH --error=logs/viz-ufno-%j.err

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
# The single switch that distinguishes this viz sbatch from the FNO one.
# visualize_3d.py reads MODEL_KIND at import time (via fno_21cm_3d) and:
#   * picks UFNOWrapped in load_model() instead of neuralop.FNO
#   * auto-selects CHECKPOINT="checkpoints_3d_ufno/best_model_state_dict.pt"
#     (unless CHECKPOINT is overridden in this env block).
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
python visualize_3d.py
