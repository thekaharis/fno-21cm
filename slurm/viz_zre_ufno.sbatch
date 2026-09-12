#!/bin/bash
# =============================================================================
# SLURM batch script — detailed viz for the density -> z_re(x, y) models.
#
# Renders 16 stratified cones per held-out split from a trained fno_zre.py
# checkpoint: per-cone map triptychs + scatters, a per-split summary grid,
# power spectra, and metrics.json.  Figures land in a unique
# figures/zre/detailed_viz/zre-<MODEL_KIND>-detailed_<timestamp>_job<id>/ folder.
#
# Submit from the project root, exporting the SAME experiment switches the
# training run used (defaults match slurm/train_zre.sbatch):
#
#     sbatch slurm/viz_zre_ufno.sbatch
#     sbatch --export=ALL,MODEL_KIND=ufno slurm/viz_zre.sbatch
#     sbatch --export=ALL,MODEL_KIND=localfno,CHECKPOINT_KIND=final slurm/viz_zre.sbatch
# =============================================================================

# NOTE: do NOT add trailing "# ..." comments on #SBATCH lines — some SLURM
# versions parse them as part of the option value and reject the job.
#
# Rendering 32 cones through a 2-D model is light; an A30 finishes in
# minutes.  Most wall time is the one-time dataset scan (params + targets).
#
#SBATCH --job-name=viz-zre-ufno
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz-zre-ufno-%j.out
#SBATCH --error=logs/viz-zre-ufno-%j.err

set -eo pipefail   # (-u omitted: conda's activate scripts trip on unbound vars)

# --- Run from the project root ----------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

# Datasets live outside the repo; see dataset/paths.py
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/../data/compressed}"
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

# --- Data & experiment configuration (keep in sync with train_zre.sbatch) ---
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
export ZRE_TARGET_CACHE="${ZRE_TARGET_CACHE:-$DATA_DIR/zre_targets.h5}"
export MODEL_KIND="${MODEL_KIND:-ufno}"
export TARGET_KIND="${TARGET_KIND:-gompertz}"
export INPUT_FEATURES="${INPUT_FEATURES:-density_params}"
export N_Z_IN="${N_Z_IN:-64}"
# CHECKPOINT_DIR is derived per MODEL_KIND inside fno_zre.py unless
# overridden here; CHECKPOINT / CHECKPOINT_KIND select the weights file.

# --- Diagnostics ------------------------------------------------------------
echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "Submit dir:         ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
echo "Model kind:         $MODEL_KIND"
echo "Target kind:        $TARGET_KIND"
echo "Lightcones:         $LIGHTCONE_DIR"
echo "Target cache:       $ZRE_TARGET_CACHE"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "================================================="

# --- Render -----------------------------------------------------------------
python -m viz.visualize_zre_detailed
