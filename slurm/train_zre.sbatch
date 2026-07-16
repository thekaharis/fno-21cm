#!/bin/bash
# =============================================================================
# SLURM batch script — train the 2-D FNO mapping density cones to z_re(x, y).
#
# Submit from the project directory (the one containing fno_zre.py):
#
#     sbatch slurm/train_zre.sbatch
#
# The job reads raw lightcone .h5 files from LIGHTCONE_DIR; no cube cache is
# needed (the dataset preloads N_Z_IN density slices per cone, ~5 MB each).
# The z_re target maps are fitted on first use and cached in ZRE_TARGET_CACHE;
# subsequent runs reuse the cache, so only the first submission pays the
# fitting cost (a few seconds per cone, CPU-only, BLAS-threaded).
#
# Experiment switches are plain env vars (see fno_zre.py header), e.g.:
#
#     sbatch --export=ALL,TARGET_KIND=step,N_EPOCHS=500 slurm/train_zre.sbatch
# =============================================================================

# NOTE: do NOT add trailing "# ..." comments on #SBATCH lines — some SLURM
# versions parse them as part of the option value and reject the job.
#
# This is a light single-GPU job: a 2-D FNO on 140x140 maps.  An A30 is
# ample; switch the gres to gpu:h200:1 only if the A30 queue is congested.
#
#SBATCH --job-name=fno-zre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fno-zre-%j.out
#SBATCH --error=logs/fno-zre-%j.err

set -eo pipefail   # (-u omitted: conda's activate scripts trip on unbound vars)

# --- Run from the project root ----------------------------------------------
# Same convention as slurm/train.sbatch: submit from the project root, and
# fall back to this script's own location when run outside SLURM.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

# --- Activate the conda environment -----------------------------------------
module load devel/miniforge
ENV_NAME="fno-env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- Thread hygiene ---------------------------------------------------------
# The one-time z_re target fitting is a large BLAS matmul per cone and uses
# these threads; training itself is GPU-bound.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# --- Data & experiment configuration ----------------------------------------
# Raw lightcones (same directory the 3-D pipeline uses; keep in sync with
# build.sbatch). Overridable at submit time via --export.
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"

# Fitted z_re target maps live on the work FS next to the checkpoints so
# every job (and the viz scripts) share one cache. Delete the file to force
# a refit after changing the fit grids in dataset/zre_target.py.
export ZRE_TARGET_CACHE="${ZRE_TARGET_CACHE:-$PROJECT_ROOT/zre_targets.h5}"

# Defaults mirror fno_zre.py; listed here so an experiment can flip them via
# `sbatch --export=ALL,...` without editing files.
export TARGET_KIND="${TARGET_KIND:-gompertz}"
export INPUT_FEATURES="${INPUT_FEATURES:-density_params}"
export N_Z_IN="${N_Z_IN:-64}"
export N_EPOCHS="${N_EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/checkpoints_zre}"

# --- Diagnostics (captured in the .out log) ---------------------------------
echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "Submit dir:         ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
echo "Lightcones:         $LIGHTCONE_DIR"
echo "Target cache:       $ZRE_TARGET_CACHE"
echo "Target kind:        $TARGET_KIND"
echo "Input features:     $INPUT_FEATURES"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || echo "(nvidia-smi unavailable)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE'))"
echo "================================================="

# --- Train ------------------------------------------------------------------
# Plain `python` (no `srun`): single-process, single-GPU job.
python fno_zre.py
