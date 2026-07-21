#!/bin/bash
# =============================================================================
# SLURM batch script — train the 2-D LocalSirenFNO mapping density cones to
# z_re(x, y): the windowed Local-FNO with SIREN-generated spectral weights in
# every branch.
#
# Submit from the project directory (the one containing fno_zre.py):
#
#     sbatch slurm/train_zre_localsirenfno.sbatch
#
# The job reads raw lightcone .h5 files from LIGHTCONE_DIR; no cube cache is
# needed. The z_re target maps are fitted on first use and cached in
# ZRE_TARGET_CACHE; subsequent runs (and the viz scripts) reuse the cache.
#
# Experiment switches are plain env vars (see fno_zre.py header), e.g.:
#
#     sbatch --export=ALL,SIREN_OMEGA=20,N_EPOCHS=300 slurm/train_zre_localsirenfno.sbatch
# =============================================================================

# NOTE: do NOT add trailing "# ..." comments on #SBATCH lines — some SLURM
# versions parse them as part of the option value and reject the job.
#
# Single-GPU job: a 2-D model on 140x140 maps. An A30 is ample.
#
#SBATCH --job-name=localsirenfno-zre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/localsirenfno-zre-%j.out
#SBATCH --error=logs/localsirenfno-zre-%j.err

set -eo pipefail   # (-u omitted: conda's activate scripts trip on unbound vars)

# --- Run from the project root ----------------------------------------------
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# --- Data & experiment configuration ----------------------------------------
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
export ZRE_TARGET_CACHE="${ZRE_TARGET_CACHE:-$PROJECT_ROOT/zre_targets.h5}"

# MODEL_KIND picks the architecture; CHECKPOINT_DIR is deliberately NOT set
# here -- fno_zre.py derives a per-kind default
# (checkpoints/checkpoints_zre_localsirenfno) so runs never overwrite others.
export MODEL_KIND="${MODEL_KIND:-localsirenfno}"
export TARGET_KIND="${TARGET_KIND:-gompertz}"
export INPUT_FEATURES="${INPUT_FEATURES:-density_params}"
export N_Z_IN="${N_Z_IN:-64}"
export N_EPOCHS="${N_EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-8}"

# Local Fourier U-Net geometry (matches train_zre_localfno.sbatch defaults).
export LOCALFNO_BASE_WIDTH="${LOCALFNO_BASE_WIDTH:-16}"
export LOCALFNO_WINDOW_X="${LOCALFNO_WINDOW_X:-16}"
export LOCALFNO_WINDOW_Y="${LOCALFNO_WINDOW_Y:-16}"
export LOCALFNO_MODES_X="${LOCALFNO_MODES_X:-6}"
export LOCALFNO_MODES_Y="${LOCALFNO_MODES_Y:-6}"
export LOCALFNO_GLOBAL_MODES_X="${LOCALFNO_GLOBAL_MODES_X:-16}"
export LOCALFNO_GLOBAL_MODES_Y="${LOCALFNO_GLOBAL_MODES_Y:-16}"
export LOCALFNO_SPECTRAL_RANK="${LOCALFNO_SPECTRAL_RANK:-16}"

# SIREN weight-network hyperparameters (shared by all six spectral branches).
export SIREN_HIDDEN_DIM="${SIREN_HIDDEN_DIM:-64}"
export SIREN_OMEGA="${SIREN_OMEGA:-30.0}"
export SIREN_N_HIDDEN="${SIREN_N_HIDDEN:-1}"
export SIREN_FEATURE_DIM="${SIREN_FEATURE_DIM:-16}"
export SIREN_FF_SIGMA="${SIREN_FF_SIGMA:-128.0}"
export SIREN_LEARNABLE_FF="${SIREN_LEARNABLE_FF:-1}"
# GRAD_CLIP_NORM defaults to 1.0 for localsirenfno inside fno_zre.py.

# --- Diagnostics (captured in the .out log) ---------------------------------
echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "Submit dir:         ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
echo "Lightcones:         $LIGHTCONE_DIR"
echo "Target cache:       $ZRE_TARGET_CACHE"
echo "Model kind:         $MODEL_KIND"
echo "SIREN:              hidden=${SIREN_HIDDEN_DIM} n_hidden=${SIREN_N_HIDDEN} omega=${SIREN_OMEGA} ff=${SIREN_FEATURE_DIM}@${SIREN_FF_SIGMA}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || echo "(nvidia-smi unavailable)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE'))"
echo "================================================="

# --- Train ------------------------------------------------------------------
python fno_zre.py
