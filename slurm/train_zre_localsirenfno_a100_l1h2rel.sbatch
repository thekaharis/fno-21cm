#!/bin/bash
# =============================================================================
# SLURM batch script — train the 2-D LocalSirenFNO mapping density cones to
# z_re(x, y) maps, relative L1 + H2 loss ablation on a single A100.
#
# Submit from the project directory (the one containing fno_zre.py):
#
#     sbatch slurm/train_zre_localsirenfno_a100_l1h2rel.sbatch
#
# Companion ablation: slurm/train_zre_localsirenfno_a100_l2only.sbatch
# (absolute L2 only). Experiment switches are plain env vars (see fno_zre.py
# header).
#
# Note: relative norms divide by the target norm, which is small for
# late-reionizing cones (the normalized target is zero over clamped
# regions); the neuralop eps guards the division. This ablation exists to
# measure exactly that effect.
# =============================================================================

# NOTE: do NOT add trailing "# ..." comments on #SBATCH lines — some SLURM
# versions parse them as part of the option value and reject the job.
#
#SBATCH --job-name=localsirenfno-zre-l1h2rel
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/localsirenfno-zre-l1h2rel-%j.out
#SBATCH --error=logs/localsirenfno-zre-l1h2rel-%j.err

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

# Datasets live outside the repo; see dataset/paths.py
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/../data/compressed}"
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
export ZRE_TARGET_CACHE="${ZRE_TARGET_CACHE:-$DATA_DIR/zre_targets.h5}"

# Model and loss ablation. CHECKPOINT_DIR is explicit so the run never
# collides with the default per-kind directory or the L2-only companion run.
export MODEL_KIND="${MODEL_KIND:-localsirenfno}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/checkpoints_zre_localsirenfno_l1h2rel}"
export TARGET_KIND="${TARGET_KIND:-gompertz}"
export INPUT_FEATURES="${INPUT_FEATURES:-density_params}"
export N_Z_IN="${N_Z_IN:-64}"
export N_EPOCHS="${N_EPOCHS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export LOSS_L2_WEIGHT="${LOSS_L2_WEIGHT:-0.0}"
export LOSS_H1_WEIGHT="${LOSS_H1_WEIGHT:-0.0}"
export LOSS_L1_WEIGHT="${LOSS_L1_WEIGHT:-0.5}"
export LOSS_H2_WEIGHT="${LOSS_H2_WEIGHT:-0.5}"
export LOSS_RELATIVE="${LOSS_RELATIVE:-1}"

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
echo "Checkpoint dir:     $CHECKPOINT_DIR"
echo "Target kind:        $TARGET_KIND"
echo "Input features:     $INPUT_FEATURES"
echo "Loss:               L2=${LOSS_L2_WEIGHT} H1=${LOSS_H1_WEIGHT} L1=${LOSS_L1_WEIGHT} H2=${LOSS_H2_WEIGHT} relative=${LOSS_RELATIVE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || echo "(nvidia-smi unavailable)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE'))"
echo "================================================="

# --- Train ------------------------------------------------------------------
# Plain `python` (no `srun`): single-process, single-GPU job.
python fno_zre.py
