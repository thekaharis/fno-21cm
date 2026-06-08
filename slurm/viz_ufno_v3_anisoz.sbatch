#!/bin/bash
# =============================================================================
# SLURM batch script -- render figures from the v3 U-FNO **anisotropic-Z
# U-Net** (option D) checkpoint.
#
# Submit from the project root:
#
#     sbatch slurm/viz_ufno_v3_anisoz.sbatch
#
# Differences vs slurm/viz_ufno.sbatch (the v1 U-FNO viz):
#   * Architecture env vars must match the training-time config exactly so
#     load_model() reconstructs the right model topology before loading the
#     state dict.  v3-anisoz training was:
#         UFNO_UNET_VARIANT=anisotropic_z
#         UFNO_NORM=groupnorm                (v2 baseline, carried forward)
#         N_MODES_Z=32                       (v2 baseline, carried forward)
#     If any of these are missing the load_model() shape-match check fails
#     loudly (0 matched keys), which is the right behavior -- it's safer
#     than silently loading a partial checkpoint.
#
#   * CHECKPOINT points at ./checkpoints_3d_ufno_v3_anisoz/, not the v1
#     ./checkpoints_3d_ufno/.
#
#   * VIZ_TAG=ufno-v3-anisoz so the per-run figure folder name encodes
#     the variant (e.g. figures/ufno-v3-anisoz_20260607-..._jobNNN/),
#     making it obvious in the figures/ listing which architectural
#     variant produced which images.  Without this, every U-FNO render
#     would just be "ufno_<timestamp>" and you'd have to open run_info.txt
#     to disambiguate -- painful when comparing variants side by side.
#
# Pairs with slurm/viz_ufno_v3_anisoz_detailed.sbatch (16-cone, active-z
# slice picker).
# =============================================================================

#SBATCH --job-name=viz-ufno-v3-anisoz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz-ufno-v3-anisoz-%j.out
#SBATCH --error=logs/viz-ufno-v3-anisoz-%j.err

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

# v3 (Tier 2) Option D: anisotropic Z U-Net.  The matching training sbatch
# is slurm/train_ufno_v3_anisoz_h200_4gpu.sbatch -- any change there must
# be mirrored here or load_model() will fail the shape-match check.
export UFNO_UNET_VARIANT=anisotropic_z

# v2 baseline overrides carried forward into all v3 runs.
export N_MODES_Z=32
export UFNO_NORM=groupnorm

# --- Checkpoint + figure-tag wiring -----------------------------------------
export CHECKPOINT_DIR=./checkpoints_3d_ufno_v3_anisoz
export CHECKPOINT="${CHECKPOINT_DIR}/best_model_state_dict.pt"
export VIZ_TAG=ufno-v3-anisoz

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
