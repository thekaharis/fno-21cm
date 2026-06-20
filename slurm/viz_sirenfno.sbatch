#!/bin/bash
# =============================================================================
# Normal visualization for the 3-D SirenFNO checkpoint.
#
# Submit from the project root:
#
#     sbatch slurm/viz_sirenfno.sbatch
# =============================================================================

#SBATCH --job-name=viz-sirenfno
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz-sirenfno-%j.out
#SBATCH --error=logs/viz-sirenfno-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs figures

module load devel/miniforge
ENV_NAME="fno-env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# Run metadata reconstructs the exact trained architecture. These values also
# provide the correct fallback contract for legacy checkpoints without metadata.
export MODEL_KIND=sirenfno
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_3d_sirenfno}"
export CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"
export VIZ_TAG="${VIZ_TAG:-sirenfno}"

export N_MODES_X="${N_MODES_X:-16}"
export N_MODES_Y="${N_MODES_Y:-16}"
export N_MODES_Z="${N_MODES_Z:-16}"
export SIREN_HIDDEN_DIM="${SIREN_HIDDEN_DIM:-64}"
export SIREN_OMEGA="${SIREN_OMEGA:-30.0}"
export SIREN_N_HIDDEN="${SIREN_N_HIDDEN:-1}"
export SIREN_FEATURE_DIM="${SIREN_FEATURE_DIM:-16}"
export SIREN_FF_SIGMA="${SIREN_FF_SIGMA:-128.0}"
export SIREN_LEARNABLE_FF="${SIREN_LEARNABLE_FF:-true}"
export SIREN_PADDING_X="${SIREN_PADDING_X:-0}"
export SIREN_PADDING_Y="${SIREN_PADDING_Y:-0}"
export SIREN_PADDING_Z="${SIREN_PADDING_Z:-8}"
export SIREN_MLP_DROPOUT="${SIREN_MLP_DROPOUT:-0.0}"

export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
export CUBES_CACHE="${CUBES_CACHE:-/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5}"

echo "================ SirenFNO visualization =========="
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "MODEL_KIND:         ${MODEL_KIND}"
echo "CHECKPOINT_DIR:     ${CHECKPOINT_DIR}"
echo "CHECKPOINT_KIND:    ${CHECKPOINT_KIND}"
echo "VIZ_TAG:            ${VIZ_TAG}"
echo "CUBES_CACHE:        ${CUBES_CACHE}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "==================================================="

python -m viz.visualize_3d
