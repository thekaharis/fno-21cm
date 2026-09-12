#!/bin/bash
# Render Z/LOS-only epoch-by-epoch Fourier-weight diagnostics.
#
# Submit from the project root:
#
#   sbatch --export=ALL,CHECKPOINT_DIR=./checkpoints/3d_xhi/ufno/checkpoints_3d_ufno_v2 \
#     slurm/viz_spectral_weights_z.sbatch

#SBATCH --job-name=viz-spectral-z
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/viz-spectral-z-%j.out
#SBATCH --error=logs/viz-spectral-z-%j.err

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

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/3d_xhi/fno/checkpoints_3d}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$MPLCONFIGDIR"

python -m viz.visualize_spectral_weights_z
