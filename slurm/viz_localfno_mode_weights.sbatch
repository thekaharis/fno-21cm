#!/bin/bash
# Render per-branch mode-weight diagnostics from a trained 3-D LocalFNO
# checkpoint (weights only -- no dataset needed).
#
# Submit from the project root, exporting the SAME LOCALFNO_* switches the
# training run used (defaults match slurm/train_localfno_h200_4gpu.sbatch):
#
#     sbatch slurm/viz_localfno_mode_weights.sbatch
#     sbatch --export=ALL,CHECKPOINT_KIND=final slurm/viz_localfno_mode_weights.sbatch
#
# Override CHECKPOINT_DIR to inspect another LocalFNO run.

#SBATCH --job-name=viz-localfno-mode-weights
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/viz-localfno-mode-weights-%j.out
#SBATCH --error=logs/viz-localfno-mode-weights-%j.err

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

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/3d_xhi/localfno/checkpoints_3d_localfno}"
export MODEL_KIND=localfno
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$MPLCONFIGDIR"

python -m viz.localfno_mode_weights --task 3d
