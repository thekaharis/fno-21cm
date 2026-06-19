#!/bin/bash
# Render epoch-by-epoch Fourier-weight diagnostics for the basic U-FNO run.
#
# Submit from the project root:
#
#     sbatch slurm/viz_spectral_weights_ufno.sbatch
#
# CHECKPOINT_DIR remains overridable for another U-FNO variant.

#SBATCH --job-name=viz-spectral-ufno
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/viz-spectral-ufno-%j.out
#SBATCH --error=logs/viz-spectral-ufno-%j.err

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

export MODEL_KIND=ufno
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_3d_ufno}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$MPLCONFIGDIR"

OUTPUT_DIR="figures/spectral-weights_ufno_$(date +%Y%m%d-%H%M%S)"

echo "MODEL_KIND:         ${MODEL_KIND}"
echo "CHECKPOINT_DIR:     ${CHECKPOINT_DIR}"
echo "SPECTRAL_HISTORY:   ${CHECKPOINT_DIR}/spectral_weight_history.npz"
echo "OUTPUT_DIR:         ${OUTPUT_DIR}"

python visualize_spectral_weights.py --output-dir "$OUTPUT_DIR"
