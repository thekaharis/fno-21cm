#!/bin/bash
# Survey ionization-front sharpness across all 6600 raw 21cmFAST lightcones.
# CPU only -- reads chunk-aligned z-slabs and reduces them; no model involved.
#
# One task per shard; merge with viz/report_truth_survey.py afterwards.
#   sbatch --array=0-32 slurm/survey_truth_sharpness.sbatch

#SBATCH --job-name=truth-survey
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/truth-survey-%A_%a.out
#SBATCH --error=logs/truth-survey-%A_%a.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs figures/shared/diagnostics/truth_survey

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

NUM_SHARDS="${NUM_SHARDS:-33}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

python -m legacy.xhi2d.survey_truth_sharpness \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --out figures/shared/diagnostics/truth_survey
