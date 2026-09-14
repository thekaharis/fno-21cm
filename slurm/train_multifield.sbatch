#!/bin/bash
# Run one indexed mapping/seed from a precomputed plan. Supply --array=0-N
# at submission, where N is the final index printed by the planner.
# Example: sbatch --array=0-35 --export=ALL,MULTIFIELD_PLAN=/path/pairwise.json \
#            slurm/train_multifield.sbatch
#SBATCH --job-name=multifield
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=96:00:00
#SBATCH --output=logs/multifield-%A_%a.out
#SBATCH --error=logs/multifield-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1
python -m util.multifield_experiments run \
    --plan "${MULTIFIELD_PLAN:?provide the prepared experiment plan}" \
    --index "${SLURM_ARRAY_TASK_ID:?submit this script as an array job}" \
    --workers "${NUM_WORKERS:-4}" --device cuda --skip-complete
