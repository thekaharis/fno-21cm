#!/bin/bash
# Derive (theta, tau) from truth edge geometry, and test whether a pointwise
# map can undo pure band-limiting. Analysis only -- no training.

#SBATCH --job-name=ctr-from-truth
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/ctr-from-truth-%j.out
#SBATCH --error=logs/ctr-from-truth-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

python legacy/probes/probe_contrast_from_truth.py
