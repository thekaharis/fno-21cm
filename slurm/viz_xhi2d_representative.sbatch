#!/bin/bash
# Render representative held-out x_HI slices for one completed 2-D run.
# Override RUN_DIR and VIZ_OUTPUT when submitting another checkpoint.

#SBATCH --job-name=xhi2d-repr-viz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/xhi2d-repr-viz-%j.out
#SBATCH --error=logs/xhi2d-repr-viz-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs figures

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-checkpoints/xhi2d_localwno_lr2e4_e30}"
VIZ_OUTPUT="${VIZ_OUTPUT:-figures/xhi2d_localwno_lr2e4_e30_representative}"

python -m viz.visualize_xhi2d_representative \
    --run "$RUN_DIR" \
    --output "$VIZ_OUTPUT"
