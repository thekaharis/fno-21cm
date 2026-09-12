#!/bin/bash
# Compare the completed 2-D x_HI LocalWNO sweep with matched baselines.
# Normally submitted by submit_xhi2d_wno_sweep.sh with afterany dependencies.

#SBATCH --job-name=xhi2d-wno-viz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/xhi2d-wno-viz-%j.out
#SBATCH --error=logs/xhi2d-wno-viz-%j.err

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

python -m viz.compare_xhi2d_models \
    --output figures/2d_xhi/eval/xhi2d_wno_sweep_comparison \
    --run default-wno=checkpoints/2d_xhi/localwno/checkpoints_2d_xhi_localwno \
    --run wno-lr2e4=checkpoints/2d_xhi/localwno/xhi2d_localwno_lr2e4_e30 \
    --run wno-lr3e4=checkpoints/2d_xhi/localwno/xhi2d_localwno_lr3e4_e30 \
    --run wno-r32=checkpoints/2d_xhi/localwno/xhi2d_localwno_lr3e4_r32 \
    --run wno-w48r24=checkpoints/2d_xhi/localwno/xhi2d_localwno_lr3e4_w48r24 \
    --run wno-h1e3=checkpoints/2d_xhi/localwno/xhi2d_localwno_lr3e4_h1e3 \
    --run localfno=checkpoints/2d_xhi/localfno/xhi2d_localfno_lr3e4_e30 \
    --run ufno=checkpoints/2d_xhi/ufno/xhi2d_ufno_lr3e4_e30
