#!/bin/bash
# Compare the completed 2-D LocalWNO with the best uniform-grid 3-D models.

#SBATCH --job-name=xhi-wno-vs-3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/xhi-wno-vs-3d-%j.out
#SBATCH --error=logs/xhi-wno-vs-3d-%j.err

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

WNO_RUN="${WNO_RUN:-checkpoints/xhi2d_localwno_lr2e4_e30}"
UFNO_CHECKPOINT="${UFNO_CHECKPOINT:-checkpoint-archive/checkpoints_3d_ufno_16modes_weights_viz/best_model_state_dict.pt}"
LOCALFNO_CHECKPOINT="${LOCALFNO_CHECKPOINT:-checkpoints/checkpoints_3d_localfno_1gpu/best_model_state_dict.pt}"
CUBE_CACHE="${CUBE_CACHE:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"
VIZ_OUTPUT="${VIZ_OUTPUT:-figures/xhi2d_localwno_vs_best_3d_100slices}"
SLICES_PER_STAGE="${SLICES_PER_STAGE:-20}"
# train (in-sample for all three) | test | heldout (val+test)
CONE_SPLIT="${CONE_SPLIT:-train}"

python -m legacy.xhi2d.compare_xhi2d_to_3d \
    --wno-run "$WNO_RUN" \
    --ufno-checkpoint "$UFNO_CHECKPOINT" \
    --localfno-checkpoint "$LOCALFNO_CHECKPOINT" \
    --cube-cache "$CUBE_CACHE" \
    --output "$VIZ_OUTPUT" \
    --patch-chunk-size 32 \
    --slices-per-stage "$SLICES_PER_STAGE" \
    --cone-split "$CONE_SPLIT" \
    --page-size 10 \
    --batch-size-2d 16
