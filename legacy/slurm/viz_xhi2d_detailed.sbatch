#!/bin/bash
# Full detailed visualisation of one 2-D x_HI run: representative held-out
# slices, then the inference test against the best 3-D models with the physical
# diagnostics and metrics.json.
#
#   RUN_DIR   checkpoint directory of the 2-D run
#   TAG       short name used for the figure directories
#
#   sbatch --export=ALL,RUN_DIR=checkpoints/xhi2d_whno_glob_expwall16,TAG=expwall16 \
#          slurm/viz_xhi2d_detailed.sbatch
#
# CONE_SPLIT defaults to test. Note the cap: selection needs globally distinct
# ionized cones and only 12 are eligible in the test split, so
# SLICES_PER_STAGE cannot exceed 9 there (45 slices over 5 stages). Asking for
# more fails outright rather than silently truncating.

#SBATCH --job-name=xhi2d-detailed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/xhi2d-detailed-%j.out
#SBATCH --error=logs/xhi2d-detailed-%j.err

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

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/../data/compressed}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the 2-D run checkpoint directory}"
TAG="${TAG:-$(basename "$RUN_DIR")}"
CUBE_CACHE="${CUBE_CACHE:-$DATA_DIR/cubes_3d.h5}"
UFNO_CHECKPOINT="${UFNO_CHECKPOINT:-checkpoint-archive/checkpoints_3d_ufno_16modes_weights_viz/best_model_state_dict.pt}"
LOCALFNO_CHECKPOINT="${LOCALFNO_CHECKPOINT:-checkpoints/checkpoints_3d_localfno_1gpu/best_model_state_dict.pt}"
CONE_SPLIT="${CONE_SPLIT:-test}"
SLICES_PER_STAGE="${SLICES_PER_STAGE:-9}"

echo "=========== detailed viz: $TAG ==========="
echo "run        : $RUN_DIR"
echo "cone split : $CONE_SPLIT ($SLICES_PER_STAGE slices/stage)"
echo "cubes      : $CUBE_CACHE"
echo "=========================================="

echo "[1/2] representative held-out slices"
python -m viz.visualize_xhi2d_representative \
    --run "$RUN_DIR" \
    --output "figures/xhi2d_${TAG}_representative"

echo "[2/2] inference test vs best 3-D + physical diagnostics"
python -m legacy.xhi2d.compare_xhi2d_to_3d \
    --wno-run "$RUN_DIR" \
    --ufno-checkpoint "$UFNO_CHECKPOINT" \
    --localfno-checkpoint "$LOCALFNO_CHECKPOINT" \
    --cube-cache "$CUBE_CACHE" \
    --output "figures/xhi2d_${TAG}_vs_best_3d" \
    --patch-chunk-size 32 \
    --slices-per-stage "$SLICES_PER_STAGE" \
    --cone-split "$CONE_SPLIT" \
    --page-size 10 \
    --batch-size-2d 16

echo "done -> figures/xhi2d_${TAG}_representative, figures/xhi2d_${TAG}_vs_best_3d"
