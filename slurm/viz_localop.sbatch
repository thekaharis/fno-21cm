#!/bin/bash
# Visualization for any modular local/global operator pairing (MODEL_KIND=localop).
#
# Pass the checkpoint directory, nothing else. viz.visualize_3d rebuilds the
# model from the run_metadata.json written next to the checkpoint, so both
# operator slots and all their hyperparameters come from the training run --
# there is nothing about the architecture to configure here.
#
#   sbatch --export=ALL,CHECKPOINT_DIR=./checkpoints/checkpoints_3d_local_whno_cnn \
#       slurm/viz_localop.sbatch
#
# VIZ_TAG defaults to the directory's basename minus the "checkpoints_3d_"
# prefix (-> figures/local_whno_cnn_<timestamp>_job.../). Set it explicitly to
# name the folder something else; without it every localop pairing would land
# in an identically-named folder, since the tag otherwise falls back to the
# model kind, which is just "localop" for all of them.

#SBATCH --job-name=viz-localop
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/viz-localop-%j.out
#SBATCH --error=logs/viz-localop-%j.err

set -eo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs figures

if [ -z "${CHECKPOINT_DIR:-}" ]; then
    echo "CHECKPOINT_DIR is required, e.g. --export=ALL,CHECKPOINT_DIR=./checkpoints/checkpoints_3d_local_whno_cnn" >&2
    exit 1
fi
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "CHECKPOINT_DIR does not exist: $CHECKPOINT_DIR" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT_DIR/run_metadata.json" ]; then
    echo "warning: no run_metadata.json in $CHECKPOINT_DIR -- the architecture" >&2
    echo "         will fall back to the MODEL_KIND/*_OPERATOR environment" >&2
    echo "         variables, which must then match the training run." >&2
fi

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

export CHECKPOINT_DIR
export CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT_DIR%/}")"
export VIZ_TAG="${VIZ_TAG:-${CHECKPOINT_NAME#checkpoints_3d_}}"
# Inference-only knob: chunk size changes execution memory, not parameters,
# so it overrides whatever the training run recorded (64 fits an A30).
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-64}"
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
export CUBES_CACHE="${CUBES_CACHE:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"

echo "======== localop viz ========"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR ($CHECKPOINT_KIND)"
echo "VIZ_TAG:        $VIZ_TAG"
echo "CUBES_CACHE:    $CUBES_CACHE"
echo "============================="

python -m viz.visualize_3d
