#!/bin/bash
# Visualization for any modular local/global operator pairing (MODEL_KIND=localop).
#
# The architecture itself is *not* configured here: viz.visualize_3d rebuilds
# the model from the run_metadata.json written next to the checkpoint, so both
# operator slots and their hyperparameters come from the training run. What
# LOCAL_OPERATOR/GLOBAL_OPERATOR do here is pick the default checkpoint
# directory and the figures/ tag, derived from the same OPERATOR_TAGS mapping
# the trainer names its directory with -- so the two cannot drift apart, and a
# typo'd operator name fails before the job reaches a GPU.
#
#   sbatch --export=ALL,LOCAL_OPERATOR=hadamard,GLOBAL_OPERATOR=cnn \
#       slurm/viz_localop.sbatch
#
# Aliases are accepted (fno, wno, whno/walsh, siren, unet). Runs that overrode
# CHECKPOINT_DIR at training time need it overridden here too, e.g.
#
#   sbatch --export=ALL,GLOBAL_OPERATOR=hadamard,\
# CHECKPOINT_DIR=./checkpoints/checkpoints_3d_localop_whno_expwall16,\
# VIZ_TAG=localop-whno-expwall16 slurm/viz_localop.sbatch

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
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

export MODEL_KIND=localop
export LOCAL_OPERATOR="${LOCAL_OPERATOR:-fourier}"
export GLOBAL_OPERATOR="${GLOBAL_OPERATOR:-hadamard}"

# Resolve the aliases and derive "<checkpoint dir> <figures tag>" from the
# resolved pair.  Plain assignment (not `export FOO=$(...)`) so that a bad
# operator name propagates its non-zero exit through set -e.
DERIVED="$(python - <<'PY'
from modeling import OPERATOR_TAGS, ModelConfig

config = ModelConfig.from_env()
local = OPERATOR_TAGS[config.local_operator]
global_ = OPERATOR_TAGS[config.global_operator]
print(config.default_checkpoint_dir, f"local-{local}-{global_}")
PY
)"
read -r DERIVED_CHECKPOINT_DIR DERIVED_TAG <<< "$DERIVED"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./$DERIVED_CHECKPOINT_DIR}"
export CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"
export VIZ_TAG="${VIZ_TAG:-$DERIVED_TAG}"
# Inference-only knob: chunk size changes execution memory, not parameters,
# so it overrides whatever the training run recorded (64 fits an A30).
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-64}"
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
export CUBES_CACHE="${CUBES_CACHE:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"

echo "======== localop viz: $LOCAL_OPERATOR local / $GLOBAL_OPERATOR global ========"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR ($CHECKPOINT_KIND)"
echo "VIZ_TAG:        $VIZ_TAG"
echo "CUBES_CACHE:    $CUBES_CACHE"
echo "=============================================================================="

python -m viz.visualize_3d
