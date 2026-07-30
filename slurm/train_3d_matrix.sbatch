#!/bin/bash
# One sbatch for the 3-D architecture x loss matrix, 50 epochs each.
#
#   ARCH  ufno | fno_whno | wno_whno | whno_whno
#   LOSS  plain | expwall | hybrid | expwall_theta
#
#   sbatch --export=ALL,ARCH=fno_whno,LOSS=hybrid slurm/train_3d_matrix.sbatch
#
# A 50-epoch 3-D run took 2.7-3.2 days on one H200 in past sweeps (~5100 s per
# epoch), so the whole 4x4 matrix is roughly 48 GPU-days.
#
# The hybrid weight is measured, not guessed. On real cubes L2/expwall is 3.0 for
# a constant-0.5 prediction and 85.8 once the prediction is merely blurred -- the
# ratio inverts over training. A fixed weight therefore swamps L2 early or
# vanishes late, so the hybrid ramps expwall in over EXPWALL_WARMUP_EPOCHS and
# then holds it at ~25% of the total.

#SBATCH --job-name=3dmatrix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=logs/3dmatrix-%j.out
#SBATCH --error=logs/3dmatrix-%j.err

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

ARCH="${ARCH:?set ARCH to ufno|fno_whno|wno_whno|whno_whno}"
LOSS="${LOSS:?set LOSS to plain|expwall|hybrid|expwall_theta}"

# ---- architecture ------------------------------------------------------
case "$ARCH" in
  ufno)       export MODEL_KIND=ufno
              # --export=ALL can carry stale operator vars in from the calling
              # shell; an empty LOCAL_OPERATOR is a hard error in from_env().
              unset LOCAL_OPERATOR GLOBAL_OPERATOR ;;
  fno_whno)   export MODEL_KIND=localop LOCAL_OPERATOR=fourier  GLOBAL_OPERATOR=hadamard ;;
  wno_whno)   export MODEL_KIND=localop LOCAL_OPERATOR=wavelet  GLOBAL_OPERATOR=hadamard ;;
  whno_whno)  export MODEL_KIND=localop LOCAL_OPERATOR=hadamard GLOBAL_OPERATOR=hadamard ;;
  *) echo "unknown ARCH=$ARCH" >&2; exit 2 ;;
esac

# shared 3-D geometry, matching the existing 3-D sweeps so results compare
export LOCALFNO_WINDOW_X=16 LOCALFNO_WINDOW_Y=16 LOCALFNO_WINDOW_Z=32
export LOCALFNO_MODES_X=6  LOCALFNO_MODES_Y=6  LOCALFNO_MODES_Z=12
export LOCALFNO_BASE_WIDTH="${LOCALFNO_BASE_WIDTH:-16}"
export LOCALFNO_SPECTRAL_RANK="${LOCALFNO_SPECTRAL_RANK:-16}"
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-128}"
export N_MODES_X=16 N_MODES_Y=16 N_MODES_Z=16
export HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-32}"
export N_LAYERS="${N_LAYERS:-4}"
export UFNO_WIDTH="${UFNO_WIDTH:-32}"
export LOCALWNO_LEVELS="${LOCALWNO_LEVELS:-2}"

# ---- training ----------------------------------------------------------
export N_EPOCHS="${N_EPOCHS:-50}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
export LOCALFNO_LEARNING_RATE="${LOCALFNO_LEARNING_RATE:-1e-4}"
export LOCALFNO_GRAD_CLIP_NORM="${LOCALFNO_GRAD_CLIP_NORM:-1.0}"
export LOCALFNO_H1_WARMUP_EPOCHS=0
export RUN_SEED="${RUN_SEED:-0}"
export SPLIT_SEED="${SPLIT_SEED:-42}"

# ---- loss --------------------------------------------------------------
# Absolute L2 throughout, so every cell of the matrix is on one scale. The older
# 3-D runs used a relative L2, which is why their numbers are not comparable.
export LOSS_L2_MODE=absolute LOSS_H1_MODE=absolute
export LOSS_H1_WEIGHT=0.0 LOSS_BCE_WEIGHT=0.0 LOSS_IONIZED_WALL_WEIGHT=0.0
export EXPWALL_SCALE="${EXPWALL_SCALE:-16.0}" WALL_CAP="${WALL_CAP:-32}"
case "$LOSS" in
  plain)
    export LOSS_L2_WEIGHT=1.0 LOSS_EXPWALL_WEIGHT=0.0
    export EXPWALL_WARMUP_EPOCHS=0 ;;
  expwall)
    export LOSS_L2_WEIGHT=0.0 LOSS_EXPWALL_WEIGHT=1.0
    export EXPWALL_WARMUP_EPOCHS=0 ;;
  hybrid)
    export LOSS_L2_WEIGHT=1.0 LOSS_EXPWALL_WEIGHT="${EXPWALL_HYBRID_W:-30.0}"
    export EXPWALL_WARMUP_EPOCHS="${EXPWALL_WARMUP_EPOCHS:-5}" ;;
  expwall_theta)
    export LOSS_L2_WEIGHT=0.0 LOSS_EXPWALL_WEIGHT=1.0
    export EXPWALL_WARMUP_EPOCHS=0
    export CONTRAST_MODE=xhi CONTRAST_REFIT=1 CONTRAST_SCHEDULE_KIND=stepped
    export CONTRAST_BINS="${CONTRAST_BINS:-14}" ;;
  *) echo "unknown LOSS=$LOSS" >&2; exit 2 ;;
esac

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/checkpoints_3d_${ARCH}_${LOSS}}"

export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
PROJECT_CUBES="${PROJECT_CUBES:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"
LOCAL_SCRATCH="${LOCAL_SCRATCH:-yes}"
if [ "$LOCAL_SCRATCH" = "yes" ]; then
    LOCAL_CUBES="$TMPDIR/cubes_3d.h5"
    echo "Staging cube cache to local scratch: $LOCAL_CUBES"
    t0=$(date +%s)
    cp "$PROJECT_CUBES" "$LOCAL_CUBES"
    echo "  staged ($(du -h "$LOCAL_CUBES" | cut -f1)) in $(( $(date +%s) - t0 ))s"
    export CUBES_CACHE="$LOCAL_CUBES"
else
    export CUBES_CACHE="${CUBES_CACHE:-$PROJECT_CUBES}"
fi

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============ 3-D matrix: ARCH=$ARCH  LOSS=$LOSS ============"
echo "MODEL_KIND:   $MODEL_KIND ${LOCAL_OPERATOR:+($LOCAL_OPERATOR local / $GLOBAL_OPERATOR global)}"
echo "LOSS:         L2=$LOSS_L2_WEIGHT expwall=$LOSS_EXPWALL_WEIGHT"
echo "              scale=$EXPWALL_SCALE cap=$WALL_CAP warmup=$EXPWALL_WARMUP_EPOCHS"
echo "CONTRAST:     ${CONTRAST_MODE:-off} ${CONTRAST_SCHEDULE_KIND:+/$CONTRAST_SCHEDULE_KIND}"
echo "EPOCHS:       $N_EPOCHS"
echo "CHECKPOINT:   $CHECKPOINT_DIR"
echo "============================================================"

python fno_21cm_3d.py
