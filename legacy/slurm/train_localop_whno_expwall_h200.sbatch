#!/bin/bash
# 3-D lightcone training: localop with a Fourier local branch and a
# Walsh-Hadamard global branch (the best 2-D configuration), trained under the
# exponential wall-placement loss instead of L2.
#
# Why this pairing. On 2-D slices whno_glob was the strongest architecture, and
# ExponentialWallDistance at scale=16 was the only loss that moved sharpness:
# transition width 1.33 vs truth 1.47 (L2 baseline 3.21) and wall-placement
# error 0.00282 vs 0.00604, for ~20% more RMSE. This is the 3-D version of that
# experiment.
#
# The loss is L2-free by design. It is minimised exactly at the target and its
# gradient is zero there, so it anchors level and position on its own -- unlike
# the SWD/highK edge terms, which need an L2 anchor and diverged without one.
#
# Override anything via --export, e.g.
#   sbatch --export=ALL,EXPWALL_SCALE=8,N_EPOCHS=30 slurm/train_localop_whno_expwall_h200.sbatch

#SBATCH --job-name=localop-whno-expwall-3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=logs/localop-whno-expwall-3d-%j.out
#SBATCH --error=logs/localop-whno-expwall-3d-%j.err

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

# ---- architecture: the 2-D winner, in 3-D -------------------------------
export MODEL_KIND="${MODEL_KIND:-localop}"
export LOCAL_OPERATOR="${LOCAL_OPERATOR:-fourier}"
export GLOBAL_OPERATOR="${GLOBAL_OPERATOR:-hadamard}"
export LOCALFNO_WINDOW_X="${LOCALFNO_WINDOW_X:-16}"
export LOCALFNO_WINDOW_Y="${LOCALFNO_WINDOW_Y:-16}"
export LOCALFNO_WINDOW_Z="${LOCALFNO_WINDOW_Z:-32}"
export LOCALFNO_MODES_X="${LOCALFNO_MODES_X:-6}"
export LOCALFNO_MODES_Y="${LOCALFNO_MODES_Y:-6}"
export LOCALFNO_MODES_Z="${LOCALFNO_MODES_Z:-12}"
export LOCALFNO_BASE_WIDTH="${LOCALFNO_BASE_WIDTH:-16}"
export LOCALFNO_SPECTRAL_RANK="${LOCALFNO_SPECTRAL_RANK:-16}"
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-128}"
export N_MODES_X="${N_MODES_X:-16}"
export N_MODES_Y="${N_MODES_Y:-16}"
export N_MODES_Z="${N_MODES_Z:-16}"
export HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-32}"
export N_LAYERS="${N_LAYERS:-4}"

# ---- training -----------------------------------------------------------
export N_EPOCHS="${N_EPOCHS:-50}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
export LOCALFNO_LEARNING_RATE="${LOCALFNO_LEARNING_RATE:-1e-4}"
export LOCALFNO_GRAD_CLIP_NORM="${LOCALFNO_GRAD_CLIP_NORM:-1.0}"
export LOCALFNO_H1_WARMUP_EPOCHS="${LOCALFNO_H1_WARMUP_EPOCHS:-0}"
export RUN_SEED="${RUN_SEED:-0}"
export SPLIT_SEED="${SPLIT_SEED:-42}"

# ---- loss: wall placement only, no L2 -----------------------------------
export LOSS_L2_WEIGHT="${LOSS_L2_WEIGHT:-0.0}"
export LOSS_H1_WEIGHT="${LOSS_H1_WEIGHT:-0.0}"
export LOSS_BCE_WEIGHT="${LOSS_BCE_WEIGHT:-0.0}"
export LOSS_IONIZED_WALL_WEIGHT="${LOSS_IONIZED_WALL_WEIGHT:-0.0}"
export LOSS_EXPWALL_WEIGHT="${LOSS_EXPWALL_WEIGHT:-1.0}"
export EXPWALL_SCALE="${EXPWALL_SCALE:-16.0}"
export WALL_CAP="${WALL_CAP:-32}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/3d_xhi/fno_whno/checkpoints_3d_localop_whno_expwall16}"

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

echo "======== localop[fourier/hadamard] + expwall, 3-D, 1x H200 ========"
echo "MODEL_KIND:   $MODEL_KIND ($LOCAL_OPERATOR local / $GLOBAL_OPERATOR global)"
echo "LOSS:         ${LOSS_EXPWALL_WEIGHT}*expwall[scale=${EXPWALL_SCALE},cap=${WALL_CAP}]"
echo "              L2=$LOSS_L2_WEIGHT H1=$LOSS_H1_WEIGHT BCE=$LOSS_BCE_WEIGHT"
echo "CHECKPOINT:   $CHECKPOINT_DIR"
echo "CUBES_CACHE:  $CUBES_CACHE"
echo "=================================================================="

python fno_21cm_3d.py
