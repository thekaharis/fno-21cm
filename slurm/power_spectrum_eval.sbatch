#!/bin/bash
# =============================================================================
# Scale- and redshift-resolved power-spectrum evaluation: Local-FNO vs U-FNO.
#
# For each model the script predicts the same test cones and compares the
# transverse power ratio P_pred/P_true and cross-correlation r(k) per k bin,
# per redshift slab, and per reionization stage (transverse-mean x_HI).
#
# Default submission:
#
#   sbatch slurm/power_spectrum_eval.sbatch
#
# Compare against another U-FNO checkpoint:
#
#   sbatch --export=ALL,\
# UFNO_CHECKPOINT=checkpoints/checkpoints_3d_ufno_v3_anisoz/best_model_state_dict.pt \
#     slurm/power_spectrum_eval.sbatch
#
# Short feasibility run:
#
#   sbatch --export=ALL,N_CONES=20 slurm/power_spectrum_eval.sbatch
# =============================================================================

#SBATCH --job-name=ps-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ps-eval-%j.out
#SBATCH --error=logs/ps-eval-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs ps_out

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- checkpoints
# Both checkpoint directories must contain run_metadata.json. This lets the
# diagnostic safely reconstruct different model architectures in one process.
UFNO_CHECKPOINT="${UFNO_CHECKPOINT:-./checkpoints/checkpoints_3d_ufno/best_model_state_dict.pt}"
LOCALFNO_CHECKPOINT="${LOCALFNO_CHECKPOINT:-./checkpoints/checkpoints_3d_localfno/best_model_state_dict.pt}"

for checkpoint in "$UFNO_CHECKPOINT" "$LOCALFNO_CHECKPOINT"; do
    if [ ! -f "$checkpoint" ]; then
        echo "Checkpoint not found: $checkpoint" >&2
        exit 1
    fi
    metadata="$(dirname "$checkpoint")/run_metadata.json"
    if [ ! -f "$metadata" ]; then
        echo "Run metadata not found: $metadata" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------- experiment
N_CONES="${N_CONES:-200}"
SPLIT="${SPLIT:-test}"
N_BINS="${N_BINS:-15}"
CHUNK_Z="${CHUNK_Z:-7 9 11}"
CHUNK_CELLS="${CHUNK_CELLS:-32}"
BOX_MPC="${BOX_MPC:-200.0}"
OUT_DIR="${OUT_DIR:-ps_out/localfno-vs-ufno-${SLURM_JOB_ID:-local}}"

# Reducing this value changes execution memory only, not checkpoint weights.
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-32}"

# SAVE_CUBES=yes stores each prediction as compressed NPZ for later offline
# re-analysis (re-usable by boundary_band_diagnostic via a manifest). For
# 200 cones and two models this can consume substantial space.
SAVE_CUBES="${SAVE_CUBES:-no}"
SAVED_CUBES_DIR="${SAVED_CUBES_DIR:-${OUT_DIR}/cubes}"

# ---------------------------------------------------------------- data cache
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
PROJECT_CUBES="${PROJECT_CUBES:-/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5}"
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

if [ ! -f "$CUBES_CACHE" ]; then
    echo "Cube cache not found: $CUBES_CACHE" >&2
    exit 1
fi

# ---------------------------------------------------------------- diagnostics
echo "================ power-spectrum evaluation ================"
echo "Host:                 $(hostname)"
echo "Job ID:               ${SLURM_JOB_ID:-N/A}"
echo "U-FNO checkpoint:     $UFNO_CHECKPOINT"
echo "Local-FNO checkpoint: $LOCALFNO_CHECKPOINT"
echo "Split / cones:        $SPLIT / $N_CONES"
echo "k bins:               $N_BINS"
echo "Cyl. chunks (z):      $CHUNK_Z  ($CHUNK_CELLS cells each)"
echo "Local-FNO chunk:      $LOCALFNO_PATCH_CHUNK_SIZE"
echo "Cube cache:           $CUBES_CACHE"
echo "Save predictions:     $SAVE_CUBES"
echo "Output:               $OUT_DIR"
nvidia-smi -L || true
echo "==========================================================="

args=(
    -m viz.power_spectrum_evaluation
    --checkpoints
    "ufno=$UFNO_CHECKPOINT"
    "localfno=$LOCALFNO_CHECKPOINT"
    --split "$SPLIT"
    --n-cones "$N_CONES"
    --n-bins "$N_BINS"
    --chunk-cells "$CHUNK_CELLS"
    --box-mpc "$BOX_MPC"
    --out "$OUT_DIR"
)
# CHUNK_Z is a space-separated list, so it must word-split here.
# shellcheck disable=SC2206
args+=(--chunk-z $CHUNK_Z)

if [ "$SAVE_CUBES" = "yes" ]; then
    args+=(--save-cubes "$SAVED_CUBES_DIR")
fi

python "${args[@]}"

echo "Power-spectrum evaluation complete: $OUT_DIR"
