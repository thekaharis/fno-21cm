#!/bin/bash
# =============================================================================
# Paired bubble-wall diagnostic: Local-FNO versus U-FNO.
#
# The script reconstructs each model from its own run_metadata.json, predicts
# the same test cones, and compares boundary-band RMSE/H1 and front width.
#
# Default submission:
#
#   sbatch slurm/boundary_localfno_vs_ufno.sbatch
#
# Compare against another U-FNO checkpoint:
#
#   sbatch --export=ALL,\
# UFNO_CHECKPOINT=checkpoints_3d_ufno_v3_anisoz/best_model_state_dict.pt \
#     slurm/boundary_localfno_vs_ufno.sbatch
#
# Short feasibility run:
#
#   sbatch --export=ALL,N_CONES=20 \
#     slurm/boundary_localfno_vs_ufno.sbatch
# =============================================================================

#SBATCH --job-name=boundary-localfno
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/boundary-localfno-%j.out
#SBATCH --error=logs/boundary-localfno-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs band_out

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- checkpoints
# Both checkpoint directories must contain run_metadata.json. This lets the
# diagnostic safely reconstruct different model architectures in one process.
UFNO_CHECKPOINT="${UFNO_CHECKPOINT:-./checkpoints_3d_ufno/best_model_state_dict.pt}"
LOCALFNO_CHECKPOINT="${LOCALFNO_CHECKPOINT:-./checkpoints_3d_localfno/best_model_state_dict.pt}"

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
MODE="${MODE:-slice}"
REFERENCE="${REFERENCE:-ufno}"
DX_MPC="${DX_MPC:-1.4285714286}"
D_MAX_MPC="${D_MAX_MPC:-30.0}"
BIN_MPC="${BIN_MPC:-1.5}"
OUT_DIR="${OUT_DIR:-band_out/localfno-vs-ufno-${SLURM_JOB_ID:-local}}"

# Reducing this value changes execution memory only, not checkpoint weights.
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-64}"

# SAVE_CUBES=yes stores each prediction as compressed NPZ for later offline
# re-analysis. For 200 cones and two models this can consume substantial space.
SAVE_CUBES="${SAVE_CUBES:-no}"
SAVED_CUBES_DIR="${SAVED_CUBES_DIR:-${OUT_DIR}/cubes}"

# Optional redshift restriction. Set both values to enable it, for example:
#   --export=ALL,Z_WINDOW_LOW=6,Z_WINDOW_HIGH=12
Z_WINDOW_LOW="${Z_WINDOW_LOW:-}"
Z_WINDOW_HIGH="${Z_WINDOW_HIGH:-}"
if { [ -n "$Z_WINDOW_LOW" ] && [ -z "$Z_WINDOW_HIGH" ]; } ||
   { [ -z "$Z_WINDOW_LOW" ] && [ -n "$Z_WINDOW_HIGH" ]; }; then
    echo "Set both Z_WINDOW_LOW and Z_WINDOW_HIGH, or neither." >&2
    exit 1
fi

# ---------------------------------------------------------------- data cache
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
PROJECT_CUBES="${PROJECT_CUBES:-/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5}"
LOCAL_SCRATCH="${LOCAL_SCRATCH:-yes}"

if [ "$LOCAL_SCRATCH" = "yes" ]; then
    SCRATCH_BASE=""
    for candidate in "${TMPDIR:-}" "/scratch/${SLURM_JOB_ID}" "/local/${SLURM_JOB_ID}"; do
        if [ -n "$candidate" ] && [ -d "$candidate" ] && [ -w "$candidate" ]; then
            SCRATCH_BASE="$candidate"
            break
        fi
    done
    if [ -n "$SCRATCH_BASE" ]; then
        LOCAL_CUBES="$SCRATCH_BASE/cubes_3d.h5"
        echo "Staging cube cache to local storage: $LOCAL_CUBES"
        cp --reflink=auto "$PROJECT_CUBES" "$LOCAL_CUBES" ||
            cp "$PROJECT_CUBES" "$LOCAL_CUBES"
        export CUBES_CACHE="$LOCAL_CUBES"
    else
        echo "No writable local scratch found; using project storage."
        export CUBES_CACHE="${CUBES_CACHE:-$PROJECT_CUBES}"
    fi
else
    export CUBES_CACHE="${CUBES_CACHE:-$PROJECT_CUBES}"
fi

if [ ! -f "$CUBES_CACHE" ]; then
    echo "Cube cache not found: $CUBES_CACHE" >&2
    exit 1
fi

# ---------------------------------------------------------------- diagnostics
echo "================ boundary-band diagnostic ================"
echo "Host:                 $(hostname)"
echo "Job ID:               ${SLURM_JOB_ID:-N/A}"
echo "U-FNO checkpoint:     $UFNO_CHECKPOINT"
echo "Local-FNO checkpoint: $LOCALFNO_CHECKPOINT"
echo "Reference:            $REFERENCE"
echo "Split / cones:        $SPLIT / $N_CONES"
echo "Mode:                 $MODE"
echo "Z window:             ${Z_WINDOW_LOW:-all} ${Z_WINDOW_HIGH:-all}"
echo "Local-FNO chunk:      $LOCALFNO_PATCH_CHUNK_SIZE"
echo "Cube cache:           $CUBES_CACHE"
echo "Save predictions:     $SAVE_CUBES"
echo "Output:               $OUT_DIR"
nvidia-smi -L || true
echo "==========================================================="

args=(
    boundary_band_diagnostic.py
    --checkpoints
    "ufno=$UFNO_CHECKPOINT"
    "localfno=$LOCALFNO_CHECKPOINT"
    --reference "$REFERENCE"
    --split "$SPLIT"
    --n-cones "$N_CONES"
    --mode "$MODE"
    --dx-mpc "$DX_MPC"
    --d-max-mpc "$D_MAX_MPC"
    --bin-mpc "$BIN_MPC"
    --out "$OUT_DIR"
)

if [ "$SAVE_CUBES" = "yes" ]; then
    args+=(--save-cubes "$SAVED_CUBES_DIR")
fi

if [ -n "$Z_WINDOW_LOW" ]; then
    args+=(--z-window "$Z_WINDOW_LOW" "$Z_WINDOW_HIGH")
fi

python "${args[@]}"

echo "Boundary diagnostic complete: $OUT_DIR"
