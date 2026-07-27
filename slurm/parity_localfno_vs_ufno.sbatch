#!/bin/bash
# =============================================================================
# Binned parity plot (calibration curve): Local-FNO versus U-FNO.
#
# Pairs (true, predicted) x_HI per voxel over the test cones, bins by the true
# value, and plots E[pred | true] with a 16-84% band against the identity
# line, plus per-bin occupancy. Shows which x_HI range each model gets wrong.
#
# Default submission:
#
#   sbatch slurm/parity_localfno_vs_ufno.sbatch
#
# Short feasibility run / different checkpoints:
#
#   sbatch --export=ALL,N_CONES=20 slurm/parity_localfno_vs_ufno.sbatch
#   sbatch --export=ALL,\
# UFNO_CHECKPOINT=checkpoint-archive/checkpoints_3d_ufno_v2/best_model_state_dict.pt \
#     slurm/parity_localfno_vs_ufno.sbatch
#
# Restrict to the active reionization window:
#
#   sbatch --export=ALL,Z_WINDOW_LOW=6,Z_WINDOW_HIGH=12 \
#     slurm/parity_localfno_vs_ufno.sbatch
# =============================================================================

#SBATCH --job-name=parity-localfno
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/parity-localfno-%j.out
#SBATCH --error=logs/parity-localfno-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs parity_out

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- checkpoints
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
TRUE_BINS="${TRUE_BINS:-50}"
PRED_BINS="${PRED_BINS:-400}"
OUT_DIR="${OUT_DIR:-parity_out/localfno-vs-ufno-${SLURM_JOB_ID:-local}}"

# Reducing this value changes execution memory only, not checkpoint weights.
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-32}"

# Optional redshift restriction. Set both values to enable it.
Z_WINDOW_LOW="${Z_WINDOW_LOW:-}"
Z_WINDOW_HIGH="${Z_WINDOW_HIGH:-}"
if { [ -n "$Z_WINDOW_LOW" ] && [ -z "$Z_WINDOW_HIGH" ]; } ||
   { [ -z "$Z_WINDOW_LOW" ] && [ -n "$Z_WINDOW_HIGH" ]; }; then
    echo "Set both Z_WINDOW_LOW and Z_WINDOW_HIGH, or neither." >&2
    exit 1
fi

# ---------------------------------------------------------------- data cache
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
PROJECT_CUBES="${PROJECT_CUBES:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"
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
echo "================ parity diagnostic ================"
echo "Host:                 $(hostname)"
echo "Job ID:               ${SLURM_JOB_ID:-N/A}"
echo "U-FNO checkpoint:     $UFNO_CHECKPOINT"
echo "Local-FNO checkpoint: $LOCALFNO_CHECKPOINT"
echo "Split / cones:        $SPLIT / $N_CONES"
echo "Bins (true/pred):     $TRUE_BINS / $PRED_BINS"
echo "Z window:             ${Z_WINDOW_LOW:-all} ${Z_WINDOW_HIGH:-all}"
echo "Local-FNO chunk:      $LOCALFNO_PATCH_CHUNK_SIZE"
echo "Cube cache:           $CUBES_CACHE"
echo "Output:               $OUT_DIR"
nvidia-smi -L || true
echo "==================================================="

args=(
    -m viz.parity_diagnostic
    --checkpoints
    "ufno=$UFNO_CHECKPOINT"
    "localfno=$LOCALFNO_CHECKPOINT"
    --split "$SPLIT"
    --n-cones "$N_CONES"
    --true-bins "$TRUE_BINS"
    --pred-bins "$PRED_BINS"
    --out "$OUT_DIR"
)

if [ -n "$Z_WINDOW_LOW" ]; then
    args+=(--z-window "$Z_WINDOW_LOW" "$Z_WINDOW_HIGH")
fi

python "${args[@]}"

echo "Parity diagnostic complete: $OUT_DIR"
