#!/bin/bash
# =============================================================================
# Stage-resolved transverse mean-free-path bubble-size evaluation.
#
# Defaults to the two current 3-D LocalSirenFNO loss variants. Override model
# labels/checkpoints, cone count, sampling density, or output directory with
# sbatch --export=ALL,...
#
#   sbatch slurm/bubble_size_eval.sbatch
#   sbatch --export=ALL,N_CONES=20,RAYS_PER_SLICE=128 \
#     slurm/bubble_size_eval.sbatch
# =============================================================================
#SBATCH --job-name=bubble-size-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bubble-size-eval-%j.out
#SBATCH --error=logs/bubble-size-eval-%j.err

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

MODEL_A_NAME="${MODEL_A_NAME:-localsirenfno-l2h1}"
MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-./checkpoints/3d_xhi/localsirenfno/checkpoints_3d_localsirenfno/best_model_state_dict.pt}"
MODEL_B_NAME="${MODEL_B_NAME:-localsirenfno-l2only}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-./checkpoints/3d_xhi/localsirenfno/checkpoints_3d_localsirenfno_l2only/best_model_state_dict.pt}"

# Optional third model: the granulometry experiment needs plain / hybrid / bsd
# side by side, and comparing them pairwise would recompute the truth BSD twice
# and give two incomparable percentile bands.
MODEL_C_NAME="${MODEL_C_NAME:-}"
MODEL_C_CHECKPOINT="${MODEL_C_CHECKPOINT:-}"

for checkpoint in "$MODEL_A_CHECKPOINT" "$MODEL_B_CHECKPOINT" ${MODEL_C_CHECKPOINT:+"$MODEL_C_CHECKPOINT"}; do
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

N_CONES="${N_CONES:-200}"
SPLIT="${SPLIT:-test}"
RAYS_PER_SLICE="${RAYS_PER_SLICE:-256}"
SLICES_PER_STAGE="${SLICES_PER_STAGE:-6}"
N_BINS="${N_BINS:-24}"
THRESHOLD="${THRESHOLD:-0.5}"
BOX_MPC="${BOX_MPC:-200.0}"
MAX_DISTANCE_MPC="${MAX_DISTANCE_MPC:-200.0}"
OUT_DIR="${OUT_DIR:-figures/3d_xhi/eval/bubble_size_out/job${SLURM_JOB_ID:-local}}"
SAVE_CUBES="${SAVE_CUBES:-no}"
SAVED_CUBES_DIR="${SAVED_CUBES_DIR:-${OUT_DIR}/cubes}"

# Reduces inference memory only; checkpoint architecture is unchanged.
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-32}"

export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
PROJECT_CUBES="${PROJECT_CUBES:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"
LOCAL_SCRATCH="${LOCAL_SCRATCH:-auto}"
if [ "$LOCAL_SCRATCH" != "auto" ] && [ "$LOCAL_SCRATCH" != "yes" ] && [ "$LOCAL_SCRATCH" != "no" ]; then
    echo "LOCAL_SCRATCH must be auto, yes, or no" >&2
    exit 1
fi
if [ ! -f "$PROJECT_CUBES" ]; then
    echo "Cube cache not found: $PROJECT_CUBES" >&2
    exit 1
fi

stage_cache="no"
if [ "$LOCAL_SCRATCH" = "yes" ]; then
    stage_cache="yes"
elif [ "$LOCAL_SCRATCH" = "auto" ] && [ -n "${TMPDIR:-}" ]; then
    cache_bytes="$(stat -c %s "$PROJECT_CUBES")"
    scratch_bytes="$(df --output=avail -B1 "$TMPDIR" | tail -n 1 | tr -d ' ')"
    reserve_bytes=$((10 * 1024 * 1024 * 1024))
    if [ "$scratch_bytes" -gt $((cache_bytes + reserve_bytes)) ]; then
        stage_cache="yes"
    else
        echo "Local scratch cannot hold the cache plus 10 GiB headroom; reading PFS directly"
    fi
fi

if [ "$stage_cache" = "yes" ]; then
    if [ -z "${TMPDIR:-}" ]; then
        echo "LOCAL_SCRATCH=yes requires TMPDIR" >&2
        exit 1
    fi
    cache_bytes="$(stat -c %s "$PROJECT_CUBES")"
    scratch_bytes="$(df --output=avail -B1 "$TMPDIR" | tail -n 1 | tr -d ' ')"
    reserve_bytes=$((10 * 1024 * 1024 * 1024))
    if [ "$scratch_bytes" -le $((cache_bytes + reserve_bytes)) ]; then
        echo "Insufficient local scratch for cache plus 10 GiB headroom" >&2
        exit 1
    fi
    LOCAL_CUBES="$TMPDIR/cubes_3d.h5"
    echo "Staging cube cache to local scratch: $LOCAL_CUBES"
    cp "$PROJECT_CUBES" "$LOCAL_CUBES"
    export CUBES_CACHE="$LOCAL_CUBES"
else
    export CUBES_CACHE="${CUBES_CACHE:-$PROJECT_CUBES}"
fi

echo "================ bubble-size evaluation ================"
echo "Host / job:          $(hostname) / ${SLURM_JOB_ID:-N/A}"
echo "Model A:             $MODEL_A_NAME=$MODEL_A_CHECKPOINT"
echo "Model B:             $MODEL_B_NAME=$MODEL_B_CHECKPOINT"
[ -n "$MODEL_C_CHECKPOINT" ] && echo "Model C:             $MODEL_C_NAME=$MODEL_C_CHECKPOINT"
echo "Split / cones:       $SPLIT / $N_CONES"
echo "Rays / slices:       $RAYS_PER_SLICE / $SLICES_PER_STAGE per stage"
echo "Threshold / max MFP: $THRESHOLD / $MAX_DISTANCE_MPC cMpc"
echo "Output:              $OUT_DIR"
echo "========================================================"

args=(
    -m viz.bubble_size_evaluation
    --checkpoints
    "${MODEL_A_NAME}=${MODEL_A_CHECKPOINT}"
    "${MODEL_B_NAME}=${MODEL_B_CHECKPOINT}"
    ${MODEL_C_CHECKPOINT:+"${MODEL_C_NAME}=${MODEL_C_CHECKPOINT}"}
    --split "$SPLIT"
    --n-cones "$N_CONES"
    --rays-per-slice "$RAYS_PER_SLICE"
    --slices-per-stage "$SLICES_PER_STAGE"
    --n-bins "$N_BINS"
    --threshold "$THRESHOLD"
    --box-mpc "$BOX_MPC"
    --max-distance-mpc "$MAX_DISTANCE_MPC"
    --out "$OUT_DIR"
)
if [ "$SAVE_CUBES" = "yes" ]; then
    args+=(--save-cubes "$SAVED_CUBES_DIR")
fi

python "${args[@]}"
echo "Bubble-size evaluation complete: $OUT_DIR"
