#!/bin/bash
# Edge / boundary + parity diagnostics for finished 3-D runs.
#
# The BSD and power-spectrum suites already have their own sbatch; these two
# diagnostics only had CLI entry points, so this wraps them the same way.
#
#   sbatch --export=ALL,\
#     MODEL_A_NAME=ufno-gnorm,MODEL_A_CHECKPOINT=./checkpoints/3d_xhi/ufno/checkpoints_3d_ufno_plain_gnorm/best_model_state_dict.pt,\
#     MODEL_B_NAME=fno_whno-plain,MODEL_B_CHECKPOINT=./checkpoints/3d_xhi/fno_whno/checkpoints_3d_fno_whno_plain/best_model_state_dict.pt \
#     slurm/edge_metrics_eval.sbatch
#
# Both scripts rebuild each model from the run_metadata.json beside its
# checkpoint, so no architecture env vars are needed -- a groupnorm U-FNO and a
# batchnorm one reconstruct correctly from their own metadata.
#
#SBATCH --job-name=edge-metrics-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/edge-metrics-eval-%j.out
#SBATCH --error=logs/edge-metrics-eval-%j.err

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
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
export CUBES_CACHE="${CUBES_CACHE:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"
# Inference-only knob: chunk size changes execution memory, not parameters.
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-64}"

MODEL_A_NAME="${MODEL_A_NAME:?set MODEL_A_NAME}"
MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:?set MODEL_A_CHECKPOINT}"
MODEL_B_NAME="${MODEL_B_NAME:-}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-}"

SPLIT="${SPLIT:-test}"
N_CONES="${N_CONES:-200}"
BOUNDARY_MODE="${BOUNDARY_MODE:-3d}"
OUT_DIR="${OUT_DIR:-figures/3d_xhi/eval/edge_metrics_out/job${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT_DIR"

checkpoints=("${MODEL_A_NAME}=${MODEL_A_CHECKPOINT}")
if [ -n "$MODEL_B_CHECKPOINT" ]; then
    checkpoints+=("${MODEL_B_NAME}=${MODEL_B_CHECKPOINT}")
fi
for spec in "${checkpoints[@]}"; do
    path="${spec#*=}"
    if [ ! -f "$path" ]; then
        echo "Checkpoint not found: $path" >&2
        exit 1
    fi
done

echo "================ edge metrics eval ================"
echo "Host:        $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Checkpoints: ${checkpoints[*]}"
echo "Split:       $SPLIT   cones: $N_CONES   boundary mode: $BOUNDARY_MODE"
echo "Out:         $OUT_DIR"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "==================================================="

echo "--- boundary band diagnostic (front width, wall RMSE, gradient error) ---"
python -m viz.boundary_band_diagnostic \
    --checkpoints "${checkpoints[@]}" \
    --split "$SPLIT" \
    --n-cones "$N_CONES" \
    --mode "$BOUNDARY_MODE" \
    --reference "$MODEL_A_NAME" \
    --out "$OUT_DIR/boundary"

echo "--- parity diagnostic (hedging bias per truth bin) ---"
python -m viz.parity_diagnostic \
    --checkpoints "${checkpoints[@]}" \
    --split "$SPLIT" \
    --n-cones "$N_CONES" \
    --out "$OUT_DIR/parity"

echo "Edge metrics evaluation complete: $OUT_DIR"
