#!/bin/bash
# Run one evaluation family over an arbitrary number of checkpoints.
#
# The per-family sbatches (bubble_size_eval, power_spectrum_eval,
# edge_metrics_eval) each hardcode two model slots, which is fine for an A/B
# but not for a leaderboard. Every viz entry point already takes
# `--checkpoints name=path ...` with nargs="+", and each loads its model lazily
# inside a per-model generator, so N models cost N sequential forward passes
# rather than N models resident at once.
#
#   SPECS_FILE  file with one `name=path` per line (blank lines/# ignored)
#   SUITE       bsd | ps | edge3d | edgeslice | parity | rmse
#
#   sbatch --export=ALL,SUITE=bsd,SPECS_FILE=figures/final_eval/specs.txt \
#     slurm/final_eval_suite.sbatch
#
# A100 rather than A30: the bw48om60 cells peaked at 42 GB in training, and
# while inference is far lighter, an OOM halfway through a 12-model sweep costs
# more than the queue wait.
#
#SBATCH --job-name=final-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/final-eval-%j.out
#SBATCH --error=logs/final-eval-%j.err

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
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-64}"

SUITE="${SUITE:?set SUITE to bsd|ps|edge3d|edgeslice|parity|rmse}"
SPECS_FILE="${SPECS_FILE:?set SPECS_FILE}"
[ -f "$SPECS_FILE" ] || { echo "no such SPECS_FILE: $SPECS_FILE" >&2; exit 2; }

SPECS=()
while read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    path="${line#*=}"
    [ -f "$path" ] || { echo "checkpoint missing: $path" >&2; exit 2; }
    SPECS+=("$line")
done < "$SPECS_FILE"
[ ${#SPECS[@]} -gt 0 ] || { echo "SPECS_FILE is empty" >&2; exit 2; }

SPLIT="${SPLIT:-test}"
N_CONES="${N_CONES:-200}"
OUT_ROOT="${OUT_ROOT:-figures/final_eval}"
OUT="$OUT_ROOT/$SUITE"
mkdir -p "$OUT"

echo "================ final eval: $SUITE ================"
echo "Host: $(hostname)   Job: ${SLURM_JOB_ID:-N/A}"
echo "Models (${#SPECS[@]}): ${SPECS[*]%%=*}"
echo "Split: $SPLIT   cones: $N_CONES   out: $OUT"
nvidia-smi -L || true
echo "===================================================="

case "$SUITE" in
  bsd)
    python -m viz.bubble_size_evaluation --checkpoints "${SPECS[@]}" \
        --split "$SPLIT" --n-cones "$N_CONES" --out "$OUT" ;;
  ps)
    python -m viz.power_spectrum_evaluation --checkpoints "${SPECS[@]}" \
        --split "$SPLIT" --n-cones "$N_CONES" --out "$OUT" ;;
  edge3d)
    python -m viz.boundary_band_diagnostic --checkpoints "${SPECS[@]}" \
        --split "$SPLIT" --n-cones "$N_CONES" --mode 3d \
        --reference "${SPECS[0]%%=*}" --out "$OUT" ;;
  edgeslice)
    python -m viz.boundary_band_diagnostic --checkpoints "${SPECS[@]}" \
        --split "$SPLIT" --n-cones "$N_CONES" --mode slice \
        --reference "${SPECS[0]%%=*}" --out "$OUT" ;;
  parity)
    python -m viz.parity_diagnostic --checkpoints "${SPECS[@]}" \
        --split "$SPLIT" --n-cones "$N_CONES" --out "$OUT" ;;
  rmse)
    python -m viz.rmse_r2_eval --checkpoints "${SPECS[@]}" \
        --split "$SPLIT" --n-cones "$N_CONES" --out "$OUT" ;;
  *) echo "unknown SUITE=$SUITE" >&2; exit 2 ;;
esac

echo "final eval complete: $SUITE -> $OUT"
