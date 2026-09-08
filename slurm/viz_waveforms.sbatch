#!/bin/bash
# Checkpoint-only plots of every learned branch/axis; no GPU or dataset needed.
# mkdir -p logs
# sbatch --export=ALL,CHECKPOINT_DIR=checkpoints/my_run slurm/viz_waveforms.sbatch
# For older metadata: export WAVEFORM_INPUT_SHAPE="140 140 256" before sbatch,
# using the actual training shape (z_re has only TWO spatial dimensions).
#SBATCH --job-name=viz-waveforms
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/viz-waveforms-%j.out
#SBATCH --error=logs/viz-waveforms-%j.err

set -eo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR to a learned-waveform run directory}"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "CHECKPOINT_DIR does not exist: $CHECKPOINT_DIR" >&2
    exit 2
fi
mkdir -p logs figures
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-waveforms-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

args=(--checkpoint-dir "$CHECKPOINT_DIR" --checkpoint-kind "${CHECKPOINT_KIND:-best}"
      --max-modes "${WAVEFORM_VIZ_MODES:-6}")
if [ -n "${WAVEFORM_INPUT_SHAPE:-}" ]; then
    read -r -a spatial_shape <<< "$WAVEFORM_INPUT_SHAPE"
    args+=(--input-shape "${spatial_shape[@]}")
fi
if [ -n "${OUT_DIR:-}" ]; then
    args+=(--out-dir "$OUT_DIR")
fi
python -m viz.learned_waveforms "${args[@]}"
