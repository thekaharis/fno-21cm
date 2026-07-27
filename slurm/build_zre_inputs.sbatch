#!/bin/bash
# Build the z_re-task INPUT cache (zre_inputs.h5): per-cone density slices
# interpolated to the training z-grid + sampled params, so training jobs
# open one sidecar instead of re-reading 6600 raw lightcones (~1.5-2 h)
# at every start.  CPU-only, idempotent per cone (resubmit resumes).
# Submit: sbatch slurm/build_zre_inputs.sbatch

#SBATCH --job-name=zre-inputs
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zre-inputs-%j.out
#SBATCH --error=logs/zre-inputs-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

# Datasets live outside the repo; see dataset/paths.py
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/../data/compressed}"
mkdir -p logs

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
ZRE_INPUT_CACHE="${ZRE_INPUT_CACHE:-$DATA_DIR/zre_inputs.h5}"
N_Z_IN="${N_Z_IN:-64}"

echo "lightcones: $LIGHTCONE_DIR"
echo "cache:      $ZRE_INPUT_CACHE"
echo "n_z_in:     $N_Z_IN"

python -m dataset.zre_input_cache \
    --data "$LIGHTCONE_DIR" \
    --out "$ZRE_INPUT_CACHE" \
    --n-z-in "$N_Z_IN"
