#!/bin/bash
# Build the z_re(x, y) target cache (zre_targets.h5) for the 2-D z_re task.
# CPU-only: fits the per-pixel front midpoint (Gompertz) for each of the
# 6600 lightcones; idempotent per cone, so a resubmit resumes where it left
# off.  Training jobs (slurm/train_zre.sbatch) read the cache read-only.
# Submit: sbatch slurm/build_zre_targets.sbatch

#SBATCH --job-name=zre-targets
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/zre-targets-%j.out
#SBATCH --error=logs/zre-targets-%j.err

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

# each per-cone fit is one large BLAS matmul
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
ZRE_TARGET_CACHE="${ZRE_TARGET_CACHE:-$DATA_DIR/zre_targets.h5}"
TARGET_KIND="${TARGET_KIND:-gompertz}"

echo "lightcones: $LIGHTCONE_DIR"
echo "cache:      $ZRE_TARGET_CACHE"
echo "kind:       $TARGET_KIND"

python -m dataset.zre_target \
    --data "$LIGHTCONE_DIR" \
    --out "$ZRE_TARGET_CACHE" \
    --kind "$TARGET_KIND"
