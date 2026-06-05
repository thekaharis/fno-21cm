#!/bin/bash
# =============================================================================
# SLURM job -- merge cube shards into the single cubes_3d.h5 cache.
# Single-process; ~30 min for ~6600 cones at gzip-4.  Submit after the array
# build:
#
#     AID=$(sbatch --parsable slurm/build_cubes.sbatch)
#     sbatch --dependency=afterok:"$AID" slurm/build_cubes_merge.sbatch
#
# Or interactively after the array finishes.
# =============================================================================

#SBATCH --job-name=merge-cubes
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge-cubes-%j.out
#SBATCH --error=logs/merge-cubes-%j.err

set -eo pipefail
# Run from the project root (sbatch scripts live in slurm/, the Python entry
# points one level up).  Resolves regardless of submission directory.
# Under SLURM, SLURM_SUBMIT_DIR is set to the dir from which sbatch was
# invoked (the convention is to submit from the project root).  Outside
# SLURM, fall back to this script's own location -> ../
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

# MUST match build_cubes.sbatch.
NUM_SHARDS=33
OUT_PATH=/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5

python build_cubes.py --out "$OUT_PATH" --merge --num-shards "$NUM_SHARDS"
