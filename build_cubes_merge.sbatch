#!/bin/bash
# =============================================================================
# SLURM job -- merge cube shards into the single cubes_3d.h5 cache.
# Single-process; ~30 min for ~6600 cones at gzip-4.  Submit after the array
# build:
#
#     AID=$(sbatch --parsable build_cubes.sbatch)
#     sbatch --dependency=afterok:"$AID" build_cubes_merge.sbatch
#
# Or interactively after the array finishes.
# =============================================================================

#SBATCH --job-name=merge-cubes
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge-cubes-%j.out
#SBATCH --error=logs/merge-cubes-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

# MUST match build_cubes.sbatch.
NUM_SHARDS=33
OUT_PATH=/pfs/10/project/bw24d007/hk/cubes_3d.h5

python build_cubes.py --out "$OUT_PATH" --merge --num-shards "$NUM_SHARDS"
