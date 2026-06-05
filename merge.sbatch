#!/bin/bash
# =============================================================================
# Combine the shard caches from build.sbatch into a single trainset.h5.
# Run after the build array completes (chain it with a dependency):
#
#     AID=$(sbatch --parsable build.sbatch)
#     sbatch --dependency=afterok:"$AID" merge.sbatch
#
# NUM_SHARDS must match the value used in build.sbatch.
# =============================================================================

#SBATCH --job-name=merge-trainset
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge-%j.out
#SBATCH --error=logs/merge-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

NUM_SHARDS=33                                       # MUST match build.sbatch

python build_trainset.py --out trainset.h5 --merge --num-shards "$NUM_SHARDS"
echo "merge complete -> trainset.h5"
