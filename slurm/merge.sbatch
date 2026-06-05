#!/bin/bash
# =============================================================================
# Combine the shard caches from build.sbatch into a single trainset.h5.
# Run after the build array completes (chain it with a dependency):
#
#     AID=$(sbatch --parsable slurm/build.sbatch)
#     sbatch --dependency=afterok:"$AID" slurm/merge.sbatch
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

NUM_SHARDS=33                                       # MUST match build.sbatch

python build_trainset.py --out trainset.h5 --merge --num-shards "$NUM_SHARDS"
echo "merge complete -> trainset.h5"
