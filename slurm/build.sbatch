#!/bin/bash
# =============================================================================
# SLURM ARRAY job — extract slices from the lightcones into shard caches.
# CPU only (no GPU); this is an I/O-bound, single-process pass per task.
#
# Each array task handles a strided subset of the files and writes one
# trainset.shardNNN.h5.  After the array finishes, merge.sbatch combines them.
# Submit both as a dependency chain so the merge runs automatically:
#
#     AID=$(sbatch --parsable slurm/build.sbatch)
#     sbatch --dependency=afterok:"$AID" slurm/merge.sbatch
#
# IMPORTANT: NUM_SHARDS (below) MUST equal the array width.
#   --array=0-32  -> 33 tasks -> NUM_SHARDS=33
# Set DATA_DIR and --partition before the first run (`sinfo -s` for partitions).
# Optional: throttle concurrency with e.g. --array=0-32%16
# =============================================================================

#SBATCH --job-name=build-trainset
#SBATCH --partition=compute
#SBATCH --array=0-32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build-%A_%a.out
#SBATCH --error=logs/build-%A_%a.err

set -eo pipefail
# Run from the project root (sbatch scripts live in slurm/, the Python entry
# points one level up).  Resolves regardless of submission directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

DATA_DIR=/pfs/10/work/hd_id260-fno_training/data/data   # dir with the 6600 .h5 files
NUM_SHARDS=33                                       # MUST equal the array width

echo "shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} on $(hostname)"
python build_trainset.py \
    --data "$DATA_DIR" \
    --out trainset.h5 \
    --k 6 \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS"
