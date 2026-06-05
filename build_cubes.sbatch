#!/bin/bash
# =============================================================================
# SLURM ARRAY job -- precompute 3-D cubes from raw lightcones.
# CPU only (I/O bound, single-process per task).
#
# Each array task handles a strided subset of the files and writes one
# cubes_3d.shardNNN.h5.  After the array finishes, run the merge step:
#
#     AID=$(sbatch --parsable build_cubes.sbatch)
#     sbatch --dependency=afterok:"$AID" build_cubes_merge.sbatch
#
# Or run the merge as a single follow-up command in an interactive session
# (it's serial and ~30 min for ~6600 cones).
#
# IMPORTANT: NUM_SHARDS (below) MUST equal the array width.
#   --array=0-32  -> 33 tasks -> NUM_SHARDS=33
# Set LIGHTCONE_DIR and --partition before the first run (`sinfo -s`).
# =============================================================================

#SBATCH --job-name=build-cubes
#SBATCH --partition=cpu
#SBATCH --array=0-32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build-cubes-%A_%a.out
#SBATCH --error=logs/build-cubes-%A_%a.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

# Lightcone directory (same as train.sbatch / build.sbatch).
LIGHTCONE_DIR=/pfs/10/project/bw24d007/hk/lightcones
NUM_SHARDS=33                       # MUST equal the array width above

# Pick where the cache lives.  Project FS is the right home -- the file will
# be 100-250 GB depending on compression.  The training script reads
# CUBES_CACHE if set, otherwise looks for ./cubes_3d.h5 next to the script.
OUT_PATH=/pfs/10/project/bw24d007/hk/cubes_3d.h5

echo "shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} on $(hostname)"
python build_cubes.py \
    --data "$LIGHTCONE_DIR" \
    --out "$OUT_PATH" \
    --n-z 256 --z-min 5.0 --z-max 25.0 \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS"
