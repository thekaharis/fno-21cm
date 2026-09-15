#!/bin/bash
# Identify raw lightcones containing non-finite voxels, before the cache build.
#   sbatch slurm/scan_nonfinite.sbatch
#SBATCH --job-name=scan-nonfinite
#SBATCH --partition=compute
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/scan-nonfinite-%A_%a.out
#SBATCH --error=logs/scan-nonfinite-%A_%a.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
srun python tools_scan_nonfinite.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 16 \
     --out experiments/multifield/nonfinite_scan
