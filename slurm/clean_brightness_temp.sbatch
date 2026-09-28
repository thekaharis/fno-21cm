#!/bin/bash
# Rebuild brightness_temp with the dv/dr term clipped (tools_clean_brightness_temp.py)
# into data/native_mirror_4f_tbclean; all other fields copied unchanged.
#   sbatch slurm/clean_brightness_temp.sbatch
#SBATCH --job-name=tb-clean
#SBATCH --partition=compute
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tb-clean-%A_%a.out
#SBATCH --error=logs/tb-clean-%A_%a.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
srun python tools_clean_brightness_temp.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 16
