#!/bin/bash
# Rewrite the 2000-cone subset into a window-friendly native mirror (4 fields).
#   sbatch slurm/build_native_mirror.sbatch
#SBATCH --job-name=native-mirror
#SBATCH --partition=compute
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/native-mirror-%A_%a.out
#SBATCH --error=logs/native-mirror-%A_%a.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
OUT="${OUT:-/pfs/10/work/hd_id260-fno_training/data/native_mirror_4f}"
srun python tools_build_native_mirror.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 16 --out "$OUT"
