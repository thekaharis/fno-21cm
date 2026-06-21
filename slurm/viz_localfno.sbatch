#!/bin/bash
# Standard Local-FNO visualization.

#SBATCH --job-name=viz-localfno
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/viz-localfno-%j.out
#SBATCH --error=logs/viz-localfno-%j.err

set -eo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs figures
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1
export MODEL_KIND=localfno
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_3d_localfno}"
export CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"
export VIZ_TAG="${VIZ_TAG:-localfno}"
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-64}"
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
export CUBES_CACHE="${CUBES_CACHE:-/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5}"
python -m viz.visualize_3d
