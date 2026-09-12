#!/bin/bash
# One-epoch 4xH200 Local-FNO smoke test with production cube shape.

#SBATCH --job-name=smoke-localfno
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke-localfno-%j.out
#SBATCH --error=logs/smoke-localfno-%j.err

set -eo pipefail

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

export MODEL_KIND=localfno
export N_EPOCHS=1
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/3d_xhi/localfno/checkpoints_3d_localfno_smoke}"
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-128}"
export LOCALFNO_LEARNING_RATE="${LOCALFNO_LEARNING_RATE:-1e-4}"
export LOCALFNO_H1_WARMUP_EPOCHS=0
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
export CUBES_CACHE="${CUBES_CACHE:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "Running one-epoch Local-FNO DDP smoke test; inspect peak HBM with nvidia-smi."
nvidia-smi -L || true
srun python fno_21cm_3d.py
