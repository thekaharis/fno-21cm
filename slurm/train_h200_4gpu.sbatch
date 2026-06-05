#!/bin/bash
# =============================================================================
# SLURM batch script -- 4-GPU H200 DDP training of the 3-D FNO.
#
# Submit from the project root:
#
#     sbatch slurm/train_h200_4gpu.sbatch
#
# Each of the 4 SLURM tasks becomes one PyTorch DDP rank, pinned to one of
# the node's four H200 NVL GPUs.  Gradients are averaged via NVLink at every
# backward step; data is split via DistributedSampler so each rank sees a
# disjoint shard of the train / val / test cones each epoch.
#
# Expected throughput on the H200 node (141 GB/GPU, 2x AMD EPYC Milan 9555):
#   ~3.5-4x speedup over single H200  ->  ~25-28 samples/s end-to-end
#   -> ~4-5 min per epoch  ->  100 epochs in ~7-8 h  (single submission).
#
# If anything goes sideways with DDP and you need to fall back to single GPU,
# `sbatch slurm/train.sbatch` still works -- the Python entry point auto-
# detects multi-task SLURM and runs single-process otherwise.
# =============================================================================

#SBATCH --job-name=fno-3d-4gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4                  # ONE task per GPU -- becomes one DDP rank
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16          # 64 cores total, comfortable on this node
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G                  # node has 1.5 TB; 256 GB is generous
#SBATCH --time=12:00:00
#SBATCH --output=logs/fno3d-4gpu-%j.out
#SBATCH --error=logs/fno3d-4gpu-%j.err

set -eo pipefail

# --- Run from the project root ----------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

# --- Conda env --------------------------------------------------------------
module load devel/miniforge
ENV_NAME="fno-env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- Thread / I/O hygiene ---------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# --- Data paths -------------------------------------------------------------
export LIGHTCONE_DIR=/pfs/10/work/hd_id260-fno_training/data/data
export CUBES_CACHE=/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5

# --- Distributed master coordinates -----------------------------------------
# The first node in SLURM_NODELIST becomes the rendezvous master.  Port is
# offset by job id so two concurrent jobs on the same node don't collide.
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- Diagnostics (printed once by the bash driver, before srun fans out) ----
echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "Submit dir:         ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Tasks (ranks):      ${SLURM_NTASKS:-1}"
echo "Tasks/node:         ${SLURM_NTASKS_PER_NODE:-1}"
echo "CPUs/task:          ${SLURM_CPUS_PER_TASK:-1}"
echo "MASTER_ADDR:        ${MASTER_ADDR}"
echo "MASTER_PORT:        ${MASTER_PORT}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "================================================="

# --- Launch -----------------------------------------------------------------
# srun spawns SLURM_NTASKS processes; each one sees SLURM_PROCID,
# SLURM_LOCALID, SLURM_NTASKS in its env, which fno_21cm_3d.py uses to
# initialize torch.distributed and pin to the local GPU.
srun python fno_21cm_3d.py
