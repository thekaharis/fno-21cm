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

# Node spec (per cluster docs): 4x H200 NVL (141 GB HBM3e ea., NVLink),
#                               2x AMD EPYC Milan 9555 = 128 cores / 256 threads,
#                               1536 GB DDR5 RAM, 28 TB local NVMe SSD,
#                               HDR 200 IB + 100 GbE.
# Each task pinned to one GPU (one DDP rank), 32 cores per rank evenly tiles
# the 128-core node; 512 GB RAM is comfortable for 4 ranks of cube loaders +
# OS page cache for the 180 GB cube cache (when copied to local NVMe).

#SBATCH --job-name=fno-3d-4gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=24:00:00
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
PROJECT_CUBES=/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5

# Stage cube cache on local NVMe scratch (node has 28 TB NVMe SSD) so all 4
# ranks read from ~3 GB/s local storage instead of contending on the shared
# project FS.  One-time copy cost (~5-10 min for 180 GB) is paid back many
# times over across the run.  Skip by setting LOCAL_SCRATCH=no in the env.
LOCAL_SCRATCH="${LOCAL_SCRATCH:-yes}"

# Discover the per-job NVMe scratch path -- conventions vary by cluster.
# Most SLURM sites set TMPDIR; some expose /scratch/$SLURM_JOB_ID; some both.
if [ "$LOCAL_SCRATCH" = "yes" ]; then
    SCRATCH_BASE=""
    for candidate in "${TMPDIR:-}" "/scratch/${SLURM_JOB_ID}" "/local/${SLURM_JOB_ID}"; do
        if [ -n "$candidate" ] && [ -d "$candidate" ] && [ -w "$candidate" ]; then
            SCRATCH_BASE="$candidate"; break
        fi
    done
    if [ -n "$SCRATCH_BASE" ]; then
        LOCAL_CUBES="$SCRATCH_BASE/cubes_3d.h5"
        echo "Staging cube cache to local NVMe: $LOCAL_CUBES"
        t0=$(date +%s)
        # Use cp -L to follow symlinks; --reflink=auto saves time on CoW FS.
        cp --reflink=auto "$PROJECT_CUBES" "$LOCAL_CUBES" || cp "$PROJECT_CUBES" "$LOCAL_CUBES"
        echo "  staged ($(du -h "$LOCAL_CUBES" | cut -f1)) in $(( $(date +%s) - t0 ))s"
        export CUBES_CACHE="$LOCAL_CUBES"
    else
        echo "No local scratch detected (TMPDIR/scratch unset); reading cubes from project FS."
        export CUBES_CACHE="$PROJECT_CUBES"
    fi
else
    echo "LOCAL_SCRATCH=no: reading cubes from project FS."
    export CUBES_CACHE="$PROJECT_CUBES"
fi

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
