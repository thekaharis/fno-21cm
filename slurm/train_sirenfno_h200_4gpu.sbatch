#!/bin/bash
# =============================================================================
# 4-GPU H200 DDP training for the 3-D SirenFNO.
#
# Submit from the project root:
#
#     sbatch slurm/train_sirenfno_h200_4gpu.sbatch
#
# Override individual defaults with --export, for example:
#
#     sbatch --export=ALL,N_MODES_Z=32,SIREN_HIDDEN_DIM=96 \
#         slurm/train_sirenfno_h200_4gpu.sbatch
# =============================================================================

#SBATCH --job-name=sirenfno-3d-4gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sirenfno3d-4gpu-%j.out
#SBATCH --error=logs/sirenfno3d-4gpu-%j.err

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

# --- SirenFNO architecture --------------------------------------------------
export MODEL_KIND=sirenfno
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/checkpoints_3d_sirenfno_m64_stable}"
# Set INIT_CHECKPOINT to warm-start from a prior feasibility run. Optimizer
# and cosine scheduler state are intentionally reset for the new full run.
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

export N_MODES_X="${N_MODES_X:-64}"
export N_MODES_Y="${N_MODES_Y:-64}"
export N_MODES_Z="${N_MODES_Z:-64}"

export SIREN_HIDDEN_DIM="${SIREN_HIDDEN_DIM:-64}"
export SIREN_OMEGA="${SIREN_OMEGA:-30.0}"
export SIREN_N_HIDDEN="${SIREN_N_HIDDEN:-1}"
export SIREN_FEATURE_DIM="${SIREN_FEATURE_DIM:-16}"
export SIREN_FF_SIGMA="${SIREN_FF_SIGMA:-128.0}"
export SIREN_LEARNABLE_FF="${SIREN_LEARNABLE_FF:-true}"
export SIREN_PADDING_X="${SIREN_PADDING_X:-0}"
export SIREN_PADDING_Y="${SIREN_PADDING_Y:-0}"
export SIREN_PADDING_Z="${SIREN_PADDING_Z:-8}"
export SIREN_MLP_DROPOUT="${SIREN_MLP_DROPOUT:-0.0}"
export SIREN_OUTPUT_SIGMOID="${SIREN_OUTPUT_SIGMOID:-true}"
export SIREN_SIGMOID_TEMPERATURE="${SIREN_SIGMOID_TEMPERATURE:-2.0}"

# --- SirenFNO optimization stability ----------------------------------------
# On 4 GPUs the default sqrt LR scaling turns 1e-4 into an effective 2e-4.
export SIRENFNO_LEARNING_RATE="${SIRENFNO_LEARNING_RATE:-1e-4}"
export SIRENFNO_H1_WARMUP_EPOCHS="${SIRENFNO_H1_WARMUP_EPOCHS:-5}"
export SIRENFNO_GRAD_CLIP_NORM="${SIRENFNO_GRAD_CLIP_NORM:-1.0}"
# At roughly 17 minutes per epoch, 70 epochs fit inside the 24-hour request.
export N_EPOCHS="${N_EPOCHS:-70}"

# --- Data paths -------------------------------------------------------------
export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
PROJECT_CUBES="${PROJECT_CUBES:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"

# Stage the cube cache on the node-local NVMe, matching the plain-FNO job.
LOCAL_SCRATCH="${LOCAL_SCRATCH:-yes}"
if [ "$LOCAL_SCRATCH" = "yes" ]; then
    LOCAL_CUBES="$TMPDIR/cubes_3d.h5"
    echo "Staging cube cache to local scratch: $LOCAL_CUBES"
    t0=$(date +%s)
    cp "$PROJECT_CUBES" "$LOCAL_CUBES"
    echo "  staged ($(du -h "$LOCAL_CUBES" | cut -f1)) in $(( $(date +%s) - t0 ))s"
    export CUBES_CACHE="$LOCAL_CUBES"
else
    export CUBES_CACHE="${CUBES_CACHE:-$PROJECT_CUBES}"
fi

# --- Distributed rendezvous -------------------------------------------------
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- Diagnostics ------------------------------------------------------------
echo "================ SirenFNO training ==============="
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "Tasks (ranks):      ${SLURM_NTASKS:-1}"
echo "CPUs/task:          ${SLURM_CPUS_PER_TASK:-1}"
echo "MODEL_KIND:         ${MODEL_KIND}"
echo "CHECKPOINT_DIR:     ${CHECKPOINT_DIR}"
echo "INIT_CHECKPOINT:    ${INIT_CHECKPOINT:-fresh initialization}"
echo "N_MODES:            (${N_MODES_X}, ${N_MODES_Y}, ${N_MODES_Z})"
echo "SIREN_HIDDEN_DIM:   ${SIREN_HIDDEN_DIM}"
echo "SIREN_OMEGA:        ${SIREN_OMEGA}"
echo "SIREN_N_HIDDEN:     ${SIREN_N_HIDDEN}"
echo "SIREN_FEATURE_DIM:  ${SIREN_FEATURE_DIM}"
echo "SIREN_FF_SIGMA:     ${SIREN_FF_SIGMA}"
echo "SIREN_PADDING:      (${SIREN_PADDING_X}, ${SIREN_PADDING_Y}, ${SIREN_PADDING_Z})"
echo "OUTPUT_SIGMOID:     ${SIREN_OUTPUT_SIGMOID}"
echo "SIGMOID_TEMP:       ${SIREN_SIGMOID_TEMPERATURE}"
echo "BASE_LR:            ${SIRENFNO_LEARNING_RATE}"
echo "H1_WARMUP_EPOCHS:   ${SIRENFNO_H1_WARMUP_EPOCHS}"
echo "GRAD_CLIP_NORM:     ${SIRENFNO_GRAD_CLIP_NORM}"
echo "N_EPOCHS:           ${N_EPOCHS}"
echo "CUBES_CACHE:        ${CUBES_CACHE}"
echo "MASTER_ADDR:        ${MASTER_ADDR}"
echo "MASTER_PORT:        ${MASTER_PORT}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "==================================================="

# --- Launch -----------------------------------------------------------------
srun python fno_21cm_3d.py
