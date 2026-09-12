#!/bin/bash
# 4-GPU H200 DDP training for the LocalSirenFNO U-Net: the windowed Local-FNO
# with SIREN-generated spectral weights in every branch.
# Submit: sbatch slurm/train_localsirenfno_h200_4gpu.sbatch

#SBATCH --job-name=localsirenfno-3d-4gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/localsirenfno3d-4gpu-%j.out
#SBATCH --error=logs/localsirenfno3d-4gpu-%j.err

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

export MODEL_KIND=localsirenfno
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/3d_xhi/localsirenfno/checkpoints_3d_localsirenfno}"
export N_EPOCHS="${N_EPOCHS:-70}"

# Whole-volume bottleneck modes at 35x35x64.
export N_MODES_X="${N_MODES_X:-16}"
export N_MODES_Y="${N_MODES_Y:-16}"
export N_MODES_Z="${N_MODES_Z:-16}"

# Local Fourier U-Net defaults (identical geometry to the dense LocalFNO;
# only the spectral-weight parameterization differs).
export LOCALFNO_WINDOW_X="${LOCALFNO_WINDOW_X:-16}"
export LOCALFNO_WINDOW_Y="${LOCALFNO_WINDOW_Y:-16}"
export LOCALFNO_WINDOW_Z="${LOCALFNO_WINDOW_Z:-32}"
export LOCALFNO_MODES_X="${LOCALFNO_MODES_X:-6}"
export LOCALFNO_MODES_Y="${LOCALFNO_MODES_Y:-6}"
export LOCALFNO_MODES_Z="${LOCALFNO_MODES_Z:-12}"
export LOCALFNO_BASE_WIDTH="${LOCALFNO_BASE_WIDTH:-16}"
export LOCALFNO_SPECTRAL_RANK="${LOCALFNO_SPECTRAL_RANK:-16}"
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-128}"

# SIREN weight-network hyperparameters (shared by all six spectral branches).
export SIREN_HIDDEN_DIM="${SIREN_HIDDEN_DIM:-64}"
export SIREN_OMEGA="${SIREN_OMEGA:-30.0}"
export SIREN_N_HIDDEN="${SIREN_N_HIDDEN:-1}"
export SIREN_FEATURE_DIM="${SIREN_FEATURE_DIM:-16}"
export SIREN_FF_SIGMA="${SIREN_FF_SIGMA:-128.0}"
export SIREN_LEARNABLE_FF="${SIREN_LEARNABLE_FF:-1}"

# LocalSirenFNO reuses the LocalFNO stability schedule (the shared SIREN trunk
# can spike gradients much like the plain SirenFNO, so keep the clip on).
export LOCALFNO_LEARNING_RATE="${LOCALFNO_LEARNING_RATE:-1e-4}"
export LOCALFNO_H1_WARMUP_EPOCHS="${LOCALFNO_H1_WARMUP_EPOCHS:-5}"
export LOCALFNO_GRAD_CLIP_NORM="${LOCALFNO_GRAD_CLIP_NORM:-1.0}"
export LOSS_L2_WEIGHT="${LOSS_L2_WEIGHT:-0.5}"
export LOSS_H1_WEIGHT="${LOSS_H1_WEIGHT:-0.5}"
export LOSS_BCE_WEIGHT="${LOSS_BCE_WEIGHT:-0.0}"
export LOSS_IONIZED_WALL_WEIGHT="${LOSS_IONIZED_WALL_WEIGHT:-0.0}"
export IONIZED_WALL_KERNEL_SIZE="${IONIZED_WALL_KERNEL_SIZE:-7}"
export IONIZED_WALL_THRESHOLD="${IONIZED_WALL_THRESHOLD:-0.5}"

export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
PROJECT_CUBES="${PROJECT_CUBES:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"
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

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============== LocalSirenFNO training ============="
echo "MODEL_KIND:         ${MODEL_KIND}"
echo "CHECKPOINT_DIR:     ${CHECKPOINT_DIR}"
echo "GLOBAL_MODES:       (${N_MODES_X}, ${N_MODES_Y}, ${N_MODES_Z})"
echo "WINDOW:             (${LOCALFNO_WINDOW_X}, ${LOCALFNO_WINDOW_Y}, ${LOCALFNO_WINDOW_Z})"
echo "LOCAL_MODES:        (${LOCALFNO_MODES_X}, ${LOCALFNO_MODES_Y}, ${LOCALFNO_MODES_Z})"
echo "BASE_WIDTH:         ${LOCALFNO_BASE_WIDTH}"
echo "SPECTRAL_RANK:      ${LOCALFNO_SPECTRAL_RANK}"
echo "PATCH_CHUNK_SIZE:   ${LOCALFNO_PATCH_CHUNK_SIZE}"
echo "SIREN:              hidden=${SIREN_HIDDEN_DIM} n_hidden=${SIREN_N_HIDDEN} omega=${SIREN_OMEGA} ff=${SIREN_FEATURE_DIM}@${SIREN_FF_SIGMA}"
echo "N_EPOCHS:           ${N_EPOCHS}"
echo "CUBES_CACHE:        ${CUBES_CACHE}"
nvidia-smi -L || true
echo "===================================================="

srun python fno_21cm_3d.py
