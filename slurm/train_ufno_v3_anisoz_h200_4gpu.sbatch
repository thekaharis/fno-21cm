#!/bin/bash
# =============================================================================
# SLURM batch script -- 4-GPU H200 DDP training of U-FNO with the Tier 2
# **anisotropic Z U-Net** variant (option D).
#
# Submit from the project root:
#
#     sbatch slurm/train_ufno_v3_anisoz_h200_4gpu.sbatch
#
# What this changes vs v2 (the A+B+C bundle):
#   * UFNO_UNET_VARIANT=anisotropic_z -- replaces the upstream U_net inside
#     each of the 3 U-Fourier blocks with AnisotropicZUNet, which uses
#     stride=(2, 2, 4) on the outermost down/up sample stages.  Doubles
#     the LOS receptive field at the bottleneck (~16 cells ≈ 2.3 redshift
#     units) without adding parameters or layers.  Targets the cone-61-
#     style high-z artefact root cause directly.
#   * Padding constraint along Z tightens from "multiple of 8" to
#     "multiple of 16" -- production cube Z=256 -> padded to 272.
#
# Inherits from v2: GroupNorm in U-Net path (UFNO_NORM=groupnorm),
# H1-weighted loss (0.3*L2 + 0.7*H1), asymmetric Z modes (N_MODES_Z=32).
# Set those back to v1 defaults via env vars if you want to isolate the
# anisotropic-Z effect from the v2 changes -- but the cleaner ablation is
# v1 -> v3-anisoz comparison, holding everything except UFNO_UNET_VARIANT.
#
# Distinct CHECKPOINT_DIR (./checkpoints_3d_ufno_v3_anisoz) so this run
# doesn't collide with v1, v2, or the other Tier 2 variants.
# =============================================================================

#SBATCH --job-name=ufno-v3-anisoz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=18:00:00
#SBATCH --output=logs/ufno-v3-anisoz-%j.out
#SBATCH --error=logs/ufno-v3-anisoz-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

module load devel/miniforge
ENV_NAME="fno-env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# --- Model selection --------------------------------------------------------
export MODEL_KIND=ufno
export N_EPOCHS=30

# --- v3 (Tier 2) Option D: anisotropic Z U-Net -----------------------------
export UFNO_UNET_VARIANT=anisotropic_z

# --- v2 baseline overrides (carried forward) -------------------------------
export N_MODES_Z=32
export UFNO_NORM=groupnorm
export LOSS_L2_WEIGHT=0.3
export LOSS_H1_WEIGHT=0.7
export LOSS_BCE_WEIGHT=0.0

# bf16 mixed precision (~1.5-2x speedup on H200 tensor cores; bf16 keeps
# fp32 exponent range -> no GradScaler).
export AUTOCAST=bf16

export CHECKPOINT_DIR=./checkpoints_3d_ufno_v3_anisoz

# --- Data paths + NVMe staging ---------------------------------------------
export LIGHTCONE_DIR=/pfs/10/work/hd_id260-fno_training/data/data
PROJECT_CUBES=/pfs/10/work/hd_id260-fno_training/data/data/cubes_3d.h5

LOCAL_SCRATCH="${LOCAL_SCRATCH:-yes}"
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
        cp --reflink=auto "$PROJECT_CUBES" "$LOCAL_CUBES" || cp "$PROJECT_CUBES" "$LOCAL_CUBES"
        echo "  staged ($(du -h "$LOCAL_CUBES" | cut -f1)) in $(( $(date +%s) - t0 ))s"
        export CUBES_CACHE="$LOCAL_CUBES"
    else
        echo "No local scratch detected; reading cubes from project FS."
        export CUBES_CACHE="$PROJECT_CUBES"
    fi
else
    echo "LOCAL_SCRATCH=no: reading cubes from project FS."
    export CUBES_CACHE="$PROJECT_CUBES"
fi

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "MODEL_KIND:         ${MODEL_KIND}"
echo "N_EPOCHS:           ${N_EPOCHS}"
echo "N_MODES_Z:          ${N_MODES_Z}"
echo "UFNO_NORM:          ${UFNO_NORM}"
echo "UFNO_UNET_VARIANT:  ${UFNO_UNET_VARIANT}"
echo "LOSS L2 / H1 / BCE: ${LOSS_L2_WEIGHT} / ${LOSS_H1_WEIGHT} / ${LOSS_BCE_WEIGHT}"
echo "AUTOCAST:           ${AUTOCAST:-off}"
echo "CHECKPOINT_DIR:     ${CHECKPOINT_DIR}"
nvidia-smi -L || echo "(nvidia-smi unavailable)"
echo "================================================="

srun python fno_21cm_3d.py
