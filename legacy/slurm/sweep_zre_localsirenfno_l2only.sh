#!/bin/bash
# =============================================================================
# Hyperparameter sweep driver — z_re LocalSirenFNO, L2-only loss (A100).
#
# Stage 1: one-factor-at-a-time variations around the baseline run
# (slurm/train_zre_localsirenfno_a100_l2only.sbatch, checkpoint dir
# checkpoints/checkpoints_zre_localsirenfno_l2only). Each variation reuses
# that sbatch script with env overrides, so every run shares: MODEL_KIND=
# localsirenfno, 1.0*absL2, N_EPOCHS=200, BATCH_SIZE=8, RUN_SEED=0, and the
# same train/val/test split (SPLIT_SEED=42).
#
# Usage (from the project root):
#     bash slurm/sweep_zre_localsirenfno_l2only.sh            # submit all
#     bash slurm/sweep_zre_localsirenfno_l2only.sh lr2e4 om60 # submit subset
#
# Re-running resubmits jobs (SBATCH refuses nothing); the script skips a tag
# whose checkpoint dir already has a metrics.jsonl unless FORCE=1 is set.
#
# Stage 2 (submitted 2026-07-21, jobs 4323085-4323087 + 4323097-4323098,
# 300 epochs, 6 h):
#   bw32lr2e4      LOCALFNO_BASE_WIDTH=32 LEARNING_RATE=2e-4  (stage-1 combo)
#   bw48lr2e4      LOCALFNO_BASE_WIDTH=48 LEARNING_RATE=2e-4  (width trend probe)
#   bw32lr3e4      LOCALFNO_BASE_WIDTH=32 LEARNING_RATE=3e-4  (LR trend probe)
#   bw48om60lr2e4  LOCALFNO_BASE_WIDTH=48 SIREN_OMEGA=60 LEARNING_RATE=2e-4
#   bw32om60lr2e4  LOCALFNO_BASE_WIDTH=32 SIREN_OMEGA=60 LEARNING_RATE=2e-4
# (the om60 pair completes the width {32,48} x omega {30,60} grid at LR 2e-4)
# Compare min val_l2 across all checkpoint dirs when finished.
# =============================================================================
set -euo pipefail

SBATCH_SCRIPT="slurm/train_zre_localsirenfno_a100_l2only.sbatch"
CKPT_BASE="checkpoints/checkpoints_zre_localsirenfno_l2only"

# tag -> extra env overrides (comma-free values only)
declare -A VARIANTS=(
    [lr5e5]="LEARNING_RATE=5e-5"
    [lr2e4]="LEARNING_RATE=2e-4"
    [bw24]="LOCALFNO_BASE_WIDTH=24"
    [bw32]="LOCALFNO_BASE_WIDTH=32"
    [m88]="LOCALFNO_MODES_X=8,LOCALFNO_MODES_Y=8"
    [win32m1212]="LOCALFNO_WINDOW_X=32,LOCALFNO_WINDOW_Y=32,LOCALFNO_MODES_X=12,LOCALFNO_MODES_Y=12"
    [om15]="SIREN_OMEGA=15"
    [om60]="SIREN_OMEGA=60"
)

tags=("$@")
if [ "${#tags[@]}" -eq 0 ]; then
    tags=(lr5e5 lr2e4 bw24 bw32 m88 win32m1212 om15 om60)
fi

for tag in "${tags[@]}"; do
    if [ -z "${VARIANTS[$tag]:-}" ]; then
        echo "unknown variant tag: $tag" >&2
        exit 1
    fi
    ckpt="${CKPT_BASE}_${tag}"
    if [ -f "${ckpt}/metrics.jsonl" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "skip $tag (${ckpt}/metrics.jsonl exists; FORCE=1 to resubmit)"
        continue
    fi
    sbatch --job-name="zre-ls-${tag}" \
        --export="ALL,${VARIANTS[$tag]},CHECKPOINT_DIR=${ckpt}" \
        "$SBATCH_SCRIPT"
done
