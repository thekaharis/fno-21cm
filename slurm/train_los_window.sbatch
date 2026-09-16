#!/bin/bash
# =============================================================================
# Windowed native-LOS x_HI training: density -> neutral_fraction, cnn/fourier.
#
# Both variants share ONE native preparation (notes/los-window-sampling.md);
# SAMPLING selects between them:
#   contiguous      256 consecutive native LOS cells, 32-cell halos excluded
#                   from the loss, 8 random draws per training cone per epoch
#   coarse_context  same fine window plus a 4x-decimated 1024-slice surrounding
#                   region through a small trainable encoder
#
# Budget: the full ensemble costs ~6.7 h/epoch (5280 train cones x 8 draws),
# i.e. 134 h for 20 epochs -- not viable. This uses the 2000-cone subset
# preparation: 1600 train x 4 draws = 6400 batches/epoch, ~1 h/epoch. Cones were
# preferred over draws because extra draws add coverage of the same simulations,
# not independent ones. The 48 h wall leaves ~2.4x headroom over the ~20 h
# estimate, so the cosine schedule anneals inside the budget rather than being
# truncated mid-schedule.
#
#   sbatch --export=ALL,SAMPLING=contiguous slurm/train_los_window.sbatch
#   sbatch --export=ALL,SAMPLING=coarse_context slurm/train_los_window.sbatch
# =============================================================================
#SBATCH --job-name=los-window
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/los-window-%j.out
#SBATCH --error=logs/los-window-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

SAMPLING="${SAMPLING:?set SAMPLING=contiguous|coarse_context}"
PREPARATION="${PREPARATION:-experiments/los_windows/preparation_xhi_2000.json}"
RUN_DIR="${RUN_DIR:-checkpoints/3d_xhi/los_windows/${SAMPLING}_cnn_fno_2000_s0}"

extra=()
if [[ "$SAMPLING" == "coarse_context" ]]; then
    extra=(--context-factor "${CONTEXT_FACTOR:-4}"
           --context-xy "${CONTEXT_XY:-4}"
           --context-features "${CONTEXT_FEATURES:-8}")
fi

echo "sampling:    $SAMPLING"
echo "preparation: $PREPARATION"
echo "run dir:     $RUN_DIR"
echo "started:     $(date)"

srun python fno_multifield.py train \
    --preparation "$PREPARATION" \
    --inputs density --targets neutral_fraction \
    --model-settings "$(cat experiments/los_windows/model_cnn_fno.json)" \
    --sampling "$SAMPLING" \
    --window-size "${WINDOW_SIZE:-256}" --window-halo "${WINDOW_HALO:-32}" \
    --windows-per-cone "${WINDOWS_PER_CONE:-4}" \
    "${extra[@]}" \
    --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-1}" \
    --lr "${LR:-3e-4}" --weight-decay "${WEIGHT_DECAY:-1e-5}" \
    --seed "${SEED:-0}" --workers "${WORKERS:-4}" --device cuda \
    --run-dir "$RUN_DIR"
echo "finished: $(date)"
