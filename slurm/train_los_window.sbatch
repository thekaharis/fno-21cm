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
# Budget: uses the 2000-cone subset preparation (1600 train / 200 val / 200 test).
#
# Per-epoch cost was measured (jobs 4952130/4952131, 5096595) and has two
# different bottlenecks:
#   training     GPU compute, ~0.6 s fwd+bwd per 140x140x256 window. Raw-file
#                loading (1.3-2.1 s/sample) runs on DataLoader workers and hides
#                behind it.
#   validation   I/O, serial: every validation cone is walked through ~15
#                overlapping windows at 12-15 s of reads per cone, because the
#                raw fields are gzip-compressed in 9x9x307 chunks and a single
#                window read costs about as much as reading the whole field.
# At 4 windows/cone and full 200-cone validation that came to 2.4-2.7 h/epoch
# (~60% of it validation), and both 48 h runs were killed before finishing.
#
# Defaults (2026-09-22): WINDOWS_PER_CONE=4, VAL_CONES=40, EPOCHS=30, 48 h wall.
# 1 window/cone (run los-coarse-r3) made epochs 4.4x cheaper but learned ~4x
# less per epoch -- no better per GPU hour -- so 4 windows is back; the 40-cone
# validation subset is kept because it removed ~1.5 h/epoch of pure I/O.
#
# Cost is dominated by the model's forward pass (train ~2.9x forward per 256-cell
# window; validation and the final 400-cone pass are ~31 s/cone of raw-file I/O
# plus 15 forwards). Projected 30-epoch totals incl. the final pass:
#   cnn/whno, cnn/swhno (116 ms fwd)              ~33 h   fits 48 h
#   cnn/fno bw48, cnn/sfno bw48om60 (~317 ms)     ~65 h   does NOT fit
#   sfno/swhno bw48om60 (458 ms)                  ~87 h   does NOT fit
# Set --time per model accordingly; a run killed mid-schedule never anneals.
#
# AUGMENT=transverse applies a random transverse periodic shift, 90-degree
# rotation and reflection to each training window (exact symmetries of the
# periodic box faces; LOS untouched) and appends _augtransverse to the run dir.
#
# Select the model with MODEL_TAG (reads experiments/los_windows/model_<tag>.json).
#
# Multi-field (density,los_velocity -> neutral_fraction,brightness_temp) uses the
# chunked native mirror (data/native_mirror_4f, 1992 cones, 140x140x64 chunks;
# window reads ~20x faster than the raw gzip files):
#   sbatch --export=ALL,SAMPLING=contiguous,MODEL_TAG=cnn_whno,FIELDSET=multifield \
#          slurm/train_los_window.sbatch
# FIELDSET=multifield sets PREPARATION, INPUTS, TARGETS and the run root; each can
# still be overridden. INPUTS/TARGETS are comma lists, so pass them through an
# exported variable rather than --export (which splits on commas).
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
FIELDSET="${FIELDSET:-xhi}"
case "$FIELDSET" in
    xhi)        PREPARATION="${PREPARATION:-experiments/los_windows/preparation_xhi_2000.json}"
                INPUTS="${INPUTS:-density}"; TARGETS="${TARGETS:-neutral_fraction}"
                RUN_ROOT="${RUN_ROOT:-checkpoints/3d_xhi/los_windows}" ;;
    multifield) PREPARATION="${PREPARATION:-experiments/los_windows/preparation_multifield_2000.json}"
                INPUTS="${INPUTS:-density,los_velocity}"; TARGETS="${TARGETS:-neutral_fraction,brightness_temp}"
                RUN_ROOT="${RUN_ROOT:-checkpoints/3d_xhi/mf_los_windows}" ;;
    *) echo "unknown FIELDSET=$FIELDSET (xhi|multifield)" >&2; exit 1 ;;
esac
MODEL_TAG="${MODEL_TAG:-cnn_fno}"
MODEL_JSON="${MODEL_JSON:-experiments/los_windows/model_${MODEL_TAG}.json}"
[[ -f "$MODEL_JSON" ]] || { echo "missing model settings: $MODEL_JSON" >&2; exit 1; }
AUGMENT="${AUGMENT:-none}"
AUG_SUFFIX=""; [[ "$AUGMENT" != none ]] && AUG_SUFFIX="_aug${AUGMENT}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${SAMPLING}_${MODEL_TAG}_2000_w${WINDOWS_PER_CONE:-4}e${EPOCHS:-30}_s${SEED:-0}${AUG_SUFFIX}}"

extra=()
if [[ "$SAMPLING" == "coarse_context" ]]; then
    extra=(--context-factor "${CONTEXT_FACTOR:-4}"
           --context-xy "${CONTEXT_XY:-4}"
           --context-features "${CONTEXT_FEATURES:-8}")
fi

echo "sampling:    $SAMPLING"
echo "preparation: $PREPARATION"
echo "fields:      $INPUTS -> $TARGETS"
echo "augment:     $AUGMENT"
echo "model:       $MODEL_JSON"
echo "run dir:     $RUN_DIR"
echo "started:     $(date)"

srun python fno_multifield.py train \
    --preparation "$PREPARATION" \
    --inputs "$INPUTS" --targets "$TARGETS" \
    --model-settings "$(cat "$MODEL_JSON")" \
    --sampling "$SAMPLING" \
    --window-size "${WINDOW_SIZE:-256}" --window-halo "${WINDOW_HALO:-32}" \
    --windows-per-cone "${WINDOWS_PER_CONE:-4}" \
    --augment "$AUGMENT" \
    --val-cones "${VAL_CONES:-40}" \
    "${extra[@]}" \
    --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-1}" \
    --lr "${LR:-3e-4}" --weight-decay "${WEIGHT_DECAY:-1e-5}" \
    --seed "${SEED:-0}" --workers "${WORKERS:-4}" --device cuda \
    --run-dir "$RUN_DIR"
echo "finished: $(date)"
