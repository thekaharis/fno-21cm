#!/bin/bash
# =============================================================================
# Standard post-training evaluation for one multi-field checkpoint:
#   1. held-out test metrics (with transverse spectra)
#   2. predictions on a fixed set of test cones
#   3. detailed per-field pages, slice comparisons (transverse + LOS),
#      lightcone strips (full range + active band)
#   4. inference throughput
# Works for both full-grid and native-LOS window checkpoints; the viz draws on
# true redshift cell edges, so native per-cone axes render correctly.
#
#   sbatch --export=ALL,CHECKPOINT=x.pt,TAG=name,CONES="27 1892 1625 1684" \
#          slurm/eval_multifield_run.sbatch
# NOTE must not contain commas: sbatch --export splits on them and silently
# truncates the value. SKIP_SPEED=1 skips step 4 (e.g. when only adding cones).
# =============================================================================
#SBATCH --job-name=eval-mf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval-mf-%j.out
#SBATCH --error=logs/eval-mf-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
TAG="${TAG:?set TAG (output folder name)}"
CONES="${CONES:?set CONES}"
STUDY="${STUDY:-los_windows}"
NOTE="${NOTE:-}"
FIG="figures/3d_xhi/$STUDY/$TAG"
PRED="experiments/$STUDY/predictions/$TAG"
METRICS="experiments/$STUDY/${TAG}_test_metrics.json"
mkdir -p "$FIG" "$PRED"
echo "checkpoint: $CHECKPOINT"; echo "tag: $TAG"; echo "cones: $CONES"; date

echo "== [1] test metrics"
if [[ -e "$METRICS" ]]; then echo "exists, skipping: $METRICS"
else python fno_multifield.py evaluate --checkpoint "$CHECKPOINT" --split test \
         --spectral-bins 12 --out "$METRICS"; fi
date

echo "== [2] predictions"
FILES=""
for c in $CONES; do
    f="$PRED/cone_$c.h5"
    [[ -e "$f" ]] || python fno_multifield.py predict --checkpoint "$CHECKPOINT" --cone-id "$c" --out "$f"
    FILES="$FILES $f"
done
date

echo "== [3] figures"
python -m viz.multifield_detailed --prediction $FILES --out-dir "$FIG/detailed" --note "$NOTE"
python -m viz.multifield_slices   --prediction $FILES --out-dir "$FIG/slices" --n-slices 5
python -m viz.multifield_slices   --prediction $FILES --out-dir "$FIG/slices" --orientation los --n-slices 4
python -m viz.multifield_lightcone_strips --prediction $FILES --out-dir "$FIG/strips" --note "$NOTE"
python -m viz.multifield_lightcone_strips --prediction $FILES --out-dir "$FIG/strips" --crop --note "$NOTE"
date

echo "== [4] inference speed"
if [[ "${SKIP_SPEED:-0}" == "1" ]]; then echo "skipped"; exit 0; fi
python -m viz.multifield_inference_speed --checkpoints "$TAG=$CHECKPOINT" \
       --split test --n-samples 4 --warmup 2 --iters 8 --out "$FIG/speed"
date
echo "done"
