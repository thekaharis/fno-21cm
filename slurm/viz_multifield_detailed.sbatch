#!/bin/bash
# Detailed per-field diagnostics for a multi-field checkpoint.
# Runs on a GPU node: 3-D inference over 140x140x256 is impractical on a login node.
#
#   sbatch --export=ALL,CHECKPOINT=path.pt,CONES="541 444 47 67" \
#          slurm/viz_multifield_detailed.sbatch
#SBATCH --job-name=viz-multifield
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/viz-multifield-%j.out
#SBATCH --error=logs/viz-multifield-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
CONES="${CONES:-541 444 47 67}"
PRED_DIR="${PRED_DIR:-experiments/multifield/predictions}"
OUT_DIR="${OUT_DIR:-figures/3d_xhi/mf_cnn_fno/detailed}"
NOTE="${NOTE:-}"
mkdir -p "$PRED_DIR" "$OUT_DIR"

echo "checkpoint: $CHECKPOINT"
echo "cones:      $CONES"
for c in $CONES; do
    if [[ -e "$PRED_DIR/cone_$c.h5" ]]; then
        echo "  cone $c: prediction exists, reusing"
    else
        echo "  cone $c: predicting"
        python fno_multifield.py predict --checkpoint "$CHECKPOINT" \
               --cone-id "$c" --out "$PRED_DIR/cone_$c.h5"
    fi
done

FILES=""
for c in $CONES; do FILES="$FILES $PRED_DIR/cone_$c.h5"; done
python -m viz.multifield_detailed --prediction $FILES --out-dir "$OUT_DIR" --note "$NOTE"
echo "---- figures ----"
ls -la "$OUT_DIR"
