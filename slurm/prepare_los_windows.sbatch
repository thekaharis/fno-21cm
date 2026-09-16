#!/bin/bash
# =============================================================================
# Native-LOS preparation for windowed x_HI training (density -> neutral_fraction).
#
# ONE artifact serves BOTH sampling modes: `contiguous` and `coarse_context` are
# train-time flags (notes/los-window-sampling.md, "Reuse the same native
# preparation for both sampling modes"). Do not prepare twice.
#
# No cache is written. --native-los reads the raw /lightcone/<field> arrays
# directly at full native coverage; it is incompatible with --n-z/--target-z/
# --z-min/--z-max/--cache and those combinations are rejected.
#
# Population: the FULL 6600 raw lightcones. The 33 simulations excluded from the
# four-field cache had non-finite voxels only in brightness_temp; density and
# neutral_fraction are clean in all 6600 (experiments/multifield/nonfinite_report.json).
# Cone IDs follow the sorted raw file list, so here they equal sample IDs.
#
#   sbatch slurm/prepare_los_windows.sbatch
# =============================================================================

#SBATCH --job-name=prep-los-win
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/prep-los-win-%j.out
#SBATCH --error=logs/prep-los-win-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

RAW_LIGHTCONES="${RAW_LIGHTCONES:-/pfs/10/work/hd_id260-fno_training/data/data}"
OUT="${OUT:-experiments/los_windows/preparation_xhi.json}"

if [[ -e "$OUT" ]]; then
    echo "ERROR: $OUT exists; move it aside first." >&2
    exit 1
fi
mkdir -p "$(dirname "$OUT")"

echo "================ native-LOS preparation ================"
echo "Host:    $(hostname)   Job: ${SLURM_JOB_ID:-none}"
echo "Raw:     $RAW_LIGHTCONES"
echo "Out:     $OUT"
echo "Fields:  density,neutral_fraction   split seed 42"
echo "Started: $(date)"
echo "========================================================"

srun python fno_multifield.py prepare \
    --data "$RAW_LIGHTCONES" --native-los \
    --fields density,neutral_fraction \
    --conditioning z_params --split-seed 42 \
    --out "$OUT"

echo "Finished: $(date)"
ls -la "$OUT"
