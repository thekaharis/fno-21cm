#!/bin/bash
# Native-LOS preparation on a 2000-cone SUBSET, for feasible windowed training.
#
# Full ensemble: 5280 train cones x 8 draws = 42,240 batches/epoch ~= 6.7 h/epoch,
# i.e. 134 h for 20 epochs. Not viable. This subset gives 1600 x 4 = 6400
# batches/epoch ~= 1 h/epoch.
#
# Cones over draws: notes/los-window-sampling.md states that more draws give more
# coverage of the SAME simulations, not more independent ones. At equal cost,
# 2000 cones x 4 draws therefore carries more statistical power than 1000 x 8.
#
# Population: samples 000000-001999, contiguous from the start of the ensemble.
# Cone IDs follow the sorted file list, so they equal sample IDs here.
#   sbatch slurm/prepare_los_windows_subset.sbatch
#SBATCH --job-name=prep-los-sub
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prep-los-sub-%j.out
#SBATCH --error=logs/prep-los-sub-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
RAW_LIGHTCONES="${RAW_LIGHTCONES:-/pfs/10/work/hd_id260-fno_training/data/data}"
OUT="${OUT:-experiments/los_windows/preparation_xhi_2000.json}"
[[ -e "$OUT" ]] && { echo "ERROR: $OUT exists" >&2; exit 1; }
echo "raw: $RAW_LIGHTCONES   out: $OUT   started: $(date)"
srun python fno_multifield.py prepare \
    --data "$RAW_LIGHTCONES" --glob '21cmfast_11d_sample00[01]*.h5' --native-los \
    --fields density,neutral_fraction \
    --conditioning z_params --split-seed 42 \
    --out "$OUT"
echo "finished: $(date)"; ls -la "$OUT"
