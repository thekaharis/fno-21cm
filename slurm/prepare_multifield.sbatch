#!/bin/bash
# Preparation artifact for the four-field 256-slice cache: splits by simulation
# ID (seed 42), source fingerprints, conditioning, and TRAIN-ONLY normalization.
# One artifact is reused by every mapping and training seed.
#   sbatch slurm/prepare_multifield.sbatch
#SBATCH --job-name=prep-multifield
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prep-multifield-%j.out
#SBATCH --error=logs/prep-multifield-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
CACHE="${CACHE:-/pfs/10/work/hd_id260-fno_training/data/compressed/multifield_z256.h5}"
OUT="${OUT:-experiments/multifield/preparation.json}"
[[ -e "$OUT" ]] && { echo "ERROR: $OUT exists" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
echo "cache: $CACHE"; echo "out:   $OUT"; echo "started: $(date)"
srun python fno_multifield.py prepare --cache "$CACHE" \
    --fields density,neutral_fraction,brightness_temp,los_velocity \
    --conditioning z_params --split-seed 42 --out "$OUT"
echo "finished: $(date)"; ls -la "$OUT"
