#!/bin/bash
# Multi-field native-LOS preparation over the rechunked mirror (4 fields).
#   sbatch slurm/prepare_multifield_windows.sbatch
#SBATCH --job-name=prep-mf-win
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/prep-mf-win-%j.out
#SBATCH --error=logs/prep-mf-win-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
MIRROR="${MIRROR:-/pfs/10/work/hd_id260-fno_training/data/native_mirror_4f}"
OUT="${OUT:-experiments/los_windows/preparation_multifield_2000.json}"
[[ -e "$OUT" ]] && { echo "ERROR: $OUT exists" >&2; exit 1; }
echo "mirror: $MIRROR   out: $OUT   started: $(date)"
srun python fno_multifield.py prepare \
    --data "$MIRROR" --glob '*.h5' --native-los \
    --fields density,los_velocity,neutral_fraction,brightness_temp \
    --conditioning z_params --split-seed 42 \
    --out "$OUT"
echo "finished: $(date)"; ls -la "$OUT"
