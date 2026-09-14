#!/bin/bash
# =============================================================================
# Build the full four-field multi-field cache from the raw 21cmFAST lightcones.
#
# CPU only and I/O bound; gzip compression is the main CPU cost. Single process,
# because fno_multifield.py has no shard/merge path -- a hand-rolled merge on a
# ~400 GiB artifact is more risk than the wall time saves.
#
#   sbatch slurm/build_multifield_cache.sbatch
#
# Configuration is the one agreed in notes/multifield-data-configuration.md and
# validated by the pilot (experiments/multifield/pilot_report.md): four plain
# fields, 256 LOS points over z = 5.001 .. 24.97. The endpoints matter -- a
# 5..25 grid silently zero-fills the 10 simulations whose native axis stops
# below 25.
# =============================================================================

#SBATCH --job-name=build-multifield
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=logs/build-multifield-%j.out
#SBATCH --error=logs/build-multifield-%j.err

set -eo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

RAW_LIGHTCONES="${RAW_LIGHTCONES:-/pfs/10/work/hd_id260-fno_training/data/data}"
MULTIFIELD_CACHE="${MULTIFIELD_CACHE:-/pfs/10/work/hd_id260-fno_training/data/compressed/multifield_z256.h5}"
FIELDS="${FIELDS:-density,neutral_fraction,brightness_temp,los_velocity}"
N_Z="${N_Z:-256}"
Z_MIN="${Z_MIN:-5.001}"
Z_MAX="${Z_MAX:-24.97}"

module load devel/miniforge >/dev/null 2>&1 || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env

echo "================ multi-field cache build ================"
echo "Host:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-none}"
echo "Raw:       $RAW_LIGHTCONES"
echo "Out:       $MULTIFIELD_CACHE"
echo "Fields:    $FIELDS"
echo "LOS grid:  $N_Z points, z = $Z_MIN .. $Z_MAX"
echo "Started:   $(date)"
echo "========================================================="

# Refuse to clobber an existing cache; a partial overwrite of this artifact is
# far more expensive than a failed submission.
if [[ -e "$MULTIFIELD_CACHE" ]]; then
    echo "ERROR: $MULTIFIELD_CACHE already exists; move it aside first." >&2
    exit 1
fi

srun python fno_multifield.py cache \
    --data "$RAW_LIGHTCONES" \
    --fields "$FIELDS" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" --n-z "$N_Z" \
    --conditioning z_params \
    --out "$MULTIFIELD_CACHE"

echo "Finished:  $(date)"
ls -la "$MULTIFIELD_CACHE"
