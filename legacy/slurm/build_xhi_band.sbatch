#!/bin/bash
# Build an x_HI-band slice cache for the late-reionization analysis.
# CPU only -- reads two 2-D planes per selected LOS index, no model involved.

#SBATCH --job-name=build-xhi-band
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build-xhi-band-%j.out
#SBATCH --error=logs/build-xhi-band-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

RUN="${RUN:-checkpoints/2d_xhi/fno_whno/xhi2d_whno_glob_lr3e4}"

python -m dataset.build_xhi_band \
    --lo "${BAND_LO:-0.0}" --hi "${BAND_HI:-0.36}" \
    --k "${K_PER_CONE:-12}" \
    --split-from "$RUN" --splits "${SPLITS:-val,test}" \
    --out "${OUT:-xhi_band_000_036.h5}"
