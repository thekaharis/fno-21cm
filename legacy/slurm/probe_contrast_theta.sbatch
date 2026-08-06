#!/bin/bash
# Optimal contrast theta vs ionisation state. Analysis only -- no training.
#
# Env: RUN, CACHE, BINS, XHI_MAX, EXTRA (extra flags for the probe)

#SBATCH --job-name=ctr-theta
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ctr-theta-%j.out
#SBATCH --error=logs/ctr-theta-%j.err

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

python legacy/probes/probe_contrast_theta.py \
    --run "${RUN:-checkpoints/xhi2d_whno_glob_lr3e4}" \
    ${CACHE:+--cache "$CACHE"} \
    ${BINS:+--bins "$BINS"} \
    ${XHI_MAX:+--xhi-max "$XHI_MAX"} \
    ${EXTRA:-}
