#!/bin/bash
# Global reionization history emulator (tools_global_history_emulator.py):
# collect -> train (5-member monotone MLP ensemble) -> evaluate -> relevel.
#SBATCH --job-name=gh-emu
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gh-emu-%j.out
#SBATCH --error=logs/gh-emu-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
python -u tools_global_history_emulator.py ${STEPS:-train evaluate relevel} \
    --workers 16 --members "${MEMBERS:-5}" --epochs "${EPOCHS:-400}"
