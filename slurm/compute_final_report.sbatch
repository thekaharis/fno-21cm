#!/bin/bash
#   sbatch --export=ALL,RUN=checkpoints/..._eval_snapshot slurm/compute_final_report.sbatch
#SBATCH --job-name=final-report
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/final-report-%j.out
#SBATCH --error=logs/final-report-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
srun python -m viz.compute_final_report --run "${RUN:?set RUN}"
