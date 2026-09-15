#!/bin/bash
# Boundary-band edge metrics for 2-D x_HI runs (the 3-D edge_metrics_eval.sbatch
# is hardwired to the cube cache and cannot score a slice model).
#   sbatch --export=ALL,RUNS="a=checkpoints/x b=checkpoints/y" slurm/edge_metrics_xhi2d.sbatch
#SBATCH --job-name=edge-xhi2d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/edge-xhi2d-%j.out
#SBATCH --error=logs/edge-xhi2d-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
RUNS="${RUNS:?set RUNS to a space-separated list of LABEL=CHECKPOINT_DIR}"
read -r -a run_array <<< "$RUNS"
srun python -m viz.edge_metrics_xhi2d --runs "${run_array[@]}" \
    --n-slices "${N_SLICES:-600}" --out "${OUT:-figures/2d_xhi/eval/edge_metrics}"
