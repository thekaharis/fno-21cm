#!/bin/bash
# Coherence r(k) vs small-scale power fidelity for 2-D x_HI checkpoints.
#
#   sbatch --export=ALL,RUNS="a=checkpoints/x b=checkpoints/y" slurm/viz_ps_coherence.sbatch
#
# Runs on a GPU node rather than the login node: SliceCache reads the whole
# trainset into RAM, which is antisocial on a shared login node and slow.
#SBATCH --job-name=ps-coherence
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ps-coherence-%j.out
#SBATCH --error=logs/ps-coherence-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
mkdir -p logs figures

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID:-local}"; mkdir -p "$MPLCONFIGDIR"

RUNS="${RUNS:?set RUNS to a space-separated list of LABEL=CHECKPOINT_DIR}"
read -r -a run_array <<< "$RUNS"
python -m viz.ps_coherence_highk --runs "${run_array[@]}" \
    --n-slices "${N_SLICES:-400}" --out "${OUT:-figures/summary/ps_coherence_highk.png}"
