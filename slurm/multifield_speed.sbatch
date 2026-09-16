#!/bin/bash
# Inference throughput for multi-field checkpoints. GPU node: a login-node
# measurement is CPU-only and not comparable with anything else in the campaign.
#   sbatch --export=ALL,CHECKPOINTS="name=path ..." slurm/multifield_speed.sbatch
#SBATCH --job-name=mf-speed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/mf-speed-%j.out
#SBATCH --error=logs/mf-speed-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
CHECKPOINTS="${CHECKPOINTS:?set CHECKPOINTS as name=path pairs}"
OUT="${OUT:-figures/3d_xhi/mf_cnn_fno/speed}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m viz.multifield_inference_speed --checkpoints $CHECKPOINTS \
       --split "${SPLIT:-test}" --n-samples "${N_SAMPLES:-4}" \
       --warmup "${WARMUP:-3}" --iters "${ITERS:-15}" --out "$OUT"
