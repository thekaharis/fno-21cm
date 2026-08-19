#!/bin/bash
# Peak GPU memory of the 3-D localop run, to decide the A100 question.
#SBATCH --job-name=probe-3d-mem
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=logs/probe-3d-mem-%j.out
#SBATCH --error=logs/probe-3d-mem-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

# the training configuration, verbatim
export MODEL_KIND=localop LOCAL_OPERATOR=fourier GLOBAL_OPERATOR=hadamard
export LOCALFNO_WINDOW_X=16 LOCALFNO_WINDOW_Y=16 LOCALFNO_WINDOW_Z=32
export LOCALFNO_MODES_X=6 LOCALFNO_MODES_Y=6 LOCALFNO_MODES_Z=12
export LOCALFNO_BASE_WIDTH=16 LOCALFNO_SPECTRAL_RANK=16
export LOCALFNO_PATCH_CHUNK_SIZE=128
export N_MODES_X=16 N_MODES_Y=16 N_MODES_Z=16
export HIDDEN_CHANNELS=32 N_LAYERS=4
export EXPWALL_SCALE=16.0 WALL_CAP=32

python legacy/probes/probe_3d_memory.py
