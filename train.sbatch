#!/bin/bash
# =============================================================================
# SLURM batch script — train the 2-D FNO on 21cm lightcone slices (1 GPU).
#
# Submit from the project directory (the one containing fno_21cm.py AND the
# data/ folder of lightcone .h5 files):
#
#     sbatch train.sbatch
#
# Edit the lines marked  # TODO  for your cluster before the first run.
# Discover the values with:
#     sinfo -s                                   # partition names + state
#     sinfo -o "%P %G"                           # partitions and their GPU gres
#     sacctmgr -n show assoc user=$USER \
#              format=account,partition          # account(s) you may use
#     conda env list                             # env names on the server
# =============================================================================

# NOTE: do NOT add trailing "# ..." comments on #SBATCH lines — some SLURM
# versions parse them as part of the option value and reject the job.
#
#   partition  : TODO set to your GPU partition (see `sinfo -s`)
#   account    : TODO uncomment the line below if your cluster requires one
#   cpus       : cores used for data interpolation / loading
#   gres=gpu:1 : 1 GPU — the training code is single-GPU (no DDP)
#   mem        : generous; the model is small — lower if you are quota-limited
#   time       : walltime HH:MM:SS — raise if you increase N_EPOCHS
#   %j         : expands to the job id in the log filenames
#
#SBATCH --job-name=fno-21cm
#SBATCH --partition=gpu
##SBATCH --account=my_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fno-%j.out
#SBATCH --error=logs/fno-%j.err

set -eo pipefail   # (-u omitted: conda's activate scripts trip on unbound vars)

# --- Run from the submission directory --------------------------------------
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# --- Activate the conda environment -----------------------------------------
# Robust activation in a non-interactive shell. If `conda` is not on PATH in
# batch jobs on your cluster, either load it first (e.g. `module load anaconda3`)
# or source it by absolute path, e.g.:
#     source "$HOME/miniforge3/etc/profile.d/conda.sh"
ENV_NAME="FNO_env"                   # TODO: change if the server env has another name
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- Thread hygiene ---------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --- Diagnostics (captured in the .out log) ---------------------------------
echo "================ job environment ================"
echo "Host:               $(hostname)"
echo "Job ID:             ${SLURM_JOB_ID:-N/A}"
echo "Submit dir:         ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Conda env:          ${CONDA_DEFAULT_ENV:-?}"
echo "Python:             $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || echo "(nvidia-smi unavailable)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE'))"
echo "================================================="

# --- Train ------------------------------------------------------------------
srun python fno_21cm.py
