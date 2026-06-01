#!/bin/bash
# =============================================================================
# SLURM batch script — train the 2-D FNO on 21cm lightcone slices (1 GPU).
#
# Submit from the project directory (the one containing fno_21cm.py AND the
# data/ folder of lightcone .h5 files):
#
#     sbatch train.sbatch
#
# Before the first run, confirm ENV_NAME (below) matches the server env:
#     conda env list
# =============================================================================

# NOTE: do NOT add trailing "# ..." comments on #SBATCH lines — some SLURM
# versions parse them as part of the option value and reject the job.
#
# GPU allocation on this cluster (per the `gpu` queue docs):
#   - nodes hold 2 or 4 NVIDIA A30 / A100 / H200 GPUs
#   - you MUST request GPUs explicitly with a TYPED gres: gpu:<type>:<count>
#         gpu:a30:1   -> one A30      gpu:a100:1 -> one A100   gpu:h200:1 -> one H200
#   - this job is single-process / single-GPU (no DDP), so it requests ONE.
#     An A30 is plenty for a 21cm FNO; leave the A100/H200 for bigger jobs.
#     To use a different type, change `a30` below (or try `--gres=gpu:1` for any).
#
#   ntasks=1     : ONE process. Do NOT raise this — the training script is a
#                  single Python process; a higher --ntasks under srun would
#                  launch duplicate trainers fighting over the GPU/checkpoints.
#                  (On this cluster --ntasks is the MPI core count; not for us.)
#   cpus-per-task: cores for data interpolation / loading
#   mem          : the model is small; raise only if preload OOMs
#   time         : walltime HH:MM:SS — raise if you increase N_EPOCHS
#   %j           : expands to the job id in the log filenames
#
#SBATCH --job-name=fno-21cm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=16G
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
# Plain `python` (no `srun`): this is a single-process job, and launching it
# under srun would spawn one trainer per task if --ntasks is ever raised.
python fno_21cm.py
