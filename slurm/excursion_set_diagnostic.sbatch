#!/bin/bash
# Sharp-k excursion-set diagnostic on the 12 contiguous_ep18 test cones
# (tools_excursion_set_diagnostic.py). CPU only; ~2.5 s per slab per worker.
#SBATCH --job-name=es-diag
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/es-diag-%j.out
#SBATCH --error=logs/es-diag-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
python -u tools_excursion_set_diagnostic.py --workers 12
