#!/bin/bash
# Parameters vs inference throughput for every operator pairing.
#
# A30 is not a preference, it is a requirement for comparability: the recorded
# rows carry `device`, and throughput across GPU generations is not comparable,
# so re-running on anything else silently invalidates every existing point.
#
#   sbatch slurm/bench_operator_variants.sbatch
#
#SBATCH --job-name=bench-operators
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench-operators-%j.out
#SBATCH --error=logs/bench-operators-%j.err

set -eo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then PROJECT_ROOT="$SLURM_SUBMIT_DIR";
else PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"; fi
cd "$PROJECT_ROOT"
mkdir -p logs figures

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "device: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo unknown)"
python legacy/probes/bench_operator_variants.py \
    --json "${JSON_OUT:-figures/summary/operator_variant_benchmark.json}" \
    ${BENCH_ARGS:-}
echo "benchmark complete"
