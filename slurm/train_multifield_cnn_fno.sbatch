#!/bin/bash
# Four-field training on the 256-slice cache: CNN local / Fourier global.
# Proof of concept for the multi-field pipeline, not a tuned comparison.
# cnn/fno was one of the two best cells in the 3-D operator matrix.
#
# Mapping: {density, los_velocity} -> {neutral_fraction, brightness_temp}.
# Disjoint and uses all four cached fields. Two caveats recorded in
# experiments/multifield/pilot_report.md apply to the brightness_temp target:
#   * at 256 LOS points it retains ~51% of its variance (round-trip rel L2 0.49),
#     so achievable error on it is floored by the cache, not the model;
#   * at z <~ 8 it is close to an algebraic function of x_HI, density and dv/dr
#     (R^2 up to 0.9985), so low-z skill on it is near-trivial.
# Report brightness_temp metrics by redshift band, against the algebraic baseline.
#
#   sbatch --dependency=afterok:<prep> slurm/train_multifield_fno_cnn.sbatch
#SBATCH --job-name=mf-cnn-fno
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mf-cnn-fno-%j.out
#SBATCH --error=logs/mf-cnn-fno-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load devel/miniforge >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
PREPARATION="${PREPARATION:-experiments/multifield/preparation.json}"
RUN_DIR="${RUN_DIR:-checkpoints/3d_xhi/mf_cnn_fno/multifield_cnn_fno_s0}"
echo "preparation: $PREPARATION"; echo "run dir:     $RUN_DIR"; echo "started: $(date)"
srun python fno_multifield.py train \
    --preparation "$PREPARATION" \
    --inputs "${INPUTS:-density,los_velocity}" \
    --targets "${TARGETS:-neutral_fraction,brightness_temp}" \
    --model-settings "$(cat experiments/multifield/model_cnn_fno.json)" \
    --sampling full \
    --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-1}" \
    --lr "${LR:-3e-4}" --weight-decay "${WEIGHT_DECAY:-1e-5}" \
    --seed "${SEED:-0}" --workers "${WORKERS:-4}" --device cuda \
    --run-dir "$RUN_DIR"
echo "finished: $(date)"
