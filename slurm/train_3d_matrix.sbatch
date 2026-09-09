#!/bin/bash
# One sbatch for the 3-D architecture x loss matrix, 50 epochs each.
#
#   ARCH  ufno | fno_fno | fno_whno | wno_whno | whno_whno
#         swhno_swhno | sfno_swhno | whno_swhno   (SIREN Walsh weights)
#         cnn_whno | cnn_swhno | cnn_fno | cnn_sfno   (U-FNO shape, global slot swapped)
#         lwf_lwf                              (learned orthonormal waveforms)
#   LOSS  plain | expwall | hybrid | expwall_theta | bsd
#
#   sbatch --export=ALL,ARCH=fno_whno,LOSS=hybrid slurm/train_3d_matrix.sbatch
#
# Runs on A100s: the measured peak is 22.3 GiB per rank, so an 80 GB A100 sits at
# ~28% while a 141 GB H200 is heavily over-provisioned, and the A100 pool is
# twice the size. An A30 is not an option -- 24 GB peaks at 94% on the first step
# and OOMs on the second when Adam allocates its moment buffers.
#
# Default is 20 epochs for a first pass. To continue a finished run:
#   sbatch --export=ALL,ARCH=fno_whno,LOSS=plain,N_EPOCHS=30,\
#     CONTINUE_FROM=checkpoints/checkpoints_3d_fno_whno_plain/final_model_state_dict.pt,\
#     CHECKPOINT_DIR=checkpoints/checkpoints_3d_fno_whno_plain_cont \
#     slurm/train_3d_matrix.sbatch
# That warm-starts the weights via INIT_CHECKPOINT. Note it restarts the epoch
# counter and the cosine LR schedule, so it is a fresh anneal from the loaded
# weights rather than a seamless resume.
#
# The hybrid weight is measured, not guessed. On real cubes L2/expwall is 3.0 for
# a constant-0.5 prediction and 85.8 once the prediction is merely blurred -- the
# ratio inverts over training. A fixed weight therefore swamps L2 early or
# vanishes late, so the hybrid ramps expwall in over EXPWALL_WARMUP_EPOCHS and
# then holds it at ~25% of the total.

#SBATCH --job-name=3dmatrix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
# No node exclusion. gpu2-07 and gpu3-03 both went through a spell of killing
# every job in 1-3 seconds with ExitCode 0:53 (which caught 12 of 12 matrix jobs
# at once, because a node that instantly kills its jobs is permanently the
# emptiest node and the scheduler preferentially fills it). Both have since been
# patched and are healthy. Excluding them costs 1 of 4 usable A100 nodes and 1 of
# 2 usable H200 nodes, which pushed the estimated start out by eight days -- so
# check `sacct -a -N <node>` for a run of 0:53 failures before excluding again
# rather than leaving a stale exclusion in place.
#SBATCH --mem=200G
#SBATCH --time=96:00:00
#SBATCH --output=logs/3dmatrix-%j.out
#SBATCH --error=logs/3dmatrix-%j.err

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fno-env
export PYTHONUNBUFFERED=1

ARCH="${ARCH:?set ARCH to ufno|fno_fno|fno_whno|wno_whno|whno_whno|swhno_swhno|sfno_swhno|whno_swhno|cnn_whno|cnn_swhno|cnn_fno|cnn_sfno|lwf_lwf}"
LOSS="${LOSS:?set LOSS to plain|expwall|hybrid|expwall_theta|bsd}"

# ---- architecture ------------------------------------------------------
case "$ARCH" in
  ufno)       export MODEL_KIND=ufno
              # --export=ALL can carry stale operator vars in from the calling
              # shell; an empty LOCAL_OPERATOR is a hard error in from_env().
              unset LOCAL_OPERATOR GLOBAL_OPERATOR ;;
  fno_fno)    export MODEL_KIND=localop LOCAL_OPERATOR=fourier  GLOBAL_OPERATOR=fourier ;;
  fno_whno)   export MODEL_KIND=localop LOCAL_OPERATOR=fourier  GLOBAL_OPERATOR=hadamard ;;
  wno_whno)   export MODEL_KIND=localop LOCAL_OPERATOR=wavelet  GLOBAL_OPERATOR=hadamard ;;
  whno_whno)  export MODEL_KIND=localop LOCAL_OPERATOR=hadamard GLOBAL_OPERATOR=hadamard ;;
  lwf_lwf)    export MODEL_KIND=localop LOCAL_OPERATOR=learned_waveform GLOBAL_OPERATOR=learned_waveform ;;
  # SIREN-generated Walsh weights: sequency is the Walsh analogue of
  # frequency, so the same hypernetwork trick the SirenFNO applies to
  # Fourier modes applies here (operators.SirenWalshHadamardOperator).
  swhno_swhno) export MODEL_KIND=localop LOCAL_OPERATOR=siren_hadamard GLOBAL_OPERATOR=siren_hadamard ;;
  sfno_swhno)  export MODEL_KIND=localop LOCAL_OPERATOR=siren_fourier  GLOBAL_OPERATOR=siren_hadamard ;;
  whno_swhno)  export MODEL_KIND=localop LOCAL_OPERATOR=hadamard       GLOBAL_OPERATOR=siren_hadamard ;;
  # U-FNO's shape with the global slot swapped. The U-FNO is a Fourier
  # bottleneck plus a real-space U-Net; "cnn" in the local slot IS that U-Net
  # (ConvUNetOperator, windowed=False so it runs at full width), so these two
  # isolate the global basis: Walsh-Hadamard, or SIREN-generated Walsh, against
  # the U-FNO's Fourier.
  cnn_whno)    export MODEL_KIND=localop LOCAL_OPERATOR=cnn GLOBAL_OPERATOR=hadamard ;;
  cnn_swhno)   export MODEL_KIND=localop LOCAL_OPERATOR=cnn GLOBAL_OPERATOR=siren_hadamard ;;
  # Same shape with the Fourier global slots, closing the cnn row: cnn_fno is
  # the closest thing in the matrix to a U-FNO built out of localop parts, and
  # cnn_sfno swaps in the SIREN-generated Fourier kernel.
  # NOTE: cnn_fno contains no SIREN, so SIREN_OMEGA is inert for it -- only
  # LOCALFNO_BASE_WIDTH carries over from the bw48om60 recipe.
  cnn_fno)     export MODEL_KIND=localop LOCAL_OPERATOR=cnn GLOBAL_OPERATOR=fourier ;;
  cnn_sfno)    export MODEL_KIND=localop LOCAL_OPERATOR=cnn GLOBAL_OPERATOR=siren_fourier ;;
  *) echo "unknown ARCH=$ARCH" >&2; exit 2 ;;
esac

# shared 3-D geometry, matching the existing 3-D sweeps so results compare
export LOCALFNO_WINDOW_X=16 LOCALFNO_WINDOW_Y=16 LOCALFNO_WINDOW_Z=32
export LOCALFNO_MODES_X=6  LOCALFNO_MODES_Y=6  LOCALFNO_MODES_Z=12
export LOCALFNO_BASE_WIDTH="${LOCALFNO_BASE_WIDTH:-16}"
export LOCALFNO_SPECTRAL_RANK="${LOCALFNO_SPECTRAL_RANK:-16}"
export LOCALFNO_PATCH_CHUNK_SIZE="${LOCALFNO_PATCH_CHUNK_SIZE:-128}"
export N_MODES_X=16 N_MODES_Y=16 N_MODES_Z=16
export HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-32}"
export N_LAYERS="${N_LAYERS:-4}"
export UFNO_WIDTH="${UFNO_WIDTH:-32}"
export LOCALWNO_LEVELS="${LOCALWNO_LEVELS:-2}"

# ---- training ----------------------------------------------------------
export N_EPOCHS="${N_EPOCHS:-20}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
export LOCALFNO_LEARNING_RATE="${LOCALFNO_LEARNING_RATE:-1e-4}"
export LOCALFNO_GRAD_CLIP_NORM="${LOCALFNO_GRAD_CLIP_NORM:-1.0}"
export LOCALFNO_H1_WARMUP_EPOCHS=0
export RUN_SEED="${RUN_SEED:-0}"
export SPLIT_SEED="${SPLIT_SEED:-42}"

# ---- loss --------------------------------------------------------------
# Absolute L2 throughout, so every cell of the matrix is on one scale. The older
# 3-D runs used a relative L2, which is why their numbers are not comparable.
export LOSS_L2_MODE=absolute LOSS_H1_MODE=absolute
export LOSS_H1_WEIGHT=0.0 LOSS_BCE_WEIGHT=0.0 LOSS_IONIZED_WALL_WEIGHT=0.0
export EXPWALL_SCALE="${EXPWALL_SCALE:-16.0}" WALL_CAP="${WALL_CAP:-32}"
case "$LOSS" in
  plain)
    export LOSS_L2_WEIGHT=1.0 LOSS_EXPWALL_WEIGHT=0.0
    export EXPWALL_WARMUP_EPOCHS=0 ;;
  expwall)
    export LOSS_L2_WEIGHT=0.0 LOSS_EXPWALL_WEIGHT=1.0
    export EXPWALL_WARMUP_EPOCHS=0 ;;
  hybrid)
    export LOSS_L2_WEIGHT=1.0 LOSS_EXPWALL_WEIGHT="${EXPWALL_HYBRID_W:-30.0}"
    export EXPWALL_WARMUP_EPOCHS="${EXPWALL_WARMUP_EPOCHS:-5}" ;;
  bsd)
    # BSD is auxiliary by construction: a size spectrum is blind to where the
    # bubbles are, so it rides on the expwall anchor rather than replacing it.
    export LOSS_L2_WEIGHT=1.0 LOSS_EXPWALL_WEIGHT="${EXPWALL_HYBRID_W:-30.0}"
    export EXPWALL_WARMUP_EPOCHS="${EXPWALL_WARMUP_EPOCHS:-5}"
    export LOSS_BSD_WEIGHT="${LOSS_BSD_WEIGHT:-1.0}"
    export BSD_WARMUP_EPOCHS="${BSD_WARMUP_EPOCHS:-5}"
    export BSD_RADII="${BSD_RADII:-1,2,4,8}"
    export BSD_DOWNSAMPLE="${BSD_DOWNSAMPLE:-2}"
    export BSD_MAX_SLICES="${BSD_MAX_SLICES:-64}" ;;
  expwall_theta)
    export LOSS_L2_WEIGHT=0.0 LOSS_EXPWALL_WEIGHT=1.0
    export EXPWALL_WARMUP_EPOCHS=0
    export CONTRAST_MODE=xhi CONTRAST_REFIT=1 CONTRAST_SCHEDULE_KIND=stepped
    export CONTRAST_BINS="${CONTRAST_BINS:-14}"
    # monotone: smooth the per-slice x_HI key along the LOS axis before it
    # picks a bin. Set CONTRAST_KEY=mean for the unsmoothed 2-D behaviour.
    export CONTRAST_KEY="${CONTRAST_KEY:-monotone}"
    export CONTRAST_REFIT_BATCH="${CONTRAST_REFIT_BATCH:-2}" ;;
  *) echo "unknown LOSS=$LOSS" >&2; exit 2 ;;
esac

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/checkpoints_3d_${ARCH}_${LOSS}}"
# Continuation: warm-start from a previous run's weights.
if [ -n "${CONTINUE_FROM:-}" ]; then
    if [ ! -f "$CONTINUE_FROM" ]; then
        echo "CONTINUE_FROM=$CONTINUE_FROM does not exist" >&2; exit 2
    fi
    export INIT_CHECKPOINT="$CONTINUE_FROM"
fi

export LIGHTCONE_DIR="${LIGHTCONE_DIR:-/pfs/10/work/hd_id260-fno_training/data/data}"
PROJECT_CUBES="${PROJECT_CUBES:-/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d.h5}"
LOCAL_SCRATCH="${LOCAL_SCRATCH:-yes}"
if [ "$LOCAL_SCRATCH" = "yes" ]; then
    LOCAL_CUBES="$TMPDIR/cubes_3d.h5"
    echo "Staging cube cache to local scratch: $LOCAL_CUBES"
    t0=$(date +%s)
    cp "$PROJECT_CUBES" "$LOCAL_CUBES"
    echo "  staged ($(du -h "$LOCAL_CUBES" | cut -f1)) in $(( $(date +%s) - t0 ))s"
    export CUBES_CACHE="$LOCAL_CUBES"
else
    export CUBES_CACHE="${CUBES_CACHE:-$PROJECT_CUBES}"
fi

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============ 3-D matrix: ARCH=$ARCH  LOSS=$LOSS ============"
echo "MODEL_KIND:   $MODEL_KIND ${LOCAL_OPERATOR:+($LOCAL_OPERATOR local / $GLOBAL_OPERATOR global)}"
echo "LOSS:         L2=$LOSS_L2_WEIGHT expwall=$LOSS_EXPWALL_WEIGHT"
echo "              scale=$EXPWALL_SCALE cap=$WALL_CAP warmup=$EXPWALL_WARMUP_EPOCHS"
echo "BSD:          ${LOSS_BSD_WEIGHT:-0} radii=${BSD_RADII:--} down=${BSD_DOWNSAMPLE:--} slices=${BSD_MAX_SLICES:--}"
echo "              axes=${EXPWALL_AXES:-3d}"
echo "CONTRAST:     ${CONTRAST_MODE:-off} ${CONTRAST_SCHEDULE_KIND:+/$CONTRAST_SCHEDULE_KIND}${CONTRAST_KEY:+ key=$CONTRAST_KEY}"
echo "EPOCHS:       $N_EPOCHS"
echo "CONTINUE:     ${INIT_CHECKPOINT:-fresh initialization}"
echo "CHECKPOINT:   $CHECKPOINT_DIR"
echo "============================================================"

python fno_21cm_3d.py
