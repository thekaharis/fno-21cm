#!/bin/bash
# Continuation of checkpoints_3d_localfno_warped256_volw (job 4227743, hit
# 72 h walltime at epoch 47/50; best val_l2 at epoch 44).  Warm-starts from
# the epoch-44 best weights into a NEW checkpoint dir (the training entry
# point does not resume optimizer/scheduler state and clears metrics.jsonl
# in its target dir).  Cosine LR restarts at half the original peak; H1
# warmup disabled since the loaded model already trained with full H1.
# Submit: sbatch slurm/train_localfno_warped256_volw_cont.sbatch

#SBATCH --job-name=localfno-w256-cont
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=logs/localfno-w256-cont-%j.out
#SBATCH --error=logs/localfno-w256-cont-%j.err

ROOT=/pfs/10/work/hd_id260-fno_training/fno-21cm

export CHECKPOINT_DIR="$ROOT/checkpoints_3d_localfno_warped256_volw_cont"
export INIT_CHECKPOINT="$ROOT/checkpoints/checkpoints_3d_localfno_warped256_volw/best_model_state_dict.pt"
export N_EPOCHS=50

# match the original warped256_volw run (job 4227743)
export LOCALFNO_MODES_X=8
export LOCALFNO_MODES_Y=8
export LOCALFNO_MODES_Z=16
export LOSS_L2_MODE=relative
export LOSS_H1_MODE=relative
export LOSS_LOS_VOLUME_WEIGHTS=1
export PROJECT_CUBES=/pfs/10/work/hd_id260-fno_training/data/compressed/cubes_3d_warped256.h5

# continuation-specific: restarted cosine at half peak, no H1 re-ramp
export LOCALFNO_LEARNING_RATE=1e-4
export LOCALFNO_H1_WARMUP_EPOCHS=0

exec bash "$ROOT/slurm/train_localfno_h200_1gpu.sbatch"
