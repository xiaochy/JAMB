#!/bin/bash
set -e

TASK=lift_pot
CONFIG=demo_clean
N=100
SEED=0
GPU=1
ACTION_DIM=14
CKPT=50

cd /data/vxiao/bimanual_manipulation/RoboTwin/policy/DP

echo "=== [1/3] process_data ==="
python process_data.py $TASK $CONFIG $N

echo "=== [2/3] train (50 epochs) ==="
export CUDA_VISIBLE_DEVICES=$GPU
python train.py --config-name=robot_dp_${ACTION_DIM}.yaml \
    task.name=$TASK \
    task.dataset.zarr_path="data/${TASK}-${CONFIG}-${N}.zarr" \
    training.debug=False \
    training.seed=$SEED \
    training.device="cuda:0" \
    exp_name=${TASK}-robot_dp-train \
    logging.mode=online \
    setting=$CONFIG \
    expert_data_num=$N \
    head_camera_type=D435 \
    training.checkpoint_every=$CKPT \
    training.num_epochs=$CKPT

echo "=== [3/3] eval (100 rollouts, ckpt=${CKPT}) ==="
conda run -n RoboTwin bash eval.sh $TASK $CONFIG $CONFIG $N $SEED $GPU $CKPT
