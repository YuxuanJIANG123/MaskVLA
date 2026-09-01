#!/bin/bash

policy_name=maskvla_policy
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}

# MaskVLA specific parameters
pretrained_checkpoint=${6:-""}  # Path to checkpoint
unnorm_key=${7:-""}             # Action unnormalization key

continue_evaling=${8:-false}    # 断点续训功能，Whether to continue evaluation from the latest directory if it exists

use_proprio=${9:-true}          # Use proprioceptive input
num_images_in_input=${10:-3}     # Number of camera views
use_l1_regression=${11:-true}   # Use L1 regression action head
center_crop=${12:-true}         # Apply center crop
use_film=${13:-true}           # Use FiLM conditioning

# checkpoint_name=${13} # hjy Checkpoint name for saving

echo "第8个参数是: $8"
echo "continue_evaling值是: ${8:-false}"

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mMaskVLA checkpoint: ${pretrained_checkpoint}\033[0m"
echo -e "\033[33mUnnorm key: ${unnorm_key}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --pretrained_checkpoint ${pretrained_checkpoint} \
    --unnorm_key ${unnorm_key} \
    --use_proprio ${use_proprio} \
    --num_images_in_input ${num_images_in_input} \
    --use_l1_regression ${use_l1_regression} \
    --center_crop ${center_crop} \
    --use_film ${use_film}  \
    --continue_evaling ${continue_evaling}


