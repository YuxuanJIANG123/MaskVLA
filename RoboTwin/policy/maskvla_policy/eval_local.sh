#!/bin/bash

policy_name=maskvla_policy
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5:-0}  # Default to GPU 0 if not specified

# MaskVLA specific parameters
pretrained_checkpoint=${6:-""}  # Path to checkpoint
unnorm_key=${7:-""}             # Action unnormalization key


use_proprio=${8:-true}          # Use proprioceptive input
num_images_in_input=${9:-3}     # Number of camera views
use_l1_regression=${10:-true}   # Use L1 regression action head
center_crop=${11:-true}         # Apply center crop
use_film=${12:-false}           # Use FiLM conditioning


export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mMaskVLA checkpoint: ${pretrained_checkpoint}\033[0m"
echo -e "\033[33mUnnorm key: ${unnorm_key}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_only_local.py --config policy/$policy_name/deploy_policy.yml \
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
    --use_film ${use_film}
    # --checkpoint_name ${checkpoint_name} # hjy