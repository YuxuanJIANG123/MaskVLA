export HF_HOME=/data/250010219/yuxuan/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# 指向你的镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_URL=$HF_ENDPOINT

# 离线模式开关（防止访问官方服务器）
export HF_HUB_OFFLINE=0          # 0 表示可联网，1 表示完全离线
export TRANSFORMERS_OFFLINE=0    

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /data/250010219/yuxuan/Better-OFT/MaskVLA/ckpt/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--50000_chkpt \
  --task_suite_name libero_spatial