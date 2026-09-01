## 多进程评估
export HF_HOME=/data/250010219/yuxuan/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_URL=$HF_ENDPOINT
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export MUJOCO_GL=egl

NUM_GPUS=$(nvidia-smi -L | wc -l)
MAX_PROCS_PER_GPU=3 
MAX_PROCS=$((NUM_GPUS * MAX_PROCS_PER_GPU))
TASK_NAME=libero_spatial

# 填你的 ckpt 列表
CKPTS=(
  "/data/250010219/yuxuan/Better-OFT/MaskVLA/ckpt/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--50000_chkpt"
  "/data/250010219/yuxuan/Better-OFT/MaskVLA/ckpt/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--100000_chkpt"
  "/data/250010219/yuxuan/Better-OFT/MaskVLA/ckpt/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--150000_chkpt"
)

job_idx=0
for CKPT in "${CKPTS[@]}"; do
    GPU_ID=$((job_idx % NUM_GPUS))
    echo "[INFO] Submit JOB: ckpt=$CKPT, gpu=$GPU_ID"

    CUDA_VISIBLE_DEVICES=$GPU_ID python experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint "$CKPT" \
        --task_suite_name $TASK_NAME &

    job_idx=$((job_idx+1))

    # 控制总并发进程数
    while [ $(jobs -rp | wc -l) -ge $MAX_PROCS ]; do
        sleep 2
        wait -n
    done
done

# 等待所有后台进程完成
wait
echo "All evaluations finished."
