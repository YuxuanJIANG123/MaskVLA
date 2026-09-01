#!/bin/bash

# 评测 loss_siqlip_dino_1:1 文件夹
BASE_DIR="/new_data/ckpt/MaskVLA/sim/loss_siqlip_dino_1:1"
GPU_ID=0

echo "=== 开始评测 loss_siqlip_dino_1:1 文件夹 ==="
echo "时间: $(date)"
echo "=================================="

for checkpoint in "$BASE_DIR/"*chkpt; do
    if [ -f "$checkpoint" ]; then
        filename=$(basename "$checkpoint")
        echo "正在评测: $filename"
        
        if [[ $filename == *"10000"* ]]; then
            name="loss_siqlip_dinov2_10000"
        elif [[ $filename == *"20000"* ]]; then
            name="loss_siqlip_dinov2_20000"
        elif [[ $filename == *"30000"* ]]; then
            name="loss_siqlip_dinov2_30000"
        else
            name="loss_siqlip_dinov2"
        fi
        
        bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 $GPU_ID \
            "$checkpoint" \
            "my_aloha_sim_adjust_bottle" \
            "$name"
        
        echo "完成: $name"
        echo "---"
    fi
done

echo "=================================="
echo "loss_siqlip_dino_1:1 评测完成!"
echo "完成时间: $(date)"
echo "=================================="