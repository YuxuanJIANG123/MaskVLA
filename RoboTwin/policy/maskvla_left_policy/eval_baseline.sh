#!/bin/bash

# 评测 baseline-oft 文件夹
BASE_DIR="/new_data/ckpt/MaskVLA/sim/baseline-oft"
GPU_ID=0

echo "=== 开始评测 baseline-oft 文件夹 ==="
echo "时间: $(date)"
echo "=================================="

for checkpoint in "$BASE_DIR/"*chkpt; do
    if [ -f "$checkpoint" ]; then
        filename=$(basename "$checkpoint")
        echo "正在评测: $filename"
        
        if [[ $filename == *"10000"* ]]; then
            name="baseline_472_10000"
        elif [[ $filename == *"20000"* ]]; then
            name="baseline_472_20000"
        elif [[ $filename == *"30000"* ]]; then
            name="baseline_472_30000"
        elif [[ $filename == *"40000"* ]]; then
            name="baseline_472_40000"
        elif [[ $filename == *"50000"* ]]; then
            name="baseline_472_50000"
        else
            name="baseline_472"
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
echo "baseline-oft 评测完成!"
echo "完成时间: $(date)"
echo "=================================="