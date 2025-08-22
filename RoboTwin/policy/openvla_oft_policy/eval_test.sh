#!/bin/bash
set -e  # 启用严格模式

echo "▶️ 启动任务1 (zero-3w)"
bash eval.sh hanging_mug demo_clean my_aloha_sim_hanging_mug 42 7 \
    "/new_data/ckpt/openvla-oft/sim/hanging_mug/zero_0.05/openvla-7b+my_aloha_sim_hanging_mug+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0810_my_aloha_sim_hanging_mug_zero_0.05--30000_chkpt" \
    "my_aloha_sim_hanging_mug"

# 显式等待和清理
sleep 5  # 缓冲时间
pkill -f "python.*eval_policy" || true  # 忽略无进程可杀的错误

echo "▶️ 启动任务2 (zero-4w)"
bash eval.sh hanging_mug demo_clean my_aloha_sim_hanging_mug 42 7 \
    "/new_data/ckpt/openvla-oft/sim/hanging_mug/zero_0.05/openvla-7b+my_aloha_sim_hanging_mug+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0810_my_aloha_sim_hanging_mug_zero_0.05--40000_chkpt" \
    "my_aloha_sim_hanging_mug"

echo "✅ 所有任务顺序执行完成"

# bash eval.sh hanging_mug demo_clean my_aloha_sim_hanging_mug 42 5 \
#     "/new_data/ckpt/openvla-oft/sim/hanging_mug/baseline-new/openvla-7b+my_aloha_sim_hanging_mug+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0811_my_aloha_sim_hanging_mug_baseline--30000_chkpt" \
#     "my_aloha_sim_hanging_mug"

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 5 \
    "/new_data/ckpt/openvla-oft/sim/adjust_bottle/loss_siglip_dino_0.3:0.2/openvla-7b+my_aloha_2_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0806_loss_siglip_dinov2_0.3_0.5--30000_chkpt" \
    "my_aloha_sim_adjust_bottle"

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 5 \
    "/new_data/ckpt/openvla-oft/sim/adjust_bottle/set_zero_0.5/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--473--30000_chkpt" \
    "my_aloha_sim_adjust_bottle"

bash eval.sh adjust_bottle demo_clean my_aloha_sim_adjust_bottle_and_stapler_pad 42 5 \
    "/new_data/ckpt/openvla-oft/sim/adjust_bottle/baseline-oft/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--472--20000_chkpt" \
    "my_aloha_sim_adjust_bottle"

bash eval.sh stack_bowls_three demo_randomized my_aloha_sim_stack_bowls_three 42 3 \
    "/new_data/ckpt/openvla-oft/sim/stack_bowls_three/zero_0.1/openvla-7b+my_aloha_sim_stack_bowls_three+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0812_my_aloha_sim_stack_bowls_three_zero_0.1--20000_chkpt"\
    "my_aloha_sim_stack_bowls_three"


bash eval.sh stack_bowls_three demo_clean my_aloha_sim_stack_bowls_three 42 5 \
    "/new_data/ckpt/openvla-oft/sim/stack_bowls_three/zero_0.1/openvla-7b+my_aloha_sim_stack_bowls_three+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0812_my_aloha_sim_stack_bowls_three_zero_0.1--30000_chkpt"\
    "my_aloha_sim_stack_bowls_three"