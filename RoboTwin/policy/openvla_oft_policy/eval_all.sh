# baseline-oft 文件夹的5个命令
bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/baseline-oft/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--472--10000_chkpt" \
    "my_aloha_sim_adjust_bottle"

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/baseline-oft/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--472--20000_chkpt" \
    "my_aloha_sim_adjust_bottle" 

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/baseline-oft/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--472--30000_chkpt" \
    "my_aloha_sim_adjust_bottle" 

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/baseline-oft/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--472--40000_chkpt" \
    "my_aloha_sim_adjust_bottle" 

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/baseline-oft/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--472--50000_chkpt" \
    "my_aloha_sim_adjust_bottle" 

# loss_siqlip_dino_1:1 文件夹的4个命令

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/loss_siqlip_dino_1:1/openvla-7b+my_aloha_2_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0801_loss_siqlip_dinov2--10000_chkpt" \
    "my_aloha_sim_adjust_bottle" 

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/loss_siqlip_dino_1:1/openvla-7b+my_aloha_2_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0801_loss_siqlip_dinov2--20000_chkpt" \
    "my_aloha_sim_adjust_bottle" 

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/loss_siqlip_dino_1:1/openvla-7b+my_aloha_2_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0801_loss_siqlip_dinov2--30000_chkpt" \
    "my_aloha_sim_adjust_bottle" 


# 还有一个单独的文件

# set_zero_0.5 文件夹的5个命令
bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/set_zero_0.5/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--473--10000_chkpt" \
    "my_aloha_sim_adjust_bottle" 

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/set_zero_0.5/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--473--20000_chkpt" \
    "my_aloha_sim_adjust_bottle"

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/set_zero_0.5/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--473--30000_chkpt" \
    "my_aloha_sim_adjust_bottle" 

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/set_zero_0.5/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--473--40000_chkpt" \
    "my_aloha_sim_adjust_bottle" 

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 5 \
    "/new_data/ckpt/openvla-oft/sim/adjust_bottle/baseline-oft/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--472--30000_chkpt" \
    "my_aloha_sim_adjust_bottle"

bash eval.sh adjust_bottle demo_clean my_aloha_sim_adjust_bottle_and_stapler_pad 42 5 \
    "/new_data/ckpt/openvla-oft/sim/adjust_bottle/baseline-oft/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--472--30000_chkpt" \
    "my_aloha_sim_adjust_bottle"

bash eval.sh hanging_mug demo_randomized my_aloha_sim_hanging_mug 42 5 \
    "/new_data/ckpt/openvla-oft/sim/hanging_mug/loss_siglip_dino_0.3_0.2/openvla-7b+my_aloha_sim_hanging_mug+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0806_hanging_mug_loss_0.3_0.5--50000_chkpt" \
    "my_aloha_sim_hanging_mug"

bash eval.sh hanging_mug demo_clean my_aloha_sim_hanging_mug 42 1 \
    "/new_data/ckpt/openvla-oft/sim/hanging_mug/baseline-new/openvla-7b+my_aloha_sim_hanging_mug+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0811_my_aloha_sim_hanging_mug_baseline--30000_chkpt" \
    "my_aloha_sim_hanging_mug"

bash eval.sh stack_bowls_three demo_randomized my_aloha_sim_stack_bowls_three 42 0 \
    "/new_data/ckpt/openvla-oft/sim/stack_bowls_three/baseline/openvla-7b+my_aloha_sim_stack_bowls_three+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0812_my_aloha_sim_stack_bowls_three_baseline--30000_chkpt" \
    "my_aloha_sim_stack_bowls_three" \
    False

bash eval.sh stack_bowls_three demo_randomized my_aloha_sim_stack_bowls_three 42 0 \
    "/new_data/ckpt/openvla-oft/sim/stack_bowls_three/zero_0.1/openvla-7b+my_aloha_sim_stack_bowls_three+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0812_my_aloha_sim_stack_bowls_three_zero_0.1--30000_chkpt" \
    "my_aloha_sim_stack_bowls_three" \
    False

bash eval.sh grab_roller demo_randomized my_aloha_sim_grab_roller 42 0 \
    "/new_data/ckpt/openvla-oft/sim/grab_roller/baseline/openvla-7b+my_aloha_sim_grab_roller+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--my_aloha_sim_grab_roller_baseline_0816--20000_chkpt" \
    "my_aloha_sim_grab_roller" \
    False

bash eval.sh hanging_mug demo_randomized my_aloha_sim_hanging_mug 42 0 \
    "/new_data/ckpt/openvla-oft/sim/hanging_mug/zero_0.05/openvla-7b+my_aloha_sim_hanging_mug+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0810_my_aloha_sim_hanging_mug_zero_0.05--20000_chkpt" \
    "my_aloha_sim_hanging_mug" \
    True

bash eval.sh hanging_mug demo_randomized my_aloha_sim_hanging_mug 42 0 \
    "/new_data/ckpt/openvla-oft/sim/hanging_mug/zero_0.05/openvla-7b+my_aloha_sim_hanging_mug+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0810_my_aloha_sim_hanging_mug_zero_0.05--20000_chkpt" \
    "my_aloha_sim_hanging_mug" \
    True