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

bash eval.sh adjust_bottle demo_randomized my_aloha_sim_adjust_bottle_and_stapler_pad 42 0 \
    "/new_data/ckpt/openvla-oft/sim/set_zero_0.5/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--473--50000_chkpt" \
    "my_aloha_sim_adjust_bottle"