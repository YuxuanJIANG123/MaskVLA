##LIBERO
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /home/jyx/openvla-oft/modified_libero_rlds \
  --dataset_name libero_10_no_noops \
  --run_root_dir /data/ckpt/openvla-oft \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 1000 \
  --max_steps 2000 \
  --save_freq 1000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "wge2002" \
  --wandb_project "LIBERO_openvla_oft" \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state

## kinova
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "/data/dataset/rlds_datasets" \
  --dataset_name lerobot_dataset \
  --run_root_dir "/data/ckpt/openvla-oft/lerobot_finetune" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 1 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 3000 \
  --max_steps 6000 \
  --use_val_set True \
  --val_freq 2000 \
  --save_freq 2000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity wge2002 \
  --wandb_project kinova_openvla-oft_1 \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--proprio_state--film

  ##aloha my_finetune resume
  torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/my_finetune.py \
  --vla_path  /new_data/ckpt/openvla-oft/sim/loss_siglip_dino_1:1/openvla-7b+my_aloha_2_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0801_loss_siglip_dinov2--30000_chkpt \
  --resume True   \
  --resume_step 30000 \
  --data_root_dir /new_data/dataset/openvla-rlds \
  --dataset_name my_aloha_sim_adjust_bottle_and_stapler_pad \
  --run_root_dir /new_data/ckpt/openvla-oft/sim/loss_siglip_dino_1:1 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 20000 \
  --max_steps 50000 \
  --use_val_set True \
  --val_freq 5000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "wge2002" \
  --wandb_project "aloha_test_loss_siglip_dinov2" \
  --run_id_note 0803_loss_siglip_dinov2
#  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film
# 记得修改/home/jyx/openvla-oft/prismatic/vla/constants.py里面的NUM_ACTIONS_CHUNK

  ##aloha my_finetune CUDA_VISIBLE_DEVICES="2,3" 
  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/my_finetune.py \
  --vla_path  /data/ckpt/openvla/models--openvla--openvla-7b/snapshots/openvla-7b \
  --data_root_dir /new_data/dataset/openvla-rlds \
  --dataset_name my_aloha_sim_hanging_mug \
  --run_root_dir /new_data/ckpt/openvla-oft/sim/hanging_mug/loss_siglip_dino_0.3:0.2 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 25000 \
  --max_steps 50000 \
  --use_val_set True \
  --val_freq 5000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "wge2002" \
  --wandb_project "aloha_hanging_mug_loss_0.3_0.5" \
  --run_id_note 0806_hanging_mug_loss_0.3_0.5

  ##aloha origin_finetune
  torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path  /data/ckpt/openvla/models--openvla--openvla-7b-oft/snapshots/openvla-7b \
  --data_root_dir /new_data/dataset/openvla-rlds \
  --dataset_name my_aloha_sim_stack_bowls_three \
  --run_root_dir /new_data/ckpt/openvla-oft/sim/stack_bowls_three/zero_0.1 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 15000 \
  --max_steps 30000 \
  --use_val_set True \
  --val_freq 5000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "wge2002" \
  --wandb_project "my_aloha_sim_stack_bowls_three_zero_0.1" \
  --run_id_note 0812_my_aloha_sim_stack_bowls_three_zero_0.1

  #my_aloha_sim_open_laptop
    torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path  /data/ckpt/openvla/models--openvla--openvla-7b-oft/snapshots/openvla-7b \
  --data_root_dir /new_data/dataset/openvla-rlds \
  --dataset_name my_aloha_sim_put_object_cabinet \
  --run_root_dir /new_data/ckpt/openvla-oft/sim/put_object_cabinet/zero_primary_0.1 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 15000 \
  --max_steps 30000 \
  --use_val_set True \
  --val_freq 5000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "wge2002" \
  --wandb_project "my_aloha_sim_put_object_cabinet_zero_primary_0.1" \
  --run_id_note my_aloha_sim_put_object_cabinet_zero_primary_0.1_0822