python vla-scripts/deploy.py \
  --pretrained_checkpoint /data/ckpt/MaskVLA/aloha_4_tasks/100000steps/openvla-7b+my_aloha_4_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--452+my_aloha_4_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--455 \
  --use_l1_regression True \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key my_aloha_picking_banana_new
  
/data/ckpt/MaskVLA/aloha_pick_banana_new/50000steps/openvla-7b+my_aloha_picking_banana_new+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--408

/data/ckpt/MaskVLA/aloha_4_tasks/100000steps/openvla-7b+my_aloha_4_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--452+my_aloha_4_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--455

/data/ckpt/MaskVLA/aloha_4_tasks/50000steps/openvla-7b+my_aloha_4_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--452

/data/ckpt/MaskVLA/picking_banana_wooden+sink/50000steps/openvla-7b+my_aloha_2_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--456