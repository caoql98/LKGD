#!/bin/bash

# # Define the name of the first program
# first_program="annotate/annotate_segmentation.py"

# # Function to check if the first program is running
# is_running() {
#     pgrep -f $first_program > /dev/null 2>&1
#     return $?
# }

# # Wait until the first program is no longer running
# while is_running; do
#     sleep 5  # Check every 5 seconds
# done

# Run the second program
accelerate launch train_models/train_depth_lora_joint.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --train_data_dir="/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/PascalVOC/VOC2012/hed-safe/train_dataset.json" \
 --dataset_type="json" \
 --image_column="image" \
 --caption_column="text" \
 --validation_prompt="a large modern house with a stone walkway and a stone patio" \
 --output_dir="output_final/output_hed_safe_lora_joint_rank64_nta_convfuse" \
 --random_flip \
 --train_batch_size 16 \
 --num_train_epochs 40 \
 --mixed_precision fp16 \
 --rank 64 \
 --x_column "original_image" \
 --snr_gamma 5.0 \
 --cond_image_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/House.jpg" \
 --cond_depth_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/House-hed.png" \
 --checkpointing_steps 5000 \
 --mask_dropout_prob 0.1 \
 --prompt_dropout_prob 0.1 \
 --seed 142847 \
 --trigger_word "hed_map, " \
 --rand_transform \
 --report_to wandb \
 --train_y \
 --no_timestep_align \
 --post_joint conv_fuse \
 --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_hed_safe_lora_joint_rank64_nta_convfuse/halfstone-10000"
#  --y_lora "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger/y_lora" \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_bugfix2_no_noise_offset/checkpoint-9000"
#  --noise_offset 0.1 \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint/checkpoint-13000"
#   \

#  
#  --resume_from_checkpoint latest
