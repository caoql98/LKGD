#!/bin/bash

# # Define the name of the first program
# first_program="/opt/conda/envs/jointdiff/bin/accelerate"

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
accelerate launch train_models/train_mask_depth_lora_joint.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting" \
 --dataset_type track \
 --dataset_config "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/train_models/train_configs/track_dataset.yaml" \
 --image_column="target_frame" \
 --caption_column="text" \
 --validation_prompt="a goat standing on top of a rocky hill" \
 --output_dir="output_dir/output_mask_correspondence_lora_joint_rank16_skipe" \
 --random_flip \
 --train_batch_size 16 \
 --num_train_epochs 10 \
 --mixed_precision fp16 \
 --rank 16 \
 --x_column "source_frame" \
 --snr_gamma 5.0 \
 --cond_image_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/example_inputs/goat-00057.png" \
 --cond_depth_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/example_inputs/goat-00037.png" \
 --checkpointing_steps 1000 \
 --mask_dropout_prob 0.1 \
 --prompt_dropout_prob 0.1 \
 --seed 142857 \
 --trigger_word "corres_map, " \
 --train_y \
 --report_to wandb \
 --skip_encoder
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_correspondence_lora_joint_rank64/checkpoint-11000"


#  --y_lora "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger/y_lora" \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64/halfstone-10000"
#  --noise_offset 0.1 \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint/checkpoint-13000"
#  --resume_from_checkpoint latest