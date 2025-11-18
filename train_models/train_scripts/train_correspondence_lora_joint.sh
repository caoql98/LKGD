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
accelerate launch train_models/train_track_lora_joint.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --dataset_type PandaN \
 --train_data_dir "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/Panda70M/validation_videos.json" \
 --image_column="target_frame" \
 --caption_column="text" \
 --validation_prompt="a golden retriever is running through the grass" \
 --output_dir="output_final/output_corres_lora_joint_rank64_nta_sym_pandaval" \
 --random_flip \
 --train_batch_size 2 \
 --num_train_epochs 1 \
 --mixed_precision fp16 \
 --rank 64 \
 --x_column "source_frame" \
 --snr_gamma 5.0 \
 --cond_image_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/corres_input/dog_00022.png" \
 --cond_depth_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/corres_input/dog_00037.png" \
 --cond_image_track_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/corres_input/dog_00022_track.png" \
 --cond_depth_track_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/corres_input/dog_00037_track.png" \
 --checkpointing_steps 1000 \
 --mask_dropout_prob 0.1 \
 --prompt_dropout_prob 0.1 \
 --seed 142857 \
 --train_y \
 --no_timestep_align \
 --symmetric \
 --report_to wandb \
 --resume_from_checkpoint "latest"
#  --with_add_cond \
#  
#  --skip_encoder
#  --post_joint conv_fuse \
#  --skip_encoder
#  --y_lora "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger/y_lora" \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64/halfstone-10000"
#  --noise_offset 0.1 \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint/checkpoint-13000"
#  --resume_from_checkpoint latest