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
accelerate launch train_models/train_depth_lora_joint.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --train_data_dir="/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/PascalVOC/VOC2012/openpose" \
 --dataset_type="imagefolder" \
 --image_column="image" \
 --caption_column="text" \
 --validation_prompt="a man and a little girl sitting on a bench" \
 --output_dir="output_final/output_pose_lora_joint_rank64_nta_convfuse" \
 --random_flip \
 --train_batch_size 16 \
 --num_train_epochs 40 \
 --mixed_precision fp16 \
 --rank 64 \
 --x_column "original_image" \
 --snr_gamma 5.0 \
 --cond_image_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/raw/PascalVOC/VOC2012/JPEGImages/2007_006744.jpg" \
 --cond_depth_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/PascalVOC/VOC2012/openpose/2007_006744.jpg" \
 --checkpointing_steps 5000 \
 --mask_dropout_prob 0.1 \
 --prompt_dropout_prob 0.1 \
 --seed 142857 \
 --trigger_word "keypoint_map, " \
 --rand_transform \
 --train_y \
 --report_to wandb \
 --post_joint conv_fuse \
 --no_timestep_align
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64/halfstone-10000"
#  --post_joint conv_fuse \
#  --skip_encoder
#  --y_lora "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger/y_lora" \

#  --noise_offset 0.1 \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint/checkpoint-13000"
#  --resume_from_checkpoint latest