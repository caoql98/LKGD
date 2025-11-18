#!/bin/bash

# # Define the name of the first program
first_program="/opt/conda/envs/jointdiff/bin/accelerate"

# Function to check if the first program is running
is_running() {
    pgrep -f $first_program > /dev/null 2>&1
    return $?
}

# Wait until the first program is no longer running
while is_running; do
    sleep 5  # Check every 5 seconds
done

# Run the second program
accelerate launch train_models/train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --train_data_dir="/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/PascalVOC/VOC2012/depth_anything_v2_gray/train_dataset.json" \
 --dataset_type="json" \
 --image_column="image" \
 --caption_column="text" \
 --validation_prompt="a man riding a horse in a field" \
 --output_dir="output_final/output_depth_gray_controlnet_ds001" \
 --random_flip \
 --train_batch_size 16 \
 --num_train_epochs 2000 \
 --mixed_precision fp16 \
 --rank 64 \
 --x_column "original_image" \
 --snr_gamma 5.0 \
 --cond_image_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/raw/PascalVOC/VOC2012/JPEGImages/2012_004104.jpg" \
 --cond_depth_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/pseudo_labels/PascalVOC/depth_anything_v2_gray/2012_004104.jpg" \
 --checkpointing_steps 5000 \
 --mask_dropout_prob 0.1 \
 --prompt_dropout_prob 0.1 \
 --seed 142857 \
 --rand_transform \
 --report_to wandb \
 --dataset_rescale 0.01
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_gray_controlnet_ds01/checkpoint-5000"
#  --train_lora \

#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_controlnet/checkpoint-8500"
#  --y_lora "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger/y_lora" \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64/halfstone-10000"
#  --noise_offset 0.1 \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint/checkpoint-13000"
#  --resume_from_checkpoint latest