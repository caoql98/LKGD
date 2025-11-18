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
output_dir="output_dir/output_id_lora_joint_celeba_r64_yr16_nta_sym"
echo "$(realpath $0)"
cp $(realpath $0) $output_dir

# Run the second program
accelerate launch train_models/train_depth_lora_joint.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --dataset_type="json" \
 --train_data_dir="/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/CelebA/dataset_celeba_idpairs_200k.json" \
 --image_column "image1_wild" "image1_align" \
 --caption_column "text0_wild" "text0_align" \
 --y_caption_column "text1_wild" "text1_align" \
 --x_column "image0_wild" "image0_align" \
 --validation_prompt="a close up of a woman with blond hair and blue eye" \
 --output_dir=$output_dir \
 --random_flip \
 --train_batch_size 16 \
 --num_train_epochs 2 \
 --mixed_precision fp16 \
 --rank 64 \
 --ylora_rank 16 \
 --snr_gamma 5.0 \
 --cond_image_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/id_inspect/0.png" \
 --cond_depth_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/id_inspect/0.png" \
 --checkpointing_steps 5000 \
 --mask_dropout_prob 0.1 \
 --prompt_dropout_prob 0.1 \
 --seed 142857 \
 --trigger_word "person_id, " \
 --no_timestep_align \
 --separate_xy_trans \
 --symmetric \
 --train_y \
 --report_to wandb \
 --resume_from_checkpoint "latest"
#  --random_crop
#  
#  --post_joint conv_fuse \
#  --skip_encoder
#  --y_lora "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger/y_lora" \

#  --noise_offset 0.1 \
#  --resume_from_checkpoint "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint/checkpoint-13000"
#  --resume_from_checkpoint latest