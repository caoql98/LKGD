accelerate launch train_models/train_mask_depth_lora.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting" \
 --train_data_dir="/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/pseudo_labels/PascalVOC/depth_anything_v2_color" \
 --image_column="image" \
 --caption_column="text" \
 --validation_prompt="depth_map, a man riding a horse in a field" \
 --output_dir="output_dir/output_mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger" \
 --train_batch_size 32 \
 --num_train_epochs 20 \
 --mixed_precision fp16 \
 --rank 32 \
 --x_column "original_image" \
 --cond_image_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/pseudo_labels/PascalVOC/depth_anything_v2_color/2012_004104.jpg" \
 --checkpointing_steps 1000 \
 --mask_dropout_prob 0.25 \
 --prompt_dropout_prob 0.1 \
 --snr_gamma 5.0 \
 --random_flip \
 --seed 142847 \
 --trigger_word "depth_map, " \
 --rand_transform \
 --report_to wandb
#  --noise_offset 0.1
#  --trigger_word depth_map

# accelerate launch train_models/train_mask_depth_lora.py \
#  --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting" \
#  --train_data_dir="/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/pseudo_labels/PascalVOC/depth_anything_v2_gray" \
#  --image_column="image" \
#  --caption_column="text" \
#  --validation_prompt="depth_map, a man riding a horse in a field" \
#  --output_dir="output_dir/output_mask_depth_lora_depthanythingv2_gray_rank64_randtrans_trigger" \
#  --train_batch_size 32 \
#  --num_train_epochs 20 \
#  --mixed_precision fp16 \
#  --rank 32 \
#  --x_column "original_image" \
#  --cond_image_path "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/pseudo_labels/PascalVOC/depth_anything_v2_gray/2012_004104.jpg" \
#  --checkpointing_steps 1000 \
#  --mask_dropout_prob 0.25 \
#  --prompt_dropout_prob 0.1 \
#  --random_flip \
#  --snr_gamma 5.0 \
#  --seed 142847 \
#  --trigger_word "depth_map, " \
#  --rand_transform \
#  --report_to wandb
#    --noise_offset 0.1 \
#   
#  --resume_from_checkpoint latest
#   \

#  
#  --resume_from_checkpoint latest
