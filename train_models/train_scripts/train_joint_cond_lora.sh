accelerate launch train_joint_cond_lora.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --train_data_dir="/data/juicefs_sharing_data/72179586/repos/readout_guidance/readout_training/data/pseudo_labels/PascalVOC/depth" \
 --image_column="image" \
 --caption_column="text" \
 --validation_prompt="a close up of the handlebars of a bike" \
 --output_dir="output_lora_joint_depth_image_trainy_samet" \
 --random_flip \
 --train_batch_size 16 \
 --num_train_epochs 10 \
 --mixed_precision fp16 \
 --rank 128 \
 --y_lora "/data/vjuicefs_ai_camera/11162591/public_datasets/share_files/depth_lora/checkpoint-17000" \
 --x_column "original_image" \
 --snr_gamma 5.0 \
 --cond_image_path "/data/juicefs_sharing_data/72179586/repos/readout_guidance/readout_training/data/pseudo_labels/PascalVOC/depth/2009_004496.jpg" \
 --resume_model "output_lora_joint_depth_image_trainy_samet/checkpoint-9000" \
 --report_to wandb \
 --checkpointing_steps 1000 \
#  
#  --resume_from_checkpoint latest
