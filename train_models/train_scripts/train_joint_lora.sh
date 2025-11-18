accelerate launch --multi_gpu --num_processes 2 train_joint_lora.py \
 --pretrained_model_name_or_path="/root/data/juicefs_sharing_data/11162591/code/models/runwayml/stable-diffusion-v1-5" \
 --train_data_dir="/data/juicefs_sharing_data/11162591/code/lxr/svd-train/data/depth" \
 --image_column="image" \
 --caption_column="text" \
 --validation_prompt="A room" \
 --output_dir="output_lora_joint_depth_image_clean_cond_trainall" \
 --random_flip \
 --train_batch_size 16 \
 --num_train_epochs 100 \
 --mixed_precision fp16 \
 --rank 128 \
 --y_lora "/data/juicefs_sharing_data/11162591/code/lxr/svd-train/output_depth_lora/checkpoint-17000" \
 --x_column "original_image" \
 --report_to wandb \
 --snr_gamma 5.0 \
 --resume_from_checkpoint latest \
 --clean_cond
#  --checkpointing_steps 5000 \

#  
#  --resume_from_checkpoint latest
