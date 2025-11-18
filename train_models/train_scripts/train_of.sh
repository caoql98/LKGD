accelerate launch --multi_gpu --num_processes 2 --main_process_port 31500 train_svd_of.py \
 --pretrained_model_name_or_path="/root/data/juicefs_sharing_data/11162591/code/models/stabilityai/stable-video-diffusion-img2vid" \
 --output_dir="output_flow_lora_flow_1f_cond_fix_fixmotion_retrain" \
 --video_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/pandas70m" \
 --condition_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/pandas70m" \
 --validation_image_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/DAVIS/JPEGImages/walking" \
 --validation_control_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/DAVIS/OpticalFlows/480p/walking" \
 --width=576 \
 --height=320 \
 --learning_rate=1e-4 \
 --per_gpu_batch_size=2 \
 --num_train_epochs=40 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=2 \
 --checkpointing_steps=500 \
 --validation_steps=400 \
 --gradient_checkpointing \
 --conditioning_channels=4 \
 --train_flow_diffusion \
 --report_to wandb \
 --rank 128 \
 --resume_model output_flow_lora_flow_1f_cond_fix_fixmotion/checkpoint-41500 \
 --fix_motion_bucket_id \
#  --mix_webvid 100000 \

