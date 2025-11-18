accelerate launch --multi_gpu --num_processes 4 train_svd_trans_lora.py \
 --pretrained_model_name_or_path="/root/data/juicefs_sharing_data/11162591/code/models/stabilityai/stable-video-diffusion-img2vid" \
 --output_dir="output_svd_translation_lora_panda_finetune" \
 --video_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/pandas70m" \
 --condition_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/pandas70m" \
 --validation_image_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/DAVIS/JPEGImages/walking" \
 --validation_control_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/DAVIS/OpticalFlows/480p/walking" \
 --width=576 \
 --height=320 \
 --learning_rate=2e-5 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=40 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=2 \
 --checkpointing_steps=500 \
 --validation_steps=400 \
 --gradient_checkpointing \
 --conditioning_channels=4 \
 --rank 64 \
 --report_to wandb \
 --resume_model output_svd_translation_lora/checkpoint-47000
#  --mix_webvid 2000000 \
#  --resume_from_checkpoint latest
