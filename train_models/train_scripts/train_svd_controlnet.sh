accelerate launch --multi_gpu --num_processes 3 --main_process_port 32500 train_svd_controlnet.py \
 --pretrained_model_name_or_path="/data/vjuicefs_ai_camera_lgroup/11162591/gsy_workdir/hf_models_backup/stabilityai/stable-video-diffusion-img2vid-xt" \
 --video_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/pandas70m" \
 --output_dir "output_dir/output_optical_flow_controlnet_refine" \
 --condition_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/pandas70m" \
 --motion_folder="path-to-your-motion" \
 --validation_image_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/DAVIS/JPEGImages/walking" \
 --validation_control_folder="/data/vjuicefs_ai_camera/11162591/public_datasets/DAVIS/OpticalFlows/480p/walking" \
 --width=576 \
 --height=320 \
 --learning_rate=2e-5 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=20 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=2 \
 --checkpointing_steps=1000 \
 --validation_steps=400 \
 --gradient_checkpointing \
 --conditioning_channels=1000 \
 --num_frames 25 \
 --resume_from_checkpoint latest
#  --resume_model output_dir/output_optical_flow_controlnet/checkpoint-139000/model.safetensors
#  --resume_from_checkpoint latest
#  --report_to wandb \
#  --resume_from_checkpoint latest