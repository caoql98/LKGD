accelerate launch --multi_gpu --num_processes 4 --main_process_port 32500 train_next_keyframe_lora.py \
 --pretrained_model_name_or_path="/data/vjuicefs_ai_camera_lgroup/11162591/gsy_workdir/hf_models_backup/runwayml/stable-diffusion-v1-5" \
 --train_data_dir="/data/vjuicefs_ai_camera/11162591/public_datasets/pandas70m" \
 --validation_prompt="a man walking on the road." \
 --output_dir="output_dir/output_lora_next_keyframe_v2" \
 --random_flip \
 --train_batch_size 2 \
 --num_train_epochs 10 \
 --mixed_precision fp16 \
 --rank 64 \
 --snr_gamma 5.0 \
 --cond_image_path "data/test_images/walk.jpg" \
 --fps_high 30 \
 --checkpointing_steps 1000 \
 --report_to wandb \

