python train_svd_lora.py \
 --pretrained_model_name_or_path="/code/weights/SVD-XT" \
 --output_dir="output_34fuse_svd_lora_depthCharge_vision" \
 --video_folder="/code/datasets/fluid_videos/depthCharge_vision/train_videos" \
 --validation_image="/code/datasets/fluid_videos/depthCharge_vision/test_imgs/125.jpg" \
 --width=512 \
 --height=512 \
 --learning_rate=2e-4 \
 --per_gpu_batch_size=2 \
 --num_train_epochs=15 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=2 \
 --checkpointing_steps=100 \
 --validation_steps=50 \
 --gradient_checkpointing \
 --conditioning_channels=4 \
 --rank 4 \
#  --report_to wandb \
#  --resume_from_checkpoint latest
#  --mix_webvid 100000 \