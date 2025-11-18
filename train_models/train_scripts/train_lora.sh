accelerate launch train_lora.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --variant="fp16" \
 --train_data_dir="/data/juicefs_sharing_data/72179586/repos/readout_guidance/readout_training/data/pseudo_labels/PascalVOC/depth"\
 --image_column="original_image" \
 --caption_column="text" \
 --validation_prompt="A photo of a cat" \
 --output_dir="noise_lora" \
 --random_flip \
 --train_batch_size 32 \
 --max_train_steps 10000 \
 --mixed_precision fp16 \
 --rank 32 \
 --report_to wandb \
 --lora_name noise_lora \
 --fix_noise 123
#  --resume_from_checkpoint latest

