import os
import random
import torch
import datetime
import numpy as np
from PIL import Image
import os
import sys
from safetensors import safe_open
# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
# from models.controlnet_sdv import ControlNetSDVModel
# from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import cv2
import re
# from pipeline.pipeline_stable_video_diffusion_tmp import StableVideoDiffusionPipeline
from diffusers.pipelines import StableVideoDiffusionPipeline
from diffusers.pipelines import StableDiffusionPipeline

from utils.dataset import readBatchedFlow 
from utils.util import validate_and_convert_image, save_gifs_side_by_side
from utils.util import load_lora_weights

def create_image_grid(images, rows, cols, target_size=(256, 256)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

def tensor_to_pil(tensor):
    """ Convert a PyTorch tensor to a PIL Image. """
    # Convert tensor to numpy array
    if len(tensor.shape) == 4:  # batch of images
        images = [Image.fromarray(img.numpy().transpose(1, 2, 0)) for img in tensor]
    else:  # single image
        images = Image.fromarray(tensor.numpy().transpose(1, 2, 0))
    return images

def save_combined_frames(batch_output, validation_images, validation_control_images, output_folder):
    # Flatten batch_output to a list of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Convert tensors in lists to PIL Images
    validation_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_images]
    validation_control_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_control_images]
    flattened_batch_output = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in batch_output]

    # Flatten lists if they contain sublists (for tensors converted to multiple images)
    validation_images = [img for sublist in validation_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    validation_control_images = [img for sublist in validation_control_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    flattened_batch_output = [img for sublist in flattened_batch_output for img in (sublist if isinstance(sublist, list) else [sublist])]

    # Combine frames into a list
    combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3
    rows = (num_images + cols - 1) // cols

    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols, target_size=(256, 256))
    if grid is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"combined_frames_{timestamp}.png"
        output_path = os.path.join(output_folder, filename)
        grid.save(output_path)
    else:
        print("Failed to create image grid")

def load_images_from_folder(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        # First, try the pattern 'frame_x_7fps'
        new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))

        # If the new pattern is not found, use the original digit extraction method
        matches = re.findall(r'\d+', filename)
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images

# Usage example
def convert_list_bgra_to_rgba(image_list):
    """
    Convert a list of PIL Image objects from BGRA to RGBA format.

    Parameters:
    image_list (list of PIL.Image.Image): A list of images in BGRA format.

    Returns:
    list of PIL.Image.Image: The list of images converted to RGBA format.
    """
    rgba_images = []
    for image in image_list:
        if image.mode == 'RGBA' or image.mode == 'BGRA':
            # Split the image into its components
            b, g, r, a = image.split()
            # Re-merge in RGBA order
            converted_image = Image.merge("RGBA", (r, g, b, a))
        else:
            # For non-alpha images, assume they are BGR and convert to RGB
            b, g, r = image.split()
            converted_image = Image.merge("RGB", (r, g, b))

        rgba_images.append(converted_image)

    return rgba_images

# Main script
if __name__ == "__main__":
    args = {
        "pretrained_model_name_or_path": "/code/weights/SVD-XT",
        # "validation_image_folder": "/data/juicefs_sharing_data/72179586/repos/VidToMe/data/tea-pour/frames",
        # "validation_control_folder": "/data/juicefs_sharing_data/72179586/repos/VidToMe/outputs/tea-pour/depth_image",
        "validation_image": "/code/datasets/test_imgs/900.jpg",
        "output_dir": "./output",
        "height": 512,
        "width": 512,
        "seed": 12345,
        "batch_size":1
        # cant be bothered to add the args in myself, just use notepad
    }

    # Load validation images and control images
    # validation_images = load_images_from_folder_to_pil(args["validation_image_folder"])
    #validation_images = convert_list_bgra_to_rgba(validation_images)
    # validation_control_images = load_images_from_folder_to_pil(args["validation_control_folder"])

    validation_image = Image.open(args["validation_image"]).convert('RGB')


    # Load and set up the pipeline
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("CiaraRowles/temporal-controlnet-depth-svd-v1",subfolder="controlnet")
    # unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args["pretrained_model_name_or_path"],subfolder="unet")
    pipeline = StableVideoDiffusionPipeline.from_pretrained(args["pretrained_model_name_or_path"], 
                        torch_dtype=torch.float16,     low_cpu_mem_usage=False,
                         device_map=None)
    # lora_path = "/code/svd-train-main/svd-train-main/train_models/output_svd_lora_exp2/pytorch_lora_weights.safetensors"  # 替换为你保存 LoRA 权重的路径
    # pipeline.load_lora_weights(lora_path)
    pipeline.enable_model_cpu_offload()
    lora_weights_path = "/code/svd-train-main/svd-train-main/train_models/output_svd_lora_exp2/pytorch_lora_weights.safetensors" 
    
    unet = pipeline.unet  # 提取 UNet 模型


    # lora_state_dict, lora_network_alphas = pipeline.lora_state_dict(lora_weights_path )
    # StableVideoDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet = pipeline.unet)

 # 加载 LoRA 权重（使用 safetensors 格式）
    lora_weights_path = "/code/svd-train-main/svd-train-main/train_models/output_svd_lora_exp2/pytorch_lora_weights.safetensors" 
    # load_lora_weights(unet, lora_weights_path, adapter_name = "y_lora")
    pipeline.unet.load_attn_procs(lora_weights_path)
    lora_state_dict = {}
    with safe_open(lora_weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state_dict[key] = f.get_tensor(key)
    # lora_state_dict1, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(lora_weights_path)
    # StableDiffusionPipeline.load_lora_into_unet(lora_state_dict1, lora_network_alphas, unet = pipeline.unet)
    # 遍历并更新 UNet 模型的参数
    # for name, param in unet_state_dict.items():
    #     print(name)
    unet_state_dict = unet.state_dict()

    # 只更新 LoRA 权重中存在的部分
    for name, param in lora_state_dict.items():
        # 去掉前缀 'unet.' 
        unet_param_name = name.replace('unet.', '', 1)
        
        if unet_param_name in unet_state_dict:
            print(f"Updating parameter: {unet_param_name}")
            unet_state_dict[unet_param_name].copy_(param)  # 更新模型参数
        # else:
        #     print(f"Parameter {unet_param_name} not found in UNet, skipping.")

    # 将更新后的参数加载回 UNet
    unet.load_state_dict(unet_state_dict)


    # Additional pipeline configurations can be added here
    #pipeline.enable_xformers_memory_efficient_attention()
    # Create output directory if it doesn't exist
    val_save_dir = os.path.join(args["output_dir"], "validation_images")
    os.makedirs(val_save_dir, exist_ok=True)

    # Inference and saving loop
    #print(validation_control_images.shape)
    # validation_image2 = Image.open("/data/juicefs_sharing_data/72179586/repos/VidToMe/outputs/dog/VG/frames/0013.png").convert('RGB')
    seed = args["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    start_f = 0
    n_frames = 14
    motion_bucket_id = 127
    # video_frames = pipeline([validation_image] * batch_size, motion_bucket_id = motion_bucket_id,decode_chunk_size=8,num_frames=n_frames,width=args["width"],height=args["height"], num_inference_steps=25, latents = latents).frames
    video_frames = pipeline([validation_image], 
                            # motion_bucket_id = motion_bucket_id,
                            # decode_chunk_size=8,
                            # num_frames=n_frames,
                            width=args["width"],
                            height=args["height"], 
                            # num_inference_steps=25
                            ).frames
    # video_frames = []
    # motion_bucket_ids = [i for i in range(8, 255, 9)]
    # for motion_bucket_id in motion_bucket_ids:
    #     batch_size = 1
    #     video_frame = pipeline([validation_image] * batch_size, motion_bucket_id = motion_bucket_id,decode_chunk_size=8,num_frames=n_frames,width=args["width"],height=args["height"], num_inference_steps=25, latents = latents[[0]]).frames
    #     video_frames += video_frame
    # video_frames = pipeline([validation_image, validation_image2], validation_control_images[:14], decode_chunk_size=8,num_frames=14,motion_bucket_id=10,controlnet_cond_scale=1.0,width=args["width"],height=args["height"]).frames
    save_gifs_side_by_side(video_frames,val_save_dir)
