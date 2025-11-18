import os
import random
import torch
import datetime
import numpy as np
from PIL import Image
# from pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
# from models.controlnet_sdv import ControlNetSDVModel
# from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import cv2
import re
from pipeline.pipeline_stable_video_diffusion_smooth import StableVideoDiffusionPipeline
# from diffusers.pipelines import StableVideoDiffusionPipeline

from utils.dataset import process_frames, readBatchedFlow
from utils.util import load_images_from_folder_to_pil, load_input, load_joint_lora_to_pipeline 

def save_gifs_side_by_side(videos, output_folder, global_step = ""):
    # Helper function to convert tensors to PIL images and save as GIF
    # if isinstance(batch_output[0], list):
    #     batch_output = batch_output[0]
    # flattened_batch_output = [img for img in batch_output]



    def create_gif(image_list, gif_path, duration=100):
        pil_images = [validate_and_convert_image(img) for img in image_list]
        pil_images = [img for img in pil_images if img is not None]
        if pil_images:
            pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], loop=0, duration=duration)

    # Creating GIFs for each image list
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gif_paths = []
    for idx, image_list in enumerate(videos):
        if isinstance(image_list[0], list):
            image_list = image_list[0]
        gif_path = os.path.join(output_folder, f"temp_{idx}_{timestamp}.gif")
        create_gif(image_list, gif_path)
        gif_paths.append(gif_path)

    # Function to combine GIFs side by side
    def combine_gifs_side_by_side(gif_paths, output_path):
        gifs = [Image.open(gif) for gif in gif_paths]
    
        # Find the minimum frame count among all GIFs
        min_frames = min(gif.n_frames for gif in gifs)
    
        frames = []
        for frame_idx in range(min_frames):
            combined_frame = None
            for gif in gifs:
                gif.seek(frame_idx)
                if combined_frame is None:
                    combined_frame = gif.copy()
                else:
                    combined_frame = get_concat_v(combined_frame, gif.copy())
            frames.append(combined_frame)
    
        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=gifs[0].info['duration'])


    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    # Helper function to concatenate images vertically
    def get_concat_v(im1, im2):
        dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    # Combine the GIFs into a single file
    combined_gif_path = os.path.join(output_folder, f"combined_frames_{timestamp}_{global_step}.gif")
    combine_gifs_side_by_side(gif_paths, combined_gif_path)

    # Clean up temporary GIFs
    for gif_path in gif_paths:
        os.remove(gif_path)

    return combined_gif_path

# Define functions
def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        # image = image.resize(target_size)
        pass
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

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
        "pretrained_model_name_or_path": "/root/data/juicefs_sharing_data/11162591/code/models/stabilityai/stable-video-diffusion-img2vid",
        "validation_path": "data/test_images/video_trans_input/badminton.mp4",
        # "validation_control_folder": "/data/juicefs_sharing_data/72179586/repos/VidToMe/outputs/tea-pour/depth_image",
        # "validation_image": "/data/vjuicefs_ai_camera/11162591/public_datasets/DAVIS/JPEGImages/walking/00000.jpg",
        "output_dir": "./output",
        "height": 512,
        "width": 512,
        "seed": 111,
        "batch_size":1
        # cant be bothered to add the args in myself, just use notepad
    }

    # Load validation images and control images
    validation_images = load_input(args["validation_path"])

    validation_images = process_frames(validation_images, args["height"], args["width"])
    #validation_images = convert_list_bgra_to_rgba(validation_images)
    # validation_control_images = load_images_from_folder_to_pil(args["validation_control_folder"])

    # validation_image = Image.open(args["validation_image"]).convert('RGB')


    # Load and set up the pipeline
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("CiaraRowles/temporal-controlnet-depth-svd-v1",subfolder="controlnet")
    # unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args["pretrained_model_name_or_path"],subfolder="unet")
    pipeline = StableVideoDiffusionPipeline.from_pretrained(args["pretrained_model_name_or_path"], torch_dtype=torch.float16)
    pipeline.enable_model_cpu_offload()
    # Additional pipeline configurations can be added here
    #pipeline.enable_xformers_memory_efficient_attention()
    # Create output directory if it doesn't exist
    val_save_dir = os.path.join(args["output_dir"], "validation_images")
    os.makedirs(val_save_dir, exist_ok=True)

    output_path = "output_svd_translation_lora_panda_finetune/checkpoint-5500"
    load_joint_lora_to_pipeline(pipeline, checkpoint_path=output_path)
    pipeline.to(torch.float16)

    # Inference and saving loop
    #print(validation_control_images.shape)
    # validation_image2 = Image.open("/data/juicefs_sharing_data/72179586/repos/VidToMe/outputs/dog/VG/frames/0013.png").convert('RGB')
    seed = args["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    start_f = 0
    n_frames = 14
    batch_size = args["batch_size"]

    # motion_bucket_id = 255
    # video_frames = pipeline([validation_image] * batch_size, motion_bucket_id = motion_bucket_id,decode_chunk_size=8,num_frames=n_frames,width=args["width"],height=args["height"], num_inference_steps=25, latents = latents).frames
    video_frames = pipeline(validation_images,
                            height = args["height"],
                            width = args["width"],
                            start_step = 15,
                            motion_bucket_id = 127,
                            num_inference_steps=25,
                            # max_guidance_scale = 1.0,
                            fps = 30,
                            noise_aug_strength = 0.1).frames
    # video_frames = []
    # motion_bucket_ids = [i for i in range(8, 255, 9)]
    # for motion_bucket_id in motion_bucket_ids:
    #     batch_size = 1
    #     video_frame = pipeline([validation_image] * batch_size, motion_bucket_id = motion_bucket_id,decode_chunk_size=8,num_frames=n_frames,width=args["width"],height=args["height"], num_inference_steps=25, latents = latents[[0]]).frames
    #     video_frames += video_frame
    # video_frames = pipeline([validation_image, validation_image2], validation_control_images[:14], decode_chunk_size=8,num_frames=14,motion_bucket_id=10,controlnet_cond_scale=1.0,width=args["width"],height=args["height"]).frames
    save_gifs_side_by_side(video_frames,val_save_dir)
