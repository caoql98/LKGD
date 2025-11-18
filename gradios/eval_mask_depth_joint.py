
import sys
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/.cache/huggingface"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import datetime
from PIL import Image
import gradio as gr

from utils.util import load_joint_diffusion_pipeline, load_scheduler, load_controlnet_inpaint_pipeline
# from diffusers import StableDiffusionInpaintPipeline
# from pipeline import pipeline_stable_diffusion_inpaint_controlnet as PipelineControlNet
# from pipeline import pipeline_stable_diffusion_inpaint_guidance as PipelineGuidance

from pipeline.pipeline_stable_diffusion_inpaint_controlnet import StableDiffusionInpaintControlNetPipeline
from pipeline.pipeline_stable_diffusion_inpaint_guidance import StableDiffusionInpaintPipeline
from PIL import Image, ImageOps
from patch import patch
# from diffusers import AutoPipelineForText2Image
import torch
# from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
# from diffusers.loaders import UNet2DConditionLoadersMixin, PeftAdapterMixin, FromSingleFileMixin
# from pipeline.pipeline_stable_diffusion_joint_control import StableDiffusionPipelineJointControl
import torchvision.transforms as T
# from safetensors import safe_open
# from diffusers import StableDiffusionInpaintPipeline
# from pipeline.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid
from utils.dataset import process_frames
# from utils.gaussian_2d import get_guassian_2d_rand_mask
from utils.util import blip_cap, load_image_folder
import random
import numpy as np
from utils.gaussian_2d import get_rand_masks
import torchvision.transforms.v2 as transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
# from torchvision import
from gradio_config import checkpoint_dict, additional_kwargs_dict, trigger_word_map, checkpoint_default_ylora_map, y_lora_dict, base_model_id_map, base_model_dict
from datasets import Image as DatasetImage
import json
import pdb
import argparse
# pdb.set_trace()





# blip_processor = BlipProcessor.from_pretrained(
#         "Salesforce/blip-image-captioning-large")
# blip_model = BlipForConditionalGeneration.from_pretrained(
#         "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

def set_joint_mask(model, lora_names, symmetric = False, do_guidance = False):
    joint_attention_mask = [1,0]
    if symmetric:
        y_mask = [1,1]
        x_mask = [1,1]
    else:
        y_mask = [0,1]
        x_mask = [1,0]
    
    if do_guidance:
        joint_attention_mask *= 2
        y_mask *= 2
        x_mask *= 2
    
    print("Joint attention mask", joint_attention_mask)
    print("Y mask", y_mask)
    print("X mask", x_mask)


    patch.set_joint_attention_mask(model, joint_attention_mask)
    for lora in lora_names:
        if lora[0] == "y":
            patch.set_patch_lora_mask(model, lora, y_mask)
        elif lora[0] == "x":
            patch.set_patch_lora_mask(model, lora, x_mask)

def process(pipe, depth_images, raw_images, image_mask, depth_mask, prompts, additional_prompt, negative_prompt, use_blip_cap, height, width, seed, inference_step, model_selection, scheduler_selection, y_lora_selection, base_model_selection, strength, guidance_scale, y_advance, eta, fix_yt, replace_end):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    print("Generate Image")
    
    # depth_image = Image.open(depth_image["image"])
    # raw_image = Image.open(raw_image["image"])
    if image_mask is None:
        image_mask = Image.new('RGB', (width, height), (255, 255, 255))
    else:
        image_mask = Image.open(image_mask)
    if depth_mask is None:
        depth_mask = Image.new('RGB', (width, height), (0, 0, 0))
    else:
        depth_mask = Image.open(depth_mask)
    
    # print(depth_image.mode)
    with torch.no_grad():
        # if depth_image.mode == "I":
        #     depth_tensor = T.ToTensor()(depth_image)
        #     depth_tensor = 1 - (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
        #     depth_image = T.ToPILImage()(depth_tensor)
        # depth_image, raw_image = depth_image["background"].convert("RGB"), raw_image["background"].convert("RGB")

        # if use_blip_cap:
        #     prompts = blip_cap(raw_images, blip_processor, blip_model)

        prompts = [additional_prompt + prompt for prompt in prompts]
        # print(prompt)
        # control_frames = process_frames(control_frames, h = height, w = width, verbose = True, div = 64)

        

        init_images = [*raw_images,  *depth_images]
        # init_images = [process_frames([init_image], height, width,  verbose = True, div = 64)[0] for init_image in init_images]
        init_images = [process_frames([init_image], height, width,  verbose = False, div = 8)[0] for init_image in init_images]
        # n_pairs = int(batch_size // 2)
        # init_images = [init_images[0]] * n_pairs + [init_images[1]] * n_pairs

        # mask_images = mask_gen(batch_size, height, width, patch_size, gaussian_thresh, mask_selection)
        # mask_images = generated_masks
        n_pairs = len(init_images) // 2
        mask_images = [image_mask] * n_pairs + [depth_mask] * n_pairs

        if not use_controlnet():
            patch.set_joint_attention(pipe.unet, enable = True)
            patch.set_joint_scale(pipe.unet, scale = 1.0)


            # if guidance_scale <= 1.0:
                # patch.set_joint_attention(pipe.unet, enable = enable_joint_diffusion)
                # patch.set_joint_scale(pipe.unet, scale = joint_scale)

                # if skip_encoder:
                #     patch.set_joint_attention(pipe.unet, enable = False, name_filter = "down_blocks")
                
            set_joint_mask(pipe.unet, lora_names = ["y_lora", "yx_lora", "xy_lora"], symmetric = "sym" in model_id, do_guidance=guidance_scale > 1.0)


        load_scheduler(pipe, mode = scheduler_selection)
        # print(pipeline.scheduler)
        if seed is not None:
            generator = torch.Generator(device="cuda")
            generator = generator.manual_seed(seed)
        else:
            generator = None

        if negative_prompt is not None:
            negative_prompt = [negative_prompt] * len(prompts)

        trigger_word = trigger_word_map[y_lora_id]
        prompts = prompts + [trigger_word + prompt for prompt in prompts]

        y_advance = None if y_advance == 0 else y_advance

        with torch.autocast(device_type = "cuda", dtype = torch.float16):
            images = pipe(
                prompts,
                image = init_images,
                mask_image = mask_images,
                height = height,
                width = width,
                num_inference_steps=inference_step,
                guidance_scale = guidance_scale,
                strength = strength,
                generator=generator,
                negative_prompt = negative_prompt,
                y_advance = y_advance,
                eta = eta,
                fix_yt = fix_yt,
                replace_end = replace_end
            ).images
        
        if len(images) > len(prompts):
            images = images[:n_pairs]
        
        
    return images

def load_pipeline():
    print(additional_kwargs_dict[model_id])
    if use_controlnet():
        pipe = load_controlnet_inpaint_pipeline(base_model_dict[base_model_id], checkpoint_dict[model_id], StableDiffusionInpaintControlNetPipeline, y_lora_path = y_lora_dict[y_lora_id], **additional_kwargs_dict[model_id])
    else:
        pipe = load_joint_diffusion_pipeline(base_model_dict[base_model_id], checkpoint_dict[model_id], StableDiffusionInpaintPipeline, y_lora_path = y_lora_dict[y_lora_id], **additional_kwargs_dict[model_id])

    return pipe


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default=None,
        required=True,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model_id = args.model_id
    y_lora_id = checkpoint_default_ylora_map[model_id]
    base_model_id = base_model_id_map[model_id]
    use_controlnet = lambda: "controlnet" in model_id

    pipe = load_pipeline()
    val_dataset_path = args.val_dataset

    output_dir = os.path.join(checkpoint_dict[model_id], "eval")

    print("output folder", output_dir)
    os.makedirs(output_dir, exist_ok = True)
    val_batch_size = 16
    # pdb.set_trace()
    dataset = load_image_folder(val_dataset_path, dataset_type="json")["train"]
    column_names = dataset.column_names
    image_column = "image"
    x_column = "original_image"
    scheduler_selection = "ddim"
    # prompt_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/annotate/PascalVOC_prompt.json"

    default_dict = {
        "image_mask" : None,
        "depth_mask" : None,
        "additional_prompt" : "", 
        "negative_prompt" : None, 
        "use_blip_cap" : False,
        "height" : 512,
        "width" : 512,
        "seed" : None, 
        "inference_step" : 50,
        "model_selection" : model_id, 
        "scheduler_selection" : scheduler_selection, 
        "y_lora_selection" : "default", 
        "base_model_selection" : "default", 
        "strength" : 1.0,
        "guidance_scale" : 7.5, 
        "y_advance" : 1.0, 
        "eta" : 1.0, 
        "fix_yt" : False,
        "replace_end": 0.0
    }

    seed = 142857
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for column in column_names:
        if "image" in column:
            dataset = dataset.cast_column(column, DatasetImage())

    def collate_fn(examples):
        raw_images = [example[x_column].convert("RGB") for example in examples]
        depth_images = [example[image_column].convert("RGB") for example in examples]
        file_names = [example["file_name"] for example in examples]
        prompts = [example["text"] for example in examples]
        return {"raw_images": raw_images, "depth_images": depth_images, "file_names": file_names, "prompts": prompts}

    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=val_batch_size,
        collate_fn = collate_fn
    )

    # with open(prompt_path, "r") as f:
    #     prompt_dict = json.load(f)



    for batch in val_dataloader:
        raw_images, depth_images, file_names = batch["raw_images"], batch["depth_images"], batch["file_names"]
        # prompts = [prompt_dict[fn][0] for fn in file_names]
        prompts = batch["prompts"]
        print(prompts)
        images = process(pipe = pipe, depth_images = depth_images, raw_images = raw_images, prompts = prompts, **default_dict)
        for file_name, image in zip(file_names, images):
            image.save(os.path.join(output_dir, file_name))
    
    # pdb.set_trace()


