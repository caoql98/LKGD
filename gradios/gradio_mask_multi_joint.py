
import sys
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/.cache/huggingface"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import datetime
from PIL import Image
import gradio as gr

from utils.util import load_joint_diffusion_pipeline, load_scheduler
from diffusers import StableDiffusionInpaintPipeline
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
from pipeline.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid
from utils.dataset import process_frames
# from utils.gaussian_2d import get_guassian_2d_rand_mask
from utils.util import blip_cap
import random
import numpy as np
from utils.gaussian_2d import get_rand_masks
import torchvision.transforms.v2 as transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
# from torchvision import

# if gr.NO_RELOAD:
checkpoint_dict = {
    "mask_depth_joint": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint",
    "mask_depth_joint_freezey": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_freezey",
    "mask_depth_joint_depth_anything_v2_color_bugfix2": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_bugfix2/checkpoint-3000",
    "mask_depth_joint_depth_anything_v2_color_bugfix2_no_noise_offset": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_bugfix2_no_noise_offset",
    "mask_normal_lora_joint_rank64" : "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_rank64",
    "mask_sr_lora_joint": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_sr_lora_joint",
    "mask_normak_lora_joint_rank64_skipencoder": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64_skipencoder",
    "mask_seg_lora_joint_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_seg_lora_joint_rank64"
}

trigger_word_map = {
        "mask_depth_joint": "",
        "mask_depth_joint_freezey": "",
        "mask_depth_joint_depth_anything_v2_color_bugfix2": "",
        "mask_depth_joint_depth_anything_v2_color_bugfix2_no_noise_offset": "",
        "mask_normal_lora_joint_rank64": "normal_map, ",
        "mask_depth_lora_joint_depth_anything_v2_color_rank64": "depth_map, ",
        "mask_sr_lora_joint": "",
        "mask_normak_lora_joint_rank64_skipencoder": "normal_map, ", 
        "mask_seg_lora_joint_rank64": "seg_map, "
}

checkpoint_default_ylora_map = {
    "mask_depth_joint": "mask_depth_joint",
    "mask_depth_joint_freezey": "mask_depth",
    "mask_depth_joint_depth_anything_v2_color_bugfix2": "mask_depth_noise_offset_mask_fix_with_trigger_depth_anythingv2_color_fixbug2",
    "mask_depth_joint_depth_anything_v2_color_bugfix2_no_noise_offset": "mask_depth_noise_offset_mask_fix_with_trigger_depth_anythingv2_color_fixbug2",
    "mask_normal_lora_joint_rank64": "mask_normal_lora_joint_rank64",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64": "mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger",
    "mask_sr_lora_joint": "mask_sr_lora_joint",
    "mask_normak_lora_joint_rank64_skipencoder": "mask_normak_lora_joint_rank64_skipencoder",
    "mask_seg_lora_joint_rank64": "mask_seg_lora_joint_rank64"
}

y_lora_dict = {
    "mask_depth_joint": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint/y_lora",
    "mask_depth": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora/y_lora",
    "mask_depth_noise_offset": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_noise_offset/y_lora",
    "mask_depth_noise_offset_mask_fix": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_noise_offset_mask_fix/y_lora",
    "mask_depth_noise_offset_mask_fix_with_trigger": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_noise_offset_mask_fix_with_trigger/y_lora",
    "mask_depth_noise_offset_mask_fix_with_trigger_depth_anythingv2_gray_fixbug2": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_noise_offset_mask_fix_with_trigger_depthanythingv2_gray_fixbug2/y_lora",
    "mask_depth_noise_offset_mask_fix_with_trigger_depth_anythingv2_color_fixbug2": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_noise_offset_mask_fix_with_trigger_depthanythingv2_color_fixbug2/y_lora",
    "mask_depth_joint_depth_anything_v2_color_rank32": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank32/y_lora",
    "mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger/y_lora",
    "mask_normal_lora_joint_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64/y_lora",
    "mask_sr_lora_joint": None,
    "mask_normak_lora_joint_rank64_skipencoder": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64_skipencoder/y_lora",
    "mask_seg_lora_joint_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_seg_lora_joint_rank64/y_lora"
}

model_id = "mask_depth_joint_depth_anything_v2_color_bugfix2_no_noise_offset"
y_lora_id = checkpoint_default_ylora_map[model_id]
base_model_id = "runwayml/stable-diffusion-inpainting"


pipeline = load_joint_diffusion_pipeline(base_model_id, checkpoint_dict[model_id], StableDiffusionInpaintPipeline, y_lora_path = y_lora_dict[y_lora_id])
output_dir = "output_dir/gradio_output"

ToTensor = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

# generated_masks = None

def mask_gen(depth_image, raw_image, depth_mask, image_mask, height, width, mask_selection, image_or_depth, enforce_complementary_mask,enforce_identical_mask, patch_size, gaussian_thresh):
    print("Generate Mask")

    height, width, patch_size = int(height), int(width), int(patch_size)
    batch_size = 2
    # n_pairs = batch_size // 2

    anchor_id = 1 if image_or_depth == "depth" else 0
    image_or_depth = "both" if (enforce_complementary_mask or enforce_identical_mask) else image_or_depth

    empty_mask = torch.empty([batch_size, 1, height, width], dtype = torch.float32)

    # if mask_selection == "from sketch":
    #     # depth_masks = [depth_images]
    #     image_mask, depth_mask = Image.open(raw_image["mask"]), Image.open(depth_image["mask"])
    #     # masks = [raw_image["mask"], depth_image["mask"]]
    #     masks = [image_mask, depth_mask]

    #     # masks = [raw_image["layer"][0], depth_image["layer"][0]]
    #     masks = process_frames(masks, height, width, verbose = True, div = 64)

    #     # depth_mask = ToTensor(depth_image["mask"])
    #     # image_mask = ToTensor(raw_image["mask"])
    #     masks = ToTensor(masks)
    #     masks = torch.stack(masks, dim = 0)
    if image_mask is None:
        image_mask = Image.new('L', (width, height), (255))
    else:
        image_mask = Image.open(image_mask).convert("L")
    if depth_mask is None:
        depth_mask = Image.new('L', (width, height), (255))
    else:
        depth_mask = Image.open(depth_mask).convert("L")
    cur_masks = [image_mask, depth_mask]
    cur_masks = [process_frames([mask], height, width, verbose = True, div = 64)[0] for mask in cur_masks]
    cur_masks = ToTensor(cur_masks)
    cur_masks = torch.stack(cur_masks, dim = 0)

    if mask_selection == "keep mask":
        masks = cur_masks
    elif mask_selection == "mask all":
        masks = torch.ones_like(empty_mask)
    elif mask_selection == "mask none":
        masks = torch.zeros_like(empty_mask)
    elif mask_selection == "mask half":
        masks = torch.zeros_like(empty_mask)
        masks[...,:width // 2] = 1.0
    elif mask_selection == "random mask":
        masks = get_rand_masks(batch_size, (max(height, width) // patch_size) + 1, noise_patch_size = int(patch_size), thresh = gaussian_thresh)
        masks = masks[...,:height,:width]

    if image_or_depth == "depth":
        masks[0] = cur_masks[0]
    elif image_or_depth == "image":
        masks[1] = cur_masks[1]
    
    if enforce_complementary_mask:
        masks[1 - anchor_id] = 1 - masks[anchor_id]
    if enforce_identical_mask:
        masks[1 - anchor_id] = masks[anchor_id]

    mask_images =  [T.ToPILImage()(mask) for mask in masks]

    # # rand_mask = [get_guassian_2d_rand_mask((max(height, width) // 8 // patch_size) + 1, patch_size, thresh = guassian_thresh) for i in range(batch_size)]
    # rand_mask = get_rand_masks(batch_size, (max(height, width) // patch_size) + 1, noise_patch_size = patch_size, thresh = gaussian_thresh)
    # # rand_mask = torch.stack(rand_mask, dim = 0)
    # rand_mask = rand_mask[...,:height,:width]
    # # .expand(batch_size, 3, height // 8, width // 8).to(torch.float32)
    # if mask_selection != "upload_mask":
    #     if mask_selection == "image mask":
    #         rand_mask[:n_pairs] = 1
    #         rand_mask[n_pairs:] = 0
    #     elif mask_selection == "depth mask":
    #         rand_mask[:n_pairs] = 0
    #         rand_mask[n_pairs:] = 1
    #     elif mask_selection == "half mask":
    #         rand_mask[:,:,:,rand_mask.shape[-1] // 2:] = 0.0
    #         rand_mask[:,:,:,:rand_mask.shape[-1] // 2] = 1.0

    #     if "complementary" in mask_selection:
    #         rand_mask[n_pairs:] = 1 - rand_mask[:n_pairs]
        
    #     if "identical" in mask_selection:
    #         rand_mask[n_pairs:] = rand_mask[:n_pairs]
            
    #     mask_images =  [T.ToPILImage()(mask) for mask in rand_mask]
    # else:
    #     mask_images = [mask_image] * batch_size
    # global generated_masks
    # generated_masks = mask_images
    return mask_images


def process(depth_image, raw_image, image_mask, depth_mask, prompt, additional_prompt, negative_prompt, use_blip_cap, height, width, seed, inference_step, batch_size, model_selection, scheduler_selection, y_lora_selection, strength, guidance_scale, enable_joint_diffusion, joint_scale, skip_encoder):
    global model_id, y_lora_id
    if y_lora_selection == "default":
        y_lora_selection = checkpoint_default_ylora_map[model_selection]
    if model_id != model_selection or y_lora_id != y_lora_selection:
        global pipeline
        model_id = model_selection
        y_lora_id = y_lora_selection
        pipeline = load_joint_diffusion_pipeline(base_model_id, checkpoint_dict[model_id], StableDiffusionInpaintPipeline, y_lora_path = y_lora_dict[y_lora_id])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Generate Image")
    
    # depth_image = Image.open(depth_image["image"])
    # raw_image = Image.open(raw_image["image"])
    depth_image = Image.open(depth_image)
    raw_image = Image.open(raw_image)
    if image_mask is None:
        image_mask = Image.new('RGB', (width, height), (255, 255, 255))
    else:
        image_mask = Image.open(image_mask)
    if depth_mask is None:
        depth_mask = Image.new('RGB', (width, height), (255, 255, 255))
    else:
        depth_mask = Image.open(depth_mask)
    
    # print(depth_image.mode)
    with torch.no_grad():
        # if depth_image.mode == "I":
        #     depth_tensor = T.ToTensor()(depth_image)
        #     depth_tensor = 1 - (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
        #     depth_image = T.ToPILImage()(depth_tensor)
        print(depth_image, raw_image)
        # depth_image, raw_image = depth_image["background"].convert("RGB"), raw_image["background"].convert("RGB")
        depth_image, raw_image = depth_image.convert("RGB"), raw_image.convert("RGB")

        if use_blip_cap:
            prompt = blip_cap([raw_image], blip_processor, blip_model)[0]

        prompt = additional_prompt + prompt
        # print(prompt)
        # control_frames = process_frames(control_frames, h = height, w = width, verbose = True, div = 64)

        

        init_images = [raw_image,  depth_image]
        init_images = [process_frames([init_image], height, width,  verbose = True, div = 64)[0] for init_image in init_images]

        n_pairs = int(batch_size // 2)
        init_images = [init_images[0]] * n_pairs + [init_images[1]] * n_pairs

        # mask_images = mask_gen(batch_size, height, width, patch_size, gaussian_thresh, mask_selection)
        # mask_images = generated_masks
        mask_images = [image_mask] * n_pairs + [depth_mask] * n_pairs


        

        patch.set_joint_attention(pipeline.unet, enable = enable_joint_diffusion)
        patch.set_joint_scale(pipeline.unet, scale = joint_scale)

        if skip_encoder:
            patch.set_joint_attention(pipeline.unet, enable = False, name_filter = "down_blocks")

        if guidance_scale <= 1.0:
            patch.set_patch_lora_mask(pipeline.unet, "y_lora", [0,1])
            patch.set_patch_lora_mask(pipeline.unet, "yx_lora", [0,1])
            patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,0])
        else:
            patch.set_patch_lora_mask(pipeline.unet, "y_lora", [0,1,0,1])
            patch.set_patch_lora_mask(pipeline.unet, "yx_lora", [0,1,0,1])
            patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,0,1,0])

        load_scheduler(pipeline, mode = scheduler_selection)
        # print(pipeline.scheduler)

        generator = torch.Generator(device="cuda")
        generator = generator.manual_seed(seed)

        if negative_prompt is not None:
            negative_prompts = [negative_prompt] * batch_size

        trigger_word = trigger_word_map[model_id]
        prompts = [prompt if i < (batch_size // 2) else trigger_word + prompt for i in range(batch_size)]
        print(prompts)
        with torch.autocast(device_type = "cuda", dtype = torch.float16):
            images = pipeline(
                prompts,
                image = init_images,
                mask_image = mask_images,
                height = height,
                width = width,
                num_inference_steps=inference_step,
                guidance_scale = guidance_scale,
                strength = strength,
                generator=generator,
                negative_prompt = negative_prompts
            ).images
        
        mask_images_upscale = [process_frames([mask_image], height, width)[0] for mask_image in mask_images]
        images_with_mask = []
        for im, miu in zip(images, mask_images_upscale):
            # Create a mask with transparency
            inverted_mask = ImageOps.invert(miu.convert("L"))
            mask_with_alpha = Image.new('RGBA', inverted_mask.size)
            mask_with_alpha.paste((255, 0, 0, 128), (0, 0), inverted_mask)  # Red color with 50% opacity

            # Composite the overlay with the original image
            image_with_mask = Image.alpha_composite(im.convert('RGBA'), mask_with_alpha)
            images_with_mask.append(image_with_mask)

        image_grid = make_image_grid([*init_images, *mask_images_upscale, *images, *images_with_mask], rows=4, cols=batch_size)
    return image_grid


demo = gr.Blocks()
with demo:
    with gr.Row():
        gr.Markdown("## Joint Depth-Image Generation")
    gr.Markdown("""
Need an image and a depth image. The depth should be visualized as RGB image. For example you can extract the RGB depth image by [Depth-Anything-V2](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2).
    
Then you can upload mask images or generate random mask for image and depth by \"Get Mask\". Change the mask generation type in Mask options. 
""")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                depth_image = gr.Image(type='filepath', label='depth image')
                raw_image = gr.Image(type='filepath', label='input image')
            with gr.Row():
                depth_mask = gr.Image(type='filepath', label='depth mask', value = None)
                image_mask = gr.Image(type='filepath', label='image mask', value = None)
            prompt = gr.Textbox(label="Prompt")
            with gr.Accordion("Advanced options", open=False):
                additional_prompt = gr.Textbox(label="Additional prompt", value = "best quality, extremely detailed, ")
                negative_prompt = gr.Textbox(label="Negative prompt", value = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
                use_blip_cap = gr.Checkbox(label='BLIP caption', value=False)
                enable_joint_diffusion = gr.Checkbox(label='Enable Joint Diffusion', value=True)
                batch_size = gr.Number(label="Batch size", value=2, precision=0)
                height = gr.Slider(label="Height", minimum=128, maximum=1024, value=512, step=64)
                width = gr.Slider(label="Width ", minimum=128, maximum=1024, value=512, step=64)
                inference_step = gr.Slider(label="Inference Step", minimum=1, maximum=100, value=50, step=1)
                # seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)\8621329
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=8621329)
                model_selection = gr.Dropdown(choices =list(checkpoint_dict.keys()), label="Model Selection", value=model_id)
                scheduler_selection = gr.Dropdown(choices= ["ddim", "ddpm", "pndm", "ead"], label="Scheduler Selection", value="ead")
                y_lora_selection = gr.Dropdown(choices =list(y_lora_dict.keys()) + ["default"], label="Y LoRA Selection", value="default")
                strength = gr.Number(label="Inpaint strength", value=1.0)
                guidance_scale = gr.Number(label="Guidance sacle", value=7.5)
                skip_encoder = gr.Checkbox(label="Skip encoder")
            with gr.Accordion("Mask options", open=False):
                # base = gr.Image(label="Background", source="upload", tool="sketch", type="pil", height=512, brush_color='#FFFFFF', mask_opacity=0.5, brush_radius = 100)
                # mask_selection = gr.Dropdown(choices = ["from sketch", "mask all", "random mask", "mask half", "mask none"], label="Mask choices", value="from sketch")
                mask_selection = gr.Dropdown(choices = ["mask all", "random mask", "mask half", "mask none", "keep mask"], label="Mask choices", value="random mask")
                image_or_depth = gr.Dropdown(choices = ["both", "image", "depth"], label="Generate mask for which", value="both")
                enforce_complementary_mask = gr.Checkbox(label="Complementary mask")
                enforce_identical_mask = gr.Checkbox(label="Identical mask")

                patch_size = gr.Number(label="Mask patch size for rand mask", value=1)
                gaussian_thresh = gr.Number(label="2D guaussian threshold for rand mask", value=0)
                joint_scale = gr.Slider(label="Joint scale", minimum=0, maximum = 2, value = 1, step = 0.1)
            with gr.Row():
                mask_gen_button = gr.Button(value="Get Mask")
                run_button = gr.Button()
           
        with gr.Column():
            # result_video = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            # result_image = gr.Image(type='pil', label='result image', format="png")
            result_image = gr.Image(type='pil', label='result image', format="png")
    
    ips = [depth_image, raw_image, image_mask, depth_mask, prompt,additional_prompt,negative_prompt, use_blip_cap, height, width, seed, inference_step, batch_size, model_selection,scheduler_selection, y_lora_selection, strength, guidance_scale, enable_joint_diffusion, joint_scale, skip_encoder]
    run_button.click(fn=process, inputs=ips, outputs=[result_image])
    mask_gen_button.click(fn=mask_gen, inputs = [depth_image, raw_image, depth_mask, image_mask, height, width, mask_selection, image_or_depth, enforce_complementary_mask, enforce_identical_mask, patch_size, gaussian_thresh], outputs = [image_mask, depth_mask])


demo.launch(allowed_paths=["."], server_name='0.0.0.0', share=True)