
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
from utils.util import blip_cap
import random
import numpy as np
from utils.gaussian_2d import get_rand_masks
import torchvision.transforms.v2 as transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
# from torchvision import
from gradio_config import checkpoint_dict, additional_kwargs_dict, trigger_word_map, checkpoint_default_ylora_map, y_lora_dict, base_model_id_map, base_model_dict
import pdb
# pdb.set_trace()




model_id = "id_lora_joint_celebawild_r64_nta"
y_lora_id = checkpoint_default_ylora_map[model_id]
base_model_id = base_model_id_map[model_id]
use_controlnet = lambda: "controlnet" in model_id
sym_lora = lambda: "sym" in model_id

example_dataset = [
    ["inputs/example_inputs/House.jpg", "inputs/example_inputs/House-hed.png"],
    ["inputs/example_inputs/House.jpg", "inputs/example_inputs/House-depth-color.png"],
    ["inputs/example_inputs/horse.jpeg", "inputs/example_inputs/horse_depth_anything_v2_color.webp"]
]

def load_pipeline():
    # print(additional_kwargs_dict[model_id])
    if use_controlnet():
        pipe = load_controlnet_inpaint_pipeline(base_model_dict[base_model_id], checkpoint_dict[model_id], StableDiffusionInpaintControlNetPipeline, y_lora_path = y_lora_dict[y_lora_id], **additional_kwargs_dict[model_id])
    else:
        pipe = load_joint_diffusion_pipeline(base_model_dict[base_model_id], checkpoint_dict[model_id], StableDiffusionInpaintPipeline, y_lora_path = y_lora_dict[y_lora_id], **additional_kwargs_dict[model_id])

    return pipe

pipe = load_pipeline()


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

from importlib import reload
def reload_pipeline(reload_module = False):
    global StableDiffusionInpaintControlNetPipeline, StableDiffusionInpaintPipeline
    if reload_module:
        if use_controlnet():
            reload(sys.modules['pipeline.pipeline_stable_diffusion_inpaint_controlnet'])
            from pipeline.pipeline_stable_diffusion_inpaint_controlnet import StableDiffusionInpaintControlNetPipeline
        else:
            reload(sys.modules['pipeline.pipeline_stable_diffusion_inpaint_guidance'])
            from pipeline.pipeline_stable_diffusion_inpaint_guidance import StableDiffusionInpaintPipeline
    global pipe
    pipe = load_pipeline()

def reload_pipeline_module(model_selection, y_lora_selection, base_model_selection):
    global model_id, y_lora_id, base_model_id
    if y_lora_selection == "default":
        y_lora_selection = checkpoint_default_ylora_map[model_selection]
    if base_model_selection == "default":
        base_model_selection = base_model_id_map[model_selection]

    if model_id != model_selection or y_lora_id != y_lora_selection or base_model_id != base_model_selection:
        model_id = model_selection
        y_lora_id = y_lora_selection
        base_model_id = base_model_selection
    reload_pipeline(reload_module=True)

def mask_gen(depth_image, raw_image, depth_mask, image_mask, height, width, mask_selection, image_or_depth, enforce_complementary_mask,enforce_identical_mask, patch_size, gaussian_thresh, max_value, min_value):
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
    elif mask_selection == "mask 1":
        masks = torch.ones_like(empty_mask)
    elif mask_selection == "mask 0":
        masks = torch.zeros_like(empty_mask)
    elif mask_selection == "half mask":
        masks = torch.zeros_like(empty_mask)
        masks[...,:width // 2] = 1.0
    elif mask_selection == "random mask":
        masks = get_rand_masks(batch_size, (max(height, width) // patch_size) + 1, noise_patch_size = int(patch_size), thresh = gaussian_thresh)
        masks = masks[...,:height,:width]
    

    min_value = min(min_value, max_value)
    masks = masks.clamp(min = min_value, max = max_value)

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

def process(depth_image, raw_image, image_mask, depth_mask, prompt, additional_prompt, negative_prompt, use_blip_cap, height, width, seed, inference_step, batch_size, model_selection, scheduler_selection, y_lora_selection, base_model_selection, strength, guidance_scale, enable_joint_diffusion, joint_scale, skip_encoder, with_guidance, guidance_lr, num_optimizer_steps, guidance_end, reg_weight, replace_latents, y_advance, eta, remove_mask, replace_end, fix_yt, y_prompt, blip_from_y):
    global model_id, y_lora_id, base_model_id
    if y_lora_selection == "default":
        y_lora_selection = checkpoint_default_ylora_map[model_selection]
    if base_model_selection == "default":
        base_model_selection = base_model_id_map[model_selection]

    if model_id != model_selection or y_lora_id != y_lora_selection or base_model_id != base_model_selection:
        model_id = model_selection
        y_lora_id = y_lora_selection
        base_model_id = base_model_selection

        reload_pipeline()

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
            if blip_from_y:
                y_prompt = blip_cap([depth_image], blip_processor, blip_model)[0]
            else:
                prompt = blip_cap([raw_image], blip_processor, blip_model)[0]

        prompt = additional_prompt + prompt
        if y_prompt == "None":
            y_prompt = prompt
        else:
            y_prompt = additional_prompt + y_prompt
        # print(prompt)
        # control_frames = process_frames(control_frames, h = height, w = width, verbose = True, div = 64)

        

        init_images = [raw_image,  depth_image]
        init_images = [process_frames([init_image], height, width,  verbose = True, div = 64)[0] for init_image in init_images]

        n_pairs = int(batch_size // 2)
        init_images = [init_images[0]] * n_pairs + [init_images[1]] * n_pairs

        # mask_images = mask_gen(batch_size, height, width, patch_size, gaussian_thresh, mask_selection)
        # mask_images = generated_masks
        mask_images = [image_mask] * n_pairs + [depth_mask] * n_pairs


        
        if not use_controlnet():
            patch.set_joint_attention(pipe.unet, enable = enable_joint_diffusion)
            patch.set_joint_scale(pipe.unet, scale = joint_scale)

            if skip_encoder:
                patch.set_joint_attention(pipe.unet, enable = False, name_filter = "down_blocks")

            
            set_joint_mask(pipe.unet, lora_names = ["y_lora", "yx_lora", "xy_lora"], symmetric = sym_lora(), do_guidance=guidance_scale > 1.0)

            # if guidance_scale <= 1.0:
            #     patch.set_patch_lora_mask(pipe.unet, "y_lora", [0,1])
            #     patch.set_patch_lora_mask(pipe.unet, "yx_lora", [0,1])
            #     patch.set_patch_lora_mask(pipe.unet, "xy_lora", [1,0])
            #     patch.set_joint_attention_mask(pipe.unet, [1,0])
            # else:
            #     patch.set_patch_lora_mask(pipe.unet, "y_lora", [0,1,0,1])
            #     patch.set_patch_lora_mask(pipe.unet, "yx_lora", [0,1,0,1])
            #     patch.set_patch_lora_mask(pipe.unet, "xy_lora", [1,0,1,0])
            #     patch.set_joint_attention_mask(pipe.unet, [1,0,1,0])

        load_scheduler(pipe, mode = scheduler_selection)
        # print(pipeline.scheduler)

        generator = torch.Generator(device="cuda")
        generator = generator.manual_seed(seed)

        if negative_prompt is not None:
            negative_prompts = [negative_prompt] * batch_size

        trigger_word = trigger_word_map[y_lora_id]
        if sym_lora():
            prompts = [trigger_word + prompt if i < (batch_size // 2) else trigger_word + y_prompt for i in range(batch_size)]
        else:
            prompts = [prompt if i < (batch_size // 2) else trigger_word + y_prompt for i in range(batch_size)]
        print(prompts)

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
                negative_prompt = negative_prompts,
                with_guidance = with_guidance,
                guidance_lr = guidance_lr, 
                num_optimizer_steps = num_optimizer_steps,
                guidance_end = guidance_end,
                replace_end = replace_end,
                reg_weight = reg_weight,
                replace_latents = replace_latents,
                y_advance = y_advance,
                eta = eta,
                remove_mask = remove_mask,
                fix_yt = fix_yt
            ).images
        
        if len(images) < len(prompts):
            images = images + init_images[batch_size // 2:]

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
                y_prompt = gr.Textbox(label="Prompt for y", value="None")
                use_blip_cap = gr.Checkbox(label='Use BLIP caption', value=False)
                blip_From_y = gr.Checkbox(label='BLIP caption from y', value=False)
                enable_joint_diffusion = gr.Checkbox(label='Enable Joint Diffusion', value=True)
                batch_size = gr.Number(label="Batch size", value=2, precision=0)
                height = gr.Slider(label="Height", minimum=128, maximum=1024, value=512, step=64)
                width = gr.Slider(label="Width ", minimum=128, maximum=1024, value=512, step=64)
                inference_step = gr.Slider(label="Inference Step", minimum=1, maximum=100, value=50, step=1)
                # seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)\8621329
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=8621329)
                model_selection = gr.Dropdown(choices =list(checkpoint_dict.keys()), label="Model Selection", value=model_id)
                scheduler_selection = gr.Dropdown(choices= ["ddim", "ddpm", "pndm", "ead", "ed", "dpm"], label="Scheduler Selection", value="ead")
                y_lora_selection = gr.Dropdown(choices =list(y_lora_dict.keys()) + ["default"], label="Y LoRA Selection", value="default")
                base_model_selection = gr.Dropdown(choices =list(base_model_dict.keys()) + ["default"], label="Base model Selection", value="default")
                strength = gr.Number(label="Inpaint strength", value=1.0)
                guidance_scale = gr.Number(label="Guidance sacle", value=7.5)
                skip_encoder = gr.Checkbox(label="Skip encoder")
                joint_scale = gr.Slider(label="Joint scale", minimum=0, maximum = 2, value = 1, step = 0.1)
                y_advance = gr.Slider(label="Y advance time", minimum=-1, maximum = 1, value = 0, step = 0.1)
                fix_yt = gr.Checkbox(label="Wether to fix yt")
                replace_latents = gr.Checkbox(label="Wether to enforce latents replacement")
                replace_end = gr.Number(label="Replace end", value=1.0)

            with gr.Accordion("Guidance options", open=False):
                with_guidance = gr.Checkbox(label="With guidance")
                guidance_lr = gr.Number(label="lr", value=4e-2)
                num_optimizer_steps = gr.Number(label="num_optimizer_steps", value=1)
                guidance_end = gr.Number(label="Guidance end", value=0.5)

                reg_weight = gr.Number(label="Regular loss weight", value=0.0)
                remove_mask = gr.Checkbox(label="Set input mask to 1 for inpainting model", value=True)
                eta = gr.Number(label="eta for DDIM Scheduler", value=1.0)


            with gr.Accordion("Mask options", open=False):
                # base = gr.Image(label="Background", source="upload", tool="sketch", type="pil", height=512, brush_color='#FFFFFF', mask_opacity=0.5, brush_radius = 100)
                # mask_selection = gr.Dropdown(choices = ["from sketch", "mask all", "random mask", "mask half", "mask none"], label="Mask choices", value="from sketch")
                mask_selection = gr.Dropdown(choices = ["mask 1", "mask 0", "random mask", "half mask", "keep mask"], label="Mask choices", value="random mask")
                image_or_depth = gr.Dropdown(choices = ["both", "image", "depth"], label="Generate mask for which", value="both")
                enforce_complementary_mask = gr.Checkbox(label="Complementary mask")
                enforce_identical_mask = gr.Checkbox(label="Identical mask")
                max_value = gr.Slider(label="Max value of mask", minimum=0, maximum=1, value=1, step=0.1)
                min_value = gr.Slider(label="Min value of mask", minimum=0, maximum=1, value=0, step=0.1)
                patch_size = gr.Number(label="Mask patch size for rand mask", value=1)
                gaussian_thresh = gr.Number(label="2D guaussian threshold for rand mask", value=0)
                
            with gr.Row():
                reload_pipeline_button = gr.Button(value = "Reload Pipeline")
                mask_gen_button = gr.Button(value="Get Mask")
                run_button = gr.Button()

            
           
        with gr.Column():
            # result_video = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            # result_image = gr.Image(type='pil', label='result image', format="png")
            result_image = gr.Image(type='pil', label='result image', format="png")
    
    with gr.Row():
        examples = gr.Examples(example_dataset, [raw_image, depth_image], label = "Image Examples")
    
    ips = [depth_image, raw_image, image_mask, depth_mask, prompt,additional_prompt,negative_prompt, use_blip_cap, height, width, seed, inference_step, batch_size, model_selection,scheduler_selection, y_lora_selection, base_model_selection, strength, guidance_scale, enable_joint_diffusion, joint_scale, skip_encoder, with_guidance, guidance_lr, num_optimizer_steps, guidance_end, reg_weight, replace_latents, y_advance, eta, remove_mask, replace_end, fix_yt, y_prompt, blip_From_y]
    run_button.click(fn=process, inputs=ips, outputs=[result_image])
    mask_gen_button.click(fn=mask_gen, inputs = [depth_image, raw_image, depth_mask, image_mask, height, width, mask_selection, image_or_depth, enforce_complementary_mask, enforce_identical_mask, patch_size, gaussian_thresh, max_value, min_value], outputs = [image_mask, depth_mask])
    reload_pipeline_button.click(fn=reload_pipeline_module, inputs=[model_selection, y_lora_selection, base_model_selection])


demo.launch(allowed_paths=["."], server_name='0.0.0.0', share=True)