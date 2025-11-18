import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
import torch.distributed as dist
from accelerate.logging import get_logger
from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange
from diffusers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
import json
import glob
from datasets import load_dataset

logger = get_logger(__name__, log_level="INFO")

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents

from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image, convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from peft import LoraConfig
from utils.peft_utils import get_peft_model_state_dict, set_peft_model_state_dict
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline
# logger = get_logger(__name__, log_level="INFO")
def load_lora_weights(unet, lora_path, adapter_name = "optical_flow_lora", logger = None):
    state_dict, network_alphas = StableDiffusionPipeline.lora_state_dict(lora_path)

    # unet.delete_adapters("optical_flow_lora")

    cls = StableDiffusionPipeline
    keys = list(state_dict.keys())
    if all(key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name) for key in keys):
        # Load the layers corresponding to UNet.
        if logger is not None:
            logger.info(f"Loading {cls.unet_name}.")

        unet_keys = [k for k in keys if k.startswith(cls.unet_name)]
        state_dict = {k.replace(f"{cls.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

        if network_alphas is not None:
            alpha_keys = [k for k in network_alphas.keys() if k.startswith(cls.unet_name)]
            network_alphas = {
                k.replace(f"{cls.unet_name}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
            }
        state_dict = convert_unet_state_dict_to_peft(state_dict)
        if network_alphas is not None:
            # The alphas state dict have the same structure as Unet, thus we convert it to peft format using
            # `convert_unet_state_dict_to_peft` method.
            network_alphas = convert_unet_state_dict_to_peft(network_alphas)
        incompatible_keys = set_peft_model_state_dict(unet, state_dict, adapter_name)
        # logger.warning(f"Incompatible keys in loading LoRA:", incompatible_keys)

# def load_weights(
#     animation_pipeline,
#     # motion module
#     motion_module_path         = "",
#     motion_module_lora_configs = [],
#     # image layers
#     dreambooth_model_path = "",
#     lora_model_path       = "",
#     lora_alpha            = 0.8,
# ):
#     # 1.1 motion module
#     unet_state_dict = {}
#     if motion_module_path != "":
#         print(f"load motion module from {motion_module_path}")
#         motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
#         motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
#         unet_state_dict.update({name.replace("module.", ""): param for name, param in motion_module_state_dict.items()})
    
#     missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
#     assert len(unexpected) == 0
#     del unet_state_dict

#    # if dreambooth_model_path != "":
#    #     print(f"load dreambooth model from {dreambooth_model_path}")
#   #      if dreambooth_model_path.endswith(".safetensors"):
#   #          dreambooth_state_dict = {}
#   #          with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
#    #             for key in f.keys():
#    #                 dreambooth_state_dict[key.replace("module.", "")] = f.get_tensor(key)
#    #     elif dreambooth_model_path.endswith(".ckpt"):
#    #         dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
#    #         dreambooth_state_dict = {k.replace("module.", ""): v for k, v in dreambooth_state_dict.items()}
            
#         # 1. vae
#     #    converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, animation_pipeline.vae.config)
#     #    animation_pipeline.vae.load_state_dict(converted_vae_checkpoint)
#         # 2. unet
#     #    converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, animation_pipeline.unet.config)
#     #    animation_pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
#         # 3. text_model
#      #   animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
#      #   del dreambooth_state_dict
        
#     if lora_model_path != "":
#         print(f"load lora model from {lora_model_path}")
#         assert lora_model_path.endswith(".safetensors")
#         lora_state_dict = {}
#         with safe_open(lora_model_path, framework="pt", device="cpu") as f:
#             for key in f.keys():
#                 lora_state_dict[key.replace("module.", "")] = f.get_tensor(key)
                
#         animation_pipeline = convert_lora(animation_pipeline, lora_state_dict, alpha=lora_alpha)
#         del lora_state_dict

#     for motion_module_lora_config in motion_module_lora_configs:
#         path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
#         print(f"load motion LoRA from {path}")

#         motion_lora_state_dict = torch.load(path, map_location="cpu")
#         motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
#         motion_lora_state_dict = {k.replace("module.", ""): v for k, v in motion_lora_state_dict.items()}

#         animation_pipeline = convert_motion_lora_ckpt_to_diffusers(animation_pipeline, motion_lora_state_dict, alpha)

#     return animation_pipeline

from PIL import Image
import torchvision.transforms as T
def load_condition_latents(vae, cond_image_path, resolution = None):
    # cond_image_name = lines[cond_id]["original_image"] if cond_x else lines[cond_id]["file_name"]
    # prompt = lines[cond_id]["text"][0]

    # depth_path = "output/joint_depth_0.png"
    print(f"Load cond image from {cond_image_path}")
    cond_image = Image.open(cond_image_path).convert("RGB")

    if isinstance(resolution, int):
        resolution = [resolution, resolution]
    height, width = (cond_image.height, cond_image.width) if resolution is None else resolution
    height, width = height // 8 * 8, width // 8 * 8
    # height, width = 512, 512

    # cond_image = T.Resize((height, width))(cond_image)
    cond_image = T.Resize((height))(cond_image)
    # cond_image = T.FiveCrop(512)(cond_image)[0]
    cond_image = T.CenterCrop((height, width))(cond_image)

    cond_image = T.ToTensor()(cond_image).to("cuda").to(torch.float16).unsqueeze(0)
    height, width = cond_image.shape[-2:]

    cond_image = (cond_image - 0.5) / 0.5
    cond_latent = vae.encode(cond_image).latent_dist.sample()
    cond_latent = cond_latent * vae.config.scaling_factor

    return cond_latent

def normalize_image(image):
    return (image - 0.5) / 0.5

def denormalize_image(image):
    return (image + 1) / 2

@torch.no_grad()
def tensor_to_vae_latent(t, vae, normalize = False):
    video_length = t.shape[1]
    if normalize:
        assert t.min() >= 0 and t.max() <= 1, f"Input tensor should be normalized to [0, 1]. Tensor min {t.min()}, tensor max {t.max()}"
        t = t * 2.0 - 1.0
    else:
        assert t.min() >= -1 and t.max() <= 1, f"Input tensor should be in range [-1, 1]. Tensor min {t.min()}, tensor max {t.max()}"
        if t.min() >= 0:
            logger.warning(f"Input tensor may not be normalized to [-1, 1], please make sure the input tensor is normalized to [-1, 1]. Tensor min {t.min()}, tensor max {t.max()}")
    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents

def _get_add_time_ids(
        fps,
        motion_bucket_ids,  # Expecting a list of tensor floats
        noise_aug_strength,
        dtype,
        batch_size,
        unet=None,
        device=None,  # Add a device parameter
    ):
    # Determine the target device
    target_device = device if device is not None else 'cpu'

    # Ensure motion_bucket_ids is a tensor and on the target device
    if not isinstance(motion_bucket_ids, torch.Tensor):
        motion_bucket_ids = torch.tensor(motion_bucket_ids, dtype=dtype, device=target_device)
    else:
        motion_bucket_ids = motion_bucket_ids.to(device=target_device)
        

    # Reshape motion_bucket_ids if necessary

    motion_bucket_ids = motion_bucket_ids.view(-1, 1)

    # Check for batch size consistency
    if motion_bucket_ids.size(0) != batch_size:
        motion_bucket_ids = motion_bucket_ids.expand(batch_size, -1)
    
    if not isinstance(fps, torch.Tensor):
        fps = torch.tensor(fps, dtype=dtype, device=target_device)
    else:
        fps = fps.to(device=target_device)

    # Reshape motion_bucket_ids if necessary

    fps = fps.view(-1, 1)

    # Check for batch size consistency
    if fps.size(0) != batch_size:
        fps = fps.expand(batch_size, -1)


    # Create fps and noise_aug_strength tensors on the target device
    noise_aug_strength = torch.tensor([noise_aug_strength], dtype=dtype, device=target_device).repeat(batch_size, 1)

    # Concatenate with motion_bucket_ids
    add_time_ids = torch.cat([fps, noise_aug_strength, motion_bucket_ids], dim=1)

    # Checking the dimensions of the added time embedding
    # passed_add_embed_dim = unet.config.addition_time_embed_dim * add_time_ids.size(1)
    # expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    # if expected_add_embed_dim != passed_add_embed_dim:
    #     raise ValueError(
    #         f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
    #         f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. "
    #         "Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
    #     )

    return add_time_ids

from transformers import BlipProcessor, BlipForConditionalGeneration
@torch.no_grad()
def blip_cap(frames, processor = None, model = None):
    processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large") if processor is None else processor
    model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda") if model is None else model
    if not isinstance(frames[0], Image.Image):
        raw_images = [T.ToPILImage()(frame) for frame in frames]
    else:
        raw_images = frames
    inputs = processor(raw_images, return_tensors="pt").to(
        "cuda", torch.float16)

    out = model.generate(**inputs)

    inversion_prompt = [processor.decode(
        outi, skip_special_tokens=True) for outi in out]
    # print("Blip Caption", inversion_prompt)
    return inversion_prompt

import cv2
import re
def load_images_from_folder_to_pil(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    def frame_number(filename):
        # Try the pattern 'frame_x_7fps'
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

    # Load, resize, and convert images
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image with original channels
            if img is not None:
                # Resize image
                # img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Convert to uint8 if necessary
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                # Ensure all images are in RGB format
                if len(img.shape) == 2:  # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert the numpy array to a PIL image
                pil_img = Image.fromarray(img)
                images.append(pil_img)

    return images

IMAGE_EXT = [".jpg", ".jpeg", ".png", "webp"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

from torchvision.io import read_video
from PIL import ImageSequence
def load_input(input_path):
    if os.path.isdir(input_path):
    # if "validation_image_folder" in args:
        validation_images = load_images_from_folder_to_pil(input_path)

    elif os.path.splitext(input_path)[1] in IMAGE_EXT:
        validation_image = Image.open(input_path).convert('RGB')
        validation_images = [validation_image]
    elif os.path.splitext(input_path)[1] in VIDEO_EXT:
        validation_images = read_video(input_path, pts_unit="sec")[0]
        validation_images = validation_images.permute(0, 3, 1, 2) / 255.
        validation_images = [T.ToPILImage()(frame) for frame in validation_images]
    elif os.path.splitext(input_path)[1] == ".gif":
        validation_images = Image.open(input_path)
        validation_images_ls = []
        for frame in ImageSequence.Iterator(validation_images):
            validation_images_ls += [frame.convert("RGB")]
        validation_images = validation_images_ls
    else:
        assert False, f"Unsupported file type or file not found: {os.path.splitext(input_path)[1]}"
    return validation_images

from patch import patch
def load_joint_lora_to_pipeline(pipeline, checkpoint_path, do_classifier_free_guidance = True):
    patch.apply_patch(pipeline.unet, flip=True)
    patch.initialize_joint_layers(pipeline.unet)

    y_lora_path = os.path.join(checkpoint_path, "y_lora")
    y_lora_state_dict, y_lora_network_alphas = StableDiffusionPipeline.lora_state_dict(y_lora_path)
    StableDiffusionPipeline.load_lora_into_unet(y_lora_state_dict, y_lora_network_alphas, unet = pipeline.unet, adapter_name = "y_lora")

    for lora_name in ["xy_lora", "yx_lora"]:
        save_dir = os.path.join(checkpoint_path, f"{lora_name}")
        lora_state_dict, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(save_dir)
        StableDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet = pipeline.unet, adapter_name = lora_name)

    state_dict = {}
    with safe_open(os.path.join(checkpoint_path, "model.safetensors"), framework="pt", device=0) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    pipeline.unet.load_state_dict(state_dict)

    active_adapters = ["xy_lora", "yx_lora", "y_lora"]

    pipeline.unet.set_adapters(active_adapters)
    patch.hack_lora_forward(pipeline.unet)
    if do_classifier_free_guidance:
        patch.set_patch_lora_mask(pipeline.unet, "y_lora", [0,1,0,1])
        patch.set_patch_lora_mask(pipeline.unet, "yx_lora", [0,1,0,1])
        patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,0,1,0])
    else:
        patch.set_patch_lora_mask(pipeline.unet, "y_lora", [0,1])
        patch.set_patch_lora_mask(pipeline.unet, "yx_lora", [0,1])
        patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,0])       


def get_points_on_a_grid(
    size,
    extent,
    center = None,
    device = torch.device("cpu"),
    margin = None,
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    # if size == 1:
    #     return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]
    if isinstance(size, int):
        size = (size, size)

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    if margin is None:
        margin = extent[0] // size[0] // 2
    # margin = 0
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size[0], device=device),
        torch.linspace(*range_x, size[1], device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)    

def get_track_queries(video, grid_query_frame = None, random_select = None, downscale_rate = 8, margin = None):
    batch_size, frame_num, channel, height, width = video.shape

    grid_pts = get_points_on_a_grid((height // downscale_rate, width // downscale_rate), (height, width), device=video.device, margin = margin)

    grid_query_frame = 0 if grid_query_frame is None else grid_query_frame

    queries = torch.cat(
        [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
        dim=2,
    ).repeat(batch_size, 1, 1)

    # print(f"Num query points : {queries.shape[1]}")
    if random_select is not None:
        rand_queries = torch.rand((queries.shape[1]), device=video.device) < random_select
        queries = queries[:,rand_queries,:]
    
    return queries

    # print(f"Num query points after rand choice : {queries.shape[1]}")

from pipeline.pipeline_stable_video_diffusion_trans import StableVideoDiffusionPipeline as StableVideoDiffusionPipelineTrans
from pipeline.pipeline_stable_video_diffusion_trans_controlnet import StableVideoDiffusionPipeline as StableVideoDiffusionPipelineTransControlNet

from models.controlnet_sdv import ControlNetSDVModel
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel

def load_trans_svd_pipeline(svd_path, checkpoint_path, single_lora = True):
    pipeline = StableVideoDiffusionPipelineTrans.from_pretrained(svd_path, torch_dtype=torch.float16, local_files_only = True)
    pipeline.enable_model_cpu_offload()
    # Additional pipeline configurations can be added here
    #pipeline.enable_xformers_memory_efficient_attention()
    # Create output directory if it doesn't exist

    output_path = checkpoint_path
    # output_path = "output_svdxt_translation_lora_noflip_temporal/checkpoint-9500"


    if "noflip" not in output_path and "flip" in output_path:
        flip = True
    else:
        flip = False

    if "notemporal" not in output_path and "temporal" in output_path:
        include_temporal = True
    else:
        include_temporal = False
    
    if "nospatial" in output_path:
        include_spatial = False
    else:
        include_spatial = True

    # flip = False
    # include_temporal = False

    print(f"flip {flip}", "temporal", include_temporal, "spatial", include_spatial)
    patch.apply_patch(pipeline.unet, flip=flip, with_temporal_block=include_temporal, with_spatial_block=include_spatial)
    patch.initialize_joint_layers(pipeline.unet)


    # y_lora_path = os.path.join(output_path, "y_lora")
    # y_lora_state_dict, y_lora_network_alphas = StableDiffusionPipeline.lora_state_dict(y_lora_path)
    # StableDiffusionPipeline.load_lora_into_unet(y_lora_state_dict, y_lora_network_alphas, unet = pipeline.unet, adapter_name = "y_lora")

    for lora_name in ["y_lora", "xy_lora", "yx_lora"]:
        save_dir = os.path.join(output_path, f"{lora_name}")
        if os.path.exists(save_dir):
            lora_state_dict, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(save_dir)
            StableDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet = pipeline.unet, adapter_name = lora_name)
        # pipeline.load_lora_weights(save_dir, adapter_name=lora_name)

    # state_dict = torch.load(os.path.join(output_path, "model.safetensors"), map_location="cpu")

    # rec_txt1 = open('output_lora_joint_depth_image_clean_cond1.txt', 'w')

    # for name, para in pipeline.unet.named_parameters():
    #     rec_txt1.write(f'{name}\n')

    # rec_txt1.close()

    if os.path.exists(os.path.join(output_path, "model.pth")):
        state_dict = torch.load(os.path.join(output_path, "model.pth"), map_location="cpu")
        pipeline.unet.load_state_dict(state_dict, strict=False)
    else:
        state_dict = {}
        with safe_open(os.path.join(output_path, "model.safetensors"), framework="pt", device=0) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        pipeline.unet.load_state_dict(state_dict)

    active_adapters = ["xy_lora", "yx_lora", "y_lora"]

    pipeline.unet.set_adapters(active_adapters)
    patch.hack_lora_forward(pipeline.unet)
    # patch.set_patch_lora_mask(pipeline.unet, "y_lora", [0,1,0,1])

    if single_lora:
        patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,1,1,1])
    else:
        patch.set_patch_lora_mask(pipeline.unet, "yx_lora", [0,1,0,1])
        patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,0,1,0])
    patch.set_joint_attention_mask(pipeline.unet, [0,1,0,1])

    return pipeline

from safetensors.torch import load_file
def load_controlnet_trans_svd_pipeline(svd_path, checkpoint_path, controlnet_path, single_lora = True):
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(svd_path,subfolder="unet")
    # controlnet = ControlNetSDVModel.from_pretrained(controlnet_path, subfolder="controlnet",
    #                                                 cache_dir="/data/vjuicefs_ai_camera_lgroup/11162591/gsy_workdir/hf_models_backup/")
    # controlnet = ControlNetSDVModel.from_pretrained(controlnet_path)

    controlnet = ControlNetSDVModel.from_unet(unet, conditioning_channels=2)
    state_dict = load_file(controlnet_path)
    controlnet.load_state_dict(state_dict)

    pipeline = StableVideoDiffusionPipelineTransControlNet.from_pretrained(svd_path, controlnet = controlnet, unet = unet, torch_dtype=torch.float16, local_files_only = True)
    pipeline.enable_model_cpu_offload()
    # Additional pipeline configurations can be added here
    #pipeline.enable_xformers_memory_efficient_attention()
    # Create output directory if it doesn't exist

    output_path = checkpoint_path
    # output_path = "output_svdxt_translation_lora_noflip_temporal/checkpoint-9500"


    if "noflip" not in output_path and "flip" in output_path:
        flip = True
    else:
        flip = False

    if "notemporal" not in output_path and "temporal" in output_path:
        include_temporal = True
    else:
        include_temporal = False
    
    if "nospatial" in output_path:
        include_spatial = False
    else:
        include_spatial = True

    # flip = False
    # include_temporal = False

    print(f"flip {flip}", "temporal", include_temporal, "spatial", include_spatial)
    patch.apply_patch(pipeline.unet, flip=flip, with_temporal_block=include_temporal, with_spatial_block=include_spatial)
    patch.initialize_joint_layers(pipeline.unet)


    # y_lora_path = os.path.join(output_path, "y_lora")
    # y_lora_state_dict, y_lora_network_alphas = StableDiffusionPipeline.lora_state_dict(y_lora_path)
    # StableDiffusionPipeline.load_lora_into_unet(y_lora_state_dict, y_lora_network_alphas, unet = pipeline.unet, adapter_name = "y_lora")

    for lora_name in ["y_lora", "xy_lora", "yx_lora"]:
        save_dir = os.path.join(output_path, f"{lora_name}")
        if os.path.exists(save_dir):
            lora_state_dict, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(save_dir)
            StableDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet = pipeline.unet, adapter_name = lora_name)
        # pipeline.load_lora_weights(save_dir, adapter_name=lora_name)

    # state_dict = torch.load(os.path.join(output_path, "model.safetensors"), map_location="cpu")

    # rec_txt1 = open('output_lora_joint_depth_image_clean_cond1.txt', 'w')

    # for name, para in pipeline.unet.named_parameters():
    #     rec_txt1.write(f'{name}\n')

    # rec_txt1.close()

    if os.path.exists(os.path.join(output_path, "model.pth")):
        state_dict = torch.load(os.path.join(output_path, "model.pth"), map_location="cpu")
        pipeline.unet.load_state_dict(state_dict, strict=False)
    else:
        state_dict = {}
        with safe_open(os.path.join(output_path, "model.safetensors"), framework="pt", device=0) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        pipeline.unet.load_state_dict(state_dict)

    active_adapters = ["xy_lora", "yx_lora", "y_lora"]

    pipeline.unet.set_adapters(active_adapters)
    patch.hack_lora_forward(pipeline.unet)
    # patch.set_patch_lora_mask(pipeline.unet, "y_lora", [0,1,0,1])

    if single_lora:
        patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,1,1,1])
    else:
        patch.set_patch_lora_mask(pipeline.unet, "yx_lora", [0,1,0,1])
        patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,0,1,0])
    patch.set_joint_attention_mask(pipeline.unet, [1,0,1,0])

    return pipeline

from safetensors.torch import load_file
def load_joint_diffusion_pipeline(base_model_id, checkpoint_path, pipeline_class, y_lora_path = None, skip_encoder = False, post_joint = "conv", add_norm = False):
    print(f"Base model: {base_model_id}")
    pipeline = pipeline_class.from_pretrained(base_model_id, safety_checker=None, torch_dtype=torch.float16).to("cuda")

    output_path = checkpoint_path

    if skip_encoder:
        name_skip = "down_blocks"
    else:
        name_skip = None
    patch.apply_patch(pipeline.unet, name_skip = name_skip)
    
    patch.initialize_joint_layers(pipeline.unet, post = post_joint, add_norm = add_norm)

    if y_lora_path is None:
        y_lora_path = os.path.join(output_path, "y_lora")
    print(f"Y LoRA path: {y_lora_path}")
    if os.path.exists(y_lora_path):
        lora_state_dict, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(y_lora_path)
        StableDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet = pipeline.unet, adapter_name = "y_lora")

    print(f"Joint LoRA path: {output_path}")
    for lora_name in ["xy_lora", "yx_lora"]:
        save_dir = os.path.join(output_path, f"{lora_name}")
        if os.path.exists(save_dir):
            lora_state_dict, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(save_dir)
            StableDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet = pipeline.unet, adapter_name = lora_name)

    if os.path.exists(os.path.join(output_path, "model.pth")):
        state_dict = torch.load(os.path.join(output_path, "model.pth"), map_location="cpu")
        # print(state_dict.keys())
        state_dict = {key: value for key, value in state_dict.items() if 'lora' not in key}
        pipeline.unet.load_state_dict(state_dict, strict=False)
    else:
        state_dict = {}
        with safe_open(os.path.join(output_path, "model.safetensors"), framework="pt", device=0) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        pipeline.unet.load_state_dict(state_dict)

    active_adapters = ["xy_lora", "yx_lora", "y_lora"]

    pipeline.unet.set_adapters(active_adapters)
    patch.hack_lora_forward(pipeline.unet)
    # patch.set_patch_lora_mask(pipeline.unet, "y_lora", [0,1,0,1])

    patch.set_patch_lora_mask(pipeline.unet, "y_lora", [0,1,0,1])
    patch.set_patch_lora_mask(pipeline.unet, "yx_lora", [0,1,0,1])
    patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,0,1,0])

    pipeline = pipeline.to("cuda").to(torch.float16)
    return pipeline

from models.controlnet import ControlNetModel
from diffusers import UNet2DConditionModel
def load_controlnet_inpaint_pipeline(base_model_id, checkpoint_path, pipeline_class, y_lora_path = None, skip_encoder = False, post_joint = "conv", conditioning_channels = 3):

    # controlnet_class = ControlNetModel
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", safety_checker=None, torch_dtype=torch.float16).to("cuda")
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels = conditioning_channels).to("cuda")
    pipeline = pipeline_class.from_pretrained(base_model_id, unet = unet, controlnet = controlnet, safety_checker=None, torch_dtype=torch.float16).to("cuda")

    output_path = checkpoint_path

    if y_lora_path is None:
        y_lora_path = os.path.join(output_path, "y_lora")
    print(f"Y LoRA path: {y_lora_path}")
    if os.path.exists(y_lora_path):
        lora_state_dict, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(y_lora_path)
        StableDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet = pipeline.controlnet, adapter_name = "y_lora")

        active_adapters = ["y_lora"]

        pipeline.unet.set_adapters(active_adapters)

    if os.path.exists(os.path.join(output_path, "model.pth")):
        state_dict = torch.load(os.path.join(output_path, "model.pth"), map_location="cpu")
        # print(state_dict.keys())
        state_dict = {key: value for key, value in state_dict.items() if 'lora' not in key}
        pipeline.controlnet.load_state_dict(state_dict, strict=False)
    else:
        state_dict = {}
        with safe_open(os.path.join(output_path, "model.safetensors"), framework="pt", device=0) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        pipeline.controlnet.load_state_dict(state_dict)

    pipeline = pipeline.to("cuda").to(torch.float16)
    return pipeline

import datetime
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

def load_scheduler(pipeline, mode="ddim"):
    if mode == "ddim":
        scheduler_cls = DDIMScheduler
    elif mode == "ddpm":
        scheduler_cls = DDPMScheduler
    elif mode == "pndm":
        scheduler_cls = PNDMScheduler
    elif mode == "ead":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif mode == "ed":
        scheduler_cls = EulerDiscreteScheduler
    elif mode == "dpm":
        scheduler_cls = DPMSolverMultistepScheduler
    pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

def load_image_folder(train_data_dir = None, cache_dir = None, dataset_type = None, rescale = None):
    data_files = {}

    if train_data_dir is not None:
        if dataset_type == "json":
            data_files["train"] = train_data_dir
        elif dataset_type == "imagefolder": 
            data_files["train"] = os.path.join(train_data_dir, "**")
        elif dataset_type == "parquet":
            data_files["train"] = glob.glob(os.path.join(train_data_dir, "*.parquet"))
        else:
            assert False, "Dataset type not defined"
        # elif dataset_type is None:


    dataset = load_dataset(
        dataset_type,
        data_files=data_files,
        cache_dir=cache_dir,
    )

    if rescale is not None:
        dataset = dataset["train"].train_test_split(train_size=rescale, shuffle=True)

    return dataset