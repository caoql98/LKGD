from PIL import Image
from diffusers import AutoPipelineForText2Image
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patch import patch

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import UNet2DConditionLoadersMixin, PeftAdapterMixin, FromSingleFileMixin
from pipeline.pipeline_stable_diffusion_joint_control import StableDiffusionPipelineJointControl
import torchvision.transforms as T
from safetensors import safe_open
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid
from utils.dataset import process_frames
from utils.gaussian_2d import get_guassian_2d_rand_mask
from utils.util import blip_cap
import random
import numpy as np

seed = 12345

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


cond_x = False

image_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/House.jpg"
depth_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/inputs/House_depth.png"
if cond_x:
    input_path = image_path
else:
    input_path = depth_path

pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                      safety_checker=None, 
                                                      torch_dtype=torch.float16).to("cuda")

# depth_lora_path = "/data/juicefs_sharing_data/11162591/code/lxr/svd-train/output_depth_lora/checkpoint-17000"
# pipeline.load_lora_weights(depth_lora_path, adapter_name="default")

checkpoint_dir = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora/checkpoint-5000"

if not cond_x:
    mask_depth_lora_path = os.path.join(checkpoint_dir, "y_lora")
    pipeline.load_lora_weights(mask_depth_lora_path, adapter_name = "mask_depth_lora")

# state_dict = {}
# with safe_open("/data/juicefs_sharing_data/11162591/code/lxr/svd-train/output_depth_lora_rank64/checkpoint-2500/model.safetensors", framework="pt", device=0) as f:
#     for k in f.keys():
#         state_dict[k] = f.get_tensor(k)
# pipeline.unet.load_state_dict(state_dict)

pipeline = pipeline.to("cuda").to(torch.float16)

meta_data_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/metadata.jsonl"
import json
with open(meta_data_path, "r") as f:
    lines = list(f)
    lines = [json.loads(line) for line in lines]


use_noise_lora = False

cond_id = 1428
cond_image_name = lines[cond_id]["original_image"] if cond_x else lines[cond_id]["file_name"]
prompt = lines[cond_id]["text"]

depth_root = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/readout_guidance/readout_training/data/pseudo_labels/PascalVOC/depth"
depth_path = os.path.join(depth_root, cond_image_name)

if input_path is not None:
    depth_path = input_path
    depth_image = Image.open(depth_path).convert("RGB")
    prompt = blip_cap([depth_image])[0]
else:
    depth_image = Image.open(depth_path).convert("RGB")

print(f"Load cond image from {depth_path}")

# depth_image = T.Resize(512)(depth_image)
# depth_image = T.CenterCrop(512)(depth_image)

# depth_image = T.ToTensor()(depth_image).to("cuda").to(torch.float16)
# height, width = depth_image.shape[-2:]

batch_size = 4
# depth_images = torch.stack([depth_image] * batch_size)

# depth_images = (depth_images - 0.5) / 0.5
# depth_latents = pipeline.vae.encode(depth_images).latent_dist.sample()
# depth_latents = depth_latents * pipeline.vae.config.scaling_factor
# depth_latents = None
print(prompt)

# generator = torch.Generator(device=depth_latents.device)
# generator = generator.manual_seed(123)

# channel = 4


# torch.random.manual_seed(1111)
# latents = torch.randn(batch_size // 2, channel, height // 8, width // 8).to("cuda").to(torch.float16)
# latents = torch.cat([latents] * 2)

# if use_noise_lora:
#     torch.manual_seed(123)
#     fix_noise = torch.randn(4, 512, 512)
#     condition_noise = fix_noise[None,...,:depth_latents.shape[-2],:depth_latents.shape[-1]].expand_as(depth_latents)
#     cond_mask = torch.zeros_like(condition_noise).to(torch.bool)
#     cond_mask[..., depth_latents.shape[-2] // 2:,:depth_latents.shape[-1]] = 1
#     # cond_mask = 1
# else:
# condition_noise = latents


height, width = 512, 512
init_image = depth_image
init_image = process_frames([init_image], height, width)[0]
# height, width = image.height, image.width
# rand_mask = get_guassian_2d_rand_mask(grid_size, noise_patch_size * 8)
# rand_mask = rand_mask[None,None,:height,:width].expand(batch_size, 1, height. width)
# init_image = load_image("https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
height, width = init_image.height, init_image.width

patch_size = 1

rand_mask = get_guassian_2d_rand_mask((max(height, width) // 8 // patch_size) + 1, patch_size, thresh = 0)
rand_mask = rand_mask[None,:height // 8,:width // 8].expand(3, height // 8, width // 8).to(torch.float32)
mask_image = T.ToPILImage()(rand_mask)

mask_image_resize = process_frames([mask_image], height, width)[0]
# mask_image = load_image("https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")
# generator = torch.Generator("cuda").manual_seed(92)
# prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
# prompt = args.validation_prompt

# for _ in range(args.num_validation_images):
images = pipeline(
        [prompt] * batch_size,
        image = [init_image] * batch_size,
        mask_image = [mask_image] * batch_size,
        height = height,
        width = width,
        # num_inference_steps=30, 
        # generator=generator
    ).images
image_grid = make_image_grid([init_image, mask_image_resize, *images], rows=1, cols=2 + len(images))

expnum = 0
save_path = os.path.join(checkpoint_dir, "test_results")
if not os.path.exists(save_path):
    os.mkdir(save_path)

image_grid.save(os.path.join(save_path, f"depth_test_{expnum}.png"))

