from PIL import Image
from diffusers import AutoPipelineForText2Image
import torch
from patch import patch
import os
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import UNet2DConditionLoadersMixin, PeftAdapterMixin, FromSingleFileMixin
from pipeline.pipeline_stable_diffusion_joint_control import StableDiffusionPipelineJointControl
import torchvision.transforms as T
from safetensors import safe_open





pipeline = StableDiffusionPipelineJointControl.from_pretrained("/root/data/juicefs_sharing_data/11162591/code/models/runwayml/stable-diffusion-v1-5",
                                                      safety_checker=None, 
                                                      torch_dtype=torch.float16).to("cuda")

# depth_lora_path = "/data/juicefs_sharing_data/11162591/code/lxr/svd-train/output_depth_lora/checkpoint-17000"
# pipeline.load_lora_weights(depth_lora_path, adapter_name="default")

# noise_mask_lora_path = "/data/vjuicefs_ai_camera/11162591/public_datasets/share_files/noise_mask_lora_fix"
# pipeline.load_lora_weights(noise_mask_lora_path, adapter_name = "noise_mask_lora")

# state_dict = {}
# with safe_open("/data/juicefs_sharing_data/11162591/code/lxr/svd-train/output_depth_lora_rank64/checkpoint-2500/model.safetensors", framework="pt", device=0) as f:
#     for k in f.keys():
#         state_dict[k] = f.get_tensor(k)
# pipeline.unet.load_state_dict(state_dict)

pipeline = pipeline.to("cuda").to(torch.float16)

meta_data_path = "/data/juicefs_sharing_data/11162591/code/lxr/svd-train/data/depth/metadata.jsonl"
import json
with open(meta_data_path, "r") as f:
    lines = list(f)
    lines = [json.loads(line) for line in lines]

cond_x = True
use_noise_lora = True

cond_id = 12
cond_image_name = lines[cond_id]["original_image"] if cond_x else lines[cond_id]["file_name"]
prompt = lines[cond_id]["text"][0]

depth_root = "/data/juicefs_sharing_data/11162591/code/lxr/svd-train/data/depth"
depth_path = os.path.join(depth_root, cond_image_name)
print(f"Load cond image from {depth_path}")
depth_image = Image.open(depth_path).convert("RGB")
depth_image = T.Resize(512)(depth_image)
depth_image = T.CenterCrop(512)(depth_image)

depth_image = T.ToTensor()(depth_image).to("cuda").to(torch.float16)
height, width = depth_image.shape[-2:]

batch_size = 4
depth_images = torch.stack([depth_image] * batch_size)

depth_images = (depth_images - 0.5) / 0.5
depth_latents = pipeline.vae.encode(depth_images).latent_dist.sample()
depth_latents = depth_latents * pipeline.vae.config.scaling_factor
# depth_latents = None
print(prompt)

# generator = torch.Generator(device=depth_latents.device)
# generator = generator.manual_seed(123)

channel = 4
# torch.random.manual_seed(1111)
# latents = torch.randn(batch_size // 2, channel, height // 8, width // 8).to("cuda").to(torch.float16)
# latents = torch.cat([latents] * 2)
latents = None

if use_noise_lora:
    torch.manual_seed(123)
    noise_patch_size = 4
    latent_resolution = 512 // 8
    grid_size = latent_resolution // noise_patch_size + 1
    # fix_noise = torch.randn(4, noise_patch_size, noise_patch_size).repeat([1, grid_size, grid_size])
    fix_noise = torch.randn(4, latent_resolution, latent_resolution)



    # torch.manual_seed(123)
    # fix_noise = torch.randn(4, 512, 512)
    condition_noise = fix_noise[None,...,:depth_latents.shape[-2],:depth_latents.shape[-1]].expand_as(depth_latents)
    cond_mask = torch.zeros_like(condition_noise).to(torch.bool)
    cond_mask[..., depth_latents.shape[-2] // 2:,:depth_latents.shape[-1]] = 1
    # cond_mask = 1
else:
    condition_noise = latents

images = pipeline([prompt] * batch_size, 
                  height = height, 
                  width = width,
                  num_inference_steps=30,
                  cond_x = cond_x,
                  latents = latents,
                  condition_noise = condition_noise,
                  cond_mask = cond_mask,
                  condition_latents = depth_latents, 
                  ).images
for i, image in enumerate(images):
    image.save(f"output/joint_depth_{i}.png")