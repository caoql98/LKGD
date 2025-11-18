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

from utils.util import blip_cap



use_noise_lora = False

# model_id = "/root/data/juicefs_sharing_data/11162591/code/models/majicmixRealistic_v4/"
model_id = "/data/vjuicefs_ai_camera_lgroup/11162591/gsy_workdir/hf_models_backup/runwayml/stable-diffusion-v1-5"

pipeline = StableDiffusionPipelineJointControl.from_pretrained(model_id,
                                                      safety_checker=None, 
                                                      torch_dtype=torch.float16).to("cuda")

# images = pipeline(["three cars are parked on the side of the road"] * 1, 

#                   ).images
# for i, image in enumerate(images):
#     image.save(f"output/joint_depth_{i}.png")
# exit(0)
# rec_txt1 = open('output_lora_joint_depth_image_clean_cond0.txt', 'w')

# for name, para in pipeline.unet.named_parameters():
#     rec_txt1.write(f'{name}\n')

# rec_txt1.close()

patch.apply_patch(pipeline.unet)
patch.initialize_joint_layers(pipeline.unet)

# y_lora_path = "/data/juicefs_sharing_data/11162591/code/lxr/svd-train/output_depth_lora/pytorch_lora_weights.safetensors"
# pipeline.load_lora_weights(y_lora_path, adapter_name="y_lora")

# load attention processors
# output_path = "output_lora_joint_depth_image_joint_randt/checkpoint-10500"
# output_path = "output_lora_joint_depth_image_jointfix/checkpoint-5000"

# output_path = "/data/juicefs_sharing_data/11162591/code/lxr/svd-train/output_lora_joint_depth_image_clean_cond"
# output_path = "output_lora_joint_depth_image_clean_cond/checkpoint-5000"
output_path = "output_dir/output_lora_next_keyframe_v2/checkpoint-29000"
y_lora_path = os.path.join(output_path, "y_lora")
pipeline.load_lora_weights(y_lora_path, adapter_name="y_lora")
x_lora_path = os.path.join(output_path, "x_lora")
pipeline.load_lora_weights(x_lora_path, adapter_name="x_lora")
for lora_name in ["xy_lora", "yx_lora"]:
    save_dir = os.path.join(output_path, f"{lora_name}")
    # lora_state_dict, lora_network_alphas = StableDiffusionPipeline.lora_state_dict(save_dir)
    # StableDiffusionPipeline.load_lora_into_unet(lora_state_dict, lora_network_alphas, unet = pipeline.unet, adapter_name = lora_name)
    pipeline.load_lora_weights(save_dir, adapter_name=lora_name)

# state_dict = torch.load(os.path.join(output_path, "model.safetensors"), map_location="cpu")

# rec_txt1 = open('output_lora_joint_depth_image_clean_cond1.txt', 'w')

# for name, para in pipeline.unet.named_parameters():
#     rec_txt1.write(f'{name}\n')

# rec_txt1.close()

state_dict = {}
with safe_open(os.path.join(output_path, "model.safetensors"), framework="pt", device=0) as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)
pipeline.unet.load_state_dict(state_dict)

# state_dict_dir = os.path.join(output_path, "model.pth")
# state_dict = torch.load(state_dict_dir)
# pipeline.unet.load_state_dict(state_dict, strict = False)


# rec_txt1 = open('output_lora_joint_depth_image_clean_cond2.txt', 'w')

# for name, para in pipeline.unet.named_parameters():
#     rec_txt1.write(f'{name}\n')

# rec_txt1.close()



active_adapters = ["x_lora", "xy_lora", "yx_lora", "y_lora"]


pipeline.set_adapters(active_adapters)
patch.hack_lora_forward(pipeline.unet)
patch.set_patch_lora_mask(pipeline.unet, "y_lora", [0,1,0,1])
patch.set_patch_lora_mask(pipeline.unet, "x_lora", [1,0,1,0])
patch.set_patch_lora_mask(pipeline.unet, "yx_lora", [0,1,0,1])
patch.set_patch_lora_mask(pipeline.unet, "xy_lora", [1,0,1,0])
patch.set_joint_attention_mask(pipeline.unet, [0,1,0,1])
rec_txt1 = open('output_lora_joint_depth_image_clean_cond1.txt', 'w')

for name, para in pipeline.unet.named_parameters():
    rec_txt1.write(f'{name}\n')

rec_txt1.close()

channel = 4

# latents = torch.randn(batch, channel, height, width).to("cuda").to(torch.float16)
# latents[batch // 2:] = latents[:batch // 2]

pipeline = pipeline.to("cuda").to(torch.float16)

meta_data_path = "/data/juicefs_sharing_data/11162591/code/lxr/svd-train/data/depth/metadata.jsonl"
import json
with open(meta_data_path, "r") as f:
    lines = list(f)
    lines = [json.loads(line) for line in lines]




cond_x = False

if use_noise_lora:
    mask = [1,0,1,0] if cond_x else [0,1,0,1]
    patch.set_patch_lora_mask(pipeline.unet, "noise_lora", mask)


cond_image_path = ["data/test_images/video_trans_input/face_cut.png"]
cond_image_prompt = [None]
# cond_ids = [1, 1234, 1428] # 1, 1234, 1428
cond_ids = [0]
for cond_id in cond_ids:
    # cond_image_name = lines[cond_id]["original_image"] if cond_x else lines[cond_id]["file_name"]
    # prompt = lines[cond_id]["text"][0]

    # depth_root = "/data/juicefs_sharing_data/11162591/code/lxr/svd-train/data/depth"
    # depth_path = os.path.join(depth_root, cond_image_name)
    depth_path = cond_image_path[cond_id]
    prompt = cond_image_prompt[cond_id]


    # depth_path = "output/joint_depth_0.png"
    print(f"Load cond image from {depth_path}")
    depth_image = Image.open(depth_path).convert("RGB")

    height, width = depth_image.height, depth_image.width
    height, width = height // 8 * 8, width // 8 * 8
    # height, width = 512, 512

    # depth_image = T.Resize((height, width))(depth_image)
    depth_image = T.Resize((height))(depth_image)
    # depth_image = T.FiveCrop(512)(depth_image)[0]
    depth_image = T.CenterCrop((height, width))(depth_image)

    depth_image = T.ToTensor()(depth_image).to("cuda").to(torch.float16)
    height, width = depth_image.shape[-2:]

    if prompt is None:
        prompt = blip_cap([depth_image])[0]

    batch_size = 2
    depth_images = torch.stack([depth_image] * batch_size)

    depth_images = (depth_images - 0.5) / 0.5
    depth_latents = pipeline.vae.encode(depth_images).latent_dist.sample()
    depth_latents = depth_latents * pipeline.vae.config.scaling_factor
    # depth_latents = None
    print(prompt)

    # generator = torch.Generator(device=depth_latents.device)
    # generator = generator.manual_seed(123)

    torch.random.manual_seed(111)
    latents = torch.randn(batch_size, channel, height // 8, width // 8).to("cuda").to(torch.float16)
    # latents = tpporch.cat([latents] * 2)
    # if cond_x:
    #     latents[batch_size // 2:] = latents[[0]]
    # else:
    #     latents[:batch_size // 2] = latents[[0]]

    # if use_noise_lora:
    #     torch.manual_seed(123)
    #     fix_noise = torch.randn(4, 512, 512)
    #     condition_noise = fix_noise[...,:latents.shape[-2],:latents.shape[-1]]
    #     condition_noise = torch.stack([condition_noise] * (latents.shape[0])).to(latents)
    # else:
    #     condition_noise = latents

    # prompt = "cars parked"
    images = pipeline([prompt] * batch_size, 
                    height = height, 
                    width = width,
                    cond_x = cond_x,
                    #   guidance_scale = 4.0,
                    #   num_inference_steps = 30
                    #   condition_noise = condition_noise,
                    latents = latents, 
                    condition_latents = depth_latents, 
                    ).images
    for i, image in enumerate(images):
        image.save(f"output/joint_depth_{cond_id}_{i}.png")