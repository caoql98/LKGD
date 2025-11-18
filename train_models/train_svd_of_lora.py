#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler, ConcatDataset
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import datetime
import diffusers
from diffusers import StableVideoDiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image, convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.utils.import_utils import is_xformers_available

from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import is_compiled_module

from models.unet_spatio_temporal_condition_joint import UNetSpatioTemporalConditionJointModel
from pipeline.pipeline_stable_video_diffusion_joint_vf import StableVideoDiffusionPipelineJointVF
from utils.util import save_gifs_side_by_side
from utils.dataset import DAVIS, WebVid10M, Panda, compute_flow, optical_flow_normalize, process_frames, readBatchedFlow, readFlow
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from models.controlnet_sdv import ControlNetSDVModel

from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.utils import flow_to_image
import glob
import re
from torchvision.io import read_video, read_image
from peft import LoraConfig
from utils.peft_utils import get_peft_model_state_dict, set_peft_model_state_dict
from utils.motion_helper import cal_motion_bucket_ids, flow2motion, motion2bucket
from utils.optical_flow import flow_to_image_naive, image_to_flow_naive, inference_flow_warpper, load_unimatch, optical_flow_expand, optical_flow_latent_normalize, optical_flow_unnormalize
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from utils.util import _get_add_time_ids, denormalize_image, load_lora_weights, tensor_to_vae_latent
from diffusers.training_utils import cast_training_params, compute_snr

from patch import patch

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# MODEL_SAVE = {"ControlNetSDVModel", "UNetSpatioTemporalConditionControlNetModel"}
MODEL_SAVE = dict()

SAVE_NAMES = {
    "UNetSpatioTemporalConditionControlNetModel": "unet"
}

#i should make a utility function file
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
        image = image.resize(target_size)
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

def save_combined_frames(batch_output, validation_images, validation_control_images,output_folder):
    # Flatten batch_output, which is a list of lists of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Combine frames into a list without converting (since they are already PIL Images)
    combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3  # adjust number of columns as needed
    rows = (num_images + cols - 1) // cols
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    filename = f"combined_frames_{timestamp}.png"
    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols)
    output_folder = os.path.join(output_folder, "validation_images")
    os.makedirs(output_folder, exist_ok=True)
    
    # Now define the full path for the file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"combined_frames_{timestamp}.png"
    output_loc = os.path.join(output_folder, filename)
    
    if grid is not None:
        grid.save(output_loc)
    else:
        print("Failed to create image grid")

def load_images_from_folder_tensor(folder):
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
            img = read_image(os.path.join(folder, filename))
            images.append(img)

    return torch.stack(images)

def load_images_from_folder(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    # def frame_number(filename):
    #     parts = filename.split('_')
    #     if len(parts) > 1 and parts[0] == 'frame':
    #         try:
    #             return int(parts[1].split('.')[0])  # Extracting the number part
    #         except ValueError:
    #             return float('inf')  # In case of non-integer part, place this file at the end
    #     return float('inf')  # Non-frame files are placed at the end

    def sort_frames(frame_name):
        try:
            return int(os.path.basename(frame_name).split('_')[0].split('.')[0])
        except:
            print("Error", frame_name)

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=sort_frames)

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            # img = T.ToTensor()(img)
            images.append(img)

    # return torch.stack(images)
    return images


def load_flow_from_folder(folder):
    flows = []
    valid_extensions = {".flo"}  # Add or remove extensions as needed

    def sort_frames(frame_name):
        try:
            return int(os.path.basename(frame_name).split('_')[0].split('.')[0])
        except:
            print("Error", frame_name)


    if folder.split(".")[-1] == "flo":
        flows = readBatchedFlow(folder)
    else:
        # Sorting files based on frame number
        sorted_files = sorted(glob.glob(os.path.join(folder, "*.flo")), key=sort_frames)

        # Load images in sorted order
        for filename in sorted_files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                flows += [readFlow(os.path.join(folder, filename))]
        
                    

        flows = np.stack(flows, axis = 0)
    flows = torch.from_numpy(flows.transpose(0, 3, 1, 2))
    flows = optical_flow_normalize(flows)

    return flows


# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5

def make_train_dataset(args):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = Panda(args.video_folder, sample_size = (args.height, args.width), sample_n_frames = args.num_frames)
    print("Panda70M loaded. Total length: ", len(dataset))

    if args.mix_webvid > 0:
        webvid10m_csv_path = "/data/vjuicefs_ai_camera/11162591/public_datasets/webvid/webvid/results_10M_train.csv"
        webvid10m_video_folder = "/data/vjuicefs_ai_camera/11162591/public_datasets/webvid/webvid/data/videos"
        webvid_dataset = WebVid10M(webvid10m_csv_path, 
            webvid10m_video_folder, 
            sample_size=(args.height, args.width), 
            sample_n_frames=args.num_frames,
            dataset_len = args.mix_webvid,
        )
        print("WebVid10M loaded. Total length: ", len(webvid_dataset))
        dataset = ConcatDataset([dataset, webvid_dataset])
        print("Dataset mixed with WebVid10M. Total length: ", len(dataset))


    return dataset


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)






def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "path to the dataset csv"
        ),
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help=(
            "path to the video folder"
        ),
    )
    parser.add_argument(
        "--condition_folder",
        type=str,
        default=None,
        help=(
            "path to the depth folder"
        ),
    )
    parser.add_argument(
        "--motion_folder",
        type=str,
        default=None,
        help=(
            "path to the depth folder"
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image_folder",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_control_folder",
        type=str,
        default=None,
        help=(
            "the validation control image"
        ),
    )

    parser.add_argument(
        "--conditioning_channels",
        type=int,
        default=None,
        help=(
            "number of conditioning channels"
        ),
    )


    # parser.add_argument(
    #     "--conv_in_cond",
    #     action="store_true",
    #     help=(
    #         "whether to directly conv in condition in ControlNet"
    #     ),
    # )

    parser.add_argument(
        "--train_flow_diffusion",
        action="store_true",
        help=(
            "set when train a flow diffusion model"
        ),
    )
    parser.add_argument(
        "--y_lora",
        type=str,
        default=None,
        help=(
            "Path to the y_lora weights. If not provided, do not use y_lora."
        ),
    )
    parser.add_argument(
        "--y_head",
        type=str,
        default=None,
        help=(
            "Path to the y_head weights."
        ),
    )

    parser.add_argument(
        "--mix_webvid",
        type=int,
        default=0,
        help=(
            "Mix WebVid10M dataset with the training dataset. Provide the number of samples from WebVid10M to mix."
        )
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    # parser.add_argument(
    #     "--lora_rank",
    #     type=int,
    #     default=256,
    #     help=(
    #         "Rank of LoRA layers"
    #     ),
    # )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


# def load_lora(args, unet):
#     # lora_attn_procs = {}
#     # for name in unet.attn_processors.keys():
#     #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
#     #     if name.startswith("mid_block"):
#     #         hidden_size = unet.config.block_out_channels[-1]
#     #     elif name.startswith("up_blocks"):
#     #         block_id = int(name[len("up_blocks.")])
#     #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#     #     elif name.startswith("down_blocks"):
#     #         block_id = int(name[len("down_blocks.")])
#     #         hidden_size = unet.config.block_out_channels[block_id]

#     #     lora_attn_procs[name] = LoRAAttnProcessor(
#     #         hidden_size=hidden_size,
#     #         cross_attention_dim=cross_attention_dim,
#     #         rank=args.lora_rank,
#     #     )

#     # unet.set_attn_processor(lora_attn_procs)
#     # lora_layers = AttnProcsLayers(unet.attn_processors)
#     unet_lora_config = LoraConfig(
#         r=args.lora_rank,
#         lora_alpha=args.lora_rank,
#         init_lora_weights="gaussian",
#         target_modules=["to_k", "to_q", "to_v", "to_out.0"],
#     )

#     unet.add_adapter(unet_lora_config)
#     lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
#     return lora_layers

def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(23123134)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")
    
    unet_class = UNetSpatioTemporalConditionJointModel
    unet = unet_class.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        # variant="fp16",
    )
    pipeline_class = StableVideoDiffusionPipelineJointVF

    if args.y_head is not None:
        unet.add_y_input_head()
        unet.load_y_input_head(args.y_head)
    # unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

    # unet_lora_config = LoraConfig(
    #     r=args.lora_rank,
    #     lora_alpha=args.lora_rank,
    #     init_lora_weights="gaussian",
    #     # target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    # )
    # # print("UNet Config,", unet.config)
    # # print(image_encoder.config)
    # unet.add_adapter(unet_lora_config)
    # lora_params = filter(lambda p: p.requires_grad, unet.parameters())
        
    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    for param in unet.parameters():
        param.requires_grad_(False)

    patch.apply_patch(unet)
    patch.initialize_joint_layers(unet)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["attn1n.to_k", "attn1n.to_q", "attn1n.to_v", "attn1n.to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)



    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config, adapter_name = "xy_lora")
    unet.add_adapter(unet_lora_config, adapter_name = "yx_lora")
    if args.y_lora is not None:
        print(args.y_lora)
        y_lora_state_dict, y_lora_network_alphas = StableDiffusionPipeline.lora_state_dict(args.y_lora)
        StableDiffusionPipeline.load_lora_into_unet(y_lora_state_dict, y_lora_network_alphas, unet = unet, adapter_name = "y_lora")

    patch.hack_lora_forward(unet)

    patch.initialize_joint_lora(unet, "y_lora", "yx_lora")

    unet.set_adapters(["y_lora", "xy_lora", "yx_lora"])
  

    for param in unet.parameters():
        param.requires_grad_(False)

    patch.set_joint_layer_requires_grad(unet, ["xy_lora", "yx_lora"], True)
    patch.set_patch_lora_mask(unet, "y_lora", [0,1])
    patch.set_patch_lora_mask(unet, "yx_lora", [0,1])
    patch.set_patch_lora_mask(unet, "xy_lora", [1,0])

    train_modules = {
        "conv_in": unet.conv_in_y, 
        "time_proj": unet.time_proj_y, 
        "time_embedding": unet.time_embedding_y, 
        "add_embedding": unet.add_embedding_y
    }
    for module in train_modules.values():
        module.requires_grad_(True)


    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    # lora_layers = load_lora(args, unet)

    
    flow_model = load_unimatch()
    checkpoint = torch.load("/data/juicefs_sharing_data/11162591/code/lxr/unimatch/models/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth", map_location="cpu")

    flow_model.load_state_dict(checkpoint['model'], strict=True)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def load_model_hook(models, input_dir):
            for model in models:
                if isinstance(model, unet_class):
                    unet = model

            y_head_path = os.path.join(input_dir, "unet_yhead.pth")
            unet.load_y_input_head(y_head_path)

            for lora_name in ["xy_lora", "yx_lora"]:
                lora_path = os.path.join(save_path, f"{lora_name}", "pytorch_lora_weights.safetensors")
                if os.path.exists(lora_path):
                    load_lora_weights(unet, lora_path, adapter_name=lora_name)

        # accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

        

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW



    # for name, para in unet.named_parameters():
    #     if 'temporal_transformer_block' in name and 'down_blocks' in name:
    #         parameters_list.append(para)
    #         para.requires_grad = True
    #     else:
    #         para.requires_grad = False
    # optimizer = optimizer_cls(
    #     parameters_list,
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )
    # params = list(controlnet.parameters()) + list(unet.parameters())
    # params = unet.parameters()
    params = lora_layers
    # params = controlnet.parameters()

    optimizer = optimizer_cls(
        list(params),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # check para
    if accelerator.is_main_process:

        rec_txt1 = open('rec_para_unet_of_lora.txt', 'w')
        rec_txt2 = open('rec_para_unet_of_lora_train.txt', 'w')
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()
    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = make_train_dataset(args)

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))

    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    # unet, optimizer, lr_scheduler, train_dataloader,controlnet = accelerator.prepare(
    #     unet, optimizer, lr_scheduler, train_dataloader,controlnet
    # )
    # unet, optimizer, lr_scheduler, train_dataloader,controlnet, lora_layers = accelerator.prepare(
    #     unet, optimizer, lr_scheduler, train_dataloader,controlnet, lora_layers
    # )
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
        assert pixel_values.min() >= 0, "Image values must be non-negative"
        pixel_values = pixel_values * 2.0 - 1.0
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(
            device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings= image_embeddings.unsqueeze(1)
        return image_embeddings

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model




    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

        # for name, para in unet.named_parameters():
        #     if para.requires_grad is True:
        #         print(name)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    last_timestep = 0.25 * np.log(1e3)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        # Skip steps until we reach the resumed step
        if args.resume_from_checkpoint and epoch == first_epoch:
            skipped_dataloader = accelerator.skip_first_batches(train_dataloader, num_batches=resume_step)
            cur_train_dataloader = skipped_dataloader
            progress_bar.update(resume_step // args.gradient_accumulation_steps)
        else:
            cur_train_dataloader = train_dataloader
        for step, batch in enumerate(cur_train_dataloader):

            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                pixel_values = batch["pixel_values"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )[:,:-1]



                # latents = tensor_to_vae_latent(pixel_values, vae)
                
                denormalized_video = denormalize_image(batch["pixel_values"])
                pred_flow = inference_flow_warpper(flow_model, denormalized_video, denormalized_video.shape[-2:])
                # motion_bucket_ids = torch.tensor([flow2motion(fps.item(), flow = fl) for fl, fps in zip(pred_flow, batch["fps"])])
                motion_bucket_ids = cal_motion_bucket_ids(flow_model, batch["pixel_values"], batch["fps"])
                # print(motion_bucket_ids)



                flow_image = flow_to_image_naive(pred_flow).to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )




                flow_latents = tensor_to_vae_latent(flow_image, vae)
                video_latents = tensor_to_vae_latent(pixel_values, vae)
                normalized_flow_latents = optical_flow_latent_normalize(flow_latents)
                joint_latents = torch.cat([video_latents, normalized_flow_latents], dim = 0)

                motion_bucket_ids = torch.cat([motion_bucket_ids] * 2, dim = 0)

                # SVD use fps - 1 as condition
                fps = torch.cat([batch["fps"] - 1] * 2, dim = 0)
                # latents = torch.cat([latents, controlnet_image], dim = 2)
                # latents, latents_video = flow_latents, latents

                # latents_video = latents
                latents = joint_latents


                #conditional_latents = latents[:, 0, :, :, :]
                #conditional_latents = conditional_latents / vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                sigmas = rand_cosine_interpolated(shape=[bsz // 2,], image_d=image_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high,
                                                  sigma_data=sigma_data, min_value=min_value, max_value=max_value).to(latents.device)
                sigmas = torch.cat([sigmas] * 2, dim = 0)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas_reshaped = sigmas.clone()
                while len(sigmas_reshaped.shape) < len(latents.shape):
                    sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)
                    
                train_noise_aug = 0.02
                # small_noise_latents = latents + noise * train_noise_aug
                # conditional_latents = small_noise_latents[:, 0, :, :, :]
                # conditional_latents = conditional_latents / vae.config.scaling_factor
                
                

                init_frame_noise = torch.randn_like(pixel_values[:, 0, :, :, :])

                small_noise_init_frame = pixel_values[:, 0, :, :, :] + init_frame_noise * train_noise_aug

                conditional_latents = vae.encode(small_noise_init_frame).latent_dist.mode()
                conditional_latents = torch.cat([conditional_latents] * 2, dim = 0)

                noise = torch.randn_like(latents)
                noisy_latents  = latents + noise * sigmas_reshaped
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(latents.device)

                
                
                inp_noisy_latents = noisy_latents  / ((sigmas_reshaped**2 + 1) ** 0.5)
                
                
                # Get the text embedding for conditioning.
                encoder_hidden_states = encode_image(
                    pixel_values[:, 0, :, :, :])

                encoder_hidden_states = torch.cat([encoder_hidden_states] * 2, dim = 0)

                added_time_ids = _get_add_time_ids(
                    fps,
                    # batch["motion_values"],
                    # [127] * bsz,
                    motion_bucket_ids,
                    train_noise_aug, # noise_aug_strength == 0.0
                    encoder_hidden_states.dtype,
                    bsz,
                    unet,
                    device=latents.device
                )
                added_time_ids = added_time_ids.to(latents.device)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(
                        bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(
                            image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                conditional_latents = conditional_latents.unsqueeze(
                    1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                
                # if args.train_flow_diffusion:
                #     # inp_noisy_latents_video, inp_noisy_latents = inp_noisy_latents[:,:,:4,:,:], inp_noisy_latents[:,:,4:,:,:]
                #     # latents_video, latents = latents[:,:,:4,:,:], latents[:,:,4:,:,:]

                #     # latents = latents.repeat(1, 1, 2, 1, 1)
                #     # noisy_latents = noisy_latents.repeat(1, 1, 2, 1, 1)
                #     # inp_noisy_latents = inp_noisy_latents.repeat(1, 1, 2, 1, 1)

                #     # latents = latents.repeat(1, 1, 2, 1, 1)
                #     # noisy_latents = noisy_latents.repeat(1, 1, 2, 1, 1)
                #     # inp_noisy_latents = inp_noisy_latents.repeat(1, 1, 2, 1, 1)


                #     # inp_noisy_latents_video = torch.cat(
                #     #     [inp_noisy_latents_video, conditional_latents], dim=2)
                #     latents_video = torch.cat(
                #         [latents_video, conditional_latents], dim=2)
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2)
                

                # controlnet_image = batch["depth_pixel_values"]

                

                # Get the target for loss depending on the prediction type
                # if noise_scheduler.config.prediction_type == "epsilon":
                #     target = latents  # we are computing loss against denoise latents
                # elif noise_scheduler.config.prediction_type == "v_prediction":
                #     target = noise_scheduler.get_velocity(
                #         latents, noise, timesteps)
                # else:
                #     raise ValueError(
                #         f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                target = latents
                # print(inp_noisy_latents.shape)
                # print(controlnet_image.shape)
                # if args.train_flow_diffusion:
                #     # down_block_res_samples, mid_block_res_sample = controlnet(
                #     #     inp_noisy_latents_video, timesteps, encoder_hidden_states,
                #     #     added_time_ids=added_time_ids,
                #     #     controlnet_cond=None,
                #     #     return_dict=False,
                #     # )
                #     down_block_res_samples, mid_block_res_sample = controlnet(
                #         latents_video, torch.ones_like(timesteps) * last_timestep, encoder_hidden_states,
                #         added_time_ids=added_time_ids,
                #         controlnet_cond=None,
                #         return_dict=False,
                #     )
                # else:
                #     down_block_res_samples, mid_block_res_sample = controlnet(
                #         inp_noisy_latents, timesteps, encoder_hidden_states,
                #         added_time_ids=added_time_ids,
                #         controlnet_cond=controlnet_image,
                #         return_dict=False,
                #     )

                
            
                # Predict the noise residual
                model_pred = unet(
                    inp_noisy_latents, timesteps, encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    # down_block_additional_residuals=[
                    #     sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    # ],
                    # mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                sigmas = sigmas_reshaped
                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                # For L2 loss
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
                # For L1 loss
                # weighing = ((sigmas**2 + 1)**0.5) * (sigmas**-1.0)

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() -
                     target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                # loss = torch.mean(
                #     (weighing.float() * (torch.abs(denoised_latents.float() -
                #      target.float()))).reshape(target.shape[0], -1),
                #     dim=1,
                # )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # if loss > 10:
                #     print("Time")

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if (
                        (global_step % args.validation_steps == 0)
                        or (global_step == 1)
                    ):
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                            f" {args.validation_prompt}."
                        )
                        raw_unet = unwrap_model(unet)
                        patch.set_patch_lora_mask(raw_unet, "y_lora", [0,1,0,1])
                        patch.set_patch_lora_mask(raw_unet, "yx_lora", [0,1,0,1])
                        patch.set_patch_lora_mask(raw_unet, "xy_lora", [1,0,1,0])

                        # create pipeline
                        pipeline = pipeline_class.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            image_encoder=accelerator.unwrap_model(
                                image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        validation_images = load_images_from_folder_tensor(args.validation_image_folder).to(weight_dtype).to(
                            accelerator.device, non_blocking=True
                        )[:14] / 255.
                        # validation_images = validation_images / 255.
                        validation_images = process_frames(validation_images, args.height, args.width)

                        # image_path = os.path.join(args.validation_image_folder, "00001.jpg")
                        # validation_images = Image.open(image_path).convert("RGB")

                        # validation_control_flows = inference_flow_warpper(flow_model, validation_images.unsqueeze(0), validation_images.shape[-2:]).squeeze(0)
                        # motion_bucket_id = flow2motion(validation_control_flows.norm(dim=-3).mean(), fps = 30)
                        # validation_control_images = flow_to_image(validation_control_flows)
                        # validation_control_images = validation_control_images / 255.
                        # validation_control_images = process_frames(validation_control_images, args.height, args.width)
                        # validation_control_flows = process_frames(validation_control_flows, args.height, args.width)
                        # validation_latents = tensor_to_vae_latent(validation_images.unsqueeze(0), vae).squeeze(0)
                        
                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")
                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        # run inference
                        generator = torch.Generator(device=accelerator.device)
                        if args.seed is not None:
                            generator = generator.manual_seed(args.seed)
                        videos = []
                        flows = []
                        if torch.backends.mps.is_available():
                            autocast_ctx = nullcontext()
                        else:
                            autocast_ctx = torch.autocast(accelerator.device.type)

                        with autocast_ctx:
                            # for _ in range(args.num_validation_images):
                            num_frames = args.num_frames
                            cond_frame = T.ToPILImage()(validation_images[0])
                            result = pipeline(
                                [cond_frame] * 2, 
                                # num_videos_per_prompt = 2,
                                generator=generator, 
                                height=args.height,
                                width=args.width,
                                num_frames=num_frames,
                                decode_chunk_size=8,
                                motion_bucket_id=127,
                                fps=25,
                                noise_aug_strength=0.02,
                                output_type="pt",
                                )
                            videos += result.frames
                            flows += result.flows
                            # videos += [result.frames[0]]
                            # flows += [result.frames[1]]

                        patch.set_patch_lora_mask(raw_unet, "y_lora", [0,1])
                        patch.set_patch_lora_mask(raw_unet, "yx_lora", [0,1])
                        patch.set_patch_lora_mask(raw_unet, "xy_lora", [1,0])

                        video = videos[0]
                        flow_images = flows[0]
                        # flow_frames = flow_images

                        flows = image_to_flow_naive(flow_images)
                        flows = optical_flow_unnormalize(flows)
                        # flows = optical_flow_unnormalize(flows)
                        flow_frames = flow_to_image(flows) / 255.
                        save_gifs_side_by_side(
                            [validation_images, video, flow_frames], 
                            val_save_dir
                        )

                        # for tracker in accelerator.trackers:
                        #     if tracker.name == "tensorboard":
                        #         np_images = np.stack([np.asarray(img) for img in images])
                        #         tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        #     if tracker.name == "wandb":
                        #         tracker.log(
                        #             {
                        #                 "validation": [
                        #                     wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                        #                     for i, image in enumerate(images)
                        #                 ]
                        #             }
                        #         )

                        del pipeline
                        torch.cuda.empty_cache()

                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    try:
                                        shutil.rmtree(removing_checkpoint)
                                    except OSError:
                                        logger.error(f"Remove {removing_checkpoint} failed. Maybe a symbolic link.")


                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        unwrapped_unet = unwrap_model(unet)
                        for lora_name in ["xy_lora", "yx_lora"]:
                            cur_save_path = os.path.join(save_path, f"{lora_name}")
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet, adapter_name=lora_name)
                            )


                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=cur_save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )

                        state_dict = dict()
                        for name, module in train_modules.items():
                            state_dict[name] = module.state_dict()
                        torch.save(state_dict, os.path.join(save_path, "unet_yhead.pth"))
                        logger.info(f"Saved state to {save_path}")


            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = unwrap_model(unet)
        save_path = args.output_dir

        for lora_name in ["xy_lora", "yx_lora"]:
            cur_save_path = os.path.join(save_path, f"{lora_name}")
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unwrapped_unet, adapter_name=lora_name)
            )


            StableDiffusionPipeline.save_lora_weights(
                save_directory=cur_save_path,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )

        state_dict = dict()
        for name, module in train_modules.items():
            state_dict[name] = module.state_dict()
        torch.save(state_dict, os.path.join(save_path, "unet_yhead.pth"))

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
