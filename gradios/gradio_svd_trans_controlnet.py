import datetime
import os
from PIL import Image

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from utils.util import load_controlnet_trans_svd_pipeline, load_trans_svd_pipeline
from utils.controlnet_utils import prepare_control
from utils.dataset import process_frames
from utils.optical_flow import flow_to_image, inference_flow_warpper, load_unimatch
from utils.util import load_input, normalize_image
import torchvision.transforms as T
from torchvision.io import write_video

checkpoint_dict = {
    "xt-trans": "output_dir/output_svdxt_translation_lora_noflip_temporal_nospatial/checkpoint-16500",
    "xt-consec": "output_dir/output_svdxt_consec_lora_noflip_temporal_nospatial/checkpoint-103500"
}
single_lora_dict = {
    "xt-trans": True,
    "xt-consec": False
}

controlnet_dict = {
    "optical_flow": "output_dir/output_optical_flow_controlnet/checkpoint-139000/model.safetensors",
    "optical_flow_refine": "output_dir/output_optical_flow_controlnet_refine/checkpoint-36000/model.safetensors"
}

model_id = "xt-trans"
pretrain_svd_path = "/data/vjuicefs_ai_camera_lgroup/11162591/gsy_workdir/hf_models_backup/stabilityai/stable-video-diffusion-img2vid-xt"

control_type = "optical_flow_refine"
# controlnet_path = "CiaraRowles/temporal-controlnet-depth-svd-v1"
controlnet_path = controlnet_dict[control_type]
pipeline = load_controlnet_trans_svd_pipeline(pretrain_svd_path, checkpoint_dict[model_id], controlnet_path, single_lora = single_lora_dict[model_id])
output_dir = "output_dir/gradio_output"
cur_video_input = ""
cur_video = None

flow_model = load_unimatch()
checkpoint = torch.load("/data/juicefs_sharing_data/11162591/code/lxr/unimatch/models/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth", map_location="cpu")

flow_model.load_state_dict(checkpoint['model'], strict=True)

def process(start_frame_input, end_frame_input, motion_bucket_id, fps, width, seed, inference_step, 
            num_frames, model_selection, video_input, start_frame, end_frame, start_step, direct_fusion,
            max_guidance_scale, min_guidance_scale, controlnet_scale):
    global model_id
    if model_id != model_selection:
        global pipeline

        model_id = model_selection
        pipeline = load_controlnet_trans_svd_pipeline(pretrain_svd_path, checkpoint_dict[model_id], controlnet_path, single_lora = single_lora_dict[model_id])
    with torch.no_grad():

        validation_images = [start_frame_input, end_frame_input]

        validation_images = process_frames(validation_images, w = width, verbose = True, div = 64)
        height = validation_images[0].height
        global cur_video_input
        if cur_video_input != video_input:
            global cur_video

            cur_video_input = video_input
            control_video = load_input(video_input)
            cur_video = control_video
        else:
            control_video = cur_video

        frame_ids = torch.linspace(start_frame, end_frame, num_frames).long()
        f_interval = frame_ids[1] - frame_ids[0]
        frame_ids = torch.cat([frame_ids, frame_ids[[-1]] + f_interval])
        print(frame_ids)
        control_frames = [control_video[i] for i in frame_ids]
        control_frames = process_frames(control_frames, h = height, w = width, verbose = True, div = 64)

        # control_frames = torch.stack(control_frames)
        control_frames_pt = [T.ToTensor()(img) for img in control_frames]
        control_frames_pt = torch.stack(control_frames_pt)
        # control_frames_pt = normalize_image(control_frames_pt)
        control_frames_latents = []
        for image_batch in control_frames_pt[:-1].split(8):
            image_batch = normalize_image(image_batch).to(torch.float16).to("cuda")
            image_latents = pipeline.vae.encode(image_batch).latent_dist.mode()
            image_latents = image_latents * pipeline.vae.config.scaling_factor
            control_frames_latents.append(image_latents)
        control_frames_latents = torch.cat(control_frames_latents)
        control_frames_latents = torch.stack([control_frames_latents, control_frames_latents.flip(dims = [0])])


        control_frames_pt = torch.stack([control_frames_pt, control_frames_pt.flip(dims = [0])])

        # control_images_pt = prepare_control("depth", control_frames, frame_ids, "output_dir/gradio_output")
        control_images_pt = inference_flow_warpper(flow_model, control_frames_pt, control_frames_pt.shape[-2:]).squeeze(0)
        control_images_pt = process_frames(control_images_pt, w = width, verbose = True, div = 64)

        control_images_vis_pt = flow_to_image(control_images_pt[0]) / 255.

        # print(control_images_pt.shape)
        # control_images = [T.ToPILImage()(img) for img in control_images_pt]
        # control_images = 

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        # start_f = 0
        # n_frames = num_frames
        # frame_interval = 1

        # motion_bucket_id = 32
        # fps = 25
        # sframe, eframe = T.ToPILImage()(validation_images[start_f]), T.ToPILImage()(validation_images[start_f + n_frames - 1])
        # end_f = min(len(validation_images) - 1, start_f + (n_frames - 1) * frame_interval)

        # start_f = 0
        # end_f = 14
        # end_f = min(len(validation_images) - 1, end_f)
        sframe, eframe = validation_images[0], validation_images[1]

        

        if not isinstance(sframe, Image.Image):
            sframe = T.ToPILImage()(sframe)
            eframe = T.ToPILImage()(eframe)
        height, width = sframe.height, sframe.width

        sframe, eframe = T.CenterCrop((height, width))(sframe), T.CenterCrop((height, width))(eframe)

        with torch.autocast(device_type = "cuda", dtype = torch.float16):
            video_frames = pipeline([sframe, eframe],
                                    controlnet_condition = [control_images_pt[0], control_images_pt[1]],
                                    motion_bucket_id = motion_bucket_id,
                                    decode_chunk_size=8,
                                    num_frames=num_frames,
                                    num_inference_steps=inference_step,
                                    fps = fps,
                                    height = height,
                                    width = width,
                                    min_guidance_scale = min_guidance_scale,
                                    max_guidance_scale = max_guidance_scale,
                                    output_type = "pt",
                                    original_latents = control_frames_latents,
                                    start_step = start_step,
                                    direct_fusion = direct_fusion,
                                    controlnet_scale = controlnet_scale
                                    ).frames

        print(video_frames.shape)
        video_frames = (einops.rearrange(video_frames, 'b t c h w -> b t h w c') * 255.).cpu().numpy().clip(0, 255).astype(np.uint8)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_dir, f"output_{timestamp}.mp4")

        # video_frames = np.concatenate(video_frames, axis = 2)
        video_frames = video_frames[0]
        # video_frames = np
        write_video(output_path, video_frames, fps=7)

        control_images_vis_np = (einops.rearrange(control_images_vis_pt, 't c h w -> t h w c') * 255.).cpu().numpy().clip(0, 255).astype(np.uint8)
        control_output_path = os.path.join(output_dir, f"output_{timestamp}_control.mp4")

        write_video(control_output_path, control_images_vis_np, fps=7)


        original_video_np = (einops.rearrange(control_frames_pt[0, :-1], 't c h w -> t h w c') * 255.).cpu().numpy().clip(0, 255).astype(np.uint8) 
        cat_video_np = np.concatenate([video_frames, original_video_np], axis = 1)

        compose_output_path = os.path.join(output_dir, f"output_{timestamp}_compose.mp4")
        write_video(compose_output_path, cat_video_np, fps=7)
        # results = [video_frame for video_frame in video_frames]
    return output_path, control_output_path, compose_output_path


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Stable Video Diffusion for Translation Video")
    with gr.Row():
        with gr.Column():
            start_frame_input = gr.Image(type='pil', label='subject Image (输入起始帧)')
            end_frame_input = gr.Image(type='pil', label='subject Image (输入结束帧)')
            video_input = gr.Video(label='Input Video to extract control (输入用于提取控制信息的视频)')
            # prompt = gr.Textbox(label="Prompt")
            with gr.Accordion("Advanced options", open=False):
                motion_bucket_id = gr.Slider(label="Motion Bucket ID (控制视频运动强度)", minimum=0.0, maximum=255.0, value=127.0, step=1.0)
                fps = gr.Slider(label="FPS （目标视频帧率）", minimum=7.0, maximum=30.0, value=25.0, step=1.0)
                # height = gr.Slider(label="Height", minimum=128, maximum=1024, value=320, step=8)
                width = gr.Slider(label="Width （输出视频宽度）", minimum=128, maximum=1024, value=576, step=64)
                # height = gr.Slider(label="Height （输出视频高度）", minimum=128, maximum=1024, value=320, step=64)
                inference_step = gr.Slider(label="Inference Step", minimum=1, maximum=100, value=25, step=1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                num_frames = gr.Slider(label="Frame Number", minimum=1, maximum=30, step=1, value=25)
                model_selection = gr.Dropdown(choices = ["xt-trans", "xt-consec"], label="Model Selection (选择模型)", value="xt-trans")
                start_frame = gr.Number(label="Start Frame", value=0)
                end_frame = gr.Number(label="End Frame", value=24)
                start_step = gr.Number(label="Start Denoising Step", value=0)
                direct_fusion = gr.Checkbox(label='Direct Fusion', value=False)
                max_guidance_scale = gr.Number(label="Max Guidance Sacle", value=3.0)
                min_guidance_scale = gr.Number(label="Min Guidance Sacle", value=1.0)
                controlnet_scale = gr.Slider(label="ControlNet Scale", minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            run_button = gr.Button()
            # with gr.Accordion("Advanced options", open=False):
            #     num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
            #     image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
            #     strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
            #     guess_mode = gr.Checkbox(label='Guess Mode', value=False)
            #     detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
            #     ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
            #     scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
            #     seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
            #     eta = gr.Number(label="eta (DDIM)", value=0.0)
            #     a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
            #     n_prompt = gr.Textbox(label="Negative Prompt",
            #                           value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            # result_video = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            result_video = gr.Video(label='Output')
            cond_video = gr.Video(label='Output')
            compose_video = gr.Video(label='Output')
    ips = [start_frame_input, end_frame_input, motion_bucket_id, fps, width, seed, inference_step, 
           num_frames, model_selection, video_input, start_frame, end_frame, start_step, direct_fusion,
           max_guidance_scale, min_guidance_scale, controlnet_scale]
    run_button.click(fn=process, inputs=ips, outputs=[result_video, cond_video, compose_video])


block.launch(server_name='0.0.0.0', share=False)