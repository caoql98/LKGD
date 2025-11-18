import datetime
import os
from PIL import Image

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from utils.util import load_trans_svd_pipeline
from utils.dataset import process_frames
from utils.util import load_input
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

model_id = "xt-trans"
pretrain_svd_path = "/data/vjuicefs_ai_camera_lgroup/11162591/gsy_workdir/hf_models_backup/stabilityai/stable-video-diffusion-img2vid-xt"

pipeline = load_trans_svd_pipeline(pretrain_svd_path, checkpoint_dict[model_id], single_lora = single_lora_dict[model_id])
output_dir = "output_dir/gradio_output"


def process(start_frame_input, end_frame_input, motion_bucket_id, fps, width, seed, inference_step, num_frames, model_selection):
    global model_id
    if model_id != model_selection:
        global pipeline

        model_id = model_selection
        pipeline = load_trans_svd_pipeline(pretrain_svd_path, checkpoint_dict[model_id], single_lora = single_lora_dict[model_id])
    with torch.no_grad():

        validation_images = [start_frame_input, end_frame_input]

        validation_images = process_frames(validation_images, w = width, verbose = True, div = 64)

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
                                    motion_bucket_id = motion_bucket_id,
                                    decode_chunk_size=8,
                                    num_frames=num_frames,
                                    num_inference_steps=inference_step,
                                    fps = fps,
                                    height = height,
                                    width = width,
                                    min_guidance_scale = 1.0,
                                    max_guidance_scale = 3.0,
                                    output_type = "pt").frames

        print(video_frames.shape)
        video_frames = (einops.rearrange(video_frames, 'b t c h w -> b t h w c') * 255.).cpu().numpy().clip(0, 255).astype(np.uint8)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_dir, f"output_{timestamp}.mp4")

        # video_frames = np.concatenate(video_frames, axis = 2)
        video_frames = video_frames[0]
        write_video(output_path, video_frames, fps=7)

        # results = [video_frame for video_frame in video_frames]
    return output_path


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Stable Video Diffusion for Translation Video")
    with gr.Row():
        with gr.Column():
            start_frame_input = gr.Image(type='pil', label='subject Image (输入起始帧)')
            end_frame_input = gr.Image(type='pil', label='subject Image (输入结束帧)')
            # prompt = gr.Textbox(label="Prompt")
            with gr.Accordion("Advanced options", open=False):
                motion_bucket_id = gr.Slider(label="Motion Bucket ID (控制视频运动强度)", minimum=0.0, maximum=255.0, value=127.0, step=1.0)
                fps = gr.Slider(label="FPS （目标视频帧率）", minimum=7.0, maximum=30.0, value=25.0, step=1.0)
                # height = gr.Slider(label="Height", minimum=128, maximum=1024, value=320, step=8)
                width = gr.Slider(label="Width （输出视频宽度）", minimum=128, maximum=1024, value=576, step=64)
                inference_step = gr.Slider(label="Inference Step", minimum=1, maximum=100, value=25, step=1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                num_frames = gr.Slider(label="Frame Number", minimum=1, maximum=30, step=1, value=25)
                model_selection = gr.Dropdown(choices = ["xt-trans", "xt-consec"], label="Model Selection (选择模型)", value="xt-trans")

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
    ips = [start_frame_input, end_frame_input, motion_bucket_id, fps, width, seed, inference_step, num_frames, model_selection]
    run_button.click(fn=process, inputs=ips, outputs=[result_video])


block.launch(server_name='0.0.0.0', share=False)