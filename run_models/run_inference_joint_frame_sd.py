
import torch
from patch import patch_FSM as patch
import os
import torchvision.transforms as T
from cotracker.predictor import CoTrackerPredictor
from utils.dataset import process_frames
from utils.util import get_track_queries, load_input
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from safetensors import safe_open
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

CONTROLNET_DICT = {
    "tile": "lllyasviel/control_v11f1e_sd15_tile",
    "ip2p": "lllyasviel/control_v11e_sd15_ip2p",
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    # "softedge": 
    # "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "depth": "/data/vjuicefs_ai_camera_lgroup/11162591/gsy_workdir/hf_models_backup/lllyasviel/control_v11f1p_sd15_depth",
    "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
    # "canny": "lllyasviel/control_v11p_sd15_canny",
    "canny": "/data/vjuicefs_ai_camera_lgroup/11162591/gsy_workdir/hf_models_backup/lllyasviel/control_v11p_sd15_canny",
}


use_noise_lora = False

# model_id = "/root/data/juicefs_sharing_data/11162591/code/models/majicmixRealistic_v4/"
model_id = "/data/vjuicefs_ai_camera_lgroup/11162591/gsy_workdir/hf_models_backup/runwayml/stable-diffusion-v1-5"

controlnet_path = "/data/vjuicefs_ai_camera_lgroup/11162591/gsy_workdir/hf_models_backup/lllyasviel/control_v11f1p_sd15_depth"

control_type = "canny"
controlnet_key = CONTROLNET_DICT[control_type]
print(f'[INFO] loading controlnet from: {controlnet_key}')
controlnet = ControlNetModel.from_pretrained(
    controlnet_key, torch_dtype=torch.float16)
print(f'[INFO] loaded controlnet!')
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
# pipeline = StableDiffusionPipelineJointControl.from_pretrained(model_id,
#                                                       safety_checker=None, 
#                                                       torch_dtype=torch.float16).to("cuda")

# images = pipeline(["three cars are parked on the side of the road"] * 1, 



#                   ).images
# for i, image in enumerate(images):
#     image.save(f"output/joint_depth_{i}.png")
# exit(0)
# rec_txt1 = open('output_lora_joint_depth_image_clean_cond0.txt', 'w')

# for name, para in pipeline.unet.named_parameters():
#     rec_txt1.write(f'{name}\n')

height, width = 320, 576
input_path = "data/test_images/video_trans_input/breakdance.mp4"
video = load_input(input_path)[:32]
video = torch.stack([T.ToTensor()(frame) for frame in video], 0).unsqueeze(0).to("cuda")
video = process_frames(video, h = height, w = width)
n_frames = video.shape[1]
grid_query_frame = 0
queries = get_track_queries(video, grid_query_frame = grid_query_frame, downscale_rate=8, random_select = 0.25).to("cuda")


track_model = CoTrackerPredictor(checkpoint="diskdata/cotracker/cotracker2.pth").to("cuda")

patch.apply_patch(pipe.unet)
patch.initialize_joint_layers(pipe.unet)


output_path = "/data/juicefs_sharing_data/11162591/code/lxr/svd-train/output_dir/output_joint_frame_sd/checkpoint-242000"
if os.path.exists(os.path.join(output_path, "model.pth")):
    state_dict = torch.load(os.path.join(output_path, "model.pth"), map_location="cpu")
    pipe.unet.load_state_dict(state_dict, strict=False)
else:
    state_dict = {}
    with safe_open(os.path.join(output_path, "model.safetensors"), framework="pt", device=0) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    pipe.unet.load_state_dict(state_dict)


dst_frames = [-1]

src_frame = T.ToPILImage()(video[0, 0])
src_frame.save(f"output_dir/joint_frame_sd/src.png")
dst_frame = T.ToPILImage()(video[0, -1])
dst_frame.save(f"output_dir/joint_frame_sd/dst.png")


pred_tracks, pred_visibility = track_model(
    video * 255.,
    # grid_size=args.grid_size,
    # backward_tracking = True,
    queries = queries,
    # grid_query_frame=args.grid_query_frame,
    # backward_tracking=args.backward_tracking,
    # segm_mask=segm_mask
)

seq_name = os.path.splitext(input_path.split("/")[-1])[0]
vis = Visualizer(save_dir="output_dir/track_videos", pad_value=120, linewidth=3)
vis.visualize(
    video * 255.,
    pred_tracks,
    pred_visibility,
    query_frame=0,
    filename=seq_name,
)

dst_tracks = pred_tracks[:,dst_frames]
src_tracks = pred_tracks[:,[grid_query_frame]].expand_as(dst_tracks)
dst_tracks = dst_tracks.flatten(0,1)
src_tracks = src_tracks.flatten(0,1)
dst_vis = pred_visibility[:,dst_frames].flatten(0,1)

patch.update_patch(pipe.unet, track = (src_tracks, dst_tracks, dst_vis))

patch.update_patch(pipe.unet, track_res = (height, width))
frames = video[0, -1]
# rec_txt1.close()





channel = 4

# latents = torch.randn(batch, channel, height, width).to("cuda").to(torch.float16)
# latents[batch // 2:] = latents[:batch // 2]

pipe = pipe.to("cuda").to(torch.float16)


prompt = "a man doing a trick on the ground in front of a crowd."



images = pipe(prompt = [prompt] * 2,
                image = [src_frame, dst_frame],
                # control_image = [src_frame, dst_frame],
                strength = 0.2,
                height = height, 
                width = width,
              ).images
os.makedirs("output_dir/joint_frame_sd", exist_ok = True)
for i, image in enumerate(images):
    image.save(f"output_dir/joint_frame_sd/{i}.png")