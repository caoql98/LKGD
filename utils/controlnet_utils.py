import torch.nn.functional as F
# from PyQt5.QtCore import QLibraryInfo
import cv2
import os
import torch
import torchvision.transforms as T
from PIL import Image
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
#     QLibraryInfo.PluginsPath
# )
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/lixirui/anaconda3/envs/dfwebui/lib/python3.9/site-packages/PyQt5/Qt5/plugins"

from controlnet_aux.processor import Processor
import transformers
import numpy as np

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = T.ToTensor()(image)
    return image.unsqueeze(0)

CONTROLNET_DICT = {
    "tile": "lllyasviel/control_v11f1e_sd15_tile",
    "ip2p": "lllyasviel/control_v11e_sd15_ip2p",
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "softedge": "lllyasviel/control_v11p_sd15_softedge",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "canny": "lllyasviel/control_v11p_sd15_canny"
}

processor_cache = dict()

def prepare_control(control, frames, frame_ids, save_path):
    if control not in CONTROLNET_DICT.keys():
        print(f"[WARNING] unknown controlnet type {control}")
        return None

    control_subdir = f'{save_path}/{control}_image'

    preprocess_flag = True
    if os.path.exists(control_subdir):
        print(f"[INFO] load control image from {control_subdir}.")
        control_image_ls = []
        for frame_id in frame_ids:
            image_path = os.path.join(
                control_subdir, "{:04}.png".format(frame_id))
            if not os.path.exists(image_path):
                break
            control_image_ls += [load_image(image_path)]
        else:
            preprocess_flag = False
            control_images = torch.cat(control_image_ls)

    if preprocess_flag:
        print("[INFO] preprocessing control images...")
        control_images = control_preprocess(frames, control)
        if control_image.shape[-3] == 3:
            print(f"[INFO] save control images to {control_subdir}.")
            os.makedirs(control_subdir, exist_ok=True)
            for image, frame_id in zip(control_images, frame_ids):
                image_path = os.path.join(
                    control_subdir, "{:04}.png".format(frame_id))
                T.ToPILImage()(image).save(image_path)

    return control_images



def process(image, processor_id):
    process_ls = []
    H, W = image.shape[2:]
    if processor_id in processor_cache:
        processor = processor_cache[processor_id]
    else:
        processor = Processor(processor_id, {"output_type": "numpy"})
        processor_cache[processor_id] = processor
    for img in image:
        img = img.clone().cpu().permute(1,2,0) * 255
        processed_image = processor(img)
        processed_image = cv2.resize(processed_image, (W, H), interpolation=cv2.INTER_LINEAR)
        processed_image = torch.tensor(processed_image).to(image).permute(2,0,1) / 255
        process_ls.append(processed_image)
    processed_image = torch.stack(process_ls)
    return processed_image

def tile_preprocess(image, resample_rate = 1.0, **kwargs):
    cond_image = F.interpolate(image, scale_factor=resample_rate, mode="bilinear")
    cond_image = F.interpolate(cond_image, scale_factor=1 / resample_rate)
    return cond_image
    
def ip2p_prepreocess(image, **kwargs):
    return image

def openpose_prepreocess(image, **kwargs):
    processor_id = 'openpose'
    return process(image, processor_id)

def softedge_prepreocess(image, proc = "pidsafe", **kwargs):
    processor_id = f'softedge_{proc}'
    return process(image, processor_id)

def depth_prepreocess(image, **kwargs):
    image_ls = []
    if not isinstance(image[0], Image.Image):
        for img in image:
            image_ls.append(T.ToPILImage()(img))
    else:
        image_ls = image
    depth_estimator = transformers.pipeline('depth-estimation')
    ret = depth_estimator(image_ls)
    depth_ls = []
    for r in ret:
        depth_ls.append(T.ToTensor()(r['depth']))
    depth = torch.cat(depth_ls)
    depth = torch.stack([depth, depth, depth], axis=1)
    return depth

def lineart_anime_prepreocess(image, proc = "anime",**kwargs):
    processor_id = f'lineart_{proc}'
    return process(image, processor_id)

def canny_preprocess(image, **kwargs):
    processor_id = f'canny'
    return process(image, processor_id)


PREPROCESS_DICT = {
    "tile": tile_preprocess,
    "ip2p": ip2p_prepreocess,
    "openpose": openpose_prepreocess,
    "softedge": softedge_prepreocess,
    "depth": depth_prepreocess,
    "lineart_anime": lineart_anime_prepreocess,
    "canny": canny_preprocess
}

def control_preprocess(images, control_type, **kwargs):
    return PREPROCESS_DICT[control_type](images, **kwargs)

def empty_cache():
    global processor_cache
    processor_cache = dict()
    torch.cuda.empty_cache()
