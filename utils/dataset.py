import json
import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import cv2

import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
import random
# try:
#     from utils.util import zero_rank_print
# except:
#     from util import zero_rank_print
#from torchvision.io import read_image
import glob
import random
from torchvision.io import read_video
from PIL import Image, ImageChops
import time
from .optical_flow import flow_to_image, inference_flow_warpper_v2, optical_flow_normalize, warp_frames
from torchvision.io import write_video, read_video, write_png
from .optical_flow import load_unimatch, inference_flow
from .optical_flow import args as flow_args
from . import train_helpers
import pdb
import torchvision.transforms.v2 as transforms
OPTICAL_FLOW_STD = 16.4956
OPTICAL_FLOW_MEAN = -0.0929



def flow_to_motion_strength(flow):
    motion_strength = flow.norm(dim=1).mean()
    motion_bucket_ids = int((1 + motion_strength / 3.5) * 127)
    motion_bucket_ids = motion_bucket_ids = min(300, motion_bucket_ids)
    return motion_bucket_ids

def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        # if 202021.25 != magic:
        #     print('Magic number incorrect. Invalid .flo file')
        #     return None
        # else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print 'Reading %d x %d flo file\n' % (w, h)
        data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
        # Reshape testdata into 3D array (columns, rows, bands)
        # The reshape here is for visualization, the original code is (w,h,2)
        return np.resize(data, (int(h), int(w), 2))

def readBatchedFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32)
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.reshape(data, (-1, int(h), int(w), 2))

# def process_frames(frames, h = None, w = None, verbose = False, div = None, rand_crop = False, ):
#     # Resize and crop frame to (h, w) while keeping the aspect ratio

#     if isinstance(frames[0], Image.Image):
#         fh, fw = frames[0].height, frames[0].width
#     else:
#         fh, fw = frames.shape[-2:]
#         assert len(frames.shape) >= 3
#         if len(frames.shape) == 3:
#             frames = [frames]
    
#     if h is None and w is None:
#         ratio = 1
#         h, w = fh, fw
#     elif h is None:
#         ratio = w / fw
#         h = int(fh * ratio)
#     elif w is None:
#         ratio = h / fh
#         w = int(fw * ratio)
#     else:
#         h_ratio = h / fh
#         w_ratio = w / fw
#         ratio = max(h_ratio, w_ratio)

    
#     if div is not None:
#         h = h // div * div
#         w = w // div * div
    

#     size = (int(fh * ratio + 0.5), int(fw * ratio + 0.5))
#     # print(ratio, size)
#         # if nw >= w:
#         #     size = (h, nw)
#         # else:
#         #     size = (int(fh / fw * w), w)

#     if verbose:
#         print(
#             f"[INFO] frame size {(fh, fw)} resize to {size} and centercrop to {(h, w)}")

#     frame_ls = []
#     for frame in frames:
#         if ratio <= 1 and rand_crop:
#             resized_frame = frame
#         else:
#             resized_frame = T.Resize(size, antialias=True)(frame)
#         # print(size)
#         if rand_crop:
#             cropped_frame = T.RandomCrop([h, w])(resized_frame)
#         else:
#             cropped_frame = T.CenterCrop([h, w])(resized_frame)
        
        
#         # croped_frame = T.FiveCrop([h, w])(resized_frame)[0]
#         frame_ls.append(cropped_frame)
#     if isinstance(frames[0], Image.Image):
#         return frame_ls
#     else:
#         return torch.stack(frame_ls)



def process_frames(frames, h=None, w=None, verbose=False, div=None):
    # Resize frame to (h, w) while keeping the aspect ratio

    if isinstance(frames[0], Image.Image):
        fh, fw = frames[0].height, frames[0].width
    else:
        fh, fw = frames.shape[-2:]
        assert len(frames.shape) >= 3
        if len(frames.shape) == 3:
            frames = [frames]

    if h is None and w is None:
        h, w = fh, fw
    elif h is None:
        ratio = w / fw
        h = int(fh * ratio)
    elif w is None:
        ratio = h / fh
        w = int(fw * ratio)
    else:
        h_ratio = h / fh
        w_ratio = w / fw
        ratio = max(h_ratio, w_ratio)

    if div is not None:
        h = h // div * div
        w = w // div * div

    size = (int(fw * ratio + 0.5), int(fh * ratio + 0.5))

    if verbose:
        print(f"[INFO] frame size {(fh, fw)} resized to {size}")

    frame_ls = []
    for frame in frames:
        resized_frame = T.Resize([h,w], antialias=True)(frame)
        frame_ls.append(resized_frame)

    if isinstance(frames[0], Image.Image):
        return frame_ls
    else:
        return torch.stack(frame_ls)

class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_n_frames=14,
            dataset_len = 10000,
            fps_range = [7,7]
        ):
        # zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            csv_dict = csv.DictReader(csvfile)
            # if dataset_len is None:
                # dataset_len = len(csv_dict)
            # dataset_len = min(dataset_len, len(csv_dict))
            self.dataset = [next(csv_dict) for _ in range(dataset_len)]
        
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        self.video_folder    = video_folder
        self.sample_n_frames = sample_n_frames
        # self.depth_folder = depth_folder
        # self.motion_values_folder=motion_folder
        print("length",len(self.dataset))
        self.sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)
        self.fps_range = fps_range
        self.pixel_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            # T.Resize(sample_size),
            # T.CenterCrop(sample_size),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        print("Set video fps range to", fps_range)
    




    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]
        
    
    def get_batch(self, idx):
        def sort_frames(frame_name):
            return int(frame_name.split('_')[1].split('.')[0])
    

    
        while True:
            video_dict = self.dataset[idx]
            page_dir = video_dict['page_dir']
            videoid = video_dict['videoid'] + ".mp4"
    
            preprocessed_dir = os.path.join(self.video_folder, page_dir, videoid)
            if not os.path.exists(preprocessed_dir):
                idx = random.randint(0, len(self.dataset) - 1)
                continue
            frames, _, info = read_video(preprocessed_dir, pts_unit="sec")

            frame_len = len(frames)
            if frame_len < self.sample_n_frames + 1:
                idx = random.randint(0, len(self.dataset) - 1)
                continue
            if "video_fps" in info:
                fps = info["video_fps"]
            else:
                fps = 30



            frame_len = len(frames)
            sample_len = self.sample_n_frames + 1

            assert frame_len >= sample_len, f"Require at least {sample_len} frames, but only {frame_len} frames found"
            

            target_fps = random.randint(*self.fps_range)
            
            frame_interval = int(fps // target_fps)
            frame_interval = min(max(frame_interval, 1), frame_len // sample_len) 

            # frame_interval_range = (1, frame_len // (sample_len))
            # rand_interval = random.randint(*frame_interval_range)

            # fps = fps / rand_interval
            fps = fps / frame_interval


            rand_start = random.randint(0, max(frame_len - (sample_len) * frame_interval, 0))
            frames = frames[rand_start:rand_start + sample_len * frame_interval:frame_interval]

            # rand_start = random.randint(0, max(frame_len - self.sample_n_frames - 1, 0))
            # frames = frames[rand_start:rand_start + self.sample_n_frames + 1]
    
            # Check if there are enough frames for both image and depth

    
            # Load image frames

            pixel_values = frames
        
            # Load depth frames
            
            # flow_path = preprocessed_dir.replace(".mp4", ".flo")
            # if os.path.exists(flow_path):
            #     depth_pixel_values = readBatchedFlow(flow_path)[rand_start:rand_start + self.sample_n_frames]
            #     depth_pixel_values = torch.from_numpy(depth_pixel_values.transpose(0, 3, 1, 2))
            #     # depth_pixel_values = optical_flow_normalize(depth_pixel_values)
            # else:
            #     depth_pixel_values = compute_flow(pixel_values)

            pixel_values = pixel_values.permute(0, 3, 1, 2) / 255.
            # numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(depth_folder, df))) for df in depth_files])
            # depth_pixel_values = numpy_to_pt(numpy_depth_images)

            # motion_bucket_ids = flow_to_motion_strength(depth_pixel_values)
            # print(motion_bucket_ids.shape)
    
            return pixel_values, fps

        
        
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        #while True:
           # try:
        pixel_values, fps = self.get_batch(idx)
           #     break
          #  except Exception as e:
          #      print(e)
          #      idx = random.randint(0, self.length - 1)

        
        
        # input_tensor = torch.cat([pixel_values, depth_pixel_values], dim = 1)
        # input_tensor = process_frames(input_tensor, *self.sample_size)
        # pixel_values, depth_pixel_values = input_tensor[:,:3], input_tensor[:,3:]

        pixel_values = process_frames(pixel_values, *self.sample_size)
        


        pixel_values = self.pixel_transforms(pixel_values)
        # sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values, motion_values=motion_bucket_ids)
        sample = dict(pixel_values=pixel_values, fps = fps)
        return sample

class DAVIS(Dataset):
    def __init__(
            self,
            video_folder,depth_folder,
            sample_size=512, sample_n_frames=14,
        ):
        self.dataset = glob.glob(os.path.join(video_folder, "*"))
        self.length = len(self.dataset)
        # print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        self.video_folder    = video_folder
        self.sample_n_frames = sample_n_frames
        self.depth_folder = depth_folder
        # self.motion_values_folder=motion_folder
        print("length",len(self.dataset))
        self.sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)
        # self.pixel_transforms = T.Compose([
        #     T.RandomHorizontalFlip(),
        #     T.Resize(sample_size),
        #     T.CenterCrop(sample_size),
        #     # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        # ])
        self.pixel_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            # T.Resize(sample_size),
            # T.CenterCrop(sample_size),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])





    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]
        
    
    def get_batch(self, idx):
        def sort_frames(frame_name):
            try:
                return int(os.path.basename(frame_name).split('_')[0].split('.')[0])
            except:
                print("Error", frame_name)
    

    
        while True:
            preprocessed_dir = self.dataset[idx]
            video_name = preprocessed_dir.split(os.sep)[-1]
            depth_folder = os.path.join(self.depth_folder, video_name)
    
            if not os.path.exists(depth_folder):
                idx = random.randint(0, len(self.dataset) - 1)
                continue
    
            # Sort and limit the numbqer of image and depth files to 14
            # print(preprocessed_dir)
            frame_dirs = glob.glob(os.path.join(preprocessed_dir, "*.jpg"))
            frame_len = len(frame_dirs)
            rand_start = random.randint(0, max(frame_len - self.sample_n_frames, 0))
            image_files = sorted(frame_dirs, key=sort_frames)[rand_start:rand_start + self.sample_n_frames]
            depth_files = sorted(glob.glob(os.path.join(depth_folder, "*.flo")), key=sort_frames)[rand_start:rand_start + self.sample_n_frames]
    
            # Check if there are enough frames for both image and depth
            if len(image_files) < self.sample_n_frames or len(depth_files) < self.sample_n_frames:
                idx = random.randint(0, len(self.dataset) - 1)
                continue
    
            # Load image frames
            numpy_images = np.array([pil_image_to_numpy(Image.open(img)) for img in image_files])
            pixel_values = numpy_to_pt(numpy_images)
        
            # Load depth frames
            depth_pixel_values = [readFlow(df) for df in depth_files]
            depth_pixel_values = np.stack(depth_pixel_values, axis = 0)
            depth_pixel_values = torch.from_numpy(depth_pixel_values.transpose(0, 3, 1, 2))
            depth_pixel_values = optical_flow_normalize(depth_pixel_values)
            # numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(depth_folder, df))) for df in depth_files])
            # depth_pixel_values = numpy_to_pt(numpy_depth_images)

            motion_bucket_ids = flow_to_motion_strength(depth_pixel_values)
            # print(motion_bucket_ids.shape)
    
            return pixel_values, depth_pixel_values, motion_bucket_ids

        
        
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        #while True:
           # try:
        pixel_values, depth_pixel_values, motion_bucket_ids = self.get_batch(idx)
           #     break
          #  except Exception as e:
          #      print(e)
          #      idx = random.randint(0, self.length - 1)
        
        input_tensor = torch.cat([pixel_values, depth_pixel_values], dim = 1)
        input_tensor = process_frames(input_tensor, *self.sample_size)
        pixel_values, depth_pixel_values = input_tensor[:,:3], input_tensor[:,3:]

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values, motion_values=motion_bucket_ids)
        return sample


def compute_flow(video_tensor, frame_interval = 1):
    video = video_tensor.numpy()
    flow_ls = []
    for i, f in enumerate(video):
        if i >= len(video) - frame_interval:
            break
        prev_frame = video[i]
        next_frame = video[i+frame_interval]
        prev = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        next = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # ff = flow_to_image_np(f)


        flow_ls += [torch.from_numpy(flow).permute(2,0,1)]
        # write_png(ff, os.path.join("flow", "{:04}.png".format(i)))
    flows = torch.stack(flow_ls)
    return flows

from torchvision.datasets.video_utils import VideoClips

    
class Panda(Dataset):
    def __init__(
            self,
            video_folder,
            sample_size=512, sample_n_frames=14,
            fps_range = [7,7],
            return_end_frames = False
        ):
        # self.dataset = glob.glob(os.path.join(video_folder, "**" , "*.mp4"), recursive=True)
        with open(os.path.join(video_folder, "video_files.json"), 'r') as f:
            self.dataset = json.load(f)
        random.shuffle(self.dataset)
        cache_dir = ".cache/panda.clips"
        if os.path.exists(cache_dir):
            _precomputed_metadata = torch.load(cache_dir)
        else:
            _precomputed_metadata = None
        self.video_clips = VideoClips(
            self.dataset,
            sample_n_frames + 1,
            32,
            num_workers=16,
            _precomputed_metadata = _precomputed_metadata
        )
        if not os.path.exists(cache_dir):
            os.makedirs(".cache", exist_ok=True)
            torch.save(self.video_clips.metadata, cache_dir)


        self.full_dataset = self.dataset
        self.length = self.video_clips.num_clips()
        # print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_n_frames = sample_n_frames
        # self.motion_values_folder=motion_folder
        print("length",self.video_clips.num_clips())
        self.sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)
        self.pixel_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            # T.Resize(sample_size),
            # T.CenterCrop(sample_size),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.fps_range = fps_range
        self.return_end_frames = return_end_frames

    




    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]

    def metadata(self):
        return self.video_clips.metadata

    def get_batch(self, idx):
    

    
        while True:
            # preprocessed_dir = self.dataset[idx]
            # video_name = preprocessed_dir.split(os.sep)[-1]
    
            # Sort and limit the numbqer of image and depth files to 14
            # print(preprocessed_dir)
            # video_clips = VideoClips([preprocessed_dir], self.sample_n_frames + 1, frames_between_clips=32)
            frames, audio, info, video_idx = self.video_clips.get_clip(idx)

            # frames, _, info = read_video(preprocessed_dir, pts_unit="sec", end_pts=5.0)
            fps = info["video_fps"]
            frame_len = len(frames)
            if frame_len < self.sample_n_frames + 1:
                idx = random.randint(0, len(self.dataset) - 1)
                continue
            # rand_start = random.randint(0, max(frame_len - self.sample_n_frames - 1, 0))
            # frames = frames[rand_start:rand_start + self.sample_n_frames + 1]
    
            video_path = self.video_clips.video_paths[video_idx]
            caption_path = video_path.replace(".mp4", ".txt")
            with open(caption_path, "r") as f:
                caption = f.read()

            frame_len = len(frames)
            sample_len = self.sample_n_frames + 1

            assert frame_len >= sample_len, f"Require at least {sample_len} frames, but only {frame_len} frames found"

            # Check if there are enough frames for both image and depth
            target_fps = random.randint(*self.fps_range)
            
            frame_interval = int(fps // target_fps)
            frame_interval = min(max(frame_interval, 1), frame_len // sample_len) 

            # frame_interval_range = (1, frame_len // (sample_len))
            # rand_interval = random.randint(*frame_interval_range)

            # fps = fps / rand_interval
            fps = fps / frame_interval


            rand_start = random.randint(0, max(frame_len - (sample_len) * frame_interval, 0))
            frames = frames[rand_start:rand_start + sample_len * frame_interval:frame_interval]
    
            # Load image frames

            pixel_values = frames
        
            # Load depth frames
            
            # flow_path = preprocessed_dir.replace(".mp4", ".flo")
            # if os.path.exists(flow_path):
            #     depth_pixel_values = readBatchedFlow(flow_path)[rand_start:rand_start + self.sample_n_frames]
            #     depth_pixel_values = torch.from_numpy(depth_pixel_values.transpose(0, 3, 1, 2))
            #     # depth_pixel_values = optical_flow_normalize(depth_pixel_values)
            # else:
            #     depth_pixel_values = compute_flow(pixel_values)

            pixel_values = pixel_values.permute(0, 3, 1, 2) / 255.
            # captions = [caption for i in range(len)]
            # numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(depth_folder, df))) for df in depth_files])
            # depth_pixel_values = numpy_to_pt(numpy_depth_images)

            # motion_bucket_ids = flow_to_motion_strength(depth_pixel_values)
            # print(motion_bucket_ids.shape)
    
            return pixel_values, fps, caption
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        #while True:
           # try:
        pixel_values, fps, caption = self.get_batch(idx)
           #     break
          #  except Exception as e:
          #      print(e)
          #      idx = random.randint(0, self.length - 1)
        if self.return_end_frames:
            # pixel_values = pixel_values[[0,-2]]
            video_len = len(pixel_values)
            pixel_values = torch.cat([pixel_values[[i, i + 24]] for i in range(0, video_len - 24, 2)], dim = 0)
            caption = [caption for i in range(len(pixel_values))]
        
        
        # input_tensor = torch.cat([pixel_values, depth_pixel_values], dim = 1)
        # input_tensor = process_frames(input_tensor, *self.sample_size)
        # pixel_values, depth_pixel_values = input_tensor[:,:3], input_tensor[:,3:]

        pixel_values = process_frames(pixel_values, *self.sample_size)
        


        pixel_values = self.pixel_transforms(pixel_values)
        # sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values, motion_values=motion_bucket_ids)


        sample = dict(pixel_values=pixel_values, fps = fps, caption = caption)
        return sample

    def select(self, range):
        self.dataset = [self.full_dataset[i] for i in range]
        self.length = len(self.dataset)
        return self

class MixDataset(Dataset):

    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        self.length = sum([len(dataset) for dataset in datasets])

        print("Mix Dataset Length", self.length)
    
    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError

class MiniDataset(Dataset):
    def __init__(
            self,
            video_folder,
            repeat_num = 10,
            sample_size=512, sample_n_frames=25,
        ):
        video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
        self.dataset = []
        for videof in video_files:
            frames, _, info = read_video(videof, pts_unit="sec")
            fps = info["video_fps"]
            frames = frames.permute(0, 3, 1, 2) / 255.
            self.dataset += [(frames, fps)]
        self.dataset = self.dataset * repeat_num

        self.length = len(self.dataset)
        print("length",self.length)
        self.sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)

        self.sample_n_frames = sample_n_frames
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        #while True:
           # try:
        frames, fps = self.dataset[idx]

        frame_len = len(frames)
        sample_len = self.sample_n_frames + 1

        assert frame_len >= sample_len, f"Require at least {sample_len} frames, but only {frame_len} frames found"
        
        frame_interval_range = (1, frame_len // (sample_len))
        rand_interval = random.randint(*frame_interval_range)

        fps = fps / rand_interval

        rand_start = random.randint(0, max(frame_len - (sample_len) * rand_interval, 0))
        frames = frames[rand_start:rand_start + sample_len * rand_interval:rand_interval]
        
        
        # input_tensor = torch.cat([pixel_values, depth_pixel_values], dim = 1)
        # input_tensor = process_frames(input_tensor, *self.sample_size)
        # pixel_values, depth_pixel_values = input_tensor[:,:3], input_tensor[:,3:]

        pixel_values = process_frames(frames, *self.sample_size)

        if random.random() < 0.5:
                pixel_values= torch.flip(pixel_values, dims=[-1])  # 水平方向翻转 左右翻转
        pixel_values = pixel_values * 2. - 1.
        
        sample = dict(pixel_values=pixel_values, fps = fps)
        return sample


class MSRVTT(Dataset):
    def __init__(
            self,
            video_folder,
            caption_file,
            sample_size=512, sample_n_frames=2,
            clip_length = 16
        ):
        self.dataset = glob.glob(os.path.join(video_folder, "*.mp4"))
        with open(os.path.join(caption_file), 'r') as f:
            self.captions = json.load(f)

        cache_dir = ".cache/msrvtt.clips"
        if os.path.exists(cache_dir):
            _precomputed_metadata = torch.load(cache_dir)
        else:
            _precomputed_metadata = None
        self.video_clips = VideoClips(
            self.dataset,
            clip_length_in_frames = clip_length,
            frames_between_clips = 1,
            frame_rate = 7,
            num_workers=16,
            _precomputed_metadata = _precomputed_metadata
        )
        if not os.path.exists(cache_dir):
            os.makedirs(".cache", exist_ok=True)
            torch.save(self.video_clips.metadata, cache_dir)


        self.full_dataset = self.dataset
        self.length = self.video_clips.num_clips()
        # print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_n_frames = sample_n_frames
        # self.motion_values_folder=motion_folder
        print("length",self.video_clips.num_clips())
        self.sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)
        self.to_tensor = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def metadata(self):
        return self.video_clips.metadata

    def get_batch(self, idx):
    

    
        while True:
            # preprocessed_dir = self.dataset[idx]
            # video_name = preprocessed_dir.split(os.sep)[-1]
    
            # Sort and limit the numbqer of image and depth files to 14
            # print(preprocessed_dir)
            # video_clips = VideoClips([preprocessed_dir], self.sample_n_frames + 1, frames_between_clips=32)
            frames, audio, info, video_idx = self.video_clips.get_clip(idx)

            # frames, _, info = read_video(preprocessed_dir, pts_unit="sec", end_pts=5.0)
            frame_len = len(frames)
            # rand_start = random.randint(0, max(frame_len - self.sample_n_frames - 1, 0))
            # frames = frames[rand_start:rand_start + self.sample_n_frames + 1]
            pdb.set_trace()

            video_path = self.video_clips.video_paths[video_idx]
            video_name = os.path.basename(video_path).split(".")[0]
            captions = self.captions[video_name]

            perm = torch.randperm(frames.size(0))
            sample_idx = perm[:self.sample_n_frames]
            sampled_frames = frames[sample_idx]

            pixel_values = sampled_frames.permute(0, 3, 1, 2) / 255.

    
            return pixel_values, captions
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        pixel_values, captions = self.get_batch(idx)

        pixel_values = process_frames(pixel_values, *self.sample_size)
        
        # sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values, motion_values=motion_bucket_ids)
        pixel_values = self.to_tensor(pixel_values)
        frame_len = pixel_values.shape[0]

        input_captions = [random.choice(captions), random.choice(captions)]

        input_dict = {
            "text": input_captions
        }

        # input_dict = self.transform(input_dict, only_caption = True)
        # input_dict["text"] = input_dict["text"][:frame_len]
        # pdb.set_trace()

        # input_dict["x_pixel_values"] = input_dict["x_pixel_values"][0]
        # input_dict["y_pixel_values"] = input_dict["y_pixel_values"][0]
        input_dict["x_pixel_values"] = pixel_values[:frame_len // 2]
        input_dict["y_pixel_values"] = pixel_values[frame_len // 2:]
        # input_dict["x_tracks"] = source_tracks
        # input_dict["y_tracks"] = target_tracks            

        # input_dict["x_input_ids"] = input_dict["x_input_ids"][0]
        # input_dict["y_input_ids"] = input_dict["y_input_ids"][0]
        return input_dict

    def select(self, rg):
        self.dataset = [self.full_dataset[i] for i in rg]
        self.length = len(self.dataset)
        return self

class PandaN(Dataset):
    def __init__(
            self,
            dataset_file,
            sample_size=512, sample_n_frames=16,
            clip_length = 16
        ):
        # self.dataset = glob.glob(os.path.join(video_folder, "*.mp4"))
        with open(os.path.join(dataset_file), 'r') as f:
            self.dataset = json.load(f)

        cache_dir = f".cache/panda.clips"
        if os.path.exists(cache_dir):
            _precomputed_metadata = torch.load(cache_dir)
        else:
            _precomputed_metadata = None
        self.video_clips = VideoClips(
            self.dataset,
            clip_length_in_frames = clip_length,
            frames_between_clips = 1,
            frame_rate = 7,
            num_workers=32,
            _precomputed_metadata = _precomputed_metadata
        )
        if not os.path.exists(cache_dir):
            os.makedirs(".cache", exist_ok=True)
            torch.save(self.video_clips.metadata, cache_dir)


        self.full_dataset = self.dataset
        self.length = self.video_clips.num_clips()
        # print(f"data scale: {self.length}")

        self.sample_n_frames = sample_n_frames
        # self.motion_values_folder=motion_folder
        print("length",self.video_clips.num_clips())
        self.sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)
        self.to_tensor = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def metadata(self):
        return self.video_clips.metadata

    def get_batch(self, idx):
    

    
        while True:
            # preprocessed_dir = self.dataset[idx]
            # video_name = preprocessed_dir.split(os.sep)[-1]
    
            # Sort and limit the numbqer of image and depth files to 14
            # print(preprocessed_dir)
            # video_clips = VideoClips([preprocessed_dir], self.sample_n_frames + 1, frames_between_clips=32)
            # pdb.set_trace()
            frames, audio, info, video_idx = self.video_clips.get_clip(idx)

            # frames, _, info = read_video(preprocessed_dir, pts_unit="sec", end_pts=5.0)
            frame_len = len(frames)
            # rand_start = random.randint(0, max(frame_len - self.sample_n_frames - 1, 0))
            # frames = frames[rand_start:rand_start + self.sample_n_frames + 1]
            # pdb.set_trace()

            video_path = self.video_clips.video_paths[video_idx]
            caption_path = video_path.replace(".mp4", ".txt")
            with open(caption_path, "r") as f:
                caption = f.read()

            perm = torch.randperm(frames.size(0))
            sample_idx = perm[:self.sample_n_frames]
            sampled_frames = frames[sample_idx]

            pixel_values = sampled_frames.permute(0, 3, 1, 2) / 255.

    
            return pixel_values, caption
    
    def __len__(self):
        return self.length

    def with_transform(self, transform):
        self.transform = transform
        return self

    def __getitem__(self, idx):

        pixel_values, caption = self.get_batch(idx)

        pixel_values = process_frames(pixel_values, *self.sample_size)
        
        # sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values, motion_values=motion_bucket_ids)
        pixel_values = self.to_tensor(pixel_values)
        frame_len = pixel_values.shape[0]

        input_captions = [caption for i in range(frame_len // 2)]

        input_dict = {
            "text": input_captions
        }

        input_dict = self.transform(input_dict, only_caption = True)
        # input_dict["text"] = input_dict["text"][:frame_len]
        # pdb.set_trace()

        # input_dict["x_pixel_values"] = input_dict["x_pixel_values"][0]
        # input_dict["y_pixel_values"] = input_dict["y_pixel_values"][0]
        input_dict["x_pixel_values"] = pixel_values[:frame_len // 2]
        input_dict["y_pixel_values"] = pixel_values[frame_len // 2:]
        # input_dict["x_tracks"] = source_tracks
        # input_dict["y_tracks"] = target_tracks            

        input_dict["x_input_ids"] = input_dict["x_input_ids"]
        input_dict["y_input_ids"] = input_dict["y_input_ids"]
        return input_dict

    def select(self, rg):
        self.dataset = [self.full_dataset[i] for i in rg]
        self.length = len(self.dataset)
        return self



class TrackDataset(Dataset):
    def __init__(
        self, 
        annotation_files,
        text_annotation,
        bucket_root,
        image_root = None,
        points_root=None,
        mask_root=None,
        annotation_max=-1,
        shuffle_once=True,
        min_dist=-1,
        max_dist=-1,
        max_frame_idx=128,
        num_points=None,
        mask_rate = 1.0,
        transform = None,
        resolution = 512,
        drop_track_rate = 0.1
    ):
        self.bucket_root = bucket_root
        self.image_root = image_root

        # Shuffle annotations
        self.data = []
        for f in annotation_files:
            self.data.extend(json.load(open(f)))
        self.texts = json.load(open(text_annotation))
        if shuffle_once:
            self.data = np.random.permutation(self.data).tolist()
        if annotation_max > 0:
            self.data = self.data[:annotation_max]

        self.points_root = points_root
        self.mask_root = mask_root
        self.max_frame_idx = max_frame_idx
        self.num_points = num_points

        self.data = train_helpers.filter_dist(self.data, min_dist, max_dist)
        # Filter for max frame index due to point tracking
        self.data = [item for item in self.data if max(train_helpers.get_frame_idx(item["source"]), train_helpers.get_frame_idx(item["target"])) < max_frame_idx]
        # Filter for only samples where the pseudo label exists
        if points_root:
            self.data = train_helpers.filter_anns(self.data, f"{bucket_root}/{points_root}/*", "video_name")
        
        self.mask_rate = mask_rate
        self.transform = transform

        self.to_tensor = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.resolution = resolution
        self.one_image = Image.new('L', (self.resolution, self.resolution), 255)
        self.drop_track_rate = drop_track_rate
    
    def with_transform(self, transform):
        self.transform = transform
        return self
    
    def proc_frame_with_tracks(self, source_frame, target_frame, tracks):
        # pdb.set_trace()

        # resized_source_frame, resized_target_frame = transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR)([source_frame, target_frame])
        # resize_factor = resized_source_frame.height / source_frame.height
        # tracks *= resize_factor

        # cropped_source_frame, cropped_target_frame = transforms.CenterCrop(self.resolution)([resized_source_frame, resized_target_frame])

        # height_shift, width_shift = (resized_source_frame.height - cropped_source_frame.height) // 2, (resized_source_frame.width - cropped_source_frame.width) // 2

        # tracks -= np.array([height_shift, width_shift]).reshape(1,1,-1)

        # tracks = torch.tensor(tracks).to(int)

        source_tensor, target_tensor = self.to_tensor([source_frame, target_frame])
        # source_tracks = torch.zeros([5, *source_tensor.shape[-2:]])



        # target_image_pil_track = train_helpers.draw_tracks_target(cropped_target_image, tracks)
        # source_image_pil_track = train_helpers.draw_tracks_target(cropped_source_image, tracks[::-1])
        # target_image_pil_track2 = target_tracks
        # source_image_pil_track2 = source_tracks


        source_tracks, target_tracks = Image.new('RGBA', source_frame.size, 0), Image.new('RGBA', target_frame.size, 0)
        # rand_color = (torch.rand(3) * 255).to(int).tolist()
        source_tracks, target_tracks = train_helpers.draw_points(source_tracks, target_tracks, tracks)
        # pdb.set_trace()
        source_tracks, target_tracks = source_tracks.convert("RGB"), target_tracks.convert("RGB")



        source_tracks_tensor, target_tracks_tensor = self.to_tensor([source_tracks, target_tracks])
        # target_tracks = train_helpers.draw_points(target_tracks, tracks, color = rand_color, target = True)

        # source_tracks = torch.zeros([3, *source_tensor.shape[-2:]])
        # target_tracks = torch.zeros_like(source_tracks)
        # pdb.set_trace()
        # for src_point, tgt_point in zip(tracks[0].astype(np.int32), tracks[1].astype(np.int32)):
                
        #     # if True in (src_point < 0) or True in (src_point >= self.resolution) or True in (tgt_point < 0) or True in (tgt_point >= self.resolution):
        #         # continue
        #     source_tracks[:, src_point[1], src_point[0]] = torch.rand(3)
        #     target_tracks[:, tgt_point[1], tgt_point[0]] = source_tracks[:, src_point[1], src_point[0]]
            # source_tracks[1:] /= self.resolution
            # target_tracks[1:] /= self.resolution
            # source_tracks[:, src_point[1], src_point[0]] = torch.tensor([1, src_point[0], src_point[1], tgt_point[0], tgt_point[1]])
            # target_tracks[:, tgt_point[1], tgt_point[0]] = torch.tensor([1, tgt_point[0], tgt_point[1], src_point[0], src_point[1]])
            # source_tracks[1:] /= self.resolution
            # target_tracks[1:] /= self.resolution
        
        return source_tensor, target_tensor, source_tracks_tensor, target_tracks_tensor
        


        




    def __getitem__(self, idx):
        item = self.data[idx]
        # print(item)
        source_image_pil = Image.open(os.path.join(self.bucket_root, self.image_root, item["source"]))
        target_image_pil = Image.open(os.path.join(self.bucket_root, self.image_root, item["target"]))

        resized_source_image, resized_target_image = T.Resize(self.resolution)(source_image_pil), T.Resize(self.resolution)(target_image_pil)

        resize_factor = resized_source_image.height / source_image_pil.height

        # w_range, h_range = (resized_source_image.width - self.resolution) // 2, (resized_source_image.height - self.resolution) // 2
        x_offset = random.randint(0, resized_source_image.width - self.resolution)
        y_offset = random.randint(0, resized_source_image.height - self.resolution)

        cropped_source_image, cropped_target_image = resized_source_image.crop((x_offset, y_offset, x_offset + self.resolution, y_offset + self.resolution)), resized_target_image.crop((x_offset, y_offset, x_offset + self.resolution, y_offset + self.resolution))




        i, j = train_helpers.get_frame_idx(item["source"]), train_helpers.get_frame_idx(item["target"])

        # Load point tracks
        video_name = item["source"].split("/")[0]
        tracks, visibles = train_helpers.open_points(i, j, video_name, self.max_frame_idx, self.bucket_root, self.points_root)
        

        crop_mask = Image.new('L', source_image_pil.size, 0)
        crop_mask.paste(self.one_image, (x_offset, y_offset))


        # mask = train_helpers.open_mask(i, video_name, self.bucket_root, self.mask_root)
        # pdb.set_trace()
        # if random.random() < self.mask_rate:
        if random.random() < self.mask_rate:
            obj_mask = train_helpers.open_mask(i, video_name, self.bucket_root, self.mask_root).convert("L")
            # print("With mask")
            mask = ImageChops.multiply(crop_mask, obj_mask)
        else:
            mask = crop_mask
        # else:
        #     mask = None

        height, width = source_image_pil.height, source_image_pil.width
        # print(tracks.shape)
        tracks_full = train_helpers.filter_tracks(tracks, width, height, mask, self.num_points)

        if tracks_full.shape[1] == 0:
            mask = crop_mask
            tracks_full = train_helpers.filter_tracks(tracks, width, height, mask, self.num_points)


        # print(tracks_full.shape)
        sample_idx = train_helpers.sample_track(tracks_full, num_samples=random.randint(1, 10))
        tracks = tracks_full[:, sample_idx, :]

        tracks = tracks * resize_factor - np.array([x_offset, y_offset]).reshape(1,1,-1)

        # target_image_pil_track = train_helpers.draw_tracks_target(cropped_traget_image, tracks)
        # source_image_pil_track = train_helpers.draw_tracks_target(cropped_source_image, tracks[::-1])

        # pdb.set_trace()

        input_dict = {
            "source_frame": [source_image_pil],
            # "target_frame": [target_image_pil_track],
            "target_frame": [target_image_pil],
            "text": self.texts[video_name]
        }

        # source_tensor, target_tensor, source_tracks, target_tracks = self.proc_frame_with_tracks(source_image_pil, target_image_pil, tracks)



        source_tensor, target_tensor, source_tracks, target_tracks = self.proc_frame_with_tracks(cropped_source_image, cropped_target_image, tracks)

        if random.random() < self.drop_track_rate:
            source_tracks = torch.zeros_like(source_tracks) - 1
            target_tracks = torch.zeros_like(target_tracks) - 1

        input_dict = self.transform(input_dict, only_caption = True)

        # pdb.set_trace()

        # input_dict["x_pixel_values"] = input_dict["x_pixel_values"][0]
        # input_dict["y_pixel_values"] = input_dict["y_pixel_values"][0]
        input_dict["x_pixel_values"] = source_tensor
        input_dict["y_pixel_values"] = target_tensor
        input_dict["x_tracks"] = source_tracks
        input_dict["y_tracks"] = target_tracks            

        input_dict["x_input_ids"] = input_dict["x_input_ids"][0]
        input_dict["y_input_ids"] = input_dict["y_input_ids"][0]

        return input_dict

    def __len__(self):
        return len(self.data)
    
from omegaconf import OmegaConf

if __name__ == "__main__":
    config_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/train_models/train_configs/track_dataset.yaml"
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    # train_dataset, train_dataloader = get_correspondence_loader(config, config["train_file"], True)

    dataset = TrackDataset(**config)

    data_iter = iter(dataset)
    next_data = next(data_iter)

# if __name__ == "__main__":
#     # from utils.util import save_videos_grid

#     # dataset = WebVid10M(
#     #     csv_path="/data/webvid/results_2M_train.csv",
#     #     video_folder="/data/webvid/data/videos",
#     #     sample_size=256,
#     #     sample_stride=4, sample_n_frames=16,
#     #     is_image=True,
#     # # )
#     # dataset = Panda(
#     #     # csv_path="/data/webvid/results_2M_train.csv",
#     #     video_folder="/data/juicefs_sharing_data/72179586/data/panda70m/test_split_webdataset",
#     #     depth_folder="/data/juicefs_sharing_data/72179586/data/panda70m/test_split_webdataset",
#     #     sample_size=(320, 576),
#     #     sample_n_frames=14,
#     # )

#     dataset = Panda("/data/vjuicefs_ai_camera/11162591/public_datasets/pandas70m", 
#                     sample_size = (320, 576), sample_n_frames = 31)
#     # print("Panda70M loaded. Total length: ", len(dataset))
#     # dataset = MiniDataset(
#     #     video_folder="data/test_images/lora_input",
#     #     sample_size=(512, 512),
#     #     sample_n_frames=14,
#     #     repeat_num=100
#     # )

#     # import pdb
#     # pdb.set_trace()
#     # flow_model = load_unimatch().to(torch.float16)
#     # checkpoint = torch.load("/data/juicefs_sharing_data/72179586/repos/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth", map_location="cpu")

#     # flow_model.load_state_dict(checkpoint['model'], strict=True)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4)
#     optical_flows = []
#     flow_model = load_unimatch()
#     checkpoint = torch.load("/data/juicefs_sharing_data/11162591/code/lxr/unimatch/models/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth", map_location="cpu")
#     flow_model.load_state_dict(checkpoint['model'], strict=True)
#     # video, _, _ = read_video("/data/juicefs_sharing_data/72179586/repos/VidToMe/data/breakdance.mp4")

#     # depth_pixel_values = inference_flow(
#     #     flow_model,
#     #     inference_video=video.permute(0, 3, 1, 2).unsqueeze(0).to("cuda"),
#     #     padding_factor=flow_args.padding_factor,
#     #     inference_size=(512, 512),
#     #     attn_type=flow_args.attn_type,
#     #     attn_splits_list=flow_args.attn_splits_list,
#     #     corr_radius_list=flow_args.corr_radius_list,
#     #     prop_radius_list=flow_args.prop_radius_list,
#     #     pred_bidir_flow=flow_args.pred_bidir_flow,
#     #     pred_bwd_flow=flow_args.pred_bwd_flow,
#     #     num_reg_refine=flow_args.num_reg_refine,
#     # )
#     # image = flow_to_image(depth_pixel_values[0].cpu())
#     # write_png(image[0], "tmp_image.png")
#     # image = [write_video(f"temp_flow_{i}.mp4", flow_to_image(dpv.cpu()).permute(0, 2, 3, 1), fps=8) for i, dpv in enumerate(depth_pixel_values)]
#     # exit(0)
#     depth_ls = []
#     with torch.autocast(device_type="cuda",dtype=torch.float16):
#         for idx, batch in enumerate(dataloader):
#             print(batch["pixel_values"].shape)
#             frames = batch["pixel_values"].to("cuda")
#             randord = torch.randperm(frames.shape[1])
#             frames = frames[:,randord].flatten(0,1)
            
#             forward_flow, backward_flow = inference_flow_warpper_v2(flow_model, frames, size=(320, 576))

#             wf_prev = warp_frames(forward_flow, frames[1::2])
#             warp_error1 = (wf_prev - frames[::2]).norm(dim=1).mean()
#             # warp_error2 = (wf_prev - frames[1::2]).norm(dim=1).mean()

#             wf_next = warp_frames(backward_flow, frames[::2])
#             # warp_error3 = (wf_next - frames[::2]).norm(dim=1).mean()
#             warp_error4 = (wf_next - frames[1::2]).norm(dim=1).mean()

#             # wf_prev2 = warp_frames(forward_flow, frames[::2])
#             # # warp_error5 = (wf_prev2 - frames[::2]).norm(dim=1).mean()
#             # warp_error6 = (wf_prev2 - frames[1::2]).norm(dim=1).mean()

#             # wf_next2 = warp_frames(backward_flow, frames[1::2])
#             # warp_error7 = (wf_next2 - frames[::2]).norm(dim=1).mean()
#             # warp_error8 = (wf_next2 - frames[1::2]).norm(dim=1).mean()


#             # print(warp_error1, warp_error2, warp_error3, warp_error4, warp_error5, warp_error6,
#                 #   warp_error7, warp_error8)
#             print(warp_error1, warp_error4)
            
            
            
            
            
            
            
            
            
#             # stime = time.time()
#             # depth_pixel_values = inference_flow(
#             #     flow_model,
#             #     inference_video=(batch["pixel_values"] * 255).to("cuda").to(torch.float16),
#             #     padding_factor=flow_args.padding_factor,
#             #     inference_size=(320, 576),
#             #     attn_type=flow_args.attn_type,
#             #     attn_splits_list=flow_args.attn_splits_list,
#             #     corr_radius_list=flow_args.corr_radius_list,
#             #     prop_radius_list=flow_args.prop_radius_list,
#             #     pred_bidir_flow=flow_args.pred_bidir_flow,
#             #     pred_bwd_flow=flow_args.pred_bwd_flow,
#             #     num_reg_refine=flow_args.num_reg_refine,
#             # )
#             # # image = [write_video(f"temp_flow_{i}.mp4", flow_to_image(dpv.cpu(), norm_mode="max").permute(0, 2, 3, 1), fps=8) for i, dpv in enumerate(depth_pixel_values)]
#             # # [write_video(f"temp_video_{i}.mp4", (video.permute(0, 2, 3, 1)* 255).to(torch.uint8), fps=8) for i, video in enumerate(batch["pixel_values"])]
#             # # exit(0)

#             # print("Inferemce Time:", time.time() - stime)
#             # print(depth_pixel_values.shape)

#             # depth_ls += [depth_pixel_values]

#             # if idx % 10 == 0:
#             #     cur_depth = torch.cat(depth_ls)
#             #     cur_mean = cur_depth.mean()
#             #     print(f"{len(cur_depth)} samples mean: {cur_mean}, std: {cur_depth.std()}")

#     # optical_flows = torch.cat(optical_flows, dim = 0)
#     # # print(optical_flows.mean(), optical_flows.std())
#     # mean_flows = []
#     # for of in optical_flows:
#     #     mean_flows += [of.norm(dim=1).mean()]
#     # import matplotlib.pyplot as plt
#     # plt.hist(mean_flows)
#     # plt.savefig("tmp.png")


#         # for i in range(batch["pixel_values"].shape[0]):
#         #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)