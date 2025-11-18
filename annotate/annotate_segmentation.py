# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

import argparse
import glob
import json
import numpy as np
import os
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from transformers import pipeline
import matplotlib
import torchvision.transforms as T
import torch
import sys
sys.path.append("/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/ControlNet-v1-1-nightly")
from annotator.oneformer import OneformerCOCODetector, OneformerADE20kDetector

import pdb

def filter_paths(paths, filter_file=None):
    if filter_file is not None:
        filter_names = set(json.load(open(filter_file)))
        paths = [image for image in paths if os.path.basename(image) in filter_names]
    return paths

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

def main(args):
    config = OmegaConf.load(args.config_path)
    paths = glob.glob(config["read_root"])
    paths = filter_paths(paths, config.get("filter_file"))
    paths = sorted(paths)
    lpath = len(paths)

    # Create folders
    save_folder = config["save_root"]

    make_folder(f"{save_folder}")

    # Init pipeline
    # pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    # cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    preprocessor = OneformerCOCODetector()
    for image in tqdm(paths[lpath // 2:]):
        img = Image.open(image)
        # Predict pseudo label
        img_arr = np.array(img)
        seg_arr = preprocessor(img_arr)
        seg_image = Image.fromarray(seg_arr)

        seg_image.save(os.path.join(f"{save_folder}",f"{os.path.basename(image)}"))
        # Optionally save metadata
        # pdb.set_trace()
        # if save_meta and meta is not None:
        #     meta["size"] = new_res
        #     json.dump(meta, open(f"{save_folder}_meta/{os.path.basename(image).split('.')[0]}.json", "w"))

if __name__ == "__main__":
    # conda run -n control python3 annotate_spatial.py --config_path configs/annotate_spatial.yaml
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--config_path", type=str)
	args = parser.parse_args()
	main(args)