from copy import deepcopy
import sys
import os

import numpy as np
import cv2

from collections import namedtuple
import torch
import torch.nn.functional as F
import argparse


# import modules.paths as ph
import gc


from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import Raft_Large_Weights

# RAFT_model = None
# fgbg = cv2.createBackgroundSubtractorMOG2(
#     history=500, varThreshold=16, detectShadows=True)
MODEL_PATH = "/home/lixirui/Repos/RAFT/models"
RAFT_LARGE = None

# def background_subtractor(frame, fgbg):
#     fgmask = fgbg.apply(frame)
#     return cv2.bitwise_and(frame, frame, mask=fgmask)

def RAFT_clear_memory():
    global RAFT_LARGE
    del RAFT_LARGE
    gc.collect()
    torch.cuda.empty_cache()
    RAFT_LARGE = None

@torch.no_grad()
def RAFT_estimate_flow_torchvision(img1_batch, img2_batch, bidir = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global RAFT_LARGE
    if RAFT_LARGE is None:
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
        model = model.eval()
        RAFT_LARGE = model
    
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    img1_batch, img2_batch = transforms(img1_batch, img2_batch)

    torch.cuda.empty_cache()
    if bidir:
        img1 = torch.cat([img1_batch, img2_batch])
        img2 = torch.cat([img2_batch, img1_batch])
        list_of_flows = RAFT_LARGE(img1.to(device), img2.to(device))
        forward_flow, backward_flow = torch.chunk(list_of_flows[-1], 2)

        return forward_flow, backward_flow
    else:
        list_of_flows = RAFT_LARGE(img1_batch.to(device), img2_batch.to(device))
        flow = list_of_flows[-1]
        return flow


# def frames_norm(frame): return frame / 127.5 - 1


# def flow_norm(flow): return flow / 255


# def occl_norm(occl): return occl / 127.5 - 1


# def frames_renorm(frame): return (frame + 1) * 127.5


# def flow_renorm(flow): return flow * 255


# def occl_renorm(occl): return (occl + 1) * 127.5