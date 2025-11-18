import numpy as np
import torch.nn.functional as F
import torch

from utils.optical_flow import compute_flow, inference_flow_warpper
# MOTION_LINE = [0.02601298, 0.43814413]

# A * motion_bucket_ids / fps + B * (1 / fps) + C * motion_bucket_ids + D
# MOTION_PARAM = np.array([3.38139974e-01, 5.71826948e+00, 2.87310879e-03, 1.85558449e-01]) 
# # def motion2flow(motion_bucket_id):
# #     flow_norm = motion_bucket_id * MOTION_LINE[0] + MOTION_LINE[1]
# #     return flow_norm

# # def flow2motion(flow_norm):
# #     motion_bucket_id = (flow_norm - MOTION_LINE[1]) / MOTION_LINE[0]
# #     return max(int(motion_bucket_id + 0.5), 0)


# def motion2flow(motion_bucket_id, fps):
#     motion_variables = np.array([1., motion_bucket_id, motion_bucket_id / fps, 1. / fps])
#     flow_norm = (motion_variables * MOTION_PARAM).sum()
#     return flow_norm

# def flow2motion(flow_norm, fps):
#     motion_bucket_id = (flow_norm - MOTION_LINE[1]) / MOTION_LINE[0]
#     return max(int(motion_bucket_id + 0.5), 0)


# MOTION_PARAM = np.array([3.38139974e-01, 5.71826948e+00, 2.87310879e-03, 1.85558449e-01]) 



# def motion2flow(motion_bucket_id, fps):
#     motion_variables = np.array([motion_bucket_id / fps, 1. / fps, motion_bucket_id, 1])
#     flow_norm = (motion_variables * MOTION_PARAM).sum()
#     return flow_norm

# def flow2motion(flow_norm, fps):
#     # flow_norm = MOTION_PARAM[0] * motion_bucket_id / fps + MOTION_PARAM[1] * 1. / fps + MOTION_PARAM[2] * motion_bucket_id + MOTION_PARAM[3]
#     motion_bucket_id = (flow_norm - MOTION_PARAM[3] - MOTION_PARAM[1] * 1. / fps) / (MOTION_PARAM[0] / fps + MOTION_PARAM[2])
#     return max(int(motion_bucket_id), 0)


MOTION_PARAM = np.array([0.07218373,2.6522603,0.00323807,0.2210316])


def motion2flow(fps, motion_bucket_id):
    motion_variables = np.array([motion_bucket_id / fps, 1. / fps, motion_bucket_id, 1])
    motion_score = (motion_variables * MOTION_PARAM).sum()
    return motion_score

def flow2motion(fps, motion_score = None, flow = None):
    assert motion_score is not None or flow is not None
    if motion_score is None:
        motion_score = flow            
        scale_factor = 16. / min(flow.shape[-2:])
        flow = F.interpolate(flow, scale_factor = scale_factor, mode="bilinear")
        motion_score = flow.abs().mean().item()
    # flow_norm = MOTION_PARAM[0] * motion_bucket_id / fps + MOTION_PARAM[1] * 1. / fps + MOTION_PARAM[2] * motion_bucket_id + MOTION_PARAM[3]
    motion_bucket_id = (motion_score - MOTION_PARAM[3] - MOTION_PARAM[1] * 1. / fps) / (MOTION_PARAM[0] / fps + MOTION_PARAM[2])
    return int(np.clip(motion_bucket_id, a_max=255, a_min=0))

MOTION_PARAM_SIMPLE = [0.06741976, 1.15129627]

def bucket2motion(motion_bucket_id):
    motion_score = motion_bucket_id * MOTION_PARAM_SIMPLE[0] + MOTION_PARAM_SIMPLE[1]
    return motion_score

def motion2bucket(motion_score):
    motion_bucket_id = (motion_score - MOTION_PARAM_SIMPLE[1]) / MOTION_PARAM_SIMPLE[0]
    return int(min(max(motion_bucket_id, 0), 255))


def cal_motion_bucket_ids(pixel_values, fps = 7, flow_model = None):
    frame_intervals = [int(fpsi.item() // 2) for fpsi in fps]
    if flow_model is not None:
        flow_fps2 = inference_flow_warpper(flow_model, pixel_values, pixel_values.shape[-2:], frame_interval = frame_intervals)
    else:
        flow_fps2 = compute_flow(pixel_values, frame_interval=frame_intervals)
    motion_scores = [fl.abs().mean().item() for fl in flow_fps2]
    motion_bucket_ids = torch.tensor([motion2bucket(motion_score) for motion_score in motion_scores])
    return motion_bucket_ids
