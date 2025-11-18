
from einops import rearrange
from unimatch.unimatch import UniMatch
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# FLOW_MEAN = 0.053626976907253265
FLOW_MEAN = -0.010683227330446243
# FLOW_STD = 5.732870578765869
FLOW_STD = 5.01635217666626

DEFAULT_CONFIG = {
    "feature_channels": 128,
    "num_scales": 2,
    "upsample_factor": 4,
    "num_head": 1,
    "ffn_dim_expansion": 4,
    "num_transformer_layers": 6,
    "reg_refine": True,
    "task": "flow",
    "padding_factor": 16,
    "attn_type": 'swin',
    "attn_splits_list": [2,8],
    "corr_radius_list": [-1,4],
    "prop_radius_list": [-1,1],
    "num_reg_refine": 1,
    "pred_bidir_flow": False,
    "pred_bwd_flow": False
}

args = OmegaConf.create(DEFAULT_CONFIG)



def optical_flow_normalize(tensor):
    # return (tensor - FLOW_MEAN) / FLOW_STD
    return tensor

def optical_flow_unnormalize(tensor):
    # return tensor * FLOW_STD + FLOW_MEAN
    return tensor


def optical_flow_expand(of: torch.Tensor):
    assert of.shape[-3] == 2
    flow_norm = of.norm(dim=-3)
    flow_angle = torch.atan2(of[:,:,1,:,:], of[:,:,0,:,:]) / torch.pi
    flow_polar = torch.stack([flow_norm, flow_angle], dim=-3)
    return torch.cat([of, flow_polar], dim = -3)

def optical_flow_squeeze(of: torch.Tensor, use_polar = True):
    assert of.shape[-3] == 4
    flow_norm, flow_angle = of[:,:,2,:,:], of[:,:,3,:,:] * torch.pi
    flow_x, flow_y = torch.cos(flow_angle) * flow_norm, torch.sin(flow_angle) * flow_norm

    return torch.stack([flow_x, flow_y], dim = -3)

FLOW_CLIP_MAX = 50
FLOW_NORM_CLIP_MAX = np.sqrt(2 * FLOW_CLIP_MAX**2)
FLOW_LATENT_MEAN = 0.5020191669464111
FLOW_LATENT_STD = 1.2818458080291748

def optical_flow_latent_normalize(tensor, scale = 1):
    tensor_dtype = tensor.dtype
    tensor = tensor.to(torch.float32)
    scaled_flow = tensor * scale
    normalized_scaled_flow = (scaled_flow - FLOW_LATENT_MEAN) / FLOW_LATENT_STD
    normalized_flow = normalized_scaled_flow / scale
    normalized_flow = normalized_flow.to(tensor_dtype)
    return normalized_flow
    # return tensor

def optical_flow_latent_unnormalize(tensor):
    return tensor * FLOW_LATENT_STD + FLOW_LATENT_MEAN
    # return tensor

def flow_to_image_naive(flow):
    # flow_norm = flow.norm(dim=-3)
    # flow_angle = torch.atan2(flow[...,1,:,:], flow[...,0,:,:]) / torch.pi
    # flow_angle[flow_norm < 1] = 1
    # flow_polar = torch.stack([flow_norm, flow_angle], dim=-3)

    flow_clip = torch.clip(flow, min = -FLOW_CLIP_MAX, max=FLOW_CLIP_MAX) / FLOW_CLIP_MAX
    flow_clip = (flow_clip + 1) / 2
    # flow_residual = 
    # flow_angle = (flow_angle + 1) / 2
    # flow_norm = flow.norm(dim=-3, keepdim=True) / FLOW_NORM_CLIP_MAX

    return torch.cat([torch.zeros_like(flow[...,[0],:,:]), flow_clip], dim = -3)

def image_to_flow_naive(flow_image):
    # flow_norm = flow.norm(dim=-3)
    # flow_angle = torch.atan2(flow[...,1,:,:], flow[...,0,:,:]) / torch.pi
    # flow_angle[flow_norm < 1] = 1
    # flow_polar = torch.stack([flow_norm, flow_angle], dim=-3)
    flow = flow_image[...,1:,:,:]
    flow = flow * 2 - 1
    flow = flow * FLOW_CLIP_MAX
    # flow_clip = torch.clip(flow, min = -FLOW_CLIP_MAX, max=FLOW_CLIP_MAX) / FLOW_CLIP_MAX
    # flow_clip = (flow_clip + 1) / 2
    # flow_angle = (flow_angle + 1) / 2
    # flow_norm = flow.norm(dim=-3, keepdim=True) / FLOW_NORM_CLIP_MAX

    return flow

def load_unimatch(device="cuda"):
    model = UniMatch(feature_channels=args.feature_channels,
                     num_scales=args.num_scales,
                     upsample_factor=args.upsample_factor,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_layers=args.num_transformer_layers,
                     reg_refine=args.reg_refine,
                     task=args.task).to(device)
    return model

def inference_flow_warpper(flow_model, video, size, to_uint8 = True, frame_interval = 1):
    if to_uint8:
        video = (video * 255)
        
    video = video.to("cuda")

    if len(video.shape) == 4:
        video = video.unsqueeze(0)

    with torch.autocast(device_type="cuda",dtype=torch.float16):

        flow = inference_flow(
            flow_model,
            inference_video=video,
            padding_factor=args.padding_factor,
            inference_size=size,
            attn_type=args.attn_type,
            attn_splits_list=args.attn_splits_list,
            corr_radius_list=args.corr_radius_list,
            prop_radius_list=args.prop_radius_list,
            pred_bidir_flow=args.pred_bidir_flow,
            pred_bwd_flow=args.pred_bwd_flow,
            num_reg_refine=args.num_reg_refine,
            frame_interval = frame_interval
        )
    flow = optical_flow_normalize(flow)
    return flow


def inference_flow_warpper_v2(flow_model, frames, size, to_uint8 = True):
    if to_uint8:
        frames = (frames * 255)
        
    frames = frames.to("cuda")

    # if len(video.shape) == 4:
    #     video = video.unsqueeze(0)
    image1 = frames[::2]
    image2 = frames[1::2]


    image1s = torch.cat([image1, image2])
    image2s = torch.cat([image2, image1])
    # image1s = frames
    # condition = (torch.arange(len(frames)) % 2 == 1).view(-1, 1, 1, 1).expand_as(frames).to(frames.device)
    # image2s = torch.where(condition, frames.roll(1, 0), frames.roll(-1, 0))

    with torch.autocast(device_type="cuda",dtype=torch.float16):

        flow = inference_flow_v2(
            flow_model,
            image1s,
            image2s,
            padding_factor=args.padding_factor,
            inference_size=size,
            attn_type=args.attn_type,
            attn_splits_list=args.attn_splits_list,
            corr_radius_list=args.corr_radius_list,
            prop_radius_list=args.prop_radius_list,
            pred_bidir_flow=args.pred_bidir_flow,
            pred_bwd_flow=args.pred_bwd_flow,
            num_reg_refine=args.num_reg_refine,
        )
    flow = optical_flow_normalize(flow)
    forward_flow, backward_flow = flow.chunk(2, dim=0)
    return forward_flow, backward_flow


@torch.no_grad()
def inference_flow(model,
                   inference_video=None,
                   padding_factor=8,
                   inference_size=None,
                   attn_type='swin',
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   pred_bidir_flow=False,
                   pred_bwd_flow=False,
                   frame_interval = 1
                   ):
    """ Inference on a directory or a video """
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    fixed_inference_size = inference_size
    transpose_img = False

    filenames = inference_video.to(device) # list of [H, W, 3]

    # filenames = rearrange("B T C H w -> ")
    flow_ls = []

    if not isinstance(frame_interval, list):
        frame_intervals = [frame_interval] * len(filenames)
    else:
        frame_intervals = frame_interval
    frame_intervals = [min(fi, filenames.shape[1] - 1) for fi in frame_intervals]

    # for test_id in range(0, filenames.shape[1] - 1):
    for test_id, frame_interval in zip(range(0, filenames.shape[0]), frame_intervals):



        image1s = filenames[test_id, :-frame_interval]
        image2s = filenames[test_id, frame_interval:]

        image1s = torch.split(image1s, 4)
        image2s = torch.split(image2s, 4)
        pred_flow = []
        for image1, image2 in zip(image1s, image2s):

            # image1 = np.array(image1).astype(np.uint8)
            # image2 = np.array(image2).astype(np.uint8)

            # if len(image1.shape) == 2:  # gray image
            #     image1 = np.tile(image1[..., None], (1, 1, 3))
            #     image2 = np.tile(image2[..., None], (1, 1, 3))
            # else:
            #     image1 = image1[..., :3]
            #     image2 = image2[..., :3]

            # image1 = torch.from_numpy(image1).permute(0, 3, 1, 2).float().to(device)
            # image2 = torch.from_numpy(image2).permute(0, 3, 1, 2).float().to(device)

            # the model is trained with size: width > height
            if image1.size(-2) > image1.size(-1):
                image1 = torch.transpose(image1, -2, -1)
                image2 = torch.transpose(image2, -2, -1)
                transpose_img = True

            nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                            int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

            # resize to nearest size or specified size
            inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]

            # resize before inference
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                    align_corners=True)
                image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                    align_corners=True)

            if pred_bwd_flow:
                image1, image2 = image2, image1

            results_dict = model(image1, image2,
                                attn_type=attn_type,
                                attn_splits_list=attn_splits_list,
                                corr_radius_list=corr_radius_list,
                                prop_radius_list=prop_radius_list,
                                num_reg_refine=num_reg_refine,
                                task='flow',
                                pred_bidir_flow=pred_bidir_flow,
                                )

            flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

            # resize back
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

            if transpose_img:
                flow_pr = torch.transpose(flow_pr, -2, -1)

            pred_flow.append(flow_pr)
        
        flow_pr = torch.cat(pred_flow, dim=0)
        flow_ls += [flow_pr]

    # flow = torch.stack(flow_ls, dim=1)
    if len(set([f.shape[0] for f in flow_ls])) != 1:
        return flow_ls
    else:
        flow = torch.stack(flow_ls, dim=0)
        return flow

@torch.no_grad()
def inference_flow_v2(model,
                   image1s,
                   image2s,
                   padding_factor=8,
                   inference_size=None,
                   attn_type='swin',
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   pred_bidir_flow=False,
                   pred_bwd_flow=False,
                   ):
    """ Inference on a directory or a video """
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    fixed_inference_size = inference_size
    transpose_img = False

    image1s = torch.split(image1s, 4)
    image2s = torch.split(image2s, 4)
    pred_flow = []
    for image1, image2 in zip(image1s, image2s):

        # image1 = np.array(image1).astype(np.uint8)
        # image2 = np.array(image2).astype(np.uint8)

        # if len(image1.shape) == 2:  # gray image
        #     image1 = np.tile(image1[..., None], (1, 1, 3))
        #     image2 = np.tile(image2[..., None], (1, 1, 3))
        # else:
        #     image1 = image1[..., :3]
        #     image2 = image2[..., :3]

        # image1 = torch.from_numpy(image1).permute(0, 3, 1, 2).float().to(device)
        # image2 = torch.from_numpy(image2).permute(0, 3, 1, 2).float().to(device)

        # the model is trained with size: width > height
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)

        if pred_bwd_flow:
            image1, image2 = image2, image1

        results_dict = model(image1, image2,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task='flow',
                            pred_bidir_flow=pred_bidir_flow,
                            )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        pred_flow.append(flow_pr)
    
    flow_pr = torch.cat(pred_flow, dim=0)

    # # flow = torch.stack(flow_ls, dim=1)
    # if len(set([f.shape[0] for f in flow_ls])) != 1:
    #     return flow_ls
    # else:
    #     flow = torch.stack(flow_ls, dim=0)
    #     return flow
    return flow_pr




@torch.no_grad()
def flow_to_image(flow: torch.Tensor, norm_mode: str = "max") -> torch.Tensor:

    """
    Converts a flow to an RGB image.

    Args:
        flow (Tensor): Flow of shape (N, 2, H, W) or (2, H, W) and dtype torch.float.

    Returns:
        img (Tensor): Image Tensor of dtype uint8 where each color corresponds
            to a given flow direction. Shape is (N, 3, H, W) or (3, H, W) depending on the input.
    """

    if flow.dtype != torch.float:
        raise ValueError(f"Flow should be of dtype torch.float, got {flow.dtype}.")

    orig_shape = flow.shape
    if flow.ndim == 3:
        flow = flow[None]  # Add batch dim

    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Input flow should have shape (2, H, W) or (N, 2, H, W), got {orig_shape}.")

    if norm_mode == "max":
        max_norm = torch.sum(flow**2, dim=1).sqrt().max()
        epsilon = torch.finfo((flow).dtype).eps
        normalized_flow = flow / (max_norm + epsilon)
    elif norm_mode == "ratio":
        height, width = flow.shape[-2:]
        flow[:,0] /= height
        flow[:,1] /= width
        normalized_flow = flow
    img = _normalized_flow_to_image(normalized_flow)

    if len(orig_shape) == 3:
        img = img[0]  # Remove batch dim
    return img

@torch.no_grad()
def _normalized_flow_to_image(normalized_flow: torch.Tensor) -> torch.Tensor:

    """
    Converts a batch of normalized flow to an RGB image.

    Args:
        normalized_flow (torch.Tensor): Normalized flow tensor of shape (N, 2, H, W)
    Returns:
       img (Tensor(N, 3, H, W)): Flow visualization image of dtype uint8.
    """

    N, _, H, W = normalized_flow.shape
    device = normalized_flow.device
    flow_image = torch.zeros((N, 3, H, W), dtype=torch.uint8, device=device)
    colorwheel = _make_colorwheel().to(device)  # shape [55x3]
    num_cols = colorwheel.shape[0]
    norm = torch.sum(normalized_flow**2, dim=1).sqrt()
    a = torch.atan2(-normalized_flow[:, 1, :, :], -normalized_flow[:, 0, :, :]) / torch.pi
    fk = (a + 1) / 2 * (num_cols - 1)
    k0 = torch.floor(fk).to(torch.long)
    k1 = k0 + 1
    k1[k1 == num_cols] = 0
    f = fk - k0

    for c in range(colorwheel.shape[1]):
        tmp = colorwheel[:, c]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        col = 1 - norm * (1 - col)
        flow_image[:, c, :, :] = torch.floor(255 * col)
    return flow_image


def _make_colorwheel() -> torch.Tensor:
    """
    Generates a color wheel for optical flow visualization as presented in:
    Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
    URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf.

    Returns:
        colorwheel (Tensor[55, 3]): Colorwheel Tensor.
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


import cv2
def compute_flow(video_tensor, frame_interval = 1):
    videos = video_tensor.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    flow_ls = []
    if not isinstance(frame_interval, list):
        frame_intervals = [frame_interval] * len(video_tensor)
    else:
        frame_intervals = frame_interval
    frame_intervals = [min(fi, video_tensor.shape[1] - 1) for fi in frame_intervals]
    flows = []
    for video, frame_interval in zip(videos, frame_intervals):
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
        flow = torch.stack(flow_ls)
        flows += [flow]
    return flows





def warp_frames(flow, frame, mode="nearest"):

    assert len(flow.shape) == len(frame.shape), "Flow shape length should be the same as frame"
    squeeze_flag = False
    if len(frame.shape) == 3:
        squeeze_flag = True
        frame = frame.unsqueeze(0)
    elif len(frame.shape) != 4:
        assert False, "Frame shape should be 3 or 4"
    if isinstance(frame, torch.Tensor): # B, C, H, W
        warped_frame = warp_frame_tensor(flow, frame, mode = mode)
    elif isinstance(frame, np.ndarray): # B, H, W, C
        frame_tensor = torch.from_numpy(frame).permute(0, 3, 1, 2)
        warped_frame_tensor = warp_frame_tensor(flow, frame_tensor, mode = mode)
        warped_frame = warped_frame_tensor.permute(0, 2, 3, 1).numpy()
    else:
        assert False, "Frame should be torch.Tensor or np.ndarray"
    if squeeze_flag:
        warped_frame = warped_frame.squeeze(0)
    return warped_frame
        
    


def warp_frame(flow, frame, mode="nearest"):
    # np.ndarray. B, H, W, C
    bs, h, w, c = frame.shape
    fl_w, fl_h = flow.shape[1:3]

    # flow in ([-fl_h, fl_h], [-fl_w, fl_w])
    # normalize flow to ([-1, 1], [-1, 1])
    flow = flow / np.array([fl_h, fl_w])

    # resize flow to frame size
    flow = cv2.resize(flow, (w, h))
    # flow to ([-h, h], [-w, w])
    flow = (flow * np.array([h, w])).astype(np.float32)

    # Generate sampling grids
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    flow_grid = torch.stack((grid_x, grid_y), dim=0).float()
    flow_grid = flow_grid[None, ...].expand(bs, -1, -1, -1)
    flow_grid += torch.from_numpy(flow).permute(0, 3, 1, 2)
    flow_grid[:, 0, :, :] = 2 * flow_grid[:, 0, :, :] / (w - 1) - 1
    flow_grid[:, 1, :, :] = 2 * flow_grid[:, 1, :, :] / (h - 1) - 1
    flow_grid = flow_grid.clamp(-1, 1)
    flow_grid = flow_grid.permute(0, 2, 3, 1)

    frame_tensor = torch.from_numpy(frame).float()
    frame_tensor = frame_tensor.permute(0, 3, 1, 2)  # N, C, H, W

    warped_frame = torch.nn.functional.grid_sample(
        frame_tensor, flow_grid, mode=mode, padding_mode="reflection", align_corners=True).permute(0, 2, 3, 1).numpy()

    return warped_frame


def warp_frame_tensor(flow, frame, mode="nearest"):
    bs, c, h, w = frame.shape
    fl_w, fl_h = flow.shape[-2:]
    assert flow.shape[1] == 2
    
    frame = frame.to(flow)
    flow = flow / torch.tensor([fl_h, fl_w]).view(1, -1, 1, 1).to(flow)

    # resize flow
    flow = F.interpolate(flow, size=(h, w))
    flow = flow * torch.tensor([h, w]).view(1, -1, 1, 1).to(flow)

    # Generate sampling grids
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    flow_grid = torch.stack((grid_x, grid_y), dim=0)
    flow_grid = flow_grid[None, ...].expand(bs, -1, -1, -1).to(flow)
    flow_grid += flow
    flow_grid[:, 0, :, :] = 2 * flow_grid[:, 0, :, :] / (w - 1) - 1
    flow_grid[:, 1, :, :] = 2 * flow_grid[:, 1, :, :] / (h - 1) - 1
    flow_grid = flow_grid.permute(0, 2, 3, 1)

    warped_frame = torch.nn.functional.grid_sample(
        frame, flow_grid, mode=mode, padding_mode="reflection", align_corners=True)
    return warped_frame