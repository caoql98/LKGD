from utils.optical_flow import flow_to_image_naive, load_unimatch, inference_flow, optical_flow_expand, optical_flow_normalize, optical_flow_squeeze
from utils.optical_flow import args as flow_args
from utils.dataset import Panda
import time
import torch
import pickle
from diffusers import AutoencoderKLTemporalDecoder
from einops import rearrange
from torchvision.io import write_video
from pipeline.pipeline_stable_video_diffusion_joint_vf import StableVideoDiffusionPipeline
import torchvision.transforms as T
import os 


if __name__ == "__main__":
    dataset = Panda(
        # csv_path="/data/webvid/results_2M_train.csv",
        video_folder="/data/juicefs_sharing_data/72179586/data/panda70m/test_split_webdataset",
        depth_folder="/data/juicefs_sharing_data/72179586/data/panda70m/test_split_webdataset",
        sample_size=(320, 576),
        sample_n_frames=14,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", revision=None, variant="fp16").to("cuda")

    # import pdb
    # pdb.set_trace()
    flow_model = load_unimatch().to(torch.float16)
    checkpoint = torch.load("/data/juicefs_sharing_data/72179586/repos/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth")

    flow_model.load_state_dict(checkpoint['model'], strict=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4)
    optical_flows = []

    # video, _, _ = read_video("/data/juicefs_sharing_data/72179586/repos/VidToMe/data/breakdance.mp4")

    # depth_pixel_values = inference_flow(
    #     flow_model,
    #     inference_video=video.permute(0, 3, 1, 2).unsqueeze(0).to("cuda"),
    #     padding_factor=flow_args.padding_factor,
    #     inference_size=(512, 512),
    #     attn_type=flow_args.attn_type,
    #     attn_splits_list=flow_args.attn_splits_list,
    #     corr_radius_list=flow_args.corr_radius_list,
    #     prop_radius_list=flow_args.prop_radius_list,
    #     pred_bidir_flow=flow_args.pred_bidir_flow,
    #     pred_bwd_flow=flow_args.pred_bwd_flow,
    #     num_reg_refine=flow_args.num_reg_refine,
    # )
    # image = flow_to_image(depth_pixel_values[0].cpu())
    # write_png(image[0], "tmp_image.png")
    # image = [write_video(f"temp_flow_{i}.mp4", flow_to_image(dpv.cpu()).permute(0, 2, 3, 1), fps=8) for i, dpv in enumerate(depth_pixel_values)]
    # exit(0)
    mean_ls = []
    mean_sq_ls = []
    # depth_ls = []

    pipeline = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, local_files_only = True).to("cuda")
    pipeline.enable_model_cpu_offload()


    with torch.autocast(device_type="cuda",dtype=torch.float16):
        for idx, batch in enumerate(dataloader):
            print(batch["pixel_values"].shape)
            stime = time.time()
            depth_pixel_values = inference_flow(
                flow_model,
                inference_video=(batch["pixel_values"] * 255).to("cuda").to(torch.float16),
                padding_factor=flow_args.padding_factor,
                inference_size=(320, 576),
                attn_type=flow_args.attn_type,
                attn_splits_list=flow_args.attn_splits_list,
                corr_radius_list=flow_args.corr_radius_list,
                prop_radius_list=flow_args.prop_radius_list,
                pred_bidir_flow=flow_args.pred_bidir_flow,
                pred_bwd_flow=flow_args.pred_bwd_flow,
                num_reg_refine=flow_args.num_reg_refine,
            )
            # flow = optical_flow_normalize(depth_pixel_values)
            # flow_expand = optical_flow_expand(flow)
            # flow_original = optical_flow_squeeze(flow_expand)
            # assert torch.allclose(flow, flow_original, atol = 1e-5)
            # depth_pixel_values = tensor_to_vae_latent(batch["pixel_values"][:,:14].to(torch.float16).cuda(), vae)

            print("Inferemce Time:", time.time() - stime)
            # print(depth_pixel_values.shape)
            # latents = tensor_to_vae_latent(flow_to_image_naive(depth_pixel_values), vae)
            # print(depth_pixel_values.max(), depth_pixel_values.min())
            image = [write_video(f"temp_flow_{i}.mp4", (flow_to_image_naive(dpv.cpu()).permute(0, 2, 3, 1) * 255).to(torch.uint8), fps=8) for i, dpv in enumerate(depth_pixel_values)]
            [write_video(f"temp_video_{i}.mp4", (video.permute(0, 2, 3, 1) * 255).to(torch.uint8), fps=8) for i, video in enumerate(batch["pixel_values"])]
            exit(0)



            # depth_ls += [depth_pixel_values]
            mean_ls += [latents.mean()]
            mean_sq_ls += [latents.pow(2).mean()]


            if idx % 10 == 0:
                # cur_depth = torch.cat(depth_ls)
                # cur_mean = cur_depth.mean()
                # print(f"{len(cur_depth)} samples mean: {cur_mean}, std: {cur_depth.std()}")
                cur_mean2 = torch.tensor(mean_ls).mean()
                cur_mean_sq = torch.tensor(mean_sq_ls).mean()
                cur_std = (cur_mean_sq - cur_mean2.pow(2)).sqrt()
                print(f"{len(mean_ls)} samples mean: {cur_mean2}, std: {cur_std}")
            
            if idx % 100 == 0:
                with open("flow_latent_statistics.pkl", "wb") as f:
                    pickle.dump({"mean":mean_ls, "mean_sq":mean_sq_ls}, f)


















    # motion_ids = [i for i in range(255, 500, 25)]
    # fps_ids = [i for i in range(5, 30, 5)]

    # motion_id_flow = dict()
    # # motion_id_flow_path = "experiments/motion_cal/svd_motion_statistics.pkl"
    # # if os.path.exists(motion_id_flow_path):
    # #     with open(motion_id_flow_path, "rb") as f:
    # #         motion_id_flow = pickle.load(f)
    # fps_flow = dict()
    # fps_flow_path = "experiments/motion_cal/svd_fps_flow_statistics.pkl"
    # if os.path.exists(fps_flow_path):
    #     with open(fps_flow_path, "rb") as f:
    #         fps_flow = pickle.load(f)
    # cnt = 0
    # cur_id = 0

    # norm_ls = []

    # with torch.autocast(device_type="cuda",dtype=torch.float16):
    #     for idx, batch in enumerate(dataloader):
    #         print(batch["pixel_values"].shape)
    #         stime = time.time()
    #         images = [T.ToPILImage()(video[0]) for video in batch["pixel_values"]]
    #         video = pipeline(images, motion_bucket_id = motion_ids[cur_id],decode_chunk_size=8,num_frames=14,width=576,height=320, num_inference_steps=25, output_type = "np").frames
    #         # video = pipeline(images, fps = fps_ids[cur_id],decode_chunk_size=8,num_frames=14,width=576,height=320, num_inference_steps=25, output_type = "np").frames
    #         video = torch.tensor(video).permute(0, 1, 4, 2, 3) * 255
    #         print(video.shape)
    #         print(video.max(), video.min())
    #         depth_pixel_values = inference_flow(
    #             flow_model,
    #             inference_video=video.to("cuda").to(torch.float16),
    #             padding_factor=flow_args.padding_factor,
    #             inference_size=(320, 576),
    #             attn_type=flow_args.attn_type,
    #             attn_splits_list=flow_args.attn_splits_list,
    #             corr_radius_list=flow_args.corr_radius_list,
    #             prop_radius_list=flow_args.prop_radius_list,
    #             pred_bidir_flow=flow_args.pred_bidir_flow,
    #             pred_bwd_flow=flow_args.pred_bwd_flow,
    #             num_reg_refine=flow_args.num_reg_refine,
    #         )
    #         # flow = optical_flow_normalize(depth_pixel_values)
    #         # flow_expand = optical_flow_expand(flow)
    #         # flow_original = optical_flow_squeeze(flow_expand)
    #         # assert torch.allclose(flow, flow_original, atol = 1e-5)
    #         # depth_pixel_values = tensor_to_vae_latent(batch["pixel_values"][:,:14].to(torch.float16).cuda(), vae)

    #         print("Inferemce Time:", time.time() - stime)
    #         # print(depth_pixel_values.shape)
    #         # latents = tensor_to_vae_latent(flow_to_image_naive(depth_pixel_values), vae)
    #         # print(depth_pixel_values.max(), depth_pixel_values.min())
    #         # image = [write_video(f"temp_flow_{i}.mp4", (flow_to_image_naive(dpv.cpu()).permute(0, 2, 3, 1) * 255).to(torch.uint8), fps=8) for i, dpv in enumerate(depth_pixel_values)]
    #         # [write_video(f"temp_video_{i}.mp4", (video.permute(0, 2, 3, 1)* 255).to(torch.uint8), fps=8) for i, video in enumerate(batch["pixel_values"])]
    #         # exit(0)



    #         # depth_ls += [depth_pixel_values]
    #         norm_ls += [depth_pixel_values.norm(dim=-3).mean()]

    #         # if idx % 10 == 0:
    #         #     # cur_depth = torch.cat(depth_ls)
    #         #     # cur_mean = cur_depth.mean()
    #         #     # print(f"{len(cur_depth)} samples mean: {cur_mean}, std: {cur_depth.std()}")
    #         #     cur_norm = torch.cat(norm_ls)
    #         #     norm_mean = cur_norm.mean()
    #         #     print(f"{len(norm_ls)} samples norm mean: {norm_mean}, std: {cur_norm.std()}")

    #         cnt += 1
    #         if cnt == 10:
    #             cur_norm = torch.tensor(norm_ls)
    #             norm_mean = cur_norm.mean()
    #             motion_id_flow[motion_ids[cur_id]] = (norm_mean, cur_norm.std())
    #             # fps_flow[fps_ids[cur_id]] = (norm_mean, cur_norm.std())
    #             cnt = 0
    #             cur_id += 1
    #             with open("experiments/motion_cal/svd_motion_statistics.pkl", "wb") as f:
    #                 pickle.dump(motion_id_flow, f)
            
    #         if cur_id == len(fps_ids):
    #             break
            

    #     # with open("svd_fps_flow_statistics.pkl", "wb") as f:
    #     #     pickle.dump(fps_flow, f)
