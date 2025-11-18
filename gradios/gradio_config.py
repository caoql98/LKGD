import os
from collections import defaultdict

# if gr.NO_RELOAD:
checkpoint_dict = {
    "mask_depth_joint": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint",
    "mask_depth_joint_freezey": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_freezey",
    "mask_depth_joint_depth_anything_v2_color_bugfix2": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_bugfix2/checkpoint-3000",
    "mask_depth_joint_depth_anything_v2_color_bugfix2_no_noise_offset": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_bugfix2_no_noise_offset",
    "mask_normal_lora_joint_rank64" : "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64_freezey": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_rank64_freezey",
    "mask_sr_lora_joint": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_sr_lora_joint",
    "mask_normal_lora_joint_rank64_skipencoder": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64_skipencoder",
    "mask_seg_lora_joint_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_seg_lora_joint_rank64",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64_post_scale" : "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_rank64_post_scale/checkpoint-11000",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_rank64",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse",
"mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse_skip_encoder": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse_skip_encoder",
"mask_depth_controlnet_lora": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_controlnet_lora/checkpoint-21000",
"mask_depth_controlnet" : "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_controlnet/checkpoint-21000",
"mask_hed_lora_joint_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_hed_lora_joint_rank64/",
"mask_hed_lora_joint_rank64_conv_fuse":"/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_hed_lora_joint_rank64_conv_fuse",
"mask_hed_lora_joint_rank64_no_timestep_align":"/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_hed_lora_joint_rank64_no_timestep_align",
"mask_hed_lora_joint_rank64_no_timestep_align_add_norm": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_hed_lora_joint_rank64_no_timestep_align_add_norm",
"depth_lora_joint_depth_gray_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_depth_lora_joint_depth_gray_rank64",
"depth_lora_joint_depth_gray_rank64_no_timestep_align" : "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_depth_lora_joint_depth_gray_rank64_no_timestep_align",
# "hed_lora_joint_rank64_nta": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_hed_lora_joint_rank64_nta",
"mask_correspondence_lora_joint_rank16_skipe": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_correspondence_lora_joint_rank16_skipe/checkpoint-19000",
"correspondence_lora_joint_rank16_skipe_nta": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_correspondence_lora_joint_rank16_skipe_nta/checkpoint-15000",
"depth_gray_controlnet": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_gray_controlnet",
"depth_controlnet_full": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_depth_controlnet",
"depth_controlnet_lora_full": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_depth_controlnet_lora",
"depth_lora_joint_gray_rank64_nta": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_lora_joint_gray_rank64_nta",
"hed_controlnet": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_hed_controlnet",
"hed_lora_joint_rank16_nta": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_hed_lora_joint_rank16_nta",
"depth_gray_controlnet_lora": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_gray_controlnet_lora",
"mask_correspondence_lora_joint_rank64":"/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_correspondence_lora_joint_rank64/checkpoint-11000",
"correspondence_lora_joint_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_correspondence_lora_joint_rank64/checkpoint-17000",
"depth_lora_joint_gray_rank64_nta_convfuse": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_lora_joint_gray_rank64_nta_convfuse",
"id_lora_joint_celebawild_r64_nta": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_id_lora_joint_celebawild_r64_nta/checkpoint-36000",
"id_lora_joint_celebahq_r64_nta": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_id_lora_joint_celebahq_r64_nta/checkpoint-20000",
"id_lora_joint_celebawild_r64_nta_fix": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_id_lora_joint_celebawild_r64_nta_fix",
"edit_lora_joint_r64_nta": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_edit_lora_joint_r64_nta",
"pose_controlnet": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_pose_controlnet",
"pose_lora_joint_rank64_nta_convfuse": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_pose_lora_joint_rank64_nta_convfuse",
"id_lora_joint_celebawild_r64_nta_fix_sym": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_id_lora_joint_celebawild_r64_nta_fix_sym/checkpoint-45000",
"edit_lora_joint_r64_nta_sym": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_edit_lora_joint_r64_nta_sym",
"hed_lora_joint_rank64_nta_convfuse": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_hed_lora_joint_rank64_nta_convfuse",
"hed_safe_lora_joint_rank64_nta_convfuse": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_hed_safe_lora_joint_rank64_nta_convfuse",
"hed_safe_controlnet": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_hed_safe_controlnet",
"depth_lora_joint_gray_rank64_nta_cf_only_image_loss":"/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_lora_joint_gray_rank64_nta_cf_only_image_loss",
"depth_lora_joint_gray_rank64_nta_cf_ds01": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_lora_joint_gray_rank64_nta_cf_ds01",
"depth_lora_joint_gray_rank64_nta_cf_ds001": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_lora_joint_gray_rank64_nta_cf_ds001",
"depth_lora_joint_gray_rank64_nta_cf_only_image_loss_ds001": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_lora_joint_gray_rank64_nta_cf_only_image_loss_ds001",
"depth_lora_joint_gray_rank64_nta_cf_only_image_loss_ds01": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_lora_joint_gray_rank64_nta_cf_only_image_loss_ds01",
"depth_gray_controlnet_ds001":
"/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_gray_controlnet_ds001",
"depth_gray_controlnet_ds01":
"/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_depth_gray_controlnet_ds01",
"corres_lora_joint_rank64_nta_sym": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_corres_lora_joint_rank64_nta_sym/checkpoint-15000",
"corres_lora_joint_rank16_nta_sym": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_corres_lora_joint_rank16_nta_sym/checkpoint-15000",
# "mask_hed_lora_joint_rank64_no_timestep_align_add_norm"
"corres_lora_joint_rank64_nta_sym_pandaval":"/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_final/output_corres_lora_joint_rank64_nta_sym_pandaval/checkpoint-10000",
"id_lora_joint_celeba_r64_yr16_nta_sym":"/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_id_lora_joint_celeba_r64_yr16_nta_sym/checkpoint-15000"
}



additional_kwargs_dict = defaultdict(lambda: dict())
additional_kwargs_dict.update(
    {
    "mask_normal_lora_joint_rank64_skipencoder": {"skip_encoder": True},
    "mask_depth_lora_joint_depth_anything_v2_color_rank64_post_scale": {"post_joint": "scale"},
    "mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse": {"post_joint": "conv_fuse"},
"mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse_skip_encoder": {"post_joint": "conv_fuse", "skip_encoder": True},
    "mask_hed_lora_joint_rank64_conv_fuse": {"post_joint": "conv_fuse"},
    "mask_hed_lora_joint_rank64_no_timestep_align_add_norm": {"add_norm" : "True"},
    "mask_correspondence_lora_joint_rank16_skipe": {"skip_encoder": True},
    "depth_lora_joint_gray_rank64_nta_convfuse": {"post_joint": "conv_fuse"},
    "pose_lora_joint_rank64_nta_convfuse": {"post_joint": "conv_fuse"},
    "hed_lora_joint_rank64_nta_convfuse": {"post_joint": "conv_fuse"},
    "hed_safe_lora_joint_rank64_nta_convfuse": {"post_joint": "conv_fuse"},
    "depth_lora_joint_gray_rank64_nta_cf_only_image_loss": {"post_joint": "conv_fuse"},
    "depth_lora_joint_gray_rank64_nta_cf_ds01": {"post_joint": "conv_fuse"},
    "depth_lora_joint_gray_rank64_nta_cf_ds001": {"post_joint": "conv_fuse"},
    "depth_lora_joint_gray_rank64_nta_cf_only_image_loss_ds001": {"post_joint": "conv_fuse"},
    "depth_lora_joint_gray_rank64_nta_cf_only_image_loss_ds01": {"post_joint": "conv_fuse"},
    },


)

for key in checkpoint_dict.keys():
    if "controlnet" in key and "mask" in key:
        additional_kwargs_dict[key] = {"conditioning_channels" : 4}

# skip_encoder = {"mask_normal_lora_joint_rank64_skipencoder"}
# post_scale = {"mask_depth_lora_joint_depth_anything_v2_color_rank64_post_scale"}
# trigger_word_map = {
#         "mask_depth_joint": "",
#         "mask_depth_joint_freezey": "",
#         "mask_depth_joint_depth_anything_v2_color_bugfix2": "",
#         "mask_depth_joint_depth_anything_v2_color_bugfix2_no_noise_offset": "",
#         "mask_normal_lora_joint_rank64": "normal_map, ",
#         "mask_depth_lora_joint_depth_anything_v2_color_rank64_freezey": "depth_map, ",
#         "mask_sr_lora_joint": "",
#         "mask_normal_lora_joint_rank64_skipencoder": "normal_map, ", 
#         "mask_seg_lora_joint_rank64": "seg_map, "
# }

trigger_word_map = {
    "mask_depth_joint": "",
    "mask_depth": "",
    "mask_depth_noise_offset": "",
    "mask_depth_noise_offset_mask_fix": "",
    "mask_depth_noise_offset_mask_fix_with_trigger": "depth_map, ",
    "mask_depth_noise_offset_mask_fix_with_trigger_depth_anythingv2_gray_fixbug2": "depth_map, ",
    "mask_depth_noise_offset_mask_fix_with_trigger_depth_anythingv2_color_fixbug2": "depth_map, ",
    "mask_depth_joint_depth_anything_v2_color_rank32": "depth_map, ",
    "mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger": "depth_map, ",
    "mask_normal_lora_joint_rank64": "normal_map, ",
    "mask_sr_lora_joint": "",
    "mask_normal_lora_joint_rank64_skipencoder": "normal_map, ",
    "mask_seg_lora_joint_rank64": "seg_map, ",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64_post_scale": "depth_map, ",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64": "depth_map, ",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse": "depth_map, ",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse_skip_encoder": "depth_map, ",
    "mask_depth_controlnet": "",
    "mask_depth_controlnet_lora": "",
    "mask_hed_lora_joint_rank64": "hed_map, ",
    "mask_hed_lora_joint_rank64_conv_fuse": "hed_map, ",
    "mask_hed_lora_joint_rank64_no_timestep_align": "hed_map, ",
    "mask_hed_lora_joint_rank64_no_timestep_align_add_norm": "hed_map, ",
    "depth_lora_joint_depth_gray_rank64": "depth_map, ",
    "depth_lora_joint_depth_gray_rank64_no_timestep_align" : "depth_map, ",
    "hed_lora_joint_rank64_nta": "hed_map, ",
    "mask_correspondence_lora_joint_rank16_skipe": "corres_map, ",
    "correspondence_lora_joint_rank16_skipe_nta": "corres_map, ",
    "depth_controlnet_gray": "",
    "depth_lora_joint_gray_rank64_nta": "depth_map, ",
    "hed_controlnet": "",
    "hed_lora_joint_rank16_nta": "hed_map, ",
    "depth_gray_controlnet": "",
    "depth_gray_controlnet_lora": "",
    "depth_controlnet_full": "",
    "depth_controlnet_lora_full": "",
    "mask_correspondence_lora_joint_rank64":"corres_map, ",
    "correspondence_lora_joint_rank64": "corres_map, ",
    "depth_lora_joint_gray_rank64_nta_convfuse": "depth_map, ",
    "id_lora_joint_celebawild_r64_nta": "",
    "id_lora_joint_celebahq_r64_nta": "",
    "id_lora_joint_celebawild_r64_nta_fix": "",
    "edit_lora_joint_r64_nta": "",
    "pose_controlnet": "",
    "pose_lora_joint_rank64_nta_convfuse": "keypoint_map, ",
    "id_lora_joint_celebawild_r64_nta_fix_sym": "",
    "edit_lora_joint_r64_nta_sym": "",
    "hed_lora_joint_rank64_nta_convfuse": "hed_map, ",
    "hed_safe_lora_joint_rank64_nta_convfuse": "hed_map, ",
    "hed_safe_controlnet": "",
    "depth_lora_joint_gray_rank64_nta_cf_only_image_loss": "depth_map, ",
    "depth_lora_joint_gray_rank64_nta_cf_ds01": "depth_map, ",
    "depth_lora_joint_gray_rank64_nta_cf_ds001": "depth_map, ",
    "depth_lora_joint_gray_rank64_nta_cf_only_image_loss_ds001": "depth_map, ",
    "depth_lora_joint_gray_rank64_nta_cf_only_image_loss_ds01": "depth_map, ",
    "depth_gray_controlnet_ds001":"",
    "depth_gray_controlnet_ds01":"",
    "corres_lora_joint_rank64_nta_sym": "",
    "corres_lora_joint_rank16_nta_sym": "",
    "corres_lora_joint_rank64_nta_sym_pandaval":"",
    "id_lora_joint_celeba_r64_yr16_nta_sym":"person_pair"
}

checkpoint_default_ylora_map = {
    # "mask_depth_joint": "mask_depth_joint",
    "mask_depth_joint_freezey": "mask_depth",
    "mask_depth_joint_depth_anything_v2_color_bugfix2": "mask_depth_noise_offset_mask_fix_with_trigger_depth_anythingv2_color_fixbug2",
    "mask_depth_joint_depth_anything_v2_color_bugfix2_no_noise_offset": "mask_depth_noise_offset_mask_fix_with_trigger_depth_anythingv2_color_fixbug2",
    # "mask_normal_lora_joint_rank64": "mask_normal_lora_joint_rank64",
    "mask_depth_lora_joint_depth_anything_v2_color_rank64_freezey": "mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger",
    # "mask_sr_lora_joint": "mask_sr_lora_joint",
    # "mask_normal_lora_joint_rank64_skipencoder": "mask_normal_lora_joint_rank64_skipencoder",
    # "mask_seg_lora_joint_rank64": "mask_seg_lora_joint_rank64",
    # "mask_depth_lora_joint_depth_anything_v2_color_rank64_post_scale": "mask_depth_lora_joint_depth_anything_v2_color_rank64_post_scale",
    # "mask_depth_lora_joint_depth_anything_v2_color_rank64": "mask_depth_lora_joint_depth_anything_v2_color_rank64",
    # "mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse": "mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse",
    # "mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse_skip_encoder": "mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse_skip_encoder",
    # "mask_depth_controlnet": "mask_depth_controlnet",
    # "mask_depth_controlnet_lora": "mask_depth_controlnet_lora",
    # "mask_hed_lora_joint_rank64":"mask_hed_lora_joint_rank64"
}

for key in checkpoint_dict.keys():
    if key not in checkpoint_default_ylora_map:
        checkpoint_default_ylora_map[key] = key


y_lora_dict = {
    # "mask_depth_joint": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint/y_lora",
    "mask_depth": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora/y_lora",
    # "mask_depth_noise_offset": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_noise_offset/y_lora",
    # "mask_depth_noise_offset_mask_fix": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_noise_offset_mask_fix/y_lora",
    # "mask_depth_noise_offset_mask_fix_with_trigger": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_noise_offset_mask_fix_with_trigger/y_lora",
    "mask_depth_noise_offset_mask_fix_with_trigger_depth_anythingv2_gray_fixbug2": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_noise_offset_mask_fix_with_trigger_depthanythingv2_gray_fixbug2/y_lora",
    "mask_depth_noise_offset_mask_fix_with_trigger_depth_anythingv2_color_fixbug2": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_noise_offset_mask_fix_with_trigger_depthanythingv2_color_fixbug2/y_lora",
    # "mask_depth_joint_depth_anything_v2_color_rank32": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank32/y_lora",
    "mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_depthanythingv2_color_rank64_randtrans_trigger/y_lora",
    # "mask_normal_lora_joint_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64/y_lora",
    "mask_sr_lora_joint": None,
    # "mask_normal_lora_joint_rank64_skipencoder": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_normal_lora_joint_rank64_skipencoder/y_lora",
    # "mask_seg_lora_joint_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_seg_lora_joint_rank64/y_lora",
    # "mask_depth_lora_joint_depth_anything_v2_color_rank64_post_scale": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_rank64_post_scale/checkpoint-11000/y_lora",
    # "mask_depth_lora_joint_depth_anything_v2_color_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_rank64/y_lora",
    # "mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse/y_lora",
    # "mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse_skip_encoder": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_lora_joint_depth_anything_v2_color_rank64_conv_fuse_skip_encoder/y_lora",
    "mask_depth_controlnet": None,
    # "mask_depth_controlnet_lora": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_depth_controlnet_lora/checkpoint-21000/y_lora",
    # "mask_hed_lora_joint_rank64": "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/output_dir/output_mask_hed_lora_joint_rank64/y_lora",
    # "mask_hed_lora_joint_rank64_conv_fuse": "hed_map, ",
    # "mask_hed_lora_joint_rank64_no_timestep_align": "hed_map, ",
}

for key in checkpoint_dict.keys():
    if key not in y_lora_dict:
        y_lora_dict[key] = os.path.join(checkpoint_dict[key], "y_lora")

# base_model_id_dict = {
#     "sd1.5": "runwayml/stable-diffusion-inpainting",
#     "sd-inpaint":"runwayml/stable-diffusion-v1-5"
# }

base_model_dict = {
    "sd1.5": "runwayml/stable-diffusion-v1-5",
    "sd-inpaint":"runwayml/stable-diffusion-inpainting",
    "realistic-v4": "digiplay/majicMIX_realistic_v4",
    "anything-v5": "stablediffusionapi/anything-v5"
}

base_model_id_map = dict()
for key in checkpoint_dict.keys():
    if "mask" in key:
        base_model_id_map[key] = "sd-inpaint"
    else:
        base_model_id_map[key] = "sd1.5"
