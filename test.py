# import os
# import sys
# # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# # os.environ["HF_HOME"] = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/.cache/huggingface"
# sys.path.append("/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/ControlNet-v1-1-nightly")
# from annotator.oneformer import OneformerCOCODetector, OneformerADE20kDetector

# from PIL import Image
# import requests
# import torch
# url = (
#     "https://hf-mirror.com/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
# )
# image = Image.open(requests.get(url, stream=True).raw)

# preprocessor = OneformerCOCODetector()

from utils.dataset import PandaN

dataset_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/Panda70M/validation_videos.json"
# config = OmegaConf.load(config_path)
# config = OmegaConf.to_container(config, resolve=True)
# train_dataset, train_dataloader = get_correspondence_loader(config, config["train_file"], True)

dataset = PandaN(dataset_file = dataset_path)