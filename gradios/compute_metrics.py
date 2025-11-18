import sys
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/.cache/huggingface"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image

import json
import numpy as np
from torchvision.transforms import functional as F
import torch
import torchvision.transforms.v2 as transforms
from utils.dataset import process_frames
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import pytorch_lightning as pl
import torch.nn as nn
import clip
from tabulate import tabulate
import argparse
from tqdm import tqdm
from transformers import pipeline

parser = argparse.ArgumentParser(description="Simple example of a training script.")

parser.add_argument(
    "--fake_image_path",
    type=str,
    required=True
)
parser.add_argument(
    "--real_image_path", type=str, default=None
)
parser.add_argument(
    "--dataset_path", type=str, default="/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/PascalVOC/VOC2012/depth_anything_v2_gray/val_dataset.json"
)
parser.add_argument(
    "--resolution", type=int, default=512
)

args = parser.parse_args()
print(args)




real_image_path = args.real_image_path
fake_image_path = args.fake_image_path
dataset_path = args.dataset_path
height, width = args.resolution, args.resolution
real_image_path = os.path.dirname(dataset_path) if real_image_path is None else real_image_path
# height, width = 512, 512
# # dataset_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/open-images/val/hed/val_dataset.json"
# # dataset_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/open-images/Human_body/openpose/val_dataset.json"
# dataset_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/PascalVOC/VOC2012/depth_anything_v2_gray/val_dataset.json"
# # dataset_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/PascalVOC/VOC2012/hed/val_dataset.json"
val_dataset = json.load(open(dataset_path))

# prompt_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/svd-train/annotate/PascalVOC_prompt.json"
# with open(prompt_path, "r") as f:
#     prompt_dict = json.load(f)
# image_paths = [line['original_image'] for line in val_dataset]
prompts = [line['text'] for line in val_dataset]

image_paths = [line['original_image'] for line in val_dataset]
real_images = [Image.open(path).convert("RGB") for path in image_paths]

fake_image_paths = [os.path.join(fake_image_path, line['file_name']) for line in val_dataset]

measure_depth = "depth" in fake_image_path

ToTensor = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

metrics = dict()

def preprocess_image(image):
    image = process_frames([image], height, width,  verbose = False, div = 8, rand_crop=False)[0]
    # return F.center_crop(image, (256, 256))
    return image

processed_real_images = torch.stack([ToTensor(preprocess_image(image)) for image in real_images], dim = 0).to("cuda")
print(processed_real_images.shape)
# torch.Size([10, 3, 256, 256])

fake_images = [Image.open(path).convert("RGB") for path in fake_image_paths]

processed_fake_images_pil = [preprocess_image(image) for image in fake_images]
processed_fake_images = torch.stack([ToTensor(image) for image in processed_fake_images_pil], dim = 0).to("cuda")
print(processed_real_images.shape)

# Compute depth metrics
if measure_depth:
    def get_depth_tensor(pipe, image):
        depth_tensor = pipe(image)["predicted_depth"]
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(1),
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )[0]
        return depth_tensor
    depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    fake_depth_tensors = []
    for fake_image in tqdm(processed_fake_images_pil):
        # Predict pseudo label
        fake_depth_tensor = get_depth_tensor(depth_estimator, fake_image)
        fake_depth_tensors.append(fake_depth_tensor)
    fake_depth_tensors = torch.stack(fake_depth_tensors)
    torch.save(fake_depth_tensors, os.path.join(fake_image_path, "depth_tensor.pt"))
    cmd = f"python gradios/compute_depth_metrics.py --real_image_path {real_image_path} --fake_image_path {fake_image_path}"
    print("Run", cmd)
    os.system(cmd)



# Compute FID

fid = FrechetInceptionDistance(normalize=True).to("cuda")
fid.update(processed_real_images, real=True)
fid.update(processed_fake_images, real=False)

print(f"FID: {float(fid.compute())}")

metrics["FID"] = float(fid.compute())

del fid
torch.cuda.empty_cache()

# Compute CLIP Score

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts, batch_size = 128):
    images_int = (images * 255).to(torch.uint8)
    clip_scores = []
    for st in range(0, len(images_int), batch_size):
        clip_score = clip_score_fn(images_int[st:st+batch_size], prompts[st:st+batch_size]).detach()
        clip_scores.append(clip_score)
    clip_score = torch.tensor(clip_scores).mean()
    return round(float(clip_score), 4)


sd_clip_score = calculate_clip_score(processed_fake_images, prompts)
print(f"CLIP score: {sd_clip_score}")

metrics["CLIP Score"] = sd_clip_score
# Compute CLIP Aesthetic Score

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def load_models():
    model = MLP(768)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    s = torch.load("sac+logos+ava1-l14-linearMSE.pth", map_location=device)

    model.load_state_dict(s)
    model.to(device)
    model.eval()

    model2, preprocess = clip.load("ViT-L/14", device=device)

    model_dict = {}
    model_dict['classifier'] = model
    model_dict['clip_model'] = model2
    model_dict['clip_preprocess'] = preprocess
    model_dict['device'] = device

    return model_dict

def predict(model_dict, images, batch_size = 100):
    image_inputs = [model_dict['clip_preprocess'](image).unsqueeze(0).to(model_dict['device']) for image in images]
    image_inputs = torch.cat(image_inputs)
    with torch.no_grad():
        all_embs = []
        # pdb.set_trace()
        for image_input_batch in image_inputs.split(batch_size):
            image_features = model_dict['clip_model'].encode_image(image_input_batch)
            if model_dict['device'] == 'cuda':
                im_emb_arr = normalized(image_features.detach().cpu().numpy())
                im_emb = torch.from_numpy(im_emb_arr).to(model_dict['device']).type(torch.cuda.FloatTensor)
            else:
                im_emb_arr = normalized(image_features.detach().numpy())
                im_emb = torch.from_numpy(im_emb_arr).to(model_dict['device']).type(torch.FloatTensor)
            all_embs.append(im_emb)
        all_embs = torch.cat(all_embs)
        prediction = model_dict['classifier'](all_embs)
    score = prediction.mean().item()

    score = prediction.mean().item()

    return score

model_dict = load_models()

score = predict(model_dict, fake_images)

print(f"CLIP aesthetic score: {score}")

metrics["CLIP aesthetic score"] = score

eval_text = f"Evaluation metrics:\n\
of predictions: {fake_image_path}\n\
on dataset: {real_image_path}\n"

eval_text += tabulate(
    zip(metrics.keys(), metrics.values())
)

metrics_filename = "eval_metrics.txt"

output_dir = fake_image_path
_save_to = os.path.join(output_dir, metrics_filename)
with open(_save_to, "w+") as f:
    f.write(eval_text)
    print(f"Evaluation metrics saved to {_save_to}")