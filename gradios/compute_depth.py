import argparse
from transformers import pipeline
import matplotlib
import torchvision.transforms as T
import torch
import pdb
import sys
# sys.path.append("data/deps/ControlNet")
sys.path.append("/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/Marigold")
from tabulate import tabulate
from src.util import metric
from src.util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)
from src.util.metric import MetricTracker
import torchvision.transforms.v2 as transforms
from utils.dataset import process_frames


def filter_paths(paths, filter_file=None):
    if filter_file is not None:
        filter_names = set(json.load(open(filter_file)))
        paths = [image for image in paths if os.path.basename(image) in filter_names]
    return paths

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

def get_depth_tensor(pipe, image):
    depth_tensor = pipe(image)["predicted_depth"]
    depth_tensor = torch.nn.functional.interpolate(
        depth_tensor.unsqueeze(1),
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )[0]
    return depth_tensor

def main():

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
    parser.add_argument(
        "--compute_depth", action="store_true"
    )

    args = parser.parse_args()
    print(args)

    real_image_path = args.real_image_path
    fake_image_path = args.fake_image_path
    dataset_path = args.dataset_path
    height, width = args.resolution, args.resolution
    real_image_path = os.path.dirname(dataset_path) if real_image_path is None else real_image_path

    if args.compute_depth:

        # dataset_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/open-images/val/hed/val_dataset.json"
        # dataset_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/open-images/Human_body/openpose/val_dataset.json"

        # dataset_path = "/home/bml/storage/mnt/v-95c5b44cfcff4e6c/org/data_lxr/data/PascalVOC/VOC2012/hed/val_dataset.json"
        val_dataset = json.load(open(dataset_path))
        image_paths = [line['original_image'] for line in val_dataset]
        real_images = [Image.open(path).convert("RGB") for path in image_paths]
        generate_image_paths = [os.path.join(generate_image_path, line['file_name']) for line in val_dataset]
        def preprocess_image(image):
            image = process_frames([image], height, width,  verbose = False, div = 8, rand_crop=False)[0]
            # image = ToTensor(image)
            # return F.center_crop(image, (256, 256))
            return image
        processed_real_images = [preprocess_image(image) for image in real_images]
        fake_images = [Image.open(path).convert("RGB") for path in generate_image_paths]
        processed_fake_images = [preprocess_image(image) for image in fake_images]
        
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

        real_depth_tensors = []
        fake_depth_tensors = []

        for real_image, fake_image in zip(tqdm(processed_real_images), processed_fake_images):
            # Predict pseudo label
            real_depth_tensor = get_depth_tensor(pipe, real_image)
            fake_depth_tensor = get_depth_tensor(pipe, fake_image)
            real_depth_tensors.append(real_depth_tensor)
            fake_depth_tensors.append(fake_depth_tensor)

        real_depth_tensors = torch.stack(real_depth_tensors)
        fake_depth_tensors = torch.stack(fake_depth_tensors)

        torch.save(real_depth_tensors, os.path.join(real_image_path, "depth_tensor.pt"))
        torch.save(fake_depth_tensors, os.path.join(generate_image_path, "depth_tensor.pt"))

    real_depth_tensors = torch.load(os.path.join(real_image_path, "depth_tensor.pt"))
    fake_depth_tensors = torch.load(os.path.join(generate_image_path, "depth_tensor.pt"))

    eval_metrics = [
        "abs_relative_difference",
        "squared_relative_difference",
        "rmse_linear",
        "rmse_log",
        "log10",
        "delta1_acc",
        "delta2_acc",
        "delta3_acc",
        "i_rmse",
        "silog_rmse",
    ]

    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    output_dir = generate_image_path

    # write title
    file_names = [item["file_name"].split(".")[0] for item in val_dataset]

    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")

    for real_depth_tensor, fake_depth_tensor, pred_name in zip(tqdm(real_depth_tensors), fake_depth_tensors, file_names):
        depth_raw = real_depth_tensor.clone().numpy()
        depth_pred = fake_depth_tensor.clone().numpy()
        depth_raw += 1
        valid_mask = np.ones_like(depth_raw).astype(bool)
        depth_pred, scale, shift = align_depth_least_square(
            gt_arr=depth_raw,
            pred_arr=depth_pred,
            valid_mask_arr=valid_mask,
            return_scale_shift=True,
            # max_resolution=alignment_max_res,
        )

    # clip to d > 0 for evaluation
    depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

    # Evaluate (using CUDA if available)
    sample_metric = []
    depth_pred_ts = torch.from_numpy(depth_pred).to(device)
    depth_raw_ts = torch.from_numpy(depth_raw).to(device)
    valid_mask_ts = torch.from_numpy(valid_mask).to(device)
# -------------------- Save metrics to file --------------------
    eval_text = f"Evaluation metrics:\n\
    of predictions: {generate_image_path}\n\
    on dataset: {real_image_path}\n"

    eval_text += tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )

    metrics_filename = "eval_metrics"
    metrics_filename += ".txt"
    
    output_dir = generate_image_path
    _save_to = os.path.join(output_dir, metrics_filename)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
        print(f"Evaluation metrics saved to {_save_to}")
    

