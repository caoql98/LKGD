import argparse
parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--image_column", type=str, default="image", nargs='+', help="The column of the dataset containing an image."
)
parser.add_argument(
    "--caption_column",
    type=str,
    default="text",
    nargs='+',
    help="The column of the dataset containing a caption or a list of captions.",
)

args = parser.parse_args()
print(args)

print({k:v if not isinstance(v, list) else " ".join(v) for k,v in vars(args).items()})