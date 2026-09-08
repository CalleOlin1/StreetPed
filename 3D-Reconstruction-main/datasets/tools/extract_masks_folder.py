"""
@file   extract_masks_folder.py
@brief  Extract car masks for all images in a folder.

This is a small folder-based variant of datasets/tools/extract_masks.py.
It loads every image from the input folder, runs SegFormer segmentation,
extracts the Cityscapes car class, and saves binary masks to a sibling
../masks directory.
"""

import os
from argparse import ArgumentParser
from glob import glob

import imageio
import numpy as np
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor


CAR_CLASS_ID = 13


def build_image_list(image_dir: str):
	image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
	image_paths = []
	for pattern in image_extensions:
		image_paths.extend(glob(os.path.join(image_dir, pattern)))
	return sorted(image_paths)


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("image_dir", type=str, help="Folder containing input images")
	parser.add_argument(
		"--segformer_path",
		type=str,
		default="/home/guojianfei/ai_ws/SegFormer",
		help="Path to the SegFormer repository",
	)
	parser.add_argument("--config", type=str, default=None, help="SegFormer config file")
	parser.add_argument("--checkpoint", type=str, default=None, help="SegFormer checkpoint file")
	parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="Output directory for masks. Defaults to a sibling ../masks folder.",
	)
	args = parser.parse_args()

	if args.config is None:
		args.config = os.path.join(
			args.segformer_path,
			"local_configs",
			"segformer",
			"B5",
			"segformer.b5.1024x1024.city.160k.py",
		)
	if args.checkpoint is None:
		args.checkpoint = os.path.join(
			args.segformer_path,
			"pretrained",
			"segformer.b5.1024x1024.city.160k.pth",
		)

	image_dir = os.path.abspath(args.image_dir)
	if args.output_dir is None:
		output_dir = os.path.abspath(os.path.join(image_dir, os.pardir, "masks"))
	else:
		output_dir = os.path.abspath(args.output_dir)
	os.makedirs(output_dir, exist_ok=True)

	image_paths = build_image_list(image_dir)
	if len(image_paths) == 0:
		raise FileNotFoundError(f"No images found in {image_dir}")

	model = init_segmentor(args.config, args.checkpoint, device=args.device)

	for image_path in tqdm(image_paths, desc="Extracting car masks"):
		image_name = os.path.splitext(os.path.basename(image_path))[0]
		result = inference_segmentor(model, image_path)
		mask = result[0].astype(np.uint8)
		car_mask = np.isin(mask, [CAR_CLASS_ID]).astype(np.uint8) * 255
		imageio.imwrite(os.path.join(output_dir, f"{image_name}.png"), car_mask)
