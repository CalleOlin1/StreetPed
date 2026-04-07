"""Pipe-based sky-mask extraction using SegFormer.

This module provides two pieces:

1. A subprocess worker that reads a list of images from stdin and writes a
   list of binary sky masks to stdout.
2. A convenience client that launches the worker and returns the masks.

The protocol is intentionally simple and mirrors the image-list pipe pattern
used elsewhere in the repository:

- write a 4-byte big-endian image count
- for each image, write a 4-byte big-endian payload length followed by PNG bytes

The worker responds with the same structure for the masks.
"""

from __future__ import annotations

import argparse
import io
import os
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image


ROAD_CLASS_ID = 0
SKY_CLASS_ID = 10
DEFAULT_SEGFORMER_PATH = os.path.join(os.path.dirname(__file__), "SegFormer")
DEFAULT_CONFIG_REL = os.path.join(
	"local_configs",
	"segformer",
	"B5",
	"segformer.b5.1024x1024.city.160k.py",
)
DEFAULT_CHECKPOINT_REL = os.path.join(
	"pretrained",
	"segformer.b5.1024x1024.city.160k.pth",
)

ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]


def _ensure_segformer_on_path(segformer_path: str | None) -> str:
	segformer_root = segformer_path or DEFAULT_SEGFORMER_PATH
	segformer_root = os.path.abspath(segformer_root)
	if segformer_root not in sys.path:
		sys.path.insert(0, segformer_root)
	return segformer_root


def _import_mmseg_apis(segformer_path: str | None = None):
	_ensure_segformer_on_path(segformer_path)
	try:
		mmseg_apis = importlib.import_module("mmseg.apis")
		inference_segmentor = getattr(mmseg_apis, "inference_segmentor")
		init_segmentor = getattr(mmseg_apis, "init_segmentor")
	except ImportError as error:
		raise ImportError(
			"Could not import mmseg.apis. Install SegFormer/mmseg or pass --segformer_path."
		) from error
	return inference_segmentor, init_segmentor


def _read_exact(pipe, n: int) -> bytes:
	buf = b""
	while len(buf) < n:
		chunk = pipe.read(n - len(buf))
		if not chunk:
			raise EOFError("Unexpected EOF while reading from pipe")
		buf += chunk
	return buf


def _serialize_image(img: Image.Image) -> bytes:
	buf = io.BytesIO()
	img.save(buf, format="PNG")
	return buf.getvalue()


def _deserialize_image(img_bytes: bytes) -> Image.Image:
	img = Image.open(io.BytesIO(img_bytes))
	img.load()
	return img


def _image_like_to_rgb_pil(image: ImageLike) -> Image.Image:
	if isinstance(image, Image.Image):
		return image.convert("RGB")
	if isinstance(image, (str, os.PathLike)):
		return Image.open(image).convert("RGB")
	array = np.asarray(image)
	if array.ndim == 2:
		array = np.stack([array, array, array], axis=-1)
	elif array.ndim == 3 and array.shape[2] == 4:
		array = array[:, :, :3]
	if array.dtype != np.uint8:
		array = np.clip(array, 0, 255).astype(np.uint8)
	return Image.fromarray(array).convert("RGB")


def _image_like_to_bgr_ndarray(image: ImageLike) -> np.ndarray:
	rgb = np.asarray(_image_like_to_rgb_pil(image), dtype=np.uint8)
	bgr = rgb[:, :, ::-1]
	return np.ascontiguousarray(bgr)


def _read_image_list(pipe) -> List[Image.Image]:
	count = int.from_bytes(_read_exact(pipe, 4), "big")
	images = []
	for _ in range(count):
		payload_len = int.from_bytes(_read_exact(pipe, 4), "big")
		img_bytes = _read_exact(pipe, payload_len)
		images.append(_deserialize_image(img_bytes))
	return images


def _write_image_list(pipe, images: Sequence[Image.Image]) -> None:
	pipe.write(len(images).to_bytes(4, "big"))
	for image in images:
		img_bytes = _serialize_image(image)
		pipe.write(len(img_bytes).to_bytes(4, "big"))
		pipe.write(img_bytes)
	pipe.flush()


def _binary_mask_from_class_ids(mask: np.ndarray, class_ids: Sequence[int]) -> np.ndarray:
	"""Convert a semantic mask to a binary uint8 mask for selected class ids."""
	return np.isin(mask, class_ids).astype(np.uint8) * 255


def _sky_mask_from_segmentation(mask: np.ndarray) -> np.ndarray:
	return _binary_mask_from_class_ids(mask, [SKY_CLASS_ID])


def _road_mask_from_segmentation(mask: np.ndarray) -> np.ndarray:
	return _binary_mask_from_class_ids(mask, [ROAD_CLASS_ID])


def _load_segmentation_model(
	segformer_path: str | None = None,
	config: str | None = None,
	checkpoint: str | None = None,
	device: str = "cuda:0",
):
	inference_segmentor, init_segmentor = _import_mmseg_apis(segformer_path)
	segformer_root = os.path.abspath(segformer_path or DEFAULT_SEGFORMER_PATH)
	config = config or os.path.join(segformer_root, DEFAULT_CONFIG_REL)
	checkpoint = checkpoint or os.path.join(segformer_root, DEFAULT_CHECKPOINT_REL)
	model = init_segmentor(config, checkpoint, device=device)
	return model, inference_segmentor


def extract_sky_masks(
	images: Sequence[ImageLike],
	segformer_path: str | None = None,
	config: str | None = None,
	checkpoint: str | None = None,
	device: str = "cuda:0",
) -> List[np.ndarray]:
	"""Extract binary sky masks for a list of images in-process."""
	model, inference_segmentor = _load_segmentation_model(
		segformer_path=segformer_path,
		config=config,
		checkpoint=checkpoint,
		device=device,
	)

	masks: List[np.ndarray] = []
	for image in images:
		bgr_image = _image_like_to_bgr_ndarray(image)
		result = inference_segmentor(model, bgr_image)
		mask = result[0].astype(np.uint8)
		masks.append(_sky_mask_from_segmentation(mask))
	return masks


def extract_road_masks(
	images: Sequence[ImageLike],
	segformer_path: str | None = None,
	config: str | None = None,
	checkpoint: str | None = None,
	device: str = "cuda:0",
) -> List[np.ndarray]:
	"""Extract binary road masks for a list of images in-process."""
	model, inference_segmentor = _load_segmentation_model(
		segformer_path=segformer_path,
		config=config,
		checkpoint=checkpoint,
		device=device,
	)

	masks: List[np.ndarray] = []
	for image in images:
		bgr_image = _image_like_to_bgr_ndarray(image)
		result = inference_segmentor(model, bgr_image)
		mask = result[0].astype(np.uint8)
		masks.append(_road_mask_from_segmentation(mask))
	return masks


def extract_sky_masks_via_pipe(
	images: Sequence[ImageLike],
	segformer_path: str | None = None,
	config: str | None = None,
	checkpoint: str | None = None,
	device: str = "cuda:0",
	python_executable: str | None = None,
) -> List[np.ndarray]:
	"""Launch the pipe worker as a subprocess and return sky masks.

	The images are serialized as PNG and transferred over stdin/stdout.
	"""
	script_path = os.path.abspath(__file__)
	python_executable = python_executable or sys.executable

	cmd = [
		python_executable,
		"-u",
		script_path,
		"--worker",
		f"--device={device}",
	]
	if segformer_path is not None:
		cmd.append(f"--segformer_path={segformer_path}")
	if config is not None:
		cmd.append(f"--config={config}")
	if checkpoint is not None:
		cmd.append(f"--checkpoint={checkpoint}")

	proc = subprocess.Popen(
		cmd,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
	)
	assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None

	try:
		ready_line = proc.stdout.readline().decode().strip()
		if ready_line != "READY":
			stderr = proc.stderr.read().decode(errors="replace")
			raise RuntimeError(
				f"Sky-mask worker failed to start. Received: {ready_line!r}\n{stderr}"
			)

		# Forward the input list.
		serialized_images = [_serialize_image(_image_like_to_rgb_pil(image)) for image in images]
		proc.stdin.write(len(serialized_images).to_bytes(4, "big"))
		for img_bytes in serialized_images:
			proc.stdin.write(len(img_bytes).to_bytes(4, "big"))
			proc.stdin.write(img_bytes)
		proc.stdin.flush()
		proc.stdin.close()

		# Read the mask list from stdout.
		mask_count = int.from_bytes(_read_exact(proc.stdout, 4), "big")
		masks: List[np.ndarray] = []
		for _ in range(mask_count):
			payload_len = int.from_bytes(_read_exact(proc.stdout, 4), "big")
			mask_image = _deserialize_image(_read_exact(proc.stdout, payload_len))
			masks.append(np.asarray(mask_image.convert("L"), dtype=np.uint8))

		stderr_output = proc.stderr.read().decode(errors="replace")
		if stderr_output.strip():
			print(stderr_output, file=sys.stderr, end="")
		proc.wait()
		return masks
	finally:
		if proc.poll() is None:
			proc.kill()


def _save_masks(masks: Sequence[np.ndarray], save_dir: str, names: Sequence[str] | None = None) -> None:
	os.makedirs(save_dir, exist_ok=True)
	for index, mask in enumerate(masks):
		if names is None:
			name = f"{index:06d}.png"
		else:
			name = names[index]
			if not name.lower().endswith(".png"):
				name = f"{name}.png"
		out_path = os.path.join(save_dir, name)
		Image.fromarray(mask).save(out_path)


def _worker_main() -> None:
	parser = argparse.ArgumentParser(description="Pipe worker for sky mask extraction")
	parser.add_argument("--worker", action="store_true", help="Run as stdin/stdout worker")
	parser.add_argument("--segformer_path", type=str, default=None)
	parser.add_argument("--config", type=str, default=None)
	parser.add_argument("--checkpoint", type=str, default=None)
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument("--save_dir", type=str, default=None, help="Optional directory to save masks")
	parser.add_argument(
		"--image_paths",
		nargs="*",
		default=None,
		help="Optional direct mode: process image paths without using a pipe",
	)
	args = parser.parse_args()

	if args.worker:
		print("Sky-mask worker: loading model...", file=sys.stderr, flush=True)
		model, inference_segmentor = _load_segmentation_model(
			segformer_path=args.segformer_path,
			config=args.config,
			checkpoint=args.checkpoint,
			device=args.device,
		)
		print("Sky-mask worker: model loaded", file=sys.stderr, flush=True)

		# Handshake before any binary output.
		sys.stdout.write("READY\n")
		sys.stdout.flush()
		print("Sky-mask worker: READY sent", file=sys.stderr, flush=True)

		images = _read_image_list(sys.stdin.buffer)
		print(f"Sky-mask worker: received {len(images)} images", file=sys.stderr, flush=True)
		masks: List[Image.Image] = []
		for index, image in enumerate(images, start=1):
			print(f"Sky-mask worker: processing {index}/{len(images)}", file=sys.stderr, flush=True)
			bgr_image = _image_like_to_bgr_ndarray(image)
			result = inference_segmentor(model, bgr_image)
			mask = _sky_mask_from_segmentation(result[0].astype(np.uint8))
			masks.append(Image.fromarray(mask))

		if args.save_dir is not None:
			_save_masks([np.asarray(mask, dtype=np.uint8) for mask in masks], args.save_dir)

		print("Sky-mask worker: writing masks", file=sys.stderr, flush=True)
		_write_image_list(sys.stdout.buffer, masks)
		sys.stdout.flush()
		print("Sky-mask worker: done", file=sys.stderr, flush=True)
		return

	if args.image_paths:
		masks = extract_sky_masks(
			args.image_paths,
			segformer_path=args.segformer_path,
			config=args.config,
			checkpoint=args.checkpoint,
			device=args.device,
		)
		if args.save_dir is not None:
			names = [Path(path).stem for path in args.image_paths]
			_save_masks(masks, args.save_dir, names=names)
		return

	parser.print_help()


if __name__ == "__main__":
	print("Starting sky-mask extraction worker...", file=sys.stderr)
	_worker_main()
