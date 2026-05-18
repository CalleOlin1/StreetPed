"""Compute PSNR, SSIM, and LPIPS between two folders of images.

Example:
	python datasets/benchmark/metrics_folders.py \
		--gt-folder /path/to/ground_truth \
		--pred-folder /path/to/predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compute PSNR/SSIM/LPIPS for image pairs from two folders."
	)
	parser.add_argument(
		"--gt-folder",
		type=Path,
		required=True,
		help="Path to the ground-truth image folder.",
	)
	parser.add_argument(
		"--pred-folder",
		type=Path,
		required=True,
		help="Path to the prediction/novel-view image folder.",
	)
	parser.add_argument(
		"--mask-folder",
		type=Path,
		default=None,
		help=(
			"Optional path to per-image masks. Black pixels are valid for loss/metrics and "
			"white pixels are ignored."
		),
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Recursively search for images in subfolders.",
	)
	parser.add_argument(
		"--resize-pred-to-gt",
		action="store_true",
		help="Resize prediction image to GT resolution when shapes do not match.",
	)
	parser.add_argument(
		"--device",
		choices=["auto", "cpu", "cuda"],
		default="auto",
		help="Device for LPIPS computation.",
	)
	parser.add_argument(
		"--skip-lpips",
		action="store_true",
		help="Skip LPIPS to avoid model download/loading and compute only PSNR/SSIM.",
	)
	parser.add_argument(
		"--lpips-net",
		choices=["alex", "squeeze", "vgg"],
		default="alex",
		help="Backbone network for LPIPS.",
	)
	parser.add_argument(
		"--bottom-half-only",
		action="store_true",
		help="Evaluate metrics using only the bottom 50% of each image.",
	)
	parser.add_argument(
		"--plot-best-worst",
		action="store_true",
		help="Save a comparison plot for best and worst image pairs.",
	)
	parser.add_argument(
		"--min-frame-index",
		type=int,
		default=None,
		help=(
			"Only evaluate images whose filename contains an integer frame index >= this value. "
			"If a filename has no integer, it is kept."
		),
	)
	parser.add_argument(
		"--plot-out",
		type=Path,
		default=None,
		help=(
			"Output path for the best/worst comparison plot. "
			"Default: <pred-folder>/../best_worst_comparison.png"
		),
	)
	return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
	if device_arg == "cpu":
		return torch.device("cpu")
	if device_arg == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("--device=cuda requested but CUDA is not available.")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(folder: Path, recursive: bool) -> Dict[str, Path]:
	if not folder.exists() or not folder.is_dir():
		raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")

	image_paths = folder.rglob("*") if recursive else folder.glob("*")
	image_map: Dict[str, Path] = {}
	for image_path in image_paths:
		if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
			relative_key = str(image_path.relative_to(folder)).replace("\\", "/")
			image_map[relative_key] = image_path
	return image_map


def natural_sort_key(path_like: str) -> List[object]:
	# Sort "frame2" before "frame10" to keep temporal ordering stable.
	parts = re.split(r"(\d+)", path_like)
	return [int(part) if part.isdigit() else part.lower() for part in parts]


def extract_frame_index(path_like: str) -> Optional[int]:
	stem = Path(path_like).stem
	matches = re.findall(r"\d+", stem)
	if not matches:
		return None
	# Prefer the longest numeric token, which is usually the frame id.
	# This avoids picking short incidental tokens like "...8bit".
	longest_len = max(len(token) for token in matches)
	longest_tokens = [token for token in matches if len(token) == longest_len]
	return int(longest_tokens[0])


def apply_min_frame_index_filter(keys: List[str], min_frame_index: Optional[int]) -> List[str]:
	if min_frame_index is None:
		return keys

	filtered: List[str] = []
	for key in keys:
		frame_idx = extract_frame_index(key)
		if frame_idx is None or frame_idx >= min_frame_index:
			filtered.append(key)
	return filtered


def load_rgb_image(path: Path) -> np.ndarray:
	image = Image.open(path).convert("RGB")
	return np.asarray(image, dtype=np.float32) / 255.0


def load_mask_image(path: Path) -> np.ndarray:
	mask = Image.open(path).convert("L")
	return np.asarray(mask, dtype=np.float32) / 255.0


def maybe_resize_prediction(pred: np.ndarray, gt: np.ndarray, do_resize: bool) -> np.ndarray:
	if pred.shape == gt.shape:
		return pred

	if not do_resize:
		raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")

	gt_h, gt_w = gt.shape[:2]
	pred_img = Image.fromarray((pred * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")
	pred_img = pred_img.resize((gt_w, gt_h), resample=Image.BICUBIC)
	return np.asarray(pred_img, dtype=np.float32) / 255.0


def crop_bottom_half(image: np.ndarray) -> np.ndarray:
	h = image.shape[0]
	start_row = h // 2
	return image[start_row:, :, :]


def crop_bottom_half_mask(mask: np.ndarray) -> np.ndarray:
	h = mask.shape[0]
	start_row = h // 2
	return mask[start_row:, :]


def load_for_display(path: Path, bottom_half_only: bool) -> np.ndarray:
	img = load_rgb_image(path)
	if bottom_half_only:
		img = crop_bottom_half(img)
	return img


def save_best_worst_plot(
	pair_results: List[Dict[str, object]],
	bottom_half_only: bool,
	out_path: Path,
	compute_lpips: bool,
) -> None:
	if not pair_results:
		return

	if compute_lpips:
		best = min(pair_results, key=lambda x: float(x["lpips"]))
		worst = max(pair_results, key=lambda x: float(x["lpips"]))
		sort_metric = "LPIPS (lower is better)"
	else:
		best = max(pair_results, key=lambda x: float(x["psnr"]))
		worst = min(pair_results, key=lambda x: float(x["psnr"]))
		sort_metric = "PSNR (higher is better)"

	fig, axes = plt.subplots(2, 2, figsize=(12, 9))

	best_gt = load_for_display(Path(str(best["gt_path"])), bottom_half_only)
	best_pred = load_for_display(Path(str(best["pred_path"])), bottom_half_only)
	worst_gt = load_for_display(Path(str(worst["gt_path"])), bottom_half_only)
	worst_pred = load_for_display(Path(str(worst["pred_path"])), bottom_half_only)

	axes[0, 0].imshow(best_gt)
	axes[0, 0].set_title(f"Best GT\n{best['gt_key']}")
	axes[0, 1].imshow(best_pred)
	axes[0, 1].set_title(
		"Best Pred\n"
		f"PSNR={float(best['psnr']):.3f}, SSIM={float(best['ssim']):.3f}, "
		f"LPIPS={float(best['lpips']):.3f}"
	)
	axes[1, 0].imshow(worst_gt)
	axes[1, 0].set_title(f"Worst GT\n{worst['gt_key']}")
	axes[1, 1].imshow(worst_pred)
	axes[1, 1].set_title(
		"Worst Pred\n"
		f"PSNR={float(worst['psnr']):.3f}, SSIM={float(worst['ssim']):.3f}, "
		f"LPIPS={float(worst['lpips']):.3f}"
	)

	for ax in axes.flat:
		ax.axis("off")

	fig.suptitle(f"Best vs Worst Pair Comparison by {sort_metric}")
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def resolve_plot_out_path(pred_folder: Path, plot_out: Optional[Path]) -> Path:
	if plot_out is not None:
		return plot_out
	return pred_folder.parent / "best_worst_comparison.png"


def to_lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
	return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def masked_psnr(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> float:
	valid_pixels = valid_mask.sum()
	if valid_pixels <= 0:
		raise ValueError("Mask has no valid pixels.")

	diff = gt - pred
	mse = np.sum((diff * diff) * valid_mask[..., None]) / (valid_pixels * 3.0)
	if mse <= 1e-12:
		return float("inf")
	return float(10.0 * np.log10(1.0 / mse))


def masked_ssim(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> float:
	valid_pixels = valid_mask.sum()
	if valid_pixels <= 0:
		raise ValueError("Mask has no valid pixels.")

	_, ssim_map = structural_similarity(
		gt,
		pred,
		data_range=1.0,
		channel_axis=2,
		full=True,
	)

	if ssim_map.ndim == 3:
		ssim_map = np.mean(ssim_map, axis=2)

	return float(np.sum(ssim_map * valid_mask) / valid_pixels)


def apply_mask_to_image(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
	return image * valid_mask[..., None]


def evaluate_folders(
	gt_folder: Path,
	pred_folder: Path,
	mask_folder: Optional[Path],
	recursive: bool,
	resize_pred_to_gt: bool,
	device: torch.device,
	compute_lpips: bool,
	lpips_net: str,
	bottom_half_only: bool,
	min_frame_index: Optional[int],
) -> Tuple[List[float], List[float], List[float], int, int, List[str], List[str], List[Dict[str, object]]]:
	gt_images = list_images(gt_folder, recursive)
	pred_images = list_images(pred_folder, recursive)
	mask_images: Optional[Dict[str, Path]] = None
	if mask_folder is not None:
		mask_images = list_images(mask_folder, recursive)

	gt_sorted_keys = sorted(gt_images.keys(), key=natural_sort_key)
	pred_sorted_keys = sorted(pred_images.keys(), key=natural_sort_key)
	mask_sorted_keys: List[str] = []
	if mask_images is not None:
		mask_sorted_keys = sorted(mask_images.keys(), key=natural_sort_key)
	gt_total_count = len(gt_sorted_keys)
	pred_total_count = len(pred_sorted_keys)
	mask_total_count = len(mask_sorted_keys)
	gt_sorted_keys = apply_min_frame_index_filter(gt_sorted_keys, min_frame_index)
	pred_sorted_keys = apply_min_frame_index_filter(pred_sorted_keys, min_frame_index)
	if mask_images is not None:
		mask_sorted_keys = apply_min_frame_index_filter(mask_sorted_keys, min_frame_index)

	if min_frame_index is not None:
		msg = (
			f"Applied min frame index {min_frame_index}: "
			f"GT {len(gt_sorted_keys)}/{gt_total_count}, "
			f"Pred {len(pred_sorted_keys)}/{pred_total_count}"
		)
		if mask_images is not None:
			msg += f", Mask {len(mask_sorted_keys)}/{mask_total_count}"
		print(msg, flush=True)

	if not gt_sorted_keys or not pred_sorted_keys or (mask_images is not None and not mask_sorted_keys):
		raise RuntimeError(
			"No images were found in one or both folders after filtering. "
			"Check --min-frame-index and filename format."
		)

	pair_count = min(
		len(gt_sorted_keys),
		len(pred_sorted_keys),
		len(mask_sorted_keys) if mask_images is not None else len(gt_sorted_keys),
	)
	paired_gt_keys = gt_sorted_keys[:pair_count]
	paired_pred_keys = pred_sorted_keys[:pair_count]
	paired_mask_keys: Optional[List[str]] = None
	if mask_images is not None:
		paired_mask_keys = mask_sorted_keys[:pair_count]

	missing_in_pred = gt_sorted_keys[pair_count:]
	extra_in_pred = pred_sorted_keys[pair_count:]

	lpips_model: Optional[lpips.LPIPS] = None
	if compute_lpips:
		print(f"Loading LPIPS model (net={lpips_net}) on {device}...", flush=True)
		lpips_model = lpips.LPIPS(net=lpips_net).to(device)
		lpips_model.eval()
		print("LPIPS model loaded.", flush=True)

	psnr_values: List[float] = []
	ssim_values: List[float] = []
	lpips_values: List[float] = []
	pair_results: List[Dict[str, object]] = []
	skipped_pairs = 0
	skipped_empty_mask_pairs = 0

	with torch.no_grad():
		for idx, (gt_key, pred_key) in enumerate(zip(paired_gt_keys, paired_pred_keys)):
			gt_img = load_rgb_image(gt_images[gt_key])
			pred_img = load_rgb_image(pred_images[pred_key])

			try:
				pred_img = maybe_resize_prediction(pred_img, gt_img, resize_pred_to_gt)
			except ValueError:
				skipped_pairs += 1
				continue

			valid_mask: Optional[np.ndarray] = None
			if mask_images is not None and paired_mask_keys is not None:
				mask_key = paired_mask_keys[idx]
				mask_img = load_mask_image(mask_images[mask_key])

				if mask_img.shape != gt_img.shape[:2]:
					if not resize_pred_to_gt:
						skipped_pairs += 1
						continue
					mask_pil = Image.fromarray((mask_img * 255.0).clip(0, 255).astype(np.uint8), mode="L")
					mask_pil = mask_pil.resize((gt_img.shape[1], gt_img.shape[0]), resample=Image.NEAREST)
					mask_img = np.asarray(mask_pil, dtype=np.float32) / 255.0

				valid_mask = (mask_img < 0.5).astype(np.float32)

			if bottom_half_only:
				gt_img = crop_bottom_half(gt_img)
				pred_img = crop_bottom_half(pred_img)
				if valid_mask is not None:
					valid_mask = crop_bottom_half_mask(valid_mask)

			if valid_mask is not None and valid_mask.sum() <= 0:
				skipped_empty_mask_pairs += 1
				continue

			if valid_mask is None:
				psnr_score = peak_signal_noise_ratio(gt_img, pred_img, data_range=1.0)
				ssim_score = structural_similarity(gt_img, pred_img, data_range=1.0, channel_axis=2)
			else:
				psnr_score = masked_psnr(gt_img, pred_img, valid_mask)
				ssim_score = masked_ssim(gt_img, pred_img, valid_mask)

			lpips_score = float("nan")
			if lpips_model is not None:
				if valid_mask is not None:
					gt_lpips_img = apply_mask_to_image(gt_img, valid_mask)
					pred_lpips_img = apply_mask_to_image(pred_img, valid_mask)
				else:
					gt_lpips_img = gt_img
					pred_lpips_img = pred_img

				# LPIPS has no native pixel mask, so invalid regions are neutralized to zero in both images.
				gt_tensor = to_lpips_tensor(gt_lpips_img, device)
				pred_tensor = to_lpips_tensor(pred_lpips_img, device)
				lpips_score = lpips_model(pred_tensor, gt_tensor, normalize=True).item()

			psnr_values.append(float(psnr_score))
			ssim_values.append(float(ssim_score))
			lpips_values.append(float(lpips_score))
			pair_results.append(
				{
					"gt_key": gt_key,
					"pred_key": pred_key,
					"gt_path": gt_images[gt_key],
					"pred_path": pred_images[pred_key],
					"psnr": float(psnr_score),
					"ssim": float(ssim_score),
					"lpips": float(lpips_score),
				}
			)

	return (
		psnr_values,
		ssim_values,
		lpips_values,
		skipped_pairs,
		skipped_empty_mask_pairs,
		missing_in_pred,
		extra_in_pred,
		pair_results,
	)


def main() -> None:
	args = parse_args()
	device = resolve_device(args.device)

	(
		psnr_values,
		ssim_values,
		lpips_values,
		skipped_pairs,
		skipped_empty_mask_pairs,
		missing_in_pred,
		extra_in_pred,
		pair_results,
	) = (
		evaluate_folders(
			gt_folder=args.gt_folder,
			pred_folder=args.pred_folder,
			mask_folder=args.mask_folder,
			recursive=args.recursive,
			resize_pred_to_gt=args.resize_pred_to_gt,
			device=device,
			compute_lpips=not args.skip_lpips,
			lpips_net=args.lpips_net,
			bottom_half_only=args.bottom_half_only,
			min_frame_index=args.min_frame_index,
		)
	)

	if not psnr_values:
		raise RuntimeError(
			"All matching image pairs were skipped (likely due to size mismatch). "
			"Try --resize-pred-to-gt."
		)

	print(f"Evaluated pairs: {len(psnr_values)}")
	print(f"Skipped pairs: {skipped_pairs}")
	if args.mask_folder is not None:
		print(f"Skipped pairs (empty valid mask): {skipped_empty_mask_pairs}")
	print(f"Missing in predictions: {len(missing_in_pred)}")
	print(f"Extra in predictions: {len(extra_in_pred)}")
	print(f"Average PSNR : {np.mean(psnr_values):.4f}")
	print(f"Average SSIM : {np.mean(ssim_values):.4f}")
	if args.skip_lpips:
		print("Average LPIPS: skipped")
	else:
		print(f"Average LPIPS: {np.nanmean(lpips_values):.4f}")

	if args.plot_best_worst:
		plot_out_path = resolve_plot_out_path(args.pred_folder, args.plot_out)
		save_best_worst_plot(
			pair_results=pair_results,
			bottom_half_only=args.bottom_half_only,
			out_path=plot_out_path,
			compute_lpips=not args.skip_lpips,
		)
		print(f"Saved best/worst comparison plot to: {plot_out_path}")


if __name__ == "__main__":
	main()
