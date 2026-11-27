#!/usr/bin/env python3
"""Calculate full-scene, human, and car image metrics for CARLA novel views."""

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


REGIONS = (
    ("complete", "Complete scene", None),
    ("humans", "Humans", "human"),
    ("cars", "Cars", "vehicle"),
)


@dataclass
class RegionMetrics:
    psnr: List[float] = field(default_factory=list)
    ssim: List[float] = field(default_factory=list)
    lpips: List[float] = field(default_factory=list)

    def add(self, psnr: float, ssim: float, lpips_value: float) -> None:
        self.psnr.append(psnr)
        self.ssim.append(ssim)
        self.lpips.append(lpips_value)


def infer_scene_id(run_dir: Path) -> Optional[str]:
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        config_text = config_path.read_text(encoding="utf-8")
        match = re.search(r"^\s*scene_idx:\s*([A-Za-z0-9_-]+)\s*$", config_text, re.MULTILINE)
        if match:
            return match.group(1)

    # Typical run name: carla_carla_town01_drive_001_sync.
    match = re.fullmatch(r"carla_(carla_.+)", run_dir.name)
    return match.group(1) if match else None


def image_index(path: Path) -> Optional[int]:
    match = re.fullmatch(
        r"(?:frame_?)?(\d+)(?:_\d+)?\.(?:png|jpe?g)",
        path.name,
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def indexed_images(image_dir: Path) -> Dict[int, Path]:
    images: Dict[int, Path] = {}
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        index = image_index(path)
        if index is None:
            continue
        if index in images:
            raise RuntimeError(
                f"Duplicate frame index {index} in {image_dir}: "
                f"{images[index].name}, {path.name}"
            )
        images[index] = path
    return images


def find_run_dirs(runs_dir: Path) -> List[Path]:
    run_dirs = {
        videos_eval.parent
        for videos_eval in runs_dir.rglob("videos_eval")
        if videos_eval.is_dir()
    }
    if (runs_dir / "videos_eval").is_dir():
        run_dirs.add(runs_dir)
    return sorted(run_dirs)


def find_prediction_dir(
    run_dir: Path,
    view: str,
    iteration: Optional[int],
) -> Optional[Path]:
    candidates = [
        path
        for path in run_dir.rglob(f"images_{view}")
        if path.is_dir()
        and "videos_eval" in path.parts
        and re.fullmatch(r"novel_\d+", path.parent.name)
    ]
    if iteration is not None:
        candidates = [path for path in candidates if path.parent.name == f"novel_{iteration}"]
    if not candidates:
        return None

    def sort_key(path: Path) -> Tuple[int, float]:
        step = int(path.parent.name.removeprefix("novel_"))
        return step, path.stat().st_mtime

    return max(candidates, key=sort_key)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_region_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def masked_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    mse = float(np.mean((pred[mask] - gt[mask]) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def masked_lpips(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    model,
    device: torch.device,
) -> float:
    mask_3d = mask[..., None].astype(np.float32)
    masked_pred = pred * mask_3d + gt * (1.0 - mask_3d)
    pred_tensor = (
        torch.from_numpy(masked_pred.transpose(2, 0, 1)).unsqueeze(0).to(device) * 2.0 - 1.0
    )
    gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).to(device) * 2.0 - 1.0
    with torch.no_grad():
        return float(model(pred_tensor, gt_tensor).squeeze().item())


def safe_mean(values: List[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return float("nan")
    if any(value == float("inf") for value in valid):
        return float("inf")
    return sum(valid) / len(valid)


def validate_indices(label: str, expected: set, actual: set) -> None:
    if expected == actual:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise RuntimeError(
        f"{label} frame indices differ from GT: missing={missing[:10]}, extra={extra[:10]}"
    )


def evaluate_scene(
    pred_dir: Path,
    gt_view_dir: Path,
    mask_set: str,
    model,
    device: torch.device,
    totals: Dict[str, RegionMetrics],
) -> int:
    pred_paths = indexed_images(pred_dir)
    gt_paths = indexed_images(gt_view_dir / "images")
    if not pred_paths or not gt_paths:
        raise RuntimeError(f"Empty prediction or GT image folder: {pred_dir}, {gt_view_dir / 'images'}")

    indices = set(gt_paths)
    validate_indices("Prediction", indices, set(pred_paths))

    masks_by_region: Dict[str, Dict[int, Path]] = {}
    for region_key, _, mask_name in REGIONS:
        if mask_name is None:
            continue
        mask_dir = gt_view_dir / mask_set / mask_name
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Required {region_key} mask directory not found: {mask_dir}")
        masks_by_region[region_key] = indexed_images(mask_dir)
        validate_indices(f"{region_key.title()} mask", indices, set(masks_by_region[region_key]))

    for index in sorted(indices):
        pred = load_rgb(pred_paths[index])
        gt = load_rgb(gt_paths[index])
        if pred.shape != gt.shape:
            raise RuntimeError(
                f"Frame {index} shape mismatch: prediction={pred.shape}, GT={gt.shape}"
            )

        _, ssim_map = structural_similarity(
            pred,
            gt,
            data_range=1.0,
            channel_axis=-1,
            full=True,
        )
        for region_key, _, mask_name in REGIONS:
            mask = (
                np.ones(gt.shape[:2], dtype=bool)
                if mask_name is None
                else load_region_mask(masks_by_region[region_key][index])
            )
            if mask.shape != gt.shape[:2]:
                raise RuntimeError(
                    f"Frame {index} {region_key} mask shape={mask.shape}, GT={gt.shape[:2]}"
                )
            if not np.any(mask):
                continue
            totals[region_key].add(
                masked_psnr(pred, gt, mask),
                float(ssim_map[mask].mean()),
                masked_lpips(pred, gt, mask, model, device),
            )
    return len(indices)


def print_results(totals: Dict[str, RegionMetrics]) -> None:
    print("\nCARLA metrics")
    for region_key, title, _ in REGIONS:
        values = totals[region_key]
        print(f"\n{title}")
        print("PSNR, SSIM, LPIPS")
        print(
            f"{safe_mean(values.psnr):.6f}, "
            f"{safe_mean(values.ssim):.6f}, "
            f"{safe_mean(values.lpips):.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate complete-scene, human, and car metrics for CARLA novel views."
    )
    parser.add_argument("runs_dir", type=Path, help="Folder containing CARLA run directories.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Processed CARLA root (default: <repo>/data/carla/processed).",
    )
    parser.add_argument("--view", default="right_4m", help="Novel view to evaluate.")
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Render iteration to use (default: latest available novel_<iteration>).",
    )
    parser.add_argument(
        "--mask-set",
        choices=("dynamic_masks", "fine_dynamic_masks"),
        default="dynamic_masks",
        help="CARLA mask set used for Humans and Cars.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = (
        args.data_root.resolve()
        if args.data_root
        else repo_root / "data" / "carla" / "processed"
    )
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Processed CARLA root not found: {data_root}")

    run_dirs = find_run_dirs(runs_dir)
    if not run_dirs:
        raise RuntimeError(f"No runs containing videos_eval found under {runs_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex").eval().to(device)
    totals = {region_key: RegionMetrics() for region_key, _, _ in REGIONS}
    evaluated_scenes = 0
    for run_dir in run_dirs:
        scene_id = infer_scene_id(run_dir)
        if scene_id is None:
            print(f"Skipping {run_dir}: could not infer scene_idx.")
            continue
        scene_dir = data_root / scene_id
        if not scene_dir.is_dir():
            print(f"Skipping {run_dir.name}: exact GT scene not found at {scene_dir}.")
            continue
        pred_dir = find_prediction_dir(run_dir, args.view, args.iteration)
        if pred_dir is None:
            requested = f" at iteration {args.iteration}" if args.iteration is not None else ""
            print(f"Skipping {scene_id}: no images_{args.view} render found{requested}.")
            continue
        gt_view_dir = scene_dir / "novel_views" / args.view
        if not (gt_view_dir / "images").is_dir():
            print(f"Skipping {scene_id}: GT view not found at {gt_view_dir / 'images'}.")
            continue

        print(
            f"Evaluating {scene_id} {args.view}: "
            f"{pred_dir.parent.name} -> {gt_view_dir}"
        )
        frame_count = evaluate_scene(
            pred_dir,
            gt_view_dir,
            args.mask_set,
            model,
            device,
            totals,
        )
        print(f"  matched {frame_count} frames")
        evaluated_scenes += 1

    if evaluated_scenes == 0:
        raise RuntimeError("No exact CARLA scene/view pairs were evaluated.")
    print(f"\nEvaluated {evaluated_scenes} scene(s), view={args.view}")
    print_results(totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
