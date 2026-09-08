#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover
    structural_similarity = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import lpips
except ImportError:  # pragma: no cover
    lpips = None


def infer_scene_id(run_dir: Path) -> Optional[str]:
    for candidate in [run_dir.name, str(run_dir)]:
        match = re.search(r"scene_\d+_clip_\d+", candidate)
        if match:
            return match.group(0)

    config_path = run_dir / "config.yaml"
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        match = re.search(r"scene_idx:\s*([A-Za-z0-9_\-]+)", text)
        if match:
            return match.group(1)
    return None


def parse_scene_index(scene_id: str) -> Optional[int]:
    match = re.search(r"scene_(\d+)", scene_id)
    if match is None:
        return None
    return int(match.group(1))


def load_rgb(path: Path) -> np.ndarray:
    if Image is None or np is None:
        raise RuntimeError("Pillow and NumPy are required.")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    if Image is None or np is None:
        raise RuntimeError("Pillow and NumPy are required.")
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    dynamic = mask > 0.5
    return ~dynamic


def compute_masked_psnr(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    if np is None:
        raise RuntimeError("NumPy is required.")
    valid = valid_mask.astype(bool)
    if valid.sum() == 0:
        return float("nan")
    diff = pred - gt
    mse = float(np.mean((diff[valid]) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def compute_masked_ssim(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    if structural_similarity is None:
        raise RuntimeError("scikit-image is required.")
    valid = valid_mask.astype(bool)
    if valid.sum() == 0:
        return float("nan")
    _, ssim_map = structural_similarity(
        pred,
        gt,
        data_range=1.0,
        channel_axis=-1,
        full=True,
    )
    return float(ssim_map[valid].mean())


def get_lpips_model(device):
    if lpips is None or torch is None:
        return None
    return lpips.LPIPS(net="alex").eval().to(device)


def compute_masked_lpips(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    model,
    device,
) -> float:
    if model is None or torch is None or np is None:
        return float("nan")
    valid = valid_mask[..., None].astype(np.float32)
    masked_pred = pred * valid + gt * (1.0 - valid)
    pred_t = torch.from_numpy(masked_pred.transpose(2, 0, 1)).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(pred_t, gt_t)
    return float(score.squeeze().item())


def get_gt_frame_paths(raw_root: Path, scene_id: str, clip_name: str) -> List[Path]:
    scene_idx = parse_scene_index(scene_id)
    if scene_idx is None:
        return []
    gt_root = raw_root / f"scene_{scene_idx}" / clip_name / "images"
    if not gt_root.exists():
        return []
    timestamp_dirs = sorted(
        [p for p in gt_root.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )
    frames = []
    for ts_dir in timestamp_dirs:
        img = ts_dir / "CAMERA_FRONT.png"
        if img.exists():
            frames.append(img)
    return frames


def get_mask_paths(raw_root: Path, scene_id: str, clip_name: str) -> List[Path]:
    scene_idx = parse_scene_index(scene_id)
    if scene_idx is None:
        return []
    mask_root = raw_root / f"scene_{scene_idx}" / clip_name / "foreground_labels"
    if not mask_root.exists():
        return []
    timestamp_dirs = sorted(
        [p for p in mask_root.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )
    masks = []
    for ts_dir in timestamp_dirs:
        mask = ts_dir / "CAMERA_FRONT.png"
        if mask.exists():
            masks.append(mask)
    return masks


def find_clip_rendered_images_dir(run_dir: Path, clip_name: str) -> Optional[Path]:
    candidates = [
        p
        for p in run_dir.rglob(f"images_file_{clip_name}")
        if p.is_dir() and "videos_eval" in str(p)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _frame_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"frame(\d+)\.png$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def load_rendered_frames(image_dir: Path) -> List[np.ndarray]:
    frame_paths = sorted(image_dir.glob("frame*.png"), key=_frame_sort_key)
    return [load_rgb(frame_path) for frame_path in frame_paths]


def safe_mean(values: List[float]) -> float:
    finite = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def evaluate_clip(
    run_dir: Path,
    scene_id: str,
    clip_name: str,
    raw_root: Path,
    lpips_model,
    device,
) -> Optional[Tuple[float, float, float]]:
    rendered_images_dir = find_clip_rendered_images_dir(run_dir, clip_name)
    if rendered_images_dir is None:
        print(f"Skipping {scene_id} {clip_name}: no rendered images found in videos_eval.")
        return None

    gt_paths = get_gt_frame_paths(raw_root, scene_id, clip_name)
    if not gt_paths:
        print(f"Skipping {scene_id} {clip_name}: no GT frames found.")
        return None

    mask_paths = get_mask_paths(raw_root, scene_id, clip_name)
    if not mask_paths:
        print(f"Skipping {scene_id} {clip_name}: no dynamic masks found.")
        return None

    pred_frames = load_rendered_frames(rendered_images_dir)
    frame_count = min(len(pred_frames), len(gt_paths), len(mask_paths))
    if frame_count == 0:
        print(f"Skipping {scene_id} {clip_name}: empty frame intersection.")
        return None

    psnrs: List[float] = []
    ssims: List[float] = []
    lpipss: List[float] = []
    for i in range(frame_count):
        pred = pred_frames[i]
        gt = load_rgb(gt_paths[i])
        valid_mask = load_mask(mask_paths[i])

        if pred.shape[:2] != gt.shape[:2]:
            if Image is None or np is None:
                raise RuntimeError("Pillow and NumPy are required for image resizing.")
            pred = np.asarray(
                Image.fromarray((pred * 255.0).astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]),
                    Image.BILINEAR,
                ),
                dtype=np.float32,
            ) / 255.0

        if valid_mask.shape != gt.shape[:2]:
            if Image is None or np is None:
                raise RuntimeError("Pillow and NumPy are required for mask resizing.")
            valid_mask = np.asarray(
                Image.fromarray((valid_mask.astype(np.uint8) * 255)).resize(
                    (gt.shape[1], gt.shape[0]),
                    Image.NEAREST,
                ),
                dtype=np.uint8,
            ) > 0

        psnrs.append(compute_masked_psnr(pred, gt, valid_mask))
        ssims.append(compute_masked_ssim(pred, gt, valid_mask))
        lpipss.append(compute_masked_lpips(pred, gt, valid_mask, lpips_model, device))

    mean_psnr = safe_mean(psnrs)
    mean_ssim = safe_mean(ssims)
    mean_lpips = safe_mean(lpipss)
    print(
        f"{scene_id} {clip_name}: PSNR={mean_psnr:.6f}, "
        f"SSIM={mean_ssim:.6f}, LPIPS={mean_lpips:.6f}"
    )
    return mean_psnr, mean_ssim, mean_lpips


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate masked PSNR/SSIM/LPIPS for clip_0 and clip_1 from pre-exported "
            "rendered images under videos_eval and print terminal summary."
        )
    )
    parser.add_argument("runs_dir", type=Path, help="Folder containing run directories.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Root folder containing raw GT scene folders and foreground_labels masks.",
    )
    args = parser.parse_args()

    if np is None or Image is None or structural_similarity is None:
        raise RuntimeError("Missing required dependencies: numpy, pillow, scikit-image.")

    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    raw_root = args.raw_root.resolve() if args.raw_root else repo_root / "data" / "paralane" / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else None
    lpips_model = get_lpips_model(device) if device is not None else None

    clip0_vals: Dict[str, List[float]] = {"psnr": [], "ssim": [], "lpips": []}
    clip1_vals: Dict[str, List[float]] = {"psnr": [], "ssim": [], "lpips": []}

    run_dirs = sorted([p for p in runs_dir.rglob("*") if (p / "checkpoint_final.pth").exists()])
    if not run_dirs:
        if (runs_dir / "checkpoint_final.pth").exists():
            run_dirs = [runs_dir]
        else:
            raise RuntimeError(f"No run directories with checkpoint_final.pth found under {runs_dir}")

    for run_dir in run_dirs:
        scene_id = infer_scene_id(run_dir)
        if scene_id is None:
            print(f"Skipping {run_dir}: could not infer scene id.")
            continue
        clip0 = evaluate_clip(run_dir, scene_id, "clip_0", raw_root, lpips_model, device)
        clip1 = evaluate_clip(run_dir, scene_id, "clip_1", raw_root, lpips_model, device)
        if clip0 is not None:
            clip0_vals["psnr"].append(clip0[0])
            clip0_vals["ssim"].append(clip0[1])
            clip0_vals["lpips"].append(clip0[2])
        if clip1 is not None:
            clip1_vals["psnr"].append(clip1[0])
            clip1_vals["ssim"].append(clip1[1])
            clip1_vals["lpips"].append(clip1[2])

    clip0_psnr = safe_mean(clip0_vals["psnr"])
    clip0_ssim = safe_mean(clip0_vals["ssim"])
    clip0_lpips = safe_mean(clip0_vals["lpips"])
    clip1_psnr = safe_mean(clip1_vals["psnr"])
    clip1_ssim = safe_mean(clip1_vals["ssim"])
    clip1_lpips = safe_mean(clip1_vals["lpips"])

    print("Clip 0, Clip 1,")
    print("PSNR, SSIM, LPIPS, PSNR, SSIM, LPIPS,")
    print(
        f"{clip0_psnr:.6f}, {clip0_ssim:.6f}, {clip0_lpips:.6f}, "
        f"{clip1_psnr:.6f}, {clip1_ssim:.6f}, {clip1_lpips:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
