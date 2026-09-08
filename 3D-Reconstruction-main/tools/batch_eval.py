#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

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


DEFAULT_DATASET_OPTS = [
    "data.dataset=kitti",
    "data.pixel_source.type=datasets.kitti.kitti_sourceloader.KITTIPixelSource",
    "data.lidar_source.type=datasets.kitti.kitti_sourceloader.KITTILiDARSource",
    "data.pixel_source.cameras=[0]",
    "data.pixel_source.downscale_when_loading=[2]",
    "data.pixel_source.downscale=2",
]


def infer_scene_id(run_dir: Path) -> Optional[str]:
    """Infer the processed4sep scene id from a run directory name or config file."""
    for candidate in [run_dir.name, str(run_dir)]:
        match = re.search(r"scene_\d+_clip_\d+", candidate)
        if match:
            return match.group(0)

    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        match = re.search(r"scene_idx:\s*([A-Za-z0-9_\-]+)", text)
        if match:
            return match.group(1)

    return None


def find_run_checkpoints(runs_dir: Path, checkpoint_pattern: str) -> List[Path]:
    """Return all checkpoint files under the supplied run directory."""
    checkpoints: List[Path] = []
    if runs_dir.is_file():
        if runs_dir.name == checkpoint_pattern:
            checkpoints.append(runs_dir)
        return checkpoints

    for path in sorted(runs_dir.rglob("*")):
        if path.is_file() and path.name == checkpoint_pattern:
            checkpoints.append(path)
    return checkpoints


def resolve_scene_dir(scene_id: str, data_root: Optional[Path]) -> Optional[Path]:
    if data_root is None:
        return None
    scene_dir = data_root / scene_id
    if scene_dir.exists():
        return scene_dir
    return None


def build_eval_command(
    checkpoint: Path,
    scene_dir: Path,
    repo_root: Path,
    extra_opts: Sequence[str],
    trajectory_name: str = "clip_1",
) -> List[str]:
    trajectories_dir = scene_dir / "trajectories"
    clip_0 = trajectories_dir / "clip_0.npz"
    clip_1 = trajectories_dir / "clip_1.npz"
    clip_2 = trajectories_dir / "clip_2.npz"

    missing = [path for path in (clip_0, clip_1, clip_2) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing trajectory files for {scene_dir}: {', '.join(str(p) for p in missing)}"
        )

    trajectory_file = clip_1 if trajectory_name == "clip_1" else clip_0
    raw_reference = clip_2

    cmd = [
        sys.executable,
        str(repo_root / "tools" / "eval.py"),
        "--resume_from",
        str(checkpoint),
        "--skip_original_render",
        "--lazy_dataset",
        "--save_images",
        "--original_trajectory_raw",
        str(raw_reference),
        "--trajectory_file",
        str(trajectory_file),
    ]
    cmd.extend(DEFAULT_DATASET_OPTS)
    cmd.extend(extra_opts)
    return cmd


def discover_data_root(repo_root: Path, explicit_root: Optional[Path]) -> Optional[Path]:
    if explicit_root is not None:
        return explicit_root

    candidates = [
        repo_root / "data" / "paralane" / "processed4sep",
        repo_root / "data" / "paralane" / "processed4sep copy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if (repo_root / "data").exists():
        matches = sorted((repo_root / "data").glob("**/processed4sep*"))
        if matches:
            return matches[0]
    return None


def parse_scene_index(scene_id: str) -> Optional[int]:
    match = re.search(r"scene_(\d+)", scene_id)
    if match is None:
        return None
    return int(match.group(1))


def natural_sort_key(path: Path) -> tuple:
    def parse_part(part: str):
        if part.isdigit():
            return (0, int(part))
        return (1, part.lower())

    return tuple(parse_part(piece) for piece in re.split(r"(\d+)", path.name))


def get_raw_gt_images(scene_id: str, raw_root: Path) -> Optional[Path]:
    scene_index = parse_scene_index(scene_id)
    if scene_index is None:
        return None
    gt_images_root = raw_root / f"scene_{scene_index}" / "clip_0" / "images"
    if gt_images_root.exists():
        return gt_images_root
    return None


def load_image_array(path: Path):
    if Image is None or np is None:
        raise RuntimeError("Pillow and NumPy are required to compute image metrics.")
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_psnr(pred, target) -> float:
    if np is None:
        raise RuntimeError("NumPy is required to compute PSNR.")
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(pred, target) -> float:
    if structural_similarity is None:
        raise RuntimeError("scikit-image is required to compute SSIM.")
    return float(structural_similarity(pred, target, data_range=1.0, channel_axis=-1))


def get_lpips_model(device):
    if lpips is None or torch is None:
        return None
    model = lpips.LPIPS(net="alex").eval().to(device)
    return model


def compute_lpips(pred, target, model, device) -> float:
    if model is None or torch is None or np is None:
        return float("nan")
    pred_t = torch.from_numpy(np.transpose(pred, (2, 0, 1))).unsqueeze(0).to(device)
    target_t = torch.from_numpy(np.transpose(target, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        value = model(pred_t, target_t)
    return float(value.squeeze().item())


def pair_generated_images_with_gt(pred_dir: Path, gt_images_root: Path) -> List[tuple[str, str]]:
    gt_entries = sorted(gt_images_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    gt_frames = []
    for entry in gt_entries:
        if not entry.is_dir():
            continue
        front_img = entry / "CAMERA_FRONT.png"
        if front_img.exists():
            gt_frames.append(str(front_img))

    pred_files = sorted(pred_dir.glob("*.png"), key=lambda p: natural_sort_key(p))
    paired = []
    for index, pred_file in enumerate(pred_files):
        if index < len(gt_frames):
            paired.append((str(pred_file), gt_frames[index]))
    return paired


def compute_image_metrics_for_run(checkpoint: Path, scene_id: str, raw_root: Path) -> None:
    gt_images_root = get_raw_gt_images(scene_id, raw_root)
    if gt_images_root is None:
        print(f"Skipping metric comparison for {scene_id}: no raw GT found under {raw_root}")
        return

    generated_dirs = sorted(
        checkpoint.parent.rglob("images_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not generated_dirs:
        print(f"No generated image folders found under {checkpoint.parent} for {scene_id}")
        return

    if torch is None:
        device = None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = get_lpips_model(device) if device is not None else None
    summary_entries = []
    for pred_dir in generated_dirs:
        if not pred_dir.is_dir():
            continue
        pairings = pair_generated_images_with_gt(pred_dir, gt_images_root)
        if not pairings:
            continue

        per_frame_metrics = []
        for pred_path, gt_path in pairings:
            pred_img = load_image_array(Path(pred_path))
            gt_img = load_image_array(Path(gt_path))
            if pred_img.shape != gt_img.shape:
                if Image is None or np is None:
                    raise RuntimeError("Pillow and NumPy are required for resized image comparison.")
                pred_img = np.asarray(
                    Image.open(pred_path).convert("RGB").resize((gt_img.shape[1], gt_img.shape[0]), Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
            psnr = compute_psnr(pred_img, gt_img)
            ssim = compute_ssim(pred_img, gt_img)
            lpips_score = compute_lpips(pred_img, gt_img, lpips_model, device)
            per_frame_metrics.append(
                {
                    "pred": pred_path,
                    "gt": gt_path,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips_score,
                }
            )

        if np is None:
            mean_psnr = float("nan")
            mean_ssim = float("nan")
            mean_lpips = float("nan")
        else:
            mean_psnr = float(np.mean([entry["psnr"] for entry in per_frame_metrics])) if per_frame_metrics else float("nan")
            mean_ssim = float(np.mean([entry["ssim"] for entry in per_frame_metrics])) if per_frame_metrics else float("nan")
            mean_lpips = float(np.mean([entry["lpips"] for entry in per_frame_metrics])) if per_frame_metrics else float("nan")
        summary = {
            "scene_id": scene_id,
            "pred_dir": str(pred_dir),
            "gt_dir": str(gt_images_root),
            "num_images": len(per_frame_metrics),
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "mean_lpips": mean_lpips,
            "frames": per_frame_metrics,
        }
        summary_entries.append(summary)

        metrics_path = pred_dir / "metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        csv_path = pred_dir / "metrics.csv"
        if per_frame_metrics:
            fieldnames = ["pred", "gt", "psnr", "ssim", "lpips"]
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in per_frame_metrics:
                    writer.writerow({
                        "pred": row["pred"],
                        "gt": row["gt"],
                        "psnr": row["psnr"],
                        "ssim": row["ssim"],
                        "lpips": row["lpips"],
                    })

        print(
            f"Metrics for {scene_id} [{pred_dir.name}]: "
            f"PSNR={mean_psnr:.4f}, SSIM={mean_ssim:.4f}, LPIPS={mean_lpips:.4f}"
        )

    if summary_entries:
        overall_summary = {
            "scene_id": scene_id,
            "gt_dir": str(gt_images_root),
            "results": summary_entries,
        }
        batch_metrics_path = checkpoint.parent / "batch_eval_metrics.json"
        batch_metrics_path.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

        if summary_entries:
            per_scene_rows = []
            for result in summary_entries:
                per_scene_rows.append({
                    "pred_dir": result["pred_dir"],
                    "num_images": result["num_images"],
                    "mean_psnr": result["mean_psnr"],
                    "mean_ssim": result["mean_ssim"],
                    "mean_lpips": result["mean_lpips"],
                })
            summary_csv_path = checkpoint.parent.parent / f"{scene_id}_summary.csv"
            with summary_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["pred_dir", "num_images", "mean_psnr", "mean_ssim", "mean_lpips"])
                writer.writeheader()
                for row in per_scene_rows:
                    writer.writerow(row)

            if per_scene_rows:
                psnr_vals = [row["mean_psnr"] for row in per_scene_rows if row["mean_psnr"] == row["mean_psnr"]]
                ssim_vals = [row["mean_ssim"] for row in per_scene_rows if row["mean_ssim"] == row["mean_ssim"]]
                lpips_vals = [row["mean_lpips"] for row in per_scene_rows if row["mean_lpips"] == row["mean_lpips"]]
                def summarize(values):
                    if not values:
                        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
                    arr = np.asarray(values, dtype=np.float64)
                    return {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=0)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }
                aggregate = {
                    "scene_id": scene_id,
                    "psnr": summarize(psnr_vals),
                    "ssim": summarize(ssim_vals),
                    "lpips": summarize(lpips_vals),
                }
                aggregate_path = checkpoint.parent.parent / f"{scene_id}_aggregate.json"
                aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


def has_metrics_csv_for_trajectory(checkpoint: Path, trajectory_name: str) -> bool:
    target_dir_name = f"images_file_{trajectory_name}"
    metrics_paths = checkpoint.parent.rglob("metrics.csv")
    for metrics_path in metrics_paths:
        if metrics_path.parent.name == target_dir_name:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run eval.py for each checkpoint directory in a run folder, matching run names to "
            "scene folders under data/paralane/processed4sep and using clip_0/clip_2 trajectories "
            "for world-space alignment."
        )
    )
    parser.add_argument("runs_dir", type=Path, help="Folder containing many run directories")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root folder containing processed4sep scene directories (default: auto-discover).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Root folder containing raw ground-truth data (default: data/paralane/raw).",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default="checkpoint_final.pth",
        help="Checkpoint filename to evaluate for each run directory.",
    )
    parser.add_argument(
        "--extra-opt",
        dest="extra_opts",
        action="append",
        default=[],
        help="Extra CLI override passed to eval.py, e.g. data.scene_idx=scene_001_clip_002",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the eval commands that would be started without launching them.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = discover_data_root(repo_root, args.data_root)
    raw_root = args.raw_root if args.raw_root is not None else repo_root / "data" / "paralane" / "raw"
    runs_dir = args.runs_dir.resolve()

    if not runs_dir.exists():
        parser.error(f"Runs directory does not exist: {runs_dir}")

    checkpoints = find_run_checkpoints(runs_dir, args.checkpoint_pattern)
    if not checkpoints:
        print(f"No checkpoints matched '{args.checkpoint_pattern}' under {runs_dir}")
        return 1

    if data_root is None:
        print(
            "Could not find processed4sep data automatically; pass --data-root manually. "
            f"Looked under {repo_root / 'data'}"
        )
        return 1

    if not raw_root.exists():
        print(f"Could not find raw GT images under {raw_root}")
        return 1

    print(f"Using processed4sep root: {data_root}")
    print(f"Using raw GT root: {raw_root}")
    print(f"Found {len(checkpoints)} checkpoint(s) to evaluate.")

    for checkpoint in checkpoints:
        run_dir = checkpoint.parent
        scene_id = infer_scene_id(run_dir)
        if scene_id is None:
            print(f"Skipping {checkpoint}: could not infer scene id from run folder name or config.yaml")
            continue

        scene_dir = resolve_scene_dir(scene_id, data_root)
        if scene_dir is None:
            print(f"Skipping {checkpoint}: no scene directory found for {scene_id} under {data_root}")
            continue

        for trajectory_name in ("clip_1", "clip_0"):
            if has_metrics_csv_for_trajectory(checkpoint, trajectory_name):
                print(
                    f"Skipping render for scene {scene_id} ({trajectory_name}): "
                    "metrics.csv already exists."
                )
                continue

            try:
                cmd = build_eval_command(checkpoint, scene_dir, repo_root, args.extra_opts, trajectory_name)
            except FileNotFoundError as exc:
                print(f"Skipping {checkpoint} for {trajectory_name}: {exc}")
                continue

            print(f"Starting render for scene {scene_id} ({trajectory_name}): {checkpoint}")
            print("Command:")
            print(" ".join(cmd))
            if args.dry_run:
                continue

            proc = subprocess.Popen(cmd, cwd=str(repo_root))
            rc = proc.wait()
            if rc != 0:
                print(f"Render for {checkpoint} ({trajectory_name}) exited with code {rc}")
                return rc

            compute_image_metrics_for_run(checkpoint, scene_id, raw_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
