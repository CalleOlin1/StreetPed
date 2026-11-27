#!/usr/bin/env python3
"""Measure one checkpoint's GPU inference latency for a representative frame.

Example:
    python tools/benchmark_inference.py work_dirs/run/checkpoint_final.pth
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from models.road_mesh import RoadMesh
from utils.misc import import_str


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Checkpoint with config.yaml beside it")
    parser.add_argument("--frame-index", type=int, default=0, help="Index in the full image set")
    parser.add_argument("--warmup", type=int, default=30, help="Untimed forward passes")
    parser.add_argument("--repeats", type=int, default=200, help="Timed forward passes")
    parser.add_argument("--downscale", type=float, default=1.0, help="Image downscale factor")
    parser.add_argument(
        "--refined-camera", action="store_true",
        help="Apply the checkpoint's camera refinement, as for original-view evaluation",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.downscale <= 0:
        parser.error("warmup must be >= 0, repeats >= 1, and downscale > 0")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to benchmark this renderer.")

    cfg = OmegaConf.load(args.checkpoint.parent / "config.yaml")
    cfg.data.pixel_source.downscale_when_loading = [args.downscale] * len(
        cfg.data.pixel_source.cameras
    )
    # Match eval.py's lightweight dataset setup. These inputs are unused by inference.
    cfg.data.project_lidar_on_images = False
    cfg.data.pixel_source.load_rgb_images = False
    cfg.data.pixel_source.load_dynamic_mask = False
    cfg.data.pixel_source.load_sky_mask = False
    if "load_road_mask" in cfg.data.pixel_source:
        cfg.data.pixel_source.load_road_mask = False
    cfg.data.lidar_source.load_lidar = False

    device = torch.device("cuda")
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(str(args.checkpoint), load_only_model=True)
    mesh_path = args.checkpoint.parent / "road_mesh.pth"
    if trainer.road_mesh is not None and mesh_path.is_file():
        trainer.road_mesh = RoadMesh.load_checkpoint(str(mesh_path), device=device)
    trainer.set_eval()

    if not 0 <= args.frame_index < len(dataset.full_image_set):
        raise SystemExit(f"frame-index must be in [0, {len(dataset.full_image_set) - 1}]")
    image_infos, camera_infos = dataset.full_image_set.get_image(
        args.frame_index, camera_downscale=trainer._get_downscale_factor()
    )
    image_infos = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in image_infos.items()
    }
    camera_infos = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in camera_infos.items()
    }
    height = int(camera_infos["height"])
    width = int(camera_infos["width"])
    novel_view = not args.refined_camera

    def infer():
        with torch.no_grad():
            return trainer(image_infos, camera_infos, novel_view=novel_view)

    for _ in range(args.warmup):
        result = infer()
        torch.cuda.synchronize()
        del result

    timings_ms = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = infer()
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000)
        del result

    mean_ms = statistics.mean(timings_ms)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Frame: {args.frame_index}, resolution: {width}x{height}, novel_view: {novel_view}")
    print(f"Warmup: {args.warmup}, timed runs: {args.repeats}")
    print(f"Inference: mean {mean_ms:.2f} ms, median {statistics.median(timings_ms):.2f} ms")
    print(f"Range: {min(timings_ms):.2f}–{max(timings_ms):.2f} ms; throughput: {1000 / mean_ms:.2f} frames/s")


if __name__ == "__main__":
    main()
