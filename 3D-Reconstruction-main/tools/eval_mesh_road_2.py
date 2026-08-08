#!/usr/bin/env python3
"""Render a road-mesh video from a saved camera trajectory."""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval import _load_novel_trajectory_from_file
from tools.eval_mesh_road import (
    _PreparedNovelViewImageSet,
    _TrajectoryDataset,
    _TrajectoryImageSet,
    _build_scene_dataset,
    _frame_visibility_stats,
    _load_mesh_as_tensors,
    _load_texture_npz,
    _normalize_intrinsics,
    _normalize_sizes,
)
from tools.train_mesh_road import export_mesh_render_from_camera, logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render an exported OBJ road mesh from a trajectory file."
    )
    parser.add_argument("--scene_path", type=str, default=None, help="Scene data directory.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Dataset config file used with scene_path.",
    )
    parser.add_argument(
        "--mesh_obj",
        type=str,
        required=True,
        help="Path to the exported OBJ mesh.",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Path to trajectory .npz/.npy file with camera poses.",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Output mp4 path (default: next to mesh obj).",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    parser.add_argument(
        "--max_frames", type=int, default=None, help="Limit number of frames rendered."
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="Optional directory to keep rendered PNG frames.",
    )
    return parser.parse_args()


def _filter_aligned_fields(
    fields: Dict[str, np.ndarray], keep_mask: np.ndarray
) -> Dict[str, np.ndarray]:
    keep_len = int(keep_mask.shape[0])
    filtered: Dict[str, np.ndarray] = {}
    for key, value in fields.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == keep_len:
            filtered[key] = value[keep_mask]
        else:
            filtered[key] = value
    return filtered


def _load_trajectory_bundle(
    trajectory_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ext = Path(trajectory_path).suffix.lower()
    poses = _load_novel_trajectory_from_file(trajectory_path).detach().cpu().numpy()

    if ext == ".npy":
        widths = np.full(len(poses), 1280, dtype=np.int32)
        heights = np.full(len(poses), 720, dtype=np.int32)
        intrinsics = _normalize_intrinsics({}, len(poses), widths, heights)
        return poses, intrinsics, heights, widths

    if ext != ".npz":
        raise ValueError(f"Unsupported trajectory extension: {ext}")

    data = np.load(trajectory_path, allow_pickle=True)
    traj_data: Dict[str, np.ndarray] = {k: np.asarray(data[k]) for k in data.files}

    pose_key = next((k for k in ("camera_poses", "poses", "trajectory") if k in traj_data), None)
    if pose_key is None:
        raise ValueError(
            f"No pose key found in {trajectory_path}. Expected one of ('camera_poses', 'poses', 'trajectory')"
        )

    raw_poses = np.asarray(traj_data[pose_key])
    if raw_poses.ndim != 3:
        raise ValueError(f"Trajectory must be [N,4,4] or [N,3,4], got {raw_poses.shape}")
    if raw_poses.shape[-2:] == (3, 4):
        bottom = np.zeros((raw_poses.shape[0], 1, 4), dtype=raw_poses.dtype)
        bottom[:, 0, 3] = 1.0
        raw_poses = np.concatenate([raw_poses, bottom], axis=1)
    if raw_poses.shape[-2:] != (4, 4):
        raise ValueError(f"Trajectory poses must be [N,4,4], got {raw_poses.shape}")

    cam_ids_arr = traj_data.get("cam_ids")
    cam_names_arr = traj_data.get("cam_names")
    if isinstance(cam_ids_arr, np.ndarray) and cam_ids_arr.ndim > 0 and cam_ids_arr.shape[0] == len(
        raw_poses
    ):
        cam_ids = np.asarray(cam_ids_arr).reshape(-1)
        keep = cam_ids == cam_ids[0]
        traj_data = _filter_aligned_fields(traj_data, keep)
    elif isinstance(cam_names_arr, np.ndarray) and cam_names_arr.ndim > 0 and cam_names_arr.shape[0] == len(
        raw_poses
    ):
        cam_names = np.asarray(cam_names_arr).reshape(-1)
        keep = cam_names == cam_names[0]
        traj_data = _filter_aligned_fields(traj_data, keep)

    widths, heights = _normalize_sizes(traj_data, len(poses))
    intrinsics = _normalize_intrinsics(traj_data, len(poses), widths, heights)
    return poses, intrinsics, heights, widths


def _load_texture_from_mesh_dir(mesh_obj_path: str):
    mesh_dir = os.path.dirname(mesh_obj_path)
    png_candidates = [
        os.path.join(mesh_dir, "road_textured_mesh.png"),
        os.path.join(mesh_dir, "road_textured_mesh_texture.png"),
    ]
    npz_path = os.path.join(mesh_dir, "image_buffer.npz")

    texture_buffer, texture_metadata = None, None

    if os.path.isfile(npz_path):
        texture_buffer, texture_metadata = _load_texture_npz(npz_path)
        if texture_buffer is not None:
            logger.info("Loaded texture buffer from image_buffer.npz")

    if texture_buffer is None:
        for png_path in png_candidates:
            if os.path.isfile(png_path):
                texture_buffer = np.asarray(Image.open(png_path).convert("RGB")).astype(np.float32) / 255.0
                if texture_metadata is None:
                    texture_metadata = {
                        "x_range": None,
                        "y_range": None,
                        "width": texture_buffer.shape[1],
                        "height": texture_buffer.shape[0],
                        "pixels_per_meter": None,
                    }
                logger.info(f"Using PNG texture {png_path}")
                break

    return texture_buffer, texture_metadata


def main():
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mesh_obj = os.path.abspath(args.mesh_obj)
    if not os.path.isfile(mesh_obj):
        raise FileNotFoundError(f"Mesh OBJ not found: {mesh_obj}")

    output_video = (
        os.path.abspath(args.output_video)
        if args.output_video
        else os.path.join(os.path.dirname(mesh_obj), "mesh_trajectory_render.mp4")
    )
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    vertices_tensor, faces_tensor = _load_mesh_as_tensors(mesh_obj)
    scene_dataset = _build_scene_dataset(args.scene_path, args.config_path, device)

    render_dataset = None
    poses = None
    intrinsics = None
    heights = None
    widths = None

    if args.trajectory:
        trajectory = os.path.abspath(args.trajectory)
        if not os.path.isfile(trajectory):
            raise FileNotFoundError(f"Trajectory file not found: {trajectory}")

        poses, intrinsics, heights, widths = _load_trajectory_bundle(trajectory)
        if args.max_frames is not None:
            max_frames = max(int(args.max_frames), 0)
            poses = poses[:max_frames]
            intrinsics = intrinsics[:max_frames]
            heights = heights[:max_frames]
            widths = widths[:max_frames]
        if len(poses) == 0:
            raise ValueError("No poses to render after applying frame limits.")

        if scene_dataset is not None:
            try:
                traj_tensor = torch.as_tensor(poses, dtype=torch.float32)
                prepared_frames = scene_dataset.prepare_novel_view_render_data(traj_tensor)
                source_image_set = (
                    scene_dataset.full_image_set if hasattr(scene_dataset, "full_image_set") else None
                )
                image_set = _PreparedNovelViewImageSet(
                    prepared_frames,
                    source_image_set=source_image_set,
                )
                render_dataset = _TrajectoryDataset(image_set)
                logger.info("Using dataset.prepare_novel_view_render_data() for custom trajectory poses")
            except Exception as e:
                logger.warning(
                    f"Failed to use dataset.prepare_novel_view_render_data(): {e}. "
                    "Falling back to direct trajectory shims."
                )
                source_image_set = (
                    scene_dataset.full_image_set if hasattr(scene_dataset, "full_image_set") else None
                )
                image_set = _TrajectoryImageSet(
                    poses,
                    intrinsics,
                    heights,
                    widths,
                    source_image_set=source_image_set,
                )
                render_dataset = _TrajectoryDataset(image_set)
        else:
            logger.warning(
                "No scene dataset provided; using raw trajectory data without PixelSource processing."
            )
            image_set = _TrajectoryImageSet(poses, intrinsics, heights, widths, source_image_set=None)
            render_dataset = _TrajectoryDataset(image_set)
    else:
        if scene_dataset is None:
            raise ValueError("Provide --scene_path so the original scene trajectory can be used.")
        logger.info("No trajectory file provided; using the scene's original trajectory.")
        render_dataset = scene_dataset
        if args.max_frames is not None and hasattr(render_dataset, "train_indices"):
            max_frames = max(int(args.max_frames), 0)
            render_dataset.train_indices = render_dataset.train_indices[:max_frames]
        if len(render_dataset.train_indices) == 0:
            raise ValueError("No poses to render after applying frame limits.")

    texture_buffer, texture_metadata = _load_texture_from_mesh_dir(mesh_obj)

    verts_np = vertices_tensor.cpu().numpy() if torch.is_tensor(vertices_tensor) else np.asarray(vertices_tensor)
    bbox_min = verts_np.min(axis=0)
    bbox_max = verts_np.max(axis=0)
    mesh_center = (bbox_min + bbox_max) * 0.5
    logger.info(f"Mesh bbox min:{bbox_min}, max:{bbox_max}, center:{mesh_center}")
    if poses is not None:
        logger.info(f"Intrinsics shape: {intrinsics.shape}, poses shape: {poses.shape}")
    else:
        logger.info(f"Using scene dataset trajectory with {len(render_dataset.train_indices)} frames")

    if args.trajectory:
        c2w0 = poses[0]
    else:
        c2w0 = render_dataset.full_image_set.get_image(0, camera_downscale=1.0)[1]["camera_to_world"]
    c2w0 = c2w0.detach().cpu().numpy() if torch.is_tensor(c2w0) else np.asarray(c2w0)
    w2c0 = np.linalg.inv(c2w0)
    center_cam = (w2c0[:3, :3] @ mesh_center) + w2c0[:3, 3]
    visible_frac = float(np.mean((w2c0[:3, :3] @ verts_np.T + w2c0[:3, 3:4])[2, :] > 0))
    logger.info(
        f"Frame0 camera pos: {c2w0[:3,3]}, mesh_center_cam_z: {center_cam[2]:.3f}, "
        f"verts_in_front_frac: {visible_frac:.3f}"
    )

    frames_dir_ctx = None
    if args.frames_dir:
        frames_dir = os.path.abspath(args.frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
    else:
        frames_dir_ctx = tempfile.TemporaryDirectory(prefix="eval_mesh_road_frames_")
        frames_dir = frames_dir_ctx.name

    rendered_frames = []
    num_frames = len(render_dataset.train_indices) if hasattr(render_dataset, "train_indices") else len(poses)
    logger.info(f"Rendering {num_frames} frames from trajectory...")
    for i in range(num_frames):
        frame_path = export_mesh_render_from_camera(
            dataset=render_dataset,
            vertices_tensor=vertices_tensor,
            faces_tensor=faces_tensor,
            frame_idx=i,
            cam_id=0,
            output_dir=frames_dir,
            texture_buffer=texture_buffer,
            texture_metadata=texture_metadata,
        )
        if frame_path is None or not os.path.isfile(frame_path):
            raise RuntimeError(f"Failed to render frame {i}")
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot read rendered frame: {frame_path}")
        visible, mean_intensity, nonzero_pixels = _frame_visibility_stats(frame)
        logger.info(
            f"Frame {i:04d}: visible={visible} mean_intensity={mean_intensity:.2f} "
            f"nonzero_pixels={nonzero_pixels}"
        )
        rendered_frames.append(frame)

    logger.info(f"Encoding video to {output_video}...")
    first_frame = rendered_frames[0]
    frame_h, frame_w = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize video writer for {output_video}")
    for frame in rendered_frames:
        if frame.shape[:2] != (frame_h, frame_w):
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()

    if frames_dir_ctx is not None:
        frames_dir_ctx.cleanup()

    logger.info(f"Saved mesh trajectory render video to {output_video}")


if __name__ == "__main__":
    main()