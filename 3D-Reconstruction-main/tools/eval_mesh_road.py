#!/usr/bin/env python3
"""Render a road-mesh video from a saved camera trajectory."""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

# Add parent directory to path for imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.driving_dataset import DrivingDataset
from tools.train_mesh_road import export_mesh_render_from_camera, logger


DEFAULT_TRAJECTORY_PATH = (
    "output/paralane/paralane_full/camera_poses/aligned_cam_pose.npz"
)


class _TrajectoryImageSet:
    """Minimal image-set shim to satisfy export_mesh_render_from_camera()."""

    def __init__(
        self,
        poses: np.ndarray,
        intrinsics: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        source_image_set=None,
    ) -> None:
        self._poses = poses
        self._intrinsics = intrinsics
        self._heights = heights
        self._widths = widths
        self._source_image_set = source_image_set

    def __len__(self) -> int:
        return len(self._poses)

    def get_image(self, image_idx: int, camera_downscale: float = 1.0):
        idx = int(np.clip(image_idx, 0, len(self._poses) - 1))
        if self._source_image_set is not None and len(self._source_image_set) > 0:
            source_idx = min(idx, len(self._source_image_set) - 1)
            image_infos, cam_infos = self._source_image_set.get_image(
                source_idx, camera_downscale=camera_downscale
            )
        else:
            h = int(self._heights[idx])
            w = int(self._widths[idx])
            image_infos = {"pixels": np.zeros((h, w, 3), dtype=np.float32)}
            cam_infos = {}

        cam_infos = dict(cam_infos)
        cam_infos["intrinsics"] = torch.as_tensor(
            self._intrinsics[idx], dtype=torch.float32
        )
        cam_infos["camera_to_world"] = torch.as_tensor(
            self._poses[idx], dtype=torch.float32
        )
        return image_infos, cam_infos


class _TrajectoryDataset:
    """Minimal dataset shim to satisfy export_mesh_render_from_camera()."""

    def __init__(self, image_set) -> None:
        self.full_image_set = image_set
        self.train_indices = list(range(len(image_set)))


class _PreparedNovelViewImageSet:
    """Wrapper around prepared novel view data from dataset.prepare_novel_view_render_data()."""

    def __init__(self, prepared_frames: list, source_image_set=None) -> None:
        """
        Args:
            prepared_frames: List of dicts from dataset.prepare_novel_view_render_data(),
                           each containing 'image_infos' and 'cam_infos'.
            source_image_set: Optional source image set for fallback texture data.
        """
        self._prepared_frames = prepared_frames
        self._source_image_set = source_image_set

    def __len__(self) -> int:
        return len(self._prepared_frames)

    def get_image(self, image_idx: int, camera_downscale: float = 1.0):
        idx = int(np.clip(image_idx, 0, len(self._prepared_frames) - 1))
        frame_data = self._prepared_frames[idx]
        image_infos = dict(frame_data.get("image_infos", {}))
        cam_infos = dict(frame_data.get("cam_infos", {}))

        # Ensure required fields are present
        if "pixels" not in image_infos and self._source_image_set is not None and len(self._source_image_set) > 0:
            source_idx = min(idx, len(self._source_image_set) - 1)
            source_infos, _ = self._source_image_set.get_image(
                source_idx, camera_downscale=camera_downscale
            )
            if "pixels" in source_infos:
                image_infos["pixels"] = source_infos["pixels"]

        return image_infos, cam_infos


def _extract_data_section(cfg) -> Dict:
    content = getattr(cfg, "_content", {})
    if hasattr(content, "get"):
        return content.get("data") or {}
    try:
        return dict(cfg)
    except Exception:
        return {}


def _build_scene_dataset(
    scene_path: Optional[str], config_path: Optional[str], device: torch.device
) -> Optional[DrivingDataset]:
    if not scene_path:
        return None

    scene_path = os.path.abspath(scene_path)
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene path not found: {scene_path}")

    base_config = OmegaConf.create(
        {
            "data_root": os.path.dirname(scene_path),
            "scene_idx": os.path.basename(scene_path),
            "start_timestep": 0,
            "end_timestep": -1,
            "preload_device": device.type,
        }
    )

    if config_path and os.path.exists(config_path):
        logger.info(f"Loading dataset configuration from: {config_path}")
        raw_cfg = OmegaConf.load(config_path)
        data_section = _extract_data_section(raw_cfg)
        data_dict = {k: v for k, v in data_section.items() if v is not None}
        for key, value in base_config.items():
            data_dict[key] = value
        data_cfg = OmegaConf.create(data_dict)
    else:
        logger.warning(
            "No config_path supplied; falling back to the minimal dataset config."
        )
        data_cfg = OmegaConf.merge(
            base_config,
            {
                "dataset": "paralane/1cams",
            },
        )

    dataset = DrivingDataset(data_cfg=data_cfg)
    if hasattr(dataset, "lidar_source") and dataset.lidar_source is not None:
        dataset.lidar_source.to(device)
    return dataset


def _load_trajectory_from_scene_dataset(
    scene_dataset: DrivingDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    if scene_dataset is None or not hasattr(scene_dataset, "full_image_set") or scene_dataset.full_image_set is None:
        raise ValueError("Scene dataset is required to derive the original trajectory.")

    image_set = scene_dataset.full_image_set
    num_frames = len(image_set)
    if num_frames <= 0:
        raise ValueError("Scene dataset does not contain any frames.")

    poses = []
    intrinsics = []
    heights = []
    widths = []

    for idx in range(num_frames):
        image_infos, cam_infos = image_set.get_image(idx, camera_downscale=1.0)
        c2w = cam_infos.get("camera_to_world")
        K = cam_infos.get("intrinsics")
        if c2w is None or K is None:
            raise ValueError(f"Missing camera pose or intrinsics for scene frame {idx}.")

        if torch.is_tensor(c2w):
            c2w_np = c2w.detach().cpu().numpy()
        else:
            c2w_np = np.asarray(c2w)
        if torch.is_tensor(K):
            K_np = K.detach().cpu().numpy()
        else:
            K_np = np.asarray(K)

        pixels = image_infos.get("pixels")
        if torch.is_tensor(pixels):
            h, w = pixels.shape[:2]
        elif isinstance(pixels, np.ndarray):
            h, w = pixels.shape[:2]
        else:
            height_val = cam_infos.get("height")
            width_val = cam_infos.get("width")
            if height_val is None or width_val is None:
                raise ValueError(f"Missing image size for scene frame {idx}.")
            h = int(height_val.item() if torch.is_tensor(height_val) else height_val)
            w = int(width_val.item() if torch.is_tensor(width_val) else width_val)

        poses.append(np.asarray(c2w_np, dtype=np.float32))
        intrinsics.append(np.asarray(K_np, dtype=np.float32))
        heights.append(int(h))
        widths.append(int(w))

    return (
        np.asarray(poses, dtype=np.float32),
        np.asarray(intrinsics, dtype=np.float32),
        np.asarray(heights, dtype=np.int32),
        np.asarray(widths, dtype=np.int32),
        None,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render an exported OBJ road mesh from a trajectory file."
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        default=None,
        help="Path to the scene data directory used to load the source dataset.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the dataset config file used with scene_path.",
    )
    parser.add_argument(
        "--mesh_obj",
        type=str,
        required=True,
        help="Path to the exported OBJ mesh (e.g. road_textured_mesh.obj).",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Path to trajectory .npz/.npy file with camera poses (optional; falls back to the scene trajectory).",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Output mp4 path (default: next to mesh obj).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Limit number of frames rendered.",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="Optional directory to keep rendered PNG frames.",
    )
    return parser.parse_args()


def _load_mesh_as_tensors(mesh_obj_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_obj_path, enable_post_processing=True)
    verts_np = np.asarray(mesh.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh.triangles, dtype=np.int64)
    if verts_np.size == 0 or faces_np.size == 0:
        raise ValueError(f"Loaded mesh is empty: {mesh_obj_path}")
    return torch.from_numpy(verts_np), torch.from_numpy(faces_np)


def _normalize_intrinsics(
    traj_data: Dict[str, np.ndarray], num_poses: int, widths: np.ndarray, heights: np.ndarray
) -> np.ndarray:
    for key in ("camera_intrinsics", "intrinsics", "K"):
        if key in traj_data:
            intr = np.asarray(traj_data[key], dtype=np.float32)
            if intr.ndim == 2 and intr.shape == (3, 3):
                intr = np.repeat(intr[None, ...], num_poses, axis=0)
            if intr.ndim == 3 and intr.shape[0] == num_poses and intr.shape[-2:] == (3, 3):
                return intr
    intr = np.zeros((num_poses, 3, 3), dtype=np.float32)
    intr[:, 0, 0] = widths.astype(np.float32)
    intr[:, 1, 1] = heights.astype(np.float32)
    intr[:, 0, 2] = widths.astype(np.float32) * 0.5
    intr[:, 1, 2] = heights.astype(np.float32) * 0.5
    intr[:, 2, 2] = 1.0
    logger.warning("No intrinsics found in trajectory file; using fallback intrinsics.")
    return intr


def _normalize_sizes(
    traj_data: Dict[str, np.ndarray], num_poses: int
) -> Tuple[np.ndarray, np.ndarray]:
    widths = np.asarray(traj_data.get("widths", np.full(num_poses, 1280)), dtype=np.int32).reshape(-1)
    heights = np.asarray(traj_data.get("heights", np.full(num_poses, 720)), dtype=np.int32).reshape(-1)
    if len(widths) == 1:
        widths = np.full(num_poses, int(widths[0]), dtype=np.int32)
    if len(heights) == 1:
        heights = np.full(num_poses, int(heights[0]), dtype=np.int32)
    if len(widths) != num_poses or len(heights) != num_poses:
        raise ValueError(
            f"widths/heights length must match number of poses ({num_poses}), got {len(widths)}/{len(heights)}"
        )
    return widths, heights


def _load_trajectory(trajectory_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ext = Path(trajectory_path).suffix.lower()
    if ext not in {".npz", ".npy"}:
        raise ValueError(f"Unsupported trajectory extension: {ext}")

    if ext == ".npy":
        poses = np.asarray(np.load(trajectory_path, allow_pickle=True), dtype=np.float32)
        if poses.ndim != 3 or poses.shape[-2:] not in {(3, 4), (4, 4)}:
            raise ValueError(f"Expected [N,3,4] or [N,4,4] in {trajectory_path}, got {poses.shape}")
        if poses.shape[-2:] == (3, 4):
            bottom = np.zeros((poses.shape[0], 1, 4), dtype=poses.dtype)
            bottom[:, 0, 3] = 1.0
            poses = np.concatenate([poses, bottom], axis=1)
        widths = np.full(len(poses), 1280, dtype=np.int32)
        heights = np.full(len(poses), 720, dtype=np.int32)
        intrinsics = _normalize_intrinsics({}, len(poses), widths, heights)
        return poses, intrinsics, heights, widths

    data = np.load(trajectory_path, allow_pickle=True)
    keys = ("camera_poses", "poses", "trajectory")
    pose_key = next((k for k in keys if k in data), None)
    if pose_key is None:
        raise ValueError(f"No pose key found in {trajectory_path}. Expected one of {keys}")
    poses = np.asarray(data[pose_key], dtype=np.float32)
    if poses.ndim != 3:
        raise ValueError(f"Trajectory must be [N,4,4] or [N,3,4], got {poses.shape}")
    if poses.shape[-2:] == (3, 4):
        bottom = np.zeros((poses.shape[0], 1, 4), dtype=poses.dtype)
        bottom[:, 0, 3] = 1.0
        poses = np.concatenate([poses, bottom], axis=1)
    if poses.shape[-2:] != (4, 4):
        raise ValueError(f"Trajectory poses must be [N,4,4], got {poses.shape}")

    traj_data: Dict[str, np.ndarray] = {k: np.asarray(data[k]) for k in data.files}

    def _filter_pose_aligned_fields(fields: Dict[str, np.ndarray], keep_mask: np.ndarray) -> Dict[str, np.ndarray]:
        keep_len = int(keep_mask.shape[0])
        filtered: Dict[str, np.ndarray] = {}
        for key, value in fields.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == keep_len:
                filtered[key] = value[keep_mask]
            else:
                filtered[key] = value
        return filtered

    cam_ids_arr = traj_data.get("cam_ids")
    cam_names_arr = traj_data.get("cam_names")

    if (
        isinstance(cam_ids_arr, np.ndarray)
        and cam_ids_arr.ndim > 0
        and cam_ids_arr.shape[0] == len(poses)
    ):
        cam_ids = np.asarray(cam_ids_arr).reshape(-1)
        keep = cam_ids == cam_ids[0]
        poses = poses[keep]
        traj_data = _filter_pose_aligned_fields(traj_data, keep)
    elif (
        isinstance(cam_names_arr, np.ndarray)
        and cam_names_arr.ndim > 0
        and cam_names_arr.shape[0] == len(poses)
    ):
        cam_names = np.asarray(cam_names_arr).reshape(-1)
        keep = cam_names == cam_names[0]
        poses = poses[keep]
        traj_data = _filter_pose_aligned_fields(traj_data, keep)

    widths, heights = _normalize_sizes(traj_data, len(poses))
    intrinsics = _normalize_intrinsics(traj_data, len(poses), widths, heights)

    source_npz = traj_data.get("source_npz")
    if (
        intrinsics is not None
        and ("camera_intrinsics" not in traj_data or "widths" not in traj_data or "heights" not in traj_data)
        and isinstance(source_npz, np.ndarray)
        and source_npz.ndim == 0
    ):
        source_path = str(source_npz.item())
        if os.path.isfile(source_path):
            source_data = np.load(source_path, allow_pickle=True)
            if "camera_intrinsics" in source_data:
                source_intr = np.asarray(source_data["camera_intrinsics"], dtype=np.float32)
                if source_intr.ndim == 2 and source_intr.shape == (3, 3):
                    source_intr = np.repeat(source_intr[None, ...], len(poses), axis=0)
                if source_intr.ndim == 3 and source_intr.shape[0] == len(poses):
                    intrinsics = source_intr
                    logger.info(f"Loaded camera intrinsics from source bundle: {source_path}")
            if "widths" in source_data and "heights" in source_data:
                widths = np.asarray(source_data["widths"], dtype=np.int32).reshape(-1)
                heights = np.asarray(source_data["heights"], dtype=np.int32).reshape(-1)
                if len(widths) == len(poses) and len(heights) == len(poses):
                    logger.info(f"Loaded frame sizes from source bundle: {source_path}")

    return poses, intrinsics, heights, widths


def _load_texture_npz(texture_npz_path: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    if not os.path.isfile(texture_npz_path):
        return None, None
    data = np.load(texture_npz_path, allow_pickle=True)
    if "buffer" not in data:
        return None, None
    metadata = {
        "x_range": tuple(np.asarray(data["x_range"]).tolist()) if "x_range" in data else None,
        "y_range": tuple(np.asarray(data["y_range"]).tolist()) if "y_range" in data else None,
        "width": int(np.asarray(data["width"]).item()) if "width" in data else None,
        "height": int(np.asarray(data["height"]).item()) if "height" in data else None,
        "pixels_per_meter": float(np.asarray(data["pixels_per_meter"]).item())
        if "pixels_per_meter" in data
        else None,
    }
    if metadata["x_range"] is None or metadata["y_range"] is None:
        return None, None
    return np.asarray(data["buffer"], dtype=np.float32), metadata


def _frame_visibility_stats(frame_bgr: np.ndarray) -> Tuple[bool, float, int]:
    """Return a simple visibility signal for a rendered frame."""
    if frame_bgr is None or frame_bgr.size == 0:
        return False, 0.0, 0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    nonzero_pixels = int(np.count_nonzero(gray > 5))
    mean_intensity = float(gray.mean())
    visible = nonzero_pixels > 0 and mean_intensity > 1.0
    return visible, mean_intensity, nonzero_pixels


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

    if args.trajectory:
        trajectory = os.path.abspath(args.trajectory)
        if not os.path.isfile(trajectory):
            raise FileNotFoundError(f"Trajectory file not found: {trajectory}")
        poses, intrinsics, heights, widths = _load_trajectory(trajectory)
        if args.max_frames is not None:
            max_frames = max(int(args.max_frames), 0)
            poses = poses[:max_frames]
            intrinsics = intrinsics[:max_frames]
            heights = heights[:max_frames]
            widths = widths[:max_frames]
        if len(poses) == 0:
            raise ValueError("No poses to render after applying frame limits.")

        # Use dataset.prepare_novel_view_render_data() for consistent pose handling with eval.py
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
                    f"Falling back to direct _TrajectoryImageSet."
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
            # Fallback when no scene_dataset available
            logger.warning(
                "No scene dataset provided; using raw trajectory data without PixelSource processing. "
                "For consistent pose handling, provide --scene_path."
            )
            source_image_set = None
            image_set = _TrajectoryImageSet(
                poses,
                intrinsics,
                heights,
                widths,
                source_image_set=source_image_set,
            )
            render_dataset = _TrajectoryDataset(image_set)
    else:
        if scene_dataset is None:
            raise ValueError(
                "Provide --scene_path so the original scene trajectory can be used."
            )
        logger.info("No trajectory file provided; using the scene's original trajectory.")
        render_dataset = scene_dataset
        if args.max_frames is not None:
            max_frames = max(int(args.max_frames), 0)
            if hasattr(render_dataset, "train_indices"):
                render_dataset.train_indices = render_dataset.train_indices[:max_frames]
        if len(render_dataset.train_indices) == 0:
            raise ValueError("No poses to render after applying frame limits.")

    # Prefer PNG texture over NPZ buffer
    png_candidates = [
        os.path.join(os.path.dirname(mesh_obj), "road_textured_mesh.png"),
        os.path.join(os.path.dirname(mesh_obj), "road_textured_mesh_texture.png"),
    ]
    npz_path = os.path.join(os.path.dirname(mesh_obj), "image_buffer.npz")
    texture_buffer, texture_metadata = None, None
    
    # Load metadata if present
    if os.path.isfile(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        if "x_range" in d and "y_range" in d:
            texture_metadata = {
                "x_range": tuple(np.asarray(d["x_range"]).tolist()),
                "y_range": tuple(np.asarray(d["y_range"]).tolist()),
                "width": int(d["width"]) if "width" in d else None,
                "height": int(d["height"]) if "height" in d else None,
                "pixels_per_meter": float(d["pixels_per_meter"]) if "pixels_per_meter" in d else None,
            }
    
    # Prefer PNG image as texture
    for png_path in png_candidates:
        if os.path.isfile(png_path):
            img = np.asarray(Image.open(png_path).convert("RGB")).astype(np.float32) / 255.0
            texture_buffer = img
            if texture_metadata is None:
                # Fallback metadata: use mesh bbox
                verts_np = (
                    vertices_tensor.cpu().numpy()
                    if torch.is_tensor(vertices_tensor)
                    else np.asarray(vertices_tensor)
                )
                texture_metadata = {
                    "x_range": (float(verts_np[:, 0].min()), float(verts_np[:, 0].max())),
                    "y_range": (float(verts_np[:, 1].min()), float(verts_np[:, 1].max())),
                    "width": img.shape[1],
                    "height": img.shape[0],
                    "pixels_per_meter": None,
                }
            logger.info(f"Using PNG texture {png_path}")
            break
    if texture_buffer is None:
        texture_buffer, texture_metadata = _load_texture_npz(npz_path)
        if texture_buffer is not None:
            logger.info("Loaded texture buffer from image_buffer.npz")

    # Diagnostic: mesh bbox and camera info
    verts_np = vertices_tensor.cpu().numpy() if torch.is_tensor(vertices_tensor) else np.asarray(vertices_tensor)
    bbox_min = verts_np.min(axis=0)
    bbox_max = verts_np.max(axis=0)
    mesh_center = (bbox_min + bbox_max) * 0.5
    logger.info(f"Mesh bbox min:{bbox_min}, max:{bbox_max}, center:{mesh_center}")
    if args.trajectory:
        logger.info(f"Intrinsics shape: {intrinsics.shape}, poses shape: {poses.shape}")
    else:
        logger.info(
            f"Using scene dataset trajectory with {len(render_dataset.train_indices)} frames"
        )
    
    # Check if mesh is in front of first camera
    c2w0 = (
        render_dataset.full_image_set.get_image(0, camera_downscale=1.0)[1][
            "camera_to_world"
        ]
        if not args.trajectory
        else poses[0]
    )
    c2w0 = c2w0.detach().cpu().numpy() if torch.is_tensor(c2w0) else np.asarray(c2w0)
    w2c0 = np.linalg.inv(c2w0)
    center_cam = (w2c0[:3,:3] @ mesh_center) + w2c0[:3,3]
    visible_frac = float(np.mean((w2c0[:3, :3] @ verts_np.T + w2c0[:3, 3:4])[2, :] > 0))
    logger.info(f"Frame0 camera pos: {c2w0[:3,3]}, mesh_center_cam_z: {center_cam[2]:.3f}, verts_in_front_frac: {visible_frac:.3f}")


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