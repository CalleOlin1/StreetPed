#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SCENE_PATH = "/home/hstromgr/Documents/Github/StreetPed/3D-Reconstruction-main/data/paralane/processed/scene_002_clip_002/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project LiDAR points into an RGB image and overlay them with depth-based coloring."
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        default=DEFAULT_SCENE_PATH,
        help="Path to the processed scene directory.",
    )
    parser.add_argument(
        "--image_index",
        type=int,
        default=0,
        help="Frame index to load (0-based; e.g. 0 => images/000_0.jpg).",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera index to use within the scene (default: 0).",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=80.0,
        help="Maximum depth in meters for points to overlay.",
    )
    parser.add_argument(
        "--point_radius",
        type=int,
        default=2,
        help="Radius of projected LiDAR points in pixels.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Alpha blending factor for the overlay.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output filename. If omitted, writes to scene_path/overlays/overlay_<frame>_<camera>.png",
    )
    return parser.parse_args()


def find_frame_image(scene_path: Path, frame_index: int, camera_index: int = 0) -> Path:
    images_dir = scene_path / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{frame_index:03d}_{camera_index}{suffix}"
        if candidate.exists():
            return candidate

    matches = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = path.stem
        if "_" not in stem:
            continue
        prefix, suffix = stem.rsplit("_", 1)
        if prefix.isdigit() and suffix.isdigit():
            if int(prefix) == frame_index and int(suffix) == camera_index:
                matches.append(path)
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No image file found for frame {frame_index} and camera {camera_index} under {images_dir}."
    )


def find_lidar_scan(scene_path: Path, frame_index: int) -> Path:
    lidar_dir = scene_path / "lidar"
    if not lidar_dir.exists():
        raise FileNotFoundError(f"LiDAR directory not found: {lidar_dir}")

    candidate = lidar_dir / f"{frame_index:03d}.bin"
    if candidate.exists():
        return candidate

    for path in sorted(lidar_dir.iterdir()):
        if path.suffix.lower() == ".bin" and path.stem.isdigit() and int(path.stem) == frame_index:
            return path
    raise FileNotFoundError(f"No LiDAR scan found for frame {frame_index} in {lidar_dir}.")


def load_intrinsics(scene_path: Path, camera_index: int) -> np.ndarray:
    intrinsics_file = scene_path / "intrinsics" / f"{camera_index}.txt"
    if not intrinsics_file.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_file}")

    values = np.loadtxt(intrinsics_file, dtype=np.float64)
    if values.size >= 4 and values.shape != (3, 3):
        if values.shape == (9,):
            fx, fy, cx, cy = values[0], values[1], values[2], values[3]
        elif values.shape == (4,):
            fx, fy, cx, cy = values
        else:
            raise ValueError(f"Unsupported intrinsics format in {intrinsics_file}: {values.shape}")
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    K = np.asarray(values, dtype=np.float64)
    if K.shape == (3, 3):
        return K
    raise ValueError(f"Unsupported intrinsic matrix shape {K.shape} in {intrinsics_file}.")


def load_extrinsics(scene_path: Path, camera_index: int) -> np.ndarray:
    extrinsics_file = scene_path / "extrinsics" / f"{camera_index}.txt"
    if not extrinsics_file.exists():
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsics_file}")

    T = np.loadtxt(extrinsics_file, dtype=np.float64)
    T = np.asarray(T)
    if T.shape != (4, 4):
        raise ValueError(f"Extrinsics matrix at {extrinsics_file} is not 4x4: {T.shape}")
    return T


def load_lidar_points(scan_path: Path) -> np.ndarray:
    arr = np.fromfile(scan_path, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if arr.size % 4 == 0:
        points = arr.reshape(-1, 4)
        return points[:, :3]
    if arr.size % 3 == 0:
        return arr.reshape(-1, 3)
    raise ValueError(f"Unexpected LiDAR file shape in {scan_path}: {arr.shape}")


def project_lidar_to_camera(pts: np.ndarray, K: np.ndarray, T_ego_camera: np.ndarray):
    """Project ego-frame LiDAR points into the camera using T_camera<-ego = inv(T_ego<-camera)."""
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ego_pts_h = np.hstack([pts, ones])
    T_camera_ego = np.linalg.inv(T_ego_camera)
    cam_pts_h = (T_camera_ego @ ego_pts_h.T).T

    xyz = cam_pts_h[:, :3]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 1e-3)
    x = x[valid]
    y = y[valid]
    z = z[valid]

    u = (x / z) * K[0, 0] + K[0, 2]
    v = (y / z) * K[1, 1] + K[1, 2]
    depth = z
    return u, v, depth


def depth_to_color(depth: np.ndarray, max_depth: float) -> np.ndarray:
    if depth.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    norm = np.clip(depth / max_depth, 0.0, 1.0)
    norm = 1.0 - norm
    scalar_values = np.uint8(255.0 * norm)
    colorized = cv2.applyColorMap(scalar_values.reshape(-1, 1), cv2.COLORMAP_JET)
    colors = colorized[:, 0, :].astype(np.uint8)
    return colors


def overlay_lidar_on_image(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    point_radius: int,
    alpha: float,
    max_depth: float,
):
    if len(u) == 0:
        return image

    height, width = image.shape[:2]
    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
        & (depth > 0)
        & (depth <= max_depth)
    )

    u = u[valid].astype(np.int32)
    v = v[valid].astype(np.int32)
    depth = depth[valid]
    if len(u) == 0:
        return image

    colors = depth_to_color(depth, max_depth)
    overlay = image.copy()
    for (x, y), color in zip(zip(u, v), colors):
        cv2.circle(overlay, (int(x), int(y)), point_radius, tuple(color.tolist()), thickness=-1)

    blended = cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)
    return blended


def main():
    args = parse_args()
    scene_path = Path(args.scene_path).expanduser().resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene path does not exist: {scene_path}")

    image_path = find_frame_image(scene_path, args.image_index, args.camera_index)
    scan_path = find_lidar_scan(scene_path, args.image_index)
    K = load_intrinsics(scene_path, args.camera_index)
    T_ego_camera = load_extrinsics(scene_path, args.camera_index)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    lidar_pts = load_lidar_points(scan_path)

    u, v, depth = project_lidar_to_camera(lidar_pts, K, T_ego_camera)
    overlay = overlay_lidar_on_image(
        image,
        u,
        v,
        depth,
        point_radius=args.point_radius,
        alpha=args.alpha,
        max_depth=args.max_depth,
    )

    if args.output_path is not None:
        output_path = Path(args.output_path).expanduser().resolve()
    else:
        output_dir = scene_path / "overlays"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"overlay_{args.image_index:03d}_cam_{args.camera_index}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    print(f"Saved overlay to {output_path}")
    print(f"Image: {image_path}")
    print(f"LiDAR: {scan_path}")
    print(f"Projected points: {len(u[np.isfinite(u)])}")


if __name__ == "__main__":
    main()
