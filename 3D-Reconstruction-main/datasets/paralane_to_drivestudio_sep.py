#!/usr/bin/env python3
"""
Convert ParaLane dataset to DriveStudio processed format.

This variant keeps ParaLane camera and LiDAR poses in their original shared world frame:

  - Camera world pose T_world<-camera is taken directly from COLMAP images.txt.
  - Ego/LiDAR world pose T_world<-ego is taken directly from lidar_poses.txt.
  - No Umeyama/ICP/similarity/global alignment is applied.
  - LiDAR point clouds are kept in LiDAR/ego coordinates.

If DriveStudio requires a fixed camera extrinsic, we derive T_ego<-camera from matched frames:

  T_world<-camera = T_world<-ego * T_ego<-camera
  => T_ego<-camera = inv(T_world<-ego) * T_world<-camera

We estimate one fixed T_ego<-camera by averaging per-frame candidates from matched timestamps.
"""

import argparse
import bisect
import json
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_colmap_cameras(cameras_txt):
    """Parse COLMAP cameras.txt into {camera_id: {model,width,height,params}}."""
    cameras = {}
    with open(cameras_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            cameras[cam_id] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(v) for v in parts[4:]],
            }
    return cameras


def colmap_camera_to_K(camera):
    """Convert COLMAP camera parameters to 3x3 intrinsics matrix K."""
    model = camera["model"]
    params = camera["params"]

    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model in (
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "RADIAL",
        "RADIAL_FISHEYE",
        "FOV",
    ):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        print(f"  Warning: Unknown camera model '{model}', falling back to fx,fy,cx,cy from first four params.")
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def parse_colmap_images(images_txt):
    """
    Parse COLMAP images.txt.

    Returns sorted list of dictionaries with fields:
      image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name
    """
    images = []
    with open(images_txt, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 10:
            images.append(
                {
                    "image_id": int(parts[0]),
                    "qw": float(parts[1]),
                    "qx": float(parts[2]),
                    "qy": float(parts[3]),
                    "qz": float(parts[4]),
                    "tx": float(parts[5]),
                    "ty": float(parts[6]),
                    "tz": float(parts[7]),
                    "camera_id": int(parts[8]),
                    "name": parts[9],
                }
            )
            i += 2  # second line is 2D-3D correspondences
        else:
            i += 1

    images.sort(key=lambda item: item["name"])
    return images


def colmap_w2c_to_c2w(qw, qx, qy, qz, tx, ty, tz):
    """
    Convert COLMAP world-to-camera to camera-to-world transform.
    COLMAP quaternion order is (qw, qx, qy, qz).
    """
    R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    t_w2c = np.array([tx, ty, tz], dtype=np.float64)

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c
    return np.linalg.inv(w2c)


def parse_lidar_poses(lidar_poses_txt):
    """
    Parse lidar_poses.txt.

    Supported formats:
      A: Qx Qy Qz Qw X Y Z
      B: filename.ply Qx Qy Qz Qw X Y Z

    Returns:
      poses: list[4x4]
      pose_map: {stem: 4x4}
      pose_time_map: {timestamp_int: 4x4}
    """
    poses = []
    pose_map = {}
    pose_time_map = {}
    with open(lidar_poses_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            filename = None
            try:
                float(parts[0])
                float_parts = parts
            except ValueError:
                filename = parts[0]
                float_parts = parts[1:]

            if len(float_parts) < 7:
                continue

            qx, qy, qz, qw, x, y, z = [float(v) for v in float_parts[:7]]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = R
            mat[:3, 3] = [x, y, z]
            poses.append(mat)

            if filename:
                stem = Path(filename).stem
                pose_map[stem] = mat
                ts = parse_numeric_timestamp(stem)
                if ts is not None:
                    pose_time_map[ts] = mat

    return poses, pose_map, pose_time_map


def parse_numeric_timestamp(token):
    """Parse integer timestamp tokens like '1718350987399622'; return None for non-numeric."""
    if token is None:
        return None
    token = str(token).strip()
    if token.isdigit():
        return int(token)
    return None


def get_image_timestamp_token(colmap_image_name, src_img_path):
    """
    Extract camera timestamp token from COLMAP image name, then fallback to parent directory name.
    """
    if colmap_image_name and "/" in colmap_image_name:
        return colmap_image_name.split("/")[0]
    return src_img_path.parent.name


def build_pose_time_index(pose_time_map):
    """Build sorted lidar timestamp index for nearest-neighbor matching."""
    if not pose_time_map:
        return [], []
    sorted_times = sorted(pose_time_map.keys())
    sorted_poses = [pose_time_map[t] for t in sorted_times]
    return sorted_times, sorted_poses


def compute_max_timestamp_delta(sorted_times):
    """
    Compute an adaptive max camera-lidar timestamp gap based on lidar cadence.
    """
    if len(sorted_times) < 2:
        return None
    diffs = np.diff(np.asarray(sorted_times, dtype=np.int64))
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return None
    median_step = int(np.median(positive_diffs))
    return max(3 * median_step, 1)


def get_nearest_pose_by_timestamp(target_ts, sorted_times, sorted_poses, max_delta):
    """Get nearest lidar pose by timestamp; return None if outside max_delta."""
    if target_ts is None or not sorted_times:
        return None
    idx = bisect.bisect_left(sorted_times, target_ts)
    candidates = []
    if idx < len(sorted_times):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None
    best_idx = min(candidates, key=lambda i: abs(sorted_times[i] - target_ts))
    best_delta = abs(sorted_times[best_idx] - target_ts)
    if max_delta is not None and best_delta > max_delta:
        return None
    return sorted_poses[best_idx]


def _read_ply_fallback(ply_path):
    """Fallback PLY reader that supports ASCII and binary vertex point clouds."""
    type_to_struct = {
        "char": "b",
        "uchar": "B",
        "short": "h",
        "ushort": "H",
        "int": "i",
        "uint": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }

    with open(ply_path, "rb") as f:
        fmt = None
        num_vertices = 0
        in_vertex = False
        vertex_props = []

        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Invalid PLY header: {ply_path}")
            line = raw.decode("ascii", errors="replace").strip()

            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    num_vertices = int(parts[2])
            elif line.startswith("property ") and in_vertex:
                parts = line.split()
                if len(parts) == 3:
                    ptype, pname = parts[1], parts[2]
                    if ptype not in type_to_struct:
                        raise ValueError(f"Unsupported PLY type '{ptype}' in {ply_path}")
                    vertex_props.append((pname, ptype))
                elif len(parts) >= 5 and parts[1] == "list":
                    raise ValueError(f"Unsupported list vertex property in {ply_path}")
            elif line == "end_header":
                break

        if fmt is None:
            raise ValueError(f"Missing PLY format declaration in {ply_path}")
        if num_vertices <= 0:
            return np.zeros((0, 4), dtype=np.float32)

        prop_names = [name for name, _ in vertex_props]

        def find_idx(candidates):
            for candidate in candidates:
                if candidate in prop_names:
                    return prop_names.index(candidate)
            return -1

        ix = find_idx(["x"])
        iy = find_idx(["y"])
        iz = find_idx(["z"])
        ii = find_idx(["intensity", "i", "scalar_intensity", "Intensity"])

        if min(ix, iy, iz) < 0:
            raise ValueError(f"PLY missing x/y/z in {ply_path}")

        pts = np.zeros((num_vertices, 4), dtype=np.float32)

        if fmt == "ascii":
            for row in range(num_vertices):
                line = f.readline().decode("ascii", errors="replace").strip()
                if not line:
                    continue
                vals = line.split()
                pts[row, 0] = float(vals[ix])
                pts[row, 1] = float(vals[iy])
                pts[row, 2] = float(vals[iz])
                pts[row, 3] = float(vals[ii]) if ii >= 0 and ii < len(vals) else 0.0
            return pts

        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format '{fmt}' in {ply_path}")

        endian = "<" if fmt == "binary_little_endian" else ">"
        struct_fmt = endian + "".join(type_to_struct[t] for _, t in vertex_props)
        stride = struct.calcsize(struct_fmt)

        for row in range(num_vertices):
            chunk = f.read(stride)
            if len(chunk) != stride:
                raise ValueError(f"Unexpected EOF while reading {ply_path}, row {row}")
            vals = struct.unpack(struct_fmt, chunk)
            pts[row, 0] = float(vals[ix])
            pts[row, 1] = float(vals[iy])
            pts[row, 2] = float(vals[iz])
            pts[row, 3] = float(vals[ii]) if ii >= 0 else 0.0

        return pts


def ply_to_bin(ply_path):
    """Load PLY points as float32 Nx4 [x,y,z,intensity]."""
    try:
        from plyfile import PlyData

        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]
        x = np.asarray(vertex["x"], dtype=np.float32)
        y = np.asarray(vertex["y"], dtype=np.float32)
        z = np.asarray(vertex["z"], dtype=np.float32)

        intensity = None
        for field in ("intensity", "i", "scalar_intensity", "Intensity"):
            if field in vertex.data.dtype.names:
                intensity = np.asarray(vertex[field], dtype=np.float32)
                break
        if intensity is None:
            intensity = np.zeros_like(x)

        return np.stack([x, y, z, intensity], axis=1)
    except Exception:
        return _read_ply_fallback(ply_path)


def average_rotation_matrices(rotations):
    """Average rotation matrices and project result back to SO(3)."""
    if not rotations:
        return np.eye(3, dtype=np.float64)
    M = np.zeros((3, 3), dtype=np.float64)
    for R in rotations:
        M += R
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def find_masks_dir(clip_path):
    """Find ParaLane mask directory among common naming variants."""
    for name in ("foreground_mask", "foreground_masks", "foreground_labels", "foreground_label"):
        candidate = clip_path / name
        if candidate.exists():
            return candidate
    return clip_path / "foreground_mask"


def resolve_image_path(images_dir, timestamp_dirs, img_name, timestep_idx):
    """Resolve source image path from COLMAP image name with fallbacks."""
    direct = images_dir / img_name
    if direct.exists():
        return direct

    ts_name = img_name.split("/")[0] if "/" in img_name else None
    if ts_name:
        candidate = images_dir / ts_name / "CAMERA_FRONT.png"
        if candidate.exists():
            return candidate

    if timestep_idx < len(timestamp_dirs):
        candidate = timestamp_dirs[timestep_idx] / "CAMERA_FRONT.png"
        if candidate.exists():
            return candidate

    return None


def resolve_mask_path(masks_dir, ts_part, timestep_idx):
    """Resolve foreground mask for one frame with robust fallbacks."""
    if not masks_dir.exists():
        return None

    for candidate in (
        masks_dir / ts_part / "CAMERA_FRONT.png",
        masks_dir / ts_part / "CAMERA_FRONT.jpg",
        masks_dir / ts_part / "mask.png",
        masks_dir / ts_part / "foreground.png",
    ):
        if candidate.exists():
            return candidate

    mask_dirs = sorted([d for d in masks_dir.iterdir() if d.is_dir()])
    if timestep_idx < len(mask_dirs):
        for candidate in (
            mask_dirs[timestep_idx] / "CAMERA_FRONT.png",
            mask_dirs[timestep_idx] / "CAMERA_FRONT.jpg",
            mask_dirs[timestep_idx] / "mask.png",
            mask_dirs[timestep_idx] / "foreground.png",
        ):
            if candidate.exists():
                return candidate

    for candidate in sorted(masks_dir.glob("*")):
        if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            lower_name = candidate.stem.lower()
            if candidate.name.startswith("CAMERA_FRONT") or "mask" in lower_name or "foreground" in lower_name:
                return candidate

    return None


def derive_fixed_cam_to_ego(cam_world_poses, ego_world_poses):
    """
    Derive fixed T_ego<-camera from matched world-frame trajectories without alignment.

    Each candidate is:
      T_ego<-camera_i = inv(T_world<-ego_i) @ T_world<-camera_i
    """
    if len(cam_world_poses) != len(ego_world_poses):
        raise ValueError("cam_world_poses and ego_world_poses must have equal length")
    if not cam_world_poses:
        return None

    candidates = []
    for cam_pose, ego_pose in zip(cam_world_poses, ego_world_poses):
        candidates.append(np.linalg.inv(ego_pose) @ cam_pose)

    R_avg = average_rotation_matrices([c[:3, :3] for c in candidates])
    t_stack = np.stack([c[:3, 3] for c in candidates], axis=0)
    t_med = np.median(t_stack, axis=0)

    cam_to_ego = np.eye(4, dtype=np.float64)
    cam_to_ego[:3, :3] = R_avg
    cam_to_ego[:3, 3] = t_med

    # Report consistency to catch bad pairing/mismatch issues.
    rot_deltas_deg = []
    trans_deltas = []
    for c in candidates:
        R_delta = c[:3, :3] @ R_avg.T
        angle = Rotation.from_matrix(R_delta).magnitude() * 180.0 / np.pi
        rot_deltas_deg.append(float(angle))
        trans_deltas.append(float(np.linalg.norm(c[:3, 3] - t_med)))
    stats = {
        "num_candidates": len(candidates),
        "median_rot_delta_deg": float(np.median(rot_deltas_deg)),
        "max_rot_delta_deg": float(np.max(rot_deltas_deg)),
        "median_trans_delta_m": float(np.median(trans_deltas)),
        "max_trans_delta_m": float(np.max(trans_deltas)),
    }
    return cam_to_ego, stats


def convert_clip(paralane_clip_dir, output_scene_dir, verbose=True):
    """Convert one ParaLane clip to DriveStudio format."""
    clip_path = Path(paralane_clip_dir)
    out_path = Path(output_scene_dir)
    sparse_dir = clip_path / "sparse" / "0"
    images_dir = clip_path / "images"
    masks_dir = find_masks_dir(clip_path)
    lidars_dir = clip_path / "lidars"

    if not clip_path.exists():
        raise FileNotFoundError(f"Clip directory not found: {clip_path}")
    if not sparse_dir.exists():
        raise FileNotFoundError(f"sparse/0 directory not found: {sparse_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")

    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"
    lidar_poses_txt = sparse_dir / "lidar_poses.txt"
    for required in (cameras_txt, images_txt):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    for subdir in (
        "images",
        "lidar",
        "ego_pose",
        "extrinsics",
        "intrinsics",
        "instances",
        "humanpose",
        "dynamic_masks/all",
        "dynamic_masks/human",
        "dynamic_masks/vehicle",
        "fine_dynamic_masks/all",
        "fine_dynamic_masks/human",
        "fine_dynamic_masks/vehicle",
        "sky_masks",
    ):
        (out_path / subdir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Parsing COLMAP cameras: {cameras_txt}")
    cameras = parse_colmap_cameras(str(cameras_txt))
    if not cameras:
        raise ValueError(f"No cameras found in {cameras_txt}")

    primary_camera_id = list(cameras.keys())[0]
    camera_info = cameras[primary_camera_id]
    K = colmap_camera_to_K(camera_info)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    intr = np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    np.savetxt(str(out_path / "intrinsics" / "0.txt"), intr, fmt="%.10f")
    np.savetxt(str(out_path / "intrinsics" / "1.txt"), intr, fmt="%.10f")
    if verbose:
        print(f"  Saved intrinsics/0.txt (camera model: {camera_info['model']})")

    # Default extrinsics to identity and overwrite if derivation succeeds.
    identity4 = np.eye(4, dtype=np.float64)
    np.savetxt(str(out_path / "extrinsics" / "0.txt"), identity4, fmt="%.10f")
    np.savetxt(str(out_path / "extrinsics" / "1.txt"), identity4, fmt="%.10f")

    if verbose:
        print(f"  Parsing COLMAP images: {images_txt}")
    colmap_images = parse_colmap_images(str(images_txt))
    if not colmap_images:
        if verbose:
            print(f"  WARNING: No images found in {images_txt}. Skipping clip.")
        return 0

    timestamp_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])

    lidar_poses = []
    lidar_pose_map = {}
    lidar_pose_time_map = {}
    lidar_sorted_times = []
    lidar_sorted_poses = []
    lidar_max_ts_delta = None
    if lidar_poses_txt.exists():
        lidar_poses, lidar_pose_map, lidar_pose_time_map = parse_lidar_poses(str(lidar_poses_txt))
        lidar_sorted_times, lidar_sorted_poses = build_pose_time_index(lidar_pose_time_map)
        lidar_max_ts_delta = compute_max_timestamp_delta(lidar_sorted_times)
        if verbose:
            print(f"  Found {len(lidar_poses)} poses in lidar_poses.txt")
            if lidar_sorted_times:
                print(f"  Using timestamp-nearest camera↔lidar matching (max delta: {lidar_max_ts_delta})")
            else:
                print("  lidar_poses.txt has no filename timestamps; using index-based pairing.")
    elif verbose:
        print("  NOTE: lidar_poses.txt not found; ego_pose will use camera c2w fallback.")

    n_converted = 0
    timestamp_to_timestep = {}
    matched_cam_world = []
    matched_ego_world = []

    for timestep_idx, img_entry in enumerate(colmap_images):
        c2w_cam = colmap_w2c_to_c2w(
            img_entry["qw"],
            img_entry["qx"],
            img_entry["qy"],
            img_entry["qz"],
            img_entry["tx"],
            img_entry["ty"],
            img_entry["tz"],
        )

        src_img_path = resolve_image_path(images_dir, timestamp_dirs, img_entry["name"], timestep_idx)
        if src_img_path is None:
            if verbose:
                print(f"  WARNING: Missing source image for frame {timestep_idx}, name={img_entry['name']}")
            continue

        dst_img = out_path / "images" / f"{timestep_idx:03d}_0.jpg"
        img = Image.open(str(src_img_path)).convert("RGB")
        img.save(str(dst_img), quality=100)

        ts_part = src_img_path.parent.name
        timestamp_to_timestep[ts_part] = timestep_idx

        ego_pose = None
        if lidar_poses_txt.exists():
            image_ts_token = get_image_timestamp_token(img_entry["name"], src_img_path)
            ego_pose = lidar_pose_map.get(image_ts_token, None)

            if ego_pose is None and lidar_sorted_times:
                image_ts_int = parse_numeric_timestamp(image_ts_token)
                ego_pose = get_nearest_pose_by_timestamp(
                    target_ts=image_ts_int,
                    sorted_times=lidar_sorted_times,
                    sorted_poses=lidar_sorted_poses,
                    max_delta=lidar_max_ts_delta,
                )

            if ego_pose is None and not lidar_sorted_times and timestep_idx < len(lidar_poses):
                ego_pose = lidar_poses[timestep_idx]

            if ego_pose is None:
                raise RuntimeError(
                    "Could not match camera frame to any lidar pose while lidar_poses.txt exists. "
                    f"Frame idx={timestep_idx}, image='{img_entry['name']}', resolved timestamp='{image_ts_token}'."
                )

            matched_cam_world.append(c2w_cam)
            matched_ego_world.append(ego_pose)
        else:
            ego_pose = c2w_cam

        np.savetxt(str(out_path / "ego_pose" / f"{timestep_idx:03d}.txt"), ego_pose, fmt="%.10f")

        h, w = img.height, img.width
        prefix = f"{timestep_idx:03d}_0"
        empty_u8 = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
        empty_u16 = Image.fromarray(np.zeros((h, w), dtype=np.uint16))

        src_mask_path = resolve_mask_path(masks_dir, ts_part, timestep_idx)
        if src_mask_path is not None and src_mask_path.exists():
            dyn = Image.open(str(src_mask_path)).convert("L")
            dyn.save(str(out_path / "dynamic_masks" / "all" / f"{prefix}.png"))
            dyn.save(str(out_path / "fine_dynamic_masks" / "all" / f"{prefix}.png"))
        else:
            if verbose and n_converted == 0:
                print("  NOTE: No foreground masks found; creating empty masks.")
            empty_u8.save(str(out_path / "dynamic_masks" / "all" / f"{prefix}.png"))
            empty_u8.save(str(out_path / "fine_dynamic_masks" / "all" / f"{prefix}.png"))

        empty_u8.save(str(out_path / "dynamic_masks" / "human" / f"{prefix}.png"))
        empty_u8.save(str(out_path / "dynamic_masks" / "vehicle" / f"{prefix}.png"))
        empty_u8.save(str(out_path / "fine_dynamic_masks" / "human" / f"{prefix}.png"))
        empty_u8.save(str(out_path / "fine_dynamic_masks" / "vehicle" / f"{prefix}.png"))
        empty_u16.save(str(out_path / "instances" / f"{prefix}.png"))
        empty_u8.save(str(out_path / "sky_masks" / f"{prefix}.png"))

        n_converted += 1

    if verbose:
        print(f"  Converted {n_converted} frames (images/poses/masks).")

    if lidar_poses_txt.exists():
        result = derive_fixed_cam_to_ego(matched_cam_world, matched_ego_world)
        if result is None:
            raise RuntimeError(
                "Failed to derive camera extrinsics: no matched camera/lidar pose pairs."
            )
        cam_to_ego, extrinsic_stats = result
        np.savetxt(str(out_path / "extrinsics" / "0.txt"), cam_to_ego, fmt="%.10f")
        np.savetxt(str(out_path / "extrinsics" / "1.txt"), cam_to_ego, fmt="%.10f")
        if verbose:
            print(
                f"  Derived fixed camera extrinsic from {extrinsic_stats['num_candidates']} matched pairs "
                f"(rot max Δ={extrinsic_stats['max_rot_delta_deg']:.3f} deg, "
                f"trans max Δ={extrinsic_stats['max_trans_delta_m']:.3f} m)."
            )
    elif verbose:
        print("  Using identity extrinsics (lidar_poses.txt unavailable).")

    with open(str(out_path / "instances" / "instances_info.json"), "w", encoding="utf-8") as jf:
        json.dump({}, jf)
    with open(str(out_path / "instances" / "frame_instances.json"), "w", encoding="utf-8") as jf:
        json.dump({str(i): [] for i in range(n_converted)}, jf)
    with open(str(out_path / "humanpose" / "smpl.pkl"), "wb") as f:
        pickle.dump({}, f)

    lidar_ply_files = sorted(lidars_dir.glob("*.ply")) if lidars_dir.exists() else []
    lidar_dst_map = {}
    for lidar_idx, ply_file in enumerate(lidar_ply_files):
        if ply_file.stem in timestamp_to_timestep:
            lidar_dst_map[ply_file] = timestamp_to_timestep[ply_file.stem]
        elif lidar_idx < n_converted:
            lidar_dst_map[ply_file] = lidar_idx

    if lidar_ply_files:
        if verbose:
            print(f"  Converting {len(lidar_ply_files)} lidar PLY frames to .bin")
        for ply_file in lidar_ply_files:
            pts = ply_to_bin(str(ply_file))
            dst_t = lidar_dst_map.get(ply_file, None)
            if dst_t is None:
                continue
            dst_bin = out_path / "lidar" / f"{dst_t:03d}.bin"
            pts.astype(np.float32).tofile(str(dst_bin))
        if verbose:
            print(f"  Saved {len(lidar_dst_map)} aligned lidar frames to lidar/")
    elif verbose:
        print("  No individual lidar PLY frames found in lidars/.")

    for t in range(n_converted):
        lidar_path = out_path / "lidar" / f"{t:03d}.bin"
        if not lidar_path.exists():
            np.zeros((0, 4), dtype=np.float32).tofile(str(lidar_path))

    return n_converted


def extract_clip_camera_trajectory(clip_path):
    """Return the camera-to-world trajectory for a ParaLane clip as [N,4,4]."""
    clip_path = Path(clip_path)
    sparse_dir = clip_path / "sparse" / "0"
    images_txt = sparse_dir / "images.txt"
    if not images_txt.exists():
        raise FileNotFoundError(f"Required file not found: {images_txt}")

    colmap_images = parse_colmap_images(str(images_txt))
    if not colmap_images:
        return np.empty((0, 4, 4), dtype=np.float32)

    poses = []
    for img_entry in colmap_images:
        c2w_cam = colmap_w2c_to_c2w(
            img_entry["qw"],
            img_entry["qx"],
            img_entry["qy"],
            img_entry["qz"],
            img_entry["tx"],
            img_entry["ty"],
            img_entry["tz"],
        )
        poses.append(c2w_cam)

    return np.asarray(poses, dtype=np.float32)


def save_scene_trajectories(scene_dir, clip_dirs, output_scene_dir, verbose=True):
    """Save per-clip camera trajectories plus a quick verification plot."""
    output_scene_dir = Path(output_scene_dir)
    trajectories_dir = output_scene_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    trajectory_data = []
    for clip_dir in clip_dirs:
        if clip_dir == scene_dir:
            clip_idx = 0
        else:
            clip_idx = int(clip_dir.name.split("_")[1])

        clip_poses = extract_clip_camera_trajectory(clip_dir)
        traj_path = trajectories_dir / f"clip_{clip_idx}.npz"
        np.savez(
            traj_path,
            camera_poses=clip_poses,
            poses=clip_poses,
            trajectory=clip_poses,
        )
        trajectory_data.append((clip_idx, clip_poses[:, :3, 3]))

    fig, ax = plt.subplots(figsize=(8, 6))
    for clip_idx, positions in trajectory_data:
        xs = positions[:, 0]
        ys = positions[:, 1]
        ax.plot(xs, ys, label=f"clip_{clip_idx}")
        ax.scatter([xs[0]], [ys[0]], s=18)
        ax.scatter([xs[-1]], [ys[-1]], s=18)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Camera trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = output_scene_dir / "trajectories.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"  Saved trajectories for {len(clip_dirs)} clips to {trajectories_dir}")
        print(f"  Saved verification plot to {plot_path}")


def convert_dataset(paralane_root, output_root, scenes=None, clips=None, verbose=True):
    """Convert selected/all ParaLane scenes and clips."""
    paralane_root = Path(paralane_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_scene_dirs = sorted(
        [d for d in paralane_root.iterdir() if d.is_dir() and d.name.startswith("scene_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not all_scene_dirs:
        print(f"ERROR: No 'scene_*' directories found in {paralane_root}")
        sys.exit(1)

    total_frames = 0
    total_clips = 0

    for scene_dir in all_scene_dirs:
        scene_idx = int(scene_dir.name.split("_")[1])
        if scenes is not None and scene_idx not in scenes:
            continue

        clip_dirs = sorted(
            [d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("clip_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        if not clip_dirs and (scene_dir / "sparse" / "0").exists() and (scene_dir / "images").exists():
            clip_dirs = [scene_dir]

        scene_out_dirs = {}
        for clip_dir in clip_dirs:
            clip_idx = 0 if clip_dir == scene_dir else int(clip_dir.name.split("_")[1])
            if clips is not None and clip_idx not in clips:
                continue

            output_name = f"scene_{scene_idx:03d}_clip_{clip_idx:03d}"
            out_dir = output_root / output_name
            scene_out_dirs[clip_idx] = out_dir

            print(f"\n{'=' * 60}")
            print(f"  Converting: {clip_dir}")
            print(f"  Output:     {out_dir}")
            print(f"{'=' * 60}")

            try:
                n_frames = convert_clip(str(clip_dir), str(out_dir), verbose=verbose)
                total_frames += n_frames
                total_clips += 1
            except Exception as e:
                print(f"  ERROR processing {clip_dir}: {e}")
                import traceback

                traceback.print_exc()

        for clip_idx, out_dir in scene_out_dirs.items():
            if clips is not None and clip_idx not in clips:
                continue
            save_scene_trajectories(scene_dir, clip_dirs, out_dir, verbose=verbose)

    print(f"\n{'=' * 60}")
    print(f"  Done! Converted {total_clips} clips, {total_frames} total frames.")
    print(f"  Output root: {output_root}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ParaLane to DriveStudio while preserving original world frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root ParaLane directory (contains scene_*/).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output root for DriveStudio processed scenes.",
    )
    parser.add_argument(
        "--scenes",
        type=int,
        nargs="+",
        default=None,
        help="Scene indices to convert (default: all).",
    )
    parser.add_argument(
        "--clips",
        type=int,
        nargs="+",
        default=None,
        help="Clip indices to convert (default: all).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    args = parser.parse_args()

    missing = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        from scipy.spatial.transform import Rotation as _R  # noqa: F401
    except ImportError:
        missing.append("scipy")
    try:
        from PIL import Image as _Image  # noqa: F401
    except ImportError:
        missing.append("Pillow")

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    convert_dataset(
        paralane_root=args.input_dir,
        output_root=args.output_dir,
        scenes=args.scenes,
        clips=args.clips,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()