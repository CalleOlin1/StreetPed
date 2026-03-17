#!/usr/bin/env python3
"""
Top-down trajectory + point cloud alignment visualizer.

Loads two trained output directories, discovers camera trajectory .npz files,
extracts camera-0 trajectories, loads point clouds from each run, then renders:
1) Raw top-down overlay
2) Auto-aligned top-down overlay (trajectory B aligned to trajectory A)

Example:
python tools/align_topdown_trajectories.py \
  --run-dir-a output/paralane/full_wmask \
  --run-dir-b output/paralane/full_nomask \
  --out outputs/alignment_topdown.png --show
"""

import argparse
from pathlib import Path
from pprint import pprint
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _pick_latest_npz(run_dir: Path) -> Path:
    camera_pose_dir = run_dir / "camera_poses"
    if not camera_pose_dir.exists():
        raise FileNotFoundError(f"Missing directory: {camera_pose_dir}")

    candidates = sorted(camera_pose_dir.glob("full_poses_*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No files matching 'full_poses_*.npz' found in {camera_pose_dir}"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _select_camera_zero_mask(data: np.lib.npyio.NpzFile, n: int) -> np.ndarray:
    if "cam_ids" in data:
        cam_ids = np.asarray(data["cam_ids"])
        if len(cam_ids) == n:
            return cam_ids.astype(np.int64) == 0

    if "cam_names" in data:
        cam_names = np.asarray(data["cam_names"]).astype(str)
        if len(cam_names) == n:
            exact = cam_names == "0"
            if np.any(exact):
                return exact

            # Fallback: infer first camera by first-appearance ordering.
            first = cam_names[0]
            return cam_names == first

    # Last fallback for single-camera exports.
    return np.ones(n, dtype=bool)


def _load_camera0_trajectory_bundle(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)

    if "camera_positions" in data:
        positions = np.asarray(data["camera_positions"], dtype=np.float64)
    elif "positions" in data:
        positions = np.asarray(data["positions"], dtype=np.float64)
    elif "trajectory" in data:
        positions = np.asarray(data["trajectory"], dtype=np.float64)
    elif "camera_poses" in data:
        poses = np.asarray(data["camera_poses"], dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"camera_poses in {npz_path} must be [N,4,4], got {poses.shape}"
            )
        positions = poses[:, :3, 3]
    else:
        available = sorted(list(data.keys()))
        raise KeyError(
            f"No supported trajectory key found in {npz_path}. "
            f"Expected one of: camera_positions, positions, trajectory, camera_poses. "
            f"Available keys: {available}"
        )

    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(
            f"Trajectory positions in {npz_path} must be [N,2+] but got {positions.shape}"
        )

    mask = _select_camera_zero_mask(data, len(positions))
    positions = positions[mask]

    if len(positions) == 0:
        raise ValueError(f"No camera-0 trajectory points found in {npz_path}")

    camera_poses = None
    if "camera_poses" in data:
        poses = np.asarray(data["camera_poses"], dtype=np.float64)
        if poses.ndim == 3 and poses.shape[1:] == (4, 4) and len(poses) == len(mask):
            camera_poses = poses[mask]

    frame_indices = None
    cam_names = None
    cam_ids = None

    # If frame indices are present and duplicated (multi-camera export), collapse duplicates by mean.
    if "frame_indices" in data:
        frame_indices = np.asarray(data["frame_indices"])
        if len(frame_indices) == len(mask):
            frame_indices = frame_indices[mask]
            unique_frames = np.unique(frame_indices)
            collapsed = []
            collapsed_poses = [] if camera_poses is not None else None
            for fi in unique_frames:
                fi_mask = frame_indices == fi
                collapsed.append(positions[fi_mask].mean(axis=0))
                if collapsed_poses is not None:
                    # Keep the first pose for the frame to preserve orientation consistency.
                    collapsed_poses.append(camera_poses[np.where(fi_mask)[0][0]])
            positions = np.asarray(collapsed, dtype=np.float64)
            frame_indices = unique_frames
            if collapsed_poses is not None:
                camera_poses = np.asarray(collapsed_poses, dtype=np.float64)

    if "cam_names" in data:
        arr = np.asarray(data["cam_names"]).astype(str)
        if len(arr) == len(mask):
            arr = arr[mask]
            cam_names = arr[: len(positions)]

    if "cam_ids" in data:
        arr = np.asarray(data["cam_ids"])
        if len(arr) == len(mask):
            arr = arr[mask]
            cam_ids = arr[: len(positions)]

    positions3 = (
        positions[:, :3]
        if positions.shape[1] >= 3
        else np.hstack([positions[:, :2], np.zeros((len(positions), 1))])
    )

    if frame_indices is None:
        frame_indices = np.arange(len(positions3), dtype=np.int64)
    else:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)

    if cam_names is None:
        cam_names = np.asarray(["0"] * len(positions3))
    if cam_ids is None:
        cam_ids = np.zeros(len(positions3), dtype=np.int64)

    if camera_poses is None:
        camera_poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(positions3), axis=0)
        camera_poses[:, :3, 3] = positions3

    return {
        "positions": positions3,
        "camera_poses": camera_poses,
        "frame_indices": frame_indices,
        "cam_names": np.asarray(cam_names),
        "cam_ids": np.asarray(cam_ids),
    }


def _extract_point_cloud(path: Path, npz_key: Optional[str]) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        pts = np.asarray(np.load(path), dtype=np.float64)
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        key_candidates = [npz_key] if npz_key else []
        key_candidates += ["points", "point_cloud", "xyz", "pc", "vertices"]
        key = None
        for cand in key_candidates:
            if cand is not None and cand in data:
                key = cand
                break
        if key is None:
            # fallback: first array-like value
            for k in data.keys():
                arr = np.asarray(data[k])
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    key = k
                    break
        if key is None:
            raise KeyError(
                f"Could not find point cloud array in {path}. "
                f"Try --pc-a-key/--pc-b-key. Keys: {sorted(list(data.keys()))}"
            )
        pts = np.asarray(data[key], dtype=np.float64)
    elif suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 == 0:
            pts = raw.reshape(-1, 4)[:, :3].astype(np.float64)
        elif raw.size % 3 == 0:
            pts = raw.reshape(-1, 3).astype(np.float64)
        else:
            raise ValueError(
                f"Unsupported .bin format in {path}: element count {raw.size} not divisible by 3 or 4"
            )
    elif suffix == ".ply":
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "Loading .ply requires open3d. Install it or convert the point cloud to .npy/.npz/.bin"
            ) from exc
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported point cloud file extension '{suffix}' for {path}. "
            "Supported: .npy, .npz, .bin, .ply"
        )

    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"Point cloud in {path} must be [N,2+], got {pts.shape}")

    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((len(pts), 1))])
    else:
        pts = pts[:, :3]

    return pts


def _extract_point_cloud_from_checkpoint(run_dir: Path, checkpoint_name: str) -> np.ndarray:
    ckpt_path = run_dir / checkpoint_name
    if not ckpt_path.exists() and checkpoint_name == "checkpoint_final.pth":
        ckpts = sorted(run_dir.glob("checkpoint_*.pth"))
        if ckpts:
            ckpt_path = max(ckpts, key=lambda p: p.stat().st_mtime)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint under {run_dir}. Tried {checkpoint_name}"
        )

    state = torch.load(str(ckpt_path), map_location="cpu")
    models = state.get("models", {})
    all_points = []

    # Prefer common gaussian mean keys first.
    preferred = ["_means", "means", "gaussians._means"]
    for model_state in models.values():
        if not isinstance(model_state, dict):
            continue
        for key in preferred:
            if key in model_state:
                arr = model_state[key]
                if hasattr(arr, "detach"):
                    arr = arr.detach().cpu().numpy()
                arr = np.asarray(arr)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    all_points.append(arr[:, :3].astype(np.float64))

    # Fallback: scan for any tensor that looks like xyz points.
    if not all_points:
        for model_state in models.values():
            if not isinstance(model_state, dict):
                continue
            for key, value in model_state.items():
                name = str(key).lower()
                if not any(tok in name for tok in ["mean", "point", "xyz", "position"]):
                    continue
                arr = value
                if hasattr(arr, "detach"):
                    arr = arr.detach().cpu().numpy()
                arr = np.asarray(arr)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    all_points.append(arr[:, :3].astype(np.float64))

    if not all_points:
        raise ValueError(
            f"No point-cloud-like tensors found in checkpoint: {ckpt_path}"
        )

    points = np.concatenate(all_points, axis=0)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError(f"All extracted points were non-finite in checkpoint: {ckpt_path}")
    return points


def _estimate_rigid_3d(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(src_xyz) != len(dst_xyz):
        raise ValueError("src_xyz and dst_xyz must have same length")
    if len(src_xyz) < 3:
        raise ValueError("Need at least 3 points to estimate 3D rigid transform")

    src_centroid = src_xyz.mean(axis=0)
    dst_centroid = dst_xyz.mean(axis=0)
    src_centered = src_xyz - src_centroid
    dst_centered = dst_xyz - dst_centroid

    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    # Ensure proper rotation (det=+1), no reflection and no scaling.
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    t = dst_centroid - (r @ src_centroid)
    return r, t


def _rotation_from_roll_pitch_yaw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build rotation matrix as Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _estimate_roll_pitch_match_transform_from_reference_poses(
    pose_a_ref: np.ndarray,
    pose_b_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return world-space transform (R, t) to apply to B so ref-pose B adopts ref-pose A roll/pitch.
    Keeps B ref yaw unchanged and keeps B ref position fixed.
    """
    pose_a_ref = np.asarray(pose_a_ref, dtype=np.float64)
    pose_b_ref = np.asarray(pose_b_ref, dtype=np.float64)

    roll_a, pitch_a, _ = _rotation_to_euler_xyz_deg(pose_a_ref[:3, :3])
    _, _, yaw_b = _rotation_to_euler_xyz_deg(pose_b_ref[:3, :3])

    r_target = _rotation_from_roll_pitch_yaw(
        np.radians(roll_a),
        np.radians(pitch_a),
        np.radians(yaw_b),
    )
    r_src = pose_b_ref[:3, :3]
    r = r_target @ r_src.T

    p_ref = pose_b_ref[:3, 3]
    t = p_ref - (r @ p_ref)
    return r, t


def _mean_nn_distance(src_xyz: np.ndarray, dst_xyz: np.ndarray, chunk_size: int = 1024) -> float:
    if len(src_xyz) == 0 or len(dst_xyz) == 0:
        return float("nan")

    total = 0.0
    count = 0
    dst = dst_xyz.astype(np.float64)
    for i in range(0, len(src_xyz), chunk_size):
        chunk = src_xyz[i : i + chunk_size].astype(np.float64)
        diff = chunk[:, None, :] - dst[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        mins = np.sqrt(np.min(d2, axis=1))
        total += float(np.sum(mins))
        count += len(mins)
    return total / max(1, count)


def _estimate_yaw_translation_from_correspondences(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate yaw-only rotation and full translation from paired correspondences."""
    if len(src_xyz) != len(dst_xyz):
        raise ValueError("src_xyz and dst_xyz must have same length")
    if len(src_xyz) < 2:
        raise ValueError("Need at least 2 points to estimate yaw+translation")

    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)

    src_xy = src[:, :2]
    dst_xy = dst[:, :2]
    src_xy_c = src_xy.mean(axis=0)
    dst_xy_c = dst_xy.mean(axis=0)

    src_xy_0 = src_xy - src_xy_c
    dst_xy_0 = dst_xy - dst_xy_c

    h = src_xy_0.T @ dst_xy_0
    u, _, vt = np.linalg.svd(h)
    r2 = vt.T @ u.T
    if np.linalg.det(r2) < 0:
        vt[-1, :] *= -1
        r2 = vt.T @ u.T

    r3 = np.eye(3, dtype=np.float64)
    r3[:2, :2] = r2

    t_xy = dst_xy_c - (r2 @ src_xy_c)
    t_z = float(dst[:, 2].mean() - src[:, 2].mean())
    t3 = np.asarray([t_xy[0], t_xy[1], t_z], dtype=np.float64)
    return r3, t3


def _estimate_yaw_translation_from_pointclouds(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    max_iters: int = 20,
    chunk_size: int = 1024,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Estimate yaw-only rotation + translation transform (R, t) for src -> dst using ICP.
    """
    if len(src_xyz) == 0 or len(dst_xyz) == 0:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), float("nan"), float("nan")

    src = src_xyz.astype(np.float64)
    dst = dst_xyz.astype(np.float64)
    r_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)

    err_before = _mean_nn_distance(src, dst, chunk_size=chunk_size)
    prev_err = err_before

    for _ in range(max_iters):
        moved = (r_total @ src.T).T + t_total[None, :]

        matched = []
        for i in range(0, len(moved), chunk_size):
            chunk = moved[i : i + chunk_size]
            diff = chunk[:, None, :] - dst[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            nn_idx = np.argmin(d2, axis=1)
            matched.append(dst[nn_idx])
        dst_match = np.concatenate(matched, axis=0)

        r_delta, t_delta = _estimate_yaw_translation_from_correspondences(moved, dst_match)
        r_total = r_delta @ r_total
        t_total = (r_delta @ t_total) + t_delta

        moved_new = (r_total @ src.T).T + t_total[None, :]
        err_new = _mean_nn_distance(moved_new, dst, chunk_size=chunk_size)
        if abs(prev_err - err_new) < tol:
            prev_err = err_new
            break
        prev_err = err_new

    return r_total, t_total, err_before, prev_err


def _apply_transform_xy(points_xyz: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = points_xyz.copy()
    out[:, :2] = (r @ points_xyz[:, :2].T).T + t
    return out


def _apply_translation_xy(points_xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = points_xyz.copy()
    out[:, :2] = out[:, :2] + t[None, :]
    return out


def _apply_transform_xyz(points_xyz: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = points_xyz.copy()
    out[:, :3] = (r @ points_xyz[:, :3].T).T + t
    return out


def _subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def _plot_overlay(
    ax,
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    pc_a: np.ndarray,
    pc_b: np.ndarray,
    title: str,
    pc_alpha: float,
) -> None:
    ax.scatter(pc_a[:, 0], pc_a[:, 1], s=0.5, c="tab:blue", alpha=pc_alpha, label="PointCloud A")
    ax.scatter(pc_b[:, 0], pc_b[:, 1], s=0.5, c="tab:orange", alpha=pc_alpha, label="PointCloud B")

    ax.plot(traj_a[:, 0], traj_a[:, 1], "-", c="navy", linewidth=2.0, label="Trajectory A")
    ax.plot(traj_b[:, 0], traj_b[:, 1], "-", c="darkorange", linewidth=2.0, label="Trajectory B")

    ax.scatter(traj_a[0, 0], traj_a[0, 1], c="navy", s=35, marker="o")
    ax.scatter(traj_a[-1, 0], traj_a[-1, 1], c="navy", s=35, marker="s")
    ax.scatter(traj_b[0, 0], traj_b[0, 1], c="darkorange", s=35, marker="o")
    ax.scatter(traj_b[-1, 0], traj_b[-1, 1], c="darkorange", s=35, marker="s")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _set_axis_to_point_percentile(
    ax,
    xy_arrays: list,
    keep_fraction: float = 0.8,
    min_span: float = 1.0,
) -> None:
    if not xy_arrays:
        return

    pts = []
    for arr in xy_arrays:
        arr_np = np.asarray(arr)
        if arr_np.ndim == 2 and arr_np.shape[1] >= 2 and len(arr_np) > 0:
            pts.append(arr_np[:, :2])
    if not pts:
        return

    merged = np.concatenate(pts, axis=0)
    merged = merged[np.isfinite(merged).all(axis=1)]
    if len(merged) == 0:
        return

    keep_fraction = float(np.clip(keep_fraction, 1e-3, 1.0))
    tail = (1.0 - keep_fraction) / 2.0
    low_q = 100.0 * tail
    high_q = 100.0 * (1.0 - tail)

    x_low, y_low = np.percentile(merged, low_q, axis=0)
    x_high, y_high = np.percentile(merged, high_q, axis=0)

    x_span = max(float(x_high - x_low), float(min_span))
    y_span = max(float(y_high - y_low), float(min_span))
    x_mid = 0.5 * (x_low + x_high)
    y_mid = 0.5 * (y_low + y_high)

    ax.set_xlim(x_mid - 0.5 * x_span, x_mid + 0.5 * x_span)
    ax.set_ylim(y_mid - 0.5 * y_span, y_mid + 0.5 * y_span)


def _rotation_to_euler_xyz_deg(r: np.ndarray) -> Tuple[float, float, float]:
    """Return Euler XYZ angles (roll, pitch, yaw) in degrees."""
    r = np.asarray(r, dtype=np.float64)
    pitch = np.degrees(np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return float(roll), float(pitch), float(yaw)


def _plot_middle_pose_rectangles_3d(
    ax,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    label_b: str,
    rect_w: float = 1.2,
    rect_h: float = 0.7,
) -> None:
    """Plot camera-local XY rectangles in world space for two 3D c2w camera poses."""

    def _corners_world(c2w: np.ndarray) -> np.ndarray:
        # Rectangle in camera local XY plane centered at origin.
        local = np.asarray(
            [
                [-0.5 * rect_w, -0.5 * rect_h, 0.0],
                [0.5 * rect_w, -0.5 * rect_h, 0.0],
                [0.5 * rect_w, 0.5 * rect_h, 0.0],
                [-0.5 * rect_w, 0.5 * rect_h, 0.0],
            ],
            dtype=np.float64,
        )
        rot = c2w[:3, :3]
        pos = c2w[:3, 3]
        return (rot @ local.T).T + pos[None, :]

    pose_a = np.asarray(pose_a, dtype=np.float64)
    pose_b = np.asarray(pose_b, dtype=np.float64)
    ca = _corners_world(pose_a)
    cb = _corners_world(pose_b)
    pa = pose_a[:3, 3]
    pb = pose_b[:3, 3]

    poly_a = Poly3DCollection([ca], alpha=0.25, facecolor="tab:blue", edgecolor="navy", linewidth=1.5)
    poly_b = Poly3DCollection([cb], alpha=0.25, facecolor="tab:orange", edgecolor="darkorange", linewidth=1.5)
    ax.add_collection3d(poly_a)
    ax.add_collection3d(poly_b)

    # Outline and centers.
    ca_closed = np.vstack([ca, ca[0]])
    cb_closed = np.vstack([cb, cb[0]])
    ax.plot(ca_closed[:, 0], ca_closed[:, 1], ca_closed[:, 2], c="navy", linewidth=1.5, label="Mid Pose A")
    ax.plot(cb_closed[:, 0], cb_closed[:, 1], cb_closed[:, 2], c="darkorange", linewidth=1.5, label=f"Mid Pose {label_b}")
    ax.scatter([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], c=["navy", "darkorange"], s=40)

    # Draw local +Z direction to emphasize pitch/roll orientation.
    va = pose_a[:3, 2]
    vb = pose_b[:3, 2]
    ax.quiver(pa[0], pa[1], pa[2], va[0], va[1], va[2], length=0.9, color="navy")
    ax.quiver(pb[0], pb[1], pb[2], vb[0], vb[1], vb[2], length=0.9, color="darkorange")

    roll_a, pitch_a, _ = _rotation_to_euler_xyz_deg(pose_a[:3, :3])
    roll_b, pitch_b, _ = _rotation_to_euler_xyz_deg(pose_b[:3, :3])
    ax.set_title(
        f"Middle 3D Camera Pose Rectangles\n"
        f"A roll/pitch={roll_a:.2f}/{pitch_a:.2f} deg | {label_b} roll/pitch={roll_b:.2f}/{pitch_b:.2f} deg"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=24, azim=-62)

    pts = np.concatenate([ca, cb, pa[None, :], pb[None, :]], axis=0)
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    span = np.maximum(pmax - pmin, 0.6)
    center = 0.5 * (pmin + pmax)
    half = 0.65 * float(np.max(span))
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _save_shifted_trajectory_npz(
    out_path: Path,
    shifted_positions: np.ndarray,
    source_npz: Path,
    shifted_camera_poses: Optional[np.ndarray] = None,
    frame_indices: Optional[np.ndarray] = None,
    cam_names: Optional[np.ndarray] = None,
    cam_ids: Optional[np.ndarray] = None,
    transform_r2: Optional[np.ndarray] = None,
    transform_t2: Optional[np.ndarray] = None,
    transform_r3: Optional[np.ndarray] = None,
    transform_t3: Optional[np.ndarray] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shifted_positions = np.asarray(shifted_positions, dtype=np.float64)

    n = len(shifted_positions)
    if shifted_camera_poses is not None:
        camera_poses = np.asarray(shifted_camera_poses, dtype=np.float64)
    else:
        camera_poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n, axis=0)
        camera_poses[:, :3, 3] = shifted_positions[:, :3]

    if frame_indices is None:
        frame_indices = np.arange(n, dtype=np.int64)
    if cam_names is None:
        cam_names = np.asarray(["0"] * n)
    if cam_ids is None:
        cam_ids = np.zeros(n, dtype=np.int64)

    payload = {
        "camera_positions": shifted_positions,
        "camera_poses": camera_poses,
        "cam_names": np.asarray(cam_names),
        "cam_ids": np.asarray(cam_ids),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "source_npz": str(source_npz),
    }
    if transform_r2 is not None:
        payload["transform_r2"] = np.asarray(transform_r2, dtype=np.float64)
    if transform_t2 is not None:
        payload["transform_t2"] = np.asarray(transform_t2, dtype=np.float64)
    if transform_r3 is not None:
        payload["transform_r3"] = np.asarray(transform_r3, dtype=np.float64)
    if transform_t3 is not None:
        payload["transform_t3"] = np.asarray(transform_t3, dtype=np.float64)
    np.savez(out_path, **payload)
    print(f"Saved shifted trajectory B to: {out_path}")


def _print_camera_comparison(
    source_positions: np.ndarray,
    shifted_positions: np.ndarray,
    source_poses: np.ndarray,
    shifted_poses: np.ndarray,
    frame_indices: np.ndarray,
    transform_r2: Optional[np.ndarray],
    transform_t2: Optional[np.ndarray],
) -> None:
    src_pos = np.asarray(source_positions, dtype=np.float64)
    shf_pos = np.asarray(shifted_positions, dtype=np.float64)
    src_pose = np.asarray(source_poses, dtype=np.float64)
    shf_pose = np.asarray(shifted_poses, dtype=np.float64)
    n = min(len(src_pos), len(shf_pos), len(src_pose), len(shf_pose))

    if n == 0:
        pprint({"camera_comparison": "no frames to compare"})
        return

    src_pos = src_pos[:n]
    shf_pos = shf_pos[:n]
    src_pose = src_pose[:n]
    shf_pose = shf_pose[:n]
    fi = np.asarray(frame_indices, dtype=np.int64)[:n]

    pos_delta = shf_pos - src_pos
    yaw_src = np.degrees(np.arctan2(src_pose[:, 1, 0], src_pose[:, 0, 0]))
    yaw_shf = np.degrees(np.arctan2(shf_pose[:, 1, 0], shf_pose[:, 0, 0]))
    yaw_delta = yaw_shf - yaw_src

    summary = {
        "camera_comparison": {
            "num_frames": int(n),
            "frame_start_end": [int(fi.min()), int(fi.max())] if len(fi) else [None, None],
            "source_position_min_xyz": src_pos.min(axis=0).round(6).tolist(),
            "source_position_max_xyz": src_pos.max(axis=0).round(6).tolist(),
            "shifted_position_min_xyz": shf_pos.min(axis=0).round(6).tolist(),
            "shifted_position_max_xyz": shf_pos.max(axis=0).round(6).tolist(),
            "position_delta_mean_xyz": pos_delta.mean(axis=0).round(6).tolist(),
            "position_delta_std_xyz": pos_delta.std(axis=0).round(6).tolist(),
            "position_delta_min_xyz": pos_delta.min(axis=0).round(6).tolist(),
            "position_delta_max_xyz": pos_delta.max(axis=0).round(6).tolist(),
            "mean_abs_delta_xy": float(np.linalg.norm(pos_delta[:, :2], axis=1).mean()),
            "yaw_delta_deg_mean": float(np.mean(yaw_delta)),
            "yaw_delta_deg_std": float(np.std(yaw_delta)),
            "transform_r2": None if transform_r2 is None else np.asarray(transform_r2).round(8).tolist(),
            "transform_t2": None if transform_t2 is None else np.asarray(transform_t2).round(8).tolist(),
        }
    }
    pprint(summary, sort_dicts=False)


def _print_cross_run_camera_comparison(
    poses_a: np.ndarray,
    poses_b_aligned: np.ndarray,
    positions_a: np.ndarray,
    positions_b_aligned: np.ndarray,
    frame_indices_a: np.ndarray,
    frame_indices_b: np.ndarray,
) -> None:
    """Debug print comparing camera A to aligned camera B on shared frames."""
    poses_a = np.asarray(poses_a, dtype=np.float64)
    poses_b = np.asarray(poses_b_aligned, dtype=np.float64)
    pos_a = np.asarray(positions_a, dtype=np.float64)
    pos_b = np.asarray(positions_b_aligned, dtype=np.float64)
    fi_a = np.asarray(frame_indices_a, dtype=np.int64)
    fi_b = np.asarray(frame_indices_b, dtype=np.int64)

    if len(poses_a) == 0 or len(poses_b) == 0:
        pprint({"cross_run_camera_comparison": "empty pose arrays"})
        return

    common = np.intersect1d(fi_a, fi_b)
    if len(common) == 0:
        n = min(len(poses_a), len(poses_b), len(pos_a), len(pos_b))
        if n == 0:
            pprint({"cross_run_camera_comparison": "no overlapping frames and no fallback samples"})
            return
        idx_a = np.arange(n)
        idx_b = np.arange(n)
        frame_label = "index_aligned_fallback"
    else:
        map_a = {int(f): i for i, f in enumerate(fi_a.tolist())}
        map_b = {int(f): i for i, f in enumerate(fi_b.tolist())}
        idx_a = np.asarray([map_a[int(f)] for f in common], dtype=np.int64)
        idx_b = np.asarray([map_b[int(f)] for f in common], dtype=np.int64)
        frame_label = "shared_frame_indices"

    pa = poses_a[idx_a]
    pb = poses_b[idx_b]
    xa = pos_a[idx_a]
    xb = pos_b[idx_b]

    pos_delta = xb - xa
    delta_norm = np.linalg.norm(pos_delta, axis=1)

    # Relative rotation: R_rel = R_a^T * R_b. Report Euler xyz in degrees as a compact tilt indicator.
    r_rel = np.einsum("nij,njk->nik", np.transpose(pa[:, :3, :3], (0, 2, 1)), pb[:, :3, :3])
    pitch = np.degrees(np.arcsin(np.clip(-r_rel[:, 2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(r_rel[:, 2, 1], r_rel[:, 2, 2]))
    yaw = np.degrees(np.arctan2(r_rel[:, 1, 0], r_rel[:, 0, 0]))

    summary = {
        "cross_run_camera_comparison": {
            "num_compared": int(len(idx_a)),
            "frame_mode": frame_label,
            "position_delta_mean_xyz": pos_delta.mean(axis=0).round(6).tolist(),
            "position_delta_std_xyz": pos_delta.std(axis=0).round(6).tolist(),
            "position_delta_l2_mean": float(delta_norm.mean()),
            "position_delta_l2_max": float(delta_norm.max()),
            "relative_euler_deg_mean_xyz": [
                float(np.mean(roll)),
                float(np.mean(pitch)),
                float(np.mean(yaw)),
            ],
            "relative_euler_deg_std_xyz": [
                float(np.std(roll)),
                float(np.std(pitch)),
                float(np.std(yaw)),
            ],
            "relative_pitch_deg_abs_mean": float(np.mean(np.abs(pitch))),
            "relative_roll_deg_abs_mean": float(np.mean(np.abs(roll))),
        }
    }
    pprint(summary, sort_dicts=False)


def _transform_camera_poses_3d(camera_poses: np.ndarray, r3: np.ndarray, t3: np.ndarray) -> np.ndarray:
    """Apply world-space 3D rigid transform (rotation + translation) to c2w camera poses."""
    poses = np.asarray(camera_poses, dtype=np.float64)
    tw = np.eye(4, dtype=np.float64)
    tw[:3, :3] = np.asarray(r3, dtype=np.float64)
    tw[:3, 3] = np.asarray(t3, dtype=np.float64)
    return np.einsum("ij,njk->nik", tw, poses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize top-down alignment between two trained run directories."
    )
    parser.add_argument("--run-dir-a", type=Path, required=True, help="Run directory A (contains camera_poses*/ and checkpoints)")
    parser.add_argument("--run-dir-b", type=Path, required=True, help="Run directory B (contains camera_poses*/ and checkpoints)")

    parser.add_argument("--pc-a", type=Path, default=None, help="Optional explicit point cloud A path (.npy/.npz/.bin/.ply)")
    parser.add_argument("--pc-b", type=Path, default=None, help="Optional explicit point cloud B path (.npy/.npz/.bin/.ply)")
    parser.add_argument("--pc-a-key", type=str, default=None, help="Key to load from point cloud A .npz")
    parser.add_argument("--pc-b-key", type=str, default=None, help="Key to load from point cloud B .npz")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_final.pth", help="Checkpoint name used when point clouds are extracted from run dirs")

    parser.add_argument("--max-pc-points", type=int, default=15000, help="Max sampled points per cloud for plotting")
    parser.add_argument("--pc-alpha", type=float, default=0.15, help="Point cloud alpha in plot")
    parser.add_argument("--pc-align-samples", type=int, default=1000, help="Sample size per cloud for point-cloud yaw+translation alignment")
    parser.add_argument("--pc-align-iters", type=int, default=100, help="ICP iterations for point-cloud yaw+translation alignment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for point cloud subsampling")
    parser.add_argument("--no-auto-align", action="store_true", help="Skip automatic alignment and only plot raw overlay")
    parser.add_argument("--save-shifted-traj-b", type=Path, default=None, help="Optional output .npz path for shifted trajectory B")

    parser.add_argument("--out", type=Path, default=None, help="Optional output image path")
    parser.add_argument("--show", action="store_true", help="Show matplotlib window")

    args = parser.parse_args()

    for p in [args.run_dir_a, args.run_dir_b]:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    traj_npz_a = _pick_latest_npz(args.run_dir_a)
    traj_npz_b = _pick_latest_npz(args.run_dir_b)

    traj_bundle_a = _load_camera0_trajectory_bundle(traj_npz_a)
    traj_bundle_b = _load_camera0_trajectory_bundle(traj_npz_b)
    traj_a = traj_bundle_a["positions"]
    traj_b = traj_bundle_b["positions"]

    if args.pc_a is not None:
        if not args.pc_a.exists():
            raise FileNotFoundError(f"File not found: {args.pc_a}")
        pc_a = _extract_point_cloud(args.pc_a, args.pc_a_key)
    else:
        pc_a = _extract_point_cloud_from_checkpoint(args.run_dir_a, args.checkpoint_name)

    if args.pc_b is not None:
        if not args.pc_b.exists():
            raise FileNotFoundError(f"File not found: {args.pc_b}")
        pc_b = _extract_point_cloud(args.pc_b, args.pc_b_key)
    else:
        pc_b = _extract_point_cloud_from_checkpoint(args.run_dir_b, args.checkpoint_name)

    print(f"Trajectory npz A: {traj_npz_a}")
    print(f"Trajectory npz B: {traj_npz_b}")
    print("Camera selection: camera 0")

    pc_a_plot = _subsample(pc_a, args.max_pc_points, args.seed)
    pc_b_plot = _subsample(pc_b, args.max_pc_points, args.seed + 1)

    print(f"Trajectory A points: {len(traj_a)}")
    print(f"Trajectory B points: {len(traj_b)}")
    print(f"PointCloud A points (plotting): {len(pc_a_plot)} / {len(pc_a)}")
    print(f"PointCloud B points (plotting): {len(pc_b_plot)} / {len(pc_b)}")

    if args.no_auto_align:
        fig = plt.figure(figsize=(18, 8))
        ax = fig.add_subplot(1, 2, 1)
        _plot_overlay(ax, traj_a, traj_b, pc_a_plot, pc_b_plot, "Raw Overlay", args.pc_alpha)
        _set_axis_to_point_percentile(
            ax,
            [pc_a_plot, pc_b_plot, traj_a, traj_b],
            keep_fraction=0.8,
        )
        ax_pose = fig.add_subplot(1, 2, 2, projection="3d")
        mid_a = len(traj_bundle_a["camera_poses"]) // 2
        mid_b = len(traj_bundle_b["camera_poses"]) // 2
        _plot_middle_pose_rectangles_3d(
            ax_pose,
            traj_bundle_a["camera_poses"][mid_a],
            traj_bundle_b["camera_poses"][mid_b],
            label_b="B (raw)",
        )
        ax_pose.legend(loc="best")
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="best")
        if args.save_shifted_traj_b is not None:
            _save_shifted_trajectory_npz(
                args.save_shifted_traj_b,
                traj_b,
                traj_npz_b,
                shifted_camera_poses=traj_bundle_b["camera_poses"],
                frame_indices=traj_bundle_b["frame_indices"],
                cam_names=traj_bundle_b["cam_names"],
                cam_ids=traj_bundle_b["cam_ids"],
                transform_r2=np.eye(2, dtype=np.float64),
                transform_t2=np.zeros(2, dtype=np.float64),
                transform_r3=np.eye(3, dtype=np.float64),
                transform_t3=np.zeros(3, dtype=np.float64),
            )
            _print_camera_comparison(
                source_positions=traj_bundle_b["positions"],
                shifted_positions=traj_b,
                source_poses=traj_bundle_b["camera_poses"],
                shifted_poses=traj_bundle_b["camera_poses"],
                frame_indices=traj_bundle_b["frame_indices"],
                transform_r2=np.eye(2, dtype=np.float64),
                transform_t2=np.zeros(2, dtype=np.float64),
            )
            print("Note: --no-auto-align enabled, so saved trajectory B is unshifted.")
    else:
        # Stage 1: rotate B world so the middle B camera has same roll/pitch as middle A camera.
        pc_a_fit = _subsample(pc_a, args.pc_align_samples, args.seed + 10)
        pc_b_fit = _subsample(pc_b, args.pc_align_samples, args.seed + 11)

        mid_a = len(traj_bundle_a["camera_poses"]) // 2
        mid_b = len(traj_bundle_b["camera_poses"]) // 2
        pose_a_mid = np.asarray(traj_bundle_a["camera_poses"][mid_a], dtype=np.float64)
        pose_b_mid = np.asarray(traj_bundle_b["camera_poses"][mid_b], dtype=np.float64)

        r_rp, t_rp = _estimate_roll_pitch_match_transform_from_reference_poses(pose_a_mid, pose_b_mid)

        traj_b_rp = _apply_transform_xyz(traj_b, r_rp, t_rp)
        pc_b_fit_rp = _apply_transform_xyz(pc_b_fit, r_rp, t_rp)
        pc_b_plot_rp = _apply_transform_xyz(pc_b_plot, r_rp, t_rp)
        pose_b_rp = _transform_camera_poses_3d(traj_bundle_b["camera_poses"], r_rp, t_rp)

        # Stage 2: point-cloud-only yaw+translation alignment (B -> A).
        r_yaw, t_yaw, before_refine, after_refine = _estimate_yaw_translation_from_pointclouds(
            pc_b_fit_rp[:, :3],
            pc_a_fit[:, :3],
            max_iters=args.pc_align_iters,
        )

        # Compose total transform from original B -> final aligned B.
        r_pc3 = r_yaw @ r_rp
        t_pc3 = (r_yaw @ t_rp) + t_yaw

        # Apply yaw+translation stage to roll/pitch-corrected B.
        traj_b_aligned = _apply_transform_xyz(traj_b_rp, r_yaw, t_yaw)
        pc_b_aligned = _apply_transform_xyz(pc_b_plot_rp, r_yaw, t_yaw)
        pose_b_aligned = _transform_camera_poses_3d(pose_b_rp, r_yaw, t_yaw)

        _print_cross_run_camera_comparison(
            poses_a=traj_bundle_a["camera_poses"],
            poses_b_aligned=pose_b_aligned,
            positions_a=traj_bundle_a["positions"],
            positions_b_aligned=traj_b_aligned,
            frame_indices_a=traj_bundle_a["frame_indices"],
            frame_indices_b=traj_bundle_b["frame_indices"],
        )

        roll_a_mid, pitch_a_mid, _ = _rotation_to_euler_xyz_deg(pose_a_mid[:3, :3])
        roll_b_mid_raw, pitch_b_mid_raw, _ = _rotation_to_euler_xyz_deg(pose_b_mid[:3, :3])
        roll_b_mid_rp, pitch_b_mid_rp, _ = _rotation_to_euler_xyz_deg(pose_b_rp[mid_b, :3, :3])
        yaw_deg = float(np.degrees(np.arctan2(r_yaw[1, 0], r_yaw[0, 0])))

        print("Stage 1 (roll/pitch match at middle pose):")
        print(f"  A mid roll/pitch (deg): {roll_a_mid:.6f}, {pitch_a_mid:.6f}")
        print(f"  B mid roll/pitch raw (deg): {roll_b_mid_raw:.6f}, {pitch_b_mid_raw:.6f}")
        print(f"  B mid roll/pitch after stage1 (deg): {roll_b_mid_rp:.6f}, {pitch_b_mid_rp:.6f}")
        print("Stage 2 (point-cloud yaw+translation alignment, B -> A):")
        print(f"  yaw (deg): {yaw_deg:.6f}")
        print(f"  translation xyz: tx={t_yaw[0]:.6f}, ty={t_yaw[1]:.6f}, tz={t_yaw[2]:.6f}")
        print(f"  mean NN distance: {before_refine:.6f} -> {after_refine:.6f}")

        fig = plt.figure(figsize=(26, 8))
        axes = [
            fig.add_subplot(1, 3, 1),
            fig.add_subplot(1, 3, 2),
            fig.add_subplot(1, 3, 3, projection="3d"),
        ]
        _plot_overlay(
            axes[0],
            traj_a,
            traj_b,
            pc_a_plot,
            pc_b_plot,
            "Before Alignment",
            args.pc_alpha,
        )
        _set_axis_to_point_percentile(
            axes[0],
            [pc_a_plot, pc_b_plot, traj_a, traj_b],
            keep_fraction=0.8,
        )
        _plot_overlay(
            axes[1],
            traj_a,
            traj_b_aligned,
            pc_a_plot,
            pc_b_aligned,
            "After Alignment (B -> A)",
            args.pc_alpha,
        )
        _set_axis_to_point_percentile(
            axes[1],
            [pc_a_plot, pc_b_aligned, traj_a, traj_b_aligned],
            keep_fraction=0.8,
        )

        mid_a = len(traj_bundle_a["camera_poses"]) // 2
        mid_b = len(pose_b_aligned) // 2
        _plot_middle_pose_rectangles_3d(
            axes[2],
            traj_bundle_a["camera_poses"][mid_a],
            pose_b_aligned[mid_b],
            label_b="B (aligned)",
        )

        if args.save_shifted_traj_b is not None:
            _save_shifted_trajectory_npz(
                args.save_shifted_traj_b,
                traj_b_aligned,
                traj_npz_b,
                shifted_camera_poses=pose_b_aligned,
                frame_indices=traj_bundle_b["frame_indices"],
                cam_names=traj_bundle_b["cam_names"],
                cam_ids=traj_bundle_b["cam_ids"],
                transform_r2=np.asarray(r_pc3[:2, :2], dtype=np.float64),
                transform_t2=np.asarray(t_pc3[:2], dtype=np.float64),
                transform_r3=r_pc3,
                transform_t3=t_pc3,
            )
            _print_camera_comparison(
                source_positions=traj_bundle_b["positions"],
                shifted_positions=traj_b_aligned,
                source_poses=traj_bundle_b["camera_poses"],
                shifted_poses=pose_b_aligned,
                frame_indices=traj_bundle_b["frame_indices"],
                transform_r2=np.asarray(r_pc3[:2, :2], dtype=np.float64),
                transform_t2=np.asarray(t_pc3[:2], dtype=np.float64),
            )

        handles, labels = axes[0].get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        axes[1].legend(uniq.values(), uniq.keys(), loc="best")
        axes[2].legend(loc="best")

    plt.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {args.out}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
