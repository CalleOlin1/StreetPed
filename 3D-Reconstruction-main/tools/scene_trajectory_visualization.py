#!/usr/bin/env python3
"""
Unified scene trajectory visualizer.

Visualizes camera + all instance trajectories (pedestrians, vehicles, others)
in one figure with:
- XY top-down view
- 3D trajectory view

Usage:
python tools/scene_trajectory_visualization.py \
    --camera_npz /path/to/camera_poses.npz \
    --instances_json /path/to/instances_info.json \
    --output ./scene_trajectories.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def load_camera_trajectory(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    required_keys = ["camera_positions", "frame_indices"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in camera npz: {npz_path}")

    camera_positions = np.asarray(data["camera_positions"])
    frame_indices = np.asarray(data["frame_indices"])
    cam_names = np.asarray(data["cam_names"]) if "cam_names" in data else None
    num_cameras = len(set(str(x) for x in cam_names.tolist())) if cam_names is not None else 1

    if camera_positions.ndim != 2 or camera_positions.shape[1] != 3:
        raise ValueError(
            f"camera_positions must be [N, 3], got {camera_positions.shape}"
        )

    return {
        "positions": camera_positions,
        "frame_indices": frame_indices,
        "cam_names": cam_names,
        "num_cameras": max(1, int(num_cameras)),
    }


def build_camera_reference_trajectory(
    camera: Dict[str, np.ndarray],
    camera_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    def collapse_to_unique_frames(pos: np.ndarray, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        unique_frames = np.unique(frames)
        collapsed_positions = []
        for fi in unique_frames:
            mask_frame = frames == fi
            collapsed_positions.append(pos[mask_frame].mean(axis=0))
        return np.asarray(collapsed_positions), unique_frames

    positions = np.asarray(camera["positions"])
    frame_indices = np.asarray(camera["frame_indices"])
    cam_names = camera.get("cam_names")
    num_cameras = int(camera.get("num_cameras", 1))

    if cam_names is None:
        collapsed_positions, collapsed_frames = collapse_to_unique_frames(
            positions, frame_indices
        )
        return {
            "positions": collapsed_positions,
            "frame_indices": collapsed_frames,
            "selected_camera": "unknown_collapsed",
            "num_cameras": num_cameras,
        }

    cam_names = np.asarray(cam_names)
    cam_name_strings = np.array([str(x) for x in cam_names.tolist()])
    available = sorted(set(cam_name_strings.tolist()))
    selected_camera = camera_name if camera_name else available[0]

    mask = cam_name_strings == str(selected_camera)
    if mask.sum() == 0:
        raise ValueError(
            f"Camera '{selected_camera}' not found. Available cameras: {available}"
        )

    selected_positions = positions[mask]
    selected_frames = frame_indices[mask]
    collapsed_positions, collapsed_frames = collapse_to_unique_frames(
        selected_positions, selected_frames
    )

    return {
        "positions": collapsed_positions,
        "frame_indices": collapsed_frames,
        "selected_camera": selected_camera,
        "num_cameras": num_cameras,
    }


def load_ego_start_inverse_transform(ego_pose_start_path: str) -> np.ndarray:
    ego_to_world_start = np.loadtxt(ego_pose_start_path)
    if ego_to_world_start.shape != (4, 4):
        raise ValueError(
            f"ego_pose_start must be a 4x4 matrix, got {ego_to_world_start.shape}"
        )
    return np.linalg.inv(ego_to_world_start)


def transform_points(points: np.ndarray, world_transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.concatenate([points, ones], axis=1)
    transformed_h = (world_transform @ points_h.T).T
    return transformed_h[:, :3]


def classify_instance(class_name: str) -> str:
    label = (class_name or "unknown").lower()
    if any(token in label for token in ["pedestrian", "person", "human"]):
        return "pedestrian"
    if any(
        token in label
        for token in [
            "vehicle",
            "car",
            "truck",
            "bus",
            "van",
            "motorcycle",
            "bike",
            "bicycle",
        ]
    ):
        return "vehicle"
    return "other"


def load_instance_trajectories(
    json_path: str,
    world_transform: Optional[np.ndarray] = None,
) -> List[Dict[str, np.ndarray]]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    trajectories: List[Dict[str, np.ndarray]] = []

    for instance_id, instance_data in raw.items():
        frame_annotations = instance_data.get("frame_annotations", {})
        frame_idx = frame_annotations.get("frame_idx", [])
        obj_to_world = frame_annotations.get("obj_to_world", [])

        if not frame_idx or not obj_to_world:
            continue

        positions = []
        valid_frames = []
        for fi, matrix in zip(frame_idx, obj_to_world):
            matrix_np = np.asarray(matrix)
            if matrix_np.shape != (4, 4):
                continue
            if world_transform is not None:
                matrix_np = world_transform @ matrix_np
            positions.append(matrix_np[:3, 3])
            valid_frames.append(fi)

        if len(positions) == 0:
            continue

        positions_np = np.asarray(positions)
        class_name = str(instance_data.get("class_name", "Unknown"))

        trajectories.append(
            {
                "instance_id": str(instance_id),
                "class_name": class_name,
                "group": classify_instance(class_name),
                "frame_indices": np.asarray(valid_frames),
                "positions": positions_np,
            }
        )

    return trajectories


def set_equal_3d(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x_mid = (x.max() + x.min()) / 2.0
    y_mid = (y.max() + y.min()) / 2.0
    z_mid = (z.max() + z.min()) / 2.0

    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
    if max_range == 0:
        max_range = 1.0

    ax.set_xlim(x_mid - max_range / 2.0, x_mid + max_range / 2.0)
    ax.set_ylim(y_mid - max_range / 2.0, y_mid + max_range / 2.0)
    ax.set_zlim(z_mid - max_range / 2.0, z_mid + max_range / 2.0)
    ax.set_box_aspect((1, 1, 1))


def get_position_at_frame(
    frame_indices: np.ndarray,
    positions: np.ndarray,
    frame_value: int,
) -> Optional[np.ndarray]:
    mask = frame_indices == frame_value
    if not np.any(mask):
        return None
    return positions[mask].mean(axis=0)


def get_velocity_at_frame(
    frame_indices: np.ndarray,
    positions: np.ndarray,
    frame_value: int,
) -> Optional[np.ndarray]:
    unique_frames = np.asarray(frame_indices)
    unique_positions = np.asarray(positions)
    if unique_frames.size < 2:
        return None

    frame_order = np.argsort(unique_frames)
    ordered_frames = unique_frames[frame_order]
    ordered_positions = unique_positions[frame_order]

    idx_candidates = np.where(ordered_frames == frame_value)[0]
    if idx_candidates.size == 0:
        return None

    idx = int(idx_candidates[0])
    if idx == 0:
        dt = ordered_frames[1] - ordered_frames[0]
        if dt == 0:
            return None
        return (ordered_positions[1] - ordered_positions[0]) / dt
    if idx == len(ordered_frames) - 1:
        dt = ordered_frames[-1] - ordered_frames[-2]
        if dt == 0:
            return None
        return (ordered_positions[-1] - ordered_positions[-2]) / dt

    dt = ordered_frames[idx + 1] - ordered_frames[idx - 1]
    if dt == 0:
        return None
    return (ordered_positions[idx + 1] - ordered_positions[idx - 1]) / dt


def plot_scene_trajectories(
    camera: Dict[str, np.ndarray],
    instances: List[Dict[str, np.ndarray]],
    output_path: str,
    show: bool = False,
    max_instances: int = 0,
    enable_slider: bool = False,
    highlight_size: float = 120.0,
    velocity_scale: float = 1.0,
) -> Tuple[int, int]:
    camera_positions = camera["positions"]
    camera_frames = np.asarray(camera["frame_indices"])
    num_cameras = int(camera.get("num_cameras", 1))
    camera_scene_frames = camera_frames // max(1, num_cameras)
    instance_subset = instances if max_instances <= 0 else instances[:max_instances]

    colors = {
        "pedestrian": "tab:orange",
        "vehicle": "tab:blue",
        "camera": "tab:green",
        "other": "tab:gray",
    }

    fig = plt.figure(figsize=(15, 7))
    ax_xy = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection="3d")

    ax_xy.plot(
        camera_positions[:, 0],
        camera_positions[:, 1],
        color="tab:green",
        linewidth=2.5,
        label="camera",
    )
    ax_xy.scatter(
        camera_positions[0, 0],
        camera_positions[0, 1],
        color="green",
        marker="o",
        s=45,
    )
    ax_xy.scatter(
        camera_positions[-1, 0],
        camera_positions[-1, 1],
        color="red",
        marker="s",
        s=45,
    )

    ax_3d.plot(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        color="tab:green",
        linewidth=2.0,
        label="camera",
    )

    legend_flags = {"pedestrian": False, "vehicle": False, "other": False}

    all_points = [camera_positions]

    for instance in instance_subset:
        positions = instance["positions"]
        group = instance["group"]
        color = colors[group]

        line_label = group if not legend_flags[group] else None
        legend_flags[group] = True

        ax_xy.plot(
            positions[:, 0],
            positions[:, 1],
            color=color,
            linewidth=1.1,
            alpha=0.7,
            label=line_label,
        )

        ax_3d.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color,
            linewidth=1.0,
            alpha=0.65,
            label=line_label,
        )

        all_points.append(positions)

    ax_xy.set_title("Scene Trajectories (XY Top-Down)")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis("equal")
    ax_xy.legend(loc="best")

    ax_3d.set_title("Scene Trajectories (3D)")
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.grid(True, alpha=0.3)

    merged = np.concatenate(all_points, axis=0)
    set_equal_3d(ax_3d, merged)
    ax_3d.legend(loc="best")

    total_instances = len(instances)
    shown_instances = len(instance_subset)

    base_title = (
        f"Camera + Actor Trajectories | instances shown: {shown_instances}/{total_instances}"
    )

    unique_slider_frames = (
        np.array(sorted(set(camera_scene_frames.tolist())), dtype=int)
        if enable_slider
        else np.array([], dtype=int)
    )

    timestamp_text = None
    highlighted_cam_xy = None
    highlighted_obj_xy = None
    highlighted_cam_3d = None
    highlighted_obj_3d = None
    current_velocity_artists = []

    if enable_slider and unique_slider_frames.size > 0:
        highlighted_cam_xy = ax_xy.scatter([], [], s=highlight_size, c=colors["camera"], edgecolors="black", linewidths=1.0, zorder=12, label="camera@t")
        highlighted_obj_xy = ax_xy.scatter([], [], s=highlight_size, c=[], edgecolors="black", linewidths=0.8, zorder=11, label="objects@t")

        highlighted_cam_3d = ax_3d.scatter([], [], [], s=highlight_size, c=colors["camera"], edgecolors="black", linewidths=1.0, depthshade=False, zorder=12)
        highlighted_obj_3d = ax_3d.scatter([], [], [], s=highlight_size, c=[], edgecolors="black", linewidths=0.8, depthshade=False, zorder=11)

        timestamp_text = fig.text(0.5, 0.02, "", ha="center", va="center", fontsize=11)

        ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        frame_slider = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(unique_slider_frames) - 1,
            valinit=0,
            valstep=1,
            valfmt="%d",
        )

        def update_highlight(frame_idx_position: float) -> None:
            nonlocal current_velocity_artists
            idx = int(frame_idx_position)
            frame_value = int(unique_slider_frames[idx])
            frame_slider.valtext.set_text(str(frame_value))

            for artist in current_velocity_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            current_velocity_artists = []

            cam_pos = get_position_at_frame(
                camera_scene_frames,
                camera_positions,
                frame_value,
            )
            if cam_pos is not None:
                highlighted_cam_xy.set_offsets(cam_pos[:2].reshape(1, 2))
                highlighted_cam_3d._offsets3d = (
                    np.array([cam_pos[0]]),
                    np.array([cam_pos[1]]),
                    np.array([cam_pos[2]]),
                )

                cam_vel = get_velocity_at_frame(
                    camera_scene_frames,
                    camera_positions,
                    frame_value,
                )
                if cam_vel is not None:
                    q_cam_xy = ax_xy.quiver(
                        cam_pos[0],
                        cam_pos[1],
                        cam_vel[0],
                        cam_vel[1],
                        angles="xy",
                        scale_units="xy",
                        scale=1.0 / max(1e-6, velocity_scale),
                        color=colors["camera"],
                        width=0.004,
                        zorder=13,
                    )
                    current_velocity_artists.append(q_cam_xy)
            else:
                highlighted_cam_xy.set_offsets(np.empty((0, 2)))
                highlighted_cam_3d._offsets3d = (np.array([]), np.array([]), np.array([]))

            obj_xy_list = []
            obj_z_list = []
            obj_colors = []
            for inst in instance_subset:
                inst_pos = get_position_at_frame(
                    np.asarray(inst["frame_indices"]),
                    np.asarray(inst["positions"]),
                    frame_value,
                )
                if inst_pos is None:
                    continue
                obj_xy_list.append(inst_pos[:2])
                obj_z_list.append(inst_pos[2])
                obj_colors.append(colors[inst["group"]])

                inst_vel = get_velocity_at_frame(
                    np.asarray(inst["frame_indices"]),
                    np.asarray(inst["positions"]),
                    frame_value,
                )
                if inst_vel is not None:
                    q_obj_xy = ax_xy.quiver(
                        inst_pos[0],
                        inst_pos[1],
                        inst_vel[0],
                        inst_vel[1],
                        angles="xy",
                        scale_units="xy",
                        scale=1.0 / max(1e-6, velocity_scale),
                        color=colors[inst["group"]],
                        width=0.003,
                        alpha=0.9,
                        zorder=12,
                    )
                    current_velocity_artists.append(q_obj_xy)

            if len(obj_xy_list) > 0:
                obj_xy = np.asarray(obj_xy_list)
                obj_z = np.asarray(obj_z_list)
                highlighted_obj_xy.set_offsets(obj_xy)
                highlighted_obj_xy.set_facecolor(obj_colors)
                highlighted_obj_3d._offsets3d = (obj_xy[:, 0], obj_xy[:, 1], obj_z)
                highlighted_obj_3d.set_facecolor(obj_colors)
            else:
                highlighted_obj_xy.set_offsets(np.empty((0, 2)))
                highlighted_obj_xy.set_facecolor([])
                highlighted_obj_3d._offsets3d = (np.array([]), np.array([]), np.array([]))
                highlighted_obj_3d.set_facecolor([])

            if timestamp_text is not None:
                timestamp_text.set_text(f"Selected frame: {frame_value}")

            fig.canvas.draw_idle()

        frame_slider.on_changed(update_highlight)
        update_highlight(0)

    fig.suptitle(base_title, fontsize=12)

    if enable_slider and unique_slider_frames.size > 0:
        plt.tight_layout(rect=[0, 0.10, 1, 0.96])
    else:
        plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return shown_instances, total_instances


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize camera + instance trajectories in one scene plot"
    )
    parser.add_argument(
        "--camera_npz", required=True, help="Path to camera poses npz"
    )
    parser.add_argument(
        "--instances_json", required=True, help="Path to instances_info.json"
    )
    parser.add_argument(
        "--output",
        default="scene_trajectories.png",
        help="Output image path",
    )
    parser.add_argument(
        "--camera_name",
        default=None,
        help="Camera name to use for trajectory (default: auto-select first camera)",
    )
    parser.add_argument(
        "--ego_pose_start",
        default=None,
        help="Optional path to first-frame ego pose (4x4 txt). If provided, applies inv(ego_to_world_start) normalization to instances",
    )
    parser.add_argument(
        "--normalize_camera_with_ego_start",
        action="store_true",
        help="Also apply --ego_pose_start normalization to camera trajectory (use only if camera npz is still in raw world coordinates)",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=0,
        help="Max number of instances to plot (0 = all)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively",
    )
    parser.add_argument(
        "--interactive_slider",
        action="store_true",
        help="Enable frame slider to highlight timestamp object/camera positions",
    )
    parser.add_argument(
        "--highlight_size",
        type=float,
        default=40.0,
        help="Marker size for timestamp-highlighted positions",
    )
    parser.add_argument(
        "--velocity_scale",
        type=float,
        default=20.0,
        help="Scale multiplier for velocity arrows",
    )

    args = parser.parse_args()

    camera_path = Path(args.camera_npz)
    instance_path = Path(args.instances_json)
    output_path = Path(args.output)

    if not camera_path.exists():
        raise FileNotFoundError(f"Camera npz not found: {camera_path}")
    if not instance_path.exists():
        raise FileNotFoundError(f"Instances json not found: {instance_path}")

    world_transform = None
    if args.ego_pose_start:
        world_transform = load_ego_start_inverse_transform(args.ego_pose_start)

    camera_raw = load_camera_trajectory(str(camera_path))
    camera = build_camera_reference_trajectory(camera_raw, camera_name=args.camera_name)
    if world_transform is not None and args.normalize_camera_with_ego_start:
        camera["positions"] = transform_points(camera["positions"], world_transform)

    instances = load_instance_trajectories(
        str(instance_path),
        world_transform=world_transform,
    )

    if len(instances) == 0:
        print("Warning: no valid instance trajectories found. Plotting camera only.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shown_instances, total_instances = plot_scene_trajectories(
        camera,
        instances,
        str(output_path),
        show=args.show,
        max_instances=args.max_instances,
        enable_slider=args.interactive_slider,
        highlight_size=args.highlight_size,
        velocity_scale=args.velocity_scale,
    )

    print(f"Saved scene trajectory figure to: {output_path}")
    print(f"Instances shown: {shown_instances}/{total_instances}")
    print(f"Selected camera: {camera.get('selected_camera', 'unknown')}")
    if args.interactive_slider:
        print("Interactive frame slider: enabled")
    if args.ego_pose_start:
        print(f"Applied first-ego normalization to instances from: {args.ego_pose_start}")
        if args.normalize_camera_with_ego_start:
            print("Also applied first-ego normalization to camera trajectory")


if __name__ == "__main__":
    main()
