"""
Train a specific rigid object (vehicle) with Difix-assisted synthetic views.

Workflow:
1) Load a pretrained checkpoint and select a rigid object id.
2) Render n orbit views around the object (novel views).
3) Repair the rendered views with Difix (reference = real frame).
4) Append repaired views as synthetic samples and train.
5) Save checkpoints and optionally re-render to inspect quality.
"""
import argparse
import gc
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from tools.difix_sender_receiver import difix_repair_batch
from tools.visualise_instance import (
    build_trainer_from_checkpoint,
    extract_instance_points,
    select_object_key,
)
from models.gaussians.basics import dataclass_camera, dataclass_gs, quat_mult
from utils.geometry import quat_to_rotmat


def mirror_gaussians(gs: dataclass_gs, mirror_axis: str = "y", mirror_point: Optional[torch.Tensor] = None) -> dataclass_gs:
    """
    Mirror gaussians along a symmetry axis (typically the Y-axis for vehicles).
    
    Args:
        gs: dataclass_gs object containing gaussian parameters
        mirror_axis: which axis to mirror along ("x", "y", or "z")
        mirror_point: optional center point for mirroring (defaults to zero)
    
    Returns:
        New dataclass_gs with mirrored parameters
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    if mirror_axis.lower() not in axis_map:
        raise ValueError(f"mirror_axis must be one of {list(axis_map.keys())}")
    
    axis_idx = axis_map[mirror_axis.lower()]
    device = gs._means.device
    
    # Mirror means
    mirrored_means = gs._means.clone()
    if mirror_point is None:
        mirror_point = torch.tensor(0.0, device=device, dtype=gs._means.dtype)
    mirrored_means[:, axis_idx] = 2 * mirror_point - mirrored_means[:, axis_idx]
    
    # Mirror quaternions: when mirroring along an axis, we need to flip the corresponding
    # rotation component. For a quaternion q = [w, x, y, z], mirroring along:
    # - x-axis: flip y and z components
    # - y-axis: flip x and z components  
    # - z-axis: flip x and y components
    mirrored_quats = gs._quats.clone()
    
    if axis_idx == 0:  # x-axis
        mirrored_quats[:, [2, 3]] = -mirrored_quats[:, [2, 3]]  # flip y, z
    elif axis_idx == 1:  # y-axis
        mirrored_quats[:, [1, 3]] = -mirrored_quats[:, [1, 3]]  # flip x, z
    else:  # z-axis
        mirrored_quats[:, [1, 2]] = -mirrored_quats[:, [1, 2]]  # flip x, y
    
    return dataclass_gs(
        _means=mirrored_means,
        _opacities=gs._opacities.clone(),
        _rgbs=gs._rgbs.clone(),
        _scales=gs._scales.clone(),
        _quats=mirrored_quats,
        detach_keys=gs.detach_keys,
        extras=gs.extras.copy() if gs.extras else None,
    )


def mirror_camera_pose(pose: torch.Tensor, mirror_axis: str = "y", mirror_point: float = 0.0) -> torch.Tensor:
    """
    Mirror a camera pose (4x4 transformation matrix) along a symmetry axis.
    
    Args:
        pose: 4x4 camera-to-world transformation matrix
        mirror_axis: which axis to mirror along ("x", "y", or "z")
        mirror_point: center point for mirroring (defaults to 0.0)
    
    Returns:
        Mirrored 4x4 pose matrix
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    if mirror_axis.lower() not in axis_map:
        raise ValueError(f"mirror_axis must be one of {list(axis_map.keys())}")
    
    axis_idx = axis_map[mirror_axis.lower()]
    mirrored_pose = pose.clone()
    
    # Mirror the position component
    mirrored_pose[axis_idx, 3] = 2 * mirror_point - mirrored_pose[axis_idx, 3]
    
    # Mirror the corresponding column of the rotation matrix
    mirrored_pose[axis_idx, :3] = -mirrored_pose[axis_idx, :3]
    
    return mirrored_pose


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 3 and tensor.shape[0] == 3:  # CHW
        tensor = tensor.permute(1, 2, 0)
    img = tensor.detach().cpu().numpy()
    img = np.clip(img, 0, 1) * 255
    img = img.astype(np.uint8)
    return Image.fromarray(img)


def image_to_array(img: Image.Image, normalize: bool = True) -> np.ndarray:
    img_np = np.array(img)  # H x W x C, uint8
    if normalize:
        img_np = img_np.astype(np.float32) / 255.0
    else:
        img_np = img_np.astype(np.float32)
    return np.transpose(img_np, (2, 0, 1))


def log_synthetic_image(novel_image: Image.Image, repaired_image: Image.Image, run_path: str, img_no: int) -> None:
    side_by_side = Image.new(
        "RGB",
        (novel_image.width + repaired_image.width, novel_image.height),
    )
    side_by_side.paste(novel_image, (0, 0))
    side_by_side.paste(repaired_image, (novel_image.width, 0))
    side_by_side.save(os.path.join(run_path, "synthetic_image_samples", f"sidebyside_{img_no}.png"))


def _to_device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device_tree(v, device) for v in value]
    return value


def _look_at_c2w(direction: torch.Tensor, position: torch.Tensor, use_neg_front: bool, ref_up: torch.Tensor) -> torch.Tensor:
    front = torch.nn.functional.normalize(direction, dim=-1)
    up = torch.nn.functional.normalize(ref_up, dim=-1)
    if torch.abs(torch.dot(front, up)) > 0.98:
        up = torch.tensor([0.0, 0.0, 1.0], device=direction.device, dtype=direction.dtype)
        if torch.abs(torch.dot(front, up)) > 0.98:
            up = torch.tensor([0.0, 1.0, 0.0], device=direction.device, dtype=direction.dtype)
    right = torch.nn.functional.normalize(torch.cross(front, up, dim=0), dim=0)
    true_up = torch.nn.functional.normalize(torch.cross(right, front, dim=0), dim=0)
    if torch.dot(true_up, ref_up) < 0:
        right = -right
        true_up = -true_up
    pose = torch.eye(4, dtype=direction.dtype, device=direction.device)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -front if use_neg_front else front
    pose[:3, 3] = position
    return pose


def build_orbit_poses(
    center: torch.Tensor,
    radius: float,
    elevation_deg: float,
    num_views: int,
    use_neg_front: bool,
    ref_up: torch.Tensor,
) -> List[torch.Tensor]:
    azimuths = np.linspace(0.0, 360.0, num_views, endpoint=False)
    elevation_rad = np.deg2rad(elevation_deg)
    poses = []
    for azim in azimuths:
        azim_rad = np.deg2rad(azim)
        offset = torch.tensor(
            [
                radius * np.cos(azim_rad) * np.cos(elevation_rad),
                radius * np.sin(azim_rad) * np.cos(elevation_rad),
                radius * np.sin(elevation_rad),
            ],
            dtype=center.dtype,
            device=center.device,
        )
        cam_pos = center + offset
        view_dir = center - cam_pos
        poses.append(_look_at_c2w(view_dir, cam_pos, use_neg_front=use_neg_front, ref_up=ref_up))
    return poses


def _build_object_gaussians(trainer, frame_idx: int, object_id: int):
    rigid_model = trainer.models.get("RigidNodes", None)
    if rigid_model is None:
        raise ValueError("RigidNodes model is required to render rigid objects.")
    rigid_model.set_cur_frame(frame_idx)

    pts_mask = rigid_model.point_ids[..., 0] == int(object_id)
    if pts_mask.sum() == 0:
        raise ValueError(f"No points found for rigid object id {object_id}")

    local_gs = rigid_model.get_instance_activated_gs_dict(int(object_id))
    if local_gs is None:
        raise ValueError(f"Rigid object id {object_id} has too few points to render")

    cur_rot = rigid_model.quat_act(rigid_model.instances_quats[frame_idx, int(object_id)])
    cur_trans = rigid_model.instances_trans[frame_idx, int(object_id)]
    rot_mat = quat_to_rotmat(cur_rot.unsqueeze(0))[0]

    local_means = local_gs["means"]
    local_quats = local_gs["quats"]
    world_means = torch.matmul(local_means, rot_mat.transpose(0, 1)) + cur_trans
    world_quats = quat_mult(
        cur_rot.unsqueeze(0).expand(local_quats.shape[0], -1),
        local_quats,
    )

    return dataclass_gs(
        _means=world_means,
        _opacities=local_gs["opacities"],
        _rgbs=rigid_model.colors[pts_mask],
        _scales=local_gs["scales"],
        _quats=world_quats,
        detach_keys=[],
        extras=None,
    )


@torch.no_grad()
def render_object_views(
    trainer,
    dataset,
    frame_idx: int,
    object_type: str,
    object_id: int,
    points_xyz: np.ndarray,
    num_views: int,
    elevation_deg: float,
    radius_scale: float,
    min_radius: float,
):
    if object_type != "rigid":
        raise NotImplementedError("Only rigid objects are supported for vehicle training.")
    if points_xyz.size == 0:
        raise ValueError("Selected object has no points to render.")

    pixel_source = dataset.pixel_source
    cam0 = pixel_source.camera_data[0]
    ref_image_infos, ref_cam_infos = cam0.get_image(frame_idx)
    if not isinstance(ref_cam_infos["camera_to_world"], torch.Tensor):
        ref_cam_infos["camera_to_world"] = torch.as_tensor(ref_cam_infos["camera_to_world"])

    device = trainer.device if hasattr(trainer, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_cam_infos = _to_device_tree(ref_cam_infos, device)

    center = torch.tensor(points_xyz.mean(axis=0), device=device, dtype=torch.float32)
    extent = np.linalg.norm(points_xyz - points_xyz.mean(axis=0, keepdims=True), axis=1)
    radius = float(max(extent.max() * radius_scale, min_radius)) if extent.size > 0 else float(min_radius)

    object_gs = _build_object_gaussians(trainer, frame_idx, object_id)
    was_training = trainer.training
    trainer.set_eval()

    def _render_opacity_score(pose: torch.Tensor) -> float:
        cam = dataclass_camera(
            camtoworlds=pose.to(device),
            camtoworlds_gt=pose.to(device),
            Ks=ref_cam_infos["intrinsics"],
            H=int(ref_cam_infos["height"]),
            W=int(ref_cam_infos["width"]),
        )
        outputs, _ = trainer.render_gaussians(
            gs=object_gs,
            cam=cam,
            near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=0.0,
        )
        return float(outputs["opacity"].mean().item())

    ref_up = ref_cam_infos["camera_to_world"][:3, 1]
    poses_neg = build_orbit_poses(center, radius, elevation_deg, num_views, True, ref_up=ref_up)
    poses_pos = build_orbit_poses(center, radius, elevation_deg, num_views, False, ref_up=ref_up)
    score_neg = _render_opacity_score(poses_neg[0])
    score_pos = _render_opacity_score(poses_pos[0])
    poses = poses_neg if score_neg >= score_pos else poses_pos

    rendered_views = []
    intrinsics_cpu = ref_cam_infos["intrinsics"].detach().cpu()
    for pose in poses:
        cam = dataclass_camera(
            camtoworlds=pose.to(device),
            camtoworlds_gt=pose.to(device),
            Ks=ref_cam_infos["intrinsics"],
            H=int(ref_cam_infos["height"]),
            W=int(ref_cam_infos["width"]),
        )
        outputs, _ = trainer.render_gaussians(
            gs=object_gs,
            cam=cam,
            near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=0.0,
        )
        rendered_views.append(
            {
                "reference_frame_idx": int(frame_idx),
                "rendered_rgb": outputs["rgb_gaussians"].detach().cpu(),
                "alpha_mask": outputs["opacity"].detach().cpu(),
                "novel_c2w": pose.detach().cpu(),
                "intrinsics": intrinsics_cpu,
            }
        )

    if was_training:
        trainer.set_train()

    reference_rgb = ref_image_infos["pixels"].detach().cpu()
    return rendered_views, reference_rgb


def repair_views_with_difix(view_samples, reference_image: Image.Image, run_path: str) -> List[dict]:
    novel_imgs = [tensor_to_image(sample["rendered_rgb"]) for sample in view_samples]
    ref_imgs = [reference_image] * len(novel_imgs)
    repaired_images = difix_repair_batch(novel_imgs, ref_imgs)

    for idx, (sample, novel_img, repaired_image) in enumerate(zip(view_samples, novel_imgs, repaired_images)):
        log_synthetic_image(novel_img, repaired_image, run_path, idx)
        repaired_array = image_to_array(repaired_image, normalize=True)
        sample["rendered_rgb"] = repaired_array
        alpha = sample.get("alpha_mask", None)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().numpy()
        if alpha is not None:
            if alpha.ndim == 3 and alpha.shape[-1] == 1:
                alpha = alpha[..., 0]
            sample["alpha_mask"] = alpha.astype(np.float32)
    return view_samples

@torch.no_grad()
def render_mirrored_object_views(
    trainer,
    dataset,
    frame_idx: int,
    object_type: str,
    object_id: int,
    points_xyz: np.ndarray,
    num_views: int,
    elevation_deg: float,
    radius_scale: float,
    min_radius: float,
    mirror_axis: str = "y",
):
    """
    Render mirrored views of an object for data augmentation.
    
    Args:
        trainer: trainer object
        dataset: dataset object
        frame_idx: frame index
        object_type: type of object ("rigid", etc.)
        object_id: id of the object
        points_xyz: 3D points of the object
        num_views: number of views
        elevation_deg: elevation angle for orbit
        radius_scale: radius scaling factor
        min_radius: minimum radius
        mirror_axis: axis to mirror along ("x", "y", or "z")
    
    Returns:
        List of rendered views from mirrored gaussian configuration
    """
    if object_type != "rigid":
        raise NotImplementedError("Only rigid objects are supported for vehicle training.")
    if points_xyz.size == 0:
        raise ValueError("Selected object has no points to render.")

    pixel_source = dataset.pixel_source
    cam0 = pixel_source.camera_data[0]
    ref_image_infos, ref_cam_infos = cam0.get_image(frame_idx)
    if not isinstance(ref_cam_infos["camera_to_world"], torch.Tensor):
        ref_cam_infos["camera_to_world"] = torch.as_tensor(ref_cam_infos["camera_to_world"])

    device = trainer.device if hasattr(trainer, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_cam_infos = _to_device_tree(ref_cam_infos, device)

    # Build original gaussians
    object_gs = _build_object_gaussians(trainer, frame_idx, object_id)
    
    # Mirror the gaussians along the specified axis
    # Use the center of the points as the mirror point
    mirror_point = torch.tensor(points_xyz.mean(axis=0)[{"x": 0, "y": 1, "z": 2}[mirror_axis.lower()]], 
                                device=device, dtype=object_gs._means.dtype)
    mirrored_gs = mirror_gaussians(object_gs, mirror_axis=mirror_axis, mirror_point=mirror_point)
    
    center = torch.tensor(points_xyz.mean(axis=0), device=device, dtype=torch.float32)
    extent = np.linalg.norm(points_xyz - points_xyz.mean(axis=0, keepdims=True), axis=1)
    radius = float(max(extent.max() * radius_scale, min_radius)) if extent.size > 0 else float(min_radius)

    was_training = trainer.training
    trainer.set_eval()

    def _render_opacity_score(pose: torch.Tensor) -> float:
        cam = dataclass_camera(
            camtoworlds=pose.to(device),
            camtoworlds_gt=pose.to(device),
            Ks=ref_cam_infos["intrinsics"],
            H=int(ref_cam_infos["height"]),
            W=int(ref_cam_infos["width"]),
        )
        outputs, _ = trainer.render_gaussians(
            gs=mirrored_gs,
            cam=cam,
            near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=0.0,
        )
        return float(outputs["opacity"].mean().item())

    ref_up = ref_cam_infos["camera_to_world"][:3, 1]
    poses_neg = build_orbit_poses(center, radius, elevation_deg, num_views, True, ref_up=ref_up)
    poses_pos = build_orbit_poses(center, radius, elevation_deg, num_views, False, ref_up=ref_up)
    score_neg = _render_opacity_score(poses_neg[0])
    score_pos = _render_opacity_score(poses_pos[0])
    poses = poses_neg if score_neg >= score_pos else poses_pos

    # Mirror the camera poses as well for symmetry
    mirrored_poses = [mirror_camera_pose(pose, mirror_axis=mirror_axis, mirror_point=mirror_point.item()) 
                      for pose in poses]

    rendered_views = []
    intrinsics_cpu = ref_cam_infos["intrinsics"].detach().cpu()
    for pose in mirrored_poses:
        cam = dataclass_camera(
            camtoworlds=pose.to(device),
            camtoworlds_gt=pose.to(device),
            Ks=ref_cam_infos["intrinsics"],
            H=int(ref_cam_infos["height"]),
            W=int(ref_cam_infos["width"]),
        )
        outputs, _ = trainer.render_gaussians(
            gs=mirrored_gs,
            cam=cam,
            near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=0.0,
        )
        rendered_views.append(
            {
                "reference_frame_idx": int(frame_idx),
                "rendered_rgb": outputs["rgb_gaussians"].detach().cpu(),
                "alpha_mask": outputs["opacity"].detach().cpu(),
                "novel_c2w": pose.detach().cpu(),
                "intrinsics": intrinsics_cpu,
                "is_mirrored": True,
            }
        )

    if was_training:
        trainer.set_train()

    return rendered_views


def _resolve_true_instance_id(dataset, object_id: int, prefer_true_id: bool) -> int:
    if (
        prefer_true_id
        and hasattr(dataset.pixel_source, "instances_true_id")
        and dataset.pixel_source.instances_true_id is not None
    ):
        if object_id >= len(dataset.pixel_source.instances_true_id):
            raise ValueError(
                f"Object id {object_id} is out of range for instances_true_id "
                f"(len={len(dataset.pixel_source.instances_true_id)})"
            )
        return int(dataset.pixel_source.instances_true_id[object_id].item())
    return int(object_id)


def load_instance_frame_indices(
    dataset,
    object_id: int,
    instances_dir: Optional[str],
    prefer_true_id: bool,
) -> List[int]:
    base_dir = instances_dir or os.path.join(dataset.data_path, "instances")
    frame_instances_path = os.path.join(base_dir, "frame_instances.json")
    instances_info_path = os.path.join(base_dir, "instances_info.json")

    if not os.path.exists(frame_instances_path) and not os.path.exists(instances_info_path):
        raise FileNotFoundError(
            f"Instance metadata not found. Expected {frame_instances_path} or {instances_info_path}"
        )

    target_id = _resolve_true_instance_id(dataset, object_id, prefer_true_id)
    frames: List[int] = []

    if os.path.exists(frame_instances_path):
        frame_instances_raw = json.load(open(frame_instances_path, "r"))
        for frame_index, instance_ids in frame_instances_raw.items():
            try:
                if target_id in [int(x) for x in instance_ids]:
                    frames.append(int(frame_index))
            except Exception:
                continue

    if not frames and os.path.exists(instances_info_path):
        instances_info = json.load(open(instances_info_path, "r"))
        if str(target_id) not in instances_info:
            raise ValueError(
                f"Instance id {target_id} not found in instances_info.json "
                f"(keys: {len(instances_info)})"
            )
        frame_indices = instances_info[str(target_id)].get("frame_annotations", {}).get("frame_idx", [])
        frames = [int(f) for f in frame_indices]

    if not frames:
        raise ValueError(f"No frames found for instance id {target_id}")

    max_frame = dataset.num_img_timesteps
    frames = [f for f in frames if 0 <= f < max_frame]
    return sorted(set(frames))


def normalize_frame_indices(
    dataset,
    args: argparse.Namespace,
) -> List[int]:
    if args.frame_indices:
        frames = [int(f) for f in args.frame_indices]
    elif args.frames_from_instances:
        frames = load_instance_frame_indices(
            dataset=dataset,
            object_id=args.object_id,
            instances_dir=args.instances_dir,
            prefer_true_id=not args.no_true_id_map,
        )
    else:
        frames = [int(args.frame_idx)]

    frames = [f for f in frames if 0 <= f < dataset.num_img_timesteps]
    if not frames:
        raise ValueError("No valid frame indices resolved.")

    if args.frame_stride > 1:
        frames = frames[:: args.frame_stride]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    return frames


def train_synthetic(
    synthetic_samples: List[dict],
    checkpoint_path: str,
    frame_index: int,
    lateral_offset: float,
    num_iters: int,
    prune_steps: int,
    from_scratch: bool,
    selected_object_type: str,
    selected_object_id: int,
    synthetic_ratio: float = 0.3,
):
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_in:
        input_path = tmp_in.name
        pickle.dump(synthetic_samples, tmp_in)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_out:
        output_path = tmp_out.name

    cmd = [
        sys.executable, "-u", os.path.join(os.path.dirname(__file__), "standalone_trainer_fixed_step.py"),
        "train_synthetic",
        f"--checkpoint_path={checkpoint_path}",
        f"--input_path={input_path}",
        f"--output_path={output_path}",
        f"--lateral_offset={lateral_offset}",
        f"--ref_frame={frame_index}",
        f"--split_steps={num_iters}",
        f"--prune_steps={prune_steps}",
        f"--selected_object_type={selected_object_type}",
        f"--selected_object_id={selected_object_id}",
        f"--synthetic_ratio={synthetic_ratio}",
    ]
    if from_scratch:
        cmd.append("--from_scratch")

    result = subprocess.run(cmd, capture_output=False, text=False)
    if result.returncode != 0:
        raise RuntimeError("train_synthetic subprocess failed")

    with open(output_path, "rb") as f:
        _ = pickle.load(f)
    os.remove(output_path)
    os.remove(input_path)


def create_run_folders(run_path: str) -> None:
    os.makedirs(run_path, exist_ok=True)
    for folder in [
        "images",
        "videos",
        "metrics",
        "configs_bk",
        "buffer_maps",
        "backup",
        "synthetic_image_samples",
    ]:
        os.makedirs(os.path.join(run_path, folder), exist_ok=True)


def copy_pretrained_checkpoint(pretrained_checkpoint_path: str, new_checkpoint_path: str) -> None:
    same_file = (
        os.path.exists(new_checkpoint_path)
        and os.path.samefile(pretrained_checkpoint_path, new_checkpoint_path)
    )
    if not same_file:
        shutil.copy(pretrained_checkpoint_path, new_checkpoint_path)
        print(f"Copied pretrained checkpoint from {pretrained_checkpoint_path} to {new_checkpoint_path}")


def copy_config_file(pretrained_checkpoint_path: str, run_path: str) -> None:
    pretrained_dir = os.path.dirname(pretrained_checkpoint_path)
    pretrained_config_path = os.path.join(pretrained_dir, "config.yaml")
    new_config_path = os.path.join(run_path, "config.yaml")
    same_file = (
        os.path.exists(new_config_path)
        and os.path.samefile(pretrained_config_path, new_config_path)
    )
    if not same_file:
        shutil.copy(pretrained_config_path, new_config_path)
        print(f"Copied config file from {pretrained_config_path} to {new_config_path}")


def find_latest_checkpoint(run_path: str) -> str:
    ckpts = []
    for fname in os.listdir(run_path):
        if not fname.startswith("checkpoint_") or not fname.endswith(".pth"):
            continue
        if fname == "checkpoint_final.pth":
            continue
        try:
            step = int(fname.replace("checkpoint_", "").replace(".pth", ""))
        except ValueError:
            continue
        ckpts.append((step, os.path.join(run_path, fname)))
    if not ckpts:
        return ""
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def main(args: argparse.Namespace) -> None:
    run_path = os.path.join(args.output_root, args.project, args.run_name)
    create_run_folders(run_path)
    checkpoint_path = os.path.join(run_path, "checkpoint_final.pth")
    copy_pretrained_checkpoint(args.resume_from, checkpoint_path)
    copy_config_file(args.resume_from, run_path)

    current_ckpt = checkpoint_path
    resolved_frames: List[int] = []

    for round_idx in range(args.num_rounds):
        round_synthetic_samples: List[dict] = []
        _, dataset, trainer, _ = build_trainer_from_checkpoint(current_ckpt, [])
        if not resolved_frames:
            resolved_frames = normalize_frame_indices(dataset, args)

        key = None
        for frame_idx in resolved_frames:
            instances = extract_instance_points(trainer=trainer, frame_idx=frame_idx)
            if not instances:
                raise RuntimeError(
                    f"No rigid instances found in the checkpoint for frame {frame_idx}."
                )

            key = select_object_key(
                instances=instances,
                object_id=args.object_id,
                object_type=args.object_type,
            )
            if key[0] != "rigid":
                raise ValueError(
                    f"Object {args.object_id} is not rigid; only rigid objects are supported."
                )

            points = instances[key]
            view_samples, reference_rgb = render_object_views(
                trainer=trainer,
                dataset=dataset,
                frame_idx=frame_idx,
                object_type=key[0],
                object_id=key[1],
                points_xyz=points,
                num_views=args.num_views,
                elevation_deg=args.elevation_deg,
                radius_scale=args.radius_scale,
                min_radius=args.min_radius,
            )

            reference_image = tensor_to_image(reference_rgb)
            view_samples = repair_views_with_difix(view_samples, reference_image, run_path)
            round_synthetic_samples.extend(view_samples)
            
            # Optionally render and add mirrored views for data augmentation
            if args.enable_mirror:
                print(f"Rendering mirrored views along {args.mirror_axis}-axis...")
                mirrored_view_samples = render_mirrored_object_views(
                    trainer=trainer,
                    dataset=dataset,
                    frame_idx=frame_idx,
                    object_type=key[0],
                    object_id=key[1],
                    points_xyz=points,
                    num_views=args.num_views,
                    elevation_deg=args.elevation_deg,
                    radius_scale=args.radius_scale,
                    min_radius=args.min_radius,
                    mirror_axis=args.mirror_axis,
                )
                mirrored_view_samples = repair_views_with_difix(mirrored_view_samples, reference_image, run_path)
                round_synthetic_samples.extend(mirrored_view_samples)
                print(f"Added {len(mirrored_view_samples)} mirrored views to training set")

        train_synthetic(
            synthetic_samples=round_synthetic_samples,
            checkpoint_path=current_ckpt,
            frame_index=resolved_frames[0],
            lateral_offset=args.lateral_offset,
            num_iters=args.num_iters,
            prune_steps=args.prune_steps,
            from_scratch=args.from_scratch and round_idx == 0,
            selected_object_type=key[0],
            selected_object_id=key[1],
            synthetic_ratio=args.synthetic_ratio,
        )

        latest_ckpt = find_latest_checkpoint(run_path)
        if latest_ckpt:
            current_ckpt = latest_ckpt
            if not os.path.samefile(latest_ckpt, checkpoint_path):
                shutil.copy(latest_ckpt, checkpoint_path)

        del trainer, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.rerender_after:
            _, dataset, trainer, _ = build_trainer_from_checkpoint(current_ckpt, [])
            if not resolved_frames:
                resolved_frames = normalize_frame_indices(dataset, args)
            for frame_idx in resolved_frames:
                instances = extract_instance_points(trainer=trainer, frame_idx=frame_idx)
                key = select_object_key(
                    instances=instances,
                    object_id=args.object_id,
                    object_type=args.object_type,
                )
                points = instances[key]
                view_samples, _ = render_object_views(
                    trainer=trainer,
                    dataset=dataset,
                    frame_idx=frame_idx,
                    object_type=key[0],
                    object_id=key[1],
                    points_xyz=points,
                    num_views=args.num_views,
                    elevation_deg=args.elevation_deg,
                    radius_scale=args.radius_scale,
                    min_radius=args.min_radius,
                )
                render_dir = os.path.join(
                    run_path,
                    "images",
                    f"vehicle_views_round{round_idx+1}_frame{frame_idx:03d}",
                )
                os.makedirs(render_dir, exist_ok=True)
                for idx, sample in enumerate(view_samples):
                    img = tensor_to_image(sample["rendered_rgb"])
                    img.save(os.path.join(render_dir, f"view_{idx:02d}.png"))

            del trainer, dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Difix-based vehicle training")
    parser.add_argument("--resume_from", required=True, type=str, help="Path to pretrained checkpoint")
    parser.add_argument("--object_id", required=True, type=int, help="Rigid object id to refine")
    parser.add_argument("--object_type", default="rigid", choices=["rigid", "auto", "smpl"], help="Object type selector")
    parser.add_argument("--frame_idx", default=0, type=int, help="Reference frame index (used when no frame list is provided)")
    parser.add_argument("--frame_indices", nargs="+", type=int, help="Explicit list of frame indices to use")
    parser.add_argument("--frames_from_instances", action="store_true", help="Use instances json to find frames for the object")
    parser.add_argument("--instances_dir", type=str, default=None, help="Override instances/ directory (contains instances_info.json)")
    parser.add_argument("--frame_stride", type=int, default=1, help="Stride when using multiple frames")
    parser.add_argument("--max_frames", type=int, default=0, help="Cap number of frames (0 = no cap)")
    parser.add_argument("--no_true_id_map", action="store_true", help="Do not map object_id through instances_true_id")
    parser.add_argument("--num_views", default=12, type=int, help="Number of orbit views to render")
    parser.add_argument("--elevation_deg", default=18.0, type=float, help="Orbit elevation in degrees")
    parser.add_argument("--radius_scale", default=3.0, type=float, help="Radius multiplier based on object extent")
    parser.add_argument("--min_radius", default=4.0, type=float, help="Minimum orbit radius in meters")
    parser.add_argument("--num_rounds", default=1, type=int, help="Number of render/train rounds")
    parser.add_argument("--num_iters", default=3000, type=int, help="Training iterations per round")
    parser.add_argument("--prune_steps", default=500, type=int, help="Pruning steps during training")
    parser.add_argument("--from_scratch", action="store_true", help="Initialize gaussians from scratch in first round")
    parser.add_argument("--synthetic_ratio", default=0.3, type=float, help="Fraction of synthetic samples to use (0.0-1.0). 0.3 = ~70%% real / ~30%% synthetic split")
    parser.add_argument("--lateral_offset", default=0.0, type=float, help="Eval offset for trainer quality snapshots")
    parser.add_argument("--rerender_after", action="store_true", help="Re-render object views after each round")
    parser.add_argument("--enable_mirror", action="store_true", help="Enable gaussian mirroring for data augmentation")
    parser.add_argument("--mirror_axis", default="y", choices=["x", "y", "z"], help="Axis along which to mirror the gaussians (typically 'y' for vehicle symmetry)")
    parser.add_argument("--output_root", default="./work_dirs/", type=str, help="Root folder for outputs")
    parser.add_argument("--project", default="drivestudio", type=str, help="Project name")
    parser.add_argument("--run_name", default="vehicle_difix", type=str, help="Run name")
    main(parser.parse_args())
