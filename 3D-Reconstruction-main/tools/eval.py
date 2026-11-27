from pathlib import Path
from typing import List, Optional, Tuple
import sys

# Make repository-local packages importable when invoked as ``python tools/eval.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse
import h5py
from datetime import datetime 

import numpy as np        

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.road_mesh import RoadMesh
from models.video_utils import render_images, save_videos, render_novel_views, extract_camera_poses_from_dataset, save_camera_poses, analyze_camera_trajectory  
from utils.geometry import rotation_6d_to_matrix

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def _restore_textured_road_mesh(trainer: BasicTrainer, checkpoint_dir: str):
    """Restore textured road mesh from checkpoint if available."""
    if not hasattr(trainer, 'road_mesh') or trainer.road_mesh is None:
        return
    
    road_mesh_path = os.path.join(checkpoint_dir, 'road_mesh.pth')
    if os.path.exists(road_mesh_path):
        try:
            trainer.road_mesh = RoadMesh.load_checkpoint(road_mesh_path, device=trainer.device)
            logger.info(f"Loaded textured road mesh from {road_mesh_path} (has_texture={trainer.road_mesh.texture_buffer is not None})")
        except Exception as e:
            logger.warning(f"Failed to load textured road mesh from {road_mesh_path}: {e}")
    else:
        logger.debug(f"Road mesh texture file not found: {road_mesh_path}")


@torch.no_grad()
def hold_last_dynamic_node_states(trainer: BasicTrainer) -> None:
    """Keep each instance's last valid state for later evaluation timestamps.

    This only changes the loaded model in memory. Earlier visibility gaps and
    checkpoints on disk are left as they are.
    """
    for class_name in ("RigidNodes", "DeformableNodes", "SMPLNodes"):
        if class_name not in trainer.models:
            continue
        model = trainer.models[class_name]
        frame_valid = model.instances_fv
        pose_names = ["instances_trans", "instances_quats"]
        if class_name == "SMPLNodes":
            pose_names.append("smpl_qauts")
        held_instances = 0
        for instance_id in range(frame_valid.shape[1]):
            valid_frames = torch.nonzero(frame_valid[:, instance_id], as_tuple=True)[0]
            if valid_frames.numel() == 0:
                continue
            last_frame = int(valid_frames[-1])
            if last_frame == frame_valid.shape[0] - 1:
                continue
            for pose_name in pose_names:
                pose = getattr(model, pose_name)
                pose[last_frame + 1:, instance_id] = pose[last_frame, instance_id].clone()
            frame_valid[last_frame + 1:, instance_id] = True
            held_instances += 1
        logger.info("Held final state for %d %s instances", held_instances, class_name)



def _load_novel_trajectory_from_file(
    trajectory_file: str, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Load a novel trajectory from .npy/.npz and return a tensor of shape (N, 4, 4)."""
    if not os.path.isfile(trajectory_file):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

    ext = os.path.splitext(trajectory_file)[1].lower()
    if ext == ".npy":
        poses = np.asarray(np.load(trajectory_file, allow_pickle=True))
    elif ext == ".npz":
        data = np.load(trajectory_file, allow_pickle=True)
        candidate_keys = ["camera_poses", "poses", "trajectory"]
        key = next((k for k in candidate_keys if k in data), None)
        if key is None:
            raise ValueError(
                f"Unsupported trajectory file format in {trajectory_file}. "
                f"Expected one of keys: {candidate_keys}"
            )
        poses = np.asarray(data[key])

        # If the file stores interleaved multi-camera poses, keep a single camera stream.
        if key == "camera_poses" and poses.ndim == 3 and poses.shape[-2:] == (4, 4):
            if "cam_ids" in data and len(np.asarray(data["cam_ids"])) == len(poses):
                cam_ids = np.asarray(data["cam_ids"]).reshape(-1)
                poses = poses[cam_ids == cam_ids[0]]
            elif "cam_names" in data and len(np.asarray(data["cam_names"])) == len(poses):
                cam_names = np.asarray(data["cam_names"]).reshape(-1)
                poses = poses[cam_names == cam_names[0]]
    else:
        raise ValueError(
            f"Unsupported trajectory file extension: {ext}. Use .npy or .npz"
        )

    if poses.ndim != 3:
        raise ValueError(f"Trajectory must be a 3D array, got shape {poses.shape}")

    if poses.shape[-2:] == (3, 4):
        bottom_row = np.zeros((poses.shape[0], 1, 4), dtype=poses.dtype)
        bottom_row[:, 0, 3] = 1.0
        poses = np.concatenate([poses, bottom_row], axis=1)

    if poses.shape[-2:] != (4, 4):
        raise ValueError(
            f"Trajectory poses must be [N,4,4] or [N,3,4], got shape {poses.shape}"
        )

    return torch.as_tensor(poses, dtype=torch.float32, device=device)


def load_custom_ego_trajectory(
    dataset: DrivingDataset,
    ego_pose_dir: str,
    camera_id: int = 0,
    extrinsics_path: Optional[str] = None,
    ego_lateral_position: Optional[float] = None,
) -> torch.Tensor:
    """Convert an on-disk ego trajectory to the model's camera coordinates.

    CARLA novel-view exports keep the vehicle trajectory in ``ego_pose`` and
    encode the camera offset (for example, ``right_4m``) in
    ``extrinsics/0.txt``. Training poses are relative to the first ego pose,
    so apply the same normalization and camera extrinsic as the data loader.
    """
    pose_dir = Path(ego_pose_dir).expanduser().resolve()
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Custom ego-pose directory does not exist: {pose_dir}")

    pose_paths = sorted(
        pose_dir.glob("*.txt"),
        key=lambda path: (
            0 if path.stem.isdigit() else 1,
            int(path.stem) if path.stem.isdigit() else path.stem,
        ),
    )
    if not pose_paths:
        raise ValueError(f"No .txt pose files found in {pose_dir}")

    if extrinsics_path is None:
        exported_extrinsics = pose_dir.parent / "extrinsics" / f"{camera_id}.txt"
        scene_extrinsics = Path(dataset.data_path) / "extrinsics" / f"{camera_id}.txt"
        extrinsics_file = (
            exported_extrinsics if exported_extrinsics.is_file() else scene_extrinsics
        )
    else:
        extrinsics_file = Path(extrinsics_path).expanduser().resolve()
    if not extrinsics_file.is_file():
        raise FileNotFoundError(f"Camera extrinsics do not exist: {extrinsics_file}")

    origin_path = (
        Path(dataset.data_path) / "ego_pose" / f"{dataset.start_timestep:03d}.txt"
    )
    if not origin_path.is_file():
        raise FileNotFoundError(f"Training trajectory origin does not exist: {origin_path}")
    origin_from_world = np.linalg.inv(np.loadtxt(origin_path))
    camera_to_ego = np.loadtxt(extrinsics_file)
    if camera_to_ego.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 camera extrinsics in {extrinsics_file}, "
            f"got {camera_to_ego.shape}"
        )
    if ego_lateral_position is not None:
        camera_to_ego = camera_to_ego.copy()
        camera_to_ego[1, 3] = ego_lateral_position
        logger.info(
            "Overriding camera ego-Y position with %.3f m (positive is left)",
            ego_lateral_position,
        )

    trajectory = []
    for pose_path in pose_paths:
        ego_to_world = np.loadtxt(pose_path)
        if ego_to_world.shape != (4, 4):
            raise ValueError(
                f"Expected a 4x4 pose matrix in {pose_path}, got {ego_to_world.shape}"
            )
        trajectory.append(origin_from_world @ ego_to_world @ camera_to_ego)

    logger.info(
        "Loaded %d custom poses from %s with extrinsics %s",
        len(trajectory),
        pose_dir,
        extrinsics_file,
    )
    return torch.from_numpy(np.stack(trajectory)).float()


def resolve_custom_trajectory(
    args: argparse.Namespace, dataset: DrivingDataset
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve an explicit ego-pose directory or a named per-scene export."""
    if args.custom_ego_pose_dir:
        pose_dir = Path(args.custom_ego_pose_dir)
        default_name = pose_dir.parent.name
    elif args.custom_trajectory_name:
        pose_dir = (
            Path(dataset.data_path)
            / "novel_views"
            / args.custom_trajectory_name
            / "ego_pose"
        )
        default_name = args.custom_trajectory_name
    else:
        return None, None
    return str(pose_dir), args.custom_trajectory_label or default_name


def _log_trajectory_camera_positions(traj_name: str, traj: torch.Tensor) -> None:
    if traj.ndim == 2 and traj.shape[-1] == 3:
        positions = traj
        num_points = positions.shape[0]
        start_pos = positions[0].detach().cpu().numpy()
        end_pos = positions[-1].detach().cpu().numpy()
        message = (
            f"Trajectory '{traj_name}': {num_points} points | "
            f"start_xyz=[{start_pos[0]:.6f}, {start_pos[1]:.6f}, {start_pos[2]:.6f}] | "
            f"end_xyz=[{end_pos[0]:.6f}, {end_pos[1]:.6f}, {end_pos[2]:.6f}]"
        )
        logger.info(message)
        print(message)
        return

    if traj.ndim != 3 or traj.shape[-2:] != (4, 4):
        raise ValueError(
            f"Trajectory '{traj_name}' must have shape [N,4,4] or [N,3], got {tuple(traj.shape)}"
        )
    if traj.shape[0] == 0:
        raise ValueError(f"Trajectory '{traj_name}' is empty")

    start_pos = traj[0, :3, 3].detach().cpu().numpy()
    end_pos = traj[-1, :3, 3].detach().cpu().numpy()
    message = (
        f"Trajectory '{traj_name}': {traj.shape[0]} poses | "
        f"start_xyz=[{start_pos[0]:.6f}, {start_pos[1]:.6f}, {start_pos[2]:.6f}] | "
        f"end_xyz=[{end_pos[0]:.6f}, {end_pos[1]:.6f}, {end_pos[2]:.6f}]"
    )
    logger.info(message)
    print(message)


def _apply_cam_pose_correction_from_embeds(
    raw_traj: torch.Tensor,
    cam_pose_embeds: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Apply the checkpoint's per-frame CamPose correction to a raw trajectory."""
    if raw_traj.ndim != 3 or raw_traj.shape[-2:] != (4, 4):
        return raw_traj
    if cam_pose_embeds.ndim != 2 or cam_pose_embeds.shape[-1] != 9:
        return raw_traj

    device = raw_traj.device if device is None else device
    raw_traj = raw_traj.to(device=device)
    cam_pose_embeds = cam_pose_embeds.to(device=device)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=raw_traj.dtype)

    corrected = raw_traj.clone()
    with torch.no_grad():
        for i in range(raw_traj.shape[0]):
            delta = cam_pose_embeds[i]
            dx = delta[:3]
            drot = delta[3:]
            rot = rotation_6d_to_matrix(drot + identity.expand(1, -1))
            T = torch.eye(4, device=device, dtype=raw_traj.dtype)
            T[:3, :3] = rot[0]
            T[:3, 3] = dx
            corrected[i] = raw_traj[i] @ T
    return corrected


def _estimate_rigid_alignment_from_reference(
    source_raw_traj: torch.Tensor,
    target_dataset_traj: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate a global rigid transform (rotation + translation) from a raw
    reference trajectory to the dataset-loaded trajectory using corresponding
    camera poses. Camera orientations determine the rotation even when the
    camera positions follow a nearly straight path.
    """
    if source_raw_traj.ndim != 3 or source_raw_traj.shape[-2:] != (4, 4):
        raise ValueError(
            f"source_raw_traj must be [N,4,4], got {tuple(source_raw_traj.shape)}"
        )
    if target_dataset_traj.ndim != 3 or target_dataset_traj.shape[-2:] != (4, 4):
        raise ValueError(
            f"target_dataset_traj must be [N,4,4], got {tuple(target_dataset_traj.shape)}"
        )

    n_src = source_raw_traj.shape[0]
    n_dst = target_dataset_traj.shape[0]
    if n_src <= 0 or n_dst <= 0:
        raise ValueError("Cannot align empty trajectories")

    n_pairs = min(max(min(n_src, n_dst), 1), 256)
    src_idx = (
        torch.linspace(0, n_src - 1, n_pairs, device=source_raw_traj.device)
        .round()
        .long()
    )
    dst_idx = (
        torch.linspace(0, n_dst - 1, n_pairs, device=target_dataset_traj.device)
        .round()
        .long()
    )
    src_pos = source_raw_traj[src_idx, :3, 3]
    dst_pos = target_dataset_traj[dst_idx, :3, 3]
    src_rot = source_raw_traj[src_idx, :3, :3]
    dst_rot = target_dataset_traj[dst_idx, :3, :3]

    # Each matching pose implies dst_rot = rotation @ src_rot. Average those
    # relative rotations and project the result onto SO(3). Position-only SVD
    # leaves roll around a straight trajectory unconstrained.
    relative_rotations = dst_rot @ src_rot.transpose(-1, -2)
    u, _, vh = torch.linalg.svd(relative_rotations.mean(dim=0))
    correction = torch.eye(3, device=u.device, dtype=u.dtype)
    correction[-1, -1] = torch.det(u @ vh).sign()
    rotation = u @ correction @ vh
    translation = dst_pos.mean(dim=0) - rotation @ src_pos.mean(dim=0)
    return rotation, translation


def _log_post_transform_alignment_stats(
    aligned_source_traj: torch.Tensor,
    target_dataset_traj: torch.Tensor,
    source_name: str = "source",
    target_name: str = "target",
) -> None:
    if aligned_source_traj.ndim != 3 or aligned_source_traj.shape[-2:] != (4, 4):
        return
    if target_dataset_traj.ndim != 3 or target_dataset_traj.shape[-2:] != (4, 4):
        return

    n_src = aligned_source_traj.shape[0]
    n_dst = target_dataset_traj.shape[0]
    n_pairs = min(max(min(n_src, n_dst), 1), 256)
    src_idx = (
        torch.linspace(0, n_src - 1, n_pairs, device=aligned_source_traj.device)
        .round()
        .long()
    )
    dst_idx = (
        torch.linspace(0, n_dst - 1, n_pairs, device=target_dataset_traj.device)
        .round()
        .long()
    )

    src_pos = aligned_source_traj[src_idx, :3, 3]
    dst_pos = target_dataset_traj[dst_idx, :3, 3].to(device=src_pos.device, dtype=src_pos.dtype)
    displacement = dst_pos - src_pos
    displacement_norm = torch.linalg.norm(displacement, dim=-1)

    src_rot = aligned_source_traj[src_idx, :3, :3]
    dst_rot = target_dataset_traj[dst_idx, :3, :3].to(device=src_rot.device, dtype=src_rot.dtype)
    relative_rot = torch.matmul(dst_rot, src_rot.transpose(-1, -2))
    trace = relative_rot[:, 0, 0] + relative_rot[:, 1, 1] + relative_rot[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation_angle_deg = torch.rad2deg(torch.acos(cos_theta))

    stats_msg = (
        f"Post-transform alignment residuals {source_name} -> {target_name}: "
        f"mean_displacement={displacement_norm.mean().item():.6f} m, "
        f"std_displacement={displacement_norm.std(unbiased=False).item():.6f} m, "
        f"mean_rotational_displacement={rotation_angle_deg.mean().item():.6f} deg, "
        f"std_rotational_displacement={rotation_angle_deg.std(unbiased=False).item():.6f} deg"
    )
    logger.info(stats_msg)
    print(stats_msg)


def _apply_rigid_alignment_to_trajectory(
    traj: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply a global rigid transform to all poses in trajectory."""
    if traj.ndim != 3 or traj.shape[-2:] != (4, 4):
        raise ValueError(f"traj must be [N,4,4], got {tuple(traj.shape)}")
    aligned = traj.clone()
    rotation = rotation.to(device=traj.device, dtype=traj.dtype)
    translation = translation.to(
        device=traj.device, dtype=traj.dtype
    )
    aligned[:, :3, :3] = torch.matmul(rotation.unsqueeze(0), aligned[:, :3, :3])
    aligned[:, :3, 3] = (
        torch.matmul(rotation, aligned[:, :3, 3].transpose(0, 1)).transpose(0, 1)
        + translation
    )
    return aligned


def apply_render_frame_limit(dataset: DrivingDataset, max_render_frames: Optional[int]) -> Optional[int]:
    if max_render_frames is None:
        return None
    max_render_frames = int(max_render_frames)
    if max_render_frames <= 0:
        logger.warning("max_render_frames <= 0, skipping frame limiting.")
        return None
    max_render_frames = min(max_render_frames, dataset.num_img_timesteps)
    max_img_idx = max_render_frames * dataset.pixel_source.num_cams

    dataset.train_indices = [i for i in dataset.train_indices if i < max_img_idx]
    dataset.train_timesteps = dataset.train_timesteps[dataset.train_timesteps < max_render_frames]
    dataset.train_image_set.split_indices = dataset.train_indices

    dataset.full_image_set.split_indices = list(range(max_img_idx))

    if dataset.test_image_set is not None:
        dataset.test_indices = [i for i in dataset.test_indices if i < max_img_idx]
        dataset.test_timesteps = dataset.test_timesteps[dataset.test_timesteps < max_render_frames]
        dataset.test_image_set.split_indices = dataset.test_indices

    logger.info(f"Limiting rendering to first {max_render_frames} frames.")
    return max_render_frames


@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True,
    extract_camera_poses: bool = True,  # new parameter
    max_render_frames: Optional[int] = None,
    trajectory_file: Optional[str] = None,
    original_trajectory_raw: Optional[str] = None,
    output_root: Optional[str] = None,
):
    print("Save images is", args.save_images)
    trainer.set_eval()
    output_root = output_root or cfg.log_dir
    # New: camera pose extraction feature
    if extract_camera_poses:
        logger.info("Extracting camera poses...")
        
        # Extract poses for each dataset
        pose_save_dir = f"{output_root}/camera_poses{post_fix}"
        os.makedirs(pose_save_dir, exist_ok=True)
        
        # Extract test set poses
        if dataset.test_image_set is not None:
            logger.info("Extracting poses from test set...")
            test_poses = extract_camera_poses_from_dataset(
                dataset=dataset.test_image_set,
                trainer=trainer
            )
            test_pose_file = os.path.join(pose_save_dir, f"test_poses_{current_time}.npz")
            save_camera_poses(test_poses, test_pose_file)
            analyze_camera_trajectory(test_poses)
        
        # Extract full dataset poses
        if cfg.render.render_full:
            logger.info("Extracting poses from full set...")
            full_poses = extract_camera_poses_from_dataset(
                dataset=dataset.full_image_set,
                trainer=trainer
            )
            full_pose_file = os.path.join(pose_save_dir, f"full_poses_{current_time}.npz")
            save_camera_poses(full_poses, full_pose_file)
            analyze_camera_trajectory(full_poses)

    logger.info("Evaluating Pixels...")
    if dataset.test_image_set is not None and cfg.render.render_test:
        logger.info("Evaluating Test Set Pixels...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.test_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
            extract_poses=extract_camera_poses,  # new parameter
        )

        # New: save test set pose information (if extracted during render)
        if extract_camera_poses and "camera_poses" in render_results:
            test_render_pose_file = os.path.join(pose_save_dir, f"test_render_poses_{current_time}.npz")
            poses_dict = {
                'frame_indices': render_results['frame_indices'],
                'camera_poses': render_results['camera_poses'],
                'camera_positions': render_results['camera_positions'],
                'camera_rotations': render_results['camera_rotations'],
                'camera_intrinsics': render_results['camera_intrinsics'],
                'cam_names': render_results['cam_names'],
                'cam_ids': render_results['cam_ids'],
                'heights': render_results['heights'],
                'widths': render_results['widths']
            }
            save_camera_poses(poses_dict, test_render_pose_file)
  
    logger.info("Evaluating Pixels...")
    if dataset.test_image_set is not None and cfg.render.render_test:
        logger.info("Evaluating Test Set Pixels...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.test_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
        )

    
        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/test/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = (
                f"{output_root}/metrics{post_fix}/images_test_{current_time}.json"
            )
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{output_root}/videos{post_fix}/test_set_{step}.mp4"
        else:
            video_output_pth = f"{output_root}/videos{post_fix}/test_set_{step}_{args.render_video_postfix}.mp4"
        num_test_frames = dataset.num_test_timesteps
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=num_test_frames,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=2,
            verbose=True,
            save_images=False,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/test/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

    if cfg.render.render_full:
        logger.info("Evaluating Full Set...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
        )

        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/full/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            full_metrics_file = (
                f"{output_root}/metrics{post_fix}/images_full_{current_time}.json"
            )
            with open(full_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {full_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{output_root}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = f"{output_root}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
        num_full_frames = max_render_frames or dataset.num_img_timesteps
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=num_full_frames,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=cfg.render.fps,
            verbose=True,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/full/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

    render_novel_cfg = cfg.render.get("render_novel", None)
    if render_novel_cfg is not None or trajectory_file is not None:
        logger.info("Rendering novel views...")
        if trajectory_file is not None:
            loaded_traj = _load_novel_trajectory_from_file(
                trajectory_file, device=dataset.pixel_source.device
            )
            _log_trajectory_camera_positions("file_trajectory_raw", loaded_traj)
            ref_cam_id = dataset.pixel_source.camera_list[0]
            reference_traj = dataset.pixel_source.camera_data[ref_cam_id].cam_to_worlds.to(
                loaded_traj.device
            )

            if original_trajectory_raw is not None:
                raw_reference_traj = _load_novel_trajectory_from_file(
                    original_trajectory_raw, device=dataset.pixel_source.device
                )
                rotation_offset, translation_offset = _estimate_rigid_alignment_from_reference(
                    source_raw_traj=raw_reference_traj,
                    target_dataset_traj=reference_traj,
                )
                offset_msg = (
                    "Estimated trajectory rigid alignment "
                    f"[r00={rotation_offset[0, 0].item():.6f}, "
                    f"r01={rotation_offset[0, 1].item():.6f}, "
                    f"r02={rotation_offset[0, 2].item():.6f}, "
                    f"r10={rotation_offset[1, 0].item():.6f}, "
                    f"r11={rotation_offset[1, 1].item():.6f}, "
                    f"r12={rotation_offset[1, 2].item():.6f}, "
                    f"r20={rotation_offset[2, 0].item():.6f}, "
                    f"r21={rotation_offset[2, 1].item():.6f}, "
                    f"r22={rotation_offset[2, 2].item():.6f}; "
                    f"[dx={translation_offset[0].item():.6f}, "
                    f"dy={translation_offset[1].item():.6f}, "
                    f"dz={translation_offset[2].item():.6f}]"
                )
                logger.info(offset_msg)
                print(offset_msg)
                loaded_traj = _apply_rigid_alignment_to_trajectory(
                    loaded_traj,
                    rotation_offset,
                    translation_offset,
                )
                _log_post_transform_alignment_stats(
                    aligned_source_traj=_apply_rigid_alignment_to_trajectory(
                        raw_reference_traj,
                        rotation_offset,
                        translation_offset,
                    ),
                    target_dataset_traj=reference_traj,
                    source_name=os.path.splitext(os.path.basename(original_trajectory_raw))[0],
                    target_name=f"dataset_cam_{ref_cam_id}",
                )
                _log_trajectory_camera_positions("file_trajectory_aligned", loaded_traj)
            else:
                logger.info(
                    "No --original_trajectory_raw provided; using trajectory_file poses without rigid alignment."
                )
            traj_name = os.path.splitext(os.path.basename(trajectory_file))[0]
            render_traj = {f"file_{traj_name}": loaded_traj}
            logger.info(
                f"Using custom trajectory file for novel rendering: {trajectory_file}"
            )
        else:
            render_traj = dataset.get_novel_render_traj(
                traj_types=render_novel_cfg.traj_types,
                target_frames=render_novel_cfg.get("frames", dataset.frame_num),
            )
        video_output_dir = f"{output_root}/videos{post_fix}/novel_{step}"
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        for traj_type, traj in render_traj.items():
            _log_trajectory_camera_positions(traj_type, traj)
            # Prepare rendering data lazily to avoid materializing all frames in memory.
            render_data = dataset.iter_novel_view_render_data(traj)

            # Render and save video
            save_path = os.path.join(video_output_dir, f"{traj_type}.mp4")
            print("eval.py > Started rendering novel view")
            render_novel_views(
                trainer,
                render_data,
                save_path,
                fps=(
                    render_novel_cfg.get("fps", cfg.render.fps)
                    if render_novel_cfg is not None
                    else cfg.render.fps
                ),
                traj_type=traj_type,
                save_images=args.save_images
            )
            logger.info(
                f"Saved novel view video for trajectory type: {traj_type} to {save_path}"
            )

    custom_pose_dir, custom_traj_name = resolve_custom_trajectory(args, dataset)
    if custom_pose_dir is not None:
        logger.info("Rendering custom trajectory %s...", custom_traj_name)
        custom_traj = load_custom_ego_trajectory(
            dataset=dataset,
            ego_pose_dir=custom_pose_dir,
            camera_id=args.custom_camera_id,
            extrinsics_path=args.custom_extrinsics,
            ego_lateral_position=args.custom_ego_lateral_position,
        ).to(dataset.pixel_source.device)
        _log_trajectory_camera_positions(custom_traj_name, custom_traj)
        if len(custom_traj) != dataset.frame_num:
            logger.warning(
                "Custom trajectory has %d poses but the scene has %d frames; "
                "timestamps will be linearly mapped across the trained sequence.",
                len(custom_traj),
                dataset.frame_num,
            )

        render_data = dataset.iter_novel_view_render_data(custom_traj)
        video_output_dir = f"{output_root}/videos{post_fix}/novel_{step}"
        os.makedirs(video_output_dir, exist_ok=True)
        save_path = os.path.join(video_output_dir, f"{custom_traj_name}.mp4")
        render_novel_views(
            trainer,
            render_data,
            save_path,
            fps=args.custom_trajectory_fps or cfg.render.fps,
            traj_type=custom_traj_name,
            save_raw=args.custom_save_raw_frames,
            save_images=args.custom_save_images,
        )
        logger.info("Saved custom trajectory video to %s", save_path)


def main(args):
    ckpt_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    
    args.enable_wandb = False
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(ckpt_dir, folder), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    has_custom_trajectory = bool(
        args.custom_ego_pose_dir or args.custom_trajectory_name
    )
    if has_custom_trajectory:
        render_downscale = float(args.custom_render_downscale)
        if render_downscale <= 0:
            raise ValueError("--custom_render_downscale must be greater than zero")
        camera_count = len(cfg.data.pixel_source.cameras)
        cfg.data.pixel_source.downscale_when_loading = [render_downscale] * camera_count
        logger.info(
            "Custom trajectory render: using source image resolution with downscale %.3g.",
            render_downscale,
        )
    lazy_dataset_mode = args.lazy_dataset or (
        args.skip_original_render
        and (args.trajectory_file is not None or has_custom_trajectory)
    )
    if lazy_dataset_mode:
        logger.info("Using lazy dataset mode for eval.")
        cfg.data.project_lidar_on_images = False
        cfg.data.pixel_source.load_rgb_images = False
        cfg.data.pixel_source.load_dynamic_mask = False
        cfg.data.pixel_source.load_sky_mask = False
        if "load_road_mask" in cfg.data.pixel_source:
            cfg.data.pixel_source.load_road_mask = False
        if cfg.render.render_test:
            logger.info("Disabling test-set rendering in lazy dataset mode.")
            cfg.render.render_test = False
        cfg.render.render_full = False
    if args.skip_lidar:
        logger.info("Skipping LiDAR loading; using camera-derived scene bounds for eval.")
        cfg.data.lidar_source.load_lidar = False
        cfg.data.project_lidar_on_images = False

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)
    original_train_traj = dataset.pixel_source.front_camera_trajectory
    _log_trajectory_camera_positions("training_original_trajectory", original_train_traj)

    # Log the corrected trajectory that the checkpoint's CamPose module applies on top.
    ckpt_state = torch.load(args.resume_from, map_location="cpu")
    cam_pose = ckpt_state.get("models", {}).get("CamPose", {})
    cam_pose_embeds = cam_pose.get("embeds.weight", None)
    if cam_pose_embeds is not None:
        corrected_train_traj = _apply_cam_pose_correction_from_embeds(
            raw_traj=original_train_traj,
            cam_pose_embeds=cam_pose_embeds,
            device=original_train_traj.device,
        )
        _log_trajectory_camera_positions(
            "training_original_trajectory_corrected_by_CamPose",
            corrected_train_traj,
        )

    # setup trainer
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

    # Resume from checkpoint
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )
    if args.hold_dynamic_nodes:
        hold_last_dynamic_node_states(trainer)
    
    # Restore textured road mesh if available
    _restore_textured_road_mesh(trainer, ckpt_dir)

    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)

    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    # Override config values
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    if hasattr(trainer, 'road_mesh') and trainer.road_mesh is not None:
        render_keys.append("road_mesh_rgb")

    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False
    if args.skip_original_render:
        print("Skipping rendering and evaluation")
        cfg.render.render_full = False

    max_render_frames = apply_render_frame_limit(dataset, args.max_render_frames)

    do_evaluation(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix="_eval"+args.render_video_postfix,
        max_render_frames=max_render_frames,
        trajectory_file=args.trajectory_file,
        original_trajectory_raw=args.original_trajectory_raw,
        output_root=ckpt_dir,
        extract_camera_poses=not args.skip_original_render,
    )

    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    # eval
    parser.add_argument(
        "--resume_from",
        default=None,
        help="path to checkpoint to resume from",
        type=str,
        required=True,
    )
    # Custom file end name
    parser.add_argument(
        "--render_video_postfix",
        type=str,
        default="",
        help="an optional postfix for video",
    )
    # Save images from rendered video
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Saves images in a folder at the resultant video output folder.",
    )
    parser.add_argument(
        "--save_catted_videos",
        type=bool,
        default=False,
        help="visualize lidar on image",
    )
    parser.add_argument(
        "--skip_original_render",
        action="store_true",
        help="Skips running some metrics as well as rendering the original viewpoint.",
    )
    parser.add_argument(
        "--hold_dynamic_nodes",
        action="store_true",
        help="Hold each rigid, deformable, and SMPL instance at its last valid state through later render frames.",
    )
    parser.add_argument(
        "--max_render_frames",
        type=int,
        default=None,
        help="limit rendering to the first N frames without changing dataset timesteps",
    )
    parser.add_argument(
        "--trajectory_file",
        type=str,
        default=None,
        help="path to a .npy/.npz camera trajectory file used for novel-view rendering",
    )
    parser.add_argument(
        "--original_trajectory_raw",
        type=str,
        default=None,
        help=(
            "path to the raw training trajectory (.npy/.npz) used as reference to estimate "
            "a rigid transform into the dataset-loaded frame"
        ),
    )
    custom_trajectory_group = parser.add_mutually_exclusive_group()
    custom_trajectory_group.add_argument(
        "--custom_trajectory_name",
        type=str,
        default=None,
        help=(
            "Per-scene trajectory name resolved as "
            "<scene>/novel_views/<name>/ego_pose (for example: right_4m)."
        ),
    )
    custom_trajectory_group.add_argument(
        "--custom_ego_pose_dir",
        type=str,
        default=None,
        help="Explicit directory containing numbered 4x4 ego-pose .txt files.",
    )
    parser.add_argument(
        "--custom_extrinsics",
        type=str,
        default=None,
        help=(
            "Camera-to-ego extrinsics for the custom trajectory. Defaults to "
            "the calibration beside ego_pose, then the scene calibration."
        ),
    )
    parser.add_argument(
        "--custom_camera_id",
        type=int,
        default=0,
        help="Camera calibration ID used for the custom trajectory (default: 0).",
    )
    parser.add_argument(
        "--custom_ego_lateral_position",
        type=float,
        default=None,
        metavar="METERS",
        help="Override camera translation along ego Y (positive is left).",
    )
    parser.add_argument(
        "--custom_trajectory_label",
        type=str,
        default=None,
        help="Optional output filename stem for the custom trajectory.",
    )
    parser.add_argument(
        "--custom_trajectory_fps",
        type=int,
        default=None,
        help="Override FPS for the custom trajectory video.",
    )
    parser.add_argument(
        "--custom_render_downscale",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Custom-trajectory output downscale relative to source images "
            "(default: 1, full source resolution)."
        ),
    )
    parser.add_argument(
        "--custom_save_raw_frames",
        action="store_true",
        help="Also save raw per-frame outputs for the custom trajectory.",
    )
    parser.add_argument(
        "--custom_save_images",
        action="store_true",
        help="Save every custom-trajectory RGB frame as a PNG image.",
    )
    parser.add_argument(
        "--lazy_dataset",
        action="store_true",
        help="skip loading RGB images/masks and image-based lidar projection; best for trajectory-only novel rendering",
    )
    parser.add_argument(
        "--skip-lidar",
        action="store_true",
        help="Skip LiDAR loading and use camera-derived scene bounds (for renders that do not need LiDAR).",
    )

    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # misc
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    print("Save images status", args.save_images)
    print("--- Args: ---", args)
    main(args)
