"""
Fine-tune camera poses to minimize RGB render error.

The goal of this script is to align the camera poses to ensure proper alignment with the GT view.
This implementation tries to finetune camera poses to minimize the RGB difference between the
rendered images and the GT images.

Inputs:
- Camera traj estimate (slightly different from the GT traj)
- GT images
- Gaussian/mesh model of the scene

Outputs:
- A new traj estimate that is more aligned with camera poses of the GT images

Method:
- Render the scene from the estimated camera poses
- Compare the rendered images with the GT images
- Generate n random perturbations of distance d rotation r of camera poses
- Render the scene from the perturbed camera poses
- Compare the rendered images with the GT images
- Keep the perturbation with lowest RGB difference
- Decrease d and r
- Repeat until convergence or max iterations reached
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets.driving_dataset import DrivingDataset
from models.road_mesh import RoadMesh
from utils.misc import import_str

logger = logging.getLogger(__name__)


def _restore_textured_road_mesh(trainer, ckpt_dir: str) -> None:
    """Restore the textured road mesh from the dedicated mesh checkpoint if available."""
    mesh_checkpoint_path = os.path.join(ckpt_dir, "road_mesh.pth")
    if not os.path.exists(mesh_checkpoint_path):
        logger.warning("No road mesh checkpoint found at %s", mesh_checkpoint_path)
        return

    try:
        trainer.road_mesh = RoadMesh.load_checkpoint(mesh_checkpoint_path, device=trainer.device)
        logger.info(
            "Loaded textured road mesh from %s (has_texture=%s)",
            mesh_checkpoint_path,
            trainer.road_mesh.texture_buffer is not None,
        )
    except Exception as exc:
        logger.warning("Failed to load textured road mesh from %s: %s", mesh_checkpoint_path, exc)


def _load_pose_bundle(trajectory_path: str) -> Dict[str, np.ndarray]:
    """Load camera poses and metadata from npy/npz file."""
    if not os.path.isfile(trajectory_path):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    ext = os.path.splitext(trajectory_path)[1].lower()
    if ext == ".npy":
        poses = np.asarray(np.load(trajectory_path, allow_pickle=True))
        data = {"camera_poses": poses}
    elif ext == ".npz":
        with np.load(trajectory_path, allow_pickle=True) as data_npz:
            data = {key: np.asarray(data_npz[key]) for key in data_npz.files}
        candidate_keys = ["camera_poses", "poses", "trajectory"]
        pose_key = next((key for key in candidate_keys if key in data), None)
        if pose_key is None:
            raise KeyError(
                f"No supported pose key found in {trajectory_path}. "
                f"Expected one of: {candidate_keys}"
            )
        poses = np.asarray(data[pose_key])
        data["camera_poses"] = poses
    else:
        raise ValueError(f"Unsupported trajectory file extension: {ext}. Use .npy or .npz")

    poses = np.asarray(data["camera_poses"])
    if poses.ndim == 2 and poses.shape == (4, 4):
        poses = poses[None, ...]
    if poses.ndim != 3:
        raise ValueError(f"Trajectory must be a 3D array, got shape {poses.shape}")
    if poses.shape[-2:] == (3, 4):
        bottom_row = np.zeros((poses.shape[0], 1, 4), dtype=poses.dtype)
        bottom_row[:, 0, 3] = 1.0
        poses = np.concatenate([poses, bottom_row], axis=1)
    if poses.shape[-2:] != (4, 4):
        raise ValueError(f"Trajectory poses must be [N,4,4] or [N,3,4], got shape {poses.shape}")

    data["camera_poses"] = poses
    if "frame_indices" not in data:
        data["frame_indices"] = np.arange(len(poses), dtype=np.int64)

    if "cam_ids" in data and len(np.asarray(data["cam_ids"])) == len(poses):
        cam_ids = np.asarray(data["cam_ids"]).reshape(-1)
        if len(np.unique(cam_ids)) > 1:
            keep = cam_ids == cam_ids[0]
            data = {
                key: value[keep] if isinstance(value, np.ndarray) and len(value) == len(poses) else value
                for key, value in data.items()
            }

    if "cam_names" in data and len(np.asarray(data["cam_names"])) == len(poses):
        cam_names = np.asarray(data["cam_names"]).astype(str).reshape(-1)
        if len(np.unique(cam_names)) > 1:
            keep = cam_names == cam_names[0]
            data = {
                key: value[keep] if isinstance(value, np.ndarray) and len(value) == len(poses) else value
                for key, value in data.items()
            }

    return data


def _as_pose_tensor(poses: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert pose array to tensor on device."""
    return torch.as_tensor(poses, dtype=torch.float32, device=device)


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotation to 3x3 rotation matrix using Rodrigues' formula."""
    axis = axis / (torch.linalg.norm(axis) + 1e-8)
    x, y, z = axis
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c

    return torch.stack(
        [
            torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s]),
            torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s]),
            torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c]),
        ]
    )


def _sample_translation_step(step_m: float, device: torch.device, rng: np.random.Generator) -> torch.Tensor:
    """Sample a random translation perturbation within radius step_m."""
    direction = torch.as_tensor(rng.normal(size=3), dtype=torch.float32, device=device)
    direction = direction / (torch.linalg.norm(direction) + 1e-8)
    radius = torch.as_tensor(rng.uniform(0.0, step_m), dtype=torch.float32, device=device)
    return direction * radius


def _sample_rotation_step(step_rad: float, device: torch.device, rng: np.random.Generator) -> torch.Tensor:
    """Sample a random rotation perturbation within angle step_rad."""
    axis = torch.as_tensor(rng.normal(size=3), dtype=torch.float32, device=device)
    axis = axis / (torch.linalg.norm(axis) + 1e-8)
    angle = torch.as_tensor(rng.uniform(-step_rad, step_rad), dtype=torch.float32, device=device)
    return _axis_angle_to_matrix(axis, angle)


def _perturb_pose(pose: torch.Tensor, translation_m: float, rotation_rad: float, rng: np.random.Generator) -> torch.Tensor:
    """Apply random translation and rotation perturbations to a camera pose (4x4 matrix)."""
    device = pose.device
    delta_t = _sample_translation_step(translation_m, device, rng)
    delta_r = _sample_rotation_step(rotation_rad, device, rng)

    out = pose.clone()
    out[:3, :3] = out[:3, :3] @ delta_r
    out[:3, 3] = out[:3, 3] + out[:3, :3] @ delta_t
    return out


def _render_pose_loss(dataset, trainer, frame_idx: int, pose: torch.Tensor) -> float:
    """Render a frame from a pose and return MSE loss vs GT image."""
    camera_downscale = trainer._get_downscale_factor()
    image_infos, cam_infos = dataset.get_image(frame_idx, camera_downscale)
    for key, value in image_infos.items():
        if isinstance(value, torch.Tensor):
            image_infos[key] = value.to(trainer.device)
    for key, value in cam_infos.items():
        if isinstance(value, torch.Tensor):
            cam_infos[key] = value.to(trainer.device)

    cam_infos["camera_to_world"] = pose.to(trainer.device)

    with torch.no_grad():
        outputs = trainer(image_infos, cam_infos, novel_view=True)
        pred = outputs["rgb"]
        target = image_infos["pixels"]
        return torch.nn.functional.mse_loss(pred, target).item()


def _pose_search(
    dataset,
    trainer,
    initial_poses: torch.Tensor,
    frame_indices: Sequence[int],
    num_candidates: int,
    translation_step_m: float,
    rotation_step_deg: float,
    translation_decay: float,
    rotation_decay: float,
    min_translation_step_m: float,
    min_rotation_step_deg: float,
    max_iterations: int,
    seed: int,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """
    Search for better camera poses by minimizing render loss.
    
    For each iteration:
    - Sample random perturbations for each pose
    - Render and evaluate loss
    - Keep improvements
    - Decay perturbation sizes
    - Stop when sizes become too small or no improvements
    """
    rng = np.random.default_rng(seed)
    current_poses = initial_poses.clone()
    
    logger.info("Computing initial losses for %d frames...", len(frame_indices))
    current_losses = torch.tensor(
        [_render_pose_loss(dataset, trainer, int(frame_idx), current_poses[i]) for i, frame_idx in tqdm(enumerate(frame_indices), total=len(frame_indices), desc="Initial loss")],
        dtype=torch.float32,
    )
    history: List[Dict[str, float]] = []

    translation_step = float(translation_step_m)
    rotation_step = math.radians(float(rotation_step_deg))
    min_translation_step = float(min_translation_step_m)
    min_rotation_step = math.radians(float(min_rotation_step_deg))
    
    initial_mean_loss = float(current_losses.mean().item())
    logger.info("Initial mean loss: %.6f", initial_mean_loss)

    for iteration in range(int(max_iterations)):
        improved = 0
        total_before = float(current_losses.mean().item()) if len(current_losses) else 0.0
        improved_frames_list = []

        frame_pbar = tqdm(enumerate(frame_indices), total=len(frame_indices), desc=f"Iteration {iteration+1}/{int(max_iterations)}")
        for i, frame_idx in frame_pbar:
            base_pose = current_poses[i]
            base_loss = float(current_losses[i].item())
            best_pose = base_pose
            best_loss = base_loss

            for _ in range(int(num_candidates)):
                candidate_pose = _perturb_pose(base_pose, translation_step, rotation_step, rng)
                candidate_loss = _render_pose_loss(dataset, trainer, int(frame_idx), candidate_pose)
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_pose = candidate_pose

            if best_loss + 1e-10 < base_loss:
                current_poses[i] = best_pose
                current_losses[i] = best_loss
                improved += 1
                improved_frames_list.append((i, base_loss, best_loss, base_loss - best_loss))
            
            frame_pbar.set_postfix({
                "improved": improved,
                "mean_loss": f"{current_losses[:i+1].mean():.6f}"
            })

        total_after = float(current_losses.mean().item()) if len(current_losses) else 0.0
        improvement_pct = 100.0 * (1.0 - total_after / (total_before + 1e-8))
        
        history.append(
            {
                "iteration": float(iteration),
                "mean_loss_before": total_before,
                "mean_loss_after": total_after,
                "translation_step_m": translation_step,
                "rotation_step_deg": math.degrees(rotation_step),
                "improved_frames": float(improved),
            }
        )
        
        logger.info(
            "Iteration %d: mean_loss %.6f -> %.6f (%.2f%% improvement), "
            "improved_frames=%d/%d, step=(%.4fm, %.4fdeg)",
            iteration,
            total_before,
            total_after,
            improvement_pct,
            improved,
            len(frame_indices),
            translation_step,
            math.degrees(rotation_step),
        )
        
        if improved > 0:
            losses_reduced = [delta for _, _, _, delta in improved_frames_list]
            logger.info(
                "  Top improvements: %.6f, %.6f, %.6f (avg: %.6f)",
                max(losses_reduced) if losses_reduced else 0,
                sorted(losses_reduced)[-2] if len(losses_reduced) > 1 else 0,
                sorted(losses_reduced)[-3] if len(losses_reduced) > 2 else 0,
                np.mean(losses_reduced) if losses_reduced else 0,
            )

        translation_step *= float(translation_decay)
        rotation_step *= float(rotation_decay)
        if translation_step <= min_translation_step and rotation_step <= min_rotation_step:
            logger.info("Stopping: search steps reached minimum thresholds (translation=%.6fm, rotation=%.6fdeg)", translation_step, math.degrees(rotation_step))
            break
        if improved == 0:
            logger.info("Stopping: no frame improved in this iteration.")
            break

    final_mean_loss = float(current_losses.mean().item())
    total_improvement = initial_mean_loss - final_mean_loss
    total_improvement_pct = 100.0 * (1.0 - final_mean_loss / (initial_mean_loss + 1e-8))
    logger.info(
        "\n=== Search Complete ===\n"
        "  Initial mean loss: %.6f\n"
        "  Final mean loss:   %.6f\n"
        "  Total improvement: %.6f (%.2f%%)\n"
        "  Iterations completed: %d\n"
        "  Total frames improved: %d",
        initial_mean_loss,
        final_mean_loss,
        total_improvement,
        total_improvement_pct,
        iteration + 1,
        sum(int(h["improved_frames"]) for h in history),
    )

    return current_poses, history


def _build_minimal_trainer(cfg, dataset: DrivingDataset, device: torch.device):
    """Build and return a minimal trainer instance."""
    trainer_cfg = dict(cfg.trainer)
    trainer_cfg.pop("resume_from", None)
    trainer_cfg.pop("resume_workflow_from", None)

    return import_str(cfg.trainer.type)(
        **trainer_cfg,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )


def _save_aligned_trajectory(
    output_path: str,
    original_bundle: Dict[str, np.ndarray],
    aligned_poses: torch.Tensor,
) -> None:
    """Save aligned poses to npz file, preserving metadata from original bundle."""
    aligned_np = aligned_poses.detach().cpu().numpy()
    payload = dict(original_bundle)
    payload["camera_poses"] = aligned_np
    payload["camera_positions"] = aligned_np[:, :3, 3]
    payload["camera_rotations"] = aligned_np[:, :3, :3]
    payload.setdefault("frame_indices", np.arange(len(aligned_np), dtype=np.int64))
    np.savez(output_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune camera poses by minimizing RGB render error")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--trajectory_path", type=str, required=True, help="Path to the estimated camera pose npy/npz file")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the aligned trajectory")
    parser.add_argument("--num_candidates", type=int, default=16, help="Random perturbations to sample per frame and iteration")
    parser.add_argument("--max_iterations", type=int, default=8, help="Maximum outer search iterations")
    parser.add_argument("--translation_step_m", type=float, default=0.5, help="Initial translation perturbation radius in meters")
    parser.add_argument("--rotation_step_deg", type=float, default=5.0, help="Initial rotation perturbation range in degrees")
    parser.add_argument("--translation_decay", type=float, default=0.5, help="Per-iteration translation step decay")
    parser.add_argument("--rotation_decay", type=float, default=0.5, help="Per-iteration rotation step decay")
    parser.add_argument("--min_translation_step_m", type=float, default=0.01, help="Stop when translation perturbations shrink below this")
    parser.add_argument("--min_rotation_step_deg", type=float, default=0.1, help="Stop when rotation perturbations shrink below this")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    ckpt_dir = os.path.dirname(args.checkpoint_path)
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = _build_minimal_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=args.checkpoint_path, load_only_model=True)
    _restore_textured_road_mesh(trainer, ckpt_dir)
    trainer.set_eval()

    pose_bundle = _load_pose_bundle(args.trajectory_path)
    initial_poses = _as_pose_tensor(np.asarray(pose_bundle["camera_poses"]), device=device)
    frame_indices = np.asarray(pose_bundle.get("frame_indices", np.arange(len(initial_poses))), dtype=np.int64).reshape(-1)
    if len(frame_indices) != len(initial_poses):
        raise ValueError(
            f"frame_indices length ({len(frame_indices)}) does not match camera_poses length ({len(initial_poses)})"
        )

    if len(initial_poses) > len(dataset.full_image_set):
        raise ValueError(
            f"Trajectory has {len(initial_poses)} poses but dataset only has {len(dataset.full_image_set)} frames"
        )

    aligned_poses, history = _pose_search(
        dataset=dataset.full_image_set,
        trainer=trainer,
        initial_poses=initial_poses,
        frame_indices=frame_indices,
        num_candidates=args.num_candidates,
        translation_step_m=args.translation_step_m,
        rotation_step_deg=args.rotation_step_deg,
        translation_decay=args.translation_decay,
        rotation_decay=args.rotation_decay,
        min_translation_step_m=args.min_translation_step_m,
        min_rotation_step_deg=args.min_rotation_step_deg,
        max_iterations=args.max_iterations,
        seed=args.seed,
    )

    _save_aligned_trajectory(args.output_path, pose_bundle, aligned_poses)
    history_path = os.path.splitext(args.output_path)[0] + "_history.npz"
    np.savez(history_path, **{
        "iterations": np.array([h["iteration"] for h in history]),
        "mean_loss_before": np.array([h["mean_loss_before"] for h in history]),
        "mean_loss_after": np.array([h["mean_loss_after"] for h in history]),
        "translation_step_m": np.array([h["translation_step_m"] for h in history]),
        "rotation_step_deg": np.array([h["rotation_step_deg"] for h in history]),
        "improved_frames": np.array([h["improved_frames"] for h in history]),
    })

    print(f"Saved aligned trajectory to {args.output_path}")
    print(f"Saved search history to {history_path}")


if __name__ == "__main__":
    main()
