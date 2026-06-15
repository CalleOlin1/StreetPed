import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from models.gaussians.basics import dataclass_camera, dataclass_gs, quat_mult
from utils.misc import import_str
from utils.camera import look_at_rotation
from utils.geometry import quat_to_rotmat


ObjectKey = Tuple[str, int]


def _load_model_state_tolerant(model: torch.nn.Module, model_state: dict, class_name: str) -> None:
    """Load model weights while skipping keys that are missing or shape-incompatible."""
    target_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for key, value in model_state.items():
        if key not in target_state:
            skipped.append((key, "missing_in_current_model"))
            continue
        if target_state[key].shape != value.shape:
            skipped.append(
                (
                    key,
                    f"shape_mismatch ckpt={tuple(value.shape)} current={tuple(target_state[key].shape)}",
                )
            )
            continue
        filtered_state[key] = value

    msg = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded {class_name} with tolerance: {msg}")
    if skipped:
        print(f"Skipped {len(skipped)} keys for {class_name}:")
        for key, reason in skipped:
            print(f"  - {key}: {reason}")


def load_checkpoint_tolerant(trainer, resume_from: str) -> None:
    """Best-effort checkpoint load for visualization-only workflows."""
    ckpt = torch.load(resume_from, map_location=trainer.device)
    step = int(ckpt.get("step", 0))
    trainer.step = step

    model_state_dict = ckpt.get("models", {})
    if not isinstance(model_state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid 'models' dictionary")

    for class_name, model in trainer.models.items():
        model.step = step
        if class_name not in model_state_dict:
            print(f"Cannot find {class_name} in the checkpoint")
            continue

        try:
            msg = model.load_state_dict(model_state_dict[class_name], strict=True)
            print(f"Loaded {class_name} strictly: {msg}")
        except RuntimeError as err:
            print(f"Strict load failed for {class_name}: {err}")
            _load_model_state_tolerant(model, model_state_dict[class_name], class_name)

    print(f"Checkpoint loaded in tolerant mode from {resume_from} at step {step}")


def prune_broken_optional_models(trainer) -> None:
    """Remove optional model branches that are present but unusable in this checkpoint."""
    broken_models = []

    smpl_model = trainer.models.get("SMPLNodes", None)
    if smpl_model is not None:
        required_attrs = ["instances_fv", "instances_quats", "instances_trans"]
        if any(not hasattr(smpl_model, attr) for attr in required_attrs):
            broken_models.append("SMPLNodes")

    for class_name in broken_models:
        print(f"Disabling broken optional model: {class_name}")
        trainer.models.pop(class_name, None)
        if hasattr(trainer, "gaussian_classes"):
            trainer.gaussian_classes.pop(class_name, None)


def build_trainer_from_checkpoint(resume_from: str, opts: List[str]):
    """Build dataset + trainer and load model weights from checkpoint."""
    log_dir = os.path.dirname(resume_from)
    config_path = os.path.join(log_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config at: {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(opts))

    dataset = DrivingDataset(data_cfg=cfg.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Visualisation can proceed even when dataset cardinality differs from training,
    # e.g. AffineTransform embedding size mismatch after synthetic-frame finetuning.
    load_checkpoint_tolerant(trainer=trainer, resume_from=resume_from)
    prune_broken_optional_models(trainer)
    return cfg, dataset, trainer, log_dir


def extract_instance_points(trainer, frame_idx: int) -> Dict[ObjectKey, np.ndarray]:
    """Collect object-instance point clouds for RigidNodes and SMPLNodes."""
    instances: Dict[ObjectKey, np.ndarray] = {}

    try:
        rigid_info = trainer.get_rigid_info(frame_idx=frame_idx)
    except Exception as err:
        print(f"Skipping rigid extraction due to error: {err}")
        rigid_info = None

    if rigid_info is not None and rigid_info.get("instance_ids") is not None:
        rigid_ids = rigid_info["instance_ids"].detach().cpu().long()
        rigid_points = rigid_info["positions"].detach().cpu()
        for obj_id in torch.unique(rigid_ids).tolist():
            mask = rigid_ids == int(obj_id)
            pts = rigid_points[mask]
            if pts.numel() > 0:
                instances[("rigid", int(obj_id))] = pts.numpy()

    # SMPLNodes may exist in config but be absent/partially initialized in checkpoint.
    try:
        smpl_info = trainer.get_smpl_info(frame_idx=frame_idx)
    except Exception as err:
        print(f"Skipping smpl extraction due to error: {err}")
        smpl_info = None

    if smpl_info is not None and smpl_info.get("instance_ids") is not None:
        smpl_ids = smpl_info["instance_ids"].detach().cpu().long()
        smpl_points = smpl_info["positions"].detach().cpu()
        for obj_id in torch.unique(smpl_ids).tolist():
            mask = smpl_ids == int(obj_id)
            pts = smpl_points[mask]
            if pts.numel() > 0:
                instances[("smpl", int(obj_id))] = pts.numpy()

    return instances


def print_available_objects(instances: Dict[ObjectKey, np.ndarray]) -> None:
    rigid_ids = sorted([obj_id for obj_type, obj_id in instances.keys() if obj_type == "rigid"])
    smpl_ids = sorted([obj_id for obj_type, obj_id in instances.keys() if obj_type == "smpl"])

    print("Available objects to visualise:")
    if rigid_ids:
        print(f"  rigid: {rigid_ids}")
    else:
        print("  rigid: []")

    if smpl_ids:
        print(f"  smpl : {smpl_ids}")
    else:
        print("  smpl : []")


def select_object_key(
    instances: Dict[ObjectKey, np.ndarray], object_id: int, object_type: str
) -> ObjectKey:
    if object_type in {"rigid", "smpl"}:
        key = (object_type, object_id)
        if key not in instances:
            raise ValueError(f"Object id {object_id} not found in {object_type} objects")
        return key

    # auto mode: resolve by unique id across types.
    candidates = [k for k in instances.keys() if k[1] == object_id]
    if len(candidates) == 0:
        raise ValueError(f"Object id {object_id} not found in rigid or smpl objects")
    if len(candidates) > 1:
        raise ValueError(
            f"Object id {object_id} exists in multiple types {candidates}; "
            "please set --object_type rigid or --object_type smpl"
        )
    return candidates[0]


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0) / 4.0
    radius = max(radius, 1e-3)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_8_views(points_xyz: np.ndarray, title: str):
    raise NotImplementedError("This helper is replaced by renderer-based novel view rendering.")


def _to_device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device_tree(v, device) for v in value]
    return value


def build_orbit_poses(center: torch.Tensor, radius: float, elevation_deg: float = 18.0):
    """Build 8 look-at camera poses around a 3D center point."""
    azimuths = np.linspace(0.0, 315.0, 8)
    poses = []
    elevation_rad = np.deg2rad(elevation_deg)

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
        direction = center - cam_pos
        pose = torch.eye(4, dtype=center.dtype, device=center.device)
        pose[:3, 3] = cam_pos
        pose[:3, :3] = look_at_rotation(direction)
        poses.append(pose)

    return poses


def _look_at_c2w(
    direction: torch.Tensor,
    position: torch.Tensor,
    use_neg_front: bool,
    ref_up: torch.Tensor,
) -> torch.Tensor:
    """Build camera-to-world pose looking along direction with selectable forward-axis convention."""
    front = torch.nn.functional.normalize(direction, dim=-1)
    up = torch.nn.functional.normalize(ref_up, dim=-1)

    # Avoid singularity when view direction is almost parallel to up.
    if torch.abs(torch.dot(front, up)) > 0.98:
        up = torch.tensor([0.0, 0.0, 1.0], device=direction.device, dtype=direction.dtype)
        if torch.abs(torch.dot(front, up)) > 0.98:
            up = torch.tensor([0.0, 1.0, 0.0], device=direction.device, dtype=direction.dtype)

    right = torch.nn.functional.normalize(torch.cross(front, up, dim=0), dim=0)
    true_up = torch.nn.functional.normalize(torch.cross(right, front, dim=0), dim=0)

    # Keep roll aligned with reference camera up to avoid upside-down flips.
    if torch.dot(true_up, ref_up) < 0:
        right = -right
        true_up = -true_up

    pose = torch.eye(4, dtype=direction.dtype, device=direction.device)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -front if use_neg_front else front
    pose[:3, 3] = position
    return pose


def build_orbit_poses_with_convention(
    center: torch.Tensor,
    radius: float,
    elevation_deg: float,
    use_neg_front: bool,
    ref_up: torch.Tensor,
):
    azimuths = np.linspace(0.0, 315.0, 8)
    poses = []
    elevation_rad = np.deg2rad(elevation_deg)
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
        poses.append(
            _look_at_c2w(
                view_dir,
                cam_pos,
                use_neg_front=use_neg_front,
                ref_up=ref_up,
            )
        )
    return poses


@torch.no_grad()
def render_8_views_with_renderer(
    trainer,
    dataset,
    frame_idx: int,
    object_type: str,
    object_id: int,
    points_xyz: np.ndarray,
    title: str,
):
    """Render 8 orbit views around the selected object using the project renderer."""
    pixel_source = dataset.pixel_source
    cam0 = pixel_source.camera_data[0]

    ref_image_infos, ref_cam_infos = cam0.get_image(frame_idx)
    if not isinstance(ref_cam_infos["camera_to_world"], torch.Tensor):
        ref_cam_infos["camera_to_world"] = torch.as_tensor(ref_cam_infos["camera_to_world"])

    center = torch.tensor(points_xyz.mean(axis=0), device=ref_cam_infos["camera_to_world"].device, dtype=torch.float32)
    extent = np.linalg.norm(points_xyz - points_xyz.mean(axis=0, keepdims=True), axis=1)
    radius = float(max(extent.max() * 3.0, 4.0)) if extent.size > 0 else 4.0

    device = trainer.device if hasattr(trainer, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_cam_infos = _to_device_tree(ref_cam_infos, device)

    if object_type != "rigid":
        raise NotImplementedError(
            "Single-object rendering currently supports rigid instances only."
        )

    rigid_model = trainer.models.get("RigidNodes", None)
    if rigid_model is None:
        raise ValueError("RigidNodes model is required to render a rigid object")

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

    object_gs = dataclass_gs(
        _means=world_means,
        _opacities=local_gs["opacities"],
        _rgbs=rigid_model.colors[pts_mask],
        _scales=local_gs["scales"],
        _quats=world_quats,
        detach_keys=[],
        extras=None,
    )

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

    # Auto-select axis convention that actually sees the object.
    ref_up = ref_cam_infos["camera_to_world"][:3, 1]
    test_poses_neg = build_orbit_poses_with_convention(
        center,
        radius,
        18.0,
        use_neg_front=True,
        ref_up=ref_up,
    )
    test_poses_pos = build_orbit_poses_with_convention(
        center,
        radius,
        18.0,
        use_neg_front=False,
        ref_up=ref_up,
    )
    score_neg = _render_opacity_score(test_poses_neg[0])
    score_pos = _render_opacity_score(test_poses_pos[0])
    use_neg_front = score_neg >= score_pos
    poses = test_poses_neg if use_neg_front else test_poses_pos
    print(f"Selected orbit convention use_neg_front={use_neg_front} (scores: neg={score_neg:.6f}, pos={score_pos:.6f})")

    rendered_images = []

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
        rendered_images.append(outputs["rgb_gaussians"].detach().cpu())

    if was_training:
        trainer.set_train()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for idx, (ax, image, azim) in enumerate(zip(axes.flat, rendered_images, np.linspace(0.0, 315.0, 8))):
        img = image.numpy()
        img = np.clip(img, 0.0, 1.0)
        ax.imshow(img)
        ax.set_title(f"azim={int(azim)}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise object instances from checkpoint")
    parser.add_argument(
        "--resume_from",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame index used to query per-instance positions",
    )
    parser.add_argument(
        "--object_id",
        type=int,
        default=None,
        help="If provided, render 8 views of this object id",
    )
    parser.add_argument(
        "--object_type",
        type=str,
        default="auto",
        choices=["auto", "rigid", "smpl"],
        help="Object family to use when selecting object_id",
    )
    parser.add_argument(
        "--save_image",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "If set, save the rendered 8-view image. "
            "Provide optional output path; if omitted, a default name is used."
        ),
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open a display window (useful on headless servers)",
    )
    parser.add_argument(
        "opts",
        help="Override config options from command line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    _, dataset, trainer, log_dir = build_trainer_from_checkpoint(args.resume_from, args.opts)
    instances = extract_instance_points(trainer=trainer, frame_idx=args.frame_idx)

    if not instances:
        print("No rigid/smpl instances found in checkpoint for the selected frame.")
        return

    print_available_objects(instances)

    if args.object_id is None:
        return

    key = select_object_key(
        instances=instances,
        object_id=args.object_id,
        object_type=args.object_type,
    )
    points = instances[key]

    fig_title = f"Object {key[1]} ({key[0]}) - 8 views"
    fig = render_8_views_with_renderer(
        trainer=trainer,
        dataset=dataset,
        frame_idx=args.frame_idx,
        object_type=key[0],
        object_id=key[1],
        points_xyz=points,
        title=fig_title,
    )

    if args.save_image is not None:
        if args.save_image == "auto":
            out_name = f"instance_{key[0]}_{key[1]}_frame{args.frame_idx}_8views.png"
            out_path = os.path.join(log_dir, out_name)
        else:
            out_path = args.save_image
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved image to: {out_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
