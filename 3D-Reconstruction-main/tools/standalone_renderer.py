import multiprocessing
from venv import logger
import torch
import numpy as np
import os
from datasets.base.pixel_source import get_rays

import argparse
import pickle
from omegaconf import OmegaConf
from PIL import Image
from utils.misc import import_str
from datasets.driving_dataset import DrivingDataset
from models.road_mesh import RoadMesh


_ROAD_MASK_WARNING_EMITTED = False


def _to_binary_mask(opacity_tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert an opacity map to a float32 binary mask in {0, 1}."""
    mask = opacity_tensor
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    return (mask > threshold).float()


def _save_rgb_tensor_png(rgb_tensor: torch.Tensor, output_path: str) -> None:
    """Save an RGB tensor in CHW/HWC format as a PNG."""
    tensor = rgb_tensor.detach().cpu()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"Expected RGB tensor with 3 channels, got shape {tuple(tensor.shape)}")
    image = np.clip(tensor.numpy(), 0.0, 1.0)
    Image.fromarray((image * 255.0).astype(np.uint8)).save(output_path)


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

def render_single_offset_novel_view(
    dataset,
    trainer,
    frame_index: int,
    lateral_offset_m: float,
):
    global _ROAD_MASK_WARNING_EMITTED
    # This code should shift the camera left by lateral_offset_m meters.
    pixel_source = dataset.pixel_source
    cam0 = pixel_source.camera_data[0]

    def _build_novel_camera_from_reference(frame_idx: int):
        ref_image_infos_local, ref_cam_infos_local = cam0.get_image(frame_idx)
        ref_c2w_local = cam0.cam_to_worlds[frame_idx].clone()
        lateral_axis_local = ref_c2w_local[:3, 0]
        lateral_axis_local = lateral_axis_local / (torch.linalg.norm(lateral_axis_local) + 1e-8)
        novel_c2w_local = ref_c2w_local.clone()
        novel_c2w_local[:3, 3] = ref_c2w_local[:3, 3] + lateral_offset_m * lateral_axis_local

        intrinsics_local = cam0.intrinsics[frame_idx].clone()
        novel_cam_infos_local = {
            "cam_id": ref_cam_infos_local["cam_id"],
            "cam_name": ref_cam_infos_local["cam_name"],
            "camera_to_world": novel_c2w_local,
            "height": ref_cam_infos_local["height"],
            "width": ref_cam_infos_local["width"],
            "intrinsics": intrinsics_local,
        }
        return ref_image_infos_local, ref_cam_infos_local, intrinsics_local, novel_c2w_local, novel_cam_infos_local

    # Use a fixed frame number (e.g., 100)
    fixed_frame_idx = frame_index
    reference_frame_idx = fixed_frame_idx
    ref_image_infos, ref_cam_infos, intrinsics, novel_c2w, novel_cam_infos = _build_novel_camera_from_reference(fixed_frame_idx)

    logger.info(
        "Rendering novel view: frame_index=%s lateral_offset_m=%.3f has_road_mesh=%s mesh_blend_alpha=%s",
        reference_frame_idx,
        lateral_offset_m,
        bool(getattr(trainer, "road_mesh", None) is not None),
        getattr(trainer, "mesh_blend_alpha", None),
    )

    H, W = cam0.HEIGHT, cam0.WIDTH
    x, y = torch.meshgrid(
        torch.arange(W, device=cam0.device),
        torch.arange(H, device=cam0.device),
        indexing="xy",
    )
    origins, viewdirs, direction_norm = get_rays(x.flatten(), y.flatten(), novel_c2w, intrinsics)
    origins = origins.reshape(H, W, 3)
    viewdirs = viewdirs.reshape(H, W, 3)
    direction_norm = direction_norm.reshape(H, W, 1)
    pixel_coords = torch.stack([y.float() / H, x.float() / W], dim=-1)

    novel_image_infos = {
        "origins": origins,
        "viewdirs": viewdirs,
        "direction_norm": direction_norm,
        "pixel_coords": pixel_coords,
        "normed_time": ref_image_infos["normed_time"],
        "img_idx": ref_image_infos["img_idx"],
        "frame_idx": ref_image_infos["frame_idx"],
    }
    with torch.no_grad():
        trainer.set_eval()
        if hasattr(trainer, "render_each_class"):
            trainer.render_each_class = True
        has_road_class = bool(getattr(trainer, "gaussian_classes", None)) and "Road" in trainer.gaussian_classes
        for k, v in novel_image_infos.items():
            if isinstance(v, torch.Tensor):
                novel_image_infos[k] = v.cuda(non_blocking=True)
        for k, v in novel_cam_infos.items():
            if isinstance(v, torch.Tensor):
                novel_cam_infos[k] = v.cuda(non_blocking=True)
        novel_outputs = trainer(novel_image_infos, novel_cam_infos, novel_view=True)
        logger.info(
            "Novel view outputs: keys=%s road_mesh_present=%s road_mesh_opacity_present=%s road_rgb_present=%s",
            sorted(novel_outputs.keys()),
            "road_mesh_rgb" in novel_outputs,
            "road_mesh_opacity" in novel_outputs,
            "Road_rgb" in novel_outputs,
        )
        road_mesh_rgb = novel_outputs.get("road_mesh_rgb", None)
        if isinstance(road_mesh_rgb, torch.Tensor):
            road_mesh_rgb_tensor = road_mesh_rgb.detach()
            logger.info(
                "Road mesh RGB stats: shape=%s min=%.4f max=%.4f mean=%.4f",
                tuple(road_mesh_rgb_tensor.shape),
                float(road_mesh_rgb_tensor.min().item()),
                float(road_mesh_rgb_tensor.max().item()),
                float(road_mesh_rgb_tensor.mean().item()),
            )
        elif getattr(trainer, "road_mesh", None) is not None:
            logger.warning("Trainer has a road_mesh, but the forward pass did not return road_mesh_rgb.")
        if "road_mesh_opacity" in novel_outputs and isinstance(novel_outputs["road_mesh_opacity"], torch.Tensor):
            road_mesh_opacity_tensor = novel_outputs["road_mesh_opacity"].detach()
            logger.info(
                "Road mesh opacity stats: shape=%s min=%.4f max=%.4f mean=%.4f",
                tuple(road_mesh_opacity_tensor.shape),
                float(road_mesh_opacity_tensor.min().item()),
                float(road_mesh_opacity_tensor.max().item()),
                float(road_mesh_opacity_tensor.mean().item()),
            )
        elif getattr(trainer, "road_mesh", None) is not None:
            logger.warning(
                "Trainer has a road_mesh, but the forward pass did not return road_mesh_opacity/road_mesh_rgb."
            )
        rendered_rgb = novel_outputs["rgb"].detach().cpu()
        background_rgb = novel_outputs.get("Background_rgb", None)
        sky_rgb = novel_outputs.get("rgb_sky", None)
        road_rgb = novel_outputs.get("Road_rgb", None)
        road_depth = novel_outputs.get("Road_depth", None)
        road_opacity = novel_outputs.get("Road_opacity", None)
        road_mesh_rgb = novel_outputs.get("road_mesh_rgb", None)
        road_mask = None
        if "Road_opacity" in novel_outputs:
            road_mask = _to_binary_mask(novel_outputs["Road_opacity"]).detach().cpu()
        elif not _ROAD_MASK_WARNING_EMITTED:
            if has_road_class:
                logger.warning(
                    "Requested road mask from Road class, but the trainer did not return Road_opacity; road_masks will be None."
                )
            else:
                logger.warning(
                    "Requested road mask from Road class, but this checkpoint/config has no Road gaussian class; road_masks will be None."
                )
            _ROAD_MASK_WARNING_EMITTED = True

    logger.info(
        f"Rendered one novel view from cam0 reference frame {reference_frame_idx} "
        f"with lateral offset {lateral_offset_m:.3f}m"
    )
    return {
        "reference_frame_idx": reference_frame_idx,
        "rendered_rgb": rendered_rgb.cpu(),
        "background_rgb": background_rgb.detach().cpu() if isinstance(background_rgb, torch.Tensor) else None,
        "sky_rgb": sky_rgb.detach().cpu() if isinstance(sky_rgb, torch.Tensor) else None,
        "road_rgb": road_rgb.detach().cpu() if isinstance(road_rgb, torch.Tensor) else None,
        "road_depth": road_depth.detach().cpu() if isinstance(road_depth, torch.Tensor) else None,
        "road_opacity": road_opacity.detach().cpu() if isinstance(road_opacity, torch.Tensor) else None,
        "road_mesh_rgb": road_mesh_rgb.detach().cpu() if isinstance(road_mesh_rgb, torch.Tensor) else None,
        "reference_rgb": ref_image_infos["pixels"].detach().cpu(),
        "alpha_mask": novel_outputs["opacity"].detach().cpu(),
        "road_masks": road_mask,
        "novel_c2w": novel_c2w.detach().cpu(),
        "intrinsics": intrinsics.detach().cpu(),
    }

def render_multiple_offset_novel_views(
    dataset,
    trainer,
    frame_indices,
    lateral_offsets,
):
    results = []
    for frame_idx, offset in zip(frame_indices, lateral_offsets):
        result = render_single_offset_novel_view(dataset, trainer, frame_idx, -offset)
        results.append(result)
    return results


# CLI entrypoint for GPU-isolated rendering
def cli_render_novel_sample():
    parser = argparse.ArgumentParser(description="Render novel sample (GPU isolated)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--frame_index", type=int, required=True)
    parser.add_argument("--lateral_offset", type=float, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mesh_blend_alpha", type=float, default=1.0, help="Mesh blending factor (0.0-1.0, default 1.0)")
    args = parser.parse_args()

    # Load config from checkpoint directory
    ckpt_path = args.checkpoint_path
    ckpt_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)

    # Build minimal dataset
    dataset = DrivingDataset(cfg.data)

    # Remove resume_from and resume_workflow_from from cfg.trainer to prevent autoloading
    trainer_cfg = dict(cfg.trainer)
    trainer_cfg.pop('resume_from', None)
    trainer_cfg.pop('resume_workflow_from', None)
    # Build minimal trainer (SingleTrainer or as specified)
    minimal_trainer = import_str(cfg.trainer.type)(
        **trainer_cfg,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    minimal_trainer.resume_from_checkpoint(ckpt_path=ckpt_path, load_only_model=True)
    _restore_textured_road_mesh(minimal_trainer, ckpt_dir)
    logger.info(
        "Checkpoint loaded: has_road_mesh=%s mesh_blend_alpha_before_set=%s",
        bool(getattr(minimal_trainer, "road_mesh", None) is not None),
        getattr(minimal_trainer, "mesh_blend_alpha", None),
    )
 
    # Set mesh blending alpha
    minimal_trainer.mesh_blend_alpha = args.mesh_blend_alpha
    logger.info("Renderer mesh_blend_alpha set to %.3f", minimal_trainer.mesh_blend_alpha)

    # Render sample
    result = render_single_offset_novel_view(dataset, minimal_trainer, args.frame_index, -args.lateral_offset)
    if isinstance(result.get("road_mesh_rgb"), torch.Tensor):
        road_mesh_png_path = os.path.splitext(args.output_path)[0] + "_road_mesh_rgb.png"
        _save_rgb_tensor_png(result["road_mesh_rgb"], road_mesh_png_path)
        logger.info("Saved rasterized road mesh RGB to %s", road_mesh_png_path)
    # Move tensors to cpu for serialization
    result = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}

    # Save result to file (pickle)
    with open(args.output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved rendered sample to {args.output_path}")

def cli_render_novel_sample_list():
    parser = argparse.ArgumentParser(description="Render novel sample list (GPU isolated)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--frame_index_list", nargs='+', type=int, required=True, help="List of frame indices")
    parser.add_argument("--lateral_offset_list", nargs='+', type=float, required=True, help="List of lateral offsets")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mesh_blend_alpha", type=float, default=1.0, help="Mesh blending factor (0.0-1.0, default 1.0)")
    args = parser.parse_args()

    # Load config from checkpoint directory
    ckpt_path = args.checkpoint_path
    ckpt_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)

    # Build minimal dataset
    dataset = DrivingDataset(cfg.data)

    # Remove resume_from and resume_workflow_from from cfg.trainer to prevent autoloading
    trainer_cfg = dict(cfg.trainer)
    trainer_cfg.pop('resume_from', None)
    trainer_cfg.pop('resume_workflow_from', None)
    # Build minimal trainer (SingleTrainer or as specified)
    minimal_trainer = import_str(cfg.trainer.type)(
        **trainer_cfg,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    minimal_trainer.resume_from_checkpoint(ckpt_path=ckpt_path, load_only_model=True)
    _restore_textured_road_mesh(minimal_trainer, ckpt_dir)
    logger.info(
        "Checkpoint loaded for list render: has_road_mesh=%s mesh_blend_alpha_before_set=%s",
        bool(getattr(minimal_trainer, "road_mesh", None) is not None),
        getattr(minimal_trainer, "mesh_blend_alpha", None),
    )

    # Set mesh blending alpha
    minimal_trainer.mesh_blend_alpha = args.mesh_blend_alpha
    logger.info("Renderer mesh_blend_alpha set to %.3f", minimal_trainer.mesh_blend_alpha)

    # Render multiple samples
    results = render_multiple_offset_novel_views(
        dataset, minimal_trainer, args.frame_index_list, args.lateral_offset_list
    )
    for idx, result in enumerate(results):
        if isinstance(result.get("road_mesh_rgb"), torch.Tensor):
            road_mesh_png_path = f"{os.path.splitext(args.output_path)[0]}_{idx}_road_mesh_rgb.png"
            _save_rgb_tensor_png(result["road_mesh_rgb"], road_mesh_png_path)
            logger.info("Saved rasterized road mesh RGB to %s", road_mesh_png_path)
    # Move tensors to cpu for serialization
    results_cpu = [
        {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in res.items()}
        for res in results
    ]

    # Save result to file (pickle)
    with open(args.output_path, "wb") as f:
        pickle.dump(results_cpu, f)

    print(f"Saved rendered samples to {args.output_path}")

if __name__ == "__main__":
    import sys
    if "render_novel_sample" in sys.argv:
        sys.argv.remove("render_novel_sample")
        cli_render_novel_sample()
    elif "render_novel_sample_list" in sys.argv:
        sys.argv.remove("render_novel_sample_list")
        cli_render_novel_sample_list()





