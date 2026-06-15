import multiprocessing
from venv import logger
import torch
import numpy as np
import os
from datasets.base.pixel_source import get_rays

import argparse
import pickle
from omegaconf import OmegaConf
from utils.misc import import_str
from datasets.driving_dataset import DrivingDataset


_ROAD_MASK_WARNING_EMITTED = False


def _to_binary_mask(opacity_tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert an opacity map to a float32 binary mask in {0, 1}."""
    mask = opacity_tensor
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    return (mask > threshold).float()

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
        rendered_rgb = novel_outputs["rgb"].detach().cpu()
        background_rgb = novel_outputs.get("Background_rgb", None)
        sky_rgb = novel_outputs.get("rgb_sky", None)
        road_rgb = novel_outputs.get("Road_rgb", None)
        road_depth = novel_outputs.get("Road_depth", None)
        road_opacity = novel_outputs.get("Road_opacity", None)
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

    # Render sample
    result = render_single_offset_novel_view(dataset, minimal_trainer, args.frame_index, -args.lateral_offset)
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

    # Render multiple samples
    results = render_multiple_offset_novel_views(
        dataset, minimal_trainer, args.frame_index_list, args.lateral_offset_list
    )
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










