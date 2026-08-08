
import multiprocessing
from unittest import result
from venv import logger
import torch
import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional, Any
from datasets.base.pixel_source import get_rays

import argparse
import pickle
from omegaconf import OmegaConf
from utils.misc import import_str
from datasets.driving_dataset import DrivingDataset
from models.video_utils import render_images, save_videos
from models.road_mesh import RoadMesh
import imageio
from post_train_difix_loop import tensor_to_image, image_to_array
from standalone_renderer import render_single_offset_novel_view
from utils.logging import MetricLogger, setup_logging
from PIL import Image
import wandb

# Setup logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _to_tensor(value, field_name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    raise TypeError(f"Field '{field_name}' expects Tensor/ndarray, got {type(value)}")


def _normalize_image_like(sample_img: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Normalize synthetic image tensor to match camera image tensor layout."""
    if sample_img.ndim == 3 and sample_img.shape != target_shape:
        if len(target_shape) == 3 and target_shape[2] == 3 and sample_img.shape[0] == 3:
            sample_img = sample_img.permute(1, 2, 0)
        elif len(target_shape) == 3 and target_shape[0] == 3 and sample_img.shape[2] == 3:
            sample_img = sample_img.permute(2, 0, 1)
    return sample_img


def _coerce_to_base_layout(value, base_tensor: torch.Tensor, field_name: str) -> torch.Tensor:
    tensor = _to_tensor(value, field_name)

    if field_name == "rendered_rgb":
        tensor = _normalize_image_like(tensor, base_tensor.shape[1:])

    # Handle scalar-to-vector append cases (e.g. unique_img_idx).
    if tensor.ndim == 0 and base_tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    # Handle alpha masks provided as [H, W] while base is [N, H, W, 1].
    if (
        field_name == "alpha_mask"
        and base_tensor.ndim == 4
        and tensor.ndim == 2
        and base_tensor.shape[-1] == 1
    ):
        tensor = tensor.unsqueeze(-1)

    if tensor.ndim == base_tensor.ndim - 1:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != base_tensor.ndim:
        raise ValueError(
            f"Field '{field_name}' rank mismatch: got {tensor.ndim}, expected {base_tensor.ndim}"
        )
    if tensor.shape[1:] != base_tensor.shape[1:]:
        raise ValueError(
            f"Field '{field_name}' shape mismatch: got {tuple(tensor.shape[1:])}, expected {tuple(base_tensor.shape[1:])}"
        )

    return tensor.to(device=base_tensor.device, dtype=base_tensor.dtype)


def _append_block(base_tensor: torch.Tensor, new_block: torch.Tensor) -> torch.Tensor:
    if new_block.ndim != base_tensor.ndim:
        raise ValueError("Cannot append block with different rank")
    if new_block.shape[1:] != base_tensor.shape[1:]:
        raise ValueError("Cannot append block with different trailing shape")

    base_len = base_tensor.shape[0]
    total_len = base_len + new_block.shape[0]
    out = torch.empty(
        (total_len, *base_tensor.shape[1:]),
        dtype=base_tensor.dtype,
        device=base_tensor.device,
    )
    out[:base_len] = base_tensor
    out[base_len:] = new_block
    return out


def _stack_field_for_samples(
    base_tensor: torch.Tensor,
    samples,
    field_name: str,
    required: bool,
    is_synthetic_mask: bool = False,
):
    blocks = []
    for sample in samples:
        ref_idx = int(sample["reference_frame_idx"])
        value = sample.get(field_name, None)
        if value is None:
            if required:
                raise KeyError(f"Synthetic sample missing required field '{field_name}'")
            if is_synthetic_mask:
                # For synthetic samples, do NOT clone misaligned reference-frame masks.
                # Novel-view pixels don't correspond to reference-frame class masks.
                # Use zeros instead to avoid incorrect supervision.
                blocks.append(torch.zeros_like(base_tensor[ref_idx:ref_idx + 1]))
            else:
                blocks.append(base_tensor[ref_idx:ref_idx + 1].clone())
        else:
            blocks.append(_coerce_to_base_layout(value, base_tensor, field_name))
    return torch.cat(blocks, dim=0)

def sanitize_broken_models(trainer, reason: str) -> int:
    """Remove partially initialized models that break resume/save flows."""
    removable = []
    for class_name, model in list(trainer.models.items()):
        try:
            model.get_param_groups()
            model.state_dict()
        except Exception as model_err:
            logger.warning(
                "Removing model '%s' during %s due to invalid state: %s",
                class_name,
                reason,
                model_err,
            )
            removable.append(class_name)

    for class_name in removable:
        trainer.models.pop(class_name, None)
        if class_name in trainer.gaussian_classes:
            trainer.gaussian_classes.pop(class_name, None)

    return len(removable)


def freeze_trainer_to_selected_object(trainer, object_type: str, object_id: int) -> None:
    if object_type != "rigid":
        raise NotImplementedError("train_vehicles only supports rigid objects.")

    frozen_classes = [class_name for class_name in trainer.models.keys() if class_name != "RigidNodes"]
    if hasattr(trainer, "set_frozen_model_classes"):
        trainer.set_frozen_model_classes(frozen_classes)
    else:
        for class_name, model in trainer.models.items():
            is_frozen = class_name in frozen_classes
            for param in model.parameters():
                param.requires_grad_(not is_frozen)
            if is_frozen:
                model.eval()

    rigid_model = trainer.models.get("RigidNodes", None)
    if rigid_model is None:
        raise ValueError("RigidNodes model is required to train a selected rigid object.")
    if not hasattr(rigid_model, "freeze_to_instance"):
        raise AttributeError("RigidNodes model does not support instance freezing.")

    rigid_model.freeze_to_instance(object_id)

    if hasattr(rigid_model, "ctrl_cfg") and rigid_model.ctrl_cfg is not None:
        rigid_model.ctrl_cfg.warmup_steps = int(1e12)

def integrate_synthetic_samples(dataset, synthetic_samples, synthetic_ratio: float = 1.0):
    """
    Integrate synthetic samples into the dataset.
    
    Args:
        dataset: DrivingDataset instance
        synthetic_samples: List of synthetic sample dicts
        synthetic_ratio: Fraction of synthetic samples to keep (0.0 to 1.0).
                        If < 1.0, randomly samples that fraction.
    """
    if synthetic_samples is None or len(synthetic_samples) == 0:
        return []

    pixel_source = dataset.pixel_source
    cam0 = pixel_source.camera_data[0]

    # If synthetic_ratio < 1.0, randomly sample the specified fraction
    if synthetic_ratio < 1.0:
        num_to_keep = max(1, int(len(synthetic_samples) * synthetic_ratio))
        sample_indices = torch.randperm(len(synthetic_samples))[:num_to_keep].tolist()
        synthetic_samples = [synthetic_samples[i] for i in sorted(sample_indices)]
        logger.info(
            "Sampling %d/%d synthetic samples (ratio %.2f) for ~70%% real / ~30%% synthetic split",
            len(synthetic_samples),
            len(synthetic_samples) + len(cam0),
            synthetic_ratio,
        )

    # Validate and normalize reference indices once.
    for idx, sample in enumerate(synthetic_samples):
        if "reference_frame_idx" not in sample:
            raise KeyError(f"Synthetic sample #{idx} missing 'reference_frame_idx'")
        sample["reference_frame_idx"] = int(sample["reference_frame_idx"])

    old_num_frames = len(cam0)
    num_new = len(synthetic_samples)

    missing_road_masks = sum(1 for sample in synthetic_samples if sample.get("road_masks", None) is None)
    if missing_road_masks > 0:
        if getattr(cam0, "road_masks", None) is not None:
            logger.warning(
                "road_masks missing for %d/%d synthetic samples; falling back to cloning reference-frame road masks.",
                missing_road_masks,
                num_new,
            )
        else:
            logger.warning(
                "road_masks missing for %d/%d synthetic samples and base dataset has no road_masks tensor; synthetic frames will not carry road_masks.",
                missing_road_masks,
                num_new,
            )

    cam0.cam_to_worlds = _append_block(
        cam0.cam_to_worlds,
        _stack_field_for_samples(
            cam0.cam_to_worlds,
            synthetic_samples,
            field_name="novel_c2w",
            required=True,
        ),
    )
    cam0.intrinsics = _append_block(
        cam0.intrinsics,
        _stack_field_for_samples(
            cam0.intrinsics,
            synthetic_samples,
            field_name="intrinsics",
            required=True,
        ),
    )

    if cam0.distortions is not None:
        cam0.distortions = _append_block(
            cam0.distortions,
            _stack_field_for_samples(
                cam0.distortions,
                synthetic_samples,
                field_name="distortions",
                required=False,
            ),
        )

    cam0.images = _append_block(
        cam0.images,
        _stack_field_for_samples(
            cam0.images,
            synthetic_samples,
            field_name="rendered_rgb",
            required=True,
        ),
    )
    cam0.alpha_masks = _append_block(
        cam0.alpha_masks,
        _stack_field_for_samples(
            cam0.alpha_masks,
            synthetic_samples,
            field_name="alpha_mask",
            required=True,
        ),
    )

    if cam0.normalized_time is not None:
        cam0.normalized_time = _append_block(
            cam0.normalized_time,
            _stack_field_for_samples(
                cam0.normalized_time,
                synthetic_samples,
                field_name="normalized_time",
                required=False,
            ),
        )

    cam0.unique_img_idx = _append_block(
        cam0.unique_img_idx,
        _stack_field_for_samples(
            cam0.unique_img_idx,
            synthetic_samples,
            field_name="unique_img_idx",
            required=False,
        ),
    )

    for tensor_field in [
        "dynamic_masks",
        "human_masks",
        "vehicle_masks",
        "sky_masks",
        "road_masks",
        "lidar_depth_maps",
        "image_error_maps",
    ]:
        field_data = getattr(cam0, tensor_field, None)
        if field_data is not None:
            # For mask fields, use is_synthetic_mask=True to avoid misaligned reference-frame masks on novel views
            is_mask_field = tensor_field in ["dynamic_masks", "human_masks", "vehicle_masks", "sky_masks", "road_masks"]
            appended = _append_block(
                field_data,
                _stack_field_for_samples(
                    field_data,
                    synthetic_samples,
                    field_name=tensor_field,
                    required=False,
                    is_synthetic_mask=is_mask_field,
                ),
            )
            setattr(cam0, tensor_field, appended)

    new_img_indices = [
        int(frame_idx * pixel_source.num_cams + cam0.unique_cam_idx)
        for frame_idx in range(old_num_frames, old_num_frames + num_new)
    ]
    dataset.train_indices.extend(new_img_indices)
    dataset.train_image_set.split_indices = dataset.train_indices

    if pixel_source.image_error_buffer is not None:
        current_len = pixel_source.image_error_buffer.shape[0]
        max_new_img_idx = max(new_img_indices)
        if max_new_img_idx >= current_len:
            pad_len = int(max_new_img_idx + 1 - current_len)
            pad_value = pixel_source.image_error_buffer.mean() if current_len > 0 else torch.tensor(1.0, device=pixel_source.device)
            pad_tensor = torch.full(
                (pad_len,),
                float(pad_value.item()),
                dtype=pixel_source.image_error_buffer.dtype,
                device=pixel_source.image_error_buffer.device,
            )
            pixel_source.image_error_buffer = torch.cat([pixel_source.image_error_buffer, pad_tensor], dim=0)

    logger.info(
        "Integrated %d synthetic samples as img indices [%d, %d]",
        num_new,
        int(new_img_indices[0]),
        int(new_img_indices[-1]),
    )
    return new_img_indices

def cache_image_errors(cfg, dataset, trainer, step):
    logger.info("Caching image error...")
    trainer.set_eval()
    with torch.no_grad():
        dataset.pixel_source.update_downscale_factor(
            1 / dataset.pixel_source.buffer_downscale
        )
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
        )
        dataset.pixel_source.reset_downscale_factor()
        dataset.pixel_source.update_image_error_maps(render_results)
        merged_error_video = dataset.pixel_source.get_image_error_video(
            dataset.layout
        )
        imageio.mimsave(
            os.path.join(
                cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
            ),
            merged_error_video,
            fps=cfg.render.fps,
        )
    logger.info("Done caching rgb error maps")


def _blend_mesh_with_outputs(
    outputs: Dict[str, torch.Tensor],
    mesh: RoadMesh,
    image_h: int,
    image_w: int,
    mesh_blend_alpha: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """
    Blend mesh rendering with Gaussian rendering outputs.
    
    This function composites the road mesh texture onto the Gaussian renderings,
    allowing the mesh to contribute to the final rendered output while the 
    Gaussians handle dynamic content (vehicles, people, etc.).
    
    Args:
        outputs: Dictionary with Gaussian rendering outputs including 'rgbs'
        mesh: RoadMesh instance
        image_h: Image height in pixels
        image_w: Image width in pixels
        mesh_blend_alpha: Blend factor for mesh contribution (0.0-1.0)
        
    Returns:
        Updated outputs dictionary with blended RGB
    """
    try:
        # Get mesh texture
        mesh_texture = mesh.get_texture_as_tensor()
        
        if mesh_texture is None or mesh_texture.shape[1] != 3:
            return outputs
        
        # Resize mesh texture to match output image size if needed
        if mesh_texture.shape[2] != image_h or mesh_texture.shape[3] != image_w:
            import torch.nn.functional as F
            mesh_texture = F.interpolate(
                mesh_texture,
                size=(image_h, image_w),
                mode='bilinear',
                align_corners=False,
            )
        
        # Blend mesh with Gaussian output
        if 'rgbs' in outputs:
            gaussian_rgb = outputs['rgbs']
            
            # Ensure shapes match (batch_size, H, W, 3)
            if gaussian_rgb.dim() == 3:
                gaussian_rgb = gaussian_rgb.unsqueeze(0)
            
            if mesh_texture.dim() == 4:
                mesh_rgb = mesh_texture.permute(0, 2, 3, 1)  # (1, H, W, 3)
            else:
                mesh_rgb = mesh_texture
            
            # Ensure both are on same device
            mesh_rgb = mesh_rgb.to(gaussian_rgb.device)
            
            # Simple alpha blending: output = gaussian * (1-alpha) + mesh * alpha
            blended_rgb = gaussian_rgb * (1.0 - mesh_blend_alpha) + mesh_rgb * mesh_blend_alpha
            
            outputs['rgbs'] = blended_rgb
            
            # Also store mesh RGB separately for inspection
            outputs['road_mesh_rgb'] = mesh_rgb.squeeze(0) if mesh_rgb.shape[0] == 1 else mesh_rgb
        
        return outputs
        
    except Exception as e:
        logger.debug(f"Error during mesh blending: {e}")
        return outputs


def _initialize_road_mesh(dataset, device, log_dir=None):
    """
    Initialize road mesh from LiDAR data and road masks, or load from checkpoint.
    
    This function:
    1. Checks for saved mesh checkpoint (if log_dir provided)
    2. If checkpoint exists, loads and returns it
    3. Otherwise, aggregates LiDAR points from road mask regions
    4. Creates a 3D mesh from the aggregated point cloud
    5. Initializes a texture buffer for RGB projection
    6. Projects camera images onto the mesh texture
    
    Args:
        dataset: DrivingDataset instance with loaded data
        device: PyTorch device for computation
        log_dir: Optional directory containing saved mesh checkpoint
        
    Returns:
        RoadMesh instance or None if initialization fails
    """
    try:
        # Check if mesh checkpoint exists
        if log_dir is not None:
            mesh_checkpoint_path = os.path.join(log_dir, "road_mesh.pth")
            if os.path.exists(mesh_checkpoint_path):
                logger.info(f"Loading road mesh from checkpoint: {mesh_checkpoint_path}")
                try:
                    road_mesh = RoadMesh.load_checkpoint(mesh_checkpoint_path, device=device)
                    logger.info("Road mesh loaded successfully from checkpoint")
                    return road_mesh
                except Exception as e:
                    logger.warning(f"Failed to load mesh checkpoint: {e}. Creating new mesh...")
        
        # Step 1: Aggregate LiDAR points in road regions
        logger.info("Step 1: Aggregating LiDAR points from road masks...")
        road_lidar_indices = dataset.get_lidar_indices_from_mask_region(mask_attr="road_masks")
        
        if len(road_lidar_indices) == 0:
            logger.warning("No LiDAR points found in road regions")
            return None
        
        road_pts, road_colors = dataset.get_lidar_points_from_mask_region(
            mask_attr="road_masks",
            num_samples=None,  # Use all available points
            return_color=True,
            device=device
        )
        
        logger.info(f"Extracted {len(road_pts):,} road LiDAR points")
        
        # Convert to numpy for mesh creation
        pts_xyz = road_pts.cpu().numpy() if isinstance(road_pts, torch.Tensor) else road_pts
        colors_arr = road_colors.cpu().numpy() if isinstance(road_colors, torch.Tensor) else road_colors
        
        # Step 2: Create mesh from point cloud
        logger.info("Step 2: Creating 3D mesh from point cloud...")
        cell_size = 0.5  # Grid resolution in meters
        road_mesh = RoadMesh.from_pointcloud(
            pts_xyz=pts_xyz,
            colors=colors_arr,
            cell_size=cell_size,
            device=device,
        )
        
        if len(road_mesh.vertices) == 0:
            logger.warning("Failed to create mesh from point cloud")
            return None
        
        # Step 3: Initialize texture buffer for RGB projection
        logger.info("Step 3: Initializing texture buffer...")
        resolution_ppm = 100.0
        road_mesh.initialize_texture_buffer(resolution_pixels_per_meter=resolution_ppm)
        
        # Step 4: Project images onto mesh texture
        logger.info("Step 4: Projecting camera images onto mesh texture...")
        try:
            projection_stats = road_mesh.project_images_onto_buffer(
                dataset=dataset,
                num_frames=None,  # Use all available frames
            )
            logger.info(f"Texture projection complete with stats: {projection_stats}")
        except Exception as proj_error:
            logger.warning(f"Image projection failed (mesh will use default texture): {proj_error}")
            # Continue even if projection fails - mesh will still render with default texture
        
        logger.info("Road mesh initialization complete!")
        return road_mesh
        
    except Exception as e:
        logger.error(f"Error during road mesh initialization: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_trainer(dataset, cfg, args, ckpt_to_load=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dataset.num_img_timesteps)
    # setup trainer
    trainer_cfg = dict(cfg.trainer)
    trainer_cfg.pop('resume_from', None)
    trainer_cfg.pop('resume_workflow_from', None)
    trainer = import_str(cfg.trainer.type)(
        **trainer_cfg,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )

    logger.info(
        "Trainer model classes: %s",
        list(trainer.model_config.keys()),
    )
    logger.info(
        "Trainer gaussian classes: %s",
        list(trainer.gaussian_classes.keys()),
    )
    print(f"Trainer model classes: {list(trainer.model_config.keys())}")
    print(f"Trainer gaussian classes: {list(trainer.gaussian_classes.keys())}")

    # initialize gaussians
    if not args.from_scratch:
        trainer.resume_from_checkpoint(ckpt_path=ckpt_to_load, load_only_model=True)
        logger.info(f"Resuming training from {ckpt_to_load}, starting at step {trainer.step}")
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}")

    # Initialize road mesh before training starts (STEP 0)
    # Check if mesh should be enabled (default True, unless --disable_mesh_road is set)
    enable_mesh = not getattr(args, "disable_mesh_road", False)
    
    if enable_mesh:
        try:
            logger.info("Initializing road mesh from LiDAR and road masks...")
            # When training from scratch, do not load an existing road mesh checkpoint.
            mesh_log_dir = None if getattr(args, "from_scratch", False) else cfg.log_dir
            if mesh_log_dir is not None:
                logger.info(f"Road mesh checkpoint directory: {mesh_log_dir}")
            road_mesh = _initialize_road_mesh(dataset, device, log_dir=mesh_log_dir)
            
            if road_mesh is not None:
                trainer.road_mesh = road_mesh
                trainer.mesh_blend_alpha = getattr(args, "mesh_blend_alpha", 1.0)
                logger.info(f"Road mesh successfully integrated into trainer (blend alpha: {trainer.mesh_blend_alpha})")
            else:
                logger.warning("Failed to initialize road mesh, continuing without mesh")
                trainer.road_mesh = None
        except Exception as e:
            logger.error(f"Error initializing road mesh: {e}")
            trainer.road_mesh = None
    else:
        logger.info("Mesh road reconstruction disabled (--disable_mesh_road flag set)")
        trainer.road_mesh = None

    if getattr(args, "selected_object_id", None) is not None:
        freeze_trainer_to_selected_object(
            trainer=trainer,
            object_type=args.selected_object_type,
            object_id=args.selected_object_id,
        )

    if args.enable_viewer:
        trainer.init_viewer(port=args.viewer_port)

    # Always re-initialize optimizer from scratch, never load from checkpoint
    try:
        trainer.initialize_optimizer()
    except AttributeError as e:
        logger.warning(f"Optimizer init failed on first attempt: {e}")
        removed = sanitize_broken_models(trainer, reason="optimizer init")
        if removed == 0:
            raise
        # After model removal, always re-initialize optimizer
        if hasattr(trainer, 'reset_optimizer_state'):
            trainer.reset_optimizer_state()  # If your trainer supports explicit reset
        trainer.initialize_optimizer()
        logger.info("Optimizer re-initialized after model removal.")
    # --- DUMMY OPTIMIZER STEP TO INITIALIZE STATE ---
    try:
        trainer.set_train()
        # Get a single batch from the training set
        image_infos, cam_infos = dataset.train_image_set.next(1.0)
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)
        outputs = trainer(image_infos, cam_infos)
        loss_dict = trainer.compute_losses(outputs, image_infos, cam_infos)
        trainer.optimizer_zero_grad()
        trainer.backward(loss_dict)
        trainer.optimizer.step()
        trainer.optimizer_zero_grad()
        logger.info("Dummy optimizer step performed to initialize optimizer state.")
    except Exception as e:
        logger.warning(f"Dummy optimizer step failed: {e}")
    return trainer


def run_training_loop(cfg, dataset, trainer, render_keys, args):
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)
    road_preview_freq = 3000
    
    # Add mesh render key if mesh is available
    mesh_render_keys = render_keys.copy()
    if hasattr(trainer, 'road_mesh') and trainer.road_mesh is not None:
        if "road_mesh_rgb" not in mesh_render_keys:
            mesh_render_keys.insert(0, "road_mesh_rgb")

    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        # training step
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()
        train_step_camera_downscale = trainer._get_downscale_factor()
        image_infos, cam_infos = dataset.train_image_set.next(train_step_camera_downscale)
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)

        # Forward pass: Gaussian + road-mesh composition happens inside trainer.forward()
        outputs = trainer(image_infos, cam_infos)
        
        trainer.update_visibility_filter()
        loss_dict = trainer.compute_losses(outputs, image_infos, cam_infos)

        # check for NaNs/Infs
        for k, v in loss_dict.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(f"Invalid value in loss {k} at step {step}")

        trainer.backward(loss_dict)
        trainer.postprocess_per_train_step(step=step)

        # logging
        with torch.no_grad():
            metric_dict = trainer.compute_metrics(outputs, image_infos)
        metric_logger.update(**{"train_metrics/"+k: v.item() for k,v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_"+k:v for k,v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/"+k:v.item() for k,v in loss_dict.items()})
        metric_logger.update(**{"train_stats/lr_"+group["name"]:group["lr"] for group in trainer.optimizer.param_groups})
 
        if args.enable_wandb:
            wandb.log({k:v.avg for k,v in metric_logger.meters.items()})

        # saving
        do_save = step > 0 and ((step % cfg.logging.saveckpt_freq == 0) or step == trainer.num_iters)
        if do_save:
            try:
                trainer.save_checkpoint(log_dir=cfg.log_dir, save_only_model=True, is_final=step==trainer.num_iters)
            except AttributeError as e:
                logger.warning(f"Checkpoint save failed due to model state issue: {e}")
                removed = sanitize_broken_models(trainer, reason="checkpoint save")
                if removed == 0:
                    raise
                logger.warning("Retrying checkpoint save after removing %d broken models", removed)
                trainer.save_checkpoint(log_dir=cfg.log_dir, save_only_model=True, is_final=step==trainer.num_iters)
             
            # Save mesh checkpoint alongside trainer checkpoint
            if hasattr(trainer, "road_mesh") and trainer.road_mesh is not None:
                try:
                    mesh_checkpoint_path = os.path.join(cfg.log_dir, "road_mesh.pth")
                    trainer.road_mesh.save_checkpoint(mesh_checkpoint_path)
                except Exception as e:
                    logger.warning(f"Failed to save mesh checkpoint: {e}")

        if step > 0 and step % road_preview_freq == 0:
            save_road_reference_preview(dataset, trainer, cfg.log_dir, args)

        # cache image errors
        if step>0 and trainer.optim_general.cache_buffer_freq>0 and step%trainer.optim_general.cache_buffer_freq==0:
            cache_image_errors(cfg, dataset, trainer, step)

    return step

def setup_render_keys(cfg, dataset):
    render_keys = [
        "gt_rgbs", "rgbs", "Background_rgbs", "Dynamic_rgbs",
        "RigidNodes_rgbs", "DeformableNodes_rgbs", "SMPLNodes_rgbs"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs")+1, "rgb_error_maps")
    return render_keys


def get_novel_view_quality_dir(ckpt_dir, args):
    return os.path.join(ckpt_dir, args.novel_view_quality_dir)


def save_road_reference_preview(dataset, trainer, ckpt_dir, args):
    quality_dir = get_novel_view_quality_dir(ckpt_dir, args)
    os.makedirs(quality_dir, exist_ok=True)

    # Render the reference road class from the same viewpoint as the other images.
    road_sample = render_single_offset_novel_view(
        dataset=dataset,
        trainer=trainer,
        frame_index=args.ref_frame,
        lateral_offset_m=0.0,
    )

    def _save_image_if_present(tensor, filename):
        if isinstance(tensor, torch.Tensor):
            img_tensor = tensor.detach().cpu()
            if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
                img_tensor = img_tensor[0]
            if img_tensor.ndim != 3:
                return
            img = tensor_to_image(img_tensor)
            img.save(os.path.join(quality_dir, filename))

    _save_image_if_present(
        road_sample.get("rendered_rgb", None),
        f"rgb_reference_frame{args.ref_frame}_step{trainer.step}.png",
    )
    _save_image_if_present(
        road_sample.get("background_rgb", None),
        f"background_reference_frame{args.ref_frame}_step{trainer.step}.png",
    )
    _save_image_if_present(
        road_sample.get("sky_rgb", None),
        f"sky_reference_frame{args.ref_frame}_step{trainer.step}.png",
    )
    _save_image_if_present(
        road_sample.get("road_mesh_rgb", None),
        f"road_mesh_reference_frame{args.ref_frame}_step{trainer.step}.png",
    )
    road_rgb = road_sample.get("road_rgb", None)
    if isinstance(road_rgb, torch.Tensor):
        road_img = tensor_to_image(road_rgb.cpu())
        road_img_path = os.path.join(
            quality_dir,
            f"road_reference_frame{args.ref_frame}_step{trainer.step}.png",
        )
        road_img.save(road_img_path)

    road_model = getattr(trainer, "models", {}).get("Road", None)
    if road_model is not None:
        road_points = getattr(road_model, "means", None)
        road_colors = getattr(road_model, "colors", None)
        if road_points is None and hasattr(road_model, "_means"):
            road_points = road_model._means.detach()
        if road_colors is None and hasattr(road_model, "_features_dc"):
            try:
                road_colors = road_model.colors
            except Exception:
                road_colors = None

        if isinstance(road_points, torch.Tensor):
            road_npz_path = os.path.join(
                quality_dir,
                f"road_reference_frame{args.ref_frame}_step{trainer.step}.npz",
            )
            road_npz = {
                "points": road_points.detach().cpu().numpy(),
            }
            if isinstance(road_colors, torch.Tensor):
                road_npz["colors"] = road_colors.detach().cpu().numpy()
            np.savez_compressed(road_npz_path, **road_npz)


def train_from_scratch(cfg, ckpt_path, args):
    # Build dataset from config
    dataset = DrivingDataset(cfg.data)
    render_keys = setup_render_keys(cfg, dataset)
    trainer = build_trainer(dataset, cfg, args, ckpt_to_load=ckpt_path)
    trainer.num_train_images = len(dataset.train_image_set)
    step = trainer.step

    save_road_reference_preview(dataset, trainer, os.path.dirname(ckpt_path), args)

    trainer.num_iters = trainer.step + args.split_steps
    run_training_loop(cfg, dataset, trainer, render_keys, args)
    # Save checkpoint
    # trainer.reinit_road_gaussians_from_dataset(dataset=dataset) # Reset before saving (might improve performance)
    trainer.save_checkpoint(log_dir=os.path.dirname(ckpt_path), save_only_model=True, is_final=False)
    with open(args.output_path, "wb") as f:
        pickle.dump({"status": "done"}, f)
        
    save_road_reference_preview(dataset, trainer, os.path.dirname(ckpt_path), args)


def cli_start_training():
    # Print GPU usage statistics
    print("\n=== GPU Usage Statistics (torch.cuda) ===")
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary())
        try:
            import subprocess
            print("\n=== GPU Usage Statistics (nvidia-smi) ===")
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"Could not run nvidia-smi: {e}")
    else:
        print("CUDA is not available.")

    parser = argparse.ArgumentParser(description="Train with synthetic samples (GPU isolated)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lateral_offset", type=float, required=True)
    parser.add_argument("--ref_frame", type=int, required=True)
    parser.add_argument("--split_steps", type=int, required=True, help="Number of steps to train with splitting enabled")
    parser.add_argument("--prune_steps", type=int, required=True, help="Number of steps to train with pruning enabled after splitting")
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--novel_view_quality_dir", type=str, default="novel_view_quality", help="Directory to save novel view quality results")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--selected_object_type", type=str, default="rigid", choices=["rigid"], help="Type of object to keep trainable")
    parser.add_argument("--selected_object_id", type=int, default=None, help="Selected object id to keep trainable")
    parser.add_argument("--synthetic_ratio", type=float, default=0.3, help="Fraction of synthetic samples to use (0.0-1.0). Use ~0.3 for 70%% real / 30%% synthetic split.")
    parser.add_argument("--enable_mesh_road", action="store_true", default=True, help="Enable mesh-based road reconstruction (default: True)")
    parser.add_argument("--disable_mesh_road", action="store_true", help="Disable mesh-based road reconstruction")
    parser.add_argument("--mesh_blend_alpha", type=float, default=1.0, help="Mesh blending factor (0.0-1.0, default 1.0)")

    args = parser.parse_args()

    # Load config from checkpoint directory
    ckpt_path = args.checkpoint_path
    ckpt_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    # Ensure log_dir is set to checkpoint directory
    cfg.log_dir = ckpt_dir

    if args.from_scratch:
        train_from_scratch(cfg, ckpt_path, args)
        return
    # Load synthetic samples from input_path
    with open(args.input_path, "rb") as f:
        synthetic_samples = pickle.load(f)

    # Build dataset
    dataset = DrivingDataset(cfg.data)
    render_keys = setup_render_keys(cfg, dataset)
    # Create fully opaque alpha masks for real frames
    for cam in dataset.pixel_source.camera_data.values():
        N, H, W, C = cam.images.shape  # assuming NHWC
        if not hasattr(cam, "alpha_masks") or cam.alpha_masks is None:
            # initialize all ones with shape (N, H, W, 1) to match synthetic alpha masks
            cam.alpha_masks = torch.ones((N, H, W, 1), device=cam.images.device)


    # Integrate synthetic samples in one shot to avoid repeated tensor reallocation/copy.
    integrate_synthetic_samples(dataset, synthetic_samples, synthetic_ratio=args.synthetic_ratio)

    # Train on both synthetic and ground truth images for split_steps
    trainer = build_trainer(dataset, cfg, args, ckpt_to_load=ckpt_path)
    trainer.num_train_images = len(dataset.train_image_set)
    trainer.num_iters = trainer.step + args.split_steps
    print(f"Training on synthetic + ground truth images for {args.split_steps} steps...")

    save_road_reference_preview(dataset, trainer, ckpt_dir, args)

    run_training_loop(cfg, dataset, trainer, render_keys, args)

    # 4. Save checkpoint
    trainer.save_checkpoint(log_dir=ckpt_dir, save_only_model=True, is_final=False)

    # 5. Evaluate and save quality in novel_view_quality folder
    eval_sample = render_single_offset_novel_view(
        dataset=dataset,
        trainer=trainer,
        frame_index=args.ref_frame,
        lateral_offset_m=-args.lateral_offset,
    )
    eval_rgb = eval_sample["rendered_rgb"]
    # Save only the rendered novel view image
    quality_dir = get_novel_view_quality_dir(ckpt_dir, args)
    os.makedirs(quality_dir, exist_ok=True)
    eval_raw_image = tensor_to_image(eval_rgb.cpu())
    novel_view_img_path = os.path.join(quality_dir, f"novelview_frame{args.ref_frame}_offset{args.lateral_offset}_step{trainer.step}.png")
    eval_raw_image.save(novel_view_img_path)

    eval_sample = render_single_offset_novel_view(
        dataset=dataset,
        trainer=trainer,
        frame_index=args.ref_frame,
        lateral_offset_m=args.lateral_offset,
    )
    eval_rgb = eval_sample["rendered_rgb"]
    # Save only the rendered novel view image
    quality_dir = get_novel_view_quality_dir(ckpt_dir, args)
    os.makedirs(quality_dir, exist_ok=True)
    eval_raw_image = tensor_to_image(eval_rgb.cpu())
    novel_view_img_path = os.path.join(quality_dir, f"novelview_frame{args.ref_frame}_offset{-args.lateral_offset}_step{trainer.step}.png")
    eval_raw_image.save(novel_view_img_path)

    eval_sample = render_single_offset_novel_view(
        dataset=dataset,
        trainer=trainer,
        frame_index=args.ref_frame,
        lateral_offset_m=-args.lateral_offset/2,
    )
    eval_rgb = eval_sample["rendered_rgb"]
    # Save only the rendered novel view image
    quality_dir = get_novel_view_quality_dir(ckpt_dir, args)
    os.makedirs(quality_dir, exist_ok=True)
    eval_raw_image = tensor_to_image(eval_rgb.cpu())
    novel_view_img_path = os.path.join(quality_dir, f"novelview_frame{args.ref_frame}_offset{args.lateral_offset/2}_step{trainer.step}.png")
    eval_raw_image.save(novel_view_img_path)


    # Write a dummy result to output_path to signal completion
    with open(args.output_path, "wb") as f:
        pickle.dump({"status": "done"}, f)

if __name__ == "__main__":
    import sys
    if "train_synthetic" in sys.argv:
        sys.argv.remove("train_synthetic")
        cli_start_training()
