import multiprocessing
from unittest import result
from venv import logger
import argparse
import os
import pickle

import imageio
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from PIL import Image

from datasets.base.pixel_source import get_rays
from datasets.driving_dataset import DrivingDataset
from models.video_utils import render_images, save_videos
from post_train_difix_loop import image_to_array, tensor_to_image
from standalone_renderer import render_single_offset_novel_view
from utils.logging import MetricLogger, setup_logging
from utils.misc import import_str


def _append_frame_tensor_via_cpu(base_tensor, ref_idx=None, new_tensor=None):
    """Append one frame to a frame-major tensor without peak GPU concatenation."""
    if base_tensor is None:
        return None

    base_cpu = base_tensor.detach().cpu()
    if new_tensor is None:
        if ref_idx is None:
            raise ValueError("ref_idx is required when new_tensor is not provided")
        new_cpu = base_cpu[ref_idx : ref_idx + 1].clone()
    else:
        if isinstance(new_tensor, np.ndarray):
            new_tensor = torch.from_numpy(new_tensor)
        if not isinstance(new_tensor, torch.Tensor):
            raise TypeError(f"Expected Tensor or ndarray, got {type(new_tensor)}")
        new_cpu = new_tensor.detach().cpu()
        if new_cpu.ndim == base_cpu.ndim - 1:
            new_cpu = new_cpu.unsqueeze(0)

    return torch.cat([base_cpu, new_cpu], dim=0)


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


def append_novel_sample_to_training_dataset(dataset, novel_sample):
    pixel_source = dataset.pixel_source
    cam0 = pixel_source.camera_data[0]
    ref_idx = novel_sample["reference_frame_idx"]
    field_devices = {}

    new_frame_idx = len(cam0)

    field_devices["cam_to_worlds"] = cam0.cam_to_worlds.device
    cam0.cam_to_worlds = _append_frame_tensor_via_cpu(
        cam0.cam_to_worlds,
        ref_idx=ref_idx,
        new_tensor=novel_sample["novel_c2w"],
    )
    cam0.cam_to_worlds = cam0.cam_to_worlds.to(field_devices["cam_to_worlds"])

    field_devices["intrinsics"] = cam0.intrinsics.device
    cam0.intrinsics = _append_frame_tensor_via_cpu(
        cam0.intrinsics,
        ref_idx=ref_idx,
        new_tensor=novel_sample["intrinsics"],
    )
    cam0.intrinsics = cam0.intrinsics.to(field_devices["intrinsics"])

    if cam0.distortions is not None:
        field_devices["distortions"] = cam0.distortions.device
        cam0.distortions = _append_frame_tensor_via_cpu(cam0.distortions, ref_idx=ref_idx)
        cam0.distortions = cam0.distortions.to(field_devices["distortions"])

    field_devices["images"] = cam0.images.device
    cam0.images = _append_frame_tensor_via_cpu(
        cam0.images,
        ref_idx=ref_idx,
        new_tensor=novel_sample["rendered_rgb"],
    )
    cam0.images = cam0.images.to(field_devices["images"])

    field_devices["alpha_masks"] = cam0.alpha_masks.device
    cam0.alpha_masks = _append_frame_tensor_via_cpu(
        cam0.alpha_masks,
        ref_idx=ref_idx,
        new_tensor=novel_sample["alpha_mask"],
    )
    cam0.alpha_masks = cam0.alpha_masks.to(field_devices["alpha_masks"])

    if cam0.normalized_time is not None:
        field_devices["normalized_time"] = cam0.normalized_time.device
        cam0.normalized_time = _append_frame_tensor_via_cpu(cam0.normalized_time, ref_idx=ref_idx)
        cam0.normalized_time = cam0.normalized_time.to(field_devices["normalized_time"])

    field_devices["unique_img_idx"] = cam0.unique_img_idx.device
    cam0.unique_img_idx = _append_frame_tensor_via_cpu(cam0.unique_img_idx, ref_idx=ref_idx)
    cam0.unique_img_idx = cam0.unique_img_idx.to(field_devices["unique_img_idx"])

    for tensor_field in [
        "dynamic_masks",
        "human_masks",
        "vehicle_masks",
        "sky_masks",
        "lidar_depth_maps",
        "image_error_maps",
    ]:
        field_data = getattr(cam0, tensor_field, None)
        if field_data is not None:
            field_devices[tensor_field] = field_data.device
            appended = _append_frame_tensor_via_cpu(field_data, ref_idx=ref_idx)
            setattr(cam0, tensor_field, appended.to(field_devices[tensor_field]))

    new_img_idx = new_frame_idx * pixel_source.num_cams + cam0.unique_cam_idx
    dataset.train_indices.append(int(new_img_idx))
    dataset.train_image_set.split_indices = dataset.train_indices

    if pixel_source.image_error_buffer is not None:
        current_len = pixel_source.image_error_buffer.shape[0]
        if new_img_idx >= current_len:
            pad_len = int(new_img_idx + 1 - current_len)
            pad_value = pixel_source.image_error_buffer.mean() if current_len > 0 else torch.tensor(1.0, device=pixel_source.device)
            pad_tensor = torch.full(
                (pad_len,),
                float(pad_value.item()),
                dtype=pixel_source.image_error_buffer.dtype,
                device=pixel_source.image_error_buffer.device,
            )
            pixel_source.image_error_buffer = torch.cat([pixel_source.image_error_buffer, pad_tensor], dim=0)

    logger.info(f"Appended novel training sample as synthetic image idx {new_img_idx}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return int(new_img_idx)

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

    # initialize gaussians
    if ckpt_to_load is not None:
        trainer.resume_from_checkpoint(ckpt_path=ckpt_to_load, load_only_model=True)
        logger.info(f"Resuming training from {ckpt_to_load}, starting at step {trainer.step}")
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}")

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

        outputs = trainer(image_infos, cam_infos)
        trainer.update_visibility_filter()
        # loss_dict = trainer.compute_losses(outputs, image_infos, cam_infos)
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


def cli_start_training():
    # Explicitly enable gaussian splitting for all models
    def enable_gaussian_splitting_and_set_nsplits(trainer, n_split_samples=2):
        if hasattr(trainer, 'models'):
            for model in trainer.models.values():
                # Enable splitting and set parameters
                if hasattr(model, 'ctrl_cfg') and isinstance(model.ctrl_cfg, dict):
                    model.ctrl_cfg['enable_gaussian_splitting'] = True
                    model.ctrl_cfg['n_split_samples'] = n_split_samples
                    model.ctrl_cfg['stop_split_at'] = int(1e9)
                elif hasattr(model, 'ctrl_cfg'):
                    try:
                        model.ctrl_cfg.enable_gaussian_splitting = True
                        model.ctrl_cfg.n_split_samples = n_split_samples
                        model.ctrl_cfg.stop_split_at = int(1e9)
                    except Exception:
                        pass

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
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    disable_splitting_steps = 1000
    loss_threshold = 0.002

    args = parser.parse_args()

    # Load config from checkpoint directory
    ckpt_path = args.checkpoint_path
    ckpt_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    # Ensure log_dir is set to checkpoint directory
    cfg.log_dir = ckpt_dir

    # Load synthetic samples from input_path
    with open(args.input_path, "rb") as f:
        synthetic_samples = pickle.load(f)

    # Build dataset
    dataset = DrivingDataset(cfg.data)
    render_keys = setup_render_keys(cfg, dataset)
    # Create empty alpha masks for real frames
    for cam in dataset.pixel_source.camera_data:
        N, H, W, C = cam.images.shape  # assuming NHWC
        if not hasattr(cam, "alpha_masks") or cam.alpha_masks is None:
            # initialize all ones
            cam.alpha_masks = torch.ones((N, H, W), device=cam.images.device)

    for sample in synthetic_samples:
        # Move all tensors in prev_sample to the minimal_dataset's device before appending
        sample_on_device = {k: (v.to(dataset.pixel_source.camera_data[0].images.device) if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}
        # Convert numpy arrays to torch tensors
        for k, v in sample_on_device.items():
            if isinstance(v, np.ndarray):
                sample_on_device[k] = torch.from_numpy(v)
        # Ensure rendered_rgb is CHW and matches cam0.images
        img = sample_on_device.get("rendered_rgb", None)
        if img is not None and hasattr(dataset.pixel_source.camera_data[0], "images"):
            cam0 = dataset.pixel_source.camera_data[0]
            # cam0.images shape: (N, C, H, W) or (N, H, W, C)
            if cam0.images.ndim == 4:
                target_shape = cam0.images.shape[1:]
                if img.shape != target_shape:
                    # If cam0.images is NHWC and img is CHW, convert to HWC
                    if len(target_shape) == 3 and target_shape[2] == 3 and img.shape[0] == 3:
                        sample_on_device["rendered_rgb"] = img.permute(1, 2, 0)
                    # If cam0.images is NCHW and img is HWC, convert to CHW
                    elif len(target_shape) == 3 and target_shape[0] == 3 and img.shape[2] == 3:
                        sample_on_device["rendered_rgb"] = img.permute(2, 0, 1)
                    else:
                        raise ValueError(f"Image shape mismatch: {img.shape} vs {target_shape}")
        append_novel_sample_to_training_dataset(dataset, sample_on_device)
        # Free unused VRAM after each sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Fallback for synthetic images that dont have alpha masks
    for cam in dataset.pixel_source.camera_data:
        N, H, W, C = cam.images.shape  # assuming NHWC
        if not hasattr(cam, "alpha_masks") or cam.alpha_masks is None:
            # initialize all ones
            cam.alpha_masks = torch.ones((N, H, W), device=cam.images.device)

    trainer = build_trainer(dataset, cfg, args, ckpt_to_load=ckpt_path)
    enable_gaussian_splitting_and_set_nsplits(trainer, n_split_samples=2)
    trainer.num_train_images = len(dataset.train_image_set)
    step = trainer.step
    splitting_disabled = False
    # Ensure repaired_tensor is a torch.Tensor
    last_sample = synthetic_samples[-1]
    rendered_rgb = last_sample["rendered_rgb"]
    if not isinstance(rendered_rgb, torch.Tensor):
        repaired_tensor = torch.from_numpy(rendered_rgb).to(trainer.device)
    else:
        repaired_tensor = rendered_rgb.to(trainer.device)
    repaired_image = tensor_to_image(repaired_tensor.cpu())

    # Start a training loop here
    last_image_save_step = None
    while True:
        trainer.num_iters = trainer.step + 250
        step = run_training_loop(cfg, dataset, trainer, render_keys, args)

        eval_sample = render_single_offset_novel_view(
            dataset=dataset,
            trainer=trainer,
            lateral_offset_m=-args.lateral_offset,
        )
        eval_rgb = eval_sample["rendered_rgb"]
        # Ensure both tensors are CHW for loss calculation
        if eval_rgb.shape != repaired_tensor.shape:
            # If eval_rgb is HWC and repaired_tensor is CHW, permute eval_rgb
            if eval_rgb.shape[-1] == 3 and repaired_tensor.shape[0] == 3:
                eval_rgb = eval_rgb.permute(2, 0, 1)
            # If eval_rgb is CHW and repaired_tensor is HWC, permute repaired_tensor
            elif eval_rgb.shape[0] == 3 and repaired_tensor.shape[-1] == 3:
                repaired_tensor = repaired_tensor.permute(1, 2, 0)
        # Ensure both tensors are on the same device
        eval_rgb = eval_rgb.to(trainer.device)
        repaired_tensor = repaired_tensor.to(trainer.device)
        eval_loss = torch.nn.functional.mse_loss(eval_rgb, repaired_tensor).item()
        print(f"-----> Novel view loss at step {step}: {eval_loss}. Will continue training until loss < {loss_threshold} <-----")

        # Save side-by-side comparison every 1000 steps, even if not aligned
        if last_image_save_step is None or (step - last_image_save_step) >= 1000:
            eval_raw_image = tensor_to_image(eval_rgb.cpu())
            side_by_side_final = Image.new('RGB', (eval_raw_image.width + repaired_image.width, eval_raw_image.height))
            side_by_side_final.paste(eval_raw_image, (0, 0))
            side_by_side_final.paste(repaired_image, (eval_raw_image.width, 0))
            side_by_side_final.save(os.path.join(ckpt_dir, "synthetic_image_samples", f"sidebyside_final_{step}_ref_{eval_sample['reference_frame_idx']}.png"))
            last_image_save_step = step

        if eval_loss < loss_threshold and not splitting_disabled:
            set_gaussian_splitting = lambda t, e: None
            if hasattr(trainer, 'models'):
                for model in trainer.models.values():
                    if hasattr(model, 'ctrl_cfg') and isinstance(model.ctrl_cfg, dict):
                        model.ctrl_cfg['enable_gaussian_splitting'] = False
                        model.ctrl_cfg['stop_split_at'] = 0
                    elif hasattr(model, 'ctrl_cfg'):
                        try:
                            model.ctrl_cfg.enable_gaussian_splitting = False
                            model.ctrl_cfg.stop_split_at = 0
                        except Exception:
                            pass
            splitting_disabled = True
            disable_steps = 0
            print(f"Disabling gaussian splitting for {disable_splitting_steps} steps.")
            while disable_steps < disable_splitting_steps:
                trainer.num_iters = trainer.step + min(250, disable_splitting_steps - disable_steps)
                step = run_training_loop(cfg, dataset, trainer, render_keys, args)
                disable_steps += min(250, disable_splitting_steps - disable_steps)
            break
    trainer.save_checkpoint(log_dir=ckpt_dir, save_only_model=True, is_final=False)

    # Write a dummy result to output_path to signal completion
    with open(args.output_path, "wb") as f:
        pickle.dump({"status": "done"}, f)

if __name__ == "__main__":
    import sys
    if "train_synthetic" in sys.argv:
        sys.argv.remove("train_synthetic")
        cli_start_training()
