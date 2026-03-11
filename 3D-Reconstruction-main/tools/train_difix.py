from omegaconf import OmegaConf
import numpy as np
import os
import time
import wandb
import random
import imageio
import logging
import argparse

import torch
from tools.eval import do_evaluation, apply_render_frame_limit
from utils.misc import import_str
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from models.video_utils import render_images, save_videos
from datasets.driving_dataset import DrivingDataset
from datasets.base.pixel_source import get_rays
from tools.difix_sender_receiver import difix_repair
from PIL import Image

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def tensor_to_image(tensor):
    # Ensure HWC format
    if tensor.ndim == 3 and tensor.shape[0] == 3:  # CHW
        tensor = tensor.permute(1, 2, 0)
    img = tensor.numpy()
    # Clamp values to [0, 1] and scale to [0, 255]
    img = np.clip(img, 0, 1) * 255
    img = img.astype(np.uint8)
    return Image.fromarray(img)

def image_to_tensor(img: Image.Image, normalize: bool = True) -> torch.Tensor:
    img_np = np.array(img)  # H x W x C, uint8
    if normalize:
        img_np = img_np.astype(np.float32) / 255.0
    else:
        img_np = img_np.astype(np.float32)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    return tensor

def set_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup(args):
    # get config
    cfg = OmegaConf.load(args.config_file)

    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")

    assert (
        "dataset" in cfg or "data" in cfg
    ), "Please specify dataset in config or data in config"

    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)

    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)

    # direct CLI override for training iterations
    if args.num_iters is not None:
        cfg.trainer.optim.num_iters = int(args.num_iters)

    log_dir = os.path.join(args.output_root, args.project, args.run_name)

    # update config and create log dir
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in [
        "images",
        "videos",
        "metrics",
        "configs_bk",
        "buffer_maps",
        "backup",
        "synthetic_image_samples"
    ]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)

    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # also save a backup copy
    saved_cfg_path_bk = os.path.join(
        log_dir, "configs_bk", f"config_{current_time}.yaml"
    )
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")

    # Backup codes
    backup_project(
        os.path.join(log_dir, "backup"),
        "./",
        ["configs", "datasets", "models", "utils", "tools"],
        [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"],
    )
    return cfg


def build_dataset(data):
    # build dataset
    dataset = DrivingDataset(data)
    return dataset


def build_trainer(dataset, cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # initialize gaussians
    if args.resume_from is not None:
        trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
        logger.info(f"Resuming training from {args.resume_from}, starting at step {trainer.step}")
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}")

    if args.enable_viewer:
        trainer.init_viewer(port=args.viewer_port)
        
    trainer.initialize_optimizer()
    return trainer

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

def visualize_step(cfg, dataset, trainer, render_keys, step, args):
    logger.info("Visualizing...")
    vis_timestep = np.linspace(
        0,
        dataset.num_img_timesteps,
        trainer.num_iters // cfg.logging.vis_freq + 1,
        endpoint=False,
        dtype=int,
    )[step // cfg.logging.vis_freq]
    with torch.no_grad():
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
            vis_indices=[
                vis_timestep * dataset.pixel_source.num_cams + i
                for i in range(dataset.pixel_source.num_cams)
            ],
        )
    if args.enable_wandb:
        wandb.log(
            {
                "image_metrics/psnr": render_results["psnr"],
                "image_metrics/ssim": render_results["ssim"],
                "image_metrics/occupied_psnr": render_results["occupied_psnr"],
                "image_metrics/occupied_ssim": render_results["occupied_ssim"],
            }
        )

    vis_frame_dict = save_videos(
        render_results,
        save_pth=os.path.join(cfg.log_dir, "images", f"step_{step}.png"),
        layout=dataset.layout,
        num_timestamps=1,
        keys=render_keys,
        save_seperate_video=cfg.logging.save_seperate_video,
        num_cams=dataset.pixel_source.num_cams,
        fps=cfg.render.fps,
        verbose=False,
    )
    if args.enable_wandb:
        for key, value in vis_frame_dict.items():
            wandb.log({"image_rendering/" + key: wandb.Image(value)})
    del render_results
    torch.cuda.empty_cache()


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


def run_training_loop(cfg, dataset, trainer, render_keys, args):
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)

    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        # visualization
        if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0:
            visualize_step(cfg, dataset, trainer, render_keys, step, args)

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
        do_save = step>0 and ((step%cfg.logging.saveckpt_freq==0) or step==trainer.num_iters) and args.resume_from is None
        if do_save:
            trainer.save_checkpoint(log_dir=cfg.log_dir, save_only_model=True, is_final=step==trainer.num_iters)

        # cache image errors
        if step>0 and trainer.optim_general.cache_buffer_freq>0 and step%trainer.optim_general.cache_buffer_freq==0:
            cache_image_errors(cfg, dataset, trainer, step)

    return step

def run_evaluation(cfg, dataset, trainer, render_keys, step, args, max_render_frames=None):
    if max_render_frames is not None:
        for cam_id in dataset.pixel_source.camera_list:
            cam = dataset.pixel_source.camera_data[cam_id]
            cam.cam_to_worlds = cam.cam_to_worlds[:max_render_frames]

        max_render_frames = apply_render_frame_limit(dataset, max_render_frames)

    do_evaluation(
        step=step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        max_render_frames=max_render_frames,
    )
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


def render_single_offset_novel_view(dataset, trainer, lateral_offset_m: float):
    # This code should shift the camera left by lateral_offset_m meters.
    pixel_source = dataset.pixel_source
    cam0 = pixel_source.camera_data[0]

    reference_frame_idx = random.randint(0, len(cam0) - 1)
    ref_image_infos, ref_cam_infos = cam0.get_image(reference_frame_idx)

    ref_c2w = cam0.cam_to_worlds[reference_frame_idx].clone()
    lateral_axis = ref_c2w[:3, 0]
    lateral_axis = lateral_axis / (torch.linalg.norm(lateral_axis) + 1e-8)
    novel_c2w = ref_c2w.clone()
    novel_c2w[:3, 3] = ref_c2w[:3, 3] + lateral_offset_m * lateral_axis

    intrinsics = cam0.intrinsics[reference_frame_idx].clone()
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
    novel_cam_infos = {
        "cam_id": ref_cam_infos["cam_id"],
        "cam_name": ref_cam_infos["cam_name"],
        "camera_to_world": novel_c2w,
        "height": ref_cam_infos["height"],
        "width": ref_cam_infos["width"],
        "intrinsics": intrinsics,
    }

    with torch.no_grad():
        trainer.set_eval()
        for k, v in novel_image_infos.items():
            if isinstance(v, torch.Tensor):
                novel_image_infos[k] = v.cuda(non_blocking=True)
        for k, v in novel_cam_infos.items():
            if isinstance(v, torch.Tensor):
                novel_cam_infos[k] = v.cuda(non_blocking=True)
        novel_outputs = trainer(novel_image_infos, novel_cam_infos, novel_view=True)
        rendered_rgb = novel_outputs["rgb"].detach().cpu()

    logger.info(
        f"Rendered one novel view from cam0 reference frame {reference_frame_idx} with lateral offset {lateral_offset_m:.3f}m"
    )
    return {
        "reference_frame_idx": reference_frame_idx,
        "rendered_rgb": rendered_rgb,
        "reference_rgb": ref_image_infos["pixels"].detach().cpu(),
        "novel_c2w": novel_c2w.detach().cpu(),
        "intrinsics": intrinsics.detach().cpu(),
    }


def append_novel_sample_to_training_dataset(dataset, novel_sample):
    pixel_source = dataset.pixel_source
    cam0 = pixel_source.camera_data[0]
    ref_idx = novel_sample["reference_frame_idx"]

    new_frame_idx = len(cam0)
    cam0.cam_to_worlds = torch.cat(
        [cam0.cam_to_worlds, novel_sample["novel_c2w"].to(cam0.cam_to_worlds.device).unsqueeze(0)], dim=0
    )
    cam0.intrinsics = torch.cat(
        [cam0.intrinsics, novel_sample["intrinsics"].to(cam0.intrinsics.device).unsqueeze(0)], dim=0
    )
    if cam0.distortions is not None:
        cam0.distortions = torch.cat([cam0.distortions, cam0.distortions[ref_idx:ref_idx+1].clone()], dim=0)

    cam0.images = torch.cat(
        [cam0.images, novel_sample["rendered_rgb"].to(cam0.images.device).unsqueeze(0)], dim=0
    )
    if cam0.normalized_time is not None:
        cam0.normalized_time = torch.cat(
            [cam0.normalized_time, cam0.normalized_time[ref_idx:ref_idx+1].clone()], dim=0
        )
    cam0.unique_img_idx = torch.cat(
        [cam0.unique_img_idx, cam0.unique_img_idx[ref_idx:ref_idx+1].clone()], dim=0
    )

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
            setattr(cam0, tensor_field, torch.cat([field_data, field_data[ref_idx:ref_idx+1].clone()], dim=0))

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
    return int(new_img_idx)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main(args):
    cfg = setup(args)
    # Initialize configuration (OmegaConf object), dataset (DrivingDataset), and trainer (Trainer instance)
    dataset = build_dataset(cfg.data)
    original_num_img_timesteps = dataset.num_img_timesteps
    trainer = build_trainer(dataset, cfg, args)
    # Prepare the list of keys (list of strings) that will be rendered/visualized during training
    render_keys = setup_render_keys(cfg, dataset)
    # Run the main training loop: forward/backward passes, logging, saving, and optional visualization
    # Returns the last training step (int)
    step = run_training_loop(cfg, dataset, trainer, render_keys, args)

    lateral_offset_m = 5
    lateral_offset_max = 7 # We will iterate towards this value
    num_iterations_refine = 750
    synthetic_images = 100
    lateral_offset_incremenet = (lateral_offset_max - lateral_offset_m) / max(synthetic_images - 1, 1)

    for _ in range(synthetic_images):
        novel_sample = render_single_offset_novel_view(
            dataset=dataset,
            trainer=trainer,
            lateral_offset_m=-lateral_offset_m,
        )
        novel_rgb = novel_sample["rendered_rgb"]
        reference_rgb = novel_sample["reference_rgb"]
        print(f"Novel RGB shape: {novel_rgb.shape}, Reference RGB shape: {reference_rgb.shape}")
        repaired_image = difix_repair(tensor_to_image(novel_rgb), tensor_to_image(reference_rgb))
        if step % 500 == 0:
            # Save repaied image for inspection
            repaired_image.save(os.path.join(cfg.log_dir, "synthetic_image_samples", f"repaired_{step}.png"))

        repaired_tensor = image_to_tensor(repaired_image).permute(1,2,0) # H, W, C
        novel_sample["rendered_rgb"] = repaired_tensor.to(novel_rgb.device)
        print(f"Repaired image shape: {novel_sample['rendered_rgb'].shape}")

        append_novel_sample_to_training_dataset(dataset, novel_sample)

        trainer.num_train_images = len(dataset.train_image_set)
        trainer.step = step + 1
        trainer.num_iters = step + num_iterations_refine
        logger.info(
            f"Starting refinement training with updated dataset for {num_iterations_refine} iterations "
            f"(from step {trainer.step} to {trainer.num_iters})"
        )
        step = run_training_loop(cfg, dataset, trainer, render_keys, args)
        lateral_offset_m += lateral_offset_incremenet

    # Perform final evaluation (no return value) and optionally start the viewer for inspection
    run_evaluation(
        cfg,
        dataset,
        trainer,
        render_keys,
        step,
        args,
        max_render_frames=original_num_img_timesteps,
    )
    # Return the last training step (int)
    return step
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument(
        "--output_root",
        default="./work_dirs/",
        help="path to save checkpoints and logs",
        type=str,
    )

    # eval
    parser.add_argument(
        "--resume_from",
        default=None,
        help="path to checkpoint to resume from",
        type=str,
    )
    parser.add_argument(
        "--render_video_postfix",
        type=str,
        default=None,
        help="an optional postfix for video",
    )

    # wandb logging part
    parser.add_argument(
        "--enable_wandb", action="store_true", help="enable wandb logging"
    )
    parser.add_argument("--entity", default="ziyc", type=str, help="wandb entity name")
    parser.add_argument(
        "--project",
        default="drivestudio",
        type=str,
        help="wandb project name, also used to enhance log_dir",
    )
    parser.add_argument(
        "--run_name",
        default="omnire",
        type=str,
        help="wandb run name, also used to enhance log_dir",
    )
    parser.add_argument(
        "--num_iters", type=int, help="number of training iterations (overrides value specified in config file) [OPTIONAL]", default=None
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
    final_step = main(args)
