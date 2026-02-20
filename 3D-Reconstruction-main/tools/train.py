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
from tools.eval import do_evaluation
from utils.misc import import_str
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from models.video_utils import render_images, save_videos
from datasets.driving_dataset import DrivingDataset

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


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


def build_dataset_and_trainer(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

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
    return cfg, dataset, trainer

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

def run_evaluation(cfg, dataset, trainer, render_keys, step, args):
    do_evaluation(step=step, cfg=cfg, trainer=trainer, dataset=dataset, render_keys=render_keys, args=args)
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

def main(args):
    cfg, dataset, trainer = build_dataset_and_trainer(args)
    render_keys = setup_render_keys(cfg, dataset)
    step = run_training_loop(cfg, dataset, trainer, render_keys, args)
    run_evaluation(cfg, dataset, trainer, render_keys, step, args)
    return step

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
