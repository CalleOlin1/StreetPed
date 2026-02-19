from typing import List, Optional
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
from models.video_utils import render_images, save_videos, render_novel_views, extract_camera_poses_from_dataset, save_camera_poses, analyze_camera_trajectory  

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


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
):
    print("Save images is", args.save_images)
    trainer.set_eval()
    # New: camera pose extraction feature
    if extract_camera_poses:
        logger.info("Extracting camera poses...")
        
        # Extract poses for each dataset
        pose_save_dir = f"{cfg.log_dir}/camera_poses{post_fix}"
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
                f"{cfg.log_dir}/metrics{post_fix}/images_test_{current_time}.json"
            )
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/test_set_{step}.mp4"
        else:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/test_set_{step}_{args.render_video_postfix}.mp4"
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
                f"{cfg.log_dir}/metrics{post_fix}/images_full_{current_time}.json"
            )
            with open(full_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {full_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
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
    if render_novel_cfg is not None:
        logger.info("Rendering novel views...")
        render_traj = dataset.get_novel_render_traj(
            traj_types=render_novel_cfg.traj_types,
            target_frames=render_novel_cfg.get("frames", dataset.frame_num),
        )
        video_output_dir = f"{cfg.log_dir}/videos{post_fix}/novel_{step}"
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        for traj_type, traj in render_traj.items():
            # Prepare rendering data
            render_data = dataset.prepare_novel_view_render_data(traj)

            # Render and save video
            save_path = os.path.join(video_output_dir, f"{traj_type}.mp4")
            print("eval.py > Started rendering novel view")
            render_novel_views(
                trainer,
                render_data,
                save_path,
                fps=render_novel_cfg.get("fps", cfg.render.fps),
                traj_type=traj_type,
                save_images=args.save_images
            )
            logger.info(
                f"Saved novel view video for trajectory type: {traj_type} to {save_path}"
            )


def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    
    args.enable_wandb = False
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
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

    # Resume from checkpoint
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )

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
        type=bool,
        action="store_true",
        help="Skips running some metrics as well as rendering the original viewpoint.",
    )
    parser.add_argument(
        "--max_render_frames",
        type=int,
        default=None,
        help="limit rendering to the first N frames without changing dataset timesteps",
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
