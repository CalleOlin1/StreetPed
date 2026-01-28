import os
import argparse
import logging
from omegaconf import OmegaConf
import torch

from utils.misc import export_points_to_ply, import_str
from datasets.driving_dataset import DrivingDataset

logger = logging.getLogger()


def export_gaussians_to_ply(
    model,
    save_dir: str,
    alpha_thresh: float = 0.01,
    instance_id: int = None,
    frame_idx: int = 0,
    normalize: bool = False,
    export_all_frames: bool = False,
):
    """
    Export Gaussians to PLY file

    Args:
        model: Gaussian model
        save_dir: Save directory
        alpha_thresh: Opacity threshold for filtering points
        instance_id: Instance ID, exports only this instance if specified
        frame_idx: Frame index, used for SMPL model
        normalize: Whether to normalize point cloud to [0,1] range
        export_all_frames: Whether to export point clouds for all frames
    """
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(model, "smpl_qauts"):  # SMPLNodes
        if export_all_frames:
            # Export all frames
            for frame in range(model.num_frames):
                if instance_id is not None:
                    if not model.instances_fv[frame, instance_id]:
                        continue
                    point_data = model.export_gaussians_to_ply(
                        alpha_thresh=alpha_thresh,
                        instance_id=instance_id,
                        specific_frame=frame,
                    )
                    save_path = os.path.join(
                        save_dir, f"frame_{frame}_instance_{instance_id}.ply"
                    )
                    export_points_to_ply(
                        positions=point_data["positions"],
                        colors=point_data["colors"],
                        save_path=save_path,
                        normalize=normalize,
                    )
                    logger.info(
                        f"Exported frame {frame} instance {instance_id} to {save_path}"
                    )
        else:
            # Export specified frame only
            point_data = model.export_gaussians_to_ply(
                alpha_thresh=alpha_thresh,
                instance_id=instance_id,
                specific_frame=frame_idx,
            )
            save_path = os.path.join(
                save_dir, f"frame_{frame_idx}_instance_{instance_id}.ply"
            )
            export_points_to_ply(
                positions=point_data["positions"],
                colors=point_data["colors"],
                save_path=save_path,
                normalize=normalize,
            )
            logger.info(
                f"Exported frame {frame_idx} instance {instance_id} to {save_path}"
            )

    elif hasattr(model, "point_ids"):  # RigidNodes
        point_data = model.export_gaussians_to_ply(
            alpha_thresh=alpha_thresh, instance_id=instance_id
        )
        if instance_id is not None:
            save_path = os.path.join(save_dir, f"instance_{instance_id}.ply")
        else:
            save_path = os.path.join(save_dir, "all_instances.ply")
        export_points_to_ply(
            positions=point_data["positions"],
            colors=point_data["colors"],
            save_path=save_path,
            normalize=normalize,
        )
        logger.info(f"Exported to {save_path}")

    else:  # VanillaGaussians
        point_data = model.export_gaussians_to_ply(alpha_thresh=alpha_thresh)
        save_path = os.path.join(save_dir, "gaussians.ply")
        export_points_to_ply(
            positions=point_data["positions"],
            colors=point_data["colors"],
            save_path=save_path,
            normalize=normalize,
        )
        logger.info(f"Exported to {save_path}")


def main(args):
    # Load configuration
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # Setup trainer
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
    logger.info(f"Resumed from checkpoint: {args.resume_from}")

    # Get the model to export
    if args.model_name is not None:
        model = trainer.models[args.model_name]
    else:
        model = trainer.models[list(trainer.models.keys())[0]]

    # Export PLY file
    export_gaussians_to_ply(
        model=model,
        save_dir=args.save_dir,
        alpha_thresh=args.alpha_thresh,
        instance_id=args.instance_id,
        frame_idx=args.frame_idx,
        normalize=args.normalize,
        export_all_frames=args.export_all_frames,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Gaussian model to PLY file")

    # Required arguments
    parser.add_argument(
        "--resume_from",
        type=str,
        required=True,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save PLY files"
    )

    # Optional arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of model to export (e.g., 'SMPLNodes', 'RigidNodes')",
    )
    parser.add_argument(
        "--alpha_thresh",
        type=float,
        default=0.01,
        help="Opacity threshold for filtering points",
    )
    parser.add_argument(
        "--instance_id", type=int, default=None, help="ID of instance to export"
    )
    parser.add_argument(
        "--frame_idx", type=int, default=0, help="Frame index for SMPL model export"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize point cloud to [0,1] range"
    )
    parser.add_argument(
        "--export_all_frames",
        action="store_true",
        help="Export all frames for SMPL model",
    )

    # Other configuration options
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    args = parser.parse_args()
    main(args)
