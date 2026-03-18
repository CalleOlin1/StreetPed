

import argparse
import torch
import cv2
import numpy as np
import sys
import os
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import render_images

def main():

    parser = argparse.ArgumentParser(description="Live 3D Scene Viewer")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint file')
    args = parser.parse_args()

    # Step 1: Load the .pth checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        print(f"Loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # Step 2: Always infer config path from checkpoint directory
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    print(f"Using config path: {config_path}")

    # Step 3: Load config and dataset
    try:
        cfg = OmegaConf.load(config_path)
        print(f"Loaded config from {config_path}")
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    # Build dataset (use nested config structure)
    dataset = DrivingDataset(data_cfg=cfg.data)

    # Step 3: Initialize trainer/model
    trainer_class = import_str(cfg.trainer.type)
    trainer = trainer_class(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    trainer.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    trainer.eval()

    # Step 4: Render a single frame (first full image)
    from datasets.base.split_wrapper import SplitWrapper
    if dataset.full_image_set is not None and len(dataset.full_image_set) > 0:
        single_image_dataset = SplitWrapper(
            datasource=dataset.full_image_set.datasource,
            split_indices=[dataset.full_image_set.split_indices[0]],
        )
        render_results = render_images(
            trainer=trainer,
            dataset=single_image_dataset,
            compute_metrics=False,
            compute_error_map=False,
        )
        rgb_image = render_results["rgbs"][0]
        rgb_uint8 = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
    else:
        rgb_uint8 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(rgb_uint8, 'No full image found', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Step 5: Show the rendered image in a window with arrow key controls
    window_name = "Live 3D Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    num_frames = len(dataset.full_image_set)
    frame_idx = 0

    def render_frame(idx):
        single_image_dataset = SplitWrapper(
            datasource=dataset.full_image_set.datasource,
            split_indices=[dataset.full_image_set.split_indices[idx]],
        )
        render_results = render_images(
            trainer=trainer,
            dataset=single_image_dataset,
            compute_metrics=False,
            compute_error_map=False,
        )
        rgb_image = render_results["rgbs"][0]
        return (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)

    rgb_uint8 = render_frame(frame_idx)

    while True:
        cv2.imshow(window_name, rgb_uint8[..., ::-1])  # Convert RGB to BGR for OpenCV
        key = cv2.waitKey(0)
        if key == 27 or key == ord('q'):
            break
        elif key == 81:  # Left arrow
            frame_idx = (frame_idx - 1) % num_frames
            rgb_uint8 = render_frame(frame_idx)
        elif key == 83:  # Right arrow
            frame_idx = (frame_idx + 1) % num_frames
            rgb_uint8 = render_frame(frame_idx)
        # Optionally, add up/down for other controls
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()