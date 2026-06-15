

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


    # Step 1: Prepare to load the .pth checkpoint using trainer's resume_from_checkpoint
    checkpoint_path = args.checkpoint
    print(f"Loaded checkpoint from {checkpoint_path}")

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
    # Use resume_from_checkpoint for proper loading
    trainer.resume_from_checkpoint(ckpt_path=checkpoint_path, load_only_model=True)
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
    # Camera translation offsets and momentum
    cam_offset_x = 0.0
    cam_offset_y = 0.0
    cam_offset_z = 0.0
    cam_offset_forward = 0.0

    # Camera velocity for each axis
    vel_x = 0.0
    vel_y = 0.0
    vel_z = 0.0
    vel_forward = 0.0

    # Momentum parameters
    max_velocity = 1  # meters per frame
    acceleration = 0.05  # meters per frame per key press
    deceleration = 0.02  # meters per frame per tick

    def render_frame(idx, cam_offset_x, cam_offset_y, cam_offset_z, cam_offset_forward):
        split_idx = dataset.full_image_set.split_indices[idx]
        single_image_dataset = SplitWrapper(
            datasource=dataset.full_image_set.datasource,
            split_indices=[split_idx],
        )
        # Get image and camera infos
        image_infos, cam_infos = single_image_dataset.get_image(0, camera_downscale=1.0)
        device = trainer.device if hasattr(trainer, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move all tensors to the correct device
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.to(device)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.to(device)
        # Modify camera pose: apply translation in local directions
        c2w = cam_infos["camera_to_world"].clone()
        translation = (
            cam_offset_x * c2w[:3, 0] +  # X (right/left)
            cam_offset_y * c2w[:3, 1] +  # Y (up/down)
            cam_offset_z * c2w[:3, 2] +  # Z (forward/backward)
            cam_offset_forward * c2w[:3, 2]  # Forward/backward (same as Z)
        )
        c2w[:3, 3] += translation
        cam_infos["camera_to_world"] = c2w
        # Render with modified camera pose
        render_results = trainer(image_infos, cam_infos)
        rgb_image = render_results["rgb"].detach().cpu().numpy()
        rgb_image = np.clip(rgb_image, 0, 1)
        return (rgb_image * 255).astype(np.uint8)


    rgb_uint8 = render_frame(frame_idx, cam_offset_x, cam_offset_y, cam_offset_z, cam_offset_forward)

    # Key state tracking
    key_state = {
        'a': False, 'd': False, 'w': False, 's': False, ' ': False, 'z': False,
        'left': False, 'right': False
    }

    # Map key codes to actions
    key_map = {
        ord('a'): 'a',
        ord('d'): 'd',
        ord('w'): 'w',
        ord('s'): 's',
        32: ' ',  # Space
        ord('z'): 'z',
        81: 'left',
        83: 'right',
    }

    while True:
        cv2.imshow(window_name, rgb_uint8[..., ::-1])
        key = cv2.waitKey(10)  # 10 ms delay for smooth updates
        # Handle frame navigation
        if key == 27 or key == ord('q'):
            break
        elif key == 81:  # Left arrow
            frame_idx = (frame_idx - 1) % num_frames
            rgb_uint8 = render_frame(frame_idx, cam_offset_x, cam_offset_y, cam_offset_z, cam_offset_forward)
            continue
        elif key == 83:  # Right arrow
            frame_idx = (frame_idx + 1) % num_frames
            rgb_uint8 = render_frame(frame_idx, cam_offset_x, cam_offset_y, cam_offset_z, cam_offset_forward)
            continue

        # Update key state for movement keys
        for code, name in key_map.items():
            if key == code:
                key_state[name] = True
        # Release keys if no key pressed
        if key == -1:
            for k in key_state:
                key_state[k] = False

        # Update velocities based on key state
        # X axis (left/right)
        if key_state['a']:
            vel_x = max(vel_x - acceleration, -max_velocity)
        elif key_state['d']:
            vel_x = min(vel_x + acceleration, max_velocity)
        else:
            # Decelerate X
            if vel_x > 0:
                vel_x = max(vel_x - deceleration, 0)
            elif vel_x < 0:
                vel_x = min(vel_x + deceleration, 0)

        # Y axis (up/down)
        if key_state[' ']:
            vel_y = max(vel_y - acceleration, -max_velocity)
        elif key_state['z']:
            vel_y = min(vel_y + acceleration, max_velocity)
        else:
            if vel_y > 0:
                vel_y = max(vel_y - deceleration, 0)
            elif vel_y < 0:
                vel_y = min(vel_y + deceleration, 0)

        # Forward/backward (Z/forward)
        if key_state['w']:
            vel_forward = min(vel_forward + acceleration, max_velocity)
        elif key_state['s']:
            vel_forward = max(vel_forward - acceleration, -max_velocity)
        else:
            if vel_forward > 0:
                vel_forward = max(vel_forward - deceleration, 0)
            elif vel_forward < 0:
                vel_forward = min(vel_forward + deceleration, 0)

        # Optionally, add Z axis (not mapped to keys in original code)
        # cam_offset_z, vel_z can be used for future extension

        # Update camera offsets
        cam_offset_x += vel_x
        cam_offset_y += vel_y
        cam_offset_forward += vel_forward

        # Only re-render if camera moved
        if any([vel_x, vel_y, vel_forward]):
            rgb_uint8 = render_frame(frame_idx, cam_offset_x, cam_offset_y, cam_offset_z, cam_offset_forward)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()