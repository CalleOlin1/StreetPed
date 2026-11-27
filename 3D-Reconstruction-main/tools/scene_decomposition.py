"""Interactive checkpoint viewer with per-class scene visibility controls.

Run: python tools/scene_decomposition.py --checkpoint PATH
Click a class in the right panel to toggle it. Left/right change frame;
W/A/S/D and Space/Z move the camera; Q or Esc quits.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str


WINDOW = "Scene decomposition"
PANEL_WIDTH = 230
ROW_HEIGHT = 38


def load_scene(checkpoint):
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    config_path = os.path.join(os.path.dirname(checkpoint), "config.yaml")
    cfg = OmegaConf.load(config_path)
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    trainer.resume_from_checkpoint(ckpt_path=checkpoint, load_only_model=True)
    trainer.eval()
    return dataset, trainer


def render_scene(trainer, image_infos, camera_infos, visible):
    """Render selected classes with the normal occlusion and sky compositing."""
    with torch.no_grad():
        image_infos = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
                       for k, v in image_infos.items()}
        camera_infos = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
                        for k, v in camera_infos.items()}
        image_id = image_infos["img_idx"].flatten()[0]
        if hasattr(trainer, "normalized_timestamps"):
            time = image_infos["normed_time"].flatten()[0]
            trainer.cur_frame = torch.argmin(torch.abs(trainer.normalized_timestamps - time))
        for model in trainer.models.values():
            if hasattr(model, "in_test_set"):
                model.in_test_set = trainer.in_test_set
        for name in trainer.gaussian_classes:
            model = trainer.models[name]
            if hasattr(model, "set_cur_frame"):
                model.set_cur_frame(trainer.cur_frame)

        camera = trainer.process_camera(camera_infos, image_id)
        if any(visible[name] for name in trainer.gaussian_classes):
            gaussians = trainer.collect_gaussians(camera, image_id)
            mask = torch.zeros_like(trainer.pts_labels, dtype=torch.bool)
            for name, label in trainer.gaussian_classes.items():
                if visible[name]:
                    mask |= trainer.pts_labels == label
            gaussians._opacities = gaussians.opacities * mask.unsqueeze(-1)
            outputs, _ = trainer.render_gaussians(
                gs=gaussians, cam=camera,
                near_plane=trainer.render_cfg.near_plane,
                far_plane=trainer.render_cfg.far_plane,
                render_mode="RGB+ED",
                radius_clip=trainer.render_cfg.get("radius_clip", 0.0),
            )
            rgb, opacity = outputs["rgb_gaussians"], outputs["opacity"]
        else:
            rgb = torch.zeros((int(camera.H), int(camera.W), 3), device=trainer.device)
            opacity = torch.zeros_like(rgb[..., :1])

        if visible.get("Sky", False):
            rgb = rgb + trainer.models["Sky"](image_infos) * (1.0 - opacity)
        if visible.get("Road", False) and getattr(trainer, "road_mesh", None) is not None:
            mesh = trainer._render_road_mesh_from_camera(camera)
            if mesh:
                alpha = (mesh["alpha"] * float(np.clip(
                    getattr(trainer, "mesh_blend_alpha", 1.0), 0.0, 1.0
                ))).clamp(0.0, 1.0)
                rgb = mesh["rgb"] * alpha + rgb * (1.0 - alpha)
        if any(visible.values()):
            rgb = trainer.affine_transformation(rgb, image_infos)
        return (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


class SceneViewer:
    def __init__(self, dataset, trainer):
        self.dataset = dataset.full_image_set
        if self.dataset is None or len(self.dataset) == 0:
            raise ValueError("The dataset has no full images to display")
        self.trainer = trainer
        self.visible = {name: True for name in trainer.gaussian_classes}
        if "Road" not in self.visible and getattr(trainer, "road_mesh", None) is not None:
            self.visible["Road"] = True
        if "Sky" in trainer.models:
            self.visible["Sky"] = True
        self.frame = 0
        self.offset = np.zeros(3, dtype=np.float32)
        self.image_width = 0
        self.dirty = True

    def on_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or x < self.image_width:
            return
        row = (y - 66) // ROW_HEIGHT
        names = list(self.visible)
        if 0 <= row < len(names) and 66 + row * ROW_HEIGHT <= y < 98 + row * ROW_HEIGHT:
            name = names[row]
            self.visible[name] = not self.visible[name]
            self.dirty = True

    def render(self):
        images, camera = self.dataset.get_image(self.frame, camera_downscale=1.0)
        camera["camera_to_world"] = camera["camera_to_world"].clone()
        pose = camera["camera_to_world"]
        for axis in range(3):
            pose[:3, 3] += float(self.offset[axis]) * pose[:3, axis]
        rgb = render_scene(self.trainer, images, camera, self.visible)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        self.image_width = width
        canvas = np.full((height, width + PANEL_WIDTH, 3), 35, dtype=np.uint8)
        canvas[:, :width] = bgr
        cv2.putText(canvas, "Scene classes", (width + 16, 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (240, 240, 240), 2)
        cv2.putText(canvas, f"Frame {self.frame + 1}/{len(self.dataset)}", (width + 16, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
        for row, (name, enabled) in enumerate(self.visible.items()):
            top = 66 + row * ROW_HEIGHT
            if top + 32 > height:
                break
            color = (65, 115, 65) if enabled else (65, 65, 65)
            cv2.rectangle(canvas, (width + 10, top), (width + PANEL_WIDTH - 10, top + 32), color, -1)
            cv2.putText(canvas, "x" if enabled else " ", (width + 20, top + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, name, (width + 49, top + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (245, 245, 245), 1)
        for i, line in enumerate(("Arrows: frames", "WASD, Space/Z: move", "Q / Esc: quit")):
            cv2.putText(canvas, line, (width + 16, height - 56 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
        self.dirty = False
        return canvas

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                if self.dirty:
                    cv2.imshow(WINDOW, self.render())
                key = cv2.waitKeyEx(30)
                if key in (27, ord("q")):
                    break
                if key in (2424832, 65361, 81):
                    self.frame = (self.frame - 1) % len(self.dataset)
                elif key in (2555904, 65363, 83):
                    self.frame = (self.frame + 1) % len(self.dataset)
                elif key in (ord("a"), ord("d"), ord("w"), ord("s"), 32, ord("z")):
                    axis, direction = {
                        ord("a"): (0, -1), ord("d"): (0, 1),
                        ord("w"): (2, 1), ord("s"): (2, -1),
                        32: (1, -1), ord("z"): (1, 1),
                    }[key]
                    self.offset[axis] += direction * 0.25
                else:
                    continue
                self.dirty = True
        finally:
            cv2.destroyWindow(WINDOW)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint")
    args = parser.parse_args()
    dataset, trainer = load_scene(args.checkpoint)
    SceneViewer(dataset, trainer).run()


if __name__ == "__main__":
    main()
