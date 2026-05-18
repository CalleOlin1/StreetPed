import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
import lpips
from PIL import Image, ImageTk
import matplotlib.cm as cm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

def list_images(folder):
    return sorted(
        [p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED_EXT]
    )


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def to_tensor(img, device):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_lpips_map(model, gt, pred, device):
    gt_t = to_tensor(gt, device)
    pred_t = to_tensor(pred, device)

    with torch.no_grad():
        lpips_map = model(pred_t, gt_t, normalize=True)

    # Upsample to image size
    lpips_map = torch.nn.functional.interpolate(
        lpips_map,
        size=gt.shape[:2],
        mode="bilinear",
        align_corners=False,
    )

    return lpips_map.squeeze().cpu().numpy()

def heatmap_to_rgb(map_):
    norm = map_ / (map_.max() + 1e-8)
    colored = cm.inferno(norm)[..., :3]  # RGB
    return (colored * 255).astype(np.uint8)

class LPIPSViewer:
    def __init__(self, root, gt_folder, pred_folder, mask_folder=None):
        self.root = root
        self.root.title("LPIPS Viewer")

        self.gt_paths = list_images(gt_folder)
        self.pred_paths = list_images(pred_folder)
        if mask_folder:
            self.mask_paths = list_images(mask_folder)
        else:
            self.mask_paths = None

        self.length = min(len(self.gt_paths), len(self.pred_paths))
        self.index = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net="alex", spatial=True).to(self.device)
        self.model.eval()

        # UI
        self.frame = ttk.Frame(root)
        self.frame.pack()

        self.entry = ttk.Entry(self.frame, width=10)
        self.entry.pack(side=tk.LEFT)
        self.entry.bind("<Return>", self.on_enter)

        self.label = ttk.Label(self.frame, text=f"/ {self.length}")
        self.label.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(root, width=900, height=300)
        self.canvas.pack()

        self.update_display()

    def on_enter(self, event):
        try:
            idx = int(self.entry.get())
            if 0 <= idx < self.length:
                self.index = idx
                self.update_display()
        except ValueError:
            pass

    def prepare_image(self, img):
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img).resize((300, 300))
  
    def mask_image(self, target_img, mask):
        # Ensure mask is single channel
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Normalize mask to boolean
        mask_bool = mask > 0

        # Expand mask to match RGB channels
        mask_bool = np.expand_dims(mask_bool, axis=-1)

        # Apply mask
        return target_img * mask_bool

    def update_display(self):
        gt = load_image(self.gt_paths[self.index])
        pred = load_image(self.pred_paths[self.index])
        if self.mask_paths:
            # If masks defined, we 0 areas in masks
            mask = load_image(self.mask_paths[self.index])
            gt = self.mask_image(gt, mask)
            pred = self.mask_image(pred, mask)

        lpips_map = compute_lpips_map(self.model, gt, pred, self.device)
        lpips_score = lpips_map.mean()
        heatmap = heatmap_to_rgb(lpips_map)

        print(gt.shape)
        psnr_score = peak_signal_noise_ratio(gt, pred)
        ssim_score =   structural_similarity(gt, pred, data_range=1.0, channel_axis=-1)

        gt_img = self.prepare_image(gt)
        pred_img = self.prepare_image(pred)
        heat_img = Image.fromarray(heatmap).resize((300, 300))

        self.tk_gt = ImageTk.PhotoImage(gt_img)
        self.tk_pred = ImageTk.PhotoImage(pred_img)
        self.tk_heat = ImageTk.PhotoImage(heat_img)

        self.canvas.delete("all")

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_gt)
        self.canvas.create_image(300, 0, anchor=tk.NW, image=self.tk_pred)
        self.canvas.create_image(600, 0, anchor=tk.NW, image=self.tk_heat)

        self.root.title(f"Frame {self.index} | LPIPS {lpips_score} | PSNR {psnr_score} | SSIM {ssim_score}")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-folder", required=True)
    parser.add_argument("--pred-folder", required=True)
    parser.add_argument("--mask-folder", required=False)
    args = parser.parse_args()

    root = tk.Tk()
    app = LPIPSViewer(root, args.gt_folder, args.pred_folder, args.mask_folder)
    root.mainloop()

if __name__ == "__main__":
    main()