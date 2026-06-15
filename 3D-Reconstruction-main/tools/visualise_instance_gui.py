import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from visualise_instance import (
    build_trainer_from_checkpoint,
    extract_instance_points,
    render_8_views_with_renderer,
)

def np_to_tk(img_np):
    img = (img_np * 255).astype(np.uint8)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    pil_img = Image.fromarray(img)
    return ImageTk.PhotoImage(pil_img)

class VisualiseInstanceGUI(tk.Tk):
    def __init__(self, resume_from, frame_idx=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Object Visualiser")
        self.geometry("1200x700")
        self.resume_from = resume_from
        self.frame_idx = frame_idx
        self.object_keys = []
        self.instances = None
        self.trainer = None
        self.dataset = None
        self.log_dir = None
        self.image_label = None
        self._init_model()
        self._build_gui()

    def _init_model(self):
        _, dataset, trainer, log_dir = build_trainer_from_checkpoint(self.resume_from, [])
        self.dataset = dataset
        self.trainer = trainer
        self.log_dir = log_dir
        self.instances = extract_instance_points(trainer=trainer, frame_idx=self.frame_idx)
        self.object_keys = [k for k in self.instances.keys() if k[0] == "rigid"]

    def _build_gui(self):
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Select Object:").pack(side=tk.LEFT)
        self.obj_var = tk.StringVar()
        obj_ids = [str(k[1]) for k in self.object_keys]
        self.dropdown = ttk.Combobox(control_frame, textvariable=self.obj_var, values=obj_ids, state="readonly")
        self.dropdown.pack(side=tk.LEFT, padx=5)
        if obj_ids:
            self.dropdown.current(0)

        render_btn = tk.Button(control_frame, text="Render", command=self.on_render)
        render_btn.pack(side=tk.LEFT, padx=10)

        self.image_panel = tk.Label(self)
        self.image_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def on_render(self):
        sel = self.obj_var.get()
        if not sel:
            messagebox.showerror("Error", "No object selected.")
            return
        obj_id = int(sel)
        key = ("rigid", obj_id)
        points = self.instances[key]
        fig_title = f"Object {obj_id} (rigid) - 8 views"
        # Run in thread to avoid GUI freeze
        threading.Thread(target=self._render_and_show, args=(key, points, fig_title)).start()

    def _render_and_show(self, key, points, fig_title):
        fig = render_8_views_with_renderer(
            trainer=self.trainer,
            dataset=self.dataset,
            frame_idx=self.frame_idx,
            object_type=key[0],
            object_id=key[1],
            points_xyz=points,
            title=fig_title,
        )
        fig.canvas.draw()
        img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_np = img_np / 255.0
        img = np_to_tk(img_np)
        self.image_panel.config(image=img)
        self.image_panel.image = img
        plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to use")
    args = parser.parse_args()
    app = VisualiseInstanceGUI(resume_from=args.resume_from, frame_idx=args.frame_idx)
    app.mainloop()
