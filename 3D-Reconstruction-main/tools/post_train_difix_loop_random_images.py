"""
The goal of this file is to create a structured and memory efficient approach to improve the quality of a trained checkpoint file.

"""
import os
import argparse
import shutil
import gc
import subprocess
import tempfile
import pickle
import sys
from PIL import Image
import numpy as np

from tools.difix_sender_receiver import difix_repair

# Utility functions
def tensor_to_image(tensor):
    # Ensure HWC format
    if tensor.ndim == 3 and tensor.shape[0] == 3:  # CHW
        tensor = tensor.permute(1, 2, 0)
    img = tensor.numpy()
    # Clamp values to [0, 1] and scale to [0, 255]
    img = np.clip(img, 0, 1) * 255
    img = img.astype(np.uint8)
    return Image.fromarray(img)

def image_to_array(img: Image.Image, normalize: bool = True) -> np.ndarray:
    img_np = np.array(img)  # H x W x C, uint8
    if normalize:
        img_np = img_np.astype(np.float32) / 255.0
    else:
        img_np = img_np.astype(np.float32)
    # Convert HWC → CHW (same as torch.permute(2, 0, 1))
    array = np.transpose(img_np, (2, 0, 1))
    return array

def log_synthetic_image(novel_image, repaired_image, run_path, img_no):
    side_by_side = Image.new('RGB', (novel_image.width + repaired_image.width, novel_image.height))
    side_by_side.paste(novel_image, (0, 0))
    side_by_side.paste(repaired_image, (novel_image.width, 0))
    side_by_side.save(os.path.join(run_path, "synthetic_image_samples", f"sidebyside_{img_no}.png"))

def render_novel_sample(checkpoint_path, frame_index, lateral_offset):
    """
    Spawns a subprocess to run tools/standalone_renderer.py in GPU-isolated mode.
    Returns (curr_step, novel_sample)
    """
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        output_path = tmp.name

    cmd = [
        sys.executable, "-u", os.path.join(os.path.dirname(__file__), "standalone_renderer.py"),
        "render_novel_sample",
        f"--checkpoint_path={checkpoint_path}",
        f"--frame_index={frame_index}",
        f"--lateral_offset={lateral_offset}",
        f"--output_path={output_path}"
    ]
    result = subprocess.run(cmd, capture_output=False, text=False)
    if result.returncode != 0:
        print("Error in render_novel_sample subprocess:", result.stderr)
        raise RuntimeError("render_novel_sample subprocess failed")

    with open(output_path, "rb") as f:
        data = pickle.load(f)
    os.remove(output_path)
    # For compatibility, return (curr_step, novel_sample)
    curr_step = data.get("reference_frame_idx", None)
    novel_sample = data
    return curr_step, novel_sample

def render_novel_sample_list(checkpoint_path, frame_index_list, lateral_offset_list):
    """
    Spawns a subprocess to run tools/standalone_renderer.py in GPU-isolated mode.
    Returns a list of (curr_step, novel_sample) for each input pair.
    """
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        output_path = tmp.name

    # Convert lists to strings for command line
    frame_index_args = [str(idx) for idx in frame_index_list]
    lateral_offset_args = [str(offset) for offset in lateral_offset_list]
    cmd = [
        sys.executable, "-u", os.path.join(os.path.dirname(__file__), "standalone_renderer.py"),
        "render_novel_sample_list",
        f"--checkpoint_path={checkpoint_path}",
        "--frame_index_list", *frame_index_args,
        "--lateral_offset_list", *lateral_offset_args,
        f"--output_path={output_path}"
    ]
    result = subprocess.run(cmd, capture_output=False, text=False)
    if result.returncode != 0:
        print("Error in render_novel_sample subprocess:", result.stderr)
        raise RuntimeError("render_novel_sample subprocess failed")

    with open(output_path, "rb") as f:
        data_list = pickle.load(f)
    os.remove(output_path)
    # For compatibility, return a list of (curr_step, novel_sample)
    result = []
    for data in data_list:
        curr_step = data.get("reference_frame_idx", None)
        novel_sample = data
        result.append((curr_step, novel_sample))
    return result

def train_synthetic(synthetic_samples, checkpoint_path, frame_index, lateral_offset, from_scratch=False):
    # Pickle synthetic_samples to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_in:
        input_path = tmp_in.name
        pickle.dump(synthetic_samples, tmp_in)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_out:
        output_path = tmp_out.name

    cmd = [
        sys.executable, "-u", os.path.join(os.path.dirname(__file__), "standalone_trainer_fixed_step.py"),
        "train_synthetic",
        f"--checkpoint_path={checkpoint_path}",
        f"--input_path={input_path}",
        f"--output_path={output_path}",
        f"--lateral_offset={lateral_offset}",
        f"--ref_frame={frame_index}",
        f"--split_steps={3000}",
        f"--prune_steps={500}",
    ]

    if from_scratch:
        cmd.append("--from_scratch")

    result = subprocess.run(cmd, capture_output=False, text=False)
    if result.returncode != 0:
        print("Error in render_novel_sample subprocess:", result.stderr)
        raise RuntimeError("render_novel_sample subprocess failed")

    with open(output_path, "rb") as f:
        data = pickle.load(f)
    os.remove(output_path)
    os.remove(input_path)
    # For compatibility, return (curr_step, novel_sample)
    curr_step = data.get("reference_frame_idx", None)
    novel_sample = data
    return curr_step, novel_sample

def one_iteration(synthetic_samples, checkpoint_path, lateral_offset, max_lateral_offset, min_frame_index, max_frame_index,
                  batch_size=10):
    # Get random unique frame indices for this batch
    frame_indices = np.random.choice(np.arange(min_frame_index, max_frame_index), size=batch_size, replace=False)
    synthetic_samples = []
    # Use render_novel_sample_list to get all samples in one call
    batch_results = render_novel_sample_list(checkpoint_path, frame_indices.tolist(), [lateral_offset]*batch_size)
    for i, (curr_step, novel_sample) in enumerate(batch_results):
        print(f"Starting diffusion process for image {i+1}/{len(batch_results)}")
        novel_img = tensor_to_image(novel_sample["rendered_rgb"])
        ref_img   = tensor_to_image(novel_sample["reference_rgb"])
        # Send novel view image to repair and get novel image back
        repaired_image = difix_repair(novel_img, ref_img)
        log_synthetic_image(novel_img, repaired_image, run_path, len(synthetic_samples))

        # Convert repaired_image (PIL) back to numpy array (normalized float32, CHW)
        repaired_array = image_to_array(repaired_image, normalize=True)
        novel_sample["rendered_rgb"] = repaired_array

        # Append the full sample dict (with repaired image) to the list
        synthetic_samples.append(novel_sample)

    # Start training with our synthetic images
    # Use the first frame_index from this batch for training (or another policy as needed)
    train_synthetic(synthetic_samples, checkpoint_path, 154, max_lateral_offset)
    return synthetic_samples

def run_training_loop(checkpoint_path, min_frame_index, max_frame_index, min_lateral_offset, max_lateral_offset, num_synthetic_samples):
    lateral_offset = min_lateral_offset
    delta_offset = (max_lateral_offset - min_lateral_offset) / num_synthetic_samples
    synthetic_samples = []
    for synth_i in range(num_synthetic_samples):
        synthetic_samples = one_iteration(synthetic_samples, checkpoint_path, lateral_offset, max_lateral_offset, min_frame_index, max_frame_index)
        lateral_offset += delta_offset
        print(f"Completed iteration {synth_i+1}/{num_synthetic_samples}, next lateral offset: {lateral_offset:.2f}m")

# -----------------------------------------------------------------------------------
# Initialisation functions
def create_run_folders(run_path):
    os.makedirs(run_path, exist_ok=True)
    for folder in [
        "images",
        "videos",
        "metrics",
        "configs_bk",
        "buffer_maps",
        "backup",
        "synthetic_image_samples"
    ]:
        os.makedirs(os.path.join(run_path, folder), exist_ok=True)

def copy_pretrained_checkpoint(pretrained_checkpoint_path, new_checkpoint_path):
    shutil.copy(pretrained_checkpoint_path, new_checkpoint_path)
    print(f"Copied pretrained checkpoint from {pretrained_checkpoint_path} to {new_checkpoint_path}")

def copy_config_file(pretrained_checkpoint_path, run_path):
    pretrained_dir = os.path.dirname(pretrained_checkpoint_path)
    pretrained_config_path = os.path.join(pretrained_dir, "config.yaml")
    new_config_path = os.path.join(run_path, "config.yaml")
    shutil.copy(pretrained_config_path, new_config_path)
    print(f"Copied config file from {pretrained_config_path} to {new_config_path}")

def main(
    pretrained_checkpoint_path: str,
    run_path: str,
    config_path: str = None,
):
    create_run_folders(run_path)
    checkpoint_path = os.path.join(run_path, "checkpoint_final.pth")
    if pretrained_checkpoint_path is None:
        shutil.copy(config_path, os.path.join(run_path, "config.yaml"))
        train_synthetic([], checkpoint_path, frame_index=154, lateral_offset=3, from_scratch=True)
        print("Trained initial model from scratch, starting synthetic training loop...")
    else:
        copy_pretrained_checkpoint(pretrained_checkpoint_path, checkpoint_path)
        copy_config_file(pretrained_checkpoint_path, run_path)

    run_training_loop(
        checkpoint_path=checkpoint_path,
        min_frame_index=20, max_frame_index=280,
        min_lateral_offset=0.8, max_lateral_offset=3.5,
        num_synthetic_samples=10
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument(
        "--output_root",
        default="./work_dirs/",
        help="path to save checkpoints and logs",
        type=str,
    )

    parser.add_argument(
        "--resume_from",
        default=None,
        help="path to checkpoint to resume from",
        type=str,
    )

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
    args = parser.parse_args()
    # Parse arguments
    pretrained_checkpoint_path = args.resume_from
    run_path = os.path.join(args.output_root, args.project, args.run_name)
    main(pretrained_checkpoint_path, run_path, config_path=args.config_file)
