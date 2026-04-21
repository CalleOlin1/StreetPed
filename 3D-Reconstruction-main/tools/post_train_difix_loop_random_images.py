"""
The goal of this file is to create a structured and memory efficient approach to improve the quality of a trained checkpoint file.

"""
import os
import argparse
import io
import shutil
import gc
import subprocess
import tempfile
import pickle
import sys
from PIL import Image
import numpy as np

from tools.difix_sender_receiver import difix_repair, difix_repair_batch

SEGFORMER_REPO_PATH = "/home/hstromgr/Documents/Github/SegFormer"
SEGFORMER_CONDA_ENV = "segformer"
SEGFORMER_CONFIG_PATH = os.path.join(
    SEGFORMER_REPO_PATH,
    "local_configs",
    "segformer",
    "B5",
    "segformer.b5.1024x1024.city.160k.py",
)
SEGFORMER_CHECKPOINT_PATH = os.path.join(
    SEGFORMER_REPO_PATH,
    "pretrained",
    "segformer.b5.1024x1024.city.160k.pth",
)
ROAD_CLASS_ID = 0
SKY_CLASS_ID = 10

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

def _get_class_masks(
    images,
    target_class_id,
    mask_name,
    segformer_path=SEGFORMER_REPO_PATH,
    config=SEGFORMER_CONFIG_PATH,
    checkpoint=SEGFORMER_CHECKPOINT_PATH,
    device="cuda:0",
):
    """Spawn the mask worker subprocess and return one binary mask per input image.

    Args:
        images: List of image paths, PIL images, or numpy arrays.
        segformer_path: Optional path to the SegFormer checkout.
        config: Optional model config path.
        checkpoint: Optional checkpoint path.
        device: Torch device string for the worker process.

    Returns:
        List of PIL.Image.Image masks in L mode, one per input image.
    """

    def _to_pil(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, os.PathLike)):
            return Image.open(image).convert("RGB")
        image_np = np.asarray(image)
        if image_np.ndim == 2:
            image_np = np.stack([image_np, image_np, image_np], axis=-1)
        elif image_np.ndim == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        return Image.fromarray(image_np).convert("RGB")

    def _serialize_image(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _deserialize_image(img_bytes):
        image = Image.open(io.BytesIO(img_bytes))
        image.load()
        return image

    def _read_exact(pipe, n):
        buf = b""
        while len(buf) < n:
            chunk = pipe.read(n - len(buf))
            if not chunk:
                raise EOFError(f"Unexpected EOF while reading {mask_name}-mask subprocess output")
            buf += chunk
        return buf

    worker_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "tools", "pipe_extract_masks.py"))

    if shutil.which("conda"):
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            SEGFORMER_CONDA_ENV,
            "python",
            "-u",
            worker_script,
            "--worker",
            f"--device={device}",
            f"--target_class_id={target_class_id}",
        ]
    else:
        cmd = [
            sys.executable,
            "-u",
            worker_script,
            "--worker",
            f"--device={device}",
            f"--target_class_id={target_class_id}",
        ]
    if segformer_path is not None:
        cmd.append(f"--segformer_path={segformer_path}")
    if config is not None:
        cmd.append(f"--config={config}")
    if checkpoint is not None:
        cmd.append(f"--checkpoint={checkpoint}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    assert proc.stdin is not None and proc.stdout is not None

    try:
        startup_lines = []
        while True:
            ready_line = proc.stdout.readline().decode(errors="replace").strip()
            if ready_line == "READY":
                break
            if ready_line == "":
                raise RuntimeError(
                    f"{mask_name.title()}-mask worker failed to start before EOF. "
                    f"Startup output: {startup_lines}"
                )
            startup_lines.append(ready_line)
            print(f"[{mask_name}-mask worker] {ready_line}", file=sys.stderr)

        pil_images = [_to_pil(image) for image in images]
        img_bytes_list = [_serialize_image(image) for image in pil_images]

        proc.stdin.write(len(img_bytes_list).to_bytes(4, "big"))
        for img_bytes in img_bytes_list:
            proc.stdin.write(len(img_bytes).to_bytes(4, "big"))
            proc.stdin.write(img_bytes)
        proc.stdin.flush()
        proc.stdin.close()

        mask_count = int.from_bytes(_read_exact(proc.stdout, 4), "big")
        masks = []
        for _ in range(mask_count):
            mask_len = int.from_bytes(_read_exact(proc.stdout, 4), "big")
            mask_bytes = _read_exact(proc.stdout, mask_len)
            masks.append(_deserialize_image(mask_bytes))

        proc.wait()
        return masks
    finally:
        if proc.poll() is None:
            proc.kill()


def get_sky_masks(images, segformer_path=SEGFORMER_REPO_PATH, config=SEGFORMER_CONFIG_PATH, checkpoint=SEGFORMER_CHECKPOINT_PATH, device="cuda:0"):
    return _get_class_masks(
        images,
        target_class_id=SKY_CLASS_ID,
        mask_name="sky",
        segformer_path=segformer_path,
        config=config,
        checkpoint=checkpoint,
        device=device,
    )


def get_road_masks(images, segformer_path=SEGFORMER_REPO_PATH, config=SEGFORMER_CONFIG_PATH, checkpoint=SEGFORMER_CHECKPOINT_PATH, device="cuda:0"):
    return _get_class_masks(
        images,
        target_class_id=ROAD_CLASS_ID,
        mask_name="road",
        segformer_path=segformer_path,
        config=config,
        checkpoint=checkpoint,
        device=device,
    )

def train_synthetic(synthetic_samples, checkpoint_path, frame_index, lateral_offset, from_scratch=False, num_iters=3000):
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
        f"--split_steps={num_iters}",
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
                  batch_size=100):
    # Get random unique frame indices for this batch
    frame_indices = np.random.choice(np.arange(min_frame_index, max_frame_index), size=batch_size, replace=False)
    synthetic_samples = []
    lateral_offsets = np.linspace(0.1, max_lateral_offset, batch_size)
    # lateral_offsets = np.full(frame_indices.shape, lateral_offset)
    batch_results = render_novel_sample_list(checkpoint_path, frame_indices.tolist(), lateral_offsets.tolist())

    # Prepare lists of images for batch repair
    novel_imgs = [tensor_to_image(novel_sample["rendered_rgb"]) for _, novel_sample in batch_results]
    ref_imgs   = [tensor_to_image(novel_sample["reference_rgb"]) for _, novel_sample in batch_results]

    # Batch repair
    repaired_images = difix_repair_batch(novel_imgs, ref_imgs)

    # Sky / road masks
    sky_masks = get_sky_masks(repaired_images, segformer_path=SEGFORMER_REPO_PATH)
    # road_masks = [novel_sample.get("road_masks") for _, novel_sample in batch_results]
    # if any(mask is None for mask in road_masks):
    #     print("road_masks were not correctly generated, using Segformer instead")
    #     road_masks = get_road_masks(repaired_images, segformer_path=SEGFORMER_REPO_PATH)
    road_masks = get_road_masks(repaired_images, segformer_path=SEGFORMER_REPO_PATH)

    # Logging and update samples
    for i, ((curr_step, novel_sample), novel_img, repaired_image, sky_mask, road_mask) in enumerate(zip(batch_results, novel_imgs, repaired_images, sky_masks, road_masks)):
        log_synthetic_image(novel_img, repaired_image, run_path, len(synthetic_samples))
        # Convert repaired_image (PIL) back to numpy array (normalized float32, CHW)
        repaired_array = image_to_array(repaired_image, normalize=True)
        novel_sample["rendered_rgb"] = repaired_array
        novel_sample["sky_masks"] = (np.asarray(sky_mask.convert("L"), dtype=np.uint8) > 0).astype(np.float32)
        if isinstance(road_mask, Image.Image):
            novel_sample["road_masks"] = (np.asarray(road_mask.convert("L"), dtype=np.uint8) > 0).astype(np.float32)
        else:
            novel_sample["road_masks"] = np.asarray(road_mask, dtype=np.float32)
        # Append the full sample dict (with repaired image) to the list
        synthetic_samples.append(novel_sample)

    # Start training with our synthetic images
    # Use the first frame_index from this batch for training (or another policy as needed)
    train_synthetic(synthetic_samples, checkpoint_path, 154, max_lateral_offset, num_iters=3000)
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
    # os.path.samefile requires both paths to exist; guard for fresh run directories.
    same_file = (
        os.path.exists(new_checkpoint_path)
        and os.path.samefile(pretrained_checkpoint_path, new_checkpoint_path)
    )
    if not same_file:
        shutil.copy(pretrained_checkpoint_path, new_checkpoint_path)
        print(f"Copied pretrained checkpoint from {pretrained_checkpoint_path} to {new_checkpoint_path}")

def copy_config_file(pretrained_checkpoint_path, run_path):
    pretrained_dir = os.path.dirname(pretrained_checkpoint_path)
    pretrained_config_path = os.path.join(pretrained_dir, "config.yaml")
    new_config_path = os.path.join(run_path, "config.yaml")
    # os.path.samefile requires both paths to exist; guard for fresh run directories.
    same_file = (
        os.path.exists(new_config_path)
        and os.path.samefile(pretrained_config_path, new_config_path)
    )
    if not same_file:
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
        new_config_path = os.path.join(run_path, "config.yaml")
        # os.path.samefile requires both paths to exist; guard for fresh run directories.
        same_file = (
            os.path.exists(new_config_path)
            and os.path.samefile(config_path, new_config_path)
        )
        if not same_file:
            shutil.copy(config_path, os.path.join(run_path, "config.yaml"))
        train_synthetic([], checkpoint_path, frame_index=154, lateral_offset=3, from_scratch=True, num_iters=12000)
        # train_synthetic([], checkpoint_path, frame_index=154, lateral_offset=3, from_scratch=True, num_iters=50000)
        # exit()
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
