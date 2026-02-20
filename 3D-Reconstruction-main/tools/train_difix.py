"""
This file performs training on a preprocessed Kitti dataset. TODO support other datasets
It creates synthetic data using Difix3D+. This synthetic data is inserted in the training data to improve quality of 3DGS reconstruction.
Finally the checkpoint model is saved.
"""
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from tqdm import tqdm
import os
import train
import types
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_args(
    dataset_path: str,
    output_root: str = "./outputs",
    project: str = "3dgs_project",
    run_name: str = "run_001",
    resume_from: str = None,
    enable_wandb: bool = False,
    enable_viewer: bool = False,
    viewer_port: int = 8080,
    config_file: str = "./configs/train_config.yaml",
    opts: list = None,
):
    """Create a minimal args object compatible with main()."""
    
    if opts is None:
        opts = []

    args = types.SimpleNamespace()
    args.dataset_path = dataset_path          # Your dataset root folder
    args.output_root = output_root            # Where outputs/logs/checkpoints go
    args.project = project                    # Project name (used for logging)
    args.run_name = run_name                  # Run name (used for logging/checkpoints)
    args.resume_from = resume_from            # Optional checkpoint path
    args.enable_wandb = enable_wandb          # W&B logging
    args.enable_viewer = enable_viewer        # Viewer flag
    args.viewer_port = viewer_port            # Port for viewer
    args.config_file = config_file            # Path to your training config YAML
    args.opts = opts                           # Any CLI overrides (list of strings)
    
    return args
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_dataset(dataset_path: str):
    pass

def train(dataset, all_synthetic_data: list = None):
    args = make_args(
        dataset_path=dataset_path,
        enable_wandb=False,
    )
    step = train.main(args)

def render_novel(model, curr_offset):
    pass

def get_fine_dynamic_masks(novel_frames, reference_images = None):
    pass

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def iterative_training(dataset_path: str, checkpoint_path: str, num_iterations: int = 5, lateral_offset_max: float = 5, save_synthetic: bool = False):
    print(f"--- Loading dataset at {dataset_path} ---")
    dataset = load_dataset(dataset_path)

    print(f"--- Starting iterative training for 3DGS ---")
    delta_offset = lateral_offset_max / num_iterations
    curr_offset = delta_offset
    all_synthetic_data = []
    for i in range(num_iterations):
        print(f"--- Training iteration {i} out of {num_iterations} ---")
        model = train(dataset, all_synthetic_data)

        print(f"--- Rendering novel view at distance {curr_offset} m ---")
        curr_synthetic_data = {}
        unprocessed_novel_rbgs = render_novel(model, curr_offset)
        
        print(f"--- Applying Difix to novel views ---")
        curr_synthetic_data["processed_novel_rgbs"] = apply_difix(unprocessed_novel_rbgs)

        print(f"--- Applying fine dynamic masks for frames in novel view ---")
        fine_dynamic_masks = get_fine_dynamic_masks(novel_frames, reference_images = )

        print(f"--- Saving data and preparing for next iteration ---")
        all_synthetic_data.append(curr_synthetic_data)
        curr_offset += delta_offset

    print(f"--- Finished training. Saving checkpoint at {checkpoint_path} ---")
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("""
          Started diffusion training.
          Parsing arguments...
          """)
    dataset_path = "/home/"
    checkpoint_path = ""
    iterative_training(dataset_path, checkpoint_path)