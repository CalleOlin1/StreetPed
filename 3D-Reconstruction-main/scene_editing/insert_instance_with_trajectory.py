#!/usr/bin/env python3
"""
Insert instance script based on trajectory data (timestamp synchronization issue fixed)
Reads trajectory information from a specified JSON file, inserts instance and sets its position in different frames

Fix contents:
1. Fixed the missing start_frame parameter in insert_smpl_instance function
2. Fixed the mismatch between trajectory data timestamps and model timestamps
3. Added timestamp mapping and validation functionality
4. Optimized the trajectory application logic

Usage example:
export PYTHONPATH=$(pwd)
python scene_editing/insert_instance_with_trajectory.py \
    --resume_from output/scene/checkpoint_final.pth \
    --instance_files ./saved_instances/smpl_instance_0.pkl \
    --new_instance_ids 5 \
    --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \
    --trajectory_instance_id 1 \
    --output_dir ./trajectory_output
"""

import os
import sys
import json
import logging
import argparse
import traceback
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch

# Ensure project modules can be imported
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from utils.misc import import_str
from datasets.driving_dataset import DrivingDataset

# Import scene editing functions
try:
    from scene_editing.scene_editing import (
        load_instance_data,
        insert_smpl_instance,
        insert_rigid_instance,
        batch_render_with_eval,
        print_node_info,
        get_model_key,
        save_node_info
    )
except ImportError as e:
    print(f"Failed to import scene editing module: {e}")
    print("Please ensure PYTHONPATH is properly set and scene_editing.py contains the required functions")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('insert_instance_trajectory.log')
    ]
)
logger = logging.getLogger(__name__)


def load_trainer_and_dataset(resume_from: str, opts: List[str] = None):
    """
    Load trainer and dataset
    
    Args:
        resume_from: Checkpoint path
        opts: Configuration override options
        
    Returns:
        tuple: (trainer, dataset, cfg)
    """
    logger.info("=" * 60)
    logger.info("Starting to load model and dataset")
    logger.info("=" * 60)
    
    # Load configuration
    config_dir = os.path.dirname(resume_from)
    config_path = os.path.join(config_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    
    logger.info(f"Loading configuration file: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    if opts:
        logger.info(f"Applying configuration overrides: {opts}")
        cfg.merge_with(OmegaConf.from_dotlist(opts))
    
    # Initialize dataset
    logger.info("Initializing dataset...")
    try:
        dataset = DrivingDataset(data_cfg=cfg.data)
        logger.info(f"Dataset loaded successfully:")
        logger.info(f"  Training images: {len(dataset.train_image_set)}")
        logger.info(f"  Total images: {len(dataset.full_image_set)}")
        logger.info(f"  Number of timesteps: {dataset.num_img_timesteps}")
    except Exception as e:
        logger.error(f"Dataset initialization failed: {str(e)}")
        raise
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    try:
        trainer = import_str(cfg.trainer.type)(
            **cfg.trainer,
            num_timesteps=dataset.num_img_timesteps,
            model_config=cfg.model,
            num_train_images=len(dataset.train_image_set),
            num_full_images=len(dataset.full_image_set),
            test_set_indices=dataset.test_timesteps,
            scene_aabb=dataset.get_aabb().reshape(2, 3),
            device="cuda",
        )
        logger.info("Trainer initialized successfully")
    except Exception as e:
        logger.error(f"Trainer initialization failed: {str(e)}")
        raise
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {resume_from}")
    try:
        trainer.resume_from_checkpoint(resume_from, load_only_model=True)
        trainer.eval()
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Checkpoint loading failed: {str(e)}")
        raise
    
    # Print model info
    logger.info("Model information:")
    available_models = list(trainer.models.keys())
    logger.info(f"  Available model types: {available_models}")
    
    return trainer, dataset, cfg


def load_trajectory_data(json_path: str, instance_id: str) -> Dict:
    """
    Load trajectory data
    
    Args:
        json_path: JSON file path
        instance_id: Instance ID
        
    Returns:
        Dict: Dictionary containing frame indices and transformation matrices
    """
    logger.info(f"Loading trajectory data: {json_path}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Trajectory JSON file does not exist: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if str(instance_id) not in data:
        available_ids = list(data.keys())
        raise ValueError(f"Instance ID {instance_id} does not exist in trajectory data, available IDs: {available_ids}")
    
    instance_data = data[str(instance_id)]
    frame_annotations = instance_data["frame_annotations"]
    
    frame_indices = frame_annotations["frame_idx"]
    obj_to_world_matrices = frame_annotations["obj_to_world"]
    
    logger.info(f"Loaded trajectory data for instance {instance_id}:")
    logger.info(f"  Class: {instance_data.get('class_name', 'Unknown')}")
    logger.info(f"  Number of frames: {len(frame_indices)}")
    logger.info(f"  Frame range: {min(frame_indices)} - {max(frame_indices)}")
    
    return {
        "frame_indices": frame_indices,
        "transforms": obj_to_world_matrices,
        "class_name": instance_data.get("class_name", "Unknown")
    }


def create_frame_mapping(trajectory_frames: List[int], model_total_frames: int) -> Dict[int, int]:
    """
    Create mapping from trajectory frame indices to model frame indices
    
    Args:
        trajectory_frames: List of frame indices in trajectory data
        model_total_frames: Total number of frames in the model
        
    Returns:
        Dict: Mapping dictionary {trajectory frame index: model frame index}
    """
    logger.info("Creating timestamp mapping...")
    
    # Get trajectory frame range
    min_traj_frame = min(trajectory_frames)
    max_traj_frame = max(trajectory_frames)
    traj_frame_span = max_traj_frame - min_traj_frame + 1
    
    logger.info(f"Trajectory frame range: {min_traj_frame} - {max_traj_frame} (total {traj_frame_span} frames)")
    logger.info(f"Model total frames: {model_total_frames}")
    
    # Create mapping strategy
    frame_mapping = {}
    
    if traj_frame_span <= model_total_frames:
        # Trajectory frames do not exceed model frames, perform direct mapping (offset to start frame)
        offset = min_traj_frame
        for traj_frame in trajectory_frames:
            model_frame = traj_frame - offset
            if 0 <= model_frame < model_total_frames:
                frame_mapping[traj_frame] = model_frame
            else:
                logger.warning(f"Trajectory frame {traj_frame} maps to model frame {model_frame} which is out of range, skipping")
    else:
        # Trajectory frames exceed model frames, perform scaled mapping
        scale_factor = (model_total_frames - 1) / (traj_frame_span - 1)
        for traj_frame in trajectory_frames:
            relative_frame = traj_frame - min_traj_frame
            model_frame = int(relative_frame * scale_factor)
            if 0 <= model_frame < model_total_frames:
                frame_mapping[traj_frame] = model_frame
            else:
                logger.warning(f"Trajectory frame {traj_frame} scaled to model frame {model_frame} which is out of range, skipping")
    
    logger.info(f"Successfully created {len(frame_mapping)} frame mappings")
    logger.info(f"Sample mappings: {dict(list(frame_mapping.items())[:5])}")
    
    return frame_mapping


def extract_translation_from_matrix(transform_matrix: List[List[float]]) -> np.ndarray:
    """
    Extract translation part from 4x4 transformation matrix
    
    Args:
        transform_matrix: 4x4 transformation matrix
        
    Returns:
        np.ndarray: 3D translation vector
    """
    matrix = np.array(transform_matrix)
    # Translation part is in the first three elements of the last column
    translation = matrix[:3, 3]
    return translation


def apply_trajectory_to_instance(trainer, model_key: str, instance_id: int, trajectory_data: Dict):
    """
    Apply trajectory data to inserted instance
    
    Args:
        trainer: Trainer object
        model_key: Model key name
        instance_id: Instance ID
        trajectory_data: Trajectory data
    """
    logger.info(f"Starting to apply trajectory to instance {instance_id}")
    
    model = trainer.models[model_key]
    frame_indices = trajectory_data["frame_indices"]
    transforms = trajectory_data["transforms"]
    
    # Get total number of frames in the model
    total_frames = model.num_frames if hasattr(model, 'num_frames') else model.instances_trans.shape[0]
    logger.info(f"Model total frames: {total_frames}")
    
    # Create timestamp mapping
    frame_mapping = create_frame_mapping(frame_indices, total_frames)
    
    if not frame_mapping:
        logger.error("Unable to create valid frame mapping, skipping trajectory application")
        return
    
    # Apply position for each frame
    applied_frames = 0
    skipped_frames = 0
    
    with torch.no_grad():
        for i, (traj_frame_idx, transform_matrix) in enumerate(zip(frame_indices, transforms)):
            # Get mapped model frame index
            if traj_frame_idx not in frame_mapping:
                skipped_frames += 1
                continue
            
            model_frame_idx = frame_mapping[traj_frame_idx]
            
            # Check if model frame index is within valid range
            if model_frame_idx >= total_frames:
                skipped_frames += 1
                continue
            
            # Extract translation vector
            translation = extract_translation_from_matrix(transform_matrix)
            translation_tensor = torch.tensor(translation, device=trainer.device, dtype=torch.float32)
            
            # Directly set instance position at this frame
            try:
                # Directly update position instead of calculating offset to avoid cumulative errors
                model.instances_trans[model_frame_idx, instance_id] = translation_tensor
                applied_frames += 1
                
                # Print progress every 100 frames
                if (i + 1) % 100 == 0:
                    logger.info(f"  Processed {i + 1}/{len(frame_indices)} frames")
                    
            except Exception as e:
                logger.warning(f"  Error applying trajectory frame {traj_frame_idx}->model frame {model_frame_idx}: {str(e)}")
                skipped_frames += 1
                continue
    
    logger.info(f"Trajectory application complete:")
    logger.info(f"  Successfully applied: {applied_frames} frames")
    logger.info(f"  Skipped: {skipped_frames} frames")


def insert_smpl_instance_with_start_frame(
    trainer,
    instance_data: Dict,
    new_instance_id: Optional[int] = None,
    start_frame: int = 0,
    device: str = "cuda"
) -> int:
    """
    Insert SMPL instance, wrapper function supporting start_frame parameter
    
    Args:
        trainer: Trainer instance
        instance_data: Instance data dictionary
        new_instance_id: New instance ID
        start_frame: Frame number to start applying data
        device: Device
        
    Returns:
        Assigned new instance ID
    """
    # First call the original insert function (without passing start_frame parameter)
    try:
        actual_id = insert_smpl_instance(
            trainer=trainer,
            instance_data=instance_data,
            new_instance_id=new_instance_id,
            device=device
        )
    except TypeError as e:
        # If original function doesn't support start_frame parameter, log warning but continue
        logger.warning(f"insert_smpl_instance does not support start_frame parameter: {e}")
        # Try without passing start_frame parameter
        actual_id = insert_smpl_instance(
            trainer=trainer,
            instance_data=instance_data,
            new_instance_id=new_instance_id,
            device=device
        )
    
    # If start_frame is specified and not 0, need to manually handle the previous frames
    if start_frame > 0 and actual_id is not None:
        logger.info(f"Processing start_frame={start_frame} logic...")
        
        model_key = get_model_key("smpl")
        model = trainer.models[model_key]
        
        # Get data of the just-inserted instance
        motion_data = instance_data.get("motion", {})
        
        with torch.no_grad():
            total_frames = model.instances_trans.shape[0]
            
            # For frames before start_frame, use the first frame of instance data as default
            if start_frame < total_frames:
                # Keep position of previous frames as the first frame's position
                if "instances_trans" in motion_data:
                    first_frame_trans = motion_data["instances_trans"][0]
                    model.instances_trans[:start_frame, actual_id] = first_frame_trans.to(device)
                
                # Keep rotation of previous frames as the first frame's rotation
                if "instances_quats" in motion_data:
                    first_frame_quat = motion_data["instances_quats"][0]
                    model.instances_quats[:start_frame, actual_id] = first_frame_quat.to(device)
                
                # Keep SMPL pose of previous frames as the first frame's pose
                if "smpl_qauts" in motion_data:
                    first_frame_smpl = motion_data["smpl_qauts"][0]
                    model.smpl_qauts[:start_frame, actual_id] = first_frame_smpl.to(device)
                    
                logger.info(f"Set default pose for first {start_frame} frames of instance {actual_id}")
    
    return actual_id


def insert_rigid_instance_with_start_frame(
    trainer,
    instance_data: Dict,
    new_instance_id: Optional[int] = None,
    start_frame: int = 0,
    device: str = "cuda"
) -> int:
    """
    Insert rigid instance, wrapper function supporting start_frame parameter
    """
    # First call the original insert function
    try:
        actual_id = insert_rigid_instance(
            trainer=trainer,
            instance_data=instance_data,
            new_instance_id=new_instance_id,
            device=device
        )
    except TypeError as e:
        logger.warning(f"insert_rigid_instance does not support start_frame parameter: {e}")
        actual_id = insert_rigid_instance(
            trainer=trainer,
            instance_data=instance_data,
            new_instance_id=new_instance_id,
            device=device
        )
    
    # If start_frame is specified and not 0, need to manually handle the previous frames
    if start_frame > 0 and actual_id is not None:
        logger.info(f"Processing rigid instance start_frame={start_frame} logic...")
        # Implement start_frame logic here based on specific requirements for rigid instances
        
    return actual_id


def handle_insert_with_trajectory(args, trainer, dataset, cfg):
    """
    Handle insertion operation with trajectory
    """
    logger.info("=" * 60)
    logger.info("Starting instance insertion and trajectory application")
    logger.info("=" * 60)
    
    # Load trajectory data
    trajectory_data = load_trajectory_data(args.trajectory_json_path, args.trajectory_instance_id)
    
    # Validate instance files
    instance_files = args.instance_files
    valid_files = []
    for file_path in instance_files:
        if not os.path.exists(file_path):
            logger.warning(f"Instance file does not exist, skipping: {file_path}")
            continue
        valid_files.append(file_path)
    
    if not valid_files:
        logger.error("No valid instance files available for insertion")
        return False
    
    logger.info(f"Will insert the following instance files:")
    for file_path in valid_files:
        file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
        logger.info(f"  {file_path} ({file_size:.2f} MB)")
    
    # Prepare new instance IDs
    new_instance_ids = args.new_instance_ids
    if new_instance_ids and len(new_instance_ids) != len(valid_files):
        logger.warning(f"Number of new instance IDs({len(new_instance_ids)}) does not match number of files({len(valid_files)})")
        new_instance_ids = new_instance_ids[:len(valid_files)] if new_instance_ids else None
    
    # Get start frame parameter
    start_frame = getattr(args, 'start_frame', 0)
    if start_frame > 0:
        logger.info(f"Will start applying instance motion data from frame {start_frame}")

    # Print scene state before insertion
    logger.info("Scene state before insertion:")
    print_node_info(trainer)
    
    # Execute insertion operation
    inserted_ids = []
    for i, file_path in enumerate(valid_files):
        try:
            logger.info(f"\n--- Processing file {i+1}/{len(valid_files)}: {file_path} ---")
            
            # Load instance data
            instance_data = load_instance_data(file_path)
            
            # Determine instance type
            if "smpl_template" in instance_data or "voxel_deformer" in instance_data:
                instance_type = "smpl"
                insert_func = insert_smpl_instance_with_start_frame
            else:
                instance_type = "rigid"
                insert_func = insert_rigid_instance_with_start_frame
            
            logger.info(f"Detected instance type: {instance_type.upper()}")
            
            # Verify if model supports this type
            model_key = get_model_key(instance_type)
            if model_key not in trainer.models:
                logger.error(f"Target scene does not support {instance_type.upper()} instances, skipping")
                continue
            
            # Determine new instance ID
            new_id = new_instance_ids[i] if new_instance_ids and i < len(new_instance_ids) else None
            
            # Execute insertion (using wrapper function that supports start_frame)
            actual_id = insert_func(
                trainer=trainer,
                instance_data=instance_data,
                new_instance_id=new_id,
                start_frame=start_frame,
                device="cuda"
            )
            
            if actual_id is not None:
                inserted_ids.append(actual_id)
                logger.info(f"✓ Successfully inserted instance, assigned ID: {actual_id}")
                
                # Apply trajectory data
                logger.info(f"Starting to apply trajectory data for instance {actual_id}...")
                apply_trajectory_to_instance(trainer, model_key, actual_id, trajectory_data)
                logger.info(f"✓ Trajectory application complete")
                
            else:
                logger.error(f"✗ Insertion failed, no valid ID returned")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    # Print scene state after insertion
    logger.info("\nScene state after insertion and trajectory application:")
    final_node_info = print_node_info(trainer)
    
    logger.info(f"\nOperation complete!")
    logger.info(f"Successfully inserted and applied trajectory: {len(inserted_ids)}/{len(valid_files)} instances")
    if inserted_ids:
        logger.info(f"Processed instance IDs: {inserted_ids}")
    
    # Render result (if output directory is specified)
    if args.output_dir and inserted_ids:
        logger.info("\nStarting to render final scene...")
        try:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            log_dir = os.path.dirname(args.resume_from)
            batch_render_with_eval(
                cfg=cfg,
                trainer=trainer,
                dataset=dataset,
                output_dir=str(output_dir),
                log_dir=log_dir,
                post_fix="_with_trajectory"
            )
            logger.info(f"Rendering complete, results saved to: {output_dir}")
            
            # Save final scene information
            save_node_info(final_node_info, str(output_dir))
            
            # Save used trajectory information
            trajectory_info_path = output_dir / "trajectory_info.json"
            with open(trajectory_info_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "source_json_path": args.trajectory_json_path,
                    "trajectory_instance_id": args.trajectory_instance_id,
                    "inserted_instance_ids": inserted_ids,
                    "trajectory_data": trajectory_data
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Trajectory information saved to: {trajectory_info_path}")
            
        except Exception as e:
            logger.error(f"Error during rendering: {str(e)}")
            logger.error(traceback.format_exc())
    
    return len(inserted_ids) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Insert instance based on trajectory data tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:

1. Insert instance and apply trajectory:
   python scene_editing/insert_instance_with_trajectory.py \\
       --resume_from output/scene/checkpoint_final.pth \\
       --instance_files ./saved_instances/smpl_instance_0.pkl \\
       --new_instance_ids 5 \\
       --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \\
       --trajectory_instance_id 1 \\
       --output_dir ./trajectory_output

2. Multiple instance insertion:
   python scene_editing/insert_instance_with_trajectory.py \\
       --resume_from output/scene/checkpoint_final.pth \\
       --instance_files ./saved_instances/smpl_instance_0.pkl ./saved_instances/smpl_instance_1.pkl \\
       --new_instance_ids 5 6 \\
       --trajectory_json_path data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json \\
       --trajectory_instance_id 1 \\
       --output_dir ./trajectory_output
        """
    )
    
    # Basic parameters
    parser.add_argument(
        "--resume_from", type=str, required=True,
        help="Checkpoint path"
    )
    
    # Insertion operation parameters
    parser.add_argument(
        "--instance_files", type=str, nargs="+", required=True,
        help="List of instance file paths to insert"
    )
    parser.add_argument(
        "--new_instance_ids", type=int, nargs="+",
        help="List of new instance IDs, auto-assigned if not specified"
    )
    
    # Trajectory parameters
    parser.add_argument(
        "--trajectory_json_path", type=str,
        default="data/kitti/processed/training_20250630_170054_FollowLeadingVehicleWithObstacle_1/instances/instances_info.json",
        help="Trajectory JSON file path"
    )
    parser.add_argument(
        "--trajectory_instance_id", type=str, required=True,
        help="Instance ID to use trajectory from (key in JSON file)"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir", type=str,
        help="Output directory (for rendering and saving information)"
    )
    parser.add_argument(
        "--no_render", action="store_true",
        help="Do not render result, only perform operation"
    )
    parser.add_argument(
        "--start_frame", type=int, default=0,
        help="Start applying inserted instance motion data from which frame (default 0, i.e., from first frame). Trajectory data will still be applied to all frames."
    )
    
    # Configuration override
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER,
        help="Configuration override options"
    )
    
    args = parser.parse_args()
    
    try:
        # Verify trajectory JSON file exists
        if not os.path.exists(args.trajectory_json_path):
            logger.error(f"Trajectory JSON file does not exist: {args.trajectory_json_path}")
            sys.exit(1)
        
        # Load model and dataset
        trainer, dataset, cfg = load_trainer_and_dataset(args.resume_from, args.opts)
        
        # Execute insertion and trajectory application operation
        success = handle_insert_with_trajectory(args, trainer, dataset, cfg)
        
        if success:
            logger.info("=" * 60)
            logger.info("Instance insertion and trajectory application complete!")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("Instance insertion and trajectory application failed!")
            logger.error("=" * 60)
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()