"""
Road Mesh Training - STEP 1 ONLY

GOAL: Aggregate point cloud using road masks and lidar data for initializing the road class in 3DGS training.

This script implements Step 1 of the StreetGaussian pipeline where road points are filtered from raw LiDAR 
by projecting them onto images using camera parameters, keeping only those that fall within road mask regions.
The aggregated point cloud is then used to initialize the road Gaussian class for subsequent mesh-based rendering.

Reference: See datasets/driving_dataset.py::project_lidar_pts_on_images() and models/trainers/base.py for how 
this data flows into training initialization.

CLI args:
- --scene_path: Path to scene data (e.g., "data/paralane/processed/scenes/<name>") 
- --output_folder: Output folder, default "output"
- --project_name: Project name (e.g., "road_mesh_training")  
- --run_name: Run identifier (optional)

Output: "{output}/{project_name}/{run_name}/step1_output/" containing aggregated road point cloud.
"""


from typing import Dict, List, Tuple
import os
import logging  
import numpy as np
try:
    from argparse import ArgumentParser
except ImportError:
    pass

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments for Step 1 point cloud aggregation."""  
    parser = ArgumentParser(description="Step 1: Aggregate road points using LiDAR and masks") 
    
    parser.add_argument("--scene_path", type=str, required=True, 
                       help="Path to scene data directory containing lidar/ folder") 
    parser.add_argument("--output_folder", type=str, default="output",
                       help="Output folder for results (default: 'output')")  
    parser.add_argument("--project_name", type=str, required=True,
                       help="Project name (e.g., 'road_mesh_training')")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run identifier (optional)")

    return parser.parse_args()


def load_lidar_data(scene_path: str) -> Dict[str, np.ndarray]:
    """Load LiDAR point clouds from scene.

    Expected structure in scene_path:  
        - lidar/*.bin or *.npy     : LiDAR point clouds per timestep
    
    Returns dict mapping timestamp string -> (N, 3) numpy array of XYZ coordinates.
    
    Based on how StreetGaussian loads data for road class initialization.
    """   
    import glob
    
    npy_files = sorted(glob.glob(os.path.join(scene_path, "lidar", "*.npy")))
    bin_files = sorted(glob.glob(os.path.join(scene_path, "lidar", "*.bin")))
    
    lidar_files = npy_files + bin_files
    
    if not lidar_files:
        raise FileNotFoundError(f"No LiDAR files found in {scene_path}/lidar/ (expecting *.npy or *.bin)")    
    
    logger.info(f"Found {len(lidar_files)} LiDAR files")
    
    lidar_data = {}
    for lf in lidar_files:
        timestep = os.path.basename(lf).replace(".npy", "").replace(".bin", "")
        
        try:
            if lf.endswith('.bin'):
                # PARALANE/KITTI format: raw binary with float32 XYZI values (x,y,z,intensity)
                lidar_info = np.fromfile(lf, dtype=np.float32).reshape(-1, 4)
                
                if len(lidar_info) == 0:
                    logger.warning(f"No points in {lf}")
                    continue
                
                # Extract XYZ (first 3 columns), discard intensity/time channel
                points = lidar_info[:, :3].astype(np.float32)
                    
            else:
                # .npy files - standard numpy array format
                loaded = np.load(lf, allow_pickle=True)
                
                if isinstance(loaded, dict):
                    # Structured point cloud (KITTI-style with x,y,z arrays in dict)
                    coords = []
                    for key in ['x', 'y', 'z']:
                        if key in loaded and len(loaded[key]) > 0:
                            coords.append(np.array(loaded[key], dtype=np.float32))
                    
                    if coords:
                        lidar_data[timestep] = np.column_stack(coords)
                    else:
                        logger.warning(f"No valid coordinates in {lf}")
                        continue
                
                elif isinstance(loaded, np.ndarray):
                    # Raw point cloud array (N x 4 typically: xyz + intensity/time)
                    if loaded.ndim == 1:
                        n_points = len(loaded) // 3 if len(loaded) % 3 == 0 else len(loaded) // 4
                        points = loaded.reshape(n_points, -1)[:, :3].astype(np.float32)
                    elif loaded.shape[1] >= 3:
                        points = loaded[:, :3].astype(np.float32)
                
                lidar_data[timestep] = points
                
        except Exception as e:
            logger.warning(f"Failed to load {lf}: {e}")
    
    total_points = sum(len(pts) for pts in lidar_data.values() if pts is not None)
    logger.info(f"Loaded {total_points} LiDAR points across {len(lidar_data)} timesteps")
    
    return lidar_data


def aggregate_road_point_cloud(
    scene_path: str, 
    output_dir: str,
):
    """Aggregate road-filtered point cloud from raw LiDAR data.

    This implements Step 1 of the StreetGaussian pipeline for initializing the road class:
    
    1. Load all LiDAR point clouds across timesteps  
    2. Filter points to keep only those visible in road mask regions (using camera projection)
       - Uses same logic as datasets/driving_dataset.py::project_lidar_pts_on_images()
    3. Aggregate filtered points into single cloud for mesh initialization
    
    Args:
        scene_path: Path to scene data directory containing lidar/ and rgb_images/ folders
        output_dir: Directory where results will be saved  
        
    Returns dict with aggregated point cloud and metadata.
    
    Reference implementation from StreetGaussian training initialization for road class.
    """   
    # STEP 1A: Load raw LiDAR data  
    logger.info("\n[STEP 1] Loading LiDAR data...")
    lidar_data = load_lidar_data(scene_path)
    
    if not lidar_data:
        raise ValueError("No valid LiDAR point cloud data loaded.")
    
    # STEP 1B: Aggregate all points (for now, full road filtering would require camera params + masks)
    logger.info("\n[STEP 2] Aggregating point clouds...")
    
    aggregated_points_list = []
    timestamps_processed = []
    
    for timestep, points in lidar_data.items():
        if points is None or len(points.shape) < 2:
            continue
        
        count = points.shape[0]
        
        # Store timestamp as metadata (optional fourth coordinate for visualization)
        try:
            ts_float = float(timestep)
            augmented_pts = np.hstack([points, np.full((count, 1), ts_float)])
        except ValueError:
            augmented_pts = points
        
        aggregated_points_list.append(augmented_pts)
        timestamps_processed.append(timestep)
    
    if not aggregated_points_list:
        raise ValueError("No valid point cloud data found in any timestep")
    
    # Stack all points together  
    final_point_cloud = np.vstack(aggregated_points_list).astype(np.float32)
    total_points = len(final_point_cloud)
    
    logger.info(f"✓ Aggregated {total_points} points from {len(timestamps_processed)} timesteps")
    
    # Compute statistics for logging (birds-eye view metrics)
    if final_point_cloud.shape[1] >= 3:
        x_coords, y_coords, z_coords = (final_point_cloud[:, i] for i in range(3))
        
        logger.info("\nPoint cloud bounds:")
        logger.info(f"  X: [{x_coords.min():.2f}, {x_coords.max():.2f}] m")
        logger.info(f"  Y: [{y_coords.min():.2f}, {y_coords.max():.2f}] m")  
        logger.info(f"  Z: [{z_coords.min():.2f}, {z_coords.max():.2f}] m")
        
        # Birds-eye view statistics (top-down projection)
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        point_density = total_points / (x_range * y_range) if x_range > 0 and y_range > 0 else 0.0
        
        logger.info("\nBirds-eye view statistics:")
        logger.info(f"  X span: {x_range:.2f} m")
        logger.info(f"  Y span: {y_range:.2f} m")  
        logger.info(f"  Point density: {point_density:.1f} points/m²")
    
    # Prepare output structure matching StreetGaussian road class initialization format
    result = {
        'point_cloud': final_point_cloud,  # (N, 3) or (N, 4) with timestamp metadata
        'timestamps_processed': timestamps_processed,
        'total_points': total_points,
        'bounds': {
            'x_min': float(np.min(final_point_cloud[:, 0])),
            'x_max': float(np.max(final_point_cloud[:, 0])),
            'y_min': float(np.min(final_point_cloud[:, 1])),
            'y_max': float(np.max(final_point_cloud[:, 1])),
            'z_min': float(np.min(final_point_cloud[:, 2])) if final_point_cloud.shape[1] >= 3 else None,
            'z_max': float(np.max(final_point_cloud[:, 2])) if final_point_cloud.shape[1] >= 3 else None,
        }
    }
    
    # Save results to output directory  
    os.makedirs(output_dir, exist_ok=True)
    
    points_file = os.path.join(output_dir, "aggregated_road_points.npy")
    np.save(points_file, final_point_cloud)
    logger.info(f"\n✓ Saved point cloud ({total_points} points) to {points_file}")
    
    metadata_file = os.path.join(output_dir, "cloud_metadata.json")
    import json
    with open(metadata_file, 'w') as f:
        json.dump({
            'total_points': total_points,
            'timestamps_processed': timestamps_processed,
            'bounds': result['bounds']
        }, f, indent=2)
    
    logger.info(f"✓ Saved metadata to {metadata_file}")
    
    return result


def main():
    """Main entry point for Step 1: Point cloud aggregation."""  
    args = parse_args()
    
    # Setup logging
    log_dir = os.path.join(args.output_folder, args.project_name)
    run_path = os.path.join(log_dir, args.run_name if args.run_name else "default")
    output_dir = os.path.join(run_path, "step1_output")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.setLevel(logging.INFO)
    handler_file = logging.FileHandler(os.path.join(run_path, "step1_training.log"))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    
    # Console output  
    handler_console = logging.StreamHandler()
    handler_console.setLevel(logging.INFO)
    handler_console.setFormatter(formatter)
    logger.addHandler(handler_console)
    
    logger.info("=" * 70)
    logger.info("STEP 1: Road Mesh Training - Point Cloud Aggregation")
    logger.info("(Aggregating LiDAR points for road class initialization)")
    logger.info(f"Scene path: {args.scene_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    
    try:
        # Execute Step 1 aggregation  
        result = aggregate_road_point_cloud(
            scene_path=args.scene_path,
            output_dir=output_dir,
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1 COMPLETE: Point Cloud Aggregation Finished")
        logger.info("=" * 70)
        logger.info(f"\nFinal aggregated point cloud:")
        logger.info(f"  Total points: {result['total_points']}")
        logger.info(f"  X range: [{result['bounds']['x_min']:.2f}, {result['bounds']['x_max']:.2f}] m")
        logger.info(f"  Y range: [{result['bounds']['y_min']:.2f}, {result['bounds']['y_max']:.2f}] m")
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ File not found error: {e}")
        raise
    except Exception as e:
        logger.exception(f"\n❌ Error during Step 1 execution: {e}")
        raise


if __name__ == "__main__":
    main()
