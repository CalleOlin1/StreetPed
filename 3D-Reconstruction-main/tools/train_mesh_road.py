#!/usr/bin/env python3
"""
Road Mesh Training Script - Step 1 Implementation

This script implements Step 1 of the road mesh training pipeline:
- Aggregate point cloud using road masks and lidar data
- Log bird's eye view visualization of the extracted points

The basic algorithm is as follows:
- 1 Aggregate point cloud using road masks and lidar data (IMPLEMENTED HERE)
    - Use existing Driving_Dataset code to load data, masks, and images
    - Extract LiDAR points that project into road mask regions
    - Log bird's eye view of the aggregated point cloud
- 2 Create a 3d mesh using the point cloud (TODO: Step 2)
    - Assume no overlap along the up axis
    - Use reasonable amount of polys (low complexity)
    - Log mesh as .pth
- 3 Create an image buffer for the road mesh (TODO: Step 3)
    - Use 4096x4096 resolution initially
    - Map pixels to mesh using x and y axis with scaling factor
- 4 Project rays using rgb images and cam position onto the mesh (TODO: Step 4)
    - Only project road areas according to road mask
    - Fill image buffer using RGB data from images
    - Log image buffer as an image
    - Log mesh with texture applied
    - Log example render from camera positions

(DONT REMOVE THIS SPECIFICATION SHEET)
"""

import os
import sys
import argparse
import logging
from typing import Union, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.driving_dataset import DrivingDataset
from utils.misc import export_points_to_ply, import_str


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train road mesh from LiDAR and road masks - Step 1"
    )
    parser.add_argument(
        "--scene_path", 
        type=str, 
        required=True, 
        help="Path to scene data directory (e.g., 'data/paralane/processed/scene_000_clip_000')"
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        default="output",
        help="Output folder for project results"
    )
    parser.add_argument(
        "--project_name", 
        type=str, 
        default="road_mesh_training",
        help="Project name (e.g., 'road_mesh_training')"
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default="run_000",
        help="Run identifier (e.g., 'run_000')"
    )
    
    # Optional configuration overrides
    parser.add_argument(
        "--config_path", 
        type=str, 
        default=None,
        help="Path to dataset config file (optional)"
    )
    
    return parser.parse_args()


def setup_output_directories(output_folder: str) -> Tuple[str, str]:
    """Create output directories and set up logging."""
    # Construct full output path
    base_path = os.path.join(output_folder, "road_mesh_training", "run_000")
    
    step1_dir = os.path.join(base_path, "step1_pointcloud_visualization")
    os.makedirs(step1_dir, exist_ok=True)
    
    # Set up logging to file and console
    log_file = os.path.join(base_path, "training.log")
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return base_path, step1_dir


def aggregate_road_pointcloud(dataset: DrivingDataset, device: torch.device):
    """
    Aggregate LiDAR points that project into road mask regions.
    
    Args:
        dataset: Initialized DrivingDataset instance with loaded data
        device: PyTorch device for computation
        
    Returns:
        Tuple of (road_pts, road_colors) tensors containing aggregated point cloud
    """
    logger.info("Starting road LiDAR point aggregation...")
    
    # Get all LiDAR points that project into road mask regions across all frames/cameras
    road_lidar_indices = dataset.get_lidar_indices_from_mask_region(mask_attr="road_masks")
    num_road_points = len(road_lidar_indices)
    
    logger.info(f"Found {num_road_points} LiDAR points projecting to road masks")
    
    if num_road_points == 0:
        logger.warning("No road LiDAR points found! Check scene configuration and road mask availability.")
        return torch.empty(0, 3, device=device), torch.empty(0, 3, device=device)
    
    # Extract the actual point coordinates and colors from aggregated indices
    road_pts, road_colors = dataset.get_lidar_points_from_mask_region(
        mask_attr="road_masks",
        num_samples=None,  # Use all available points
        return_color=True,
        device=device
    )
    
    logger.info(f"Successfully extracted {len(road_pts)} unique road LiDAR points")
    
    if len(road_pts) == 0:
        logger.warning("Extracted point cloud is empty after processing!")
        
    return road_pts, road_colors


def log_birds_eye_view(pts_xyz: np.ndarray, colors: np.ndarray, output_dir: str):
    """
    Create and save bird's eye view visualization of the point cloud.
    
    Args:
        pts_xyz: N x 3 numpy array of point coordinates in meters
        colors: N x 3 numpy array of RGB colors (0-1 or 0-255)
        output_dir: Directory to save visualizations
    """
    if len(pts_xyz) == 0:
        logger.warning("Cannot create visualization for empty point cloud")
        return
    
    num_points = len(pts_xyz)
    
    # Create bird's eye view (top-down, looking along Y-axis or Z-axis depending on coordinate system)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # X-Y plane (bird's eye view - top down)
    ax_xy = axes[0]
    scatter = ax_xy.scatter(pts_xyz[:, 0], pts_xyz[:, 1], c=range(num_points), 
                           cmap='viridis', s=1, alpha=0.5)
    ax_xy.set_xlabel('X (meters)', fontsize=12)
    ax_xy.set_ylabel('Y (meters)', fontsize=12)
    ax_xy.set_title(f"Bird's Eye View - {num_points:,} Road Points", fontsize=14, fontweight='bold')
    ax_xy.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax_xy, label='Point Index', shrink=0.8)
    
    # X-Z plane (side view - looking from side)
    ax_xz = axes[1]
    scatter2 = ax_xz.scatter(pts_xyz[:, 0], pts_xyz[:, 2], c=range(num_points), 
                            cmap='viridis', s=1, alpha=0.5)
    ax_xz.set_xlabel('X (meters)', fontsize=12)
    ax_xz.set_ylabel('Z height (meters)', fontsize=12)
    ax_xz.set_title("Side View - X-Z Plane", fontsize=14, fontweight='bold')
    ax_xz.grid(True, alpha=0.3)
    
    # Y-Z plane (front view - looking from front/side)  
    ax_yz = axes[2]
    scatter3 = ax_yz.scatter(pts_xyz[:, 1], pts_xyz[:, 2], c=range(num_points), 
                            cmap='viridis', s=1, alpha=0.5)
    ax_yz.set_xlabel('Y (meters)', fontsize=12)
    ax_yz.set_ylabel('Z height (meters)', fontsize=12)
    ax_yz.set_title("Front View - Y-Z Plane", fontsize=14, fontweight='bold')
    ax_yz.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization with high DPI for publication quality
    viz_path = os.path.join(output_dir, "birds_eye_view.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved bird's eye view visualization to {viz_path}")
    
    # Also save a lower resolution version for quick viewing
    viz_path_lowres = os.path.join(output_dir, "birds_eye_view_preview.png")
    plt.savefig(viz_path_lowres, dpi=72, bbox_inches='tight')
    logger.info(f"Saved preview visualization to {viz_path_lowres}")


def log_point_cloud_statistics(pts_xyz: np.ndarray):
    """Log detailed statistics about the point cloud."""
    if len(pts_xyz) == 0:
        return
    
    # Bounding box analysis
    x_min, y_min, z_min = pts_xyz.min(axis=0)
    x_max, y_max, z_max = pts_xyz.max(axis=0)
    
    logger.info("=" * 60)
    logger.info("ROAD POINT CLOUD STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total points: {len(pts_xyz):,}")
    logger.info("")
    logger.info("Bounding Box (meters):")
    logger.info(f"  X-axis: [{x_min:.2f}, {x_max:.2f}] - span: {x_max - x_min:.2f} m")
    logger.info(f"  Y-axis: [{y_min:.2f}, {y_max:.2f}] - span: {y_max - y_min:.2f} m")
    logger.info(f"  Z-axis (height): [{z_min:.2f}, {z_max:.2f}] - span: {z_max - z_min:.2f} m")
    logger.info("")
    
    # Height distribution analysis for road surface characteristics
    mean_height = pts_xyz[:, 2].mean()
    std_height = pts_xyz[:, 2].std()
    median_height = np.median(pts_xyz[:, 2])
    
    logger.info("Height Distribution (Z-axis):")
    logger.info(f"  Mean height: {mean_height:.4f} ± {std_height:.4f} meters")
    logger.info(f"  Median height: {median_height:.4f} meters")
    logger.info(f"  Height range: {pts_xyz[:, 2].max() - pts_xyz[:, 2].min():.4f} meters")
    
    # Point density analysis (points per square meter in X-Y plane)
    xy_area = (x_max - x_min) * (y_max - y_min) if (x_max > x_min and y_max > y_min) else 1.0
    point_density = len(pts_xyz) / max(xy_area, 1.0)
    
    logger.info("")
    logger.info("Point Density:")
    logger.info(f"  X-Y coverage area: {xy_area:.2f} m²")
    logger.info(f"  Point density: {point_density:.2f} points/m² in X-Y plane")
    
    # Color statistics (if available)
    if pts_xyz.shape[1] >= 3 and len(pts_xyz) > 0:
        mean_color = pts_xyz[:, :3].mean(axis=0) if pts_xyz.shape[-1] == 6 else np.array([0.5, 0.5, 0.5])
        logger.info("")
        logger.info("Color Statistics (RGB):")
        logger.info(f"  Mean color: [{mean_color[0]:.3f}, {mean_color[1]:.3f}, {mean_color[2]:.3f}]")


def export_point_cloud(pts_xyz: np.ndarray, colors: np.ndarray, output_dir: str):
    """Export point cloud to PLY format for external inspection."""
    ply_path = os.path.join(output_dir, "road_pointcloud.ply")
    
    if len(pts_xyz) > 0 and pts_xyz.shape[1] >= 3:
        # Use first 3 columns as XYZ coordinates (assuming last 3 are RGB colors)
        xyz_coords = pts_xyz[:, :3].astype(np.float32)
        
        # Extract or generate colors
        if len(pts_xyz) > 0 and colors is not None and len(colors) == len(xyz_coords):
            rgb_colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
        else:
            # Generate grayscale based on height (Z coordinate) if no color data available
            z_values = xyz_coords[:, 2]
            normalized_z = (z_values - z_values.min()) / max(z_values.max() - z_values.min(), 1e-6)
            rgb_colors = np.stack([normalized_z, normalized_z, normalized_z], axis=1).astype(np.uint8) * 255
        
        export_points_to_ply(
            xyz_coords,
            rgb_colors.astype(np.float32),
            save_path=ply_path
        )
        
        logger.info(f"Exported point cloud to {ply_path}")
    else:
        # Create empty PLY file for consistency
        with open(ply_path, 'w') as f:
            f.write("PLY\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex 0\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
        
        logger.warning(f"Created empty PLY file at {ply_path}")


def save_pruned_pointcloud_image(pts_xyz: np.ndarray, colors: np.ndarray, output_dir: str):
    """
    Save a pruned/decimated point cloud image for efficient storage and quick viewing.
    
    Args:
        pts_xyz: N x 3 numpy array of point coordinates in meters
        colors: N x 3 numpy array of RGB colors (0-1)
        output_dir: Directory to save the pruned visualization
    """
    if len(pts_xyz) == 0 or len(colors) == 0:
        logger.warning("Cannot create pruned image for empty point cloud")
        return
    
    num_points = len(pts_xyz)
    
    # Prune by downsampling points while preserving spatial distribution
    max_display_points = 50000  # Limit for visualization performance
    
    if num_points > max_display_points:
        logger.info(f"Pruning point cloud from {num_points:,} to {max_display_points:,} points")
        
        # Use uniform sampling based on X-Y grid bins
        x_min, y_min = pts_xyz[:, 0].min(), pts_xyz[:, 1].min()
        x_max, y_max = pts_xyz[:, 0].max(), pts_xyz[:, 1].max()
        
        # Create a spatial hash/grid for uniform sampling
        bin_size = max((x_max - x_min) / np.sqrt(max_display_points), 
                       (y_max - y_min) / np.sqrt(max_display_points))
        if bin_size < 0.01:
            bin_size = 0.05
        
        # Assign each point to a grid cell
        grid_x = ((pts_xyz[:, 0] - x_min) / bin_size).astype(int)
        grid_y = ((pts_xyz[:, 1] - y_min) / bin_size).astype(int)
        
        # Create unique grid identifiers and sample one point per occupied cell
        grid_ids = (grid_x * 10000 + grid_y).astype(np.int64)
        unique_grids, inverse_indices = np.unique(grid_ids, return_inverse=True)
        
        # Sample up to max_display_points points uniformly from each bin
        sampled_indices = []
        for i in range(len(unique_grids)):
            cell_mask = (inverse_indices == i)
            if cell_mask.sum() > 0:
                sample_size = min(cell_mask.sum(), int(max_display_points / len(unique_grids)))
                selected_local = np.random.choice(np.where(cell_mask)[0], size=sample_size, replace=False)
                sampled_indices.extend(np.where(cell_mask)[0][selected_local])
        
        # Ensure we have enough points (at least one per occupied bin)
        if len(sampled_indices) < min(num_points, max_display_points):
            logger.warning("Pruning resulted in fewer than expected points; using all available")
            sampled_indices = np.arange(num_points)
    else:
        # No pruning needed - use all points
        sampled_indices = np.arange(num_points)
    
    pruned_pts = pts_xyz[sampled_indices]
    pruned_colors = colors[sampled_indices] if len(colors) > 0 else None
    
    logger.info(f"Saved pruned point cloud: {len(pruned_pts)} points")
    
    # Create visualization of pruned point cloud (single view - top down with color coding by height)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if len(pruned_colors) > 0:
        scatter = ax.scatter(pruned_pts[:, 0], pruned_pts[:, 1], 
                            c=pruned_pts[:, 2], cmap='viridis', s=1.5, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Z height (meters)', shrink=0.8)
    else:
        scatter = ax.scatter(pruned_pts[:, 0], pruned_pts[:, 1], 
                            c=np.arange(len(pruned_pts)), cmap='viridis', s=1.5, alpha=0.7)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f"Pruned Road Point Cloud - {len(pruned_pts):,} points", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save pruned visualization with high resolution
    pruned_path = os.path.join(output_dir, "pruned_pointcloud.png")
    plt.savefig(pruned_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved pruned point cloud image to {pruned_path}")


def main():
    """Main execution function for Step 1."""
    
    # Parse command line arguments
    args = parse_args()
    
    # Setup output directories and logging
    base_path, step1_dir = setup_output_directories(args.output_folder)
    logger.info(f"Output directory: {base_path}")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset configuration - merge CLI args with config file if provided
    base_config = OmegaConf.create({
        "data_root": os.path.dirname(args.scene_path),  # Parent directory of scene path as data root
        "scene_idx": os.path.basename(args.scene_path),   # Scene name from path
        "start_timestep": 0,
        "end_timestep": -1,
        "preload_device": device.type,
    })
    
    if args.config_path and os.path.exists(args.config_path):
        logger.info(f"Loading configuration from: {args.config_path}")
        
        # Load config file (may have 'data:' wrapper or be flat)
        raw_cfg = OmegaConf.load(args.config_path)
        
        def extract_data_section(cfg):
            """Extract the data section if it exists under a 'data' key."""
            try:
                content = getattr(cfg, '_content', {})
                if hasattr(content, 'get'):
                    return content.get('data') or {}
            except Exception as e:
                logger.debug(f"Error extracting data section: {e}")
            
            # If no wrapper found, use the config itself (flat structure)
            try:
                return dict(cfg) if hasattr(cfg, '__dict__') else cfg
            except:
                return {}
        
        data_section = extract_data_section(raw_cfg)
        
        # Build configuration dictionary from extracted section + CLI overrides
        data_dict = {k: v for k, v in data_section.items() if v is not None}
        
        # Merge with CLI overrides (CLI takes precedence for scene-specific settings)
        for key in base_config.keys():
            data_dict[key] = base_config[key]
        
        data_cfg = OmegaConf.create(data_dict)
            
    elif args.scene_path and os.path.exists(args.scene_path):
        # Try to infer dataset type from path structure if no config provided
        scene_name = os.path.basename(os.path.dirname(args.scene_path))
        
        logger.info(f"Using minimal configuration for: {args.scene_path}")
        data_cfg = OmegaConf.merge(base_config, {
            "dataset": "paralane/1cams",  # Default to paralane as it's commonly used in this project
        })
    else:
        # Minimal fallback config with required fields only
        logger.warning("No valid data path or configuration found")
        data_cfg = OmegaConf.merge(base_config, {
            "dataset": "paralane/1cams",  # Default to paralane as it's commonly used in this project
        })
    
    logger.info(f"Scene path: {args.scene_path}")
    
    try:
        # Initialize DrivingDataset with the target scene
        dataset = DrivingDataset(data_cfg=data_cfg)
        
        # Move lidar source to device for computation
        if hasattr(dataset, 'lidar_source') and dataset.lidar_source is not None:
            dataset.lidar_source.to(device)
            
        logger.info(f"Scene loaded successfully:")
        logger.info(f"  - Number of frames: {dataset.frame_num}")
        logger.info(f"  - Number of cameras: {dataset.num_cams}")
        
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {e}")
        raise
    
    # Step 1.2 & 1.3: Aggregate road point cloud using LiDAR and road masks
    try:
        road_pts, road_colors = aggregate_road_pointcloud(dataset, device)
        
        if len(road_pts) == 0:
            logger.error("No road points extracted! Cannot proceed with visualization.")
            return
            
    except Exception as e:
        logger.error(f"Error during point cloud aggregation: {e}")
        raise
    
    # Convert to numpy for processing and visualization
    pts_xyz = torch.cat([road_pts, road_colors], dim=1).cpu().numpy() if len(road_pts) > 0 else np.empty((0, 6))
    
    # Step 1.4: Log bird's eye view of point cloud
    try:
        log_birds_eye_view(pts_xyz[:, :3], pts_xyz[:, 3:], step1_dir)
        
        # Save a pruned/decimated version for efficient storage (especially useful for large point clouds)
        save_pruned_pointcloud_image(pts_xyz[:, :3], pts_xyz[:, 3:], step1_dir)
        
        # Log detailed statistics to console and file
        log_point_cloud_statistics(pts_xyz[:, :3])
        
    except Exception as e:
        logger.error(f"Error during visualization logging: {e}")
    
    # Step 1.5: Export point cloud to PLY format for external inspection
    try:
        export_point_cloud(road_pts.cpu().numpy(), road_colors.cpu().numpy() if len(road_colors) > 0 else None, step1_dir)
        
    except Exception as e:
        logger.error(f"Error during PLY export: {e}")
    
    # Final summary logging
    logger.info("=" * 60)
    logger.info("STEP 1 COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Output directory: {base_path}")
    logger.info(f"Point cloud files saved to: {step1_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Step 2: Create a 3D mesh from the aggregated point cloud")
    logger.info("  - Step 3: Generate image buffer for texture mapping (4096x4096)")
    logger.info("  - Step 4: Project RGB images onto the textured mesh")


if __name__ == "__main__":
    main()


