#!/usr/bin/env python3
"""
Road Mesh Training Script - Step 1 & 2 Implementation

This script implements Steps 1-4 of the road mesh training pipeline:
- Aggregate point cloud using road masks and lidar data (IMPLEMENTED HERE)
    - Use existing Driving_Dataset code to load data, masks, and images
    - Extract LiDAR points that project into road mask regions
    - Log bird's eye view visualization of the extracted points
- Create a 3d mesh from the aggregated point cloud (STEP 2 IMPLEMENTED HERE)
    - No overlap along Z axis by construction (heightmap-based approach)
    - Low complexity with reasonable amount of polys (~5cm grid resolution)
    - Log mesh as .pth file and PLY export for inspection
- Create an image buffer for the road mesh (STEP 3 IMPLEMENTED HERE)
    - Use x and y range of point cloud to determine dimensions
    - Resolution: 10 pixels per meter (user requirement)
    - Map pixels to mesh using scaling factor based on scene size
    - Project rays using rgb images and cam position onto the mesh (Step 4)
    - Only project road areas according to road mask
    - Fill image buffer using RGB data from images
    - Reproject the final image buffer back onto the mesh
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
from tqdm import tqdm
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
    parser.add_argument(
        "--num_projection_frames",
        type=int,
        default=None,
        help="Limit how many training images are used for Step 4 ray projection (default: all training images)",
    )
    
    return parser.parse_args()


def setup_output_directories(output_folder: str, run_name: str) -> Tuple[str, str]:
    """Create output directories and set up logging."""
    # Construct full output path
    base_path = os.path.join(output_folder, "road_mesh_training", run_name)
    
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
    
    IMPORTANT: Uses PRUNED lidar data - only includes points remaining after
    dataset.project_lidar_pts_on_images() removes out-of-view points during initialization.
    This ensures mesh creation uses valid, visible LiDAR points rather than all raw lidar data.
    
    Args:
        dataset: Initialized DrivingDataset instance with loaded data (already pruned)
        device: PyTorch device for computation
        
    Returns:
        Tuple of (road_pts, road_colors) tensors containing aggregated point cloud from pruned LiDAR source
    """
    logger.info("Starting road LiDAR point aggregation using PRUNED lidar data...")
    logger.info(f"Using dataset.lidar_source.pts_xyz with {len(dataset.lidar_source.pts_xyz)} points (pruned)")
    
    # Get all LiDAR indices that project into road mask regions across all frames/cameras
    # Note: This uses the pruned lidar source from self.lidar_source after project_lidar_pts_on_images()
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



def create_birds_eye_view(pts_xyz: np.ndarray, 
                          colors: np.ndarray = None,
                          cam_positions=None,
                          output_dir: str = None) -> str:
    """
    Create a bird's eye view visualization of the point cloud.
    
    This function generates three orthogonal views (X-Y, X-Z, Y-Z planes)
    to visualize the 3D structure of the road surface from different perspectives.
    Optionally overlays camera positions if provided.
    
    Args:
        pts_xyz: N x 3 numpy array of point coordinates
        colors: Optional N x 3 RGB color values (0-1 range)
        cam_positions: Optional M x 3 array of camera locations to overlay
        output_dir: Directory to save the visualization (optional, returns path if None)
    
    Returns:
        Path to saved PNG file or None if no points provided
    """
    if len(pts_xyz) == 0:
        logger.warning("Cannot create bird's eye view from empty point cloud")
        return None
    
    num_points = len(pts_xyz)
    cam_positions_arr = np.array(cam_positions) if cam_positions is not None else None
    
    # Create three orthogonal views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # X-Y plane (bird's eye view - top down)
    ax_xy = axes[0]
    scatter = ax_xy.scatter(pts_xyz[:, 0], pts_xyz[:, 1], c=range(num_points), 
                           cmap='viridis', s=1, alpha=0.5, label='Road Points')
    
    # Plot camera positions if available
    if cam_positions_arr is not None and len(cam_positions_arr) > 0:
        scatter_cams = ax_xy.scatter(cam_positions_arr[:, 0], cam_positions_arr[:, 1], 
                                     c='red', s=80, marker='X', linewidths=2,
                                     edgecolors='white', alpha=0.9, 
                                     label=f'Cameras ({len(cam_positions_arr)} positions)')
        ax_xy.legend(loc='upper right', fontsize=10)
    
    ax_xy.set_xlabel('X (meters)', fontsize=12)
    ax_xy.set_ylabel('Y (meters)', fontsize=12)
    title = f"Bird's Eye View - {num_points:,} Road Points"
    if cam_positions_arr is not None and len(cam_positions_arr) > 0:
        title += " + Camera Locations"
    ax_xy.set_title(title, fontsize=14, fontweight='bold')
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
    
    # Determine output path
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, "birds_eye_view.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved bird's eye view visualization to {viz_path}")
    else:
        # Return figure if no output directory specified
        return fig
    
    plt.close(fig)
    return viz_path


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


def create_camera_locations_plot(dataset, output_dir: str = None):
    """
    Create a dedicated visualization showing ONLY camera locations across all frames.
    
    This function extracts and plots ALL available camera positions from the dataset,
    providing a clear view of the data collection trajectory.
    
    Args:
        dataset: DrivingDataset instance with loaded scene
        output_dir: Directory to save the visualization (optional)
    """
    cam_positions = []
    cam_frame_ids = []
    cam_cam_ids = []
    
    try:
        total_train_indices = len(dataset.train_indices) if hasattr(dataset, 'train_indices') else 10
        logger.info(f"Extracting camera positions from {total_train_indices} frames...")
        
        for i in range(total_train_indices):
            img_index = int(i * len(dataset.train_indices) / max(total_train_indices, 1)) if total_train_indices > 0 else 0
            
            if hasattr(dataset, 'full_image_set') and dataset.full_image_set is not None:
                try:
                    image_infos, cam_infos = dataset.full_image_set.get_image(img_index, camera_downscale=1.0)
                    c2w = cam_infos.get('camera_to_world', None)
                    
                    if c2w is not None:
                        # Extract camera position from extrinsics matrix
                        if torch.is_tensor(c2w):
                            c2w_np = c2w.cpu().numpy()
                        else:
                            c2w_np = np.array(c2w)
                        
                        # Handle both single and multi-camera formats
                        if len(c2w_np.shape) == 3:  # Multi-camera frame
                            for j in range(min(5, c2w_np.shape[0])):  # Limit to first 5 cameras per frame
                                cam_positions.append(c2w_np[j][:3, 3])
                                cam_frame_ids.append(img_index)
                                cam_cam_ids.append(j)
                        elif len(c2w_np.shape) == 2:  # Single camera view
                            cam_positions.append(c2w_np[:3, 3])
                            cam_frame_ids.append(img_index)
                            cam_cam_ids.append(0)
                except Exception as e:
                    logger.debug(f"Could not extract camera position from frame {i}: {e}")
    
        if len(cam_positions) == 0:
            logger.warning("No camera positions could be extracted")
            return None
        
        cam_array = np.array(cam_positions)
        num_cams = len(cam_positions)
        
        # Create trajectory visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # X-Y plane (top-down view of camera path)
        ax_xy = axes[0]
        scatter_cams = ax_xy.scatter(cam_array[:, 0], cam_array[:, 1], 
                                     c=range(num_cams), cmap='Reds', s=50, alpha=0.7,
                                     edgecolors='black', linewidths=0.5)
        ax_xy.set_xlabel('X (meters)', fontsize=12)
        ax_xy.set_ylabel('Y (meters)', fontsize=12)
        ax_xy.set_title(f"Camera Trajectory - {num_cams} Positions", fontsize=14, fontweight='bold')
        ax_xy.grid(True, alpha=0.3)
        plt.colorbar(scatter_cams, ax=ax_xy, label='Frame Index', shrink=0.8)
        
        # X-Z plane (side view showing camera height variations)
        ax_xz = axes[1]
        scatter_heights = ax_xz.scatter(cam_array[:, 0], cam_array[:, 2],
                                        c=range(num_cams), cmap='Reds', s=50, alpha=0.7,
                                        edgecolors='black', linewidths=0.5)
        ax_xz.set_xlabel('X (meters)', fontsize=12)
        ax_xz.set_ylabel('Z height (meters)', fontsize=12)
        ax_xz.set_title("Camera Height Profile - X-Z View", fontsize=14, fontweight='bold')
        ax_xz.grid(True, alpha=0.3)
        plt.colorbar(scatter_heights, ax=ax_xz, label='Frame Index', shrink=0.8)
        
        # Y-Z plane (front view showing camera height variations along path)
        ax_yz = axes[2]
        scatter_front = ax_yz.scatter(cam_array[:, 1], cam_array[:, 2],
                                      c=range(num_cams), cmap='Reds', s=50, alpha=0.7,
                                      edgecolors='black', linewidths=0.5)
        ax_yz.set_xlabel('Y (meters)', fontsize=12)
        ax_yz.set_ylabel('Z height (meters)', fontsize=12)
        ax_yz.set_title("Camera Height Profile - Y-Z View", fontsize=14, fontweight='bold')
        ax_yz.grid(True, alpha=0.3)
        plt.colorbar(scatter_front, ax=ax_yz, label='Frame Index', shrink=0.8)
        
        # Log camera statistics
        logger.info(f"\nCamera Position Statistics:")
        logger.info(f"  Total positions: {num_cams}")
        logger.info(f"  X range: [{cam_array[:, 0].min():.2f}, {cam_array[:, 0].max():.2f}] m")
        logger.info(f"  Y range: [{cam_array[:, 1].min():.2f}, {cam_array[:, 1].max():.2f}] m")
        logger.info(f"  Z height (avg): {cam_array[:, 2].mean():.3f} ± {cam_array[:, 2].std():.3f} m")
        
        plt.tight_layout()
        
        # Save visualization
        if output_dir:
            viz_path = os.path.join(output_dir, "camera_locations.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved camera locations plot to {viz_path}")
        
        return cam_array
    
    except Exception as e:
        logger.error(f"Error creating camera locations plot: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_mesh_from_pointcloud(pts_xyz: np.ndarray, 
                                colors: np.ndarray = None):
    """
    Create a 3D mesh from the point cloud using a simpler heightmap-based approach.
    
    This function creates a grid-based mesh where each cell in the X-Y plane maps to one Z value:
    - Uses x and y ranges of the point cloud to determine scene dimensions
    - Cell size is fixed at 0.2 meters per cell
    - Creates a 2D heightmap array sized (x_range/cell_size) × (y_range/cell_size)
    - Iterates through points, marking bins with z values while keeping minimum heights
    
    Args:
        pts_xyz: N x 3 numpy array of point coordinates (x, y, z) in meters
        colors: Optional N x 3 numpy array of RGB colors
        
    Returns:
        Tuple of (vertices, faces, vertex_colors):
            - vertices: M × 3 tensor of mesh vertex positions
            - faces: F × 3 tensor of face indices  
            - vertex_colors: M × 3 tensor of RGB colors for each vertex
    """
    
    if len(pts_xyz) == 0:
        logger.warning("Cannot create mesh from empty point cloud")
        return torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 3)
    
    # Convert to numpy for processing
    points = pts_xyz if isinstance(pts_xyz, np.ndarray) else pts_xyz.cpu().numpy()
    colors_arr = colors.copy() if colors is not None and len(colors) > 0 else None
    
    logger.info(f"Creating mesh from {len(points):,} point cloud samples...")
    
    # Determine scene bounds from point cloud (x range and y range)
    x_min, y_min, z_min = points[:, 0].min(), points[:, 1].min(), points[:, 2].min()
    x_max, y_max, _ = points[:, 0].max(), points[:, 1].max(), points[:, 2].max()
    
    scene_width = max(x_max - x_min, 0.1)
    scene_height = max(y_max - y_min, 0.1)
    
    # Fixed cell size of 0.2 meters per cell (as specified by user requirement)
    cell_size = 1
    
    # Calculate grid dimensions: number of cells in x and y directions
    num_cells_x = int(np.ceil(scene_width / cell_size)) + 1
    num_cells_y = int(np.ceil(scene_height / cell_size)) + 1
    
    logger.info(f"Grid resolution: {num_cells_x} × {num_cells_y} cells (cell size: {cell_size}m)")
    
    # Create heightmap grid initialized with infinity (user requirement)
    # Shape is num_cells_x × num_cells_y, storing minimum z values per cell
    height_map = np.full((num_cells_x, num_cells_y), np.inf, dtype=np.float64)
    
    if colors_arr is not None and len(colors_arr) == len(points):
        color_map = [np.full((num_cells_x, num_cells_y), -1.0, dtype=np.int32) for _ in range(3)]
    else:
        color_map = None
    
    # Iterate through points to mark bins with z values (user requirement)
    # Keep lower z values when multiple points fall into same cell
    logger.info("Iterating through points and marking heightmap cells...")
    
    for i, point in enumerate(points):
        x_idx = int((point[0] - x_min) / scene_width * num_cells_x)
        y_idx = int((point[1] - y_min) / scene_height * num_cells_y)
        
        # Clip to valid range [0, num_cells-1]
        x_idx = np.clip(x_idx, 0, num_cells_x - 1)
        y_idx = np.clip(y_idx, 0, num_cells_y - 1)
        
        z_val = point[2]
        
        # Update heightmap if this is a lower (better) z value for this cell
        if z_val < height_map[x_idx, y_idx]:
            height_map[x_idx, y_idx] = z_val
            
            # Store color information if available
            if colors_arr is not None:
                for c in range(3):
                    color_map[c][x_idx, y_idx] = int(colors_arr[i, c])
    
    # Count valid cells (those with actual points)
    num_valid_cells = np.sum(~np.isinf(height_map))
    logger.info(f"Created {num_valid_cells:,} valid grid cells from point cloud")
    
    if num_valid_cells < 4:
        logger.warning("Insufficient valid cells to create a meaningful mesh")
        return torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 3)
    
    # Build lookup table mapping (x_idx, y_idx) -> vertex_index for face building
    height_map_dict = {}
    cell_z_values = []
    cell_colors_list = [] if color_map is not None else None
    
    # First pass: identify all valid cells that will become vertices
    # Keep boundary cells too; dropping them removes the outer ring of triangles.
    valid_cells = []
    for x_idx in range(num_cells_x):
        for y_idx in range(num_cells_y):
            z_val = height_map[x_idx, y_idx]
            if not np.isinf(z_val):
                valid_cells.append((x_idx, y_idx))
    
    # Create mapping from grid position to vertex index
    for i, (x_idx, y_idx) in enumerate(valid_cells):
        height_map_dict[(x_idx, y_idx)] = len(cell_z_values)
        cell_z_values.append(float(height_map[x_idx, y_idx]))
        
        if color_map is not None:
            mean_color = np.array([color_map[c][x_idx, y_idx] for c in range(3)], dtype=np.float32) / 255.0
            cell_colors_list.append(mean_color)
    
    if len(cell_z_values) == 0:
        logger.warning("No valid cells after boundary filtering")
        return torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 3)
    
    # Build triangular faces connecting adjacent vertices (same as original implementation)
    final_faces = []
    
    for x_idx in range(num_cells_x - 1):
        for y_idx in range(num_cells_y - 1):
            corner_indices = [
                height_map_dict.get((x_idx, y_idx)),
                height_map_dict.get((x_idx + 1, y_idx)),
                height_map_dict.get((x_idx, y_idx + 1)),
                height_map_dict.get((x_idx + 1, y_idx + 1))
            ]
            
            # Skip if any corner is missing (would create holes)
            if None in corner_indices:
                continue
            
            # Keep both triangles wound consistently so face culling doesn't drop half the quad.
            final_faces.append(
                [corner_indices[0], corner_indices[1], corner_indices[3]]
            )
            final_faces.append(
                [corner_indices[0], corner_indices[3], corner_indices[2]]
            )
    
    logger.info(f"Created {len(final_faces):,} triangular faces")
    
    if len(final_faces) == 0:
        logger.warning("No valid triangles could be created from the grid cells")
        return torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 3)
    
# Build final vertices with X-Y plane coordinates in original world space (preserving LiDAR coordinate system)
    final_vertices = []
    row_colors = cell_colors_list if color_map is not None else None

    for x_idx in range(num_cells_x):
        for y_idx in range(num_cells_y):
            vertex_key = (x_idx, y_idx)
            
            # Skip cells with no data or boundary cells
            if vertex_key not in height_map_dict:
                continue
            
            z_height = cell_z_values[height_map_dict[vertex_key]]
            
            # Map grid index to world coordinates preserving original LiDAR coordinate system (NOT centered at origin).
            # Use (num_cells - 1) so the generated mesh spans the original min/max bounds.
            world_x = x_min + (x_idx / max(num_cells_x - 1, 1)) * scene_width
            world_y = y_min + (y_idx / max(num_cells_y - 1, 1)) * scene_height
            
            final_vertices.append([world_x, world_y, z_height])
    
    logger.info(f"Created {len(final_vertices):,} mesh vertices in original world space")
    
    # Convert to PyTorch tensors for compatibility with the rest of the pipeline
    vertices_tensor = torch.tensor(final_vertices).float() if len(final_vertices) > 0 else torch.empty(0, 3)
    faces_tensor = torch.tensor(final_faces).long() if len(final_faces) > 0 else torch.empty(0, 3)
    
    # Convert colors to tensor if available
    vertex_colors_final = np.array(row_colors).astype(np.float32) / 255.0 if row_colors is not None and len(row_colors) > 0 else None
    
    logger.info(f"Mesh created successfully: {len(final_vertices):,} vertices, {len(final_faces):,} faces")
    
    return vertices_tensor, faces_tensor, torch.tensor(vertex_colors_final) if vertex_colors_final is not None else torch.empty(0, 3)


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


def create_and_log_mesh(pts_xyz: np.ndarray, 
                        colors: Union[np.ndarray, None] = None,
                        output_dir: str = ""):
    """
    Create a 3D mesh from the aggregated point cloud and log it.
    
    This implements Step 2 of the pipeline - creating a low-complexity mesh with no Z-axis overlap.
    
    Args:
        pts_xyz: N x 3 numpy array of point coordinates (x, y, z) in meters
        colors: Optional N x 3 numpy array of RGB colors (0-1 or 0-255)
        output_dir: Directory to save mesh files and visualizations
        
    Returns:
        Tuple of (vertices_tensor, faces_tensor, vertex_colors): Mesh components as tensors
    """
    
    # Create the mesh using heightmap-based approach with no Z-axis overlap
    vertices, faces, vertex_colors = create_mesh_from_pointcloud(pts_xyz, colors)
    
    if len(vertices) == 0 or len(faces) == 0:
        logger.warning("Mesh creation failed - insufficient valid points")
        return None, None, None
    
    # Save mesh as .pth file for later use in training pipeline
    mesh_path = os.path.join(output_dir, "road_mesh.pth")
    
    torch.save({
        'vertices': vertices.cpu(),
        'faces': faces.cpu(),
        'vertex_colors': vertex_colors.cpu() if len(vertex_colors) > 0 else None,
        'metadata': {
            'num_vertices': len(vertices),
            'num_faces': len(faces),
            'mesh_type': 'heightmap_road_mesh'
        }
    }, mesh_path)
    
    logger.info(f"Saved road mesh to {mesh_path}")
    
    # Log detailed mesh statistics
    x_min, y_min, z_min = vertices[:, 0].min(), vertices[:, 1].min(), vertices[:, 2].min()
    x_max, y_max, z_max = vertices[:, 0].max(), vertices[:, 1].max(), vertices[:, 2].max()
    
    logger.info("=" * 60)
    logger.info("MESH STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total vertices: {len(vertices):,}")
    logger.info(f"Total faces (triangles): {len(faces):,}")
    logger.info("")
    logger.info("Mesh Bounding Box:")
    logger.info(f"  X-axis: [{x_min:.4f}, {x_max:.4f}] - span: {(x_max-x_min):.2f} m")
    logger.info(f"  Y-axis: [{y_min:.4f}, {y_max:.4f}] - span: {(y_max-y_min):.2f} m")
    logger.info(f"  Z-axis (height): [{z_min:.4f}, {z_max:.4f}] - span: {(z_max-z_min):.4f} m")
    
    # Verify no Z-axis overlap by checking unique heights per grid position
    z_values = vertices[:, 2].cpu().numpy() if isinstance(vertices, torch.Tensor) else np.array(z_values)
    height_variance = float(np.var(z_values))
    logger.info("")
    logger.info("Mesh Quality Metrics:")
    logger.info(f"  Z-height variance: {height_variance:.6f} m²")
    
    # Calculate triangle count per square meter (mesh complexity density)
    xy_area = max(x_max - x_min, 0.1) * max(y_max - y_min, 0.1)
    triangles_per_m2 = len(faces) / xy_area
    
    logger.info("")
    logger.info("Mesh Complexity:")
    logger.info(f"  X-Y coverage area: {xy_area:.2f} m²")
    logger.info(f"  Triangle density: {triangles_per_m2:.1f} triangles/m² (low complexity)")
    
    # Create visualization of the mesh from multiple viewpoints
    try:
        visualize_mesh(vertices, faces, vertex_colors if len(vertex_colors) > 0 else None, output_dir)
        
        logger.info("")
        logger.info("Mesh visualizations saved to:")
        for filename in ['mesh_top_view.png', 'mesh_side_view.png', 'mesh_3d_view.png']:
            viz_path = os.path.join(output_dir, filename)
            if os.path.exists(viz_path):
                logger.info(f"  - {filename}")
    except Exception as e:
        logger.error(f"Error during mesh visualization: {e}")
    
    return vertices, faces, vertex_colors


def visualize_mesh(vertices: torch.Tensor, 
                   faces: torch.Tensor, 
                   colors: Union[torch.Tensor, None] = None,
                   output_dir: str = ""):
    """
    Create and save visualizations of the mesh from multiple viewpoints.
    
    Args:
        vertices: M x 3 tensor of vertex positions in meters
        faces: F x 3 tensor of face indices (triangles)
        colors: Optional M x 3 tensor of RGB colors for each vertex
        output_dir: Directory to save visualizations
        
    Returns:
        List of saved visualization file paths
    """
    
    if len(vertices) == 0 or len(faces) == 0:
        logger.warning("Cannot visualize empty mesh")
        return []
    
    # Convert tensors to numpy for matplotlib plotting
    verts = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else np.array(vertices)
    faces_np = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else np.array(faces)
    colors_arr = colors.cpu().numpy() if (colors is not None and len(colors) > 0) else None
    
    # Create multi-view visualization using matplotlib's plot_surface for mesh-like appearance
    fig = plt.figure(figsize=(18, 6))
    
    # View 1: Top-down view (bird's eye - looking along Z-axis)
    ax_top = fig.add_subplot(131, projection='3d')
    if colors_arr is not None and len(colors_arr) > 0:
        mesh_plot = ax_top.plot_trisurf(verts[:, 0], verts[:, 1], 
                                        verts[:, 2], triangles=faces_np, cmap='viridis', alpha=0.9)
    else:
        # Color by height if no color data available
        heights = (verts[:, 2] - verts[:, 2].min()) / max(verts[:, 2].max() - verts[:, 2].min(), 1e-6)
        mesh_plot = ax_top.plot_trisurf(verts[:, 0], verts[:, 1], 
                                        verts[:, 2], triangles=faces_np, cmap='viridis', alpha=0.9)
    
    x_min, y_min, z_min = verts.min(axis=0)
    x_max, y_max, z_max = verts.max(axis=0)
    
    ax_top.set_xlabel('X (meters)', fontsize=12)
    ax_top.set_ylabel('Y (meters)', fontsize=12)
    ax_top.set_zlabel('Z height (meters)', fontsize=12)
    ax_top.set_title("Top View - Bird's Eye", fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio for proper visualization
    max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
    
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    
    ax_top.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_top.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_top.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # View 2: Side view (looking along Y-axis)
    ax_side = fig.add_subplot(132, projection='3d')
    if colors_arr is not None and len(colors_arr) > 0:
        mesh_plot = ax_side.plot_trisurf(verts[:, 0], verts[:, 2], 
                                         verts[:, 1], triangles=faces_np, cmap='viridis', alpha=0.9)
    
    ax_side.set_xlabel('X (meters)', fontsize=12)
    ax_side.set_ylabel('Z height (meters)', fontsize=12)
    ax_side.set_zlabel('Y depth (meters)', fontsize=12)
    ax_side.set_title("Side View - X-Z Plane", fontsize=14, fontweight='bold')
    
    # Set aspect ratio for side view
    max_range_xz = np.array([x_max-x_min, z_max-z_min]).max() / 2.0
    
    mid_z_s = (z_max + z_min) * 0.5
    ax_side.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_side.set_ylim(z_min - max_range_xz/10, z_max + max_range_xz/10)
    
    # View 3: Isometric view (for better depth perception)
    ax_iso = fig.add_subplot(133, projection='3d')
    if colors_arr is not None and len(colors_arr) > 0:
        mesh_plot = ax_iso.plot_trisurf(verts[:, 0], verts[:, 2], 
                                        verts[:, 1], triangles=faces_np, cmap='viridis', alpha=0.9)
    
    # Set elevation for better isometric view (elevation angle and azimuth)
    ax_iso.view_init(elev=30, azim=-60)
    
    ax_iso.set_xlabel('X (meters)', fontsize=12)
    ax_iso.set_ylabel('Z height (meters)', fontsize=12)
    ax_iso.set_zlabel('Y depth (meters)', fontsize=12)
    ax_iso.set_title("Isometric View", fontsize=14, fontweight='bold')
    
    # Set aspect ratio for isometric view
    max_range_iso = np.array([x_max-x_min, y_max-y_min]).max() / 2.0
    
    mid_y_i = (y_max + y_min) * 0.5
    ax_iso.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_iso.set_ylim(z_min - max_range/10, z_max + max_range/10)
    
    plt.tight_layout()
    
    # Save visualizations with high DPI for publication quality
    viz_files = []
    
    try:
        top_view_path = os.path.join(output_dir, "mesh_top_view.png") if output_dir else None
        side_view_path = os.path.join(output_dir, "mesh_side_view.png") if output_dir else None
        iso_view_path = os.path.join(output_dir, "mesh_3d_view.png") if output_dir else None
        
        # Save individual views with high resolution (150 DPI)
        plt.savefig(top_view_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved top view visualization to {top_view_path}")
        
        plt.savefig(side_view_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved side view visualization to {side_view_path}")
        
        plt.savefig(iso_view_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved isometric view visualization to {iso_view_path}")
        
        viz_files = [top_view_path, side_view_path, iso_view_path]
    except Exception as e:
        logger.error(f"Error saving mesh visualizations: {e}")
    
    plt.close(fig)
    
    return viz_files


def export_mesh_to_ply(vertices: torch.Tensor, 
                       faces: torch.Tensor, 
                       colors: Union[torch.Tensor, None] = None,
                       output_dir: str = "",
                       filename: str = "road_mesh.ply"):
    """Export mesh to PLY format for external inspection."""
    import struct
    
    if len(vertices) == 0 or len(faces) == 0:
        logger.warning("Cannot export empty mesh")
        return
    
    ply_path = os.path.join(output_dir, filename) if output_dir else None
    
    # Convert to numpy arrays for PLY export
    verts_np = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else np.array(vertices)
    faces_np = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else np.array(faces)
    
    colors_arr = colors.cpu().numpy() if (colors is not None and len(colors) > 0) else None
    
    # Generate vertex colors for PLY export
    if colors_arr is not None:
        rgb_colors = np.clip(colors_arr * 255, 0, 255).astype(np.uint8)
    else:
        # Grayscale based on height relative to scene bounds
        z_min, z_max = verts_np[:, 2].min(), verts_np[:, 2].max()
        normalized_z = (verts_np[:, 2] - z_min) / max(z_max - z_min, 1e-6)
        rgb_colors = np.stack([normalized_z] * 3, axis=1).astype(np.float32) * 255
    
    # Write PLY file in binary format for efficiency
    with open(ply_path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(verts_np)}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(f"element face {len(faces_np)}\n".encode())
        f.write(b"property list uchar int vertex_indices\n")
        f.write(b"end_header\n")
        
        # Write vertices with RGB colors
        for i in range(len(verts_np)):
            x, y, z = verts_np[i]
            r, g, b = rgb_colors[i].astype(np.uint8) if len(rgb_colors) > 0 else (128, 128, 128)
            
            # Write as binary floats for XYZ and unsigned chars for RGB
            f.write(struct.pack('fff', float(x), float(y), float(z)))
            f.write(bytes([r, g, b]))
        
        # Write faces (triangles only - each face has 3 vertex indices + count)
        for i in range(len(faces_np)):
            v0, v1, v2 = faces_np[i]
            
            # Triangle: write number of vertices (3) followed by the three vertex indices
            f.write(bytes([3]))
            f.write(struct.pack('iii', int(v0), int(v1), int(v2)))
    
    logger.info(f"Exported mesh to PLY format at {ply_path}")


def create_image_buffer(vertices: torch.Tensor, 
                        faces: torch.Tensor,
                        colors: Union[torch.Tensor, None] = None) -> Tuple[np.ndarray, dict]:
    """
    Create an image buffer for the road mesh using point cloud dimensions as reference.
    
    This implements Step 3 of the pipeline - creating a texture-ready image buffer where:
    - Resolution is determined by x and y ranges with scaling factor (10 pixels per meter)
    - Each pixel maps to a corresponding location on the mesh surface
    - Buffer stores RGB color values for each pixel
    
    Args:
        vertices: M × 3 tensor of vertex positions in meters
        faces: F × 3 tensor of face indices (triangles)  
        colors: Optional M × 3 tensor of RGB colors for each vertex
        
    Returns:
        Tuple of (image_buffer, metadata):
            - image_buffer: H x W x 3 numpy array with float values in [0, 1] range
            - metadata: Dictionary containing buffer dimensions and scaling information
    """
    
    if len(vertices) == 0 or len(faces) == 0:
        logger.warning("Cannot create image buffer from empty mesh")
        return np.zeros((100, 100, 3), dtype=np.float32), {}
    
    # Convert to numpy for processing
    verts = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else np.array(vertices)
    
    logger.info("Creating image buffer from mesh...")
    
    # Determine scene bounds in X and Y directions (user requirement: use x and y range as reference)
    x_min, y_min, z_min = verts[:, 0].min(), verts[:, 1].min(), verts[:, 2].min()
    x_max, y_max, _ = verts[:, 0].max(), verts[:, 1].max(), verts[:, 2].max()
    
    scene_width = max(x_max - x_min, 0.1)  # Ensure minimum width to avoid division by zero
    scene_height = max(y_max - y_min, 0.1)  # Ensure minimum height
    
    logger.info(f"Scene dimensions: {scene_width:.2f}m × {scene_height:.2f}m")
    
    pixels_per_meter = 40
    
    # Calculate buffer dimensions based on scene size and pixel density (user specification)
    width_pixels = int(np.ceil(scene_width * pixels_per_meter))
    height_pixels = int(np.ceil(scene_height * pixels_per_meter))
    
    logger.info(f"Image buffer resolution: {width_pixels} × {height_pixels} ({pixels_per_meter}px/m)")
    
    # Initialize image buffer with zeros (background) - RGB channels in [0, 1] range
    image_buffer = np.zeros((height_pixels, width_pixels, 3), dtype=np.float32)
    
    # Store metadata about the buffer for later use
    metadata = {
        'width': width_pixels,
        'height': height_pixels,
        'pixels_per_meter': pixels_per_meter,
        'x_range': (float(x_min), float(x_max)),
        'y_range': (float(y_min), float(y_max)),
        'scene_width_meters': scene_width,
        'scene_height_meters': scene_height,
    }
    
    # If we have vertex colors, map them to the image buffer via rasterization-like process
    if colors is not None and len(colors) > 0:
        logger.info(f"Mapping {len(verts)} vertices with color data to image buffer...")
        
        # Convert faces to numpy array for processing
        faces_np = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else np.array(faces)
        colors_arr = colors.cpu().numpy() if isinstance(colors, torch.Tensor) else np.array(colors)
        
        # For each triangle face, rasterize it onto the image buffer
        for i in range(len(faces_np)):
            v0_idx, v1_idx, v2_idx = faces_np[i]
            
            # Get vertex positions and colors for this triangle
            v0 = verts[v0_idx]
            v1 = verts[v1_idx]
            v2 = verts[v2_idx]
            
            tri_colors = np.array([colors_arr[v0_idx], colors_arr[v1_idx], colors_arr[v2_idx]])
            
            # Bounding box of triangle in pixel coordinates (user requirement: map pixels to mesh)
            x_coords = [v0[0], v1[0], v2[0]]
            y_coords = [v0[1], v1[1], v2[1]]
            
            min_x, max_x = int(np.floor((min(x_coords) - x_min) * pixels_per_meter)), \
                          int(np.ceil((max(x_coords) - x_min) * pixels_per_meter))
            min_y, max_y = int(np.floor((min(y_coords) - y_min) * pixels_per_meter)), \
                          int(np.ceil((max(y_coords) - y_min) * pixels_per_meter))
            
            # Clip to image bounds
            min_x = np.clip(min_x, 0, width_pixels - 1)
            max_x = np.clip(max_x, 0, width_pixels - 1)
            min_y = np.clip(min_y, 0, height_pixels - 1)
            max_y = np.clip(max_y, 0, height_pixels - 1)
            
            # Rasterize triangle pixels (simple scanline approach for each pixel in bounding box)
            for py in range(min_y, max_y):
                for px in range(min_x, max_x):
                    # Convert pixel coordinates back to world coordinates
                    wx = x_min + px / pixels_per_meter
                    wy = y_min + py / pixels_per_meter
                    
                    # Check if this point is inside the triangle using barycentric coordinates
                    # Triangle vertices: v0, v1, v2 (all with same Z for heightmap mesh)
                    
                    # Vector from v0 to test point
                    d0 = np.array([wx - v0[0], wy - v0[1]])
                    d1 = np.array([v1[0] - v0[0], v1[1] - v0[1]])
                    d2 = np.array([v2[0] - v0[0], v2[1] - v0[1]])
                    
                    # Cross products for barycentric coordinate calculation (2D)
                    cp01 = d0[0]*d1[1] - d0[1]*d1[0]  # z-component of cross product
                    cp02 = d0[0]*d2[1] - d0[1]*d2[0]
                    
                    if (cp01 >= 0 and cp02 >= 0) or \
                       (cp01 <= 0 and cp02 <= 0):
                        # Point is inside triangle, interpolate color using barycentric coordinates
                        
                        # Calculate actual barycentric weights for smooth interpolation
                        denom = d1[0]*d2[1] - d1[1]*d2[0]
                        if abs(denom) > 1e-6:
                            w1 = (d2[1]*(v0[0]-wx) + d2[0]*(wy-v0[1])) / denom
                            w2 = (d1[0]*(vy:=w1*(v1[1]-v0[1])/(v1[0]-v0[0])+v0[1]-wy) - 
                                  d1[1]*(wx-v0[0])) / denom if abs(v1[0]-v0[0]) > 1e-6 else 0
                            w0 = 1.0 - w1 - max(0, min(1, w2))
                            
                            # Clamp weights to valid range and interpolate color
                            w1, w2 = np.clip([w1, w2], 0, 1)
                            w0 = 1.0 - w1 - w2
                            w0 = max(0, min(1, w0))
                            
                            interpolated_color = (w0 * tri_colors[0] + 
                                                w1 * tri_colors[1] + 
                                                w2 * tri_colors[2])
                            
                            # Store color in buffer if brighter than existing value (for overlapping triangles)
                            current_color = image_buffer[py, px]
                            if np.sum(interpolated_color - current_color) > 0:
                                image_buffer[py, px] = interpolated_color
    
    logger.info(f"Image buffer created with {np.sum(image_buffer > 0):,} non-background pixels")
    
    return image_buffer, metadata


def log_image_buffer_visualization(buffer: np.ndarray, 
                                   metadata: dict, 
                                   output_dir: str) -> None:
    """
    Create and save visualization of the image buffer.
    
    Args:
        buffer: H x W x 3 numpy array with float values in [0, 1] range
        metadata: Dictionary containing buffer dimensions and scaling information
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved visualization file
    """
    
    if len(buffer) == 0 or len(metadata) == 0:
        logger.warning("Cannot visualize empty image buffer")
        return
    
    width = metadata.get('width', buffer.shape[1])
    height = metadata.get('height', buffer.shape[0])
    pixels_per_meter = metadata.get('pixels_per_meter', 10)
    
    # Create visualization with multiple views
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # View 1: Full image buffer (top-down view of texture map)
    ax_full = axes[0, 0]
    im_full = ax_full.imshow(buffer, extent=[-width/2, width/2, -height/2, height/2], origin='lower')
    ax_full.set_xlabel('Width (pixels)', fontsize=12)
    ax_full.set_ylabel('Height (pixels)', fontsize=12)
    ax_full.set_title(f"Image Buffer ({width}×{height})", fontsize=14, fontweight='bold')
    plt.colorbar(im_full, ax=ax_full, label='RGB Intensity', shrink=0.8)
    
    # View 2: Zoomed center region (first quadrant for clarity)
    zoom_x = min(width // 4, max(50, width // 10))
    zoom_y = min(height // 4, max(50, height // 10))
    ax_zoom = axes[0, 1]
    im_zoom = ax_zoom.imshow(buffer[:zoom_y, :zoom_x], origin='lower', 
                            extent=[-width/2, -width/2+zoom_x, -height/2, -height/2+zoom_y])
    ax_zoom.set_xlabel('Width (pixels)', fontsize=12)
    ax_zoom.set_ylabel('Height (pixels)', fontsize=12)
    ax_zoom.set_title(f"Center Region ({zoom_x}×{zoom_y})", fontsize=14, fontweight='bold')
    
    # View 3: RGB channel breakdown - Red channel
    ax_r = axes[0, 2]
    im_r = ax_r.imshow(buffer[:, :, 0], cmap='Reds', origin='lower')
    ax_r.set_xlabel('Width (pixels)', fontsize=12)
    ax_r.set_ylabel('Height (pixels)', fontsize=12)
    ax_r.set_title("Red Channel", fontsize=14, fontweight='bold')
    
    # View 4: RGB channel breakdown - Green channel  
    ax_g = axes[1, 0]
    im_g = ax_g.imshow(buffer[:, :, 1], cmap='Greens', origin='lower')
    ax_g.set_xlabel('Width (pixels)', fontsize=12)
    ax_g.set_ylabel('Height (pixels)', fontsize=12)
    ax_g.set_title("Green Channel", fontsize=14, fontweight='bold')
    
    # View 5: RGB channel breakdown - Blue channel
    ax_b = axes[1, 1]
    im_b = ax_b.imshow(buffer[:, :, 2], cmap='Blues', origin='lower')
    ax_b.set_xlabel('Width (pixels)', fontsize=12)
    ax_b.set_ylabel('Height (pixels)', fontsize=12)
    ax_b.set_title("Blue Channel", fontsize=14, fontweight='bold')
    
    # View 6: Non-background pixel density heatmap
    non_bg_mask = np.any(buffer > 0.01, axis=-1).astype(float)
    ax_density = axes[1, 2]
    im_density = ax_density.imshow(non_bg_mask, cmap='viridis', origin='lower')
    ax_density.set_xlabel('Width (pixels)', fontsize=12)
    ax_density.set_ylabel('Height (pixels)', fontsize=12)
    ax_density.set_title(f"Non-Background Pixels ({np.sum(non_bg_mask):,})", 
                        fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization with high DPI for publication quality
    viz_path = os.path.join(output_dir, "image_buffer_visualization.png") if output_dir else None
    
    try:
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved image buffer visualization to {viz_path}")
        
        # Also save a lower resolution version for quick viewing
        viz_path_lowres = os.path.join(output_dir, "image_buffer_preview.png") if output_dir else None
        plt.savefig(viz_path_lowres, dpi=72, bbox_inches='tight')
        logger.info(f"Saved preview visualization to {viz_path_lowres}")
    except Exception as e:
        logger.error(f"Error saving image buffer visualization: {e}")
    
    plt.close(fig)


def save_image_buffer_as_png(buffer: np.ndarray, 
                             metadata: dict, 
                             output_path: str) -> None:
    """
    Save image buffer as a PNG file for external inspection.
    
    Args:
        buffer: H x W x 3 numpy array with float values in [0, 1] range
        metadata: Dictionary containing buffer dimensions and scaling information  
        output_path: Path to save the PNG file
        
    Returns:
        None (saves file directly)
    """
    
    # Convert from normalized [0, 1] to uint8 [0-255] for PNG format
    buffer_uint8 = np.clip(buffer * 255.0, 0, 255).astype(np.uint8)
    
    try:
        plt.figure(figsize=(buffer.shape[1]/72, buffer.shape[0]/72), dpi=72)
        plt.imshow(buffer_uint8)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        logger.info(f"Saved image buffer as PNG to {output_path}")
    except Exception as e:
        # Fallback using PIL if matplotlib fails
        try:
            from PIL import Image
            img = Image.fromarray(buffer_uint8, mode='RGB')
            img.save(output_path)
            logger.info(f"Saved image buffer as PNG (PIL fallback) to {output_path}")
        except Exception as pil_error:
            logger.error(f"Failed to save image buffer as PNG: {e}; PIL also failed: {pil_error}")


def save_texture_buffer_as_png(buffer: np.ndarray, output_path: str) -> None:
    """Save a texture image with the vertical axis flipped for UV mapping."""
    buffer_uint8 = np.clip(np.flipud(buffer) * 255.0, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(buffer_uint8, mode="RGB").save(output_path)
        logger.info(f"Saved texture image to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save texture image: {e}")


def export_textured_mesh_obj(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    image_buffer: np.ndarray,
    buffer_metadata: dict,
    output_dir: str = "",
    filename: str = "road_textured_mesh.obj",
    texture_filename: str = "road_textured_mesh_texture.png",
):
    """Export a true textured OBJ/MTL mesh with per-vertex UVs and a texture image."""
    if len(vertices) == 0 or len(faces) == 0 or len(image_buffer) == 0 or len(buffer_metadata) == 0:
        logger.warning("Cannot export textured mesh - missing mesh or buffer data")
        return None
    if not output_dir:
        logger.warning("Cannot export textured mesh - output_dir is required")
        return None

    verts_np = vertices.detach().cpu().numpy() if torch.is_tensor(vertices) else np.asarray(vertices)
    faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)

    x_min, x_max = buffer_metadata["x_range"]
    y_min, y_max = buffer_metadata["y_range"]
    width = max(int(buffer_metadata.get("width", image_buffer.shape[1])), 1)
    height = max(int(buffer_metadata.get("height", image_buffer.shape[0])), 1)

    x_span = max(float(x_max) - float(x_min), 1e-8)
    y_span = max(float(y_max) - float(y_min), 1e-8)

    # Match the saved texture image: y_min is at the bottom after vertical flip.
    u = np.clip((verts_np[:, 0] - float(x_min)) / x_span, 0.0, 1.0)
    v = np.clip((verts_np[:, 1] - float(y_min)) / y_span, 0.0, 1.0)
    uv_coords = np.stack([u, v], axis=1).astype(np.float32)

    obj_path = os.path.join(output_dir, filename)
    mtl_filename = os.path.splitext(filename)[0] + ".mtl"
    mtl_path = os.path.join(output_dir, mtl_filename)
    texture_path = os.path.join(output_dir, texture_filename)

    save_texture_buffer_as_png(image_buffer, texture_path)

    with open(mtl_path, "w") as f:
        f.write("# StreetPed textured mesh material\n")
        f.write("newmtl road_texture\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("d 1.0\n")
        f.write("illum 1\n")
        f.write(f"map_Kd {texture_filename}\n")

    with open(obj_path, "w") as f:
        f.write("# Road mesh exported from StreetPed training pipeline\n")
        f.write(f"# Vertices: {len(verts_np)}, Faces: {len(faces_np)}\n")
        f.write(f"mtllib {mtl_filename}\n")
        f.write("usemtl road_texture\n")
        f.write("# Units in meters, coordinate system: X-right, Y-forward, Z-up\n\n")

        for x, y, z in verts_np:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for uu, vv in uv_coords:
            f.write(f"vt {uu:.6f} {vv:.6f}\n")
        for v0, v1, v2 in faces_np:
            v0i, v1i, v2i = int(v0) + 1, int(v1) + 1, int(v2) + 1
            f.write(f"f {v0i}/{v0i} {v1i}/{v1i} {v2i}/{v2i}\n")

    logger.info(f"Exported textured mesh OBJ to {obj_path}")
    logger.info(f"Exported textured mesh MTL to {mtl_path}")
    return obj_path


def reproject_image_buffer_onto_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    image_buffer: np.ndarray,
    buffer_metadata: dict,
    output_dir: str = "",
):
    """
    Reproject the final image buffer back onto the mesh as per-vertex colors.

    The mesh is a heightmap, so each vertex can be sampled directly from the
    buffer using its X/Y world position.
    """
    if len(vertices) == 0 or len(faces) == 0 or len(image_buffer) == 0 or len(buffer_metadata) == 0:
        logger.warning("Cannot reproject image buffer onto mesh - missing mesh or buffer data")
        return None, {}

    verts_np = vertices.detach().cpu().numpy() if torch.is_tensor(vertices) else np.asarray(vertices)
    faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
    buffer_np = np.asarray(image_buffer)

    width = int(buffer_metadata.get("width", buffer_np.shape[1]))
    height = int(buffer_metadata.get("height", buffer_np.shape[0]))
    x_min, x_max = buffer_metadata["x_range"]
    y_min, y_max = buffer_metadata["y_range"]
    pixels_per_meter = float(buffer_metadata["pixels_per_meter"])

    if width <= 0 or height <= 0:
        logger.warning("Cannot reproject image buffer onto mesh - invalid buffer dimensions")
        return None, {}

    # Sample each vertex from the final texture buffer.
    px = np.clip((verts_np[:, 0] - x_min) * pixels_per_meter, 0.0, max(width - 1, 0))
    py = np.clip((verts_np[:, 1] - y_min) * pixels_per_meter, 0.0, max(height - 1, 0))

    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    dx = (px - x0)[:, None]
    dy = (py - y0)[:, None]

    c00 = buffer_np[y0, x0]
    c10 = buffer_np[y0, x1]
    c01 = buffer_np[y1, x0]
    c11 = buffer_np[y1, x1]
    sampled_colors = (
        c00 * (1.0 - dx) * (1.0 - dy)
        + c10 * dx * (1.0 - dy)
        + c01 * (1.0 - dx) * dy
        + c11 * dx * dy
    )
    sampled_colors = np.clip(sampled_colors, 0.0, 1.0).astype(np.float32)

    vertex_colors = torch.from_numpy(sampled_colors)
    face_colors = sampled_colors[faces_np].mean(axis=1) if len(faces_np) > 0 else None
    non_bg_vertices = int(np.sum(np.any(sampled_colors > 0.01, axis=1)))

    reprojection_stats = {
        "mesh_vertices_reprojected": int(len(verts_np)),
        "mesh_vertices_textured": non_bg_vertices,
        "mesh_texture_coverage": float(non_bg_vertices / max(len(verts_np), 1)),
        "buffer_width": width,
        "buffer_height": height,
        "buffer_x_range": (float(x_min), float(x_max)),
        "buffer_y_range": (float(y_min), float(y_max)),
    }

    if output_dir:
        textured_mesh_path = os.path.join(output_dir, "road_textured_mesh.pth")
        torch.save(
            {
                "vertices": vertices.detach().cpu() if torch.is_tensor(vertices) else torch.as_tensor(vertices),
                "faces": faces.detach().cpu() if torch.is_tensor(faces) else torch.as_tensor(faces),
                "vertex_colors": vertex_colors.cpu(),
                "metadata": {
                    "mesh_type": "road_mesh_reprojected_from_image_buffer",
                    **reprojection_stats,
                },
            },
            textured_mesh_path,
        )
        logger.info(f"Saved reprojected textured mesh tensor to {textured_mesh_path}")

        export_textured_mesh_obj(
            vertices,
            faces,
            image_buffer,
            buffer_metadata,
            output_dir,
            filename="road_textured_mesh.obj",
            texture_filename="road_textured_mesh_texture.png",
        )
        export_mesh_to_ply(vertices, faces, vertex_colors, output_dir, filename="road_textured_mesh.ply")

        try:
            facecolors_rgba = None
            if face_colors is not None and len(face_colors) > 0:
                facecolors_rgba = np.concatenate(
                    [np.clip(face_colors, 0.0, 1.0), np.ones((len(face_colors), 1), dtype=np.float32)],
                    axis=1,
                )
            fig = plt.figure(figsize=(16, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_trisurf(
                verts_np[:, 0],
                verts_np[:, 1],
                verts_np[:, 2],
                triangles=faces_np,
                facecolors=facecolors_rgba,
                linewidth=0.1,
                antialiased=False,
                shade=False,
            )
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            ax.set_zlabel("Z (meters)")
            ax.set_title("Road Mesh Reprojected from Final Image Buffer")
            plt.tight_layout()
            preview_path = os.path.join(output_dir, "road_textured_mesh_reprojection.png")
            plt.savefig(preview_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved textured mesh preview to {preview_path}")
        except Exception as preview_error:
            logger.warning(f"Could not save textured mesh preview: {preview_error}")

    return vertex_colors, reprojection_stats


def export_mesh_to_obj(vertices: torch.Tensor, 
                       faces: torch.Tensor, 
                       colors: Union[torch.Tensor, None] = None,
                       output_dir: str = "",
                       filename: str = "road_mesh.obj"):
    """Export mesh to OBJ format for external inspection.
    
    The .obj format is a widely supported 3D file format that stores geometry as vertices and faces.
    It's compatible with most 3D software including Blender, MeshLab, Maya, and more.
    
    Args:
        vertices: M x 3 tensor of vertex positions in meters
        faces: F x 3 tensor of face indices (triangles)
        colors: Optional M x 3 tensor of RGB colors for each vertex
        output_dir: Directory to save the OBJ file
        
    Returns:
        Path to the saved .obj file
    """
    
    if len(vertices) == 0 or len(faces) == 0:
        logger.warning("Cannot export empty mesh")
        return None
    
    obj_path = os.path.join(output_dir, filename) if output_dir else None
    
    # Convert to numpy arrays for OBJ export
    verts_np = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else np.array(vertices)
    faces_np = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else np.array(faces)
    
    colors_arr = colors.cpu().numpy() if (colors is not None and len(colors) > 0) else None
    
    # Generate vertex colors for OBJ export
    if colors_arr is not None:
        rgb_colors = np.clip(colors_arr * 255, 0, 255).astype(np.float32)
    else:
        # Grayscale based on height relative to scene bounds
        z_min, z_max = verts_np[:, 2].min(), verts_np[:, 2].max()
        normalized_z = (verts_np[:, 2] - z_min) / max(z_max - z_min, 1e-6)
        rgb_colors = np.stack([normalized_z] * 3, axis=1).astype(np.float32)
    
    # Write OBJ file in ASCII format for human readability and compatibility
    with open(obj_path, 'w') as f:
        # File header (optional comments)
        f.write("# Road mesh exported from StreetPed training pipeline\n")
        f.write(f"# Vertices: {len(verts_np)}, Faces: {len(faces_np)}\n")
        f.write("# Units in meters, coordinate system: X-right, Y-forward, Z-up\n")
        f.write("\n")
        
        # Write vertex positions (OBJ uses 1-indexed vertices)
        for i in range(len(verts_np)):
            x, y, z = verts_np[i]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        
        # Write vertex colors (optional - as vn normals or embedded in faces)
        if rgb_colors is not None and len(rgb_colors) > 0:
            for i in range(len(verts_np)):
                r, g, b = rgb_colors[i]
                f.write(f"vn {r:.6f} {g:.6f} {b:.6f}\n")
        
        # Write face definitions (OBJ uses 1-indexed vertex references)
        for i in range(len(faces_np)):
            v0, v1, v2 = faces_np[i] + 1  # Convert to 1-based indexing
            if rgb_colors is not None and len(rgb_colors) > 0:
                f.write(f"f {v0}//{v0} {v1}//{v1} {v2}//{v2}\n")
            else:
                f.write(f"f {v0} {v1} {v2}\n")
    
    logger.info(f"Exported mesh to OBJ format at {obj_path}")
    return obj_path


def project_rgb_onto_image_buffer(
    dataset,
    vertices_tensor,
    faces_tensor,
    image_buffer,
    buffer_metadata,
    device,
    step1_dir=None,
    num_frames=None,
):
    """Project RGB values from road-masked images onto the image buffer using ray casting."""

    logger.info("\n" + "=" * 60)
    logger.info("STEP 4 - RGB IMAGE PROJECTION")
    logger.info("=" * 60)
    
        # Extract buffer parameters from metadata
    width_pixels = buffer_metadata.get('width', image_buffer.shape[1])
    height_pixels = buffer_metadata.get('height', image_buffer.shape[0])
    x_min, y_min = buffer_metadata['x_range'][0], buffer_metadata['y_range'][0]
    scene_width_meters = buffer_metadata['scene_width_meters']
    scene_height_meters = buffer_metadata['scene_height_meters']
    
    # Store buffer bounds for debugging
    x_max_buffer = x_min + scene_width_meters
    y_max_buffer = y_min + scene_height_meters
    
    # Initialize updated buffer as copy of input (preserves initial texture)
    updated_buffer = image_buffer.copy()
    
    # Statistics tracking with ray projection details
    projection_stats = {
        'total_frames': 0,
        'total_cameras_processed': 0,
        'road_pixels_projected': 0,
        'buffer_pixels_updated': set(),  # Will convert to count later
        'cameras_with_data': [],
        'rays_cast_total': 0,           # Total rays cast from cameras
        'rays_hit_mesh': 0,             # Rays that intersected mesh triangles
        'rays_missed_mesh': 0,          # Rays that didn't hit any triangle
    }
    
    if len(vertices_tensor) == 0 or faces_tensor.numel() == 0:
        logger.warning("Cannot project RGB - mesh not available")
        return updated_buffer, projection_stats
    
    # Convert mesh to numpy for ray intersection testing
    verts_np = vertices_tensor.cpu().numpy() if isinstance(vertices_tensor, torch.Tensor) else np.array(vertices_tensor)

    # Build triangle list with barycentric info for fast lookup
    triangles = []
    faces_np = faces_tensor.cpu().numpy() if isinstance(faces_tensor, torch.Tensor) else np.array(faces_tensor)
    for i in range(len(faces_np)):
        v0_idx, v1_idx, v2_idx = faces_np[i]
        v0 = verts_np[v0_idx]
        v1 = verts_np[v1_idx]
        v2 = verts_np[v2_idx]
        
        # Precompute triangle plane and edge functions for faster ray intersection
        tri_plane_normal = np.cross(v1 - v0, v2 - v0)
        tri_plane_normal = tri_plane_normal / (np.linalg.norm(tri_plane_normal) + 1e-8)
        tri_plane_d = -np.dot(tri_plane_normal, v0)
        
        triangles.append({
            'vertices': [v0, v1, v2],
            'plane_normal': tri_plane_normal,
            'plane_d': tri_plane_d,
            'face_idx': i
        })
    
    logger.info(f"Built {len(triangles)} triangle lookup table for ray intersection")
    
    total_train_indices = len(dataset.train_indices) if hasattr(dataset, 'train_indices') else 0
    # Iterate through a configurable number of training images and cameras to project RGB values.
    num_frames = total_train_indices if num_frames is None else min(num_frames, total_train_indices)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive when projecting RGB onto the image buffer")
    
    with tqdm(total=num_frames, desc="Processing frames", unit="frame") as pbar:
        for frame_idx in range(num_frames):
            projection_stats['total_frames'] += 1
            
            try:
                # Get image data and camera information for this frame from the dataset wrapper
                # The DrivingDataset stores images in full_image_set (SplitWrapper) which has get_image() method
                
                # Access through SplitWrapper - use normalized timestep index to map to actual indices
                if not hasattr(dataset, 'full_image_set') or dataset.full_image_set is None:
                    logger.warning(f"No full_image_set available for frame {frame_idx}, skipping")
                    pbar.update(1)
                    continue
                
                # Map from our iteration index (0..num_frames-1) to actual image indices in the dataset.
                if total_train_indices == 0:
                    logger.warning(f"No train_indices available for frame {frame_idx}, skipping")
                    pbar.update(1)
                    continue
                
                # Use proportional mapping to select frames from the dataset's training indices
                img_index = int(frame_idx * total_train_indices / num_frames) if num_frames > 0 else 0
                img_index = min(img_index, total_train_indices - 1)  # Ensure within bounds
                
                image_infos, cam_infos = dataset.full_image_set.get_image(img_index, camera_downscale=1.0)
                
                if 'pixels' not in image_infos or 'road_masks' not in image_infos:
                    pbar.update(1)
                    continue
                
                # Handle both single-camera and multi-camera formats from pixel_source.get_image()
                rgb_images = image_infos['pixels']  # Shape: [H, W, 3] for single cam or [num_cams, H, W, 3] for multi-cam
                road_masks = image_infos['road_masks'] if 'road_masks' in image_infos else None
                
                # Determine if we have multiple cameras (shape starts with num_cameras)
                is_multi_camera = torch.is_tensor(rgb_images) and len(rgb_images.shape) == 4 or \
                                  (isinstance(rgb_images, np.ndarray) and rgb_images.ndim == 4)
                
                # If single camera, wrap in list for consistent processing
                if not is_multi_camera:
                    rgb_list = [rgb_images]
                    road_mask_list = [road_masks] if road_masks is not None else []
                else:
                    rgb_list = torch.unbind(rgb_images, dim=0) if torch.is_tensor(rgb_images) else list(rgb_images)
                    road_mask_list = torch.unbind(road_masks, dim=0) if (torch.is_tensor(road_masks)) and road_masks is not None else [None] * len(rgb_list)

                # Process each camera view in the frame
                for cam_id, rgb_img in enumerate(rgb_list):
                    projection_stats['total_cameras_processed'] += 1
                    
                    # Get road mask for this camera (handle None case)
                    road_mask = road_mask_list[cam_id] if cam_id < len(road_mask_list) and road_mask_list[cam_id] is not None else np.zeros(rgb_img.shape[:2], dtype=np.float32)
                    
                    # Convert tensor to numpy on CPU before processing (fixes CUDA->numpy conversion error)
                    if torch.is_tensor(road_mask):
                        road_mask = road_mask.cpu().numpy()
                    
                    # Get camera intrinsics and extrinsics for this frame/camera
                    K = None
                    camToWorld = None
                    
                    # Try different possible key names for intrinsics (from pixel_source.get_image)
                    if 'intrinsics' in cam_infos:
                        try:
                            K = cam_infos['intrinsics'].cpu().numpy() if torch.is_tensor(cam_infos['intrinsics']) else np.array(cam_infos['intrinsics'])
                        except Exception as e:
                            logger.warning(f"Failed to extract intrinsics for frame {frame_idx}: {e}")
                    
                    # Try different possible key names for extrinsics (camera-to-world transform)
                    if K is not None and 'camera_to_world' in cam_infos:
                        try:
                            # camera_to_world might be a single 4x4 matrix or per-camera array
                            c2w = cam_infos['camera_to_world']
                            if torch.is_tensor(c2w):
                                c2w_np = c2w.cpu().numpy()
                            else:
                                c2w_np = np.array(c2w)
                            
                            # Handle both single matrix and per-camera array formats
                            if len(c2w_np.shape) == 3:
                                camToWorld = c2w_np[0]  # Take first camera view
                            elif len(c2w_np.shape) == 2:
                                camToWorld = c2w_np
                            else:
                                logger.warning(f"Unexpected shape for camera_to_world: {c2w_np.shape}")
                        except Exception as e:
                            logger.warning(f"Failed to extract extrinsics for frame {frame_idx}: {e}")
                    
                    if K is None or camToWorld is None:
                        pbar.update(1)
                        continue
                    
                    # Find road pixels in this image (single camera view from pixel_source.get_image())
                    road_pixel_indices = np.where(road_mask > 0.5)

                    if len(road_pixel_indices[0]) == 0:
                        continue
                    
                    rays_per_image = 30000
                    num_road_pixels = len(road_pixel_indices[0])
                    if num_road_pixels > rays_per_image:
                        logger.info(f"Sampling {rays_per_image} of {num_road_pixels} road pixels (frame {frame_idx}, cam {cam_id})")
                        # Randomly sample up to 100 pixel indices
                        sampled_indices = np.random.choice(num_road_pixels, size=rays_per_image, replace=False)
                        selected_py = road_pixel_indices[0][sampled_indices]
                        selected_px = road_pixel_indices[1][sampled_indices]
                    else:
                        # Use all available pixels if fewer than 100
                        selected_py = road_pixel_indices[0]
                        selected_px = road_pixel_indices[1]

                    projection_stats['cameras_with_data'].append(f"frame_{frame_idx}_cam{cam_id}")

                    # Log first frame/camera for debugging (only once at the start of processing)
                    if 'first_debug_log' not in locals():
                        logger.info(f"\nProcessing frame {frame_idx}, camera {cam_id}:")
                        logger.info(f"  - RGB image shape: {rgb_img.shape}")
                        logger.info(f"  - Road mask pixels found: {len(road_pixel_indices[0])}")
                        if K is not None and len(K.shape) == 2:
                            logger.info(f"  - Camera intrinsics (K): fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
                        if camToWorld is not None:
                            logger.info(f"  - Camera position (world): {camToWorld[:3, 3]}")
                        # Log mesh bounds for debugging
                        x_min_mesh = verts_np[:, 0].min()
                        y_min_mesh = verts_np[:, 1].min()
                        z_min_mesh = verts_np[:, 2].min()
                        
                        # Define buffer coordinate variables (same as projection logic)
                        x_min_buffer = float(x_min)
                        y_min_buffer = float(y_min)
                        scene_width_meters_float = float(scene_width_meters)
                        scene_height_meters_float = float(scene_height_meters)
                        x_max_buffer = x_min_buffer + scene_width_meters_float
                        y_max_buffer = y_min_buffer + scene_height_meters_float
                        
                        logger.info(f"  - Mesh bounds: X=[{x_min_mesh:.3f}, {verts_np[:, 0].max():.3f}], Y=[{y_min_mesh:.3f}, {verts_np[:, 1].max():.3f}], Z=[{z_min_mesh:.3f}, {verts_np[:, 2].max():.3f}]")
                        logger.info(f"  - Image buffer bounds: X=[{x_min_buffer:.3f}, {x_max_buffer:.3f}], Y=[{y_min_buffer:.3f}, {y_max_buffer:.3f}]")

                    # Project each road pixel onto the image buffer via ray casting (max {rays_per_image} rays per image)
                    for py, px in tqdm(zip(selected_py, selected_px), desc=f"Projecting pixels (frame {frame_idx}, cam {cam_id})"):
                        projection_stats['rays_cast_total'] += 1
                        # Get RGB color at this pixel (already normalized to [0, 1] by dataset loader)
                        rgb_color = rgb_img[py, px]
                        
                        # Ensure rgb_color is on CPU and converted to numpy before use
                        if torch.is_tensor(rgb_color):
                            rgb_color = rgb_color.cpu().numpy()
                        
                        # Cast ray from camera through this pixel into the scene
                        # Pixel coordinates to normalized device coordinates using intrinsics K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                        
                        # Handle both numpy array and list formats for K matrix
                        if isinstance(K, np.ndarray) or (hasattr(K, '__getitem__') and len(np.shape(K)) == 2):
                            fx = float(K[0, 0])
                            fy = float(K[1, 1])
                            cx = float(K[0, 2])
                            cy = float(K[1, 2])
                        else:
                            # Fallback for unexpected format - log warning and skip this pixel
                            logger.warning(f"Unexpected K matrix shape: {type(K)}, skipping pixel ({px}, {py})")
                            continue
                        
                        # Match datasets/base/pixel_source.py::get_rays():
                        # pixel centers are offset by +0.5 and image Y grows downward.
                        x_ndc = float((px - cx + 0.5) / fx) if abs(fx) > 1e-8 else 0.0
                        y_ndc = float((py - cy + 0.5) / fy) if abs(fy) > 1e-8 else 0.0

                        # Ray direction in camera frame: [x, y, z] where z points along optical axis (forward)
                        ray_dir_cam = np.array([x_ndc, y_ndc, 1.0])

                        # Normalize the ray direction vector to unit length for accurate intersection testing
                        ray_length = float(np.linalg.norm(ray_dir_cam))
                        if abs(ray_length) > 1e-8:
                            ray_dir_cam = ray_dir_cam / ray_length
                        
                        # Transform from camera frame to world frame using the camera_to_world pose directly.
                        ray_origin_world = camToWorld[:3, 3]

                        # Rotate ray direction into world frame: R @ d where R is the camera-to-world rotation.
                        rot_matrix = camToWorld[:3, :3]
                        ray_dir_world = np.dot(rot_matrix, ray_dir_cam)
                        
                        # DEBUG LOGGING for first pixel to diagnose coordinate system issues
                        if 'first_ray_debug' not in locals():
                            logger.info(f"\n=== RAY CASTING DEBUG (frame {frame_idx}, cam {cam_id}) ===")
                            logger.info(f"Ray origin world: {ray_origin_world}")
                            logger.info(f"Ray direction world: {ray_dir_world}")
                            logger.info(f"Mesh bounds check:")
                            logger.info(f"  Ray origin X={ray_origin_world[0]:.3f} in mesh range [{x_min_mesh:.3f}, {verts_np[:, 0].max():.3f}]? {'YES' if x_min_mesh <= ray_origin_world[0] <= verts_np[:, 0].max() else 'NO'}")
                            logger.info(f"  Ray origin Y={ray_origin_world[1]:.3f} in mesh range [{y_min_mesh:.3f}, {verts_np[:, 1].max():.3f}]? {'YES' if y_min_mesh <= ray_origin_world[1] <= verts_np[:, 1].max() else 'NO'}")
                            logger.info(f"  Ray origin Z={ray_origin_world[2]:.3f} in mesh range [{z_min_mesh:.3f}, {verts_np[:, 2].max():.3f}]? {'YES' if z_min_mesh <= ray_origin_world[2] <= verts_np[:, 2].max() else 'NO'}")
                            logger.info(f"Ray direction Z component: {ray_dir_world[2]:.4f} (should be negative for downward rays)")
                            first_ray_debug = True
                        
                        # Ray-plane intersection: t = -(plane_d + n·o) / (n·d)
                        # For our case, we project onto the X-Y grid plane at mesh level
                        
                        # Project ray direction to find where it hits the buffer plane
                        # We'll use a simplified approach: cast ray and check if it intersects any triangle
                        
                        t_hit = None
                        hit_point = None
                        
                        # For heightmap meshes, we need to find the triangle that projects onto this buffer pixel
                        # Instead of ray-triangle intersection from camera, project each point in mesh plane and check visibility
                        
                        for tri_idx, tri in enumerate(triangles):
                            v0, v1, v2 = tri['vertices']
                            
                            # Ray-triangle intersection using Moller-Trumbore algorithm (optimized)
                            edge1 = v1 - v0
                            edge2 = v2 - v0
                            h_vec = np.cross(ray_dir_world, edge2)
                            det_a = np.dot(edge1, h_vec)
                            
                            # Check if ray is parallel to triangle plane
                            if abs(det_a) > 1e-8:  
                                f_inv = 1.0 / det_a
                                s_vec = ray_origin_world - v0
                                
                                u = f_inv * np.dot(s_vec, h_vec)
                                
                                # Check barycentric coordinate constraints
                                if u >= 0 and u <= 1:
                                    q_cross_edge1 = np.cross(s_vec, edge1)
                                    v = f_inv * np.dot(ray_dir_world, q_cross_edge1)
                                    
                                    # Second barycentric constraint
                                    if v >= 0 and (u + v) <= 1:
                                        # Ray intersects triangle at distance t
                                        t_hit_val = f_inv * np.dot(edge2, q_cross_edge1)
                                        
                                                                                # Hit must be in front of camera and not too close (avoid self-intersection)
                                        if t_hit_val > 0.01:  
                                            hit_point = ray_origin_world + t_hit_val * ray_dir_world
                                            projection_stats['rays_hit_mesh'] += 1
                                            logger.debug(f"Ray HIT at frame {frame_idx}, cam {cam_id}: distance={t_hit_val:.3f}m, point={hit_point}")
                                            # For heightmap meshes, prefer closer intersections first
                                            break
                                    else:
                                        v = -1.0  # Will fail check

                        if hit_point is None:
                            projection_stats['rays_missed_mesh'] += 1
                            continue

                        # Map intersection point to image buffer coordinates (world -> pixel)
                        buf_x_idx = int((hit_point[0] - x_min) / scene_width_meters * width_pixels)
                        buf_y_idx = int((hit_point[1] - y_min) / scene_height_meters * height_pixels)
                        
                        # Clamp to valid buffer range
                        buf_x_idx = np.clip(buf_x_idx, 0, width_pixels - 1)
                        buf_y_idx = np.clip(buf_y_idx, 0, height_pixels - 1)
                        
                        updated_buffer[buf_y_idx, buf_x_idx] = rgb_color
                        projection_stats['buffer_pixels_updated'].add((buf_y_idx, buf_x_idx))
                        print(f"Updated buffer pixel at ({buf_y_idx}, {buf_x_idx}) with color {rgb_color}")
                    
                    # Save the image buffer after processing each frame/camera combination
                    save_path = os.path.join(step1_dir, f"frame_{frame_idx}_cam{cam_id}_buffer.png") if step1_dir else None
                    
                    try:
                        if save_path and len(selected_py) > 0:
                            # Convert to uint8 for PNG saving
                            buffer_uint8 = np.clip(updated_buffer * 255.0, 0, 255).astype(np.uint8)
                            
                            plt.figure(figsize=(buffer_metadata['width']/72, buffer_metadata['height']/72), dpi=72)
                            plt.imshow(buffer_uint8)
                            plt.axis('off')
                            plt.tight_layout()
                            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                            plt.close()
                            
                            logger.info(f"Saved image buffer after frame {frame_idx}, camera {cam_id} to {save_path}")
                    except Exception as save_error:
                        logger.warning(f"Could not save intermediate buffer for frame {frame_idx}: {save_error}")

                    pbar.update(1)
            
            except Exception as e:
                logger.warning(f"Error processing frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
    
        # Convert set to count for stats
    projection_stats['road_pixels_projected'] = len(projection_stats['buffer_pixels_updated'])
    num_unique_buffer_pixels = len(projection_stats['buffer_pixels_updated'])
    del projection_stats['buffer_pixels_updated']  # Remove raw set from final output
    
    hit_rate = (projection_stats['rays_hit_mesh'] / max(projection_stats['rays_cast_total'], 1)) * 100 if 'rays_cast_total' in projection_stats else 0.0
    miss_rate = (projection_stats['rays_missed_mesh'] / max(projection_stats['rays_cast_total'], 1)) * 100 if 'rays_missed_mesh' in projection_stats and projection_stats['rays_cast_total'] > 0 else 0.0
    
    logger.info("\n" + "="*60)
    logger.info("RGB PROJECTION STATISTICS")
    logger.info("="*60)
    logger.info(f"Total frames processed: {projection_stats['total_frames']}")
    logger.info(f"Total cameras processed: {projection_stats['total_cameras_processed']}")
    logger.info(f"Rays cast from camera: {projection_stats.get('rays_cast_total', 0):,}")
    logger.info(f"Rays that HIT mesh triangles: {projection_stats.get('rays_hit_mesh', 0):,} ({hit_rate:.1f}%)")
    logger.info(f"Rays MISSED all triangles: {projection_stats.get('rays_missed_mesh', 0):,} ({miss_rate:.1f}%)")
    logger.info(f"Unique buffer pixels updated with RGB colors: {num_unique_buffer_pixels:,}")
    if projection_stats['total_cameras_processed'] > 0:
        avg_rays_per_camera = projection_stats.get('rays_cast_total', 0) / max(projection_stats['total_cameras_processed'], 1)
        logger.info(f"Average rays per camera: {avg_rays_per_camera:.1f}")
    
    # Log potential issues if hit rate is low
    if 'rays_hit_mesh' in projection_stats and hit_rate < 50:
        logger.warning("LOW HIT RATE DETECTED - Check coordinate system alignment!")
        logger.warning(f"Only {hit_rate:.1f}% of rays intersected the mesh")

    # Suggest fixes based on common issues
    if 'first_ray_debug' in locals():
        logger.info("")
        logger.info("POSSIBLE FIXES:")
        logger.info("  1. Check ray direction Z component - should be negative for downward rays onto heightmap")
        logger.info("  2. Verify camera extrinsics matrix orientation (world-to-camera vs camera-to-world)")
        logger.info("  3. Ensure mesh coordinate system matches world space from dataset")

    logger.info(f"Projected {projection_stats['total_cameras_processed']} camera views")
    logger.info(f"Updated {num_unique_buffer_pixels:,} unique buffer pixels with RGB colors")
    
    return updated_buffer, projection_stats


def _build_triangle_cache_gpu(vertices_tensor, faces_tensor, device):
    """Build triangle vertices directly on the target device."""
    if torch.is_tensor(vertices_tensor):
        verts = vertices_tensor.to(device=device, dtype=torch.float32)
    else:
        verts = torch.as_tensor(vertices_tensor, dtype=torch.float32, device=device)

    if torch.is_tensor(faces_tensor):
        faces = faces_tensor.to(device=device, dtype=torch.long)
    else:
        faces = torch.as_tensor(faces_tensor, dtype=torch.long, device=device)

    tri_vertices = verts[faces]
    return tri_vertices[:, 0, :], tri_vertices[:, 1, :], tri_vertices[:, 2, :]


def _intersect_rays_with_triangles_gpu(
    ray_origins,
    ray_directions,
    triangle_vertices,
    ray_chunk_size=512,
    triangle_chunk_size=4096,
    min_distance=0.01,
    eps=1e-8,
):
    """Find the nearest triangle hit for each ray using batched GPU work."""
    tri_v0, tri_v1, tri_v2 = triangle_vertices
    device = ray_origins.device
    dtype = ray_origins.dtype
    num_rays = ray_origins.shape[0]

    best_t = torch.full((num_rays,), float("inf"), device=device, dtype=dtype)
    best_hit_points = torch.full((num_rays, 3), float("nan"), device=device, dtype=dtype)

    for ray_start in range(0, num_rays, ray_chunk_size):
        ray_end = min(ray_start + ray_chunk_size, num_rays)
        origin_chunk = ray_origins[ray_start:ray_end]
        direction_chunk = ray_directions[ray_start:ray_end]

        chunk_best_t = torch.full((ray_end - ray_start,), float("inf"), device=device, dtype=dtype)
        chunk_best_points = torch.full((ray_end - ray_start, 3), float("nan"), device=device, dtype=dtype)

        for tri_start in range(0, tri_v0.shape[0], triangle_chunk_size):
            tri_end = min(tri_start + triangle_chunk_size, tri_v0.shape[0])
            v0 = tri_v0[tri_start:tri_end]
            v1 = tri_v1[tri_start:tri_end]
            v2 = tri_v2[tri_start:tri_end]

            edge1 = v1 - v0
            edge2 = v2 - v0

            h_vec = torch.cross(direction_chunk[:, None, :], edge2[None, :, :], dim=-1)
            det = torch.sum(edge1[None, :, :] * h_vec, dim=-1)
            valid_det = det.abs() > eps

            inv_det = torch.zeros_like(det)
            inv_det[valid_det] = 1.0 / det[valid_det]

            s_vec = origin_chunk[:, None, :] - v0[None, :, :]
            u = inv_det * torch.sum(s_vec * h_vec, dim=-1)

            q_vec = torch.cross(s_vec, edge1[None, :, :], dim=-1)
            v = inv_det * torch.sum(direction_chunk[:, None, :] * q_vec, dim=-1)
            t = inv_det * torch.sum(edge2[None, :, :] * q_vec, dim=-1)

            hit_mask = (
                valid_det
                & (u >= 0.0)
                & (v >= 0.0)
                & ((u + v) <= 1.0)
                & (t > min_distance)
            )

            candidate_t = torch.where(hit_mask, t, torch.full_like(t, float("inf")))
            local_best_t, _ = torch.min(candidate_t, dim=1)
            better = local_best_t < chunk_best_t

            if better.any():
                chunk_best_t = torch.where(better, local_best_t, chunk_best_t)
                candidate_points = origin_chunk + local_best_t.unsqueeze(1) * direction_chunk
                chunk_best_points = torch.where(better.unsqueeze(1), candidate_points, chunk_best_points)

        best_t[ray_start:ray_end] = chunk_best_t
        best_hit_points[ray_start:ray_end] = chunk_best_points

    hit_mask = torch.isfinite(best_t)
    return best_hit_points, hit_mask, best_t


def project_rgb_onto_image_buffer_gpu(
    dataset,
    vertices_tensor,
    faces_tensor,
    image_buffer,
    buffer_metadata,
    device,
    step1_dir=None,
    num_frames=None,
    ray_chunk_size=1024,
    triangle_chunk_size=4096,
):
    """GPU batched version of RGB projection onto the image buffer."""
    if not torch.cuda.is_available() or getattr(device, "type", str(device)) != "cuda":
        logger.info("CUDA is unavailable; falling back to the CPU projection path.")
        return project_rgb_onto_image_buffer(
            dataset,
            vertices_tensor,
            faces_tensor,
            image_buffer,
            buffer_metadata,
            device,
            step1_dir=step1_dir,
            num_frames=num_frames,
        )

    logger.info("\n" + "=" * 60)
    logger.info("STEP 4 - RGB IMAGE PROJECTION (GPU)")
    logger.info("=" * 60)

    width_pixels = buffer_metadata.get('width', image_buffer.shape[1])
    height_pixels = buffer_metadata.get('height', image_buffer.shape[0])
    x_min, y_min = buffer_metadata['x_range'][0], buffer_metadata['y_range'][0]
    scene_width_meters = buffer_metadata['scene_width_meters']
    scene_height_meters = buffer_metadata['scene_height_meters']

    updated_buffer = image_buffer.copy()
    projection_stats = {
        'total_frames': 0,
        'total_cameras_processed': 0,
        'road_pixels_projected': 0,
        'buffer_pixels_updated': set(),
        'cameras_with_data': [],
        'rays_cast_total': 0,
        'rays_hit_mesh': 0,
        'rays_missed_mesh': 0,
    }

    if len(vertices_tensor) == 0 or faces_tensor.numel() == 0:
        logger.warning("Cannot project RGB - mesh not available")
        return updated_buffer, projection_stats

    triangle_vertices = _build_triangle_cache_gpu(vertices_tensor, faces_tensor, device)
    logger.info(f"Built {triangle_vertices[0].shape[0]} triangle lookup entries on GPU")
    verts_np = vertices_tensor.cpu().numpy() if torch.is_tensor(vertices_tensor) else np.asarray(vertices_tensor)

    total_train_indices = len(dataset.train_indices) if hasattr(dataset, 'train_indices') else 0
    num_frames = total_train_indices if num_frames is None else min(num_frames, total_train_indices)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive when projecting RGB onto the image buffer")

    first_debug_log_done = False
    first_ray_debug_done = False

    with torch.no_grad():
        with tqdm(total=num_frames, desc="Processing frames", unit="frame") as pbar:
            for frame_idx in range(num_frames):
                projection_stats['total_frames'] += 1

                try:
                    if not hasattr(dataset, 'full_image_set') or dataset.full_image_set is None:
                        logger.warning(f"No full_image_set available for frame {frame_idx}, skipping")
                        pbar.update(1)
                        continue

                    if total_train_indices == 0:
                        logger.warning(f"No train_indices available for frame {frame_idx}, skipping")
                        pbar.update(1)
                        continue

                    img_index = int(frame_idx * total_train_indices / num_frames) if num_frames > 0 else 0
                    img_index = min(img_index, total_train_indices - 1)

                    image_infos, cam_infos = dataset.full_image_set.get_image(img_index, camera_downscale=1.0)

                    if 'pixels' not in image_infos or 'road_masks' not in image_infos:
                        pbar.update(1)
                        continue

                    rgb_images = image_infos['pixels']
                    road_masks = image_infos['road_masks'] if 'road_masks' in image_infos else None

                    is_multi_camera = (torch.is_tensor(rgb_images) and len(rgb_images.shape) == 4) or (
                        isinstance(rgb_images, np.ndarray) and rgb_images.ndim == 4
                    )

                    if not is_multi_camera:
                        rgb_list = [rgb_images]
                        road_mask_list = [road_masks] if road_masks is not None else []
                    else:
                        rgb_list = torch.unbind(rgb_images, dim=0) if torch.is_tensor(rgb_images) else list(rgb_images)
                        road_mask_list = (
                            torch.unbind(road_masks, dim=0)
                            if torch.is_tensor(road_masks) and road_masks is not None
                            else [None] * len(rgb_list)
                        )

                    for cam_id, rgb_img in enumerate(rgb_list):
                        projection_stats['total_cameras_processed'] += 1

                        road_mask = (
                            road_mask_list[cam_id]
                            if cam_id < len(road_mask_list) and road_mask_list[cam_id] is not None
                            else np.zeros(rgb_img.shape[:2], dtype=np.float32)
                        )
                        if torch.is_tensor(road_mask):
                            road_mask = road_mask.cpu().numpy()

                        K = None
                        camToWorld = None

                        if 'intrinsics' in cam_infos:
                            try:
                                K = cam_infos['intrinsics'].cpu().numpy() if torch.is_tensor(cam_infos['intrinsics']) else np.array(cam_infos['intrinsics'])
                            except Exception as e:
                                logger.warning(f"Failed to extract intrinsics for frame {frame_idx}: {e}")

                        if K is not None and 'camera_to_world' in cam_infos:
                            try:
                                c2w = cam_infos['camera_to_world']
                                c2w_np = c2w.cpu().numpy() if torch.is_tensor(c2w) else np.array(c2w)
                                if len(c2w_np.shape) == 3:
                                    camToWorld = c2w_np[0]
                                elif len(c2w_np.shape) == 2:
                                    camToWorld = c2w_np
                                else:
                                    logger.warning(f"Unexpected shape for camera_to_world: {c2w_np.shape}")
                            except Exception as e:
                                logger.warning(f"Failed to extract extrinsics for frame {frame_idx}: {e}")

                        if K is None or camToWorld is None:
                            pbar.update(1)
                            continue

                        road_pixel_indices = np.where(road_mask > 0.5)
                        if len(road_pixel_indices[0]) == 0:
                            continue

                        rays_per_image = 30000
                        num_road_pixels = len(road_pixel_indices[0])
                        if num_road_pixels > rays_per_image:
                            logger.info(f"Sampling {rays_per_image} of {num_road_pixels} road pixels (frame {frame_idx}, cam {cam_id})")
                            sampled_indices = np.random.choice(num_road_pixels, size=rays_per_image, replace=False)
                            selected_py = road_pixel_indices[0][sampled_indices]
                            selected_px = road_pixel_indices[1][sampled_indices]
                        else:
                            selected_py = road_pixel_indices[0]
                            selected_px = road_pixel_indices[1]

                        projection_stats['cameras_with_data'].append(f"frame_{frame_idx}_cam{cam_id}")

                        if not first_debug_log_done:
                            logger.info(f"\nProcessing frame {frame_idx}, camera {cam_id}:")
                            logger.info(f"  - RGB image shape: {rgb_img.shape}")
                            logger.info(f"  - Road mask pixels found: {len(road_pixel_indices[0])}")
                            if K is not None and len(K.shape) == 2:
                                logger.info(f"  - Camera intrinsics (K): fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
                            if camToWorld is not None:
                                logger.info(f"  - Camera position (world): {camToWorld[:3, 3]}")
                            x_min_mesh = verts_np[:, 0].min()
                            y_min_mesh = verts_np[:, 1].min()
                            z_min_mesh = verts_np[:, 2].min()
                            x_max_mesh = verts_np[:, 0].max()
                            y_max_mesh = verts_np[:, 1].max()
                            z_max_mesh = verts_np[:, 2].max()
                            logger.info(f"  - Mesh bounds: X=[{x_min_mesh:.3f}, {x_max_mesh:.3f}], Y=[{y_min_mesh:.3f}, {y_max_mesh:.3f}], Z=[{z_min_mesh:.3f}, {z_max_mesh:.3f}]")
                            logger.info(f"  - Image buffer bounds: X=[{float(x_min):.3f}, {float(x_min + scene_width_meters):.3f}], Y=[{float(y_min):.3f}, {float(y_min + scene_height_meters):.3f}]")
                            first_debug_log_done = True

                        if torch.is_tensor(rgb_img):
                            rgb_img_np = rgb_img.detach().cpu().numpy()
                        else:
                            rgb_img_np = np.asarray(rgb_img)

                        rgb_colors = rgb_img_np[selected_py, selected_px]
                        if rgb_colors.shape[0] == 0:
                            continue

                        if isinstance(K, np.ndarray) or (hasattr(K, '__getitem__') and len(np.shape(K)) == 2):
                            fx = float(K[0, 0])
                            fy = float(K[1, 1])
                            cx = float(K[0, 2])
                            cy = float(K[1, 2])
                        else:
                            logger.warning(f"Unexpected K matrix shape: {type(K)}, skipping frame {frame_idx}, cam {cam_id}")
                            continue

                        px_tensor = torch.as_tensor(selected_px, dtype=torch.float32, device=device)
                        py_tensor = torch.as_tensor(selected_py, dtype=torch.float32, device=device)

                        if abs(fx) > 1e-8:
                            x_ndc = (px_tensor - cx + 0.5) / fx
                        else:
                            x_ndc = torch.zeros_like(px_tensor)

                        if abs(fy) > 1e-8:
                            y_ndc = (py_tensor - cy + 0.5) / fy
                        else:
                            y_ndc = torch.zeros_like(py_tensor)
                        ray_dirs_cam = torch.stack([x_ndc, y_ndc, torch.ones_like(x_ndc)], dim=-1)
                        ray_dirs_cam = torch.nn.functional.normalize(ray_dirs_cam, dim=-1)

                        cam_origin = torch.as_tensor(camToWorld[:3, 3], dtype=torch.float32, device=device)
                        rot_matrix = torch.as_tensor(camToWorld[:3, :3], dtype=torch.float32, device=device)
                        ray_origins = cam_origin.unsqueeze(0).expand(ray_dirs_cam.shape[0], -1)
                        ray_dirs_world = ray_dirs_cam @ rot_matrix.T

                        if not first_ray_debug_done:
                            ray_origin_world = ray_origins[0].detach().cpu().numpy()
                            ray_dir_world = ray_dirs_world[0].detach().cpu().numpy()
                            logger.info(f"\n=== RAY CASTING DEBUG (frame {frame_idx}, cam {cam_id}) ===")
                            logger.info(f"Ray origin world: {ray_origin_world}")
                            logger.info(f"Ray direction world: {ray_dir_world}")
                            first_ray_debug_done = True

                        best_hit_points, hit_mask, _ = _intersect_rays_with_triangles_gpu(
                            ray_origins,
                            ray_dirs_world,
                            triangle_vertices,
                            ray_chunk_size=ray_chunk_size,
                            triangle_chunk_size=triangle_chunk_size,
                        )

                        num_rays = int(ray_origins.shape[0])
                        num_hits = int(hit_mask.sum().item())
                        projection_stats['rays_cast_total'] += num_rays
                        projection_stats['rays_hit_mesh'] += num_hits
                        projection_stats['rays_missed_mesh'] += num_rays - num_hits

                        if num_hits == 0:
                            continue

                        hit_points_np = best_hit_points[hit_mask].detach().cpu().numpy()
                        hit_colors_np = rgb_colors[hit_mask.detach().cpu().numpy()]

                        buf_x_idx = np.clip(
                            ((hit_points_np[:, 0] - x_min) / scene_width_meters * width_pixels).astype(np.int64),
                            0,
                            width_pixels - 1,
                        )
                        buf_y_idx = np.clip(
                            ((hit_points_np[:, 1] - y_min) / scene_height_meters * height_pixels).astype(np.int64),
                            0,
                            height_pixels - 1,
                        )

                        updated_buffer[buf_y_idx, buf_x_idx] = hit_colors_np
                        projection_stats['buffer_pixels_updated'].update(zip(buf_y_idx.tolist(), buf_x_idx.tolist()))
                        print(f"Updated buffer pixels with {num_hits} GPU ray hits for frame {frame_idx}, cam {cam_id}")

                        save_path = os.path.join(step1_dir, f"frame_{frame_idx}_cam{cam_id}_buffer.png") if step1_dir else None
                        try:
                            if save_path and len(selected_py) > 0:
                                buffer_uint8 = np.clip(updated_buffer * 255.0, 0, 255).astype(np.uint8)
                                plt.figure(figsize=(buffer_metadata['width'] / 72, buffer_metadata['height'] / 72), dpi=72)
                                plt.imshow(buffer_uint8)
                                plt.axis('off')
                                plt.tight_layout()
                                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                                plt.close()
                                logger.info(f"Saved image buffer after frame {frame_idx}, camera {cam_id} to {save_path}")
                        except Exception as save_error:
                            logger.warning(f"Could not save intermediate buffer for frame {frame_idx}: {save_error}")

                    pbar.update(1)

                except Exception as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    import traceback
                    traceback.print_exc()

    projection_stats['road_pixels_projected'] = len(projection_stats['buffer_pixels_updated'])
    num_unique_buffer_pixels = len(projection_stats['buffer_pixels_updated'])
    del projection_stats['buffer_pixels_updated']

    hit_rate = (projection_stats['rays_hit_mesh'] / max(projection_stats['rays_cast_total'], 1)) * 100 if 'rays_cast_total' in projection_stats else 0.0
    miss_rate = (projection_stats['rays_missed_mesh'] / max(projection_stats['rays_cast_total'], 1)) * 100 if 'rays_missed_mesh' in projection_stats and projection_stats['rays_cast_total'] > 0 else 0.0

    logger.info("\n" + "=" * 60)
    logger.info("RGB PROJECTION STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total frames processed: {projection_stats['total_frames']}")
    logger.info(f"Total cameras processed: {projection_stats['total_cameras_processed']}")
    logger.info(f"Rays cast from camera: {projection_stats.get('rays_cast_total', 0):,}")
    logger.info(f"Rays that HIT mesh triangles: {projection_stats.get('rays_hit_mesh', 0):,} ({hit_rate:.1f}%)")
    logger.info(f"Rays MISSED all triangles: {projection_stats.get('rays_missed_mesh', 0):,} ({miss_rate:.1f}%)")
    logger.info(f"Unique buffer pixels updated with RGB colors: {num_unique_buffer_pixels:,}")
    if projection_stats['total_cameras_processed'] > 0:
        avg_rays_per_camera = projection_stats.get('rays_cast_total', 0) / max(projection_stats['total_cameras_processed'], 1)
        logger.info(f"Average rays per camera: {avg_rays_per_camera:.1f}")

    if 'rays_hit_mesh' in projection_stats and hit_rate < 50:
        logger.warning("LOW HIT RATE DETECTED - Check coordinate system alignment!")
        logger.warning(f"Only {hit_rate:.1f}% of rays intersected the mesh")

    if first_ray_debug_done:
        logger.info("")
        logger.info("POSSIBLE FIXES:")
        logger.info("  1. Check ray direction Z component - should be negative for downward rays onto heightmap")
        logger.info("  2. Verify camera extrinsics matrix orientation (world-to-camera vs camera-to-world)")
        logger.info("  3. Ensure mesh coordinate system matches world space from dataset")

    logger.info(f"Projected {projection_stats['total_cameras_processed']} camera views")
    logger.info(f"Updated {num_unique_buffer_pixels:,} unique buffer pixels with RGB colors")

    return updated_buffer, projection_stats


def create_camera_view_comparison(dataset, vertices_tensor, faces_tensor, cam_id=2, frame_idx=None, output_dir="", num_frames=None):
    """
    Create a side-by-side comparison image of RGB view and mesh render from the same camera perspective.
    
    This function helps verify that world space alignment is correct between:
    - The actual RGB camera image
    - A rendered view of the road mesh from the same camera pose
    
    Args:
        dataset: DrivingDataset instance with loaded data and camera information
        vertices_tensor: M x 3 tensor of vertex positions in meters
        faces_tensor: F x 3 tensor of face indices (triangles)
        cam_id: Camera ID to visualize (default=2, middle camera view)
        frame_idx: Specific frame index to use. If None, uses first available frame.
        output_dir: Directory to save the comparison image
        num_frames: Total number of frames for progress tracking
        
    Returns:
        Path to saved comparison file or None if failed
    """
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d.art3d import PolyCollection
    
    logger.info(f"\nCreating camera view comparison (camera {cam_id})...")
    
    # Convert mesh to numpy for rendering
    verts_np = vertices_tensor.cpu().numpy() if isinstance(vertices_tensor, torch.Tensor) else np.array(vertices_tensor)
    faces_np = faces_tensor.cpu().numpy() if isinstance(faces_tensor, torch.Tensor) else np.array(faces_tensor)
    
    logger.info(f"Mesh dimensions: {len(verts_np)} vertices, {len(faces_np)} triangles")
    
    # Validate vertex-face consistency (debugging check)
    max_face_idx = faces_np.max() if len(faces_np) > 0 else -1
    min_face_idx = faces_np.min() if len(faces_np) > 0 else 999
    
    if len(verts_np) == 0 or len(faces_np) == 0:
        logger.warning("Cannot create comparison - empty mesh")
        return None
    
    # Check for vertex-face index mismatch (common issue with grid-based meshes)
    if max_face_idx >= len(verts_np):
        logger.warning(f"Vertex-face index MISMATCH detected! Max face index {max_face_idx} exceeds vertex count {len(verts_np)}")
        logger.warning("This indicates a bug in mesh creation where faces reference non-existent vertices.")
    
    # Determine which frame to use for visualization
    total_train_indices = len(dataset.train_indices) if hasattr(dataset, 'train_indices') else 0
    frame_count = total_train_indices if num_frames is None else min(num_frames, total_train_indices)
    if frame_count <= 0:
        logger.warning("No training frames available for comparison")
        return None
    target_frame_idx = frame_idx if frame_idx is not None and frame_idx < frame_count else min(cam_id * 2, frame_count - 1)
    
    logger.info(f"Using frame {target_frame_idx} for comparison")
    
    try:
        # Get image data from dataset
        if hasattr(dataset, 'full_image_set') and dataset.full_image_set is not None:
            img_index = int(target_frame_idx * total_train_indices / max(frame_count, 1))
            img_index = min(img_index, total_train_indices - 1)
            
            image_infos, cam_infos = dataset.full_image_set.get_image(img_index, camera_downscale=1.0)
        else:
            logger.warning("No full_image_set available for comparison")
            return None
        
        if 'pixels' not in image_infos or 'road_masks' not in image_infos:
            logger.warning("Missing required data (pixels/road_masks) for comparison")
            return None
        
        rgb_images = image_infos['pixels']
        road_masks = image_infos.get('road_masks')
        K = cam_infos.get('intrinsics', None)
        c2w = cam_infos.get('camera_to_world', None)
        
        # Handle multi-camera format
        is_multi_camera = torch.is_tensor(rgb_images) and len(rgb_images.shape) == 4 or (
            isinstance(rgb_images, np.ndarray) and rgb_images.ndim == 4)
        
        if not is_multi_camera:
            rgb_list = [rgb_images]
            road_mask_list = [road_masks] if road_masks is not None else []
        else:
            rgb_list = torch.unbind(rgb_images, dim=0) if torch.is_tensor(rgb_images) else list(rgb_images)
            road_mask_list = (torch.unbind(road_masks, dim=0) if (torch.is_tensor(road_masks)) and road_masks is not None 
                             else [None] * len(rgb_list))
        
        # Get the specified camera view
        cam_id_clamped = min(cam_id, len(rgb_list) - 1)
        rgb_img = rgb_list[cam_id_clamped]
        road_mask = road_mask_list[cam_id_clamped] if cam_id_clamped < len(road_mask_list) else None
        
        # Convert tensors to numpy on CPU before processing (fixes CUDA->numpy conversion error)
        if torch.is_tensor(rgb_img):
            rgb_img = rgb_img.cpu().numpy()
        
        # Handle road mask similarly - convert tensor to numpy on CPU first  
        if road_mask is not None and torch.is_tensor(road_mask):
            road_mask = road_mask.cpu().numpy()
        
        # Get camera parameters for this view
        K_cam = K
        c2w_cam = c2w
        
        if torch.is_tensor(K_cam):
            K_np = K_cam.cpu().numpy()
        elif isinstance(K_cam, np.ndarray) or (hasattr(K_cam, '__getitem__') and len(np.shape(K_cam)) == 2):
            K_np = np.array(K_cam)
        else:
            logger.warning(f"Invalid intrinsics format: {type(K_cam)}")
            return None
        
        if torch.is_tensor(c2w_cam):
            c2w_np = c2w_cam.cpu().numpy()
        elif isinstance(c2w_cam, np.ndarray) or (hasattr(c2w_cam, '__getitem__') and len(np.shape(c2w_cam)) >= 2):
            c2w_np = np.array(c2w_cam)
            if len(c2w_np.shape) == 3:
                c2w_np = c2w_np[0] if isinstance(cam_id_clamped, int) else c2w_np
        else:
            logger.warning(f"Invalid extrinsics format: {type(c2w_cam)}")
            return None
        
        # Create side-by-side comparison figure
        fig = plt.figure(figsize=(16, 8))
        
        # Left panel: RGB camera view with road mask overlay
        ax_rgb = fig.add_subplot(131)
        rgb_display = np.clip(rgb_img * 255, 0, 255).astype(np.uint8) if (rgb_img.max() <= 1.0 and len(rgb_img.shape) == 3) else rgb_img
        ax_rgb.imshow(rgb_display)
        
        # Overlay road mask in semi-transparent red
        if road_mask is not None:
            rm = np.array(road_mask)
            if torch.is_tensor(rm):
                rm = rm.cpu().numpy()
            overlay = np.zeros_like(rgb_display, dtype=np.uint8)
            overlay[rm > 0.5] = [255, 0, 0]
            alpha = 0.4
            blended = (rgb_display * (1 - alpha) + overlay * alpha).astype(np.uint8)
            ax_rgb.imshow(blended, alpha=alpha if road_mask.max() > 0 else 0)
        
        # Get camera position from extrinsics
        cam_pos_world = c2w_np[:3, 3] if len(c2w_np.shape) >= 2 and c2w_np.shape[0] == 4 else [0, 0, -1.6]
        ax_rgb.set_title(f"RGB View (Camera {cam_id_clamped})\nPos: [{cam_pos_world[0]:+.2f}, {cam_pos_world[1]:+.2f}, {cam_pos_world[2]:+,.2f}m]", 
                        fontsize=14, fontweight='bold')
        ax_rgb.set_xlabel('Width (pixels)', fontsize=12)
        ax_rgb.set_ylabel('Height (pixels)', fontsize=12)
        
        # Right panel: Projected mesh view from camera perspective (2D projection)
        ax_proj = fig.add_subplot(133)
        
        # Project mesh vertices to 2D pixel space using camera intrinsics
        verts_homogeneous = np.concatenate([verts_np, np.ones((len(verts_np), 1))], axis=1)  # (N, 4)
        
        # Transform from world to camera coordinates: p_cam = R @ p_world + t
        cam_to_world = c2w_np if len(c2w_np.shape) == 4 and c2w_np.shape[0] >= 1 else np.eye(4)
        rot_matrix = cam_to_world[:3, :3]
        trans_vector = cam_to_world[:3, 3]
        
        # World to camera transform
        R_cam_from_world = rot_matrix.T
        t_cam = -R_cam_from_world @ trans_vector
        p_camera = (R_cam_from_world @ verts_homogeneous[:, :3].T).T + t_cam  # (N, 3)
        
        # Project to pixel coordinates: u = K * [x/z, y/z, 1]
        pixels_2d = np.dot(K_np[:2], p_camera.T) / (p_camera[:, 2] + 1e-6)  # (N, 2)
        
        # Draw projected triangles on image plane with bounds checking
        valid_triangles = 0
        skipped_invalid = 0
        
        for i in range(len(faces_np)):
            v0_idx, v1_idx, v2_idx = faces_np[i]
            
            # Bounds check: ensure all vertex indices are within the projected vertices array
            if (v0_idx >= len(pixels_2d) or v1_idx >= len(pixels_2d) or v2_idx >= len(pixels_2d)):
                skipped_invalid += 1
                continue
            
            tri_pixels = pixels_2d[[v0_idx, v1_idx, v2_idx]]
            
            # Check if triangle is visible (facing camera)
            z_vals = p_camera[:, 2][[v0_idx, v1_idx, v2_idx]]
            avg_z = np.mean(z_vals)
            if avg_z > 0.5:  # Only draw triangles in front of camera
                ax_proj.plot(tri_pixels[:, 0], tri_pixels[:, 1], 'b-', linewidth=0.3, alpha=0.6)
                valid_triangles += 1
        
        logger.info(f"Projected {valid_triangles} valid triangles (skipped {skipped_invalid} with invalid vertex indices)")
        x_min, y_min, z_min = verts_np.min(axis=0)
        x_max, y_max, _ = verts_np.max(axis=0)
        max_range = np.array([x_max-x_min, y_max-y_min]).max() / 2.0 * 1.5
        
        ax_proj.set_xlim(0, rgb_img.shape[1])
        ax_proj.set_ylim(rgb_img.shape[0], 0)
        
        # Set axis labels
        ax_proj.set_xlabel('Width (pixels)', fontsize=12)
        ax_proj.set_ylabel('Height (pixels)', fontsize=12)
        
        # Calculate mid points for combined view projection (needed by code below)
        x_min, y_min, z_min = verts_np.min(axis=0)
        x_max, y_max, _ = verts_np.max(axis=0)
        max_range = np.array([x_max-x_min, y_max-y_min]).max() / 2.0 * 1.5
        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max + y_min) * 0.5
                
        # Calculate mid points for combined view projection
        x_min, y_min, z_min = verts_np.min(axis=0)
        x_max, y_max, _ = verts_np.max(axis=0)
        max_range = np.array([x_max-x_min, y_max-y_min]).max() / 2.0 * 1.5
        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max + y_min) * 0.5
        
        ax_proj.set_title(f"Projected Mesh from Camera {cam_id_clamped}\nSame Pose as RGB View", 
                       fontsize=14, fontweight='bold')
        
        # Center panel: Combined view showing both for comparison
        ax_combined = fig.add_subplot(132)
        combined_display = np.zeros((max(rgb_img.shape[0], 500), rgb_img.shape[1] * 2 + 100, 3), dtype=np.uint8) if len(rgb_img.shape) == 3 else None
        
        # Create a simple side-by-side composite for quick comparison
        left_half = np.zeros((max(400, rgb_img.shape[0]), 500, 3), dtype=np.uint8)
        right_half = np.zeros((max(400, rgb_img.shape[0]), 500, 3), dtype=np.uint8)
        
        # Left: RGB image scaled to fit
        if len(rgb_img.shape) == 3:
            rgb_scaled = (rgb_img * 255).astype(np.uint8)[:400, :500]
            left_half[:rgb_scaled.shape[0], :rgb_scaled.shape[1]] = rgb_scaled
        
        # Right: Mesh wireframe overlay on heightmap
        z_vals = verts_np[:, 2].reshape(int(np.sqrt(len(verts_np))), int(np.sqrt(len(verts_np))))
        x_vals = np.linspace(x_min, x_max, len(z_vals))[:int(np.sqrt(len(verts_np)))]
        y_vals = np.linspace(y_min, y_max, z_vals.shape[1])
        
        # Create a simple mesh visualization
        for i in range(min(50, len(faces_np))):
            v0_idx, v1_idx, v2_idx = faces_np[i]
            tri_verts = verts_np[[v0_idx, v1_idx, v2_idx]]
            if len(tri_verts) == 3:
                # Project to 2D for display
                x_proj = (tri_verts[:, 0] - mid_x + max_range) / (max_range * 2) * 450 + 25
                y_proj = (-tri_verts[:, 1] + mid_y + max_range) / (max_range * 2) * 450 + 25
                if all(0 <= x < 500 and 0 <= y < 400 for x, y in zip(x_proj, y_proj)):
                    right_half[y_proj[1]:y_proj[2], x_proj[1]:x_proj[2]] = [200, 200, 255]
        
        combined_display[:, :500] = left_half
        combined_display[:, 600:1100] = right_half
        ax_combined.imshow(combined_display)
        ax_combined.set_title("Quick Comparison View", fontsize=14, fontweight='bold')
        ax_combined.axis('off')
        
        plt.tight_layout()
        
        # Save comparison image
        if output_dir:
            comp_path = os.path.join(output_dir, f"camera_{cam_id_clamped}_rgb_vs_mesh_comparison.png")
            plt.savefig(comp_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved camera view comparison to {comp_path}")
        else:
            comp_path = None
        
        plt.close(fig)
        return comp_path
    
    except Exception as e:
        logger.error(f"Error creating camera view comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_mesh_render_from_camera(
    dataset,
    vertices_tensor,
    faces_tensor,
    frame_idx=0,
    cam_id=0,
    output_dir="",
    texture_buffer=None,
    texture_metadata=None,
):
    """Export a real offscreen mesh render from the camera pose of a specific frame."""
    logger.info(f"\nExporting standalone mesh render for frame {frame_idx}, camera {cam_id}...")

    try:
        os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        import open3d as o3d
    except Exception as e:
        logger.warning(f"Cannot export camera render - open3d unavailable: {e}")
        return None

    verts_np = vertices_tensor.cpu().numpy() if isinstance(vertices_tensor, torch.Tensor) else np.array(vertices_tensor)
    faces_np = faces_tensor.cpu().numpy() if isinstance(faces_tensor, torch.Tensor) else np.array(faces_tensor)

    if len(verts_np) == 0 or len(faces_np) == 0:
        logger.warning("Cannot export camera render - empty mesh")
        return None

    total_train_indices = len(dataset.train_indices) if hasattr(dataset, 'train_indices') else 0
    if total_train_indices <= 0 or not hasattr(dataset, 'full_image_set') or dataset.full_image_set is None:
        logger.warning("Cannot export camera render - dataset camera frames unavailable")
        return None

    img_index = min(max(int(frame_idx), 0), total_train_indices - 1)
    image_infos, cam_infos = dataset.full_image_set.get_image(img_index, camera_downscale=1.0)

    rgb_images = image_infos.get("pixels")
    if rgb_images is None:
        logger.warning("Cannot export camera render - missing RGB image for frame 0")
        return None

    if torch.is_tensor(rgb_images) and len(rgb_images.shape) == 4:
        rgb_img = rgb_images[min(cam_id, rgb_images.shape[0] - 1)]
    elif isinstance(rgb_images, np.ndarray) and rgb_images.ndim == 4:
        rgb_img = rgb_images[min(cam_id, rgb_images.shape[0] - 1)]
    else:
        rgb_img = rgb_images

    if torch.is_tensor(rgb_img):
        rgb_img = rgb_img.detach().cpu().numpy()
    else:
        rgb_img = np.asarray(rgb_img)

    K = cam_infos.get("intrinsics")
    c2w = cam_infos.get("camera_to_world")
    if K is None or c2w is None:
        logger.warning("Cannot export camera render - missing camera intrinsics/extrinsics")
        return None

    if torch.is_tensor(K):
        K_np = K.detach().cpu().numpy()
    else:
        K_np = np.asarray(K)
    if torch.is_tensor(c2w):
        c2w_np = c2w.detach().cpu().numpy()
    else:
        c2w_np = np.asarray(c2w)
    if len(c2w_np.shape) == 3:
        c2w_np = c2w_np[min(cam_id, c2w_np.shape[0] - 1)]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_np.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces_np.astype(np.int32))
    mesh.compute_vertex_normals()

    texture_image = None
    if texture_buffer is not None and texture_metadata:
        texture_np = np.clip(np.flipud(np.asarray(texture_buffer)) * 255.0, 0, 255).astype(np.uint8)
        texture_image = o3d.geometry.Image(texture_np)
        x_min, x_max = texture_metadata["x_range"]
        y_min, y_max = texture_metadata["y_range"]
        x_span = max(float(x_max) - float(x_min), 1e-8)
        y_span = max(float(y_max) - float(y_min), 1e-8)
        u = np.clip((verts_np[:, 0] - float(x_min)) / x_span, 0.0, 1.0)
        v = np.clip((verts_np[:, 1] - float(y_min)) / y_span, 0.0, 1.0)
        triangle_uvs = np.stack([np.column_stack([u, v])[faces_np[:, 0]],
                                 np.column_stack([u, v])[faces_np[:, 1]],
                                 np.column_stack([u, v])[faces_np[:, 2]]], axis=1).reshape(-1, 2)
        mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs.astype(np.float64))
        mesh.textures = [texture_image]
    else:
        color_np = plt.cm.viridis((verts_np[:, 2] - verts_np[:, 2].min()) / max(verts_np[:, 2].ptp(), 1e-6))[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(color_np, 0.0, 1.0).astype(np.float64))

    h, w = rgb_img.shape[:2]
    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.sRGB_color = True
    if texture_image is not None:
        material.albedo_img = texture_image
    renderer.scene.add_geometry("road_mesh", mesh, material)

    fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
    cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    extrinsic = np.linalg.inv(c2w_np)

    try:
        renderer.setup_camera(intrinsic, extrinsic)
    except Exception:
        eye = c2w_np[:3, 3]
        forward = c2w_np[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up = c2w_np[:3, :3] @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
        if np.linalg.norm(forward) < 1e-8:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.linalg.norm(up) < 1e-8:
            up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        renderer.scene.camera.look_at(eye + forward, eye, up)

    color = np.asarray(renderer.render_to_image())
    render_path = None
    if output_dir:
        render_path = os.path.join(output_dir, f"frame_{frame_idx}_camera_{cam_id}_mesh_render.png")
        try:
            from PIL import Image

            Image.fromarray(np.clip(color[:, :, :3], 0, 255).astype(np.uint8)).save(render_path)
            logger.info(f"Saved mesh render to {render_path}")
        except Exception as e:
            logger.error(f"Failed to save mesh render: {e}")
            return None

    return render_path


def log_rgb_projection_visualization(buffer: np.ndarray, 
                                      stats: dict, 
                                      output_dir: str) -> None:
    """Create and save visualization of the projected RGB image buffer.
    
    Args:
        buffer: H x W x 3 numpy array with float values in [0, 1] range
        stats: Dictionary containing projection statistics
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved visualization file
    """
    
    if len(buffer) == 0 or len(stats) == 0:
        logger.warning("Cannot visualize empty projected buffer")
        return
    
    width = stats.get('width', buffer.shape[1])
    height = stats.get('height', buffer.shape[0])
    pixels_updated = stats.get('road_pixels_projected', 'N/A')
    
    # Create visualization with multiple views
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # View 1: Full projected RGB buffer (top-down view of texture map)
    ax_full = axes[0, 0]
    im_full = ax_full.imshow(buffer, extent=[-width/2, width/2, -height/2, height/2], origin='lower')
    ax_full.set_xlabel('Width (pixels)', fontsize=12)
    ax_full.set_ylabel('Height (pixels)', fontsize=12)
    ax_full.set_title(f"Projected RGB Buffer ({width}×{height})", fontsize=14, fontweight='bold')
    plt.colorbar(im_full, ax=ax_full, label='RGB Intensity', shrink=0.8)
    
    # View 2: Zoomed center region (first quadrant for clarity)
    zoom_x = min(width // 4, max(50, width // 10))
    zoom_y = min(height // 4, max(50, height // 10))
    ax_zoom = axes[0, 1]
    im_zoom = ax_zoom.imshow(buffer[:zoom_y, :zoom_x], origin='lower', 
                            extent=[-width/2, -width/2+zoom_x, -height/2, -height/2+zoom_y])
    ax_zoom.set_xlabel('Width (pixels)', fontsize=12)
    ax_zoom.set_ylabel('Height (pixels)', fontsize=12)
    ax_zoom.set_title(f"Center Region ({zoom_x}×{zoom_y})", fontsize=14, fontweight='bold')
    
    # View 3: RGB channel breakdown - Red channel
    ax_r = axes[0, 2]
    im_r = ax_r.imshow(buffer[:, :, 0], cmap='Reds', origin='lower')
    ax_r.set_xlabel('Width (pixels)', fontsize=12)
    ax_r.set_ylabel('Height (pixels)', fontsize=12)
    ax_r.set_title("Red Channel", fontsize=14, fontweight='bold')
    
    # View 4: RGB channel breakdown - Green channel  
    ax_g = axes[1, 0]
    im_g = ax_g.imshow(buffer[:, :, 1], cmap='Greens', origin='lower')
    ax_g.set_xlabel('Width (pixels)', fontsize=12)
    ax_g.set_ylabel('Height (pixels)', fontsize=12)
    ax_g.set_title("Green Channel", fontsize=14, fontweight='bold')
    
    # View 5: RGB channel breakdown - Blue channel
    ax_b = axes[1, 1]
    im_b = ax_b.imshow(buffer[:, :, 2], cmap='Blues', origin='lower')
    ax_b.set_xlabel('Width (pixels)', fontsize=12)
    ax_b.set_ylabel('Height (pixels)', fontsize=12)
    ax_b.set_title("Blue Channel", fontsize=14, fontweight='bold')
    
    # View 6: Updated pixel density heatmap
    updated_mask = np.any(buffer > 0.01, axis=-1).astype(float)
    ax_density = axes[1, 2]
    im_density = ax_density.imshow(updated_mask, cmap='viridis', origin='lower')
    ax_density.set_xlabel('Width (pixels)', fontsize=12)
    ax_density.set_ylabel('Height (pixels)', fontsize=12)
    ax_density.set_title(f"Updated Pixels ({pixels_updated:,})", 
                        fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization with high DPI for publication quality
    viz_path = os.path.join(output_dir, "rgb_projection_visualization.png") if output_dir else None
    
    try:
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved RGB projection visualization to {viz_path}")
        
        # Also save a lower resolution version for quick viewing
        viz_path_lowres = os.path.join(output_dir, "rgb_projection_preview.png") if output_dir else None
        plt.savefig(viz_path_lowres, dpi=72, bbox_inches='tight')
        logger.info(f"Saved preview visualization to {viz_path_lowres}")
    except Exception as e:
        logger.error(f"Error saving RGB projection visualization: {e}")
    
    plt.close(fig)


def main():
    
    # Parse command line arguments
    args = parse_args()
    
    # Setup output directories and logging
    base_path, step1_dir = setup_output_directories(args.output_folder, args.run_name)
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
    
    # Step 1.4: Log bird's eye view of point cloud with camera locations
    try:
        log_birds_eye_view(pts_xyz[:, :3], pts_xyz[:, 3:], step1_dir)
        
        # Save a pruned/decimated version for efficient storage (especially useful for large point clouds)
        save_pruned_pointcloud_image(pts_xyz[:, :3], pts_xyz[:, 3:], step1_dir)
        
        # Log detailed statistics to console and file
        log_point_cloud_statistics(pts_xyz[:, :3])
    except Exception as e:
        logger.error(f"Error during visualization logging: {e}")

    # Step 1.45: Create camera locations plot showing all camera positions with point cloud
    try:
        create_camera_locations_plot(dataset, step1_dir)
        
        # Also create combined view of cameras + road points in same plot
        cam_positions = None
        if hasattr(dataset, 'full_image_set') and dataset.full_image_set is not None:
            total_train_indices = len(dataset.train_indices) if hasattr(dataset, 'train_indices') else 10
            
            for i in range(min(5, max(1, total_train_indices // 2))):
                img_index = int(i * len(dataset.train_indices) / max(total_train_indices, 1)) if total_train_indices > 0 else 0
                
                try:
                    image_infos, cam_infos = dataset.full_image_set.get_image(img_index, camera_downscale=1.0)
                    c2w = cam_infos.get('camera_to_world', None)
                    
                    if c2w is not None:
                        if torch.is_tensor(c2w):
                            c2w_np = c2w.cpu().numpy()
                        else:
                            c2w_np = np.array(c2w)
                        
                        # Extract camera positions from extrinsics matrix
                        if len(c2w_np.shape) == 3:  # Multi-camera frame
                            for j in range(min(5, c2w_np.shape[0])):
                                cam_positions.append(c2w_np[j][:3, 3]) if 'cam_positions' not in locals() else None
                        elif len(c2w_np.shape) == 2:  # Single camera view
                            cam_positions = np.array([c2w_np[:3, 3]]) if 'cam_positions' is None else np.vstack([cam_positions, c2w_np[:3, 3]])
                except Exception as e:
                    logger.debug(f"Could not extract camera position from frame {i}: {e}")
            
            # Create combined visualization with cameras and road points in same plot
            if cam_positions is not None and len(cam_positions) > 0:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # X-Y plane (bird's eye view - top down) WITH CAMERAS
                ax_xy = axes[0]
                scatter_pts = ax_xy.scatter(pts_xyz[:, 0], pts_xyz[:, 1], c=range(len(pts_xyz)), 
                                           cmap='viridis', s=1, alpha=0.5, label='Road Points')
                
                # Plot camera positions in red X markers
                cam_array = np.array(cam_positions) if not isinstance(cam_positions, np.ndarray) else cam_positions
                scatter_cams = ax_xy.scatter(cam_array[:, 0], cam_array[:, 1], 
                                             c='red', s=80, marker='X', linewidths=2,
                                             edgecolors='white', alpha=0.9, label=f'Cameras ({len(cam_positions)} positions)')
                ax_xy.legend(loc='upper right', fontsize=10)
                
                ax_xy.set_xlabel('X (meters)', fontsize=12)
                ax_xy.set_ylabel('Y (meters)', fontsize=12)
                title = f"Bird's Eye View - {len(pts_xyz):,} Road Points + Camera Locations"
                ax_xy.set_title(title, fontsize=14, fontweight='bold')
                ax_xy.grid(True, alpha=0.3)
                plt.colorbar(scatter_pts, ax=ax_xy, label='Point Index', shrink=0.8)
                
                # X-Z plane (side view - looking from side) WITH CAMERAS
                ax_xz = axes[1]
                scatter2_pts = ax_xz.scatter(pts_xyz[:, 0], pts_xyz[:, 2], c=range(len(pts_xyz)), 
                                             cmap='viridis', s=1, alpha=0.5, label='Road Points')
                
                # Plot camera positions in red X markers  
                scatter_cams_z = ax_xz.scatter(cam_array[:, 0], cam_array[:, 2], 
                                               c='red', s=80, marker='X', linewidths=2,
                                               edgecolors='white', alpha=0.9, label=f'Cameras ({len(cam_positions)} positions)')
                ax_xz.legend(loc='upper right', fontsize=10)
                
                ax_xz.set_xlabel('X (meters)', fontsize=12)
                ax_xz.set_ylabel('Z height (meters)', fontsize=12)
                title_z = "Side View - X-Z Plane"
                if len(cam_positions) > 0:
                    title_z += " + Cameras"
                ax_xz.set_title(title_z, fontsize=14, fontweight='bold')
                ax_xz.grid(True, alpha=0.3)
                
                # Y-Z plane (front view - looking from front/side) WITH CAMERAS  
                ax_yz = axes[2]
                scatter3_pts = ax_yz.scatter(pts_xyz[:, 1], pts_xyz[:, 2], c=range(len(pts_xyz)), 
                                             cmap='viridis', s=1, alpha=0.5, label='Road Points')
                
                # Plot camera positions in red X markers
                scatter_cams_front = ax_yz.scatter(cam_array[:, 1], cam_array[:, 2], 
                                                   c='red', s=80, marker='X', linewidths=2,
                                                   edgecolors='white', alpha=0.9, label=f'Cameras ({len(cam_positions)} positions)')
                ax_yz.legend(loc='upper right', fontsize=10)
                
                ax_yz.set_xlabel('Y (meters)', fontsize=12)
                ax_yz.set_ylabel('Z height (meters)', fontsize=12)
                title_front = "Front View - Y-Z Plane"
                if len(cam_positions) > 0:
                    title_front += " + Cameras"
                ax_yz.set_title(title_front, fontsize=14, fontweight='bold')
                ax_yz.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save combined visualization with high DPI for publication quality
                viz_path = os.path.join(step1_dir, "combined_pointcloud_cameras.png")
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved combined point cloud + camera locations plot to {viz_path}")
                
    except Exception as e:
        logger.warning(f"Could not create combined visualization with cameras (non-critical): {e}")
    
    # Step 1.5: Export point cloud to PLY format for external inspection
    try:
        export_point_cloud(road_pts.cpu().numpy(), road_colors.cpu().numpy() if len(road_colors) > 0 else None, step1_dir)
        
    except Exception as e:
        logger.error(f"Error during PLY export: {e}")
    
    # Step 2: Create a 3D mesh from the aggregated point cloud (NO Z-AXIS OVERLAP!)
    # IMPORTANT: This uses PRUNED lidar data - only includes points remaining after 
    # dataset.project_lidar_pts_on_images() removes out-of-view points during initialization.
    try:
        vertices_tensor, faces_tensor, vertex_colors = create_and_log_mesh(
            pts_xyz[:, :3], 
            pts_xyz[:, 3:] if len(pts_xyz) > 0 and pts_xyz.shape[1] >= 6 else None,
            step1_dir
        )
        
    except Exception as e:
        logger.error(f"Error during mesh creation (Step 2): {e}")
    
    # Step 2.5: Export mesh to OBJ format for external inspection and compatibility with other software
    try:
        export_mesh_to_obj(
            vertices_tensor, 
            faces_tensor, 
            vertex_colors if len(vertex_colors) > 0 else None,
            step1_dir
        )
        
    except Exception as e:
        logger.error(f"Error during OBJ mesh export (Step 2.5): {e}")
    
    # Step 3: Create image buffer for texture mapping (USER REQUIREMENT IMPLEMENTED HERE)
    try:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3 - IMAGE BUFFER CREATION")
        logger.info("=" * 60)
        
        # Create the image buffer using point cloud dimensions as reference
        image_buffer, buffer_metadata = create_image_buffer(
            vertices_tensor, 
            faces_tensor, 
            vertex_colors if len(vertex_colors) > 0 else None
        )
        
        # Log visualization of the image buffer
        log_image_buffer_visualization(image_buffer, buffer_metadata, step1_dir)
        
        # Save image buffer as PNG for external inspection and later use in Step 4
        png_path = os.path.join(step1_dir, "road_texture_map.png") if step1_dir else None
        
        try:
            save_image_buffer_as_png(
                image_buffer, 
                buffer_metadata, 
                png_path
            )
            
            # Also save the raw numpy array for programmatic access in Step 4
            npz_path = os.path.join(step1_dir, "image_buffer.npz") if step1_dir else None
            
            try:
                np.savez_compressed(
                    npz_path, 
                    buffer=image_buffer, 
                    **buffer_metadata
                )
                logger.info(f"Saved image buffer data to {npz_path}")
                
            except Exception as save_error:
                logger.warning(f"Could not save .npz file (non-critical): {save_error}")
            
        except Exception as png_save_error:
            logger.error(f"Error saving PNG visualization: {png_save_error}")
        
        # Log buffer statistics to console and file
        if len(buffer_metadata) > 0:
            width = buffer_metadata.get('width', 'N/A')
            height = buffer_metadata.get('height', 'N/A')
            pixels_per_meter = buffer_metadata.get('pixels_per_meter', 'N/A')
            
            logger.info(f"\nImage Buffer Statistics:")
            logger.info(f"  - Resolution: {width} × {height}")
            logger.info(f"  - Pixel density: {pixels_per_meter}px/meter")
            logger.info(f"  - Scene coverage: {buffer_metadata.get('scene_width_meters', 0):.2f}m × {buffer_metadata.get('scene_height_meters', 0):.2f}m")
            
        # Save buffer metadata to JSON for later use in Step 4
        import json
            
    except Exception as e:
        logger.error(f"Error during image buffer creation (Step 3): {e}")
    
    # Before Step 4: Create camera view verification (side-by-side RGB vs mesh render)
    try:
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION - Camera View Comparison")
        logger.info("=" * 60)
        
        comparison_path = create_camera_view_comparison(
            dataset, 
            vertices_tensor, 
            faces_tensor, 
            cam_id=2,  # Use middle camera view (index 2 is typically the third/center view in multi-camera setups)
            output_dir=step1_dir,
            num_frames=args.num_projection_frames if args.num_projection_frames is not None else (len(dataset.train_indices) if hasattr(dataset, 'train_indices') else None)
        )
        
        if comparison_path and os.path.exists(comparison_path):
            logger.info(f"Camera verification image saved to: {comparison_path}")
    except Exception as e:
        logger.warning(f"Could not create camera view comparison (non-critical): {e}")

    
    # Before Step 4: Create camera view verification (side-by-side RGB vs mesh render)
    try:
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION - Camera View Comparison")
        logger.info("=" * 60)
        
        comparison_path = create_camera_view_comparison(
            dataset, 
            vertices_tensor, 
            faces_tensor, 
            cam_id=2,  # Use middle camera view (index 2 is typically the third/center view in multi-camera setups)
            output_dir=step1_dir,
            num_frames=args.num_projection_frames if args.num_projection_frames is not None else (len(dataset.train_indices) if hasattr(dataset, 'train_indices') else None)
        )
        
        if comparison_path and os.path.exists(comparison_path):
            logger.info(f"Camera verification image saved to: {comparison_path}")
    except Exception as e:
        logger.warning(f"Could not create camera view comparison (non-critical): {e}")

    # Step 4: Project RGB images onto the mesh using ray casting from camera positions
    try:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4 - RGB IMAGE PROJECTION")
        logger.info("=" * 60)
        
        # Perform ray projection of road-masked RGB pixels onto image buffer
        projection_fn = project_rgb_onto_image_buffer_gpu if device.type == "cuda" else project_rgb_onto_image_buffer
        updated_buffer, projection_stats = projection_fn(
            dataset, 
            vertices_tensor, 
            faces_tensor, 
            image_buffer, 
            buffer_metadata,
            device,
            step1_dir,
            num_frames=args.num_projection_frames,
        )
        
        # Log visualization of the projected result
        log_rgb_projection_visualization(updated_buffer, {**buffer_metadata, **projection_stats}, step1_dir)
        
        # Save updated texture map with RGB projections
        rgb_texture_path = os.path.join(step1_dir, "road_textured_mesh.png") if step1_dir else None
        
        try:
            save_image_buffer_as_png(
                updated_buffer, 
                buffer_metadata, 
                rgb_texture_path
            )
            
            # Save projection statistics for analysis
            stats_json_path = os.path.join(step1_dir, "projection_stats.json") if step1_dir else None
            
            try:
                with open(stats_json_path, 'w') as f:
                    json.dump({
                        **projection_stats,
                        'width': buffer_metadata.get('width'),
                        'height': buffer_metadata.get('height'),
                        'pixels_per_meter': buffer_metadata.get('pixels_per_meter')
                    }, f, indent=2)
                logger.info(f"Saved projection statistics to {stats_json_path}")
                
            except Exception as stats_error:
                logger.warning(f"Could not save JSON stats (non-critical): {stats_error}")
            
        except Exception as rgb_save_error:
            logger.error(f"Error saving RGB texture map: {rgb_save_error}")

        reprojection_stats = {}
        # Reproject the final image buffer back onto the mesh surface.
        try:
            textured_vertex_colors, reprojection_stats = reproject_image_buffer_onto_mesh(
                vertices_tensor,
                faces_tensor,
                updated_buffer,
                buffer_metadata,
                step1_dir,
            )
            if textured_vertex_colors is not None:
                logger.info(
                    f"Reprojected buffer onto mesh: {reprojection_stats.get('mesh_vertices_textured', 0):,} "
                    f"of {reprojection_stats.get('mesh_vertices_reprojected', 0):,} vertices textured"
                )
        except Exception as mesh_reprojection_error:
            logger.error(f"Error reprojecting buffer back onto mesh: {mesh_reprojection_error}")

        # Export a true offscreen mesh render from the frame 0 camera pose.
        try:
            frame0_render_path = export_mesh_render_from_camera(
                dataset,
                vertices_tensor,
                faces_tensor,
                frame_idx=0,
                cam_id=0,
                output_dir=step1_dir,
                texture_buffer=updated_buffer,
                texture_metadata=buffer_metadata,
            )
            if frame0_render_path and os.path.exists(frame0_render_path):
                logger.info(f"Frame 0 mesh render saved to: {frame0_render_path}")
        except Exception as e:
            logger.warning(f"Could not export frame 0 mesh render (non-critical): {e}")

        if "stats_json_path" in locals() and stats_json_path:
            try:
                with open(stats_json_path, "w") as f:
                    json.dump(
                        {
                            **projection_stats,
                            **reprojection_stats,
                            "width": buffer_metadata.get("width"),
                            "height": buffer_metadata.get("height"),
                            "pixels_per_meter": buffer_metadata.get("pixels_per_meter"),
                        },
                        f,
                        indent=2,
                    )
                logger.info(f"Updated projection statistics to include mesh reprojection at {stats_json_path}")
            except Exception as stats_error:
                logger.warning(f"Could not update JSON stats with mesh reprojection data: {stats_error}")
        
        # Log projection statistics to console and file
        if len(projection_stats) > 0:
            total_cameras = projection_stats.get('total_cameras_processed', 'N/A')
            pixels_updated = projection_stats.get('road_pixels_projected', 'N/A')
            
            logger.info(f"\nRGB Projection Statistics:")
            logger.info(f"  - Total cameras processed: {total_cameras}")
            logger.info(f"  - Unique buffer pixels updated: {pixels_updated:,}")
            logger.info(f"  - Output file: road_textured_mesh.png")
        
        # Update image_buffer variable for final summary (now contains RGB projections)
        image_buffer = updated_buffer
            
    except Exception as e:
        logger.error(f"Error during RGB projection (Step 4): {e}")
    
    # Final summary logging
    logger.info("=" * 60)
    logger.info("STEPS 1-4 COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Output directory: {base_path}")
    logger.info(f"All files saved to: {step1_dir}")
    
    if vertices_tensor is not None and len(vertices_tensor) > 0:
        logger.info("")
        logger.info("STEP 2 - MESH CREATION COMPLETED:")
        logger.info(f"  - Mesh file: road_mesh.pth")
        logger.info(f"  - Vertices: {len(vertices_tensor):,}")
        logger.info(f"  - Faces (triangles): {len(faces_tensor):,}")
    
    if 'image_buffer' in locals() and image_buffer is not None:
        logger.info("")
        logger.info("STEP 3 & STEP 4 COMPLETED:")
        logger.info(f"  - Initial texture map: road_texture_map.png")
        logger.info(f"  - RGB-projected mesh: road_textured_mesh.png")
        logger.info(f"  - Reprojected textured mesh: road_textured_mesh.pth / .ply / .obj / .mtl")
        logger.info(f"  - Texture image: road_textured_mesh_texture.png")
        logger.info(f"  - Frame 0 mesh render: frame_0_camera_0_mesh_render.png")
        logger.info(f"  - Buffer dimensions: {buffer_metadata.get('width', 'N/A')} × {buffer_metadata.get('height', 'N/A')} pixels")
    
    logger.info("")
    logger.info("Generated files:")
    for filename in ['road_mesh.pth', 'road_pointcloud.ply', 'road_mesh.ply', 
                     'road_texture_map.png', 'road_textured_mesh.png',
                     'road_textured_mesh.pth', 'road_textured_mesh.ply', 'road_textured_mesh.obj',
                     'road_textured_mesh.mtl', 'road_textured_mesh_texture.png',
                     'frame_0_camera_0_mesh_render.png',
                     'road_textured_mesh_reprojection.png',
                     'image_buffer.npz', 'projection_stats.json']:
        filepath = os.path.join(step1_dir, filename) if step1_dir else None
        if filepath and os.path.exists(filepath):
            logger.info(f"  - {filename}")
    
    logger.info("")
    logger.info(f"  - Buffer dimensions: {buffer_metadata.get('width', 'N/A')} × {buffer_metadata.get('height', 'N/A')} pixels")
    
    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Step 4: Project RGB images onto the textured mesh using image buffer")


if __name__ == "__main__":
    main()
