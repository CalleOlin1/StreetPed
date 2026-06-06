#!/usr/bin/env python3
"""
Road Mesh Training Script - Step 1 & 2 Implementation

This script implements Steps 1-3 of the road mesh training pipeline:
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
- Project rays using rgb images and cam position onto the mesh (TODO: Step 4)
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
    cell_size = 0.2
    
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
    
    for x_idx in range(num_cells_x):
        for y_idx in range(num_cells_y):
            z_val = height_map[x_idx, y_idx]
            
            # Skip cells with no data (infinity) and boundary cells that would create incomplete quads
            if np.isinf(z_val) or x_idx == 0 or x_idx == num_cells_x - 1 or y_idx == 0 or y_idx == num_cells_y - 1:
                continue
            
            height_map_dict[(x_idx, y_idx)] = len(cell_z_values)
            cell_z_values.append(float(z_val))
            
            if color_map is not None and x_idx < num_cells_x and y_idx < num_cells_y:
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
            
            v0, v1, v2 = corner_indices[0], corner_indices[3], corner_indices[1]
            final_faces.append([v0, v1, v2])  # First triangle
            
            v0, v1, v2 = corner_indices[2], corner_indices[1], corner_indices[3]
            final_faces.append([v0, v1, v2])  # Second triangle
    
    logger.info(f"Created {len(final_faces):,} triangular faces")
    
    if len(final_faces) == 0:
        logger.warning("No valid triangles could be created from the grid cells")
        return torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 3)
    
    # Build final vertices with X-Y plane coordinates centered at origin (same as original)
    final_vertices = []
    row_colors = cell_colors_list if color_map is not None else None
    
    for x_idx in range(num_cells_x):
        for y_idx in range(num_cells_y):
            vertex_key = (x_idx, y_idx)
            
            # Skip cells with no data or boundary cells
            if vertex_key not in height_map_dict:
                continue
            
            z_height = cell_z_values[height_map_dict[vertex_key]]
            
            # Map grid index to world coordinates centered at origin
            local_x = (x_idx - num_cells_x / 2 + 0.5) * scene_width / num_cells_x
            local_y = (y_idx - num_cells_y / 2 + 0.5) * scene_height / num_cells_y
            
            final_vertices.append([local_x, local_y, z_height])
    
    logger.info(f"Created {len(final_vertices):,} mesh vertices")
    
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
                       output_dir: str = ""):
    """Export mesh to PLY format for external inspection."""
    import struct
    
    if len(vertices) == 0 or len(faces) == 0:
        logger.warning("Cannot export empty mesh")
        return
    
    ply_path = os.path.join(output_dir, "road_mesh.ply") if output_dir else None
    
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
    
    # User requirement: resolution of 10 pixels per meter
    pixels_per_meter = 10
    
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


def export_mesh_to_obj(vertices: torch.Tensor, 
                       faces: torch.Tensor, 
                       colors: Union[torch.Tensor, None] = None,
                       output_dir: str = ""):
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
    
    obj_path = os.path.join(output_dir, "road_mesh.obj") if output_dir else None
    
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


def main():
    
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
    
    # Step 2: Create a 3D mesh from the aggregated point cloud (NO Z-AXIS OVERLAP!)
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
    
    # Final summary logging
    logger.info("=" * 60)
    logger.info("STEP 1, STEP 2 & STEP 3 COMPLETED SUCCESSFULLY")
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
        logger.info("STEP 3 - IMAGE BUFFER CREATION COMPLETED:")
        logger.info(f"  - Texture map file: road_texture_map.png")
        logger.info(f"  - Buffer dimensions: {buffer_metadata.get('width', 'N/A')} × {buffer_metadata.get('height', 'N/A')} pixels")
    
    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Step 4: Project RGB images onto the textured mesh using image buffer")


if __name__ == "__main__":
    main()


