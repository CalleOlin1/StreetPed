"""
Road Mesh Module - Mesh-based road reconstruction for training and rendering.

This module implements a 3D mesh representation of the road surface that:
1. Is initialized from LiDAR point cloud data and road masks
2. Has a texture buffer that maps RGB image data onto the mesh
3. Can be rendered as part of the forward pass
4. Provides actual colored rendering (not just vertex colors)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class RoadMesh(nn.Module):
    """
    3D mesh representation of road surface with texture mapping.
    
    This class wraps mesh creation, texture buffer management, and rendering logic.
    The mesh is initialized from LiDAR data and road masks, then textured using
    RGB projections from camera images.
    """
    
    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        device: torch.device = torch.device("cuda"),
        cell_size: float = 1.0,
    ):
        """
        Initialize RoadMesh with mesh geometry.
        
        Args:
            vertices: M x 3 tensor of vertex positions in world space
            faces: F x 3 tensor of face indices (triangles)
            device: PyTorch device for computation
            cell_size: Grid cell size used during mesh creation (for reference)
        """
        super().__init__()
        
        self.device = device
        self.cell_size = cell_size
        
        # Register mesh as buffers (not trainable parameters)
        self.register_buffer("vertices", vertices.to(device))
        self.register_buffer("faces", faces.to(device))
        
        # Initialize texture buffer (will be populated during training)
        # Shape: (height, width, 3) for RGB texture
        self.texture_buffer = None
        self.texture_metadata = None
        
        # Compute mesh bounds for texture buffer initialization
        if len(vertices) > 0:
            self.bounds_min = vertices.min(dim=0)[0]
            self.bounds_max = vertices.max(dim=0)[0]
            self.scene_width = float((self.bounds_max[0] - self.bounds_min[0]).item())
            self.scene_height = float((self.bounds_max[1] - self.bounds_min[1]).item())
        else:
            self.bounds_min = torch.zeros(3, device=device)
            self.bounds_max = torch.zeros(3, device=device)
            self.scene_width = 0.0
            self.scene_height = 0.0
        
        logger.info(
            f"RoadMesh initialized: {len(vertices)} vertices, {len(faces)} faces"
        )
    
    @classmethod
    def from_pointcloud(
        cls,
        pts_xyz: np.ndarray,
        colors: Optional[np.ndarray] = None,
        cell_size: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ) -> "RoadMesh":
        """
        Create a RoadMesh from a point cloud using heightmap-based mesh creation.
        
        Args:
            pts_xyz: N x 3 numpy array of point coordinates (x, y, z) in meters
            colors: Optional N x 3 numpy array of RGB colors (0-1 or 0-255 range)
            cell_size: Grid cell size in meters
            device: PyTorch device for computation
            
        Returns:
            Initialized RoadMesh instance
        """
        vertices, faces, vertex_colors = _create_mesh_from_pointcloud(
            pts_xyz, colors, cell_size
        )
        
        mesh = cls(
            vertices=vertices,
            faces=faces,
            device=device,
            cell_size=cell_size,
        )
        
        return mesh
    
    def initialize_texture_buffer(
        self,
        resolution_pixels_per_meter: float = 10.0,
    ) -> None:
        """
        Initialize texture buffer for RGB projection.
        
        Args:
            resolution_pixels_per_meter: Texture resolution in pixels per meter
        """
        if self.scene_width <= 0 or self.scene_height <= 0:
            logger.warning("Cannot initialize texture buffer with invalid scene bounds")
            return
        
        # Calculate buffer dimensions based on scene size
        buffer_width = int(np.ceil(self.scene_width * resolution_pixels_per_meter))
        buffer_height = int(np.ceil(self.scene_height * resolution_pixels_per_meter))
        
        # Initialize with white/neutral color
        self.texture_buffer = np.ones(
            (buffer_height, buffer_width, 3), dtype=np.float32
        )
        
        self.texture_metadata = {
            "width": buffer_width,
            "height": buffer_height,
            "x_range": (float(self.bounds_min[0].item()), float(self.bounds_max[0].item())),
            "y_range": (float(self.bounds_min[1].item()), float(self.bounds_max[1].item())),
            "scene_width_meters": self.scene_width,
            "scene_height_meters": self.scene_height,
            "resolution_ppm": resolution_pixels_per_meter,
        }
        
        logger.info(
            f"Initialized texture buffer: {buffer_width}x{buffer_height} pixels "
            f"({self.scene_width:.2f}x{self.scene_height:.2f}m)"
        )
    
    def update_texture_buffer(self, new_buffer: np.ndarray) -> None:
        """
        Update the texture buffer with new RGB data.
        
        Args:
            new_buffer: Numpy array of shape (height, width, 3) with RGB values in [0, 1]
        """
        if self.texture_buffer is None:
            logger.warning("Texture buffer not initialized. Call initialize_texture_buffer first.")
            return
        
        # Ensure buffer has correct shape
        if new_buffer.shape != self.texture_buffer.shape:
            logger.warning(
                f"Buffer shape mismatch: expected {self.texture_buffer.shape}, "
                f"got {new_buffer.shape}"
            )
            return
        
        self.texture_buffer = np.clip(new_buffer, 0.0, 1.0)
    
    def get_texture_as_tensor(self) -> Optional[torch.Tensor]:
        """
        Get texture buffer as PyTorch tensor.
        
        Returns:
            Texture as (1, 3, height, width) tensor or None if not initialized
        """
        if self.texture_buffer is None:
            return None
        
        # Convert numpy (H, W, 3) to torch (1, 3, H, W) for rendering
        texture_np = np.transpose(self.texture_buffer, (2, 0, 1))  # (3, H, W)
        texture_tensor = torch.from_numpy(texture_np).float().unsqueeze(0)  # (1, 3, H, W)
        
        return texture_tensor.to(self.device)
    
    def project_images_onto_buffer(
        self,
        dataset,
        num_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Project RGB values from dataset images onto the texture buffer.
        
        This implements the ray-casting approach to map camera images onto the mesh
        surface through the texture buffer. Uses GPU parallelization when CUDA is available
        for significant performance improvement over the CPU version.
        
        Args:
            dataset: DrivingDataset instance with images and camera information
            num_frames: Number of training frames to use for projection (None = all)
            
        Returns:
            Dictionary with projection statistics
        """
        if self.texture_buffer is None:
            logger.warning("Texture buffer not initialized")
            return {}
        
        if len(self.vertices) == 0:
            logger.warning("Cannot project images - mesh is empty")
            return {}
        
        # Import both CPU and GPU projection functions from train_mesh_road
        from tools.train_mesh_road import project_rgb_onto_image_buffer, project_rgb_onto_image_buffer_gpu
        
        try:
            # Determine whether to use GPU or CPU version based on device availability
            use_gpu = torch.cuda.is_available() and self.device.type == "cuda"
            
            if use_gpu:
                logger.info("Using GPU-accelerated texture projection (batched ray tracing)")
                updated_buffer, stats = project_rgb_onto_image_buffer_gpu(
                    dataset=dataset,
                    vertices_tensor=self.vertices,
                    faces_tensor=self.faces,
                    image_buffer=self.texture_buffer.copy(),
                    buffer_metadata=self.texture_metadata,
                    device=self.device,
                    num_frames=num_frames,
                    ray_chunk_size=1024,
                    triangle_chunk_size=4096,
                )
            else:
                logger.info("Using CPU projection (GPU not available)")
                updated_buffer, stats = project_rgb_onto_image_buffer(
                    dataset=dataset,
                    vertices_tensor=self.vertices,
                    faces_tensor=self.faces,
                    image_buffer=self.texture_buffer.copy(),
                    buffer_metadata=self.texture_metadata,
                    device=self.device,
                    num_frames=num_frames,
                )
            
            self.texture_buffer = updated_buffer
            
            logger.info(
                f"Texture projection complete: {stats.get('road_pixels_projected', 0)} "
                f"pixels updated from {stats.get('total_cameras_processed', 0)} cameras"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during image projection: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_mesh_for_rendering(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get mesh components for rendering.
        
        Returns:
            Tuple of (vertices, faces, texture_tensor)
        """
        texture = self.get_texture_as_tensor()
        return self.vertices, self.faces, texture
    
    def forward(self) -> Dict[str, torch.Tensor]:
        """
        Forward pass - returns mesh components for rendering integration.
        
        Returns:
            Dictionary with mesh data for rendering
        """
        vertices, faces, texture = self.get_mesh_for_rendering()
        
        return {
            "vertices": vertices,
            "faces": faces,
            "texture": texture,
            "bounds_min": self.bounds_min,
            "bounds_max": self.bounds_max,
        }
    
    def to(self, device):
        """Move mesh to specified device."""
        super().to(device)
        self.device = device
        return self
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save mesh state to a checkpoint file.
        
        Persists vertices, faces, bounds, scene dimensions, and texture buffer.
        This enables resuming training with the same mesh across sessions.
        
        Args:
            filepath: Path where checkpoint should be saved
        """
        checkpoint = {
            "vertices": self.vertices.cpu(),
            "faces": self.faces.cpu(),
            "bounds_min": self.bounds_min.cpu(),
            "bounds_max": self.bounds_max.cpu(),
            "scene_width": self.scene_width,
            "scene_height": self.scene_height,
            "cell_size": self.cell_size,
            "texture_buffer": self.texture_buffer,
            "texture_metadata": self.texture_metadata,
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved road mesh checkpoint: {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: torch.device = torch.device("cuda")) -> "RoadMesh":
        """
        Load mesh state from a checkpoint file.
        
        Restores vertices, faces, bounds, and texture buffer from saved state.
        
        Args:
            filepath: Path to saved checkpoint
            device: PyTorch device for computation
            
        Returns:
            Restored RoadMesh instance
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        mesh = cls(
            vertices=checkpoint["vertices"].to(device),
            faces=checkpoint["faces"].to(device),
            device=device,
            cell_size=checkpoint.get("cell_size", 1.0),
        )
        
        # Restore bounds and scene dimensions
        mesh.bounds_min = checkpoint["bounds_min"].to(device)
        mesh.bounds_max = checkpoint["bounds_max"].to(device)
        mesh.scene_width = checkpoint["scene_width"]
        mesh.scene_height = checkpoint["scene_height"]
        
        # Restore texture if available
        if checkpoint["texture_buffer"] is not None:
            mesh.texture_buffer = checkpoint["texture_buffer"]
            mesh.texture_metadata = checkpoint["texture_metadata"]
        
        logger.info(f"Loaded road mesh checkpoint: {filepath}")
        return mesh


def _create_mesh_from_pointcloud(
    pts_xyz: np.ndarray,
    colors: Optional[np.ndarray] = None,
    cell_size: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a 3D mesh from point cloud using heightmap-based approach.
    
    This function creates a grid-based mesh where each cell in the X-Y plane maps to one Z value.
    
    Args:
        pts_xyz: N x 3 numpy array of point coordinates (x, y, z) in meters
        colors: Optional N x 3 numpy array of RGB colors
        cell_size: Grid cell size in meters
        
    Returns:
        Tuple of (vertices, faces, vertex_colors) as PyTorch tensors
    """
    
    if len(pts_xyz) == 0:
        logger.warning("Cannot create mesh from empty point cloud")
        return torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 3)
    
    points = pts_xyz if isinstance(pts_xyz, np.ndarray) else pts_xyz.cpu().numpy()
    colors_arr = colors.copy() if colors is not None and len(colors) > 0 else None
    
    logger.info(f"Creating mesh from {len(points):,} point cloud samples...")
    
    # Determine scene bounds
    x_min, y_min, z_min = points[:, 0].min(), points[:, 1].min(), points[:, 2].min()
    x_max, y_max, _ = points[:, 0].max(), points[:, 1].max(), points[:, 2].max()
    
    scene_width = max(x_max - x_min, 0.1)
    scene_height = max(y_max - y_min, 0.1)
    
    num_cells_x = int(np.ceil(scene_width / cell_size)) + 1
    num_cells_y = int(np.ceil(scene_height / cell_size)) + 1
    
    logger.info(f"Grid resolution: {num_cells_x} × {num_cells_y} cells (cell size: {cell_size}m)")
    
    # Create heightmap
    height_map = np.full((num_cells_x, num_cells_y), np.inf, dtype=np.float64)
    x_coord_map = np.full((num_cells_x, num_cells_y), np.nan, dtype=np.float64)
    y_coord_map = np.full((num_cells_x, num_cells_y), np.nan, dtype=np.float64)
    
    if colors_arr is not None and len(colors_arr) == len(points):
        color_map = [np.full((num_cells_x, num_cells_y), -1.0, dtype=np.int32) for _ in range(3)]
    else:
        color_map = None
    
    # Fill heightmap with minimum z values
    for i, point in enumerate(points):
        x_idx = int((point[0] - x_min) / scene_width * num_cells_x)
        y_idx = int((point[1] - y_min) / scene_height * num_cells_y)
        
        x_idx = np.clip(x_idx, 0, num_cells_x - 1)
        y_idx = np.clip(y_idx, 0, num_cells_y - 1)
        
        z_val = point[2]
        
        if z_val < height_map[x_idx, y_idx]:
            height_map[x_idx, y_idx] = z_val
            x_coord_map[x_idx, y_idx] = point[0]
            y_coord_map[x_idx, y_idx] = point[1]
            
            if colors_arr is not None:
                for c in range(3):
                    color_map[c][x_idx, y_idx] = int(colors_arr[i, c])
    
    num_valid_cells = np.sum(~np.isinf(height_map))
    logger.info(f"Created {num_valid_cells:,} valid grid cells from point cloud")
    
    if num_valid_cells < 4:
        logger.warning("Insufficient valid cells to create a meaningful mesh")
        return torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 3)
    
    # Build vertex and face lists
    height_map_dict = {}
    cell_z_values = []
    cell_colors_list = [] if color_map is not None else None
    
    # Identify valid cells
    valid_cells = []
    for x_idx in range(num_cells_x):
        for y_idx in range(num_cells_y):
            z_val = height_map[x_idx, y_idx]
            if not np.isinf(z_val):
                valid_cells.append((x_idx, y_idx))
    
    # Create vertex mapping
    for i, (x_idx, y_idx) in enumerate(valid_cells):
        height_map_dict[(x_idx, y_idx)] = len(cell_z_values)
        cell_z_values.append(float(height_map[x_idx, y_idx]))
        
        if color_map is not None:
            mean_color = np.array([color_map[c][x_idx, y_idx] for c in range(3)], dtype=np.float32) / 255.0
            cell_colors_list.append(mean_color)
    
    if len(cell_z_values) == 0:
        logger.warning("No valid cells after filtering")
        return torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 3)
    
    # Build faces
    final_faces = []
    for x_idx in range(num_cells_x - 1):
        for y_idx in range(num_cells_y - 1):
            corner_indices = [
                height_map_dict.get((x_idx, y_idx)),
                height_map_dict.get((x_idx + 1, y_idx)),
                height_map_dict.get((x_idx, y_idx + 1)),
                height_map_dict.get((x_idx + 1, y_idx + 1))
            ]
            
            if None in corner_indices:
                continue
            
            # Create two triangles per quad
            final_faces.append([corner_indices[0], corner_indices[1], corner_indices[3]])
            final_faces.append([corner_indices[0], corner_indices[3], corner_indices[2]])
    
    logger.info(f"Created {len(final_faces):,} triangular faces")
    
    if len(final_faces) == 0:
        logger.warning("No valid triangles could be created")
        return torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 3)
    
    # Build final vertices
    final_vertices = []
    for x_idx in range(num_cells_x):
        for y_idx in range(num_cells_y):
            vertex_key = (x_idx, y_idx)
            
            if vertex_key not in height_map_dict:
                continue
            
            z_height = cell_z_values[height_map_dict[vertex_key]]
            # Keep world placement faithful to the source LiDAR points.
            world_x = x_coord_map[x_idx, y_idx]
            world_y = y_coord_map[x_idx, y_idx]
            if np.isnan(world_x) or np.isnan(world_y):
                world_x = x_min + (x_idx / max(num_cells_x - 1, 1)) * scene_width
                world_y = y_min + (y_idx / max(num_cells_y - 1, 1)) * scene_height
            
            final_vertices.append([world_x, world_y, z_height])
    
    logger.info(f"Created {len(final_vertices):,} mesh vertices in original world space")
    
    # Convert to tensors
    vertices_tensor = torch.tensor(final_vertices).float() if len(final_vertices) > 0 else torch.empty(0, 3)
    faces_tensor = torch.tensor(final_faces).long() if len(final_faces) > 0 else torch.empty(0, 3)
    vertex_colors_final = np.array(cell_colors_list).astype(np.float32) / 255.0 if cell_colors_list is not None and len(cell_colors_list) > 0 else None
    
    logger.info(f"Mesh created successfully: {len(final_vertices):,} vertices, {len(final_faces):,} faces")
    
    return (
        vertices_tensor,
        faces_tensor,
        torch.tensor(vertex_colors_final) if vertex_colors_final is not None else torch.empty(0, 3)
    )
