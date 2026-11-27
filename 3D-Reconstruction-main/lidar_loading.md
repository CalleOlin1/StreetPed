# Road Mesh Training - Implementation Plan

This document outlines the implementation steps for training only the road mesh using road masks, lidar data, and RGB images.

## Overview

The basic algorithm consists of 4 main steps:
1. **Aggregate point cloud** using road masks and lidar data
2. **Create a 3D mesh** from the aggregated point cloud
3. **Create an image buffer** for texture mapping on the road mesh
4. **Project rays** using RGB images and camera positions onto the mesh

---

## Step 1: Aggregate Point Cloud Using Road Masks and Lidar Data

### Implementation Details

#### Objective
Aggregate LiDAR points that correspond to road surfaces by projecting LiDAR points into image space and filtering based on road masks.

#### Key Components from Existing Codebase

**1. Dataset Loading (driving_dataset.py)**
- Use `DrivingDataset` class to load scene data including:
  - RGB images with camera intrinsics/extrinsics
  - Road semantic masks (`road_masks`)
  - LiDAR point clouds (origins, directions, ranges)

**2. Point Cloud Initialization Pattern (models/trainers/base.py)**
```python
# Get class labels from gaussian dictionary
self.pts_labels = gs_dict.pop("class_labels")

# Identify road points using Gaussian class mapping
if "Road" in self.gaussian_classes:
    static_mask = (~static_mask) | (self.pts_labels == self.gaussian_classes["Road"])
```

**3. Road Point Extraction Methods (driving_dataset.py)**
- `get_lidar_indices_from_mask_region(mask_attr="road_masks")` - Returns global LiDAR point indices that project into road mask regions
- `get_lidar_points_from_mask_region(mask_attr="road_masks", num_samples=None, return_color=True)` - Extracts actual LiDAR points and colors from road regions

#### Implementation Steps

**Step 1.1: Initialize Dataset**
```python
# Load configuration for target scene
data_cfg = OmegaConf.load("configs/streetgs.yaml")  # or appropriate config
data_cfg.scene_idx = "scene_000"  # From CLI args

# Create DrivingDataset instance
dataset = DrivingDataset(data_cfg=data_cfg)
dataset.lidar_source.to(device)
```

**Step 1.2: Extract Road LiDAR Points**
```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Get all LiDAR points that project into road mask regions across all frames/cameras
road_lidar_indices = dataset.get_lidar_indices_from_mask_region(mask_attr="road_masks")
logger.info(f"Found {len(road_lidar_indices)} LiDAR points projecting to road masks")

# Extract the actual point coordinates and colors
road_pts, road_colors = dataset.get_lidar_points_from_mask_region(
    mask_attr="road_masks",
    num_samples=None,  # Use all available points or set a limit if needed
    return_color=True,
    device=device
)

logger.info(f"Extracted {len(road_pts)} unique road LiDAR points")
```

**Step 1.3: Aggregate Points Across All Frames**
The `get_lidar_indices_from_mask_region` method already aggregates across all frames by:
- Iterating through all camera views (`self.pixel_source.camera_data.values()`)
- For each frame, finding the closest LiDAR timestep via temporal synchronization
- Projecting 3D LiDAR points to 2D image space using camera intrinsics and extrinsics
- Checking if projected points fall within road mask regions
- Accumulating all matching point indices

**Step 1.4: Log Bird's Eye View of Point Cloud**
```python
import matplotlib.pyplot as plt
import numpy as np

# Convert to numpy for visualization
pts_xyz = road_pts.cpu().numpy()

if len(pts_xyz) > 0:
    # Create bird's eye view (top-down, looking along Y-axis or Z-axis depending on coordinate system)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # X-Y plane (bird's eye view)
    ax_xy = axes[0]
    scatter = ax_xy.scatter(pts_xyz[:, 0], pts_xyz[:, 1], c=range(len(pts_xyz)), 
                           cmap='viridis', s=1, alpha=0.5)
    ax_xy.set_xlabel('X (meters)')
    ax_xy.set_ylabel('Y (meters)')
    ax_xy.set_title(f"Bird's Eye View - {len(pts_xyz)} Road Points")
    ax_xy.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax_xy, label='Point Index')
    
    # X-Z plane (side view)
    ax_xz = axes[1]
    scatter2 = ax_xz.scatter(pts_xyz[:, 0], pts_xyz[:, 2], c=range(len(pts_xyz)), 
                            cmap='viridis', s=1, alpha=0.5)
    ax_xz.set_xlabel('X (meters)')
    ax_xz.set_ylabel('Z (height in meters)')
    ax_xz.set_title("Side View - X-Z Plane")
    ax_xz.grid(True, alpha=0.3)
    
    # Y-Z plane (front view)  
    ax_yz = axes[2]
    scatter3 = ax_yz.scatter(pts_xyz[:, 1], pts_xyz[:, 2], c=range(len(pts_xyz)), 
                            cmap='viridis', s=1, alpha=0.5)
    ax_yz.set_xlabel('Y (meters)')
    ax_yz.set_ylabel('Z (height in meters)')
    ax_yz.set_title("Front View - Y-Z Plane")
    ax_yz.grid(True, alpha=0.3)
    
    # Save visualization
    output_dir = os.path.join(output_folder, "step1_pointcloud_visualization")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "birds_eye_view.png"), dpi=150, bbox_inches='tight')
    logger.info(f"Saved bird's eye view visualization to {output_dir}/birds_eye_view.png")
    
    # Log statistics
    logger.info(f"Point cloud bounding box:")
    logger.info(f"  X: [{pts_xyz[:, 0].min():.2f}, {pts_xyz[:, 0].max():.2f}] meters")
    logger.info(f"  Y: [{pts_xyz[:, 1].min():.2f}, {pts_xyz[:, 1].max():.2f}] meters")
    logger.info(f"  Z (height): [{pts_xyz[:, 2].min():.2f}, {pts_xyz[:, 2].max():.2f}] meters")
    
    # Log height distribution for road surface analysis
    if len(pts_xyz) > 0:
        mean_height = pts_xyz[:, 2].mean()
        std_height = pts_xyz[:, 2].std()
        logger.info(f"Road point cloud statistics:")
        logger.info(f"  Mean Z (height): {mean_height:.4f} ± {std_height:.4f} meters")
        logger.info(f"  Height range: {pts_xyz[:, 2].max() - pts_xyz[:, 2].min():.4f} meters")
else:
    logger.warning("No road LiDAR points found! Check road masks and scene configuration.")

# Optional: Export point cloud to PLY format for inspection in external tools
from models.gaussians.basics import export_points_to_ply
ply_path = os.path.join(output_dir, "road_pointcloud.ply")
export_points_to_ply(
    pts_xyz.astype(np.float32),
    road_colors.cpu().numpy().astype(np.float32) if len(pts_xyz) > 0 else np.zeros((0, 3)),
    save_path=ply_path
)
logger.info(f"Exported point cloud to {ply_path}")
```

**Step 1.5: Temporal Aggregation (if training on multiple frames)**
If the scene has temporal variation in road surface appearance:
- Aggregate points across all timesteps using `get_lidar_indices_from_mask_region` which already handles this
- Consider filtering for consistent road regions if needed based on point density over time

### Expected Output from Step 1

After completing Step 1, you should have:
1. **Aggregated Road Point Cloud**: A set of 3D points (`road_pts`) representing the road surface geometry extracted via LiDAR and filtered by semantic road masks
2. **Point Colors**: Corresponding RGB colors for each point from LiDAR intensity or color data
3. **Visualization Logs**: 
   - Bird's eye view showing X-Y distribution of road points
   - Side/front views showing Z-height variations
   - Statistical summary (bounding box, mean height, etc.)
4. **Exported Files**:
   - `road_pointcloud.ply` for external visualization/inspection

### CLI Arguments Required for Step 1

```python
# Parse command line arguments
import argparse

parser = argparse.ArgumentParser(description="Train road mesh from LiDAR and road masks")
parser.add_argument("--scene_path", type=str, required=True, 
                    help="Path to scene data directory (e.g., 'data/paralane/processed/scene_000_clip_000')")
parser.add_argument("--output_folder", type=str, default="output",
                    help="Output folder for project results")
parser.add_argument("--project_name", type=str, default="road_mesh_training",
                    help="Project name (e.g., 'road_mesh_training')")
parser.add_argument("--run_name", type=str, default="run_000",
                    help="Run identifier (e.g., 'run_000')")

args = parser.parse_args()

# Construct output path
output_folder = os.path.join(args.output_folder, args.project_name, args.run_name)
os.makedirs(output_folder, exist_ok=True)
```

### Dependencies and Assumptions

1. **Dataset Configuration**: Requires a valid dataset configuration file (e.g., `configs/streetgs.yaml`) with:
   - Correct data root path
   - Lidar source enabled (`load_lidar: true`)
   - Road mask availability in pixel source
   
2. **Temporal Synchronization**: LiDAR and camera timestamps must be synchronized as per the DrivingDataset requirements

3. **Coordinate System**: Assumes standard automotive coordinate system (X-forward, Y-left, Z-up) or similar; adjust visualization axes accordingly based on actual dataset convention

4. **Memory Efficiency**: The aggregation method processes frames sequentially to avoid loading all projections simultaneously into memory

---

## Next Steps After Step 1

Once you have the aggregated road point cloud from Step 1:
- Proceed to **Step 2** for mesh creation using surface reconstruction techniques (e.g., Poisson surface reconstruction, ball-pivoting)
- The low polys assumption means we can use simplified meshes with ~100-500 triangles depending on scene complexity

---

## Notes

- This implementation leverages existing infrastructure from the StreetGaussian framework rather than duplicating functionality
- The road mask-based filtering ensures only relevant LiDAR points are used, improving mesh quality and reducing noise
- Bird's eye view logging provides immediate visual feedback for debugging point cloud extraction pipeline
