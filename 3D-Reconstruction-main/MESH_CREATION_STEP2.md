# Road Mesh Creation - Step 2 Implementation

## Overview

This document describes **Step 2** of the road mesh training pipeline, which creates a low-complexity 3D mesh from the aggregated point cloud with **no Z-axis overlap**.

## Key Features

### No Z-Axis Overlap Guarantee
The mesh creation algorithm ensures that each (x,y) position maps to exactly one z value by:
1. Grid-based discretization of the X-Y plane into uniform cells (~5cm resolution)
2. Computing mean height and color for points within each cell
3. Discarding high-variance cells (>0.05m² variance) that would cause vertical overlaps

### Heightmap-Based Mesh Construction
The mesh is constructed as a **heightmap** where:
- Each grid vertex has coordinates (x, y, z_height)
- Adjacent vertices are connected to form triangular faces
- No two points share the same (x,y) position with different heights

## Algorithm Details

### 1. Grid Cell Aggregation
```python
# For each cell in X-Y plane:
cell_center_x = x_min + (x_idx - 0.5) * scene_width / num_cells_x
cell_center_y = y_min + (y_idx - 0.5) * scene_height / num_cells_y

# Find points within this grid cell and compute statistics
mean_z = np.mean(cell_points[:, 2])
z_variance = np.var(cell_points[:, 2])

# Discard if variance too high (prevents Z-axis overlap from noise/overhangs)
if z_variance > max_height_variance:
    continue
```

### 2. Triangular Mesh Generation
- Each valid grid cell becomes a mesh vertex at position (x, y, mean_z)
- Adjacent cells are connected to form quads
- Each quad is split into two triangles for rendering compatibility
- Resulting mesh has no vertical overlaps by construction

## Output Files

The implementation generates:

1. **road_mesh.pth** - PyTorch tensor containing:
   - `vertices`: M x 3 tensor of vertex positions (meters)
   - `faces`: F x 3 tensor of triangular face indices
   - `vertex_colors`: Optional RGB colors for each vertex
   - Metadata about mesh complexity

2. **road_mesh.ply** - PLY format export for external inspection in tools like MeshLab, Blender, or CloudCompare

3. **Visualization Images**:
   - `mesh_top_view.png` - Bird's eye view (looking along Z-axis)
   - `mesh_side_view.png` - Side profile (X-Z plane)
   - `mesh_3d_view.png` - Isometric 3D perspective

## Mesh Quality Metrics

The implementation logs:
- **Total vertices**: Number of mesh vertices created from grid cells
- **Total faces**: Number of triangular faces in the mesh
- **Bounding box**: Extent of the mesh in X, Y, and Z dimensions
- **Z-height variance**: Measures vertical consistency (should be low)
- **Triangle density**: Triangles per square meter (low complexity metric)

## Usage Example

```bash
python tools/train_mesh_road.py \
    --scene_path data/paralane/processed/scene_000_clip_000 \
    --output_folder output \
    --project_name road_mesh_training \
    --run_name run_001
```

## Configuration Parameters

- `max_height_variance` (default: 0.05): Maximum allowed height variance within a grid cell before discarding the point cluster
- Grid resolution is adaptive based on scene extent, targeting ~5cm cells

## Integration with Pipeline

The created mesh can be loaded in subsequent steps for:
1. **Step 3**: Generate image buffer (4096x4096) and map pixels to mesh using X-Y coordinates
2. **Step 4**: Project RGB images onto the textured mesh from camera positions

## Technical Notes

- The heightmap approach ensures no Z-axis overlap but may lose overhanging structures or steep slopes (>85° relative to horizontal)
- For scenes with significant vertical features, consider increasing grid resolution (smaller cell size) at the cost of more triangles
- Color information is preserved from LiDAR point cloud if available; otherwise grayscale based on height

## References

See `lidar_loading.md` for complete pipeline documentation and Step 1 implementation in `tools/train_mesh_road.py`.
