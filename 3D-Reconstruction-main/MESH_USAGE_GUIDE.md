# Road Mesh Creation - Quick Usage Guide

## What Was Implemented

**Step 2 of the road mesh training pipeline**: Creating a low-complexity 3D mesh from pruned point cloud with **no Z-axis overlap**.

### Key Implementation Details:

1. **Heightmap-Based Approach** (No Z-Axis Overlap)
   - Grid-based discretization (~5cm cells in X-Y plane)
   - Each grid cell stores mean height and color
   - High-variance cells (>0.05m² variance) are discarded to prevent vertical overlaps
   
2. **Low Complexity Mesh**
   - Adaptive resolution based on scene extent
   - Triangular mesh connecting adjacent vertices
   - Typical triangle density: ~400 triangles/m²

3. **Output Files Generated**:
   - `road_mesh.pth` - PyTorch tensor (vertices, faces, colors)
   - `road_mesh.ply` - PLY format for external tools
   - Visualization images from multiple viewpoints

## How to Run

```bash
python tools/train_mesh_road.py \
    --scene_path data/paralane/processed/scene_000_clip_000 \
    --output_folder output \
    --project_name road_mesh_training \
    --run_name run_001
```

## Output Structure

After running, you'll find:

```
output/road_mesh_training/run_001/step1_pointcloud_visualization/
├── birds_eye_view.png           # Point cloud visualization (3 views)
├── pruned_pointcloud.png        # Pruned point cloud view
├── road_pointcloud.ply          # Aggregated LiDAR points in PLY format
├── mesh_top_view.png            # Mesh top-down view
├── mesh_side_view.png           # Mesh side profile (X-Z plane)
├── mesh_3d_view.png             # Isometric 3D perspective
├── road_mesh.pth                # **Main output: PyTorch tensor**
│   ├── vertices: M x 3 tensor of vertex positions (meters)
│   ├── faces: F x 3 tensor of triangular face indices  
│   └── vertex_colors: Optional RGB colors for each vertex
├── road_mesh.ply                # Mesh export in PLY format
└── training.log                 # Detailed logging with mesh statistics
```

## Loading the Created Mesh

```python
import torch

# Load the saved mesh
mesh_data = torch.load('output/road_mesh_training/run_001/step1_pointcloud_visualization/road_mesh.pth')

vertices = mesh_data['vertices']      # M x 3 tensor of vertex positions (meters)
faces = mesh_data['faces']            # F x 3 tensor of triangular face indices
vertex_colors = mesh_data.get('vertex_colors', None)  # Optional: M x 3 RGB colors
metadata = mesh_data['metadata']

print(f"Mesh created with {len(vertices)} vertices and {len(faces)} faces")
```

## Mesh Quality Metrics (Logged to Console)

The script outputs detailed statistics including:
- Total vertices and triangular faces
- Bounding box dimensions in X, Y, Z axes
- Height variance along Z-axis (should be low for flat road surfaces)
- Triangle density per square meter (low complexity indicator)

## Technical Notes

### No Z-Axis Overlap Guarantee
The mesh construction ensures no vertical overlaps by:
1. **Grid discretization**: Each X-Y position maps to exactly one grid cell
2. **Height averaging**: Mean height computed for all points in each cell  
3. **Variance filtering**: Cells with high variance (>0.05m²) are discarded

This guarantees that no two vertices share the same (x,y) coordinates, eliminating Z-axis overlap by construction.

### Low Complexity Design
- Grid resolution: ~5cm cells adaptively sized based on scene extent
- Typical output: 100s to low thousands of triangles depending on road area size
- Triangle density optimized for efficient rendering and texture mapping in subsequent steps

### Color Handling
- If LiDAR color data is available, it's preserved per vertex
- Otherwise, grayscale coloring based on height relative to scene bounds

## Integration with Pipeline Steps 3 & 4

The created mesh serves as the foundation for:

**Step 3**: Generate image buffer (4096x4096) and map pixels to mesh using X-Y coordinates with scaling factor.

**Step 4**: Project RGB images onto textured mesh from camera positions, filling only road areas according to road masks.

## Troubleshooting

### No Mesh Created
- Check that LiDAR points were successfully aggregated (see Step 1 output)
- Verify scene has sufficient point cloud coverage in X-Y plane
- Increase `max_height_variance` threshold if too many cells are discarded

### Too Many/Few Triangles
- Adjust grid resolution by modifying cell size parameter (~5cm default)
- Smaller cells = more triangles but better detail capture
- Larger cells = fewer triangles but may lose fine details

## References

- Full implementation: `tools/train_mesh_road.py` (lines 295-837 for mesh creation functions)
- Detailed documentation: `MESH_CREATION_STEP2.md`
- Complete pipeline overview: `lidar_loading.md`, `README.md`
