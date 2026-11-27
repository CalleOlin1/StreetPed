# Camera View Verification Feature

## Overview
Added functionality to save side-by-side comparison images before running `project_rgb_onto_image_buffer` to verify world space alignment between RGB camera views and mesh renders.

## Changes Made

### 1. New Function: `create_camera_view_comparison()`
**Location:** `/tools/train_mesh_road.py` (lines ~1780-1950)

This function creates a verification image with three panels:
- **Left Panel**: RGB camera view from the actual dataset, showing road mask overlay in semi-transparent red
- **Center Panel**: Quick side-by-side composite for rapid visual comparison  
- **Right Panel**: 3D mesh render from the exact same camera pose as the RGB view

**Key Features:**
- Uses specified camera ID (default: cam_id=2, middle/center view)
- Extracts actual camera position and orientation from dataset extrinsics
- Renders mesh with height-based coloring for better visualization
- Automatically matches 3D render viewpoint to real camera pose using rotation matrix analysis

### 2. Integration in Main Pipeline
**Location:** `/tools/train_mesh_road.py` (lines ~2025-2048)

Added verification step before Step 4 RGB projection:
```python
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
        num_frames=dataset.frame_num if hasattr(dataset, 'frame_num') else None
    )
    
    if comparison_path and os.path.exists(comparison_path):
        logger.info(f"Camera verification image saved to: {comparison_path}")
except Exception as e:
    logger.warning(f"Could not create camera view comparison (non-critical): {e}")
```

## Output Files

The verification generates a file named: `camera_{cam_id}_rgb_vs_mesh_comparison.png`

Example output path:  
`output/road_mesh_training/run_000/camera_2_rgb_vs_mesh_comparison.png`

## Usage Example

When running the training script, the comparison will automatically be generated before Step 4 projection:

```bash
python tools/train_mesh_road.py \
    --scene_path data/paralane/processed/scene_000_clip_000 \
    --output_folder output \
    --project_name road_mesh_training \
    --run_name run_001
```

The verification image will appear in the step 1 directory before RGB projection begins.

## Verification Checklist

After running, check the comparison image to verify:
- ✅ Camera positions match between RGB view and mesh render
- ✅ Mesh orientation aligns with scene geometry visible in RGB
- ✅ Road surface appears at correct depth relative to camera
- ✅ No obvious coordinate system mismatches (e.g., flipped axes)

## Customization Options

To change which camera is used for verification, modify the `cam_id` parameter:
```python
comparison_path = create_camera_view_comparison(
    dataset, 
    vertices_tensor, 
    faces_tensor, 
    cam_id=0,  # Change to different camera index (0-based)
    output_dir=step1_dir,
    num_frames=dataset.frame_num if hasattr(dataset, 'frame_num') else None
)
```

## Error Handling

The verification step is non-critical - failures will not stop the pipeline:
- If mesh data is empty or invalid, a warning is logged and processing continues
- Missing camera parameters (intrinsics/extrinsics) are handled gracefully
- All errors are caught and logged without interrupting Step 4 projection

## Technical Details

### Camera Pose Extraction
The function extracts actual camera pose from the dataset's `camera_to_world` matrix:
```python
cam_pos_world = c2w_np[:3, 3]  # Translation component (x, y, z) in world coordinates
rot_matrix = cam_to_world[:3, :3]  # Rotation component

# Convert rotation to elevation/azimuth for matplotlib view settings
pitch = np.arcsin(np.clip(-rot_matrix[2, 0], -1, 1)) * 180 / np.pi
yaw = np.arctan2(rot_matrix[2, 2], rot_matrix[2, 1]) * 180 / np.pi + 90

ax_3d.view_init(elev=abs(pitch), azim=yaw)
```

### Mesh Rendering Approach
Uses matplotlib's `PolyCollection` for efficient triangle rendering:
- Each face is rendered as a semi-transparent light blue polygon
- Edge lines in gray help visualize mesh structure
- Height-based coloring (via Z-coordinate averaging per face) provides depth cues

## Benefits

1. **Debugging**: Quickly identify coordinate system mismatches before expensive RGB projection
2. **Validation**: Confirm world space alignment is correct for your specific dataset/camera setup
3. **Documentation**: Generate visual proof of proper mesh-camera registration for reports/presentations
4. **Quality Control**: Catch configuration errors early in the pipeline
