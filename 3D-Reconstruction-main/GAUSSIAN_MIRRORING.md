# Gaussian Mirroring for Vehicle Training

This document describes the gaussian mirroring feature added to `tools/train_vehicles.py` for data augmentation during vehicle training.

## Overview

When training a specific vehicle with Difix-assisted synthetic views, you can now enable **gaussian mirroring** to automatically generate additional training views by mirroring the 3D gaussian representation along the vehicle's symmetry axis. This leverages the natural symmetry of vehicles to augment the training data.

## Key Concepts

### What is Gaussian Mirroring?

Gaussian mirroring creates a mirror image of the 3D gaussian representation by:

1. **Mirroring positions**: Flipping the gaussian centers (means) along a specified axis
2. **Mirroring orientations**: Adjusting the quaternion rotations to maintain proper orientation in the mirrored space
3. **Rendering new views**: Generating novel synthetic views from the mirrored gaussian configuration

### Why Mirror Along an Axis?

Most vehicles have natural symmetry:
- **Y-axis (lateral)**: Cars are typically symmetric left-right, making Y the natural mirror axis
- **X-axis (longitudinal)**: Can be used for objects with front-back symmetry
- **Z-axis (vertical)**: Less common but supported

For vehicles, **Y-axis mirroring is usually the best choice** as it preserves the car's structure while creating new viewing angles.

## Usage

### Basic Usage

Enable mirroring with the `--enable_mirror` flag:

```bash
python tools/train_vehicles.py \
    --resume_from /path/to/checkpoint.pth \
    --object_id 0 \
    --enable_mirror \
    --mirror_axis y
```

### Advanced Options

```bash
python tools/train_vehicles.py \
    --resume_from work_dirs/drivestudio/checkpoint_final.pth \
    --object_id 5 \
    --frame_idx 10 \
    --num_views 8 \
    --enable_mirror \
    --mirror_axis y \
    --num_rounds 2 \
    --num_iters 3000 \
    --synthetic_ratio 0.3
```

### Command-Line Arguments

- `--enable_mirror`: Flag to enable gaussian mirroring for data augmentation
- `--mirror_axis`: Axis along which to mirror (choices: `x`, `y`, `z`; default: `y`)
  - `y`: Mirror left-right (typical for vehicles)
  - `x`: Mirror front-back
  - `z`: Mirror top-bottom (uncommon)

### Workflow with Mirroring

When `--enable_mirror` is enabled:

1. Original views are rendered from the original gaussian representation
2. Original views are repaired using DiFix
3. **Mirrored views are rendered** from the mirror gaussian representation
4. **Mirrored views are repaired** using DiFix
5. Both original and mirrored views are combined for training
6. Training proceeds with an augmented dataset

### Example: Training a Vehicle with Mirroring

```bash
# Basic vehicle training with Y-axis mirroring (vehicle left-right symmetry)
python tools/train_vehicles.py \
    --resume_from ./output/base_checkpoint.pth \
    --object_id 1 \
    --frame_idx 50 \
    --num_views 8 \
    --elevation_deg 18.0 \
    --enable_mirror \
    --mirror_axis y \
    --num_iters 3000 \
    --prune_steps 500 \
    --output_root ./work_dirs/ \
    --project vehicles \
    --run_name car_mirrored_v1
```

## Implementation Details

### Mirror Functions

#### `mirror_gaussians(gs, mirror_axis, mirror_point)`

Mirrors a gaussian representation by:
- Flipping the means (positions) along the specified axis
- Adjusting quaternions to maintain proper orientation

For quaternion mirroring along different axes:
- **Y-axis**: Flip x and z rotation components
- **X-axis**: Flip y and z rotation components
- **Z-axis**: Flip x and y rotation components

#### `mirror_camera_pose(pose, mirror_axis, mirror_point)`

Mirrors a camera pose (4x4 transformation matrix) to maintain consistency between:
- The mirrored gaussian representation
- The viewing angle from which it's rendered

#### `render_mirrored_object_views(...)`

Renders views of the mirrored gaussian configuration:
- Takes the same parameters as `render_object_views`
- Applies gaussian mirroring
- Renders multiple viewpoints around the mirrored object
- Returns rendered images with alpha masks

## Technical Details

### Quaternion Transformation

When mirroring a quaternion `q = [w, x, y, z]` along different axes:

- **Y-axis mirroring**: Set `q' = [w, -x, y, -z]` (flip x and z)
- **X-axis mirroring**: Set `q' = [w, x, -y, -z]` (flip y and z)
- **Z-axis mirroring**: Set `q' = [w, -x, -y, z]` (flip x and y)

This ensures that after mirroring, the gaussians maintain proper orientation relative to their new positions.

### Camera Pose Transformation

The camera-to-world transformation matrix (4x4) is mirrored by:
1. Flipping the position component (translation)
2. Flipping the corresponding axis of the rotation matrix

This maintains the camera's viewing direction relative to the mirrored gaussians.

## Data Augmentation Impact

### Training Data Augmentation

With mirroring enabled:
- For N original views, you get N mirrored views
- Total training views = 2N (before DiFix repair)
- Effective training set size roughly doubles

### Synthetic Ratio

The `--synthetic_ratio` parameter (default 0.3) controls the mix:
- With mirroring: Both original and mirrored synthetic views are included
- Example: `--synthetic_ratio 0.3` with 16 views (8 original + 8 mirrored)
  - ~70% real data, ~30% synthetic data during training

## Best Practices

1. **Choose the right axis**: For vehicles, Y-axis is almost always correct
2. **Monitor training**: Check if mirrored views improve convergence
3. **Validate visually**: Use `--rerender_after` to inspect results between rounds
4. **Test gradually**: Start with smaller training runs to validate the approach

## Examples

### Training with Mirroring (Y-axis)

```bash
python tools/train_vehicles.py \
    --resume_from ./work_dirs/drivestudio/checkpoint_final.pth \
    --object_id 3 \
    --frame_idx 20 \
    --num_views 8 \
    --enable_mirror \
    --mirror_axis y \
    --num_iters 2000 \
    --rerender_after \
    --run_name vehicle_3_mirrored
```

### Training Multiple Frames with Mirroring

```bash
python tools/train_vehicles.py \
    --resume_from ./work_dirs/drivestudio/checkpoint_final.pth \
    --object_id 5 \
    --frames_from_instances \
    --frame_stride 10 \
    --max_frames 5 \
    --enable_mirror \
    --mirror_axis y \
    --num_rounds 2 \
    --num_iters 3000 \
    --run_name vehicle_5_multiframe_mirrored
```

## Troubleshooting

### Issue: Mirrored views look incorrect

**Solution**: Verify the mirror axis is correct. For vehicles:
- Use `y` for left-right symmetry (most common)
- Use `x` only if the object is front-back symmetric
- Use `z` rarely, only for top-bottom symmetric objects

### Issue: Training doesn't improve with mirroring

**Solution**: 
- The object may not be symmetric along the chosen axis
- Try training without mirroring to compare
- Check if object symmetry is preserved in the data

### Issue: Memory issues with double the views

**Solution**:
- Reduce `--num_views` to keep total views manageable
- Increase `--synthetic_ratio` to use more real data
- Run with a smaller batch size if needed

## Related Code

- [train_vehicles.py](tools/train_vehicles.py) - Main training script
- [basics.py](models/gaussians/basics.py) - Gaussian data structures
- [geometry.py](utils/geometry.py) - Geometric utilities
- [standalone_trainer_fixed_step.py](tools/standalone_trainer_fixed_step.py) - Training logic
