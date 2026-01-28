# Camera pose manipulation and trajectory generation.
import os
import torch
import numpy as np
import math
from typing import Dict, List, Optional

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


def interpolate_poses(key_poses: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Interpolate between key poses to generate a smooth trajectory.

    Args:
        key_poses (torch.Tensor): Tensor of shape (N, 4, 4) containing key camera poses.
        target_frames (int): Number of frames to interpolate.

    Returns:
        torch.Tensor: Interpolated poses of shape (target_frames, 4, 4).
    """
    device = key_poses.device
    key_poses = key_poses.cpu().numpy()

    # Separate translation and rotation
    translations = key_poses[:, :3, 3]
    rotations = key_poses[:, :3, :3]

    # Create time array
    times = np.linspace(0, 1, len(key_poses))
    target_times = np.linspace(0, 1, target_frames)

    # Interpolate translations
    interp_translations = np.stack(
        [np.interp(target_times, times, translations[:, i]) for i in range(3)], axis=-1
    )

    # Interpolate rotations using Slerp
    key_rots = R.from_matrix(rotations)
    slerp = Slerp(times, key_rots)
    interp_rotations = slerp(target_times).as_matrix()

    # Combine interpolated translations and rotations
    interp_poses = np.eye(4)[None].repeat(target_frames, axis=0)
    interp_poses[:, :3, :3] = interp_rotations
    interp_poses[:, :3, 3] = interp_translations

    return torch.tensor(interp_poses, dtype=torch.float32, device=device)


def look_at_rotation(
    direction: torch.Tensor, up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])
) -> torch.Tensor:
    """Calculate rotation matrix to look at a specific direction."""
    # Ensure input tensors are on the same device
    up = up.to(direction.device)  # [!code ++]
    front = torch.nn.functional.normalize(direction, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(front, up), dim=-1)
    up = torch.cross(right, front)
    rotation_matrix = torch.stack([right, up, -front], dim=-1)
    return rotation_matrix


def get_interp_novel_trajectories(
    dataset_type: str,
    scene_idx: str,
    per_cam_poses: Dict[int, torch.Tensor],
    traj_type: str = "front_center_interp",
    target_frames: int = 100,
) -> torch.Tensor:
    original_frames = per_cam_poses[list(per_cam_poses.keys())[0]].shape[0]

    trajectory_generators = {
        "front_center_interp": front_center_interp,
        "s_curve": s_curve,
        "three_key_poses": three_key_poses_trajectory,
        # New trajectory types
        "circle_trajectory": circle_trajectory,
        "spiral_trajectory": spiral_trajectory,
        "look_around_trajectory": look_around_trajectory,
        "fixed_path_trajectory": kitti_fixed_path,
        "analyze_center_trajectory":analyze_front_center_interp,
        "analyze_npz_trajectory":analyze_kitti_trajectory,
        "fixed_offset_1": fixed_offset_trajectory_1,
        "fixed_offset_2": fixed_offset_trajectory_2,
        "fixed_offset_3": fixed_offset_trajectory_3,
        "fixed_offset_4": fixed_offset_trajectory_4,
        "fixed_offset_5": fixed_offset_trajectory_5,
        "fixed_offset_6": fixed_offset_trajectory_6,
        "fixed_offset_7": fixed_offset_trajectory_7,
        "fixed_offset_8": fixed_offset_trajectory_8,
        "fixed_offset_9": fixed_offset_trajectory_9,
        "fixed_offset_10": fixed_offset_trajectory_10,
        "fixed_offset": fixed_offset_trajectory,
        "lane_change": smooth_lane_change_trajectory,
        "double_lane_change": double_lane_change_trajectory,
    }

    if traj_type not in trajectory_generators:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return trajectory_generators[traj_type](
        dataset_type, per_cam_poses, original_frames, target_frames
    )

def kitti_fixed_path(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    num_loops: int = 1,
    npz_path = "output/Kitti/dataset=Kitti/change_line_gt/camera_poses_eval/full_poses_2025-07-02_18-00-20.npz",
    position_offset: Optional[List[float]] = None,  # New: position offset [x, y, z]
    rotation_offset: Optional[List[float]] = None,  # New: rotation offset [roll, pitch, yaw] (radians)
) -> torch.Tensor:
    """
    Read complete camera trajectory from NPZ file, no interpolation, use raw data directly
    
    Args:
        dataset_type (str): Dataset type (unused in this function)
        per_cam_poses (Dict[int, torch.Tensor]): Per-camera poses (unused in this function)
        original_frames (int): Original frame count (unused in this function)
        target_frames (int): Target frame count (if exceeds original frames, will repeat or truncate)
        num_loops (int): Number of loops (unused in this function)
        position_offset (List[float], optional): Position offset [x, y, z] in meters
        rotation_offset (List[float], optional): Rotation offset [roll, pitch, yaw] in radians
        
    Returns:
        torch.Tensor: Original trajectory data, shape (actual_frames, 4, 4)
    """
    # Hardcoded NPZ file path
    # npz_path = "output/Kitti/dataset=Kitti/change_line_gt/camera_poses_eval/full_poses_2025-07-02_18-00-20.npz"
    
    print(f"🔍 Loading complete trajectory from NPZ (no interpolation):")

    # position_offset = [0, 0, 0]
    
    try:
        # Load NPZ file
        data = np.load(npz_path, allow_pickle=True)
        camera_poses = data['camera_poses']  # Shape: (N, 4, 4)
        cam_names = data['cam_names']        # Camera name list
        frame_indices = data['frame_indices'] # Frame indices
        
        print(f"   NPZ contains {len(camera_poses)} total poses")
        print(f"   Available cameras: {set(cam_names)}")
        
        # Find front center camera (try various possible naming conventions)
        front_center_mask = None
        found_camera = None
        
        for candidate in ['CAM_LEFT', 'FRONT_CENTER', 'front_center', 'FRONT', 'front', '0', 'cam0']:
            mask = np.array([str(name) == candidate for name in cam_names])
            if mask.any():
                front_center_mask = mask
                found_camera = candidate
                break
        
        # If not found, use the first camera
        if front_center_mask is None:
            front_center_mask = np.ones(len(cam_names), dtype=bool)
            front_center_mask[1:] = False  # Only keep the first one
            found_camera = str(cam_names[0])
        
        print(f"   Using camera: {found_camera}")
        
        # Extract front center camera poses
        front_center_poses = camera_poses[front_center_mask]
        front_center_frames = np.array(frame_indices)[front_center_mask]
        
        print(f"   Found {len(front_center_poses)} poses for this camera")
        
        # Sort by frame index
        sorted_indices = np.argsort(front_center_frames)
        front_center_poses = front_center_poses[sorted_indices]
        front_center_frames = front_center_frames[sorted_indices]
        
        print(f"   Frame range: {front_center_frames[0]} - {front_center_frames[-1]}")
        
        # Display position ranges
        positions = front_center_poses[:, :3, 3]
        print(f"   Position ranges:")
        print(f"     X: [{positions[:, 0].min():.6f}, {positions[:, 0].max():.6f}] m")
        print(f"     Y: [{positions[:, 1].min():.6f}, {positions[:, 1].max():.6f}] m")
        print(f"     Z: [{positions[:, 2].min():.6f}, {positions[:, 2].max():.6f}] m")
        
        # Convert to torch tensor
        poses_tensor = torch.tensor(front_center_poses, dtype=torch.float32)
        
        # Ensure device consistency
        if per_cam_poses and len(per_cam_poses) > 0:
            sample_pose = per_cam_poses[list(per_cam_poses.keys())[0]]
            poses_tensor = poses_tensor.to(sample_pose.device)
        
        # Adjust output based on target_frames
        actual_frames = len(poses_tensor)
        
        if target_frames <= actual_frames:
            # If target frames is less than or equal to actual frames, truncate directly
            result = poses_tensor[:target_frames]
            print(f"   Truncated to {target_frames} frames (from {actual_frames})")
        else:
            # If target frames is more than actual frames, repeat last frame
            result = torch.zeros(target_frames, 4, 4, dtype=poses_tensor.dtype, device=poses_tensor.device)
            result[:actual_frames] = poses_tensor
            # Fill remaining with last frame
            for i in range(actual_frames, target_frames):
                result[i] = poses_tensor[-1]
            print(f"   Extended to {target_frames} frames (repeated last frame)")
        
        # ==================== New: Apply offsets ====================
        if position_offset is not None or rotation_offset is not None:
            print(f"   Applying offsets...")
            result = apply_trajectory_offset(result, position_offset, rotation_offset)
        
        # Output result info
        result_positions = result[:, :3, 3]
        print(f"   Output: {result.shape[0]} frames")
        print(f"   Start: {result_positions[0][0]:.3f}, {result_positions[0][1]:.3f}, {result_positions[0][2]:.3f}")
        print(f"   End:   {result_positions[-1][0]:.3f}, {result_positions[-1][1]:.3f}, {result_positions[-1][2]:.3f}")
        
        return result
        
    except Exception as e:
        print(f"Error loading from NPZ: {e}")
        # If NPZ loading fails, fall back to the original front_center_interp
        print("Falling back to front_center_interp")
        assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required for fallback"
        key_poses = per_cam_poses[0][::original_frames // 4]
        return interpolate_poses(key_poses, target_frames)


def apply_trajectory_offset(
    poses: torch.Tensor, 
    position_offset: Optional[List[float]] = None,
    rotation_offset: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Apply position and rotation offsets to trajectory
    
    Args:
        poses: Original poses tensor, shape (N, 4, 4)
        position_offset: Position offset [x, y, z], in meters
        rotation_offset: Rotation offset [roll, pitch, yaw], in radians
        
    Returns:
        torch.Tensor: Poses with applied offsets
    """
    import torch
    import math
    
    result = poses.clone()
    
    # Apply position offset
    if position_offset is not None:
        offset_tensor = torch.tensor(position_offset, dtype=poses.dtype, device=poses.device)
        print(f"     Position offset: {position_offset}")
        
        # Method 1: Simple global offset (in world coordinate system)
        result[:, :3, 3] += offset_tensor
        
        # Method 2: Offset relative to camera orientation (enable this if you want relative offset)
        # for i in range(len(result)):
        #     # Get current camera's rotation matrix
        #     rotation_matrix = result[i, :3, :3]
        #     # Transform offset to camera coordinate system
        #     relative_offset = rotation_matrix @ offset_tensor
        #     result[i, :3, 3] += relative_offset
    
    # Apply rotation offset
    if rotation_offset is not None:
        print(f"     Rotation offset (roll, pitch, yaw): {rotation_offset}")
        
        # Convert Euler angles to rotation matrix
        roll, pitch, yaw = rotation_offset
        
        # Create rotation matrix (ZYX order)
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch) 
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        
        # Roll (X-axis rotation)
        R_x = torch.tensor([
            [1, 0, 0],
            [0, cos_r, -sin_r],
            [0, sin_r, cos_r]
        ], dtype=poses.dtype, device=poses.device)
        
        # Pitch (Y-axis rotation)
        R_y = torch.tensor([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p]
        ], dtype=poses.dtype, device=poses.device)
        
        # Yaw (Z-axis rotation)
        R_z = torch.tensor([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ], dtype=poses.dtype, device=poses.device)
        
        # Combine rotation matrices (ZYX order)
        R_offset = R_z @ R_y @ R_x
        
        # Apply rotation offset to each frame
        for i in range(len(result)):
            # Original rotation matrix
            original_rotation = result[i, :3, :3]
            # Apply offset rotation
            result[i, :3, :3] = R_offset @ original_rotation
    
    return result
    
def front_center_interp(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    num_loops: int = 1,
) -> torch.Tensor:
    """Interpolate key frames from the front center camera."""
    assert (
        0 in per_cam_poses.keys()
    ), "Front center camera (ID 0) is required for front_center_interp"
    key_poses = per_cam_poses[0][
        :: original_frames // 4
    ]  # Select every 4th frame as key frame
    return interpolate_poses(key_poses, target_frames)


def fixed_offset_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    translation_offset: list = [-4.0, 0.0, 0.0],
    rotation_offset: list = [0.0, 0.0, 0.0],
) -> torch.Tensor:
    """
    Generate fixed offset trajectory relative to front camera

    Args:
        translation_offset (list): [x, y, z] translation offset (meters)
        rotation_offset (list): [pitch, yaw, roll] rotation offset (degrees)
    """
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) required"

    # Get device information
    device = per_cam_poses[0].device

    # Convert offsets to tensors
    trans_offset = torch.tensor(translation_offset, device=device, dtype=torch.float32)
    rot_offset = torch.tensor(rotation_offset, device=device, dtype=torch.float32)

    # Ensure original_frames is at least 1
    original_frames = max(1, original_frames)
    # Calculate step size, ensure at least 1
    step = max(1, original_frames // 4)
    key_poses = per_cam_poses[0][::step]

    def convert_to_tensor(data, device):
        return torch.tensor(data, device=device, dtype=torch.float32)  # [!code ++]

    # Apply offsets
    modified_poses = []
    for pose in key_poses:
        # Create new pose matrix
        new_pose = torch.eye(4, device=device)

        rot_matrix = R.from_euler(
            "xyz", rot_offset.cpu().numpy(), degrees=True
        ).as_matrix()
        rot_matrix = rot_matrix.astype(np.float32)  # [!code ++]

        # Modification 3: Keep matrix multiplication data types consistent
        new_rot = pose[:3, :3] @ convert_to_tensor(rot_matrix, device)  # [!code ++]

        # Modification 4: Ensure translation offset data type is correct
        trans_offset = convert_to_tensor(translation_offset, device)  # [!code ++]
        offset_trans = pose[:3, :3] @ trans_offset
        new_trans = pose[:3, 3] + offset_trans

        new_pose[:3, :3] = new_rot
        new_pose[:3, 3] = new_trans

        modified_poses.append(new_pose)
    # Ensure at least two poses for interpolation
    if len(modified_poses) == 1:
        # If only one pose, directly copy it to create target number of frames
        return modified_poses[0].unsqueeze(0).repeat(target_frames, 1, 1)
    return interpolate_poses(torch.stack(modified_poses), target_frames)


def analyze_front_center_interp(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    num_loops: int = 1,
) -> torch.Tensor:
    """
    Analyze front_center_interp logic, display key information
    """
    print(f"\U0001f50d Front Center Interp Analysis:")
    
    # Check input
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) required"
    front_poses = per_cam_poses[0]
    
    # Basic information
    print(f"   Input: {len(front_poses)} poses -> Target: {target_frames} frames")
    print(f"   Original frames param: {original_frames}")
    
    # Key frame selection logic
    step = original_frames // 4
    key_poses = front_poses[::step]
    print(f"   Step size: {step} -> Key frames: {len(key_poses)}")
    
    # Display key frame coordinates
    print(f"   Key frame positions:")
    for i, pose in enumerate(key_poses):
        pos = pose[:3, 3]
        print(f"     [{i}] {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")
    
    # Perform interpolation
    result = interpolate_poses(key_poses, target_frames)
    
    # Output result
    result_start = result[0][:3, 3]
    result_end = result[-1][:3, 3]
    print(f"   Output: {result.shape[0]} frames")
    print(f"   Start: {result_start[0]:.3f}, {result_start[1]:.3f}, {result_start[2]:.3f}")
    print(f"   End:   {result_end[0]:.3f}, {result_end[1]:.3f}, {result_end[2]:.3f}")
    
    return result


def analyze_kitti_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    num_loops: int = 1,
) -> torch.Tensor:
    """
    Analyze kitti trajectory logic
    """
    npz_path =  "output/Kitti/dataset=Kitti/training_20250630_162211_FollowLeadingVehicleWithObstacle_1/camera_poses_eval/full_poses_2025-07-02_18-00-20.npz"
    
    print(f"\U0001f50d Kitti Trajectory Analysis:")
    
    try:
        # Load data
        data = np.load(npz_path, allow_pickle=True)
        camera_poses = data['camera_poses']
        cam_names = data['cam_names']
        
        # Find front center camera
        front_mask = None
        for candidate in ['FRONT_CENTER', 'front_center', 'FRONT', 'front', '0']:
            mask = np.array([str(name) == candidate for name in cam_names])
            if mask.any():
                front_mask = mask
                break
        
        if front_mask is None:
            front_mask = np.ones(len(cam_names), dtype=bool)
            front_mask[1:] = False
        
        # Extract poses
        front_poses = camera_poses[front_mask]
        frame_indices = data['frame_indices']
        front_frames = np.array(frame_indices)[front_mask]
        
        # Sort
        sorted_indices = np.argsort(front_frames)
        front_poses = front_poses[sorted_indices]
        
        print(f"   NPZ input: {len(front_poses)} poses -> Target: {target_frames} frames")
        
        # Key frame selection
        actual_frames = len(front_poses)
        step = max(1, actual_frames // 4)
        key_poses = front_poses[::step]
        
        print(f"   Step size: {step} -> Key frames: {len(key_poses)}")
        
        # Display key frame coordinates
        print(f"   Key frame positions:")
        for i, pose in enumerate(key_poses):
            pos = pose[:3, 3]
            print(f"     [{i}] {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")
        
        # Convert to tensor and interpolate
        key_poses_tensor = torch.tensor(key_poses, dtype=torch.float32)
        if per_cam_poses and len(per_cam_poses) > 0:
            sample_pose = per_cam_poses[list(per_cam_poses.keys())[0]]
            key_poses_tensor = key_poses_tensor.to(sample_pose.device)
        
        result = interpolate_poses(key_poses_tensor, target_frames)
        
        # Output result
        result_start = result[0][:3, 3]
        result_end = result[-1][:3, 3]
        print(f"   Output: {result.shape[0]} frames")
        print(f"   Start: {result_start[0]:.3f}, {result_start[1]:.3f}, {result_start[2]:.3f}")
        print(f"   End:   {result_end[0]:.3f}, {result_end[1]:.3f}, {result_end[2]:.3f}")
        
        return result
        
    except Exception as e:
        print(f"   Error: {e}")
        return analyze_front_center_interp(dataset_type, per_cam_poses, original_frames, target_frames, num_loops)

# FIX_TRAJ= "output/streetgs/dataset=Kitti/training_20250628_171319_FollowLeadingVehicle_1/camera_poses_eval/full_poses_2025-07-09_00-52-47.npz"
FIX_TRAJ= "output/pvg/dataset=Kitti/training_20250628_171319_FollowLeadingVehicle_1/camera_poses/full_poses_2025-07-08_21-34-38.npz"


def fixed_offset_trajectory_1(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Based on kitti_fixed_path + [0.0, 0.0, 0.5] offset"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.5, 0.0, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_2(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Based on kitti_fixed_path + [3.2, 0.0, 0.0] offset"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, -3.2, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_3(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Based on kitti_fixed_path + [1.6, 0.0, 0.0] offset"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, -1.6, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_4(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Based on kitti_fixed_path + [-3.2, 0.0, 0.0] offset"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, 3.2, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_5(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Based on kitti_fixed_path + [-1.6, 0.0, 0.0] offset"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, 1.6, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_6(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Based on kitti_fixed_path + [0.5, 0.0, 0.0] offset"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, -0.5, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_7(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Based on kitti_fixed_path + [-0.5, 0.0, 0.0] offset"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, 0.5, 0.0],
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_8(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Based on kitti_fixed_path + [0.5, 0.0, 0.0] offset + Y-axis rotation 15 degrees"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, -0.5, 0.0],
        rotation_offset=[0.0, 0.0, math.radians(-15.0)],  # Convert to radians
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_9(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Based on kitti_fixed_path + [-0.5, 0.0, 0.0] offset + Y-axis rotation -15 degrees"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[0.0, 0.5, 0.0],
        rotation_offset=[0.0, 0.0, math.radians(15.0)],  # Convert to radians
        npz_path = FIX_TRAJ
    )

def fixed_offset_trajectory_10(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Based on kitti_fixed_path + [0.0, 0.0, -0.5] offset"""
    return kitti_fixed_path(
        dataset_type, 
        per_cam_poses, 
        original_frames, 
        target_frames,
        position_offset=[-0.5, 0.0, 0.0],
        npz_path = FIX_TRAJ
    )

def double_lane_change_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    first_change_start: int = 20,
    first_change_end: int = 50,
    second_change_start: int = 110,
    second_change_end: int = 130,
    lane_offset: float = 3.2,
    offset_vector: list = [0.0, -1.0, 0.0],  # Lane change to the left
    return_offset_vector: list = [0.0, 1.0, 0.0],  # Return to the right
    first_steer_angle: float = 5.0,   # Steering angle for first lane change (degrees)
    second_steer_angle: float = -5.0,  # Steering angle for second lane change (degrees)
) -> torch.Tensor:
    """
    Generate lane change and return trajectory with steering
    
    Each lane change includes: steering -> straightening process
    Steering starts at change_start and ends at change_end
    Steering rotates around Y-axis (horizontal steering)
    """
    import math
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required"
    assert (
        0
        <= first_change_start
        < first_change_end
        < second_change_start
        < second_change_end
        < target_frames
    ), "Invalid frame index settings"
    
    # Get device info
    device = per_cam_poses[0].device
    
    # Generate base trajectory (using front_center_interp)
    base_trajectory = front_center_interp(
        dataset_type, per_cam_poses, original_frames, target_frames
    )
    
    # Normalize the offset vector for the first lane change
    first_vector = torch.tensor(offset_vector, device=device, dtype=torch.float32)
    first_vector = first_vector / torch.norm(first_vector)
    first_full_offset = first_vector * lane_offset
    
    # Normalize the offset vector for the second lane change
    second_vector = torch.tensor(
        return_offset_vector, device=device, dtype=torch.float32
    )
    second_vector = second_vector / torch.norm(second_vector)
    second_full_offset = second_vector * lane_offset
    
    # Create lane change trajectory
    lane_change_trajectory = base_trajectory.clone()
    
    # Create steering rotation (rotation around Y-axis - horizontal steering)
    # Steering rotation for first lane change
    first_rot = R.from_euler('y', first_steer_angle, degrees=True)
    first_rot_matrix = torch.tensor(first_rot.as_matrix(), device=device, dtype=torch.float32)
    
    # Steering rotation for second lane change
    second_rot = R.from_euler('y', second_steer_angle, degrees=True)
    second_rot_matrix = torch.tensor(second_rot.as_matrix(), device=device, dtype=torch.float32)
    
    # Identity rotation matrix (straight driving state)
    identity_rot = torch.eye(3, device=device, dtype=torch.float32)
    
    # Custom spherical linear interpolation function
    def custom_slerp(rot1, rot2, t):
        """
        Custom spherical linear interpolation, compatible with older scipy versions
        """
        q1 = rot1.as_quat()
        q2 = rot2.as_quat()
        
        # Calculate quaternion dot product
        dot = np.dot(q1, q2)
        
        # If dot product is negative, flip second quaternion to ensure shortest path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are nearly identical, use linear interpolation directly
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            result /= np.linalg.norm(result)
            return R.from_quat(result)
        
        # Calculate interpolation angle
        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        # Spherical linear interpolation
        result = s0 * q1 + s1 * q2
        return R.from_quat(result)
    
    # Calculate phase length for each lane change
    first_duration = first_change_end - first_change_start
    first_steer_duration = first_duration // 2  # Steering phase (first half)
    first_return_duration = first_duration - first_steer_duration  # Straightening phase (second half)
    
    second_duration = second_change_end - second_change_start
    second_steer_duration = second_duration // 2  # Steering phase (first half)
    second_return_duration = second_duration - second_steer_duration  # Straightening phase (second half)
    
    # Apply lane change and steering
    for frame_idx in range(target_frames):
        original_rotation = base_trajectory[frame_idx, :3, :3].clone()
        
        if frame_idx < first_change_start:
            # Before first lane change, keep original trajectory
            continue
            
        elif frame_idx < first_change_start + first_steer_duration:
            # First lane change: steering phase (first half)
            progress = (frame_idx - first_change_start) / first_steer_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # Smooth displacement
            current_offset = first_full_offset * ((frame_idx - first_change_start) / first_duration)
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # Smooth steering
            start_rot = R.from_matrix(identity_rot.cpu().numpy())
            end_rot = R.from_matrix(first_rot_matrix.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # Apply smooth rotation
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
            
        elif frame_idx <= first_change_end:
            # First lane change: straightening phase (second half)
            progress = (frame_idx - first_change_start - first_steer_duration) / first_return_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # Continue moving
            current_offset = first_full_offset * ((frame_idx - first_change_start) / first_duration)
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # Return from steering to straight driving
            start_rot = R.from_matrix(first_rot_matrix.cpu().numpy())
            end_rot = R.from_matrix(identity_rot.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # Apply smooth rotation
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
            
        elif frame_idx < second_change_start:
            # Keep straight driving between the two lane changes
            lane_change_trajectory[frame_idx, :3, 3] += first_full_offset
            # Maintain straight driving state
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ identity_rot
            
        elif frame_idx < second_change_start + second_steer_duration:
            # Second lane change: steering phase (first half)
            progress = (frame_idx - second_change_start) / second_steer_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # Smooth return
            total_progress = (frame_idx - second_change_start) / second_duration
            current_offset = first_full_offset + second_full_offset * total_progress
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # Smooth steering
            start_rot = R.from_matrix(identity_rot.cpu().numpy())
            end_rot = R.from_matrix(second_rot_matrix.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # Apply smooth rotation
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
            
        elif frame_idx <= second_change_end:
            # Second lane change: straightening phase (second half)
            progress = (frame_idx - second_change_start - second_steer_duration) / second_return_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # Continue returning
            total_progress = (frame_idx - second_change_start) / second_duration
            current_offset = first_full_offset + second_full_offset * total_progress
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # Return from steering to straight driving
            start_rot = R.from_matrix(second_rot_matrix.cpu().numpy())
            end_rot = R.from_matrix(identity_rot.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # Apply smooth rotation
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
            
        else:
            # After second lane change, maintain straight driving state
            # No offset added, keep driving straight
            continue
    
    return lane_change_trajectory


def smooth_lane_change_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    start_frame: int = 20,
    end_frame: int = 50,
    lane_offset: float = -3.2,
    offset_vector: list = [0.0, 1.0, 0.0],  # Lane change to the left
    steer_angle: float = -5.0,  # Steering angle (degrees)
) -> torch.Tensor:
    """
    Generate single lane change trajectory with steering
    
    Lane change process includes: steering -> straightening
    Steering starts at start_frame and ends at end_frame
    Steering rotates around Y-axis (horizontal steering)
    """
    import math
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required"
    assert 0 <= start_frame < end_frame < target_frames, "Invalid frame index settings"
    
    # Get device info
    device = per_cam_poses[0].device
    
    # Generate base trajectory (using front_center_interp)
    base_trajectory = front_center_interp(
        dataset_type, per_cam_poses, original_frames, target_frames
    )
    
    # Normalize offset vector
    norm_vector = torch.tensor(offset_vector, device=device, dtype=torch.float32)
    norm_vector = norm_vector / torch.norm(norm_vector)
    
    # Calculate full offset
    full_offset = norm_vector * lane_offset
    
    # Create lane change trajectory
    lane_change_trajectory = base_trajectory.clone()
    
    # Create steering rotation (rotation around Y-axis - horizontal steering)
    steer_rot = R.from_euler('y', steer_angle, degrees=True)
    steer_rot_matrix = torch.tensor(steer_rot.as_matrix(), device=device, dtype=torch.float32)
    
    # Identity rotation matrix (straight driving state)
    identity_rot = torch.eye(3, device=device, dtype=torch.float32)
    
    # Custom spherical linear interpolation function
    def custom_slerp(rot1, rot2, t):
        """
        Custom spherical linear interpolation, compatible with older scipy versions
        """
        q1 = rot1.as_quat()
        q2 = rot2.as_quat()
        
        # Calculate quaternion dot product
        dot = np.dot(q1, q2)
        
        # If dot product is negative, flip second quaternion to ensure shortest path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are nearly identical, use linear interpolation directly
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            result /= np.linalg.norm(result)
            return R.from_quat(result)
        
        # Calculate interpolation angle
        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        # Spherical linear interpolation
        result = s0 * q1 + s1 * q2
        return R.from_quat(result)
    
    # Calculate phase length for lane change
    duration = end_frame - start_frame
    steer_duration = duration // 2  # Steering phase (first half)
    return_duration = duration - steer_duration  # Straightening phase (second half)
    
    # Apply smooth transition offset and steering for each frame
    for frame_idx in range(target_frames):
        original_rotation = base_trajectory[frame_idx, :3, :3].clone()
        
        if frame_idx < start_frame:
            # Before start frame, keep original trajectory
            continue
            
        elif frame_idx < start_frame + steer_duration:
            # Steering phase (first half)
            progress = (frame_idx - start_frame) / steer_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # Smooth displacement
            current_offset = full_offset * ((frame_idx - start_frame) / duration)
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # Smooth steering
            start_rot = R.from_matrix(identity_rot.cpu().numpy())
            end_rot = R.from_matrix(steer_rot_matrix.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # Apply smooth rotation
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
            
        elif frame_idx <= end_frame:
            # Straightening phase (second half)
            progress = (frame_idx - start_frame - steer_duration) / return_duration
            smooth_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            
            # Continue moving
            current_offset = full_offset * ((frame_idx - start_frame) / duration)
            lane_change_trajectory[frame_idx, :3, 3] += current_offset
            
            # Return from steering to straight driving
            start_rot = R.from_matrix(steer_rot_matrix.cpu().numpy())
            end_rot = R.from_matrix(identity_rot.cpu().numpy())
            interpolated_rot = custom_slerp(start_rot, end_rot, smooth_factor)
            interpolated_rot_matrix = torch.tensor(
                interpolated_rot.as_matrix(), 
                device=device, 
                dtype=torch.float32
            )
            
            # Apply smooth rotation
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ interpolated_rot_matrix
        else:
            # After end frame, maintain new position and straight driving state
            lane_change_trajectory[frame_idx, :3, 3] += full_offset
            # Maintain straight driving state
            lane_change_trajectory[frame_idx, :3, :3] = original_rotation @ identity_rot
    
    return lane_change_trajectory

def s_curve(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Create an S-shaped trajectory using the front three cameras."""
    assert all(
        cam in per_cam_poses.keys() for cam in [0, 1, 2]
    ), "Front three cameras (IDs 0, 1, 2) are required for s_curve"
    key_poses = torch.cat(
        [
            per_cam_poses[0][0:1],
            per_cam_poses[1][original_frames // 4 : original_frames // 4 + 1],
            per_cam_poses[0][original_frames // 2 : original_frames // 2 + 1],
            per_cam_poses[2][3 * original_frames // 4 : 3 * original_frames // 4 + 1],
            per_cam_poses[0][-1:],
        ],
        dim=0,
    )
    return interpolate_poses(key_poses, target_frames)


def three_key_poses_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """
    Create a trajectory using three key poses:
    1. First frame of front center camera
    2. Middle frame with interpolated rotation and position from camera 1 or 2
    3. Last frame of front center camera

    The rotation of the middle pose is calculated using Slerp between
    the start frame and the middle frame of camera 1 or 2.

    Args:
        dataset_type (str): Type of the dataset (e.g., "waymo", "pandaset", etc.).
        per_cam_poses (Dict[int, torch.Tensor]): Dictionary of camera poses.
        original_frames (int): Number of original frames.
        target_frames (int): Number of frames in the output trajectory.

    Returns:
        torch.Tensor: Trajectory of shape (target_frames, 4, 4).
    """
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required"
    assert (
        1 in per_cam_poses.keys() or 2 in per_cam_poses.keys()
    ), "Either camera 1 or camera 2 is required"

    # First key pose: First frame of front center camera
    start_pose = per_cam_poses[0][0]
    key_poses = [start_pose]

    # Select camera for middle frame
    middle_frame = int(original_frames // 2)
    chosen_cam = np.random.choice([1, 2])

    middle_pose = per_cam_poses[chosen_cam][middle_frame]

    # Calculate interpolated rotation for middle pose
    start_rotation = R.from_matrix(start_pose[:3, :3].cpu().numpy())
    middle_rotation = R.from_matrix(middle_pose[:3, :3].cpu().numpy())
    slerp = Slerp(
        [0, 1], R.from_quat([start_rotation.as_quat(), middle_rotation.as_quat()])
    )
    interpolated_rotation = slerp(0.5).as_matrix()

    # Create middle key pose with interpolated rotation and original translation
    middle_key_pose = torch.eye(4, device=start_pose.device)
    middle_key_pose[:3, :3] = torch.tensor(
        interpolated_rotation, device=start_pose.device
    )
    middle_key_pose[:3, 3] = middle_pose[:3, 3]  # Keep the original translation
    key_poses.append(middle_key_pose)

    # Third key pose: Last frame of front center camera
    key_poses.append(per_cam_poses[0][-1])

    # Stack the key poses and interpolate
    key_poses = torch.stack(key_poses)
    return interpolate_poses(key_poses, target_frames)


def circle_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    radius: float = 5.0,
    height: float = 2.0,
) -> torch.Tensor:
    """Generate circular trajectory around the scene"""
    # Fix 1: Correctly get center point coordinates
    center_pose = per_cam_poses[0][original_frames // 2]
    center = center_pose[:3, 3].cpu().numpy()  # [!code --]
    center = center_pose[:3, 3].cpu().numpy()  # [!code ++] Directly get position coordinates

    # Fix 2: Add debug information
    print(f"Center pose shape: {center_pose.shape}")  # Should be (4,4)
    print(f"Center coordinates: {center}")  # Should display 3D coordinates

    # Generate circular trajectory parameters
    angles = np.linspace(0, 2 * np.pi, 12)
    key_poses = []

    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + height  # [!code ++]

        # Ensure coordinate types are correct
        pose = torch.eye(4, device=center_pose.device)
        pose[:3, 3] = torch.tensor([x, y, z], device=center_pose.device)

        # Fix 3: Add direction calculation protection
        direction = center - pose[:3, 3].cpu().numpy()
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([0.0, 0.0, 1.0])  # Prevent zero vector

        pose[:3, :3] = look_at_rotation(
            torch.tensor(direction, device=center_pose.device)
        )
        key_poses.append(pose)

    return interpolate_poses(torch.stack(key_poses), target_frames)


def spiral_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    radius: float = 5.0,
    spiral_height: float = 3.0,
    num_turns: int = 2,
) -> torch.Tensor:
    """Generate spiral ascending trajectory"""
    center_pose = per_cam_poses[0][original_frames // 2]
    center = center_pose[:3, 3].mean(dim=0).cpu().numpy()

    angles = np.linspace(0, num_turns * 2 * np.pi, 12)
    key_poses = []
    for i, angle in enumerate(angles):
        r = radius * (1 - i / len(angles))  # Radius gradually decreases
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        z = center[2] + spiral_height * (i / len(angles))

        pose = torch.eye(4, device=center_pose.device)
        pose[:3, 3] = torch.tensor([x, y, z], device=center_pose.device)
        direction = center - pose[:3, 3].cpu().numpy()
        pose[:3, :3] = look_at_rotation(
            torch.tensor(direction, device=center_pose.device)
        )

        key_poses.append(pose)

    return interpolate_poses(torch.stack(key_poses), target_frames)


def look_around_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    elevation_range: tuple = (-30, 30),
    azimuth_range: tuple = (0, 360),
) -> torch.Tensor:
    """Generate look-around trajectory (fixed position, rotating viewpoint)"""
    center_pose = per_cam_poses[0][original_frames // 2]
    center = center_pose[:3, 3].cpu().numpy()

    # Generate viewing angle parameters
    elevations = np.linspace(*elevation_range, 6)
    azimuths = np.linspace(*azimuth_range, 6)

    key_poses = []
    for elev, azim in zip(elevations, azimuths):
        # Convert spherical coordinates to Cartesian coordinates
        r = np.linalg.norm(center)
        x = r * np.cos(np.radians(azim)) * np.cos(np.radians(elev))
        y = r * np.sin(np.radians(azim)) * np.cos(np.radians(elev))
        z = r * np.sin(np.radians(elev))

        pose = torch.eye(4, device=center_pose.device)
        pose[:3, 3] = torch.tensor([x, y, z], device=center_pose.device)
        direction = center - pose[:3, 3].cpu().numpy()
        pose[:3, :3] = look_at_rotation(
            torch.tensor(direction, device=center_pose.device)
        )

        key_poses.append(pose)

    return interpolate_poses(torch.stack(key_poses), target_frames)