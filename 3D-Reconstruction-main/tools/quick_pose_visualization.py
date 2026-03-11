
#!/usr/bin/env python3
"""
Quick camera pose viewer
Visualize camera trajectories easily and quickly

How to use:
python quick_pose_viewer.py pose_file.npz
python quick_pose_viewer.py pose_file.npz --show # Display the picture directly
python quick_pose_viewer.py pose_file.npz --output result.png #Specify output
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json


def load_poses(npz_path):
    """Quickly load pose data"""
    print(f"Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    poses_dict = {key: data[key] for key in data.keys()}
    
    print(f"Loaded {len(poses_dict['frame_indices'])} frames")
    print(f"Cameras: {set(poses_dict['cam_names'])}")
    
    return poses_dict


def quick_stats(poses_dict):
    """Quick statistics"""
    positions = np.array(poses_dict['camera_positions'])
    
    print("\n=== QUICK STATS ===")
    print(f"Total frames: {len(poses_dict['frame_indices'])}")
    print(f"Frame range: {min(poses_dict['frame_indices'])} - {max(poses_dict['frame_indices'])}")
    
    print(f"\nPosition ranges:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m")
    
    if len(positions) > 1:
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        print(f"\nMovement:")
        print(f"  Total distance: {distances.sum():.3f} m")
        print(f"  Average speed: {distances.mean():.3f} m/frame")
        print(f"  Max speed: {distances.max():.3f} m/frame")


def quick_plot(poses_dict, output_path=None, show=False):
    """Quickly draw trajectories"""
    print("\nGenerating plots...")
    
    positions = np.array(poses_dict['camera_positions'])
    cam_names = poses_dict['cam_names']
    frame_indices = poses_dict['frame_indices']
    
    # Create a color map
    unique_cameras = list(set(cam_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cameras)))
    camera_colors = {cam: colors[i] for i, cam in enumerate(unique_cameras)}
    
    # Create a 2x2 subplot
    fig = plt.figure(figsize=(14, 10))
    
    # 1. 3D trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    for cam_name in unique_cameras:
        mask = np.array(cam_names) == cam_name
        cam_pos = positions[mask]
        if len(cam_pos) > 0:
            ax1.plot(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2], 
                    'o-', color=camera_colors[cam_name], markersize=3, 
                    linewidth=1.5, label=cam_name, alpha=0.8)
    
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='green', s=80, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='red', s=80, marker='s', label='End')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 3D等比例坐标轴
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    x_mid = (x.max() + x.min()) / 2.0
    y_mid = (y.max() + y.min()) / 2.0
    z_mid = (z.max() + z.min()) / 2.0
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
    if max_range == 0:
        max_range = 1.0

    ax1.set_xlim(x_mid - max_range / 2.0, x_mid + max_range / 2.0)
    ax1.set_ylim(y_mid - max_range / 2.0, y_mid + max_range / 2.0)
    ax1.set_zlim(z_mid - max_range / 2.0, z_mid + max_range / 2.0)
    ax1.set_box_aspect((1, 1, 1))
    
    # 2. XY plane
    ax2 = fig.add_subplot(222)
    for cam_name in unique_cameras:
        mask = np.array(cam_names) == cam_name
        cam_pos = positions[mask]
        if len(cam_pos) > 0:
            ax2.plot(cam_pos[:, 0], cam_pos[:, 1], 
                    'o-', color=camera_colors[cam_name], markersize=4, 
                    linewidth=2, label=cam_name, alpha=0.8)
    
    ax2.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
    ax2.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY View (Top Down)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 3. Z height change
    ax3 = fig.add_subplot(223)
    for cam_name in unique_cameras:
        mask = np.array(cam_names) == cam_name
        cam_pos = positions[mask]
        cam_frames = np.array(frame_indices)[mask]
        if len(cam_pos) > 0:
            ax3.plot(cam_frames, cam_pos[:, 2], 
                    'o-', color=camera_colors[cam_name], markersize=3, 
                    linewidth=2, label=cam_name, alpha=0.8)
    
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Z Height (m)')
    ax3.set_title('Height over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Movement speed
    ax4 = fig.add_subplot(224)
    if len(positions) > 1:
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        ax4.plot(frame_indices[1:], distances, 'b-o', markersize=2, 
                linewidth=1.5, alpha=0.7)
        
        mean_speed = distances.mean()
        ax4.axhline(y=mean_speed, color='r', linestyle='--', alpha=0.7,
                   label=f'Mean: {mean_speed:.3f}m')
        
        ax4.set_xlabel('Frame Index')
        ax4.set_ylabel('Movement (m/frame)')
        ax4.set_title('Movement Speed')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Add general title
    plt.suptitle(f'Camera Trajectory Analysis\n'
                f'{len(positions)} frames, {len(unique_cameras)} cameras', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_poses_every_n_frames(poses_dict, output_dir, interval=10):
    """Save camera pose to txt file every n frames"""
    from scipy.spatial.transform import Rotation as R
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    positions = np.array(poses_dict['camera_positions'])
    rotations = poses_dict['camera_rotations']
    poses = poses_dict['camera_poses']
    frame_indices = poses_dict['frame_indices']
    cam_names = poses_dict['cam_names']
    
    print(f"\n=== SAVING POSES EVERY {interval} FRAMES ===")
    
    # Create the main pose file
    pose_txt_path = output_dir / f"camera_poses_every_{interval}_frames.txt"
    
    with open(pose_txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CAMERA POSES (Every {interval} frames)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated from: {len(frame_indices)} total frames\n")
        f.write(f"Saving interval: {interval} frames\n")
        f.write(f"Total saved frames: {len(range(0, len(frame_indices), interval))}\n\n")
        
        # Write format description
        f.write("FORMAT EXPLANATION:\n")
        f.write("-" * 40 + "\n")
        f.write("Frame Index: Original frame number\n")
        f.write("Camera Name: Camera identifier\n")
        f.write("Position: [X, Y, Z] in world coordinates (meters)\n")
        f.write("Rotation (Euler): [Roll, Pitch, Yaw] in degrees\n")
        f.write("4x4 Matrix: Camera-to-world transformation matrix\n")
        f.write("3x3 Rotation: Rotation matrix only\n\n")
        
        saved_count = 0
        for i in range(0, len(frame_indices), interval):
            frame_idx = frame_indices[i]
            cam_name = cam_names[i]
            pos = positions[i]
            rot = rotations[i]
            pose_4x4 = poses[i]
            
            # Calculate Euler angles
            euler_angles = R.from_matrix(rot).as_euler('xyz', degrees=True)
            
            f.write(f"FRAME {saved_count + 1:03d} (Original Frame {frame_idx})\n")
            f.write("-" * 50 + "\n")
            f.write(f"Camera: {cam_name}\n")
            f.write(f"Position: [{pos[0]:10.6f}, {pos[1]:10.6f}, {pos[2]:10.6f}] m\n")
            f.write(f"Rotation (Euler XYZ): [{euler_angles[0]:8.3f}, {euler_angles[1]:8.3f}, {euler_angles[2]:8.3f}] degrees\n")
            
            # Write a 4x4 transformation matrix
            f.write("4x4 Transformation Matrix (Camera-to-World):\n")
            for row in pose_4x4:
                f.write("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]\n")
            
            # Write a 3x3 rotation matrix
            f.write("3x3 Rotation Matrix:\n")
            for row in rot:
                f.write("  [" + ", ".join(f"{val:12.6f}" for val in row) + "]\n")
            
            f.write("\n")
            saved_count += 1
    
    print(f"Detailed poses saved to: {pose_txt_path}")
    
    # Create a simplified pose file (only positions and Euler angles)
    simple_txt_path = output_dir / f"simple_poses_every_{interval}_frames.txt"
    
    with open(simple_txt_path, 'w') as f:
        f.write(f"# Camera Poses (Every {interval} frames) - Simple Format\n")
        f.write("# Format: Frame_Index Camera_Name X Y Z Roll Pitch Yaw\n")
        f.write("# Position in meters, Rotation in degrees\n")
        f.write("#" + "-" * 70 + "\n")
        
        for i in range(0, len(frame_indices), interval):
            frame_idx = frame_indices[i]
            cam_name = cam_names[i]
            pos = positions[i]
            rot = rotations[i]
            
            # Calculate Euler angles
            euler_angles = R.from_matrix(rot).as_euler('xyz', degrees=True)
            
            f.write(f"{frame_idx:5d} {cam_name:12s} {pos[0]:10.6f} {pos[1]:10.6f} {pos[2]:10.6f} ")
            f.write(f"{euler_angles[0]:8.3f} {euler_angles[1]:8.3f} {euler_angles[2]:8.3f}\n")
    
    print(f"Simple poses saved to: {simple_txt_path}")
    
    # Create files in colmap format (every n frames)
    colmap_txt_path = output_dir / f"colmap_images_every_{interval}_frames.txt"
    
    with open(colmap_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image (COLMAP format):\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Generated every {interval} frames\n")
        
        image_id = 1
        for i in range(0, len(frame_indices), interval):
            frame_idx = frame_indices[i]
            cam_name = cam_names[i]
            pos = positions[i]
            rot = rotations[i]
            
            # Convert to world-to-camera (COLMAP format)
            w2c_rot = rot.T
            w2c_pos = -w2c_rot @ pos
            
            # Convert to quaternion [w, x, y, z]
            quat = R.from_matrix(w2c_rot).as_quat()  # [x, y, z, w]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            f.write(f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} ")
            f.write(f"{w2c_pos[0]:.6f} {w2c_pos[1]:.6f} {w2c_pos[2]:.6f} ")
            f.write(f"1 {cam_name}_frame_{frame_idx:04d}.jpg\n")
            f.write("\n")  # Blank lines indicate no 2D points
            
            image_id += 1
    
    print(f"COLMAP format saved to: {colmap_txt_path}")
    
    # Create files in tum format (every n frames)
    tum_txt_path = output_dir / f"tum_trajectory_every_{interval}_frames.txt"
    
    with open(tum_txt_path, 'w') as f:
        f.write(f"# TUM trajectory format (every {interval} frames)\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        f.write("# timestamp as frame_index, pose as camera-to-world\n")
        
        for i in range(0, len(frame_indices), interval):
            frame_idx = frame_indices[i]
            pos = positions[i]
            rot = rotations[i]
            
            # Convert to quaternion [x, y, z, w]
            quat = R.from_matrix(rot).as_quat()
            timestamp = float(frame_idx)
            
            f.write(f"{timestamp:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} ")
            f.write(f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")
    
    print(f"TUM format saved to: {tum_txt_path}")
    
    print(f"Total {saved_count} poses saved (every {interval} frames)")
    return saved_count


def export_summary(poses_dict, output_dir):
    """Export simple summary information"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    positions = np.array(poses_dict['camera_positions'])
    
    # export csv
    csv_path = output_dir / "trajectory.csv"
    with open(csv_path, 'w') as f:
        f.write("frame_idx,cam_name,x,y,z\n")
        for i, (frame_idx, cam_name, pos) in enumerate(zip(
            poses_dict['frame_indices'],
            poses_dict['cam_names'],
            positions
        )):
            f.write(f"{frame_idx},{cam_name},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}\n")
    
    print(f"CSV exported to: {csv_path}")
    
    # Export statistics json
    stats = {
        "total_frames": len(poses_dict['frame_indices']),
        "cameras": list(set(poses_dict['cam_names'])),
        "frame_range": [int(min(poses_dict['frame_indices'])), 
                       int(max(poses_dict['frame_indices']))],
        "position_ranges": {
            "x": [float(positions[:, 0].min()), float(positions[:, 0].max())],
            "y": [float(positions[:, 1].min()), float(positions[:, 1].max())],
            "z": [float(positions[:, 2].min()), float(positions[:, 2].max())]
        }
    }
    
    if len(positions) > 1:
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        stats["movement"] = {
            "total_distance": float(distances.sum()),
            "average_speed": float(distances.mean()),
            "max_speed": float(distances.max())
        }
    
    json_path = output_dir / "stats.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Stats exported to: {json_path}")


def print_coordinate_list(poses_dict, num_frames=10):
    """Print the coordinates of the first few frames"""
    positions = np.array(poses_dict['camera_positions'])
    frame_indices = poses_dict['frame_indices']
    cam_names = poses_dict['cam_names']
    
    print(f"\n=== FIRST {num_frames} FRAMES ===")
    print("Frame | Camera    | X        | Y        | Z")
    print("-" * 50)
    
    for i in range(min(num_frames, len(positions))):
        frame_idx = frame_indices[i]
        cam_name = cam_names[i]
        pos = positions[i]
        print(f"{frame_idx:5d} | {cam_name:9s} | {pos[0]:8.3f} | {pos[1]:8.3f} | {pos[2]:8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Quick camera pose viewer")
    parser.add_argument("npz_file", help="NPZ file containing camera poses")
    parser.add_argument("--output", "-o", help="Output image path (default: auto-generated)")
    parser.add_argument("--show", "-s", action="store_true", help="Show plot interactively")
    parser.add_argument("--export", "-e", help="Export directory for CSV/JSON")
    parser.add_argument("--list", "-l", type=int, default=0, 
                       help="Print first N frame coordinates (default: 0=off)")
    parser.add_argument("--stats-only", action="store_true", 
                       help="Only print statistics, no plotting")
    parser.add_argument("--save-poses", "-p", help="Directory to save pose txt files")
    parser.add_argument("--pose-interval", "-i", type=int, default=10,
                       help="Save poses every N frames (default: 10)")
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.npz_file).exists():
        print(f"Error: File {args.npz_file} not found!")
        sys.exit(1)
    
    try:
        # Load data
        poses_dict = load_poses(args.npz_file)
        
        # Print statistics
        quick_stats(poses_dict)
        
        # Print coordinate list
        if args.list > 0:
            print_coordinate_list(poses_dict, args.list)
        
        # Save pose file
        if args.save_poses:
            save_poses_every_n_frames(poses_dict, args.save_poses, args.pose_interval)
        
        # If you only want statistics, it’s over
        if args.stats_only:
            return
        
        # Generate pictures
        if not args.show and not args.output:
            # Automatically generate output file names
            input_path = Path(args.npz_file)
            output_path = input_path.parent / f"{input_path.stem}_quick_view.png"
        else:
            output_path = args.output
        
        quick_plot(poses_dict, output_path, args.show)
        
        # Export data
        if args.export:
            export_summary(poses_dict, args.export)
        
        print("\nDone!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()