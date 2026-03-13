#!/usr/bin/env python3
"""
Convert ParaLane dataset to DriveStudio processed format.

ParaLane structure (input):
    <paralane_root>/
    └── scene_X/
        └── clip_Y/
            ├── images/{timestamp}/CAMERA_FRONT.png
            ├── foreground_labels/{timestamp}/CAMERA_FRONT.png
            ├── lidars/{lidar_id}.ply   (individual frames, foreground removed)
            ├── video.mp4
            └── sparse/0/
                ├── cameras.txt         (COLMAP intrinsics)
                ├── images.txt          (COLMAP image poses)
                ├── lidar_poses.txt     (Qx Qy Qz Qw X Y Z per lidar frame)
                ├── merged_lidar_pcd.ply
                └── visual_points.ply

DriveStudio structure (output):
    <output_root>/
    └── scene_X_clip_Y/
        ├── images/              {timestep:03d}_0.jpg   (cam_id=0 for CAMERA_FRONT)
        ├── lidar/               {timestep:03d}.bin     (float32 Nx4: x,y,z,intensity)
        ├── ego_pose/            {timestep:03d}.txt     (4x4 cam-to-world matrix)
        ├── extrinsics/          0.txt                  (4x4 cam-to-ego, identity here)
        ├── intrinsics/          0.txt                  ([fx, fy, cx, cy, k1, k2, p1, p2, k3])
        ├── dynamic_masks/       {timestep:03d}_0.png   (coarse foreground mask)
        ├── fine_dynamic_masks/  {timestep:03d}_0.png   (copy of dynamic_masks; refine later)
        ├── instances/           {timestep:03d}_0.png   (instance seg – empty placeholder)
        ├── sky_masks/           {timestep:03d}_0.png   (sky mask – empty placeholder)
        ├── humanpose/           smpl.pkl               (empty placeholder)
        └── instances/           instances_info.json / frame_instances.json

Usage:
    python paralane_to_drivestudio.py \
        --input_dir /path/to/paralane \
        --output_dir /path/to/drivestudio/processed \
        [--scenes 0 1 2]   # optional: only convert specific scene indices
        [--clips 0 1 2]    # optional: only convert specific clip indices
"""

import os
import sys
import glob
import argparse
import shutil
import pickle
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image


# ─────────────────────────────────────────────────────────────
# COLMAP helpers
# ─────────────────────────────────────────────────────────────

def parse_colmap_cameras(cameras_txt):
    """
    Parse COLMAP cameras.txt.
    Returns dict: camera_id -> {'model': str, 'width': int, 'height': int, 'params': list}
    COLMAP camera models and their parameters:
      SIMPLE_PINHOLE: f, cx, cy
      PINHOLE:        fx, fy, cx, cy
      SIMPLE_RADIAL:  f, cx, cy, k1
      RADIAL:         f, cx, cy, k1, k2
      OPENCV:         fx, fy, cx, cy, k1, k2, p1, p2
      (others follow similarly)
    """
    cameras = {}
    with open(cameras_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            cam_id  = int(parts[0])
            model   = parts[1]
            width   = int(parts[2])
            height  = int(parts[3])
            params  = [float(p) for p in parts[4:]]
            cameras[cam_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    return cameras


def colmap_camera_to_K(camera):
    """Convert COLMAP camera params to 3x3 intrinsic matrix K."""
    model  = camera['model']
    params = camera['params']

    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE'):
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model in ('PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV', 'RADIAL', 'RADIAL_FISHEYE'):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model == 'FOV':
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        # Fallback: assume first four are fx, fy, cx, cy
        print(f"  Warning: Unknown camera model '{model}', attempting fx,fy,cx,cy from first 4 params.")
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    return K


def parse_colmap_images(images_txt):
    """
    Parse COLMAP images.txt.
    Returns list of dicts sorted by image name:
      {'image_id': int, 'qw': float, 'qx': float, 'qy': float, 'qz': float,
       'tx': float, 'ty': float, 'tz': float, 'camera_id': int, 'name': str}
    COLMAP stores world-to-camera (w2c): p_cam = R * p_world + t
    """
    images = []
    with open(images_txt, 'r') as f:
        lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 9:
            img = {
                'image_id':  int(parts[0]),
                'qw': float(parts[1]), 'qx': float(parts[2]),
                'qy': float(parts[3]), 'qz': float(parts[4]),
                'tx': float(parts[5]), 'ty': float(parts[6]), 'tz': float(parts[7]),
                'camera_id': int(parts[8]),
                'name':      parts[9] if len(parts) > 9 else '',
            }
            images.append(img)
            i += 2  # skip the 2D point line
        else:
            i += 1

    # Sort by image name (timestamp-based filenames sort chronologically)
    images.sort(key=lambda x: x['name'])
    return images


def colmap_w2c_to_c2w(qw, qx, qy, qz, tx, ty, tz):
    """
    Convert COLMAP world-to-camera quaternion+translation to a 4x4 cam-to-world matrix.
    COLMAP quaternion order: (qw, qx, qy, qz)
    scipy Rotation expects (x, y, z, w)
    """
    R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    t_w2c = np.array([tx, ty, tz])

    # Build 4x4 w2c
    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3,  3] = t_w2c

    # Invert to get cam-to-world (c2w)
    c2w = np.linalg.inv(w2c)
    return c2w


# ─────────────────────────────────────────────────────────────
# LiDAR helpers
# ─────────────────────────────────────────────────────────────

def parse_lidar_poses(lidar_poses_txt):
    """
    Parse lidar_poses.txt.
    Supports two formats:
      Format A:  Qx Qy Qz Qw X Y Z
      Format B:  filename.ply  Qx Qy Qz Qw X Y Z   (ParaLane actual format)
    Returns (list_of_matrices, dict_of_stem->matrix).
    """
    poses = []
    pose_map = {}  # filename stem (without .ply) -> matrix
    with open(lidar_poses_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # Detect if first token is a filename (non-numeric)
            filename = None
            try:
                float(parts[0])
                float_parts = parts  # all numeric, Format A
            except ValueError:
                filename = parts[0]  # first token is filename, Format B
                float_parts = parts[1:]

            if len(float_parts) < 7:
                continue
            vals = [float(v) for v in float_parts[:7]]
            qx, qy, qz, qw = vals[0], vals[1], vals[2], vals[3]
            x,  y,  z       = vals[4], vals[5], vals[6]

            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            mat = np.eye(4)
            mat[:3, :3] = R
            mat[:3,  3] = [x, y, z]
            poses.append(mat)
            if filename:
                stem = filename.replace('.ply', '')
                pose_map[stem] = mat
    return poses, pose_map


def ply_to_bin(ply_path):
    """
    Load a PLY point cloud and return an Nx4 float32 numpy array (x, y, z, intensity).
    If the PLY has no intensity field, intensity is set to 0.
    Handles both ASCII and binary PLY files.
    """
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        x = np.asarray(vertex['x'], dtype=np.float32)
        y = np.asarray(vertex['y'], dtype=np.float32)
        z = np.asarray(vertex['z'], dtype=np.float32)
        # Try common intensity field names
        intensity = None
        for field in ('intensity', 'i', 'scalar_intensity', 'Intensity'):
            if field in vertex.data.dtype.names:
                intensity = np.asarray(vertex[field], dtype=np.float32)
                break
        if intensity is None:
            intensity = np.zeros_like(x)
        return np.stack([x, y, z, intensity], axis=1)
    except Exception:
        # Fallback: manual PLY reader supporting both ASCII and binary.
        return _read_ply_fallback(ply_path)


def _read_ply_fallback(ply_path):
    """Minimal PLY reader fallback supporting ASCII and binary vertex clouds."""
    type_to_struct = {
        'char': 'b',
        'uchar': 'B',
        'short': 'h',
        'ushort': 'H',
        'int': 'i',
        'uint': 'I',
        'float': 'f',
        'float32': 'f',
        'double': 'd',
        'float64': 'd',
    }

    with open(ply_path, 'rb') as f:
        # Parse header
        fmt = None
        num_vertices = 0
        in_vertex = False
        vertex_props = []  # list of (name, type)
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Invalid PLY header in {ply_path}")
            line = raw.decode('ascii', errors='replace').strip()
            if line.startswith('format '):
                # e.g. "format binary_little_endian 1.0"
                fmt = line.split()[1]
            elif line.startswith('element '):
                parts = line.split()
                in_vertex = (len(parts) >= 3 and parts[1] == 'vertex')
                if in_vertex:
                    num_vertices = int(parts[2])
            elif line.startswith('property ') and in_vertex:
                parts = line.split()
                if len(parts) == 3:
                    ptype, pname = parts[1], parts[2]
                    if ptype not in type_to_struct:
                        raise ValueError(f"Unsupported PLY property type '{ptype}' in {ply_path}")
                    vertex_props.append((pname, ptype))
                elif len(parts) >= 5 and parts[1] == 'list':
                    # List properties are not expected for point-cloud vertices.
                    raise ValueError(f"Unsupported list property in vertex element for {ply_path}")
            elif line == 'end_header':
                break

        if fmt is None:
            raise ValueError(f"Missing PLY format declaration in {ply_path}")
        if num_vertices <= 0:
            return np.zeros((0, 4), dtype=np.float32)

        # Identify property indices
        prop_names = [n for n, _ in vertex_props]
        def _find_idx(candidates):
            for c in candidates:
                if c in prop_names:
                    return prop_names.index(c)
            return -1

        ix = _find_idx(['x'])
        iy = _find_idx(['y'])
        iz = _find_idx(['z'])
        ii = _find_idx(['intensity', 'i', 'scalar_intensity', 'Intensity'])

        if min(ix, iy, iz) < 0:
            raise ValueError(f"PLY missing x/y/z vertex properties: {ply_path}")

        points = np.zeros((num_vertices, 4), dtype=np.float32)

        if fmt == 'ascii':
            for row in range(num_vertices):
                line = f.readline().decode('ascii', errors='replace').strip()
                if not line:
                    continue
                vals = line.split()
                points[row, 0] = float(vals[ix])
                points[row, 1] = float(vals[iy])
                points[row, 2] = float(vals[iz])
                points[row, 3] = float(vals[ii]) if (ii >= 0 and ii < len(vals)) else 0.0
            return points

        if fmt not in ('binary_little_endian', 'binary_big_endian'):
            raise ValueError(f"Unsupported PLY format '{fmt}' in {ply_path}")

        endian = '<' if fmt == 'binary_little_endian' else '>'
        struct_fmt = endian + ''.join(type_to_struct[t] for _, t in vertex_props)
        stride = struct.calcsize(struct_fmt)

        for row in range(num_vertices):
            chunk = f.read(stride)
            if len(chunk) != stride:
                raise ValueError(f"Unexpected EOF while reading vertex {row} in {ply_path}")
            vals = struct.unpack(struct_fmt, chunk)
            points[row, 0] = float(vals[ix])
            points[row, 1] = float(vals[iy])
            points[row, 2] = float(vals[iz])
            points[row, 3] = float(vals[ii]) if ii >= 0 else 0.0

        return points


def estimate_similarity_transform(src_pts, dst_pts):
    """Estimate similarity transform dst ~= s * R * src + t with Umeyama alignment."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src_pts and dst_pts must have shape (N, 3)")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points for similarity transform estimation")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    if var_src < 1e-12:
        raise ValueError("Source trajectory variance too small for stable alignment")
    scale = np.sum(D * np.diag(S)) / var_src
    t = mu_dst - scale * (R @ mu_src)
    return float(scale), R, t


def average_rotation_matrices(rotations):
    """Average rotation matrices and project back to SO(3)."""
    if len(rotations) == 0:
        return np.eye(3)
    M = np.zeros((3, 3), dtype=np.float64)
    for R in rotations:
        M += R
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


# ─────────────────────────────────────────────────────────────
# Main conversion logic
# ─────────────────────────────────────────────────────────────

def convert_clip(paralane_clip_dir, output_scene_dir, verbose=True):
    """
    Convert a single ParaLane clip (e.g. scene_0/clip_0) into DriveStudio format.

    Args:
        paralane_clip_dir : str  – e.g. /data/paralane/scene_0/clip_0
        output_scene_dir  : str  – e.g. /data/drivestudio/processed/scene_0_clip_0
    """
    clip_path  = Path(paralane_clip_dir)
    out_path   = Path(output_scene_dir)
    sparse_dir = clip_path / 'sparse' / '0'
    images_dir = clip_path / 'images'
    masks_dir  = clip_path / 'foreground_labels'
    lidars_dir = clip_path / 'lidars'

    # ── Validate input ─────────────────────────────────────────
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip directory not found: {clip_path}")
    if not sparse_dir.exists():
        raise FileNotFoundError(f"sparse/0 directory not found: {sparse_dir}")

    cameras_txt    = sparse_dir / 'cameras.txt'
    images_txt     = sparse_dir / 'images.txt'
    lidar_poses_txt = sparse_dir / 'lidar_poses.txt'

    for f in [cameras_txt, images_txt]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    # ── Create output directories ───────────────────────────────
    for subdir in ['images', 'lidar', 'ego_pose', 'extrinsics', 'intrinsics',
                   'instances', 'humanpose',
                   # dynamic_masks and fine_dynamic_masks have all/human/vehicle subdirs
                   'dynamic_masks/all', 'dynamic_masks/human', 'dynamic_masks/vehicle',
                   'fine_dynamic_masks/all', 'fine_dynamic_masks/human', 'fine_dynamic_masks/vehicle',
                   'sky_masks']:
        (out_path / subdir).mkdir(parents=True, exist_ok=True)

    # KITTI one-camera setting used by this repo expects 1242x375.
    target_width = 1242
    target_height = 375

    # ── Parse COLMAP cameras (intrinsics) ───────────────────────
    if verbose:
        print(f"  Parsing COLMAP cameras: {cameras_txt}")
    cameras = parse_colmap_cameras(str(cameras_txt))

    # Use first camera (ParaLane only releases front camera data)
    cam_id_colmap = list(cameras.keys())[0]
    camera_info   = cameras[cam_id_colmap]
    K = colmap_camera_to_K(camera_info)

    # Save intrinsics for camera 0.
    # KITTI loader expects [fx, fy, cx, cy, k1, k2, p1, p2, k3].
    src_width = float(camera_info['width'])
    src_height = float(camera_info['height'])
    sx = target_width / src_width
    sy = target_height / src_height
    fx = K[0,0] * sx
    fy = K[1,1] * sy
    cx = K[0,2] * sx
    cy = K[1,2] * sy
    np.savetxt(
        str(out_path / 'intrinsics' / '0.txt'),
        np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0]),
        fmt='%.10f'
    )

    # Keep camera 1 calibration as a duplicate of camera 0 so code paths that
    # touch both cameras (e.g., optional SMPL routines) do not fail on missing files.
    np.savetxt(
        str(out_path / 'intrinsics' / '1.txt'),
        np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0]),
        fmt='%.10f'
    )
    if verbose:
        print(f"  Saved intrinsics/0.txt  (camera model: {camera_info['model']})")

    # Initialize extrinsics as identity. We will refine camera->ego from matched
    # lidar/camera poses if lidar_poses.txt is available.
    np.savetxt(str(out_path / 'extrinsics' / '0.txt'), np.eye(4), fmt='%.10f')
    np.savetxt(str(out_path / 'extrinsics' / '1.txt'), np.eye(4), fmt='%.10f')

    # ── Parse COLMAP images (per-frame poses) ───────────────────
    if verbose:
        print(f"  Parsing COLMAP images:  {images_txt}")
    colmap_images = parse_colmap_images(str(images_txt))

    if not colmap_images:
        print(f"  WARNING: No images found in {images_txt}. Skipping clip.")
        return 0

    if verbose:
        print(f"  Found {len(colmap_images)} frames in COLMAP reconstruction.")

    # ── Find actual image files on disk ─────────────────────────
    # Images are stored as images/{timestamp}/CAMERA_FRONT.png
    # We need to match COLMAP image names to actual files.
    # COLMAP image 'name' field may be like '1718346704099615/CAMERA_FRONT.png'
    # or just 'CAMERA_FRONT.png' inside a timestamped folder.

    # Get all timestamp directories sorted
    timestamp_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if verbose:
        print(f"  Found {len(timestamp_dirs)} timestamp directories in images/")

    # Parse lidar poses early so ego_pose can follow KITTI convention:
    # ego_pose is the lidar/ego pose in world coordinates for each frame.
    lidar_poses = []
    pose_map = {}
    if lidar_poses_txt.exists():
        lidar_poses, pose_map = parse_lidar_poses(str(lidar_poses_txt))
        if verbose:
            print(f"  Found {len(lidar_poses)} lidar poses in lidar_poses.txt")

    # ── Process each frame ──────────────────────────────────────
    n_converted = 0
    timestamp_to_timestep = {}
    matched_cam_poses = []
    matched_lidar_poses = []
    for timestep_idx, img_entry in enumerate(colmap_images):
        img_name = img_entry['name']  # e.g. '1718346704099615/CAMERA_FRONT.png'

        # Compute cam-to-world pose
        c2w = colmap_w2c_to_c2w(
            img_entry['qw'], img_entry['qx'], img_entry['qy'], img_entry['qz'],
            img_entry['tx'], img_entry['ty'], img_entry['tz']
        )

        # Find source image
        # Try to locate the image via the COLMAP name or by index
        src_img_path = None

        # Strategy 1: direct path relative to images_dir
        candidate = images_dir / img_name
        if candidate.exists():
            src_img_path = candidate
        else:
            # Strategy 2: strip directory part of name, look for CAMERA_FRONT.png
            # in the Nth timestamp dir
            ts_name = img_name.split('/')[0] if '/' in img_name else None
            if ts_name:
                candidate2 = images_dir / ts_name / 'CAMERA_FRONT.png'
                if candidate2.exists():
                    src_img_path = candidate2
            # Strategy 3: fall back to Nth timestamp directory
            if src_img_path is None and timestep_idx < len(timestamp_dirs):
                candidate3 = timestamp_dirs[timestep_idx] / 'CAMERA_FRONT.png'
                if candidate3.exists():
                    src_img_path = candidate3

        if src_img_path is None:
            if verbose:
                print(f"  WARNING: Could not find image for frame {timestep_idx} (name={img_name})")
            continue

        # Copy image → {timestep:03d}_0.jpg (KITTI-compatible resolution)
        dst_img = out_path / 'images' / f'{timestep_idx:03d}_0.jpg'
        img = Image.open(str(src_img_path)).convert('RGB')
        if img.size != (target_width, target_height):
            img = img.resize((target_width, target_height), Image.BILINEAR)
        img.save(str(dst_img), quality=95)
        ts_part = src_img_path.parent.name
        timestamp_to_timestep[ts_part] = timestep_idx

        # Save ego_pose. Prefer lidar pose when timestamp is available.
        ego_pose = pose_map.get(ts_part, None)
        if ego_pose is None:
            ego_pose = c2w
        else:
            matched_cam_poses.append(c2w)
            matched_lidar_poses.append(ego_pose)
        pose_filename = out_path / 'ego_pose' / f'{timestep_idx:03d}.txt'
        np.savetxt(str(pose_filename), ego_pose, fmt='%.10f')

        # Copy dynamic mask (foreground mask)
        # Try matching mask to same timestamp directory
        ts_part = src_img_path.parent.name  # the timestamp string
        src_mask_path = masks_dir / ts_part / 'CAMERA_FRONT.png'
        if not src_mask_path.exists():
            # Try matching by index
            mask_dirs = sorted([d for d in masks_dir.iterdir() if d.is_dir()]) if masks_dir.exists() else []
            if timestep_idx < len(mask_dirs):
                src_mask_path = mask_dirs[timestep_idx] / 'CAMERA_FRONT.png'

        h, w = img.height, img.width
        prefix = f'{timestep_idx:03d}_0'
        empty = Image.fromarray(np.zeros((h, w), dtype=np.uint8))

        if src_mask_path.exists():
            # all/ = combined foreground mask (everything moving)
            dyn = Image.open(str(src_mask_path)).convert('L')
            if dyn.size != (target_width, target_height):
                dyn = dyn.resize((target_width, target_height), Image.NEAREST)
            dyn.save(str(out_path / 'dynamic_masks' / 'all' / f'{prefix}.png'))
            dyn.save(str(out_path / 'fine_dynamic_masks' / 'all' / f'{prefix}.png'))
        else:
            if verbose and n_converted == 0:
                print(f"  NOTE: No foreground mask found – creating empty dynamic/fine_dynamic masks.")
            empty.save(str(out_path / 'dynamic_masks'      / 'all' / f'{prefix}.png'))
            empty.save(str(out_path / 'fine_dynamic_masks' / 'all' / f'{prefix}.png'))

        # human/ and vehicle/ – ParaLane has no per-class labels so these are empty placeholders.
        # You can populate them later with a panoptic segmentation model (e.g. Mask2Former).
        empty.save(str(out_path / 'dynamic_masks'      / 'human'   / f'{prefix}.png'))
        empty.save(str(out_path / 'dynamic_masks'      / 'vehicle' / f'{prefix}.png'))
        empty.save(str(out_path / 'fine_dynamic_masks' / 'human'   / f'{prefix}.png'))
        empty.save(str(out_path / 'fine_dynamic_masks' / 'vehicle' / f'{prefix}.png'))

        # instances: instance-segmentation label image (uint16, 0 = background).
        inst_arr = np.zeros((h, w), dtype=np.uint16)
        Image.fromarray(inst_arr).save(str(out_path / 'instances' / f'{prefix}.png'))

        # sky_masks/{timestep}_0.png  – placeholder (all zeros = no sky).
        # Replace with e.g. SegFormer sky segmentation output.
        empty.save(str(out_path / 'sky_masks' / f'{prefix}.png'))

        n_converted += 1

    if verbose:
        print(f"  Converted {n_converted} frames (images + poses + masks).")

    # Refine extrinsics from matched lidar/camera poses if possible.
    if len(matched_cam_poses) >= 3:
        cam_centers = np.stack([p[:3, 3] for p in matched_cam_poses], axis=0)
        lidar_centers = np.stack([p[:3, 3] for p in matched_lidar_poses], axis=0)
        try:
            scale, R_align, t_align = estimate_similarity_transform(cam_centers, lidar_centers)

            cam_to_ego_candidates = []
            for cam_pose, lidar_pose in zip(matched_cam_poses, matched_lidar_poses):
                cam_pose_aligned = np.eye(4, dtype=np.float64)
                cam_pose_aligned[:3, :3] = R_align @ cam_pose[:3, :3]
                cam_pose_aligned[:3, 3] = scale * (R_align @ cam_pose[:3, 3]) + t_align
                cam_to_ego = np.linalg.inv(lidar_pose) @ cam_pose_aligned
                cam_to_ego_candidates.append(cam_to_ego)

            R_avg = average_rotation_matrices([m[:3, :3] for m in cam_to_ego_candidates])
            t_avg = np.mean(np.stack([m[:3, 3] for m in cam_to_ego_candidates], axis=0), axis=0)
            cam_to_ego_final = np.eye(4, dtype=np.float64)
            cam_to_ego_final[:3, :3] = R_avg
            cam_to_ego_final[:3, 3] = t_avg

            np.savetxt(str(out_path / 'extrinsics' / '0.txt'), cam_to_ego_final, fmt='%.10f')
            np.savetxt(str(out_path / 'extrinsics' / '1.txt'), cam_to_ego_final, fmt='%.10f')
            if verbose:
                print(
                    f"  Refined extrinsics/0.txt from {len(cam_to_ego_candidates)} matched poses "
                    f"(similarity scale={scale:.6f})"
                )
        except Exception as e:
            if verbose:
                print(f"  WARNING: Failed trajectory alignment for extrinsics, keeping identity. Reason: {e}")
    elif verbose:
        print("  Using identity extrinsics (insufficient matched lidar/camera pose timestamps)")

    # Minimal KITTI-style placeholders expected by the loader.
    # Empty objects metadata is valid and keeps code paths consistent.
    import json
    with open(str(out_path / 'instances' / 'instances_info.json'), 'w') as jf:
        json.dump({}, jf)
    with open(str(out_path / 'instances' / 'frame_instances.json'), 'w') as jf:
        json.dump({str(i): [] for i in range(n_converted)}, jf)
    with open(str(out_path / 'humanpose' / 'smpl.pkl'), 'wb') as f:
        pickle.dump({}, f)

    # ── Process LiDAR ──────────────────────────────────────────
    # Individual frames are in lidars/ as PLY files.
    # lidar_poses.txt has one line per lidar frame.

    lidar_ply_files = []
    if lidars_dir.exists():
        lidar_ply_files = sorted(lidars_dir.glob('*.ply'))

    # Build destination index for each lidar frame so lidar/{t:03d}.bin aligns with image timesteps.
    lidar_dst_map = {}
    for lidar_idx, ply_file in enumerate(lidar_ply_files):
        stem = ply_file.stem
        if stem in timestamp_to_timestep:
            lidar_dst_map[ply_file] = timestamp_to_timestep[stem]
        elif lidar_idx < n_converted:
            lidar_dst_map[ply_file] = lidar_idx

    if lidar_ply_files:
        if verbose:
            print(f"  Found {len(lidar_ply_files)} lidar PLY files. Converting to .bin …")

        # Check if plyfile is available
        try:
            import plyfile
            has_plyfile = True
        except ImportError:
            has_plyfile = False
            print("  NOTE: 'plyfile' package not found. Using built-in PLY fallback reader.")
            print("        Install with: pip install plyfile")

        for lidar_idx, ply_file in enumerate(lidar_ply_files):
            # Convert PLY -> Nx4 float32
            pts = ply_to_bin(str(ply_file))

            # Keep lidar points in ego/lidar coordinates. KITTI loader applies ego_pose
            # per frame; pre-transforming to world here would transform points twice.

            dst_t = lidar_dst_map.get(ply_file, None)
            if dst_t is None:
                continue

            # Save as binary .bin  (float32 Nx4: x,y,z,intensity)
            dst_bin = out_path / 'lidar' / f'{dst_t:03d}.bin'
            pts.astype(np.float32).tofile(str(dst_bin))

        if verbose:
            print(f"  Saved {len(lidar_dst_map)} aligned lidar frames to lidar/")

    elif lidar_poses_txt.exists():
        if verbose:
            print(f"  No individual lidar PLY frames found in lidars/ – skipping lidar output.")
    else:
        if verbose:
            print(f"  No lidar data found for this clip. Skipping lidar conversion.")

    # Ensure lidar files exist for every converted frame (empty scans are allowed).
    for t in range(n_converted):
        lidar_path = out_path / 'lidar' / f'{t:03d}.bin'
        if not lidar_path.exists():
            np.zeros((0, 4), dtype=np.float32).tofile(str(lidar_path))

    return n_converted


def convert_dataset(paralane_root, output_root, scenes=None, clips=None, verbose=True):
    """
    Convert all (or selected) scenes/clips from ParaLane to DriveStudio format.

    Args:
        paralane_root : str – root of the paralane dataset
        output_root   : str – root of the drivestudio output
        scenes        : list[int] or None – scene indices to convert (None = all)
        clips         : list[int] or None – clip indices to convert (None = all)
    """
    paralane_root = Path(paralane_root)
    output_root   = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover all scene directories
    all_scene_dirs = sorted([d for d in paralane_root.iterdir()
                              if d.is_dir() and d.name.startswith('scene_')],
                            key=lambda d: int(d.name.split('_')[1]))

    if not all_scene_dirs:
        print(f"ERROR: No 'scene_*' directories found in {paralane_root}")
        sys.exit(1)

    total_frames = 0
    total_clips  = 0

    for scene_dir in all_scene_dirs:
        scene_idx = int(scene_dir.name.split('_')[1])
        if scenes is not None and scene_idx not in scenes:
            continue

        # Discover clip directories.
        # Some ParaLane exports store data directly under scene_x without clip_* folders.
        clip_dirs = sorted([d for d in scene_dir.iterdir()
                            if d.is_dir() and d.name.startswith('clip_')],
                           key=lambda d: int(d.name.split('_')[1]))

        # Fallback: treat scene directory itself as a single clip (clip_0).
        if not clip_dirs:
            if (scene_dir / 'sparse' / '0').exists() and (scene_dir / 'images').exists():
                clip_dirs = [scene_dir]

        for clip_dir in clip_dirs:
            if clip_dir == scene_dir:
                clip_idx = 0
            else:
                clip_idx = int(clip_dir.name.split('_')[1])
            if clips is not None and clip_idx not in clips:
                continue

            output_name = f"scene_{scene_idx:03d}_clip_{clip_idx:03d}"
            out_dir     = output_root / output_name

            print(f"\n{'='*60}")
            print(f"  Converting: {clip_dir}")
            print(f"  Output:     {out_dir}")
            print(f"{'='*60}")

            try:
                n = convert_clip(str(clip_dir), str(out_dir), verbose=verbose)
                total_frames += n
                total_clips  += 1
            except Exception as e:
                print(f"  ERROR processing {clip_dir}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*60}")
    print(f"  Done! Converted {total_clips} clips, {total_frames} total frames.")
    print(f"  Output root: {output_root}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert ParaLane dataset to DriveStudio processed format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input_dir',  required=True,
                        help='Root directory of the ParaLane dataset (contains scene_0/, scene_1/, …)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for DriveStudio processed data')
    parser.add_argument('--scenes', type=int, nargs='+', default=None,
                        help='Scene indices to convert (default: all). E.g. --scenes 0 1 2')
    parser.add_argument('--clips',  type=int, nargs='+', default=None,
                        help='Clip indices to convert (default: all). E.g. --clips 0 1')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()

    # Check dependencies
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        missing.append('scipy')
    try:
        from PIL import Image
    except ImportError:
        missing.append('Pillow')
    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    convert_dataset(
        paralane_root=args.input_dir,
        output_root=args.output_dir,
        scenes=args.scenes,
        clips=args.clips,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
