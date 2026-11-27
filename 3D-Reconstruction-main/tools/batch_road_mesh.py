#!/usr/bin/env python3
"""Build, render, and composite road meshes for trained runs.

For every run containing a checkpoint and ``config.yaml`` this script loads
only the scene dataset, builds a textured mesh from road-filtered LiDAR,
aligns clip_0 and clip_1 by using clip_2 as the reference (as ``eval.py``
does), saves transparent mesh renders, and composites those over the matching
Gaussian renders in ``videos_eval/novel_<step>``.

Heavy project modules and Open3D are imported lazily, allowing ``--help`` and
``--dry-run`` to work in lightweight environments.
"""
32r
from __future__ import annotations

import argparse
import gc
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("batch_road_mesh")
DEFAULT_CHECKPOINT_PATTERN = "checkpoint_final.pth"
TRAJECTORY_NAMES = ("clip_0", "clip_1")
SUPPORTED_DATASET_LAYOUTS = {
    "argoverse",
    "kitti",
    "nuplan",
    "nuscenes",
    "pandaset",
    "waymo",
}
KITTI_PIXEL_SOURCE = "datasets.kitti.kitti_sourceloader.KITTIPixelSource"
KITTI_LIDAR_SOURCE = "datasets.kitti.kitti_sourceloader.KITTILiDARSource"


@dataclass(frozen=True)
class RunSpec:
    """The files needed to process one trained run."""

    run_dir: Path
    checkpoint: Path
    config: Path
    scene_id: str
    scene_dir: Path


def natural_sort_key(path: Path) -> tuple:
    """Sort paths containing frame numbers numerically."""

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    )


def infer_scene_id(run_dir: Path, config_path: Optional[Path] = None) -> Optional[str]:
    """Infer ``scene_###_clip_###`` from a run path or its config."""

    for candidate in (run_dir.name, str(run_dir)):
        match = re.search(r"scene_\d+_clip_\d+", candidate)
        if match:
            return match.group(0)

    config_path = config_path or run_dir / "config.yaml"
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^\s*scene_idx:\s*['\"]?([^\s'\"]+)", config_text, re.MULTILINE)
    if match:
        configured_name = Path(match.group(1).rstrip("/")).name
        scene_match = re.search(r"scene_\d+_clip_\d+", configured_name)
        return scene_match.group(0) if scene_match else configured_name
    return None


def find_run_checkpoints(runs_dir: Path, checkpoint_pattern: str) -> list[Path]:
    """Find checkpoint files deterministically beneath ``runs_dir``."""

    if runs_dir.is_file():
        return [runs_dir] if runs_dir.match(checkpoint_pattern) else []
    return sorted(
        (path for path in runs_dir.rglob(checkpoint_pattern) if path.is_file()),
        key=lambda path: str(path),
    )


def configured_scene_dir(config_path: Path) -> Optional[Path]:
    """Resolve the dataset directory recorded in a run's ``config.yaml``."""

    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    root_match = re.search(
        r"^\s*data_root:\s*['\"]?([^\n#'\"]+)", config_text, re.MULTILINE
    )
    scene_match = re.search(
        r"^\s*scene_idx:\s*['\"]?([^\s#'\"]+)", config_text, re.MULTILINE
    )
    if root_match is None or scene_match is None:
        return None
    data_root = Path(root_match.group(1).strip()).expanduser()
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    scene_name = Path(scene_match.group(1).strip().rstrip("/")).name
    scene_dir = data_root / scene_name
    return scene_dir.resolve() if scene_dir.is_dir() else None


def resolve_scene_dir(scene_id: str, data_root: Path) -> Optional[Path]:
    """Resolve a scene, accepting suffixes such as ``_inpaint`` when needed."""

    exact = data_root / scene_id
    if exact.is_dir():
        return exact
    matches = sorted(path for path in data_root.glob(f"{scene_id}*") if path.is_dir())
    return matches[0] if len(matches) == 1 else None


def discover_runs(
    runs_dir: Path, data_root: Optional[Path], checkpoint_pattern: str
) -> tuple[list[RunSpec], list[str]]:
    """Return processable runs and human-readable discovery warnings."""

    specs: list[RunSpec] = []
    warnings: list[str] = []
    for checkpoint in find_run_checkpoints(runs_dir, checkpoint_pattern):
        run_dir = checkpoint.parent
        config = run_dir / "config.yaml"
        if not config.is_file():
            warnings.append(f"Skipping {checkpoint}: missing {config}")
            continue
        scene_id = infer_scene_id(run_dir, config)
        if not scene_id:
            warnings.append(f"Skipping {checkpoint}: could not infer scene_idx")
            continue
        scene_dir = (
            resolve_scene_dir(scene_id, data_root)
            if data_root is not None
            else configured_scene_dir(config)
        )
        if scene_dir is None:
            location = f"under {data_root}" if data_root is not None else "from config.yaml"
            warnings.append(f"Skipping {checkpoint}: could not resolve scene {location}")
            continue
        specs.append(RunSpec(run_dir, checkpoint, config, scene_id, scene_dir))
    return specs, warnings


def _to_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def normalize_dataset_config(data_cfg) -> None:
    """Map processed ParaLane configs to their actual DriveStudio loader.

    ParaLane scenes in this repository are converted to KITTI layout. Some
    trained configs nevertheless retain ``dataset: paralane`` (or
    ``paralane/1cams``) and source paths for a module that no longer exists.
    Infer the layout from that alias/source and update both source classes.
    """

    configured_type = str(data_cfg.get("dataset", "")).lower().strip()
    base_type = configured_type.split("/", 1)[0]
    pixel_source_type = str(data_cfg.get("pixel_source", {}).get("type", "")).lower()
    lidar_source_type = str(data_cfg.get("lidar_source", {}).get("type", "")).lower()
    uses_paralane_loader = (
        "datasets.paralane." in pixel_source_type
        or "datasets.paralane." in lidar_source_type
    )
    if base_type == "paralane" or uses_paralane_loader:
        LOGGER.info(
            "Using KITTI dataset layout and source loaders for configured type %r",
            configured_type,
        )
        data_cfg.dataset = "kitti"
        if "pixel_source" not in data_cfg:
            raise ValueError("ParaLane config has no pixel_source section")
        if "lidar_source" not in data_cfg:
            raise ValueError("ParaLane config has no lidar_source section")
        data_cfg.pixel_source.type = KITTI_PIXEL_SOURCE
        data_cfg.lidar_source.type = KITTI_LIDAR_SOURCE
        return

    if base_type in SUPPORTED_DATASET_LAYOUTS:
        # Values such as kitti/1cams still need the base layout name.
        data_cfg.dataset = base_type
        return

    raise ValueError(
        f"Unsupported configured dataset type {configured_type!r}; "
        f"could not infer a layout from pixel source {pixel_source_type!r}"
    )


def load_dataset(spec: RunSpec, device):
    """Load only the dataset described by a run config (never its checkpoint)."""

    from omegaconf import OmegaConf

    from datasets.driving_dataset import DrivingDataset

    cfg = OmegaConf.load(spec.config)
    if "data" not in cfg:
        raise ValueError(f"Config has no data section: {spec.config}")
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    # Keep the stored dataset settings, changing only the path representation to
    # the already-resolved scene. This also supports runs moved with --data-root.
    data_cfg.data_root = str(spec.scene_dir.parent)
    data_cfg.scene_idx = spec.scene_dir.name
    data_cfg.preload_device = device.type
    normalize_dataset_config(data_cfg)
    return DrivingDataset(data_cfg=data_cfg)


def build_road_mesh(dataset, device, args):
    """Create and texture a road mesh using the existing training implementation."""

    from models.road_mesh import RoadMesh
    from tools.train_mesh_road import aggregate_road_pointcloud

    points, colors = aggregate_road_pointcloud(dataset, device)
    if len(points) == 0:
        raise RuntimeError("No LiDAR points projected into road masks")
    points_np = _to_numpy(points, np.float32)
    colors_np = _to_numpy(colors, np.float32) if colors is not None and len(colors) else None
    mesh = RoadMesh.from_pointcloud(
        points_np,
        colors_np,
        cell_size=args.cell_size,
        vertex_selection=args.vertex_selection,
        device=device,
    )
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("Road point cloud did not produce any mesh faces")
    mesh.initialize_texture_buffer(args.texture_pixels_per_meter)
    stats = mesh.project_images_onto_buffer(
        dataset,
        num_frames=args.num_projection_frames,
        rays_per_image=args.rays_per_image,
    )
    if mesh.texture_buffer is None:
        raise RuntimeError("Road mesh texture buffer was not initialized")
    return mesh, stats


def load_and_align_trajectory(dataset, trajectory_path: Path, reference_path: Path, device):
    """Apply the same clip_2 rigid alignment used by ``tools/eval.py``."""

    from tools.eval import (
        _apply_rigid_alignment_to_trajectory,
        _estimate_rigid_alignment_from_reference,
        _load_novel_trajectory_from_file,
    )

    trajectory = _load_novel_trajectory_from_file(str(trajectory_path), device=device)
    raw_reference = _load_novel_trajectory_from_file(str(reference_path), device=device)
    pixel_source = dataset.pixel_source
    camera_id = pixel_source.camera_list[0]
    dataset_reference = pixel_source.camera_data[camera_id].cam_to_worlds.to(device)
    rotation, translation = _estimate_rigid_alignment_from_reference(
        raw_reference, dataset_reference
    )
    return _apply_rigid_alignment_to_trajectory(trajectory, rotation, translation)


def get_camera_calibration(dataset) -> tuple[np.ndarray, int, int]:
    """Get the same camera/intrinsics selected for novel rendering by the dataset."""

    pixel_source = dataset.pixel_source
    camera_id = 1 if getattr(dataset, "type", "") == "argoverse" else 0
    if camera_id not in pixel_source.camera_data:
        camera_id = pixel_source.camera_list[0]
    camera = pixel_source.camera_data[camera_id]
    intrinsic = _to_numpy(camera.intrinsics[0], np.float64)
    return intrinsic, int(camera.HEIGHT), int(camera.WIDTH)


def _make_open3d_mesh(mesh):
    """Convert RoadMesh geometry and its XY texture mapping to Open3D.

    Return the texture image separately so Python retains its owning pybind11
    holder. Accessing it again through ``geometry.textures[0]`` returns a
    borrowed reference with Open3D 0.16 and cannot be assigned to
    ``MaterialRecord.albedo_img``.
    """

    import cv2
    import open3d as o3d

    vertices = _to_numpy(mesh.vertices, np.float64)
    faces = _to_numpy(mesh.faces, np.int32)
    geometry = o3d.geometry.TriangleMesh()
    geometry.vertices = o3d.utility.Vector3dVector(vertices)
    geometry.triangles = o3d.utility.Vector3iVector(faces)
    geometry.compute_vertex_normals()

    metadata = mesh.texture_metadata
    texture = mesh.texture_buffer
    if texture is None or not metadata:
        raise ValueError("A textured RoadMesh is required for rendering")
    x_min, x_max = metadata["x_range"]
    y_min, y_max = metadata["y_range"]
    x_span = max(float(x_max) - float(x_min), 1e-8)
    y_span = max(float(y_max) - float(y_min), 1e-8)
    uv = np.column_stack(
        (
            np.clip((vertices[:, 0] - x_min) / x_span, 0.0, 1.0),
            1.0 - np.clip((vertices[:, 1] - y_min) / y_span, 0.0, 1.0),
        )
    )
    geometry.triangle_uvs = o3d.utility.Vector2dVector(uv[faces].reshape(-1, 2))
    texture_float = np.asarray(texture, dtype=np.float32)
    # Preserve the requested texture resolution. convertScaleAbs allocates only
    # the uint8 result, unlike a multiply/clip/astype NumPy expression which
    # creates several full-resolution temporary arrays.
    texture_u8 = cv2.convertScaleAbs(texture_float, alpha=255.0)
    texture_image = o3d.geometry.Image(texture_u8)
    geometry.textures = [texture_image]
    return geometry, texture_image


def _render_trajectory_nvdiffrast(
    mesh,
    poses,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    expected: list[Path],
    overwrite: bool,
) -> int:
    """Render with the same CUDA rasterizer used by the trained-scene renderer."""

    import torch
    import nvdiffrast.torch as dr

    device = torch.device("cuda")
    vertices = mesh.vertices.detach().to(device=device, dtype=torch.float32)
    faces_all = mesh.faces.detach().to(device=device, dtype=torch.long)
    texture_np = np.asarray(mesh.texture_buffer, dtype=np.float32)
    # The projection buffer is already contiguous HWC. Upload it in that layout
    # directly so no second full-resolution GPU texture is created.
    texture = torch.from_numpy(texture_np).unsqueeze(0).to(device=device)

    metadata = mesh.texture_metadata
    x_min, x_max = metadata["x_range"]
    y_min, y_max = metadata["y_range"]
    x_span = max(float(x_max) - float(x_min), 1e-8)
    y_span = max(float(y_max) - float(y_min), 1e-8)
    uv = torch.stack(
        (
            ((vertices[:, 0] - float(x_min)) / x_span).clamp(0.0, 1.0),
            ((vertices[:, 1] - float(y_min)) / y_span).clamp(0.0, 1.0),
        ),
        dim=-1,
    )[None].contiguous()

    K = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    vertices_h = torch.cat(
        (vertices, torch.ones((vertices.shape[0], 1), device=device)), dim=-1
    )
    glctx = dr.RasterizeCudaContext()
    try:
        for index, pose in enumerate(poses):
            if expected[index].is_file() and not overwrite:
                continue
            c2w = torch.as_tensor(pose, dtype=torch.float32, device=device)
            w2c = torch.linalg.inv(c2w)
            vertices_cam = (vertices_h @ w2c.T)[:, :3]
            z_cam = vertices_cam[:, 2]
            valid_vertices = z_cam > 1e-4
            valid_faces = valid_vertices[faces_all].all(dim=-1)
            if not bool(valid_faces.any()):
                rgba = np.zeros((height, width, 4), dtype=np.uint8)
                Image.fromarray(rgba, mode="RGBA").save(expected[index])
                continue

            faces = faces_all[valid_faces]
            z_safe = z_cam.clamp_min(1e-4)
            px = fx * vertices_cam[:, 0] / z_safe + cx
            py = fy * vertices_cam[:, 1] / z_safe + cy
            x_ndc = ((px + 0.5) / max(width, 1)) * 2.0 - 1.0
            y_ndc = 1.0 - ((py + 0.5) / max(height, 1)) * 2.0
            z_valid = z_cam[valid_vertices]
            z_min, z_max = z_valid.min(), z_valid.max()
            z_ndc = ((z_cam - z_min) / (z_max - z_min).clamp_min(1e-6)) * 2.0 - 1.0
            clip_vertices = torch.stack(
                (x_ndc, y_ndc, z_ndc, torch.ones_like(z_ndc)), dim=-1
            )[None].contiguous()
            triangles = faces.to(dtype=torch.int32).contiguous()
            rast, _ = dr.rasterize(glctx, clip_vertices, triangles, (height, width))
            interpolated_uv, _ = dr.interpolate(uv, rast, triangles)
            rgb = dr.texture(
                texture,
                interpolated_uv,
                filter_mode="linear",
                boundary_mode="clamp",
            )[0]
            alpha = (rast[0, ..., 3:4] > 0).to(torch.float32)
            rgb = torch.flip(rgb, dims=(0,))
            alpha = torch.flip(alpha, dims=(0,))
            rgba = torch.cat((rgb, alpha), dim=-1).clamp(0.0, 1.0)
            rgba_u8 = (rgba * 255.0).to(torch.uint8).cpu().numpy()
            Image.fromarray(rgba_u8, mode="RGBA").save(expected[index])
    finally:
        del glctx, texture, uv, vertices, faces_all, vertices_h
        torch.cuda.empty_cache()
    return len(expected)


def render_trajectory(
    mesh,
    poses,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    output_dir: Path,
    overwrite: bool = False,
) -> int:
    """Render a trajectory as sequential, transparent RGBA PNG files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"frame{i:04d}.png" for i in range(len(poses))]
    if not overwrite and expected and all(path.is_file() for path in expected):
        LOGGER.info("Keeping %d existing renders in %s", len(expected), output_dir)
        return len(expected)

    # Match the renderer already used by BaseTrainer for road meshes. Besides
    # avoiding Open3D's native Filament/EGL failure mode, this samples the full
    # requested texture without imposing an arbitrary dimension cap.
    try:
        import torch
        import nvdiffrast.torch  # noqa: F401

        if torch.cuda.is_available():
            LOGGER.info(
                "Rendering %s with CUDA nvdiffrast and full %dx%d texture",
                output_dir,
                mesh.texture_buffer.shape[1],
                mesh.texture_buffer.shape[0],
            )
            return _render_trajectory_nvdiffrast(
                mesh, poses, intrinsic, height, width, expected, overwrite
            )
    except ImportError as exc:
        LOGGER.warning("nvdiffrast unavailable; falling back to Open3D: %s", exc)

    import os

    os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    import open3d as o3d

    LOGGER.info("Preparing full-resolution Open3D mesh texture for %s", output_dir)
    geometry, texture_image = _make_open3d_mesh(mesh)
    LOGGER.info("Creating Open3D offscreen renderer at %dx%d", width, height)
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.sRGB_color = True
    # Use the owning object directly. geometry.textures[0] is a non-owning
    # pybind11 reference in Open3D 0.16 and raises a Holder<T> cast error.
    material.albedo_img = texture_image
    renderer.scene.add_geometry("road_mesh", geometry, material)

    pinhole = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
    )
    try:
        for index, pose in enumerate(_to_numpy(poses, np.float64)):
            renderer.setup_camera(pinhole, np.linalg.inv(pose))
            color = np.asarray(renderer.render_to_image())
            depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
            if color.dtype.kind == "f":
                color = color * 255.0
            rgb = np.clip(color[..., :3], 0, 255).astype(np.uint8)
            # Open3D reports background depth as infinity (some versions use 0).
            alpha = (np.isfinite(depth) & (depth > 0)).astype(np.uint8) * 255
            rgba = np.dstack((rgb, alpha))
            Image.fromarray(rgba).save(expected[index])
    finally:
        renderer.scene.remove_geometry("road_mesh")
        del renderer
    return len(expected)


def image_files(directory: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg"}
    return sorted(
        (path for path in directory.iterdir() if path.suffix.lower() in extensions),
        key=natural_sort_key,
    )


def composite_directories(
    background_dir: Path,
    overlay_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    frame_limit: Optional[int] = None,
) -> int:
    """Alpha composite road renders over Gaussian render frames by frame order."""

    if not background_dir.is_dir():
        raise FileNotFoundError(f"Gaussian image directory not found: {background_dir}")
    backgrounds = image_files(background_dir)
    overlays = image_files(overlay_dir)
    if frame_limit is not None:
        backgrounds = backgrounds[:frame_limit]
        overlays = overlays[:frame_limit]
    if not backgrounds:
        raise ValueError(f"No Gaussian image frames found in {background_dir}")
    if len(backgrounds) != len(overlays):
        raise ValueError(
            f"Frame-count mismatch: {len(backgrounds)} Gaussian frames in {background_dir}, "
            f"{len(overlays)} road frames in {overlay_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (background_path, overlay_path) in enumerate(zip(backgrounds, overlays)):
        output_path = output_dir / f"frame{index:04d}.png"
        if output_path.is_file() and not overwrite:
            continue
        with Image.open(background_path) as bg_image, Image.open(overlay_path) as road_image:
            background = bg_image.convert("RGBA")
            road = road_image.convert("RGBA")
            if road.size != background.size:
                road = road.resize(background.size, resample=Image.Resampling.BILINEAR)
            Image.alpha_composite(background, road).convert("RGB").save(output_path)
    return len(backgrounds)


def output_root_for_run(run_dir: Path, step: int) -> Path:
    return run_dir / "videos_eval" / f"novel_{step}"


def validate_trajectory_files(scene_dir: Path) -> dict[str, Path]:
    trajectory_dir = scene_dir / "trajectories"
    paths = {name: trajectory_dir / f"{name}.npz" for name in (*TRAJECTORY_NAMES, "clip_2")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trajectory file(s): " + ", ".join(missing))
    return paths


def finish_existing_outputs(output_root: Path, args) -> bool:
    """Complete cheap compositing work and report whether a run needs a mesh.

    This check deliberately runs before loading the dataset. Texture projection
    is the expensive part of this command and is unnecessary when every target
    road frame already exists.
    """

    if args.overwrite:
        return False

    all_complete = True
    for trajectory_name in TRAJECTORY_NAMES:
        background_dir = output_root / f"images_file_{trajectory_name}"
        road_dir = output_root / f"{trajectory_name}_road_mesh"
        composite_dir = output_root / f"composite_road_mesh_{trajectory_name}"
        if not background_dir.is_dir():
            all_complete = False
            continue

        backgrounds = image_files(background_dir)
        if args.max_frames is not None:
            backgrounds = backgrounds[: args.max_frames]
        frame_count = len(backgrounds)
        if frame_count == 0:
            all_complete = False
            continue

        expected_roads = [road_dir / f"frame{i:04d}.png" for i in range(frame_count)]
        if not all(path.is_file() for path in expected_roads):
            all_complete = False
            continue

        expected_composites = [
            composite_dir / f"frame{i:04d}.png" for i in range(frame_count)
        ]
        if not all(path.is_file() for path in expected_composites):
            composited = composite_directories(
                background_dir,
                road_dir,
                composite_dir,
                overwrite=False,
                frame_limit=args.max_frames,
            )
            LOGGER.info(
                "Reused %d existing %s road frames and completed %d composites",
                frame_count,
                trajectory_name,
                composited,
            )

    return all_complete


def process_run(spec: RunSpec, args) -> None:
    """Execute the complete road-mesh pipeline for one run."""

    import torch

    device = torch.device(args.device)
    output_root = output_root_for_run(spec.run_dir, args.step)
    output_root.mkdir(parents=True, exist_ok=True)
    if finish_existing_outputs(output_root, args):
        LOGGER.info(
            "%s already has complete road renders and composites; skipping dataset loading and projection",
            spec.scene_id,
        )
        return

    trajectories = validate_trajectory_files(spec.scene_dir)
    LOGGER.info("Loading dataset for %s from %s", spec.scene_id, spec.scene_dir)
    dataset = load_dataset(spec, device)
    try:
        mesh, projection_stats = build_road_mesh(dataset, device, args)
        LOGGER.info("Texture projection stats for %s: %s", spec.scene_id, projection_stats)
        if args.save_mesh:
            mesh.save_checkpoint(str(output_root / "road_mesh.pth"))

        intrinsic, height, width = get_camera_calibration(dataset)
        aligned_trajectories = {}
        for trajectory_name in TRAJECTORY_NAMES:
            aligned = load_and_align_trajectory(
                dataset,
                trajectories[trajectory_name],
                trajectories["clip_2"],
                device,
            )
            if args.max_frames is not None:
                aligned = aligned[: args.max_frames]
            aligned_trajectories[trajectory_name] = aligned

        # Rendering no longer needs RGB images, masks, LiDAR, or camera-source
        # objects. Release them before Open3D creates its own CPU/GPU resources.
        del dataset
        dataset = None
        del projection_stats
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        LOGGER.info("Released dataset and CUDA projection cache before road-mesh rendering")

        for trajectory_name in TRAJECTORY_NAMES:
            aligned = aligned_trajectories[trajectory_name]
            road_dir = output_root / f"{trajectory_name}_road_mesh"
            rendered = render_trajectory(
                mesh,
                aligned,
                intrinsic,
                height,
                width,
                road_dir,
                overwrite=args.overwrite,
            )
            background_dir = output_root / f"images_file_{trajectory_name}"
            composite_dir = output_root / f"composite_road_mesh_{trajectory_name}"
            composited = composite_directories(
                background_dir,
                road_dir,
                composite_dir,
                overwrite=args.overwrite,
                frame_limit=args.max_frames,
            )
            LOGGER.info(
                "%s %s: rendered %d transparent frames and wrote %d composites",
                spec.scene_id,
                trajectory_name,
                rendered,
                composited,
            )
    finally:
        if dataset is not None:
            del dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, help="Directory containing trained run folders")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the data root stored in each run's config.yaml",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default=DEFAULT_CHECKPOINT_PATTERN,
        help="Checkpoint glob used only to identify trained runs",
    )
    parser.add_argument("--step", type=int, default=30000, help="novel_<step> output folder")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for dataset loading and projection (default: cuda)",
    )
    parser.add_argument("--cell-size", type=float, default=0.25, help="Mesh grid size in metres")
    parser.add_argument(
        "--vertex-selection",
        choices=("median", "lowest"),
        default="median",
        help="Representative LiDAR height per grid cell",
    )
    parser.add_argument(
        "--texture-pixels-per-meter", type=float, default=200.0, help="Road texture resolution"
    )
    parser.add_argument(
        "--num-projection-frames",
        type=int,
        default=None,
        help="Limit source images used to texture the mesh",
    )
    parser.add_argument(
        "--rays-per-image",
        type=int,
        default=400000,
        help="Maximum road-mask rays used per source image",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Render only the first N poses")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output frames")
    parser.add_argument("--save-mesh", action="store_true", help="Save road_mesh.pth in novel_<step>")
    parser.add_argument("--dry-run", action="store_true", help="Discover and print runs only")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed run instead of continuing the batch",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.runs_dir = args.runs_dir.expanduser().resolve()
    if not args.runs_dir.exists():
        LOGGER.error("Runs path does not exist: %s", args.runs_dir)
        return 2
    if args.max_frames is not None and args.max_frames <= 0:
        LOGGER.error("--max-frames must be positive")
        return 2
    if args.cell_size <= 0 or args.texture_pixels_per_meter <= 0 or args.rays_per_image <= 0:
        LOGGER.error("--cell-size, --texture-pixels-per-meter, and --rays-per-image must be positive")
        return 2
    data_root = args.data_root.expanduser().resolve() if args.data_root is not None else None
    if data_root is not None and not data_root.is_dir():
        LOGGER.error("Data root does not exist: %s", data_root)
        return 2

    specs, warnings = discover_runs(args.runs_dir, data_root, args.checkpoint_pattern)
    for warning in warnings:
        LOGGER.warning(warning)
    if not specs:
        LOGGER.error("No processable checkpoints matched %r", args.checkpoint_pattern)
        return 1
    if data_root is None:
        LOGGER.info("Found %d run(s); using dataset paths stored in run configs", len(specs))
    else:
        LOGGER.info("Found %d run(s); overriding data root with %s", len(specs), data_root)
    if args.dry_run:
        for spec in specs:
            print(f"{spec.checkpoint} -> {spec.scene_dir} -> {output_root_for_run(spec.run_dir, args.step)}")
        return 0

    failures = []
    for index, spec in enumerate(specs, start=1):
        LOGGER.info("[%d/%d] Processing %s", index, len(specs), spec.checkpoint)
        try:
            process_run(spec, args)
        except Exception as exc:
            LOGGER.exception("Failed %s: %s", spec.checkpoint, exc)
            failures.append((spec.checkpoint, str(exc)))
            if args.fail_fast:
                break
    if failures:
        LOGGER.error("%d/%d run(s) failed", len(failures), len(specs))
        for checkpoint, message in failures:
            LOGGER.error("  %s: %s", checkpoint, message)
        return 1
    LOGGER.info("Completed all %d run(s)", len(specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
