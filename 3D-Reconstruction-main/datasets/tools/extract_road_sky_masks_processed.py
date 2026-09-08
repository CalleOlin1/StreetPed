#!/usr/bin/env python3
"""
Extract road and sky masks for all scenes inside a processed dataset folder.

This is a thin wrapper around datasets/tools/extract_masks.py:
- discovers scene directories under --data_root
- filters to scenes that contain an image folder
- invokes extract_masks.py per scene with --process_road_mask

Sky masks are always produced by extract_masks.py; road masks are enabled here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_CONFIG_REL = Path("local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py")
DEFAULT_CHECKPOINT_REL = Path("pretrained/segformer.b5.1024x1024.city.160k.pth")
DEFAULT_SEGFORMER_ROOT = Path("/home/hstromgr/Documents/Github/SegFormer")


def _discover_scenes(data_root: Path, rgb_dirname: str) -> List[str]:
    scene_names: List[str] = []
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / rgb_dirname).is_dir():
            scene_names.append(entry.name)
    return scene_names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract road and sky masks for all scenes in a processed folder."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Processed dataset root containing scene subfolders.",
    )
    parser.add_argument(
        "--extract_script",
        type=Path,
        default=Path(__file__).with_name("extract_masks.py"),
        help="Path to datasets/tools/extract_masks.py helper script.",
    )
    parser.add_argument(
        "--python_executable",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke extract_masks.py.",
    )
    parser.add_argument(
        "--scene_ids",
        nargs="+",
        default=None,
        help="Optional explicit scene names. If omitted, all scenes are discovered from --data_root.",
    )
    parser.add_argument(
        "--rgb_dirname",
        type=str,
        default="images",
        help="Image directory name inside each scene folder.",
    )
    parser.add_argument(
        "--segformer_path",
        type=str,
        default=str(DEFAULT_SEGFORMER_ROOT),
        help="Path to SegFormer repository.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--ignore_existing",
        action="store_true",
        help="Forwarded to extract_masks.py; skips when output already exists.",
    )
    return parser


def _resolve_model_paths(
    segformer_path_arg: str | None,
    config_arg: str | None,
    checkpoint_arg: str | None,
) -> tuple[Path, Path, Path]:
    segformer_root = Path(segformer_path_arg).resolve()
    if not segformer_root.is_dir():
        raise FileNotFoundError(
            "SegFormer directory does not exist. "
            f"Checked: {segformer_root}. Pass --segformer_path explicitly."
        )

    if config_arg is None:
        config_path = segformer_root / DEFAULT_CONFIG_REL
    else:
        config_path = Path(config_arg).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"SegFormer config does not exist: {config_path}. "
            "Pass --config explicitly if needed."
        )

    if checkpoint_arg is None:
        checkpoint_path = segformer_root / DEFAULT_CHECKPOINT_REL
    else:
        checkpoint_path = Path(checkpoint_arg).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SegFormer checkpoint does not exist: {checkpoint_path}. "
            "Place the checkpoint there or pass --checkpoint explicitly."
        )

    return segformer_root, config_path, checkpoint_path


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"data_root does not exist or is not a directory: {data_root}")

    extract_script = args.extract_script.resolve()
    if not extract_script.is_file():
        raise FileNotFoundError(f"extract_masks helper not found: {extract_script}")
    segformer_root, config_path, checkpoint_path = _resolve_model_paths(
        args.segformer_path, args.config, args.checkpoint
    )

    discovered_scenes = _discover_scenes(data_root, args.rgb_dirname)
    if args.scene_ids is None:
        scene_ids = discovered_scenes
    else:
        requested = set(args.scene_ids)
        scene_ids = [scene for scene in discovered_scenes if scene in requested]
        missing = sorted(requested - set(scene_ids))
        if missing:
            raise ValueError(
                "Requested scene_ids were not found (or missing image folder): "
                + ", ".join(missing)
            )

    if not scene_ids:
        raise RuntimeError(
            f"No scenes found in {data_root} with image folder '{args.rgb_dirname}'."
        )

    print(f"Found {len(scene_ids)} scene(s). Running road+sky mask extraction...")
    for index, scene_id in enumerate(scene_ids, start=1):
        print(f"[{index}/{len(scene_ids)}] {scene_id}")
        cmd = [
            args.python_executable,
            str(extract_script),
            "--data_root",
            str(data_root),
            "--scene_ids",
            scene_id,
            "--process_road_mask",
            "--rgb_dirname",
            args.rgb_dirname,
            "--device",
            args.device,
            "--segformer_path",
            str(segformer_root),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
        if args.ignore_existing:
            cmd.append("--ignore_existing")

        subprocess.run(cmd, check=True)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
