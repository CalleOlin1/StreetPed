#!/usr/bin/env python3
"""Render the same custom trajectory for every checkpoint below a directory."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def find_checkpoints(runs_dir: Path, checkpoint_name: str) -> List[Path]:
    if runs_dir.is_file():
        return [runs_dir] if runs_dir.name == checkpoint_name else []
    return sorted(path for path in runs_dir.rglob(checkpoint_name) if path.is_file())


def build_command(
    repo_root: Path,
    checkpoint: Path,
    trajectory_name: str,
    trajectory_label: str,
    camera_id: int,
    fps: Optional[int],
    render_downscale: float,
    save_images: bool,
    save_raw_frames: bool,
    extra_opts: Sequence[str],
) -> List[str]:
    command = [
        sys.executable,
        str(repo_root / "tools" / "eval.py"),
        "--resume_from",
        str(checkpoint),
        "--custom_trajectory_name",
        trajectory_name,
        "--custom_trajectory_label",
        trajectory_label,
        "--custom_camera_id",
        str(camera_id),
        "--custom_render_downscale",
        str(render_downscale),
        "--skip_original_render",
        "--lazy_dataset",
        "--skip-lidar",
    ]
    if fps is not None:
        command.extend(["--custom_trajectory_fps", str(fps)])
    if save_images:
        command.append("--custom_save_images")
    if save_raw_frames:
        command.append("--custom_save_raw_frames")
    command.extend(
        [
            "render.render_test=False",
            "render.render_full=False",
            "render.render_novel=null",
        ]
    )
    command.extend(extra_opts)
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render one named per-scene trajectory for every checkpoint found "
            "under RUNS_DIR. Checkpoints may live in another repository."
        )
    )
    parser.add_argument("runs_dir", type=Path, help="Checkpoint file or directory to search.")
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_final.pth",
        help="Checkpoint filename to find recursively (default: checkpoint_final.pth).",
    )
    parser.add_argument(
        "--trajectory-name",
        default="right_4m",
        help="Name below each scene's novel_views directory (default: right_4m).",
    )
    parser.add_argument(
        "--trajectory-label",
        default=None,
        help="Output MP4 stem (default: same as --trajectory-name).",
    )
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--render-downscale",
        type=float,
        default=1.0,
        help="Output downscale relative to source images (default: 1, full resolution).",
    )
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-raw-frames", action="store_true")
    parser.add_argument(
        "--extra-opt",
        dest="extra_opts",
        action="append",
        default=[],
        help="Config override passed to eval.py; repeat for multiple overrides.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue rendering remaining checkpoints after a failure.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.expanduser().resolve()
    if not runs_dir.exists():
        parser.error(f"Checkpoint path does not exist: {runs_dir}")

    checkpoints = find_checkpoints(runs_dir, args.checkpoint_name)
    if not checkpoints:
        print(f"No {args.checkpoint_name!r} checkpoints found under {runs_dir}")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    trajectory_label = args.trajectory_label or args.trajectory_name
    print(f"Found {len(checkpoints)} checkpoint(s).")
    failures = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        command = build_command(
            repo_root=repo_root,
            checkpoint=checkpoint,
            trajectory_name=args.trajectory_name,
            trajectory_label=trajectory_label,
            camera_id=args.camera_id,
            fps=args.fps,
            render_downscale=args.render_downscale,
            save_images=args.save_images,
            save_raw_frames=args.save_raw_frames,
            extra_opts=args.extra_opts,
        )
        print(f"[{index}/{len(checkpoints)}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue

        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode != 0:
            failures.append((checkpoint, result.returncode))
            print(f"Failed ({result.returncode}): {checkpoint}", file=sys.stderr)
            if not args.keep_going:
                return result.returncode

    if failures:
        print(f"{len(failures)} checkpoint(s) failed:", file=sys.stderr)
        for checkpoint, returncode in failures:
            print(f"  {returncode}: {checkpoint}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
