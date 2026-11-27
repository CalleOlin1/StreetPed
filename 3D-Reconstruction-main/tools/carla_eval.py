#!/usr/bin/env python3
"""Render the same CARLA right_4m novel view from three trained scenes.

Run ``python tools/carla_eval.py`` from any working directory. Videos and PNG
frames are written under each checkpoint's ``videos_eval/novel_<step>/``.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "tools" / "eval.py"
RENDER_PYTHON = Path("/home/hstromgr/miniconda3/envs/ds117/bin/python")
NOVEL_VIEW = Path(
    "/home/hstromgr/Documents/Github/StreetPed/3D-Reconstruction-main/"
    "data/carla/processed_w_smpl/carla_town10_with_PEDS_4dhumans/"
    "novel_views/right_4m"
)
SCENES = {
    "baseline": Path(
        "/home/hstromgr/Documents/Github/drivestudio/"
        "work_dirs/carla/town10_peds_4dhumans"
    ),
    "road_as_sky_difix": Path(
        "/home/hstromgr/Documents/Github/StreetPed_road_is_sky/"
        "3D-Reconstruction-main/output/CARLA_all/"
        "carla_town10_with_PEDS_4dhumans_mesh_locked_fullres"
    ),
    "road_as_sky": Path(
        "/home/hstromgr/Documents/Github/StreetPed_road_is_sky/"
        "3D-Reconstruction-main/output/CARLA_all/"
        "carla_town10_with_PEDS_4dhumans_mesh_locked_fullres_no_difix"
    ),
}


def build_command(checkpoint: Path, save_images: bool) -> list[str]:
    command = [
        str(RENDER_PYTHON),
        str(EVAL_SCRIPT),
        "--resume_from", str(checkpoint),
        "--custom_ego_pose_dir", str(NOVEL_VIEW / "ego_pose"),
        "--custom_extrinsics", str(NOVEL_VIEW / "extrinsics" / "0.txt"),
        "--custom_camera_id", "0",
        "--custom_trajectory_label", "right_4m",
        "--skip_original_render",
        "--hold_dynamic_nodes",
        "--lazy_dataset",
        "--skip-lidar",
    ]
    if save_images:
        command.append("--custom_save_images")
    command.extend([
        "render.render_test=False",
        "render.render_full=False",
        "render.render_novel=null",
    ])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", choices=tuple(SCENES), action="append",
        help="Render only this scene; repeat to select several (default: all).",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Write MP4 videos only, without individual PNG frames.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without rendering."
    )
    args = parser.parse_args()

    required = [RENDER_PYTHON, EVAL_SCRIPT, NOVEL_VIEW / "extrinsics" / "0.txt"]
    if not (NOVEL_VIEW / "ego_pose").is_dir() or not any(
        (NOVEL_VIEW / "ego_pose").glob("*.txt")
    ):
        parser.error(f"No trajectory poses in {NOVEL_VIEW / 'ego_pose'}")
    selected = args.scene or list(SCENES)
    for name in selected:
        checkpoint = SCENES[name] / "checkpoint_final.pth"
        required.extend([checkpoint, SCENES[name] / "config.yaml"])
    missing = [path for path in required if not path.is_file()]
    if missing:
        parser.error("Missing required files: " + ", ".join(map(str, missing)))

    for name in selected:
        checkpoint = SCENES[name] / "checkpoint_final.pth"
        command = build_command(checkpoint, save_images=not args.no_images)
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode:
            print(f"[{name}] render failed (exit {result.returncode})", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
