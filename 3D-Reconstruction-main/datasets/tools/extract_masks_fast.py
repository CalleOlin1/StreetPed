"""
@file   extract_masks_fast.py
@brief  Parallel scene dispatcher for extract_masks.py

This script keeps the same extraction behavior as extract_masks.py, but splits
scenes into N subprocesses so multiple scene groups are processed in parallel.
"""

import os
import subprocess
import sys
from argparse import ArgumentParser
from typing import List


def resolve_scene_ids(data_root: str, rgb_dirname: str, scene_ids, split_file: str, start_idx: int, num_scenes: int) -> List[str]:
    if scene_ids is not None:
        resolved = []
        for scene_id in scene_ids:
            scene_id = str(scene_id)
            if scene_id.isdigit():
                resolved.append(str(int(scene_id)).zfill(3))
            else:
                resolved.append(scene_id)
        return resolved

    if split_file is not None:
        split_lines = open(split_file, "r").readlines()
        if len(split_lines) > 0 and split_lines[0].lstrip().startswith("#"):
            split_lines = split_lines[1:]

        resolved = []
        for line in split_lines:
            token = line.strip().split(",")[0]
            if token == "":
                continue
            if token.isdigit():
                resolved.append(str(int(token)).zfill(3))
            else:
                resolved.append(token)
        return resolved

    discovered_scene_ids = sorted(
        scene_name
        for scene_name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, scene_name))
        and os.path.isdir(os.path.join(data_root, scene_name, rgb_dirname))
    )
    if discovered_scene_ids:
        return discovered_scene_ids[start_idx: start_idx + num_scenes]

    return [str(scene_id).zfill(3) for scene_id in range(start_idx, start_idx + num_scenes)]


def split_evenly(items: List[str], num_chunks: int) -> List[List[str]]:
    num_chunks = min(num_chunks, len(items))
    q, r = divmod(len(items), num_chunks)
    chunks = []
    start = 0
    for chunk_idx in range(num_chunks):
        chunk_size = q + (1 if chunk_idx < r else 0)
        end = start + chunk_size
        chunks.append(items[start:end])
        start = end
    return chunks


def build_base_command(args, passthrough_args: List[str], extract_script_path: str) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        extract_script_path,
        "--data_root",
        args.data_root,
        "--rgb_dirname",
        args.rgb_dirname,
        "--mask_dirname",
        args.mask_dirname,
        "--segformer_path",
        args.segformer_path,
        "--device",
        args.device,
        "--palette",
        args.palette,
    ]

    if args.split_file is not None:
        cmd.extend(["--split_file", args.split_file])
    if args.config is not None:
        cmd.extend(["--config", args.config])
    if args.checkpoint is not None:
        cmd.extend(["--checkpoint", args.checkpoint])

    if args.process_dynamic_mask:
        cmd.append("--process_dynamic_mask")
    if args.process_road_mask:
        cmd.append("--process_road_mask")
    if args.verbose:
        cmd.append("--verbose")
    if args.ignore_existing:
        cmd.append("--ignore_existing")
    if args.no_compress:
        cmd.append("--no_compress")

    cmd.extend(passthrough_args)
    return cmd


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=2, help="Number of subprocesses to launch")

    # Keep these aligned with extract_masks.py.
    parser.add_argument("--data_root", type=str, default="data/waymo/processed/training")
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=str,
        nargs="+",
        help="scene ids to be processed. Supports numeric IDs (e.g. 0 1) and named IDs (e.g. scene_001_clip_000).",
    )
    parser.add_argument("--split_file", type=str, default=None, help="Split file in data/waymo_splits")
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument("--num_scenes", type=int, default=200, help="number of scenes to be processed")
    parser.add_argument("--process_dynamic_mask", action="store_true", help="Whether to process dynamic masks")
    parser.add_argument(
        "--process_road_mask",
        "--process_road",
        dest="process_road_mask",
        action="store_true",
        help="Whether to process road masks",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument("--no_compress", action="store_true")
    parser.add_argument("--rgb_dirname", type=str, default="images")
    parser.add_argument("--mask_dirname", type=str, default="fine_dynamic_masks")
    parser.add_argument("--segformer_path", type=str, default="/home/guojianfei/ai_ws/SegFormer")
    parser.add_argument("--config", help="Config file", type=str, default=None)
    parser.add_argument("--checkpoint", help="Checkpoint file", type=str, default=None)
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--palette", default="cityscapes", help="Color palette used for segmentation map")

    args, passthrough_args = parser.parse_known_args()

    if args.num_processes <= 0:
        raise ValueError("--num_processes must be greater than 0")

    scene_ids = resolve_scene_ids(
        data_root=args.data_root,
        rgb_dirname=args.rgb_dirname,
        scene_ids=args.scene_ids,
        split_file=args.split_file,
        start_idx=args.start_idx,
        num_scenes=args.num_scenes,
    )
    if len(scene_ids) == 0:
        raise RuntimeError(f"No scenes found under data_root={args.data_root!r} with rgb_dirname={args.rgb_dirname!r}")

    scene_chunks = split_evenly(scene_ids, args.num_processes)
    extract_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_masks.py")
    base_cmd = build_base_command(args, passthrough_args, extract_script_path)

    procs = []
    try:
        for worker_idx, chunk in enumerate(scene_chunks):
            worker_cmd = base_cmd + [
                "--flat_progress",
                "--progress_position",
                str(worker_idx),
                "--progress_desc",
                f"worker[{worker_idx}]",
                "--scene_ids",
            ] + chunk
            print(f"[worker {worker_idx}] launching {len(chunk)} scenes")
            procs.append((worker_idx, subprocess.Popen(worker_cmd)))

        failed_workers = []
        for worker_idx, proc in procs:
            return_code = proc.wait()
            if return_code != 0:
                failed_workers.append((worker_idx, return_code))

        if failed_workers:
            formatted = ", ".join(f"worker={worker_idx}, exit={code}" for worker_idx, code in failed_workers)
            raise RuntimeError(f"extract_masks subprocess failed: {formatted}")
    except KeyboardInterrupt:
        for _, proc in procs:
            if proc.poll() is None:
                proc.terminate()
        raise