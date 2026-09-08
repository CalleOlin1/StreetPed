#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _safe_median(values: List[float]) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _safe_min(values: List[float]) -> float:
    if not values:
        return float("nan")
    return min(values)


def _safe_max(values: List[float]) -> float:
    if not values:
        return float("nan")
    return max(values)


def _collect_metrics(csv_path: Path, buckets: Dict[str, Dict[str, List[float]]]) -> None:
    clip_folder = csv_path.parent.name
    if "clip_0" in clip_folder:
        clip_key = "clip_0"
    elif "clip_1" in clip_folder:
        clip_key = "clip_1"
    else:
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"psnr", "ssim", "lpips"}
        if not required.issubset(set(reader.fieldnames or [])):
            return

        for row in reader:
            try:
                buckets[clip_key]["psnr"].append(float(row["psnr"]))
                buckets[clip_key]["ssim"].append(float(row["ssim"]))
                buckets[clip_key]["lpips"].append(float(row["lpips"]))
            except (ValueError, TypeError, KeyError):
                continue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print average PSNR/SSIM/LPIPS across clip_0 then clip_1 from batch-eval metrics CSVs."
    )
    parser.add_argument("runs_dir", type=Path, help="Folder containing batch-eval run outputs.")
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    buckets: Dict[str, Dict[str, List[float]]] = {
        "clip_0": {"psnr": [], "ssim": [], "lpips": []},
        "clip_1": {"psnr": [], "ssim": [], "lpips": []},
    }

    for csv_path in sorted(runs_dir.rglob("metrics.csv")):
        _collect_metrics(csv_path, buckets)

    clip0_psnr = _safe_mean(buckets["clip_0"]["psnr"])
    clip0_ssim = _safe_mean(buckets["clip_0"]["ssim"])
    clip0_lpips = _safe_mean(buckets["clip_0"]["lpips"])
    clip1_psnr = _safe_mean(buckets["clip_1"]["psnr"])
    clip1_ssim = _safe_mean(buckets["clip_1"]["ssim"])
    clip1_lpips = _safe_mean(buckets["clip_1"]["lpips"])
    clip0_psnr_median = _safe_median(buckets["clip_0"]["psnr"])
    clip0_ssim_median = _safe_median(buckets["clip_0"]["ssim"])
    clip0_lpips_median = _safe_median(buckets["clip_0"]["lpips"])
    clip1_psnr_median = _safe_median(buckets["clip_1"]["psnr"])
    clip1_ssim_median = _safe_median(buckets["clip_1"]["ssim"])
    clip1_lpips_median = _safe_median(buckets["clip_1"]["lpips"])
    clip0_psnr_worst = _safe_min(buckets["clip_0"]["psnr"])
    clip0_ssim_worst = _safe_min(buckets["clip_0"]["ssim"])
    clip0_lpips_worst = _safe_max(buckets["clip_0"]["lpips"])
    clip1_psnr_worst = _safe_min(buckets["clip_1"]["psnr"])
    clip1_ssim_worst = _safe_min(buckets["clip_1"]["ssim"])
    clip1_lpips_worst = _safe_max(buckets["clip_1"]["lpips"])
    clip0_psnr_best = _safe_max(buckets["clip_0"]["psnr"])
    clip0_ssim_best = _safe_max(buckets["clip_0"]["ssim"])
    clip0_lpips_best = _safe_min(buckets["clip_0"]["lpips"])
    clip1_psnr_best = _safe_max(buckets["clip_1"]["psnr"])
    clip1_ssim_best = _safe_max(buckets["clip_1"]["ssim"])
    clip1_lpips_best = _safe_min(buckets["clip_1"]["lpips"])

    print("Clip 0, Clip 1,")
    print("PSNR, SSIM, LPIPS, PSNR, SSIM, LPIPS,")
    print(
        f"{clip0_psnr:.6f}, {clip0_ssim:.6f}, {clip0_lpips:.6f}, "
        f"{clip1_psnr:.6f}, {clip1_ssim:.6f}, {clip1_lpips:.6f}"
    )
    print(
        f"{clip0_psnr_median:.6f}, {clip0_ssim_median:.6f}, {clip0_lpips_median:.6f}, "
        f"{clip1_psnr_median:.6f}, {clip1_ssim_median:.6f}, {clip1_lpips_median:.6f}"
    )
    print(
        f"{clip0_psnr_worst:.6f}, {clip0_ssim_worst:.6f}, {clip0_lpips_worst:.6f}, "
        f"{clip1_psnr_worst:.6f}, {clip1_ssim_worst:.6f}, {clip1_lpips_worst:.6f}"
    )
    print(
        f"{clip0_psnr_best:.6f}, {clip0_ssim_best:.6f}, {clip0_lpips_best:.6f}, "
        f"{clip1_psnr_best:.6f}, {clip1_ssim_best:.6f}, {clip1_lpips_best:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
