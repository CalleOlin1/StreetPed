import argparse
import importlib
import json
import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.geometry import get_corners, project_camera_points_to_image

HUMAN_CLASS_KEYWORDS = ("pedestrian", "person", "cyclist", "rider", "bicycl")
VEHICLE_CLASS_KEYWORDS = (
    "car",
    "van",
    "truck",
    "bus",
    "tram",
    "motor",
    "trailer",
    "vehicle",
)

SEGMENTATION_HUMAN_CLASSES = [11, 12, 17, 18]
SEGMENTATION_VEHICLE_CLASSES = [13, 14, 15]
SEGMENTATION_SKY_CLASS = 10


def create_folder(target_dir: str):
    os.makedirs(f"{target_dir}/images", exist_ok=True)
    os.makedirs(f"{target_dir}/extrinsics", exist_ok=True)
    os.makedirs(f"{target_dir}/intrinsics", exist_ok=True)
    os.makedirs(f"{target_dir}/sky_masks", exist_ok=True)
    os.makedirs(f"{target_dir}/dynamic_masks/all", exist_ok=True)
    os.makedirs(f"{target_dir}/dynamic_masks/human", exist_ok=True)
    os.makedirs(f"{target_dir}/dynamic_masks/vehicle", exist_ok=True)
    os.makedirs(f"{target_dir}/fine_dynamic_masks/all", exist_ok=True)
    os.makedirs(f"{target_dir}/fine_dynamic_masks/human", exist_ok=True)
    os.makedirs(f"{target_dir}/fine_dynamic_masks/vehicle", exist_ok=True)


def parse_frame_index(filename: str) -> int:
    stem = os.path.splitext(os.path.basename(filename))[0]
    matched = re.search(r"\d+", stem)
    if matched is None:
        raise ValueError(f"Cannot parse frame index from filename: {filename}")
    return int(matched.group(0))


def load_matrix(path: str, expected_shape: Tuple[int, int]) -> np.ndarray:
    matrix = np.loadtxt(path)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == expected_shape[0] * expected_shape[1] and matrix.shape != expected_shape:
        matrix = matrix.reshape(expected_shape)
    if matrix.shape != expected_shape:
        raise ValueError(f"Expected matrix shape {expected_shape} at {path}, got {matrix.shape}")
    return matrix


def get_intrinsics_from_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path)
    values = np.asarray(values, dtype=np.float32)
    if values.shape == (3, 3):
        intrinsic = values
        intrinsics_vector = np.array(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2], 0, 0, 0, 0, 0],
            dtype=np.float32,
        )
        return intrinsic, intrinsics_vector
    values = values.reshape(-1)
    if values.size < 4:
        raise ValueError(f"Intrinsics file must contain 3x3 matrix or at least fx,fy,cx,cy: {path}")
    fx, fy, cx, cy = values[:4]
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if values.size >= 9:
        intrinsics_vector = values[:9].astype(np.float32)
    else:
        intrinsics_vector = np.array([fx, fy, cx, cy, 0, 0, 0, 0, 0], dtype=np.float32)
    return intrinsic, intrinsics_vector


def infer_ref_file(ref_dir: str, folder: str, camera_id: int) -> str:
    candidate = os.path.join(ref_dir, folder, f"{camera_id}.txt")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Missing reference file: {candidate}")
    return candidate


def init_segmentation_model(
    segformer_path: Optional[str],
    seg_config: Optional[str],
    seg_checkpoint: Optional[str],
    seg_device: str,
) -> Tuple[Any, Any]:
    try:
        mmseg_apis = importlib.import_module("mmseg.apis")
        inference_segmentor = getattr(mmseg_apis, "inference_segmentor")
        init_segmentor = getattr(mmseg_apis, "init_segmentor")
    except ImportError as error:
        raise ImportError(
            "Segmentation refinement requires mmseg. Please use the SegFormer environment "
            "(same setup as datasets/tools/extract_masks.py)."
        ) from error

    if seg_config is None:
        if segformer_path is None:
            raise ValueError("Provide --segformer_path or --seg_config for segmentation refinement.")
        seg_config = os.path.join(
            segformer_path,
            "local_configs",
            "segformer",
            "B5",
            "segformer.b5.1024x1024.city.160k.py",
        )
    if seg_checkpoint is None:
        if segformer_path is None:
            raise ValueError("Provide --segformer_path or --seg_checkpoint for segmentation refinement.")
        seg_checkpoint = os.path.join(
            segformer_path,
            "pretrained",
            "segformer.b5.1024x1024.city.160k.pth",
        )

    model = init_segmentor(seg_config, seg_checkpoint, device=seg_device)
    return model, inference_segmentor


def build_frame_to_instances(instances_info: Dict, frame_instances_path: Optional[str]) -> Dict[int, List[int]]:
    if frame_instances_path and os.path.exists(frame_instances_path):
        frame_instances_raw = json.load(open(frame_instances_path, "r"))
        return {
            int(frame_index): [int(instance_id) for instance_id in instance_ids]
            for frame_index, instance_ids in frame_instances_raw.items()
        }

    frame_to_instances: Dict[int, List[int]] = {}
    for instance_key, instance_value in instances_info.items():
        frame_indices = instance_value.get("frame_annotations", {}).get("frame_idx", [])
        for frame_index in frame_indices:
            int_frame_index = int(frame_index)
            frame_to_instances.setdefault(int_frame_index, []).append(int(instance_key))
    return frame_to_instances


def classify_instance(class_name: str) -> str:
    lowered = class_name.lower()
    if any(keyword in lowered for keyword in HUMAN_CLASS_KEYWORDS):
        return "human"
    if any(keyword in lowered for keyword in VEHICLE_CLASS_KEYWORDS):
        return "vehicle"
    return "other"


def find_pose_file(ref_dir: str, frame_index: int) -> str:
    candidates = [
        os.path.join(ref_dir, "ego_pose", f"{frame_index:03d}.txt"),
        os.path.join(ref_dir, "ego_pose", f"{frame_index:06d}.txt"),
    ]
    for pose_file in candidates:
        if os.path.exists(pose_file):
            return pose_file
    wildcard_candidates = sorted(glob(os.path.join(ref_dir, "ego_pose", "*.txt")))
    for pose_file in wildcard_candidates:
        try:
            if int(os.path.splitext(os.path.basename(pose_file))[0]) == frame_index:
                return pose_file
        except ValueError:
            continue
    raise FileNotFoundError(f"Missing ego pose for frame {frame_index}")


def generate_dynamic_masks(
    image_shape: Tuple[int, int],
    frame_index: int,
    frame_instance_ids: List[int],
    instances_info: Dict,
    world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_height, image_width = image_shape
    all_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    human_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    vehicle_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for instance_id in frame_instance_ids:
        instance = instances_info.get(str(instance_id))
        if instance is None:
            continue
        annotations = instance.get("frame_annotations", {})
        frame_indices = [int(idx) for idx in annotations.get("frame_idx", [])]
        if frame_index not in frame_indices:
            continue

        annotation_index = frame_indices.index(frame_index)
        obj_to_world = np.asarray(annotations["obj_to_world"][annotation_index], dtype=np.float32)
        obj_to_world = obj_to_world.reshape(4, 4)
        length, width, height = [float(x) for x in annotations["box_size"][annotation_index]]

        corners_local = get_corners(length, width, height)
        corners_world = obj_to_world[:3, :3] @ corners_local + obj_to_world[:3, 3:4]
        corners_camera = world_to_camera[:3, :3] @ corners_world + world_to_camera[:3, 3:4]
        projected_points, depth = project_camera_points_to_image(corners_camera.T, intrinsic)

        if np.max(depth) <= 0:
            continue

        x_min, y_min = np.min(projected_points, axis=0)
        x_max, y_max = np.max(projected_points, axis=0)
        x_min = int(np.clip(np.floor(x_min), 0, image_width - 1))
        x_max = int(np.clip(np.ceil(x_max), 0, image_width - 1))
        y_min = int(np.clip(np.floor(y_min), 0, image_height - 1))
        y_max = int(np.clip(np.ceil(y_max), 0, image_height - 1))

        if x_max <= x_min or y_max <= y_min:
            continue

        rectangle = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.int32,
        )
        cv2.fillPoly(all_mask, [rectangle], 255)

        class_type = classify_instance(instance.get("class_name", ""))
        if class_type == "human":
            cv2.fillPoly(human_mask, [rectangle], 255)
        elif class_type == "vehicle":
            cv2.fillPoly(vehicle_mask, [rectangle], 255)

    return all_mask, human_mask, vehicle_mask


def process_images(
    data_root: str,
    target_dir: str,
    cam_id: int,
    ref_dir: str,
    instances_info: Dict,
    frame_to_instances: Dict[int, List[int]],
    intrinsic: np.ndarray,
    cam_to_ego: np.ndarray,
    segmentation_model: Any,
    segmentation_inference: Any,
):
    image_files = sorted(
        [
            filename
            for filename in os.listdir(data_root)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ],
        key=lambda filename: (parse_frame_index(filename), filename),
    )

    if not image_files:
        raise RuntimeError(f"No images found in {data_root}")

    seen_frame_indices = set()
    for image_file in tqdm(image_files, desc="Processing synthetic frames"):
        frame_index = parse_frame_index(image_file)
        if frame_index in seen_frame_indices:
            raise RuntimeError(f"Duplicate frame index {frame_index} detected in {data_root}")
        seen_frame_indices.add(frame_index)

        image_path = os.path.join(data_root, image_file)
        rgb_image = Image.open(image_path).convert("RGB")
        width, height = rgb_image.size

        image_save_name = f"{frame_index:03d}_{cam_id}.jpg"
        rgb_image.save(os.path.join(target_dir, "images", image_save_name), "JPEG")

        pose_file = find_pose_file(ref_dir, frame_index)
        ego_to_world = load_matrix(pose_file, (4, 4))

        cam_to_world = ego_to_world @ cam_to_ego
        world_to_camera = np.linalg.inv(cam_to_world)
        frame_instance_ids = frame_to_instances.get(frame_index, [])
        all_mask, human_mask, vehicle_mask = generate_dynamic_masks(
            image_shape=(height, width),
            frame_index=frame_index,
            frame_instance_ids=frame_instance_ids,
            instances_info=instances_info,
            world_to_camera=world_to_camera,
            intrinsic=intrinsic,
        )

        cv2.imwrite(
            os.path.join(target_dir, "dynamic_masks", "all", f"{frame_index:03d}_{cam_id}.png"),
            all_mask,
        )
        cv2.imwrite(
            os.path.join(target_dir, "dynamic_masks", "human", f"{frame_index:03d}_{cam_id}.png"),
            human_mask,
        )
        cv2.imwrite(
            os.path.join(target_dir, "dynamic_masks", "vehicle", f"{frame_index:03d}_{cam_id}.png"),
            vehicle_mask,
        )

        semantic_result = segmentation_inference(segmentation_model, image_path)
        semantic_mask = semantic_result[0].astype(np.uint8)

        semantic_human = np.isin(semantic_mask, SEGMENTATION_HUMAN_CLASSES)
        semantic_vehicle = np.isin(semantic_mask, SEGMENTATION_VEHICLE_CLASSES)

        coarse_human = human_mask > 0
        coarse_vehicle = vehicle_mask > 0
        fine_human = np.logical_and(semantic_human, coarse_human)
        fine_vehicle = np.logical_and(semantic_vehicle, coarse_vehicle)
        fine_all = np.logical_or(fine_human, fine_vehicle)

        cv2.imwrite(
            os.path.join(target_dir, "fine_dynamic_masks", "all", f"{frame_index:03d}_{cam_id}.png"),
            (fine_all.astype(np.uint8) * 255),
        )
        cv2.imwrite(
            os.path.join(target_dir, "fine_dynamic_masks", "human", f"{frame_index:03d}_{cam_id}.png"),
            (fine_human.astype(np.uint8) * 255),
        )
        cv2.imwrite(
            os.path.join(target_dir, "fine_dynamic_masks", "vehicle", f"{frame_index:03d}_{cam_id}.png"),
            (fine_vehicle.astype(np.uint8) * 255),
        )

        sky_mask = np.isin(semantic_mask, [SEGMENTATION_SKY_CLASS]).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(target_dir, "sky_masks", f"{frame_index:03d}_{cam_id}.png"), sky_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess synthetic novel-view data for StreetPed")
    parser.add_argument("--data_root", type=str, required=True, help="Path to generated RGB images")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to save processed output")
    parser.add_argument(
        "--ref_dir",
        type=str,
        required=True,
        help="Path to reference processed scene containing instances and ego poses",
    )
    parser.add_argument("--cam_id", type=int, required=True, help="Target camera id for output filenames")
    parser.add_argument(
        "--cam_to_ego_file",
        type=str,
        default=None,
        help="Path to 4x4 camera-to-ego extrinsic matrix for the new camera",
    )
    parser.add_argument(
        "--intrinsics_file",
        type=str,
        default=None,
        help="Path to intrinsics file (3x3 matrix or fx,fy,cx,cy[,distortions])",
    )
    parser.add_argument(
        "--ref_intrinsics_cam_id",
        type=int,
        default=None,
        help="If intrinsics_file is not set, copy intrinsics from this reference cam id",
    )
    parser.add_argument(
        "--ref_extrinsics_cam_id",
        type=int,
        default=None,
        help="If cam_to_ego_file is not set, copy extrinsics from this reference cam id",
    )
    parser.add_argument(
        "--segformer_path",
        type=str,
        default=None,
        help="Path to SegFormer repo root (used when seg config/checkpoint are not explicitly set)",
    )
    parser.add_argument(
        "--seg_config",
        type=str,
        default=None,
        help="Path to SegFormer config file",
    )
    parser.add_argument(
        "--seg_checkpoint",
        type=str,
        default=None,
        help="Path to SegFormer checkpoint file",
    )
    parser.add_argument(
        "--seg_device",
        type=str,
        default="cuda:0",
        help="Device for segmentation inference",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    if not os.path.exists(args.ref_dir):
        raise FileNotFoundError(f"Reference directory does not exist: {args.ref_dir}")

    create_folder(args.target_dir)

    if args.cam_to_ego_file is not None:
        cam_to_ego_path = args.cam_to_ego_file
    else:
        fallback_extrinsics_cam_id = args.ref_extrinsics_cam_id
        if fallback_extrinsics_cam_id is None:
            fallback_extrinsics_cam_id = args.cam_id
        cam_to_ego_path = infer_ref_file(args.ref_dir, "extrinsics", fallback_extrinsics_cam_id)
    cam_to_ego = load_matrix(cam_to_ego_path, (4, 4))

    if args.intrinsics_file is not None:
        intrinsics_path = args.intrinsics_file
    else:
        fallback_intrinsics_cam_id = args.ref_intrinsics_cam_id
        if fallback_intrinsics_cam_id is None:
            fallback_intrinsics_cam_id = args.cam_id
        intrinsics_path = infer_ref_file(args.ref_dir, "intrinsics", fallback_intrinsics_cam_id)
    intrinsic, intrinsics_vector = get_intrinsics_from_file(intrinsics_path)

    np.savetxt(os.path.join(args.target_dir, "extrinsics", f"{args.cam_id}.txt"), cam_to_ego)
    np.savetxt(os.path.join(args.target_dir, "intrinsics", f"{args.cam_id}.txt"), intrinsics_vector)

    instances_info_path = os.path.join(args.ref_dir, "instances", "instances_info.json")
    if not os.path.exists(instances_info_path):
        raise FileNotFoundError(f"Missing instances metadata: {instances_info_path}")
    instances_info = json.load(open(instances_info_path, "r"))
    frame_instances_path = os.path.join(args.ref_dir, "instances", "frame_instances.json")
    frame_to_instances = build_frame_to_instances(instances_info, frame_instances_path)

    segmentation_model, segmentation_inference = init_segmentation_model(
        segformer_path=args.segformer_path,
        seg_config=args.seg_config,
        seg_checkpoint=args.seg_checkpoint,
        seg_device=args.seg_device,
    )

    process_images(
        data_root=args.data_root,
        target_dir=args.target_dir,
        cam_id=args.cam_id,
        ref_dir=args.ref_dir,
        instances_info=instances_info,
        frame_to_instances=frame_to_instances,
        intrinsic=intrinsic,
        cam_to_ego=cam_to_ego,
        segmentation_model=segmentation_model,
        segmentation_inference=segmentation_inference,
    )

    print("Preprocessing complete.")
