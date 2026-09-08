"""
@file   extract_masks.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract semantic mask

Using SegFormer, 2021. Cityscapes 83.2%
Relies on timm==0.3.2 & pytorch 1.8.1 (buggy on pytorch >= 1.9)

Installation:
    NOTE: mmcv-full==1.2.7 requires another pytorch version & conda env.
        Currently mmcv-full==1.2.7 does not support pytorch>=1.9; 
            will raise AttributeError: 'super' object has no attribute '_specify_ddp_gpu_num'
        Hence, a seperate conda env is needed.

    git clone https://github.com/NVlabs/SegFormer

    conda create -n segformer python=3.8
    conda activate segformer
    # conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf
    pip install mmcv-full==1.2.7 --no-cache-dir
    
    cd SegFormer
    pip install .

Usage:
    Direct run this script in the newly set conda env.
"""


from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

semantic_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]
dataset_classes_in_sematic = {
    'Vehicle': [13, 14, 15],   # 'car', 'truck', 'bus'
    'human': [11, 12, 17, 18], # 'person', 'rider', 'motorcycle', 'bicycle'
}

if __name__ == "__main__":
    import os
    import imageio
    import numpy as np
    from glob import glob
    from tqdm import tqdm
    from argparse import ArgumentParser, SUPPRESS
    parser = ArgumentParser()
    # Custom configs
    parser.add_argument('--data_root', type=str, default='data/waymo/processed/training')
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=str,
        nargs="+",
        help="scene ids to be processed. Supports numeric IDs (e.g. 0 1) and named IDs (e.g. scene_001_clip_000).",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        '--process_dynamic_mask',
        action='store_true',
        help="Whether to process dynamic masks",
    )
    parser.add_argument(
        '--process_road_mask',
        '--process_road',
        dest='process_road_mask',
        action='store_true',
        help="Whether to process road masks",
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--no_compress', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--mask_dirname', type=str, default="fine_dynamic_masks")

    # Algorithm configs
    parser.add_argument('--segformer_path', type=str, default='/home/guojianfei/ai_ws/SegFormer')
    parser.add_argument('--config', help='Config file', type=str, default=None)
    parser.add_argument('--checkpoint', help='Checkpoint file', type=str, default=None)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='cityscapes', help='Color palette used for segmentation map')
    parser.add_argument('--flat_progress', action='store_true', help=SUPPRESS)
    parser.add_argument('--progress_position', type=int, default=0, help=SUPPRESS)
    parser.add_argument('--progress_desc', type=str, default=None, help=SUPPRESS)
    parser.add_argument('--progress_disable', action='store_true', help=SUPPRESS)
    parser.add_argument('--progress_leave', action='store_true', help=SUPPRESS)
    
    args = parser.parse_args()
    if args.config is None:
        args.config = os.path.join(
            args.segformer_path,
            'local_configs',
            'segformer',
            'B5',
            'segformer.b5.1024x1024.city.160k.py',
        )
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.segformer_path, 'pretrained', 'segformer.b5.1024x1024.city.160k.pth')
    
    if args.scene_ids is not None:
        scene_ids_list = []
        for scene_id in args.scene_ids:
            if scene_id.isdigit():
                scene_ids_list.append(int(scene_id))
            else:
                scene_ids_list.append(scene_id)
    elif args.split_file is not None:
        # parse the split file
        split_lines = open(args.split_file, "r").readlines()
        # Support both header + rows and plain row files.
        if len(split_lines) > 0 and split_lines[0].lstrip().startswith("#"):
            split_lines = split_lines[1:]
        scene_ids_list = []
        for line in split_lines:
            token = line.strip().split(",")[0]
            if token == "":
                continue
            if token.isdigit():
                scene_ids_list.append(int(token))
            else:
                scene_ids_list.append(token)
    else:
        discovered_scene_ids = sorted(
            scene_name
            for scene_name in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, scene_name))
            and os.path.isdir(os.path.join(args.data_root, scene_name, args.rgb_dirname))
        )
        if discovered_scene_ids:
            scene_ids_list = discovered_scene_ids[args.start_idx: args.start_idx + args.num_scenes]
        else:
            scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)
    
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    
    scene_contexts = {}
    all_jobs = []
    for scene_id in scene_ids_list:
        scene_id = str(scene_id)
        if scene_id.isdigit():
            scene_id = scene_id.zfill(3)
        img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname)

        # create mask dir
        sky_mask_dir = os.path.join(args.data_root, scene_id, "sky_masks")
        if not os.path.exists(sky_mask_dir):
            os.makedirs(sky_mask_dir)

        scene_context = {
            "sky_mask_dir": sky_mask_dir,
        }

        # create dynamic mask dir
        if args.process_dynamic_mask:
            rough_human_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "human")
            rough_vehicle_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "vehicle")

            all_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "all")
            if not os.path.exists(all_mask_dir):
                os.makedirs(all_mask_dir)
            human_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "human")
            if not os.path.exists(human_mask_dir):
                os.makedirs(human_mask_dir)
            vehicle_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "vehicle")
            if not os.path.exists(vehicle_mask_dir):
                os.makedirs(vehicle_mask_dir)

            scene_context["rough_human_mask_dir"] = rough_human_mask_dir
            scene_context["rough_vehicle_mask_dir"] = rough_vehicle_mask_dir
            scene_context["all_mask_dir"] = all_mask_dir
            scene_context["human_mask_dir"] = human_mask_dir
            scene_context["vehicle_mask_dir"] = vehicle_mask_dir

        # create road mask dir
        if args.process_road_mask:
            road_mask_dir = os.path.join(args.data_root, scene_id, "road_masks")
            if not os.path.exists(road_mask_dir):
                os.makedirs(road_mask_dir)
            scene_context["road_mask_dir"] = road_mask_dir

        flist = sorted(glob(os.path.join(img_dir, '*')))
        scene_contexts[scene_id] = scene_context
        all_jobs.append((scene_id, flist))

    progress_kwargs = {
        "position": args.progress_position,
        "leave": args.progress_leave,
        "disable": args.progress_disable,
        "dynamic_ncols": True,
    }

    if args.flat_progress:
        frame_jobs = []
        for scene_id, flist in all_jobs:
            frame_jobs.extend((scene_id, fpath) for fpath in flist)

        progress_desc = args.progress_desc if args.progress_desc is not None else "Extracting Masks ..."
        frame_iter = tqdm(frame_jobs, desc=progress_desc, **progress_kwargs)
        for scene_id, fpath in frame_iter:
            fbase = os.path.splitext(os.path.basename(os.path.normpath(fpath)))[0]
            scene_context = scene_contexts[scene_id]

            if args.ignore_existing and os.path.exists(os.path.join(args.data_root, scene_id, "fine_dynamic_masks")):
                continue

            result = inference_segmentor(model, fpath)
            mask = result[0].astype(np.uint8)   # NOTE: in the settings of "cityscapes", there are 19 classes at most.

            sky_mask = np.isin(mask, [10])
            imageio.imwrite(os.path.join(scene_context["sky_mask_dir"], f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)

            if args.process_road_mask:
                road_mask = np.isin(mask, [0])
                imageio.imwrite(os.path.join(scene_context["road_mask_dir"], f"{fbase}.png"), road_mask.astype(np.uint8) * 255)

            if args.process_dynamic_mask:
                rough_human_mask_path = os.path.join(scene_context["rough_human_mask_dir"], f"{fbase}.png")
                rough_human_mask = (imageio.imread(rough_human_mask_path) > 0)
                huamn_mask = np.isin(mask, dataset_classes_in_sematic['human'])
                valid_human_mask = np.logical_and(huamn_mask, rough_human_mask)
                imageio.imwrite(os.path.join(scene_context["human_mask_dir"], f"{fbase}.png"), valid_human_mask.astype(np.uint8) * 255)

                rough_vehicle_mask_path = os.path.join(scene_context["rough_vehicle_mask_dir"], f"{fbase}.png")
                rough_vehicle_mask = (imageio.imread(rough_vehicle_mask_path) > 0)
                vehicle_mask = np.isin(mask, dataset_classes_in_sematic['Vehicle'])
                valid_vehicle_mask = np.logical_and(vehicle_mask, rough_vehicle_mask)
                imageio.imwrite(os.path.join(scene_context["vehicle_mask_dir"], f"{fbase}.png"), valid_vehicle_mask.astype(np.uint8) * 255)

                valid_all_mask = np.logical_or(valid_human_mask, valid_vehicle_mask)
                imageio.imwrite(os.path.join(scene_context["all_mask_dir"], f"{fbase}.png"), valid_all_mask.astype(np.uint8) * 255)
    else:
        scene_desc = args.progress_desc if args.progress_desc is not None else "Extracting Masks ..."
        for scene_id, flist in tqdm(all_jobs, desc=scene_desc, **progress_kwargs):
            scene_context = scene_contexts[scene_id]
            inner_desc = f"scene[{scene_id}]"
            for fpath in tqdm(flist, desc=inner_desc, **progress_kwargs):
                fbase = os.path.splitext(os.path.basename(os.path.normpath(fpath)))[0]

                if args.ignore_existing and os.path.exists(os.path.join(args.data_root, scene_id, "fine_dynamic_masks")):
                    continue

                result = inference_segmentor(model, fpath)
                mask = result[0].astype(np.uint8)   # NOTE: in the settings of "cityscapes", there are 19 classes at most.

                sky_mask = np.isin(mask, [10])
                imageio.imwrite(os.path.join(scene_context["sky_mask_dir"], f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)

                if args.process_road_mask:
                    road_mask = np.isin(mask, [0])
                    imageio.imwrite(os.path.join(scene_context["road_mask_dir"], f"{fbase}.png"), road_mask.astype(np.uint8) * 255)

                if args.process_dynamic_mask:
                    rough_human_mask_path = os.path.join(scene_context["rough_human_mask_dir"], f"{fbase}.png")
                    rough_human_mask = (imageio.imread(rough_human_mask_path) > 0)
                    huamn_mask = np.isin(mask, dataset_classes_in_sematic['human'])
                    valid_human_mask = np.logical_and(huamn_mask, rough_human_mask)
                    imageio.imwrite(os.path.join(scene_context["human_mask_dir"], f"{fbase}.png"), valid_human_mask.astype(np.uint8) * 255)

                    rough_vehicle_mask_path = os.path.join(scene_context["rough_vehicle_mask_dir"], f"{fbase}.png")
                    rough_vehicle_mask = (imageio.imread(rough_vehicle_mask_path) > 0)
                    vehicle_mask = np.isin(mask, dataset_classes_in_sematic['Vehicle'])
                    valid_vehicle_mask = np.logical_and(vehicle_mask, rough_vehicle_mask)
                    imageio.imwrite(os.path.join(scene_context["vehicle_mask_dir"], f"{fbase}.png"), valid_vehicle_mask.astype(np.uint8) * 255)

                    valid_all_mask = np.logical_or(valid_human_mask, valid_vehicle_mask)
                    imageio.imwrite(os.path.join(scene_context["all_mask_dir"], f"{fbase}.png"), valid_all_mask.astype(np.uint8) * 255)