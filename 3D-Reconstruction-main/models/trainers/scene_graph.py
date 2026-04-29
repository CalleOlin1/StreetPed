from typing import Dict
import torch
import logging

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere, quat_to_rotmat, quaternion_multiply
from models.gaussians.basics import dataclass_camera

logger = logging.getLogger()


class MultiTrainer(BasicTrainer):
    def __init__(self, num_timesteps: int, **kwargs):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)
        self.render_each_class = True

    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(
            0, 1, num_timestamps, device=self.device
        )

    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "Road" in self.model_config:
            self.gaussian_classes["Road"] = GSModelType.Road
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes

        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)

            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device,
                )

            if class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get("params", {}),
                    n=self.num_full_images,
                    device=self.device,
                ).to(self.device)

            self.models[class_name] = model

        logger.info(f"Initialized models: {self.models.keys()}")

        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, "register_normalized_timestamps"):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, "set_bbox"):
                model.set_bbox(self.aabb)

    def safe_init_models(
        self,
        model: torch.nn.Module,
        instance_pts_dict: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(instance_pts_dict=instance_pts_dict)
            return False
        else:
            return True

    def _estimate_ground_normal_from_lidar(self, pts: torch.Tensor) -> torch.Tensor:
        if pts is None or pts.numel() == 0 or pts.shape[0] < 3:
            return None

        centered = pts - pts.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normal = eigvecs[:, torch.argmin(eigvals)]
        normal = normal / normal.norm().clamp_min(1e-8)

        if abs(normal[2].item()) > 1e-6 and normal[2] < 0:
            normal = -normal
        return normal
        
    def reinit_road_gaussians_from_dataset(self, dataset: DrivingDataset) -> None:
        """
        Reinitialise ONLY the Road Gaussians from scratch using the same logic
        as init_gaussians_from_dataset (Road branch).
        """

        if "Road" not in self.gaussian_classes:
            logger.warning("Road class not found in gaussian_classes; skipping reinit.")
            return

        model_cfg = self.model_config["Road"]
        model = self.models["Road"]

        # Optional: fully reset model state if supported
        if hasattr(model, "reset"):
            model.reset()

        init_cfg = model_cfg["init"]
        strict_mask_only = bool(init_cfg.get("strict_mask_only", True))

        # ---------------- lidar sampling ----------------
        candidate_indices = None
        sampled_pts, sampled_color, sampled_time = (
            torch.empty(0, 3).to(self.device),
            torch.empty(0, 3).to(self.device),
            None,
        )

        if init_cfg.get("from_lidar", None) is not None:
            sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                **init_cfg.from_lidar,
                candidate_indices=candidate_indices,
                device=self.device,
            )

        # ---------------- road mask sampling ----------------
        requested_num_samples = sampled_pts.shape[0]
        if init_cfg.get("from_lidar", None) is not None:
            requested_num_samples = int(init_cfg.from_lidar.get(
                "num_samples", requested_num_samples
            ))

        road_mask_pts, road_mask_colors = dataset.get_lidar_points_from_mask_region(
            mask_attr="road_masks",
            num_samples=requested_num_samples,
            return_color=True,
            device=self.device,
        )

        sampled_pts = road_mask_pts
        sampled_color = road_mask_colors

        if strict_mask_only:
            logger.info(
                "Road strict_mask_only enabled: using only road-mask LiDAR seeds (%d pts)",
                int(road_mask_pts.shape[0]),
            )
        elif road_mask_pts.shape[0] < 3 and init_cfg.get("from_lidar", None) is not None:
            logger.warning(
                "Road mask LiDAR seed too small (%d pts), falling back to generic LiDAR seeds",
                int(road_mask_pts.shape[0]),
            )
            sampled_pts, sampled_color, _ = dataset.get_lidar_samples(
                **init_cfg.from_lidar,
                candidate_indices=None,
                device=self.device,
            )

        # ensure >= 3 points for scale estimation
        if sampled_pts.shape[0] > 0 and sampled_pts.shape[0] < 3:
            repeat_count = 3 - sampled_pts.shape[0]
            sampled_pts = torch.cat(
                [sampled_pts, sampled_pts[:1].repeat(repeat_count, 1)], dim=0
            )
            sampled_color = torch.cat(
                [sampled_color, sampled_color[:1].repeat(repeat_count, 1)], dim=0
            )

        road_lidar_pts = sampled_pts

        # ---------------- random sampling ----------------
        random_pts = []
        num_near_pts = init_cfg.get("near_randoms", 0)
        if num_near_pts > 0:
            num_near_pts *= 3
            random_pts.append(uniform_sample_sphere(num_near_pts, self.device))

        num_far_pts = init_cfg.get("far_randoms", 0)
        if num_far_pts > 0:
            num_far_pts *= 3
            random_pts.append(
                uniform_sample_sphere(num_far_pts, self.device, inverse=True)
            )

        if len(random_pts) > 0:
            random_pts = torch.cat(random_pts, dim=0)
            random_pts = random_pts * self.scene_radius + self.scene_origin

            visible_mask = dataset.check_pts_visibility(random_pts)
            valid_pts = random_pts[visible_mask]

            sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
            sampled_color = torch.cat(
                [
                    sampled_color,
                    torch.rand(valid_pts.shape).to(self.device),
                ],
                dim=0,
            )

        # ---------------- filtering ----------------
        processed_init_pts = dataset.filter_pts_in_boxes(
            seed_pts=sampled_pts,
            seed_colors=sampled_color,
            valid_instances_dict={},  # Road has no instance filtering
        )

        if processed_init_pts["pts"].shape[0] < 3:
            logger.warning(
                "Road reinit has too few points (%d), aborting",
                int(processed_init_pts["pts"].shape[0]),
            )
            return

        # ---------------- create gaussians ----------------
        model.create_from_pcd(
            init_means=processed_init_pts["pts"],
            init_colors=processed_init_pts["colors"],
        )

        # ---------------- surface normal lock ----------------
        if hasattr(model, "set_surface_normal_lock"):
            requested_num_samples = 200000
            if init_cfg.get("from_lidar", None) is not None:
                requested_num_samples = int(
                    init_cfg.from_lidar.get("num_samples", requested_num_samples)
                )

            road_normal_pts = dataset.get_lidar_points_from_mask_region(
                mask_attr="road_masks",
                num_samples=min(requested_num_samples, 200000),
                device=self.device,
            )

            if road_normal_pts is None or road_normal_pts.shape[0] < 3:
                road_normal_pts = road_lidar_pts

            road_normal = self._estimate_ground_normal_from_lidar(road_normal_pts)

            if road_normal is not None:
                model.set_surface_normal_lock(road_normal)
                logger.info(
                    "Reinitialised Road surface-normal lock from %d points: %s",
                    int(road_normal_pts.shape[0]) if road_normal_pts is not None else 0,
                    road_normal.detach().cpu().tolist(),
                )

        logger.info("Reinitialised Road gaussians from dataset")

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        # get instance points
        rigidnode_pts_dict, deformnode_pts_dict, smplnode_pts_dict = {}, {}, {}
        if "RigidNodes" in self.model_config:
            rigidnode_pts_dict = dataset.get_init_objects(
                cur_node_type="RigidNodes", **self.model_config["RigidNodes"]["init"]
            )

        if "DeformableNodes" in self.model_config:
            deformnode_pts_dict = dataset.get_init_objects(
                cur_node_type="DeformableNodes",
                exclude_smpl="SMPLNodes" in self.model_config,
                **self.model_config["DeformableNodes"]["init"],
            )

        if "SMPLNodes" in self.model_config:
            smplnode_pts_dict = dataset.get_init_smpl_objects(
                **self.model_config["SMPLNodes"]["init"]
            )
        allnode_pts_dict = {
            **rigidnode_pts_dict,
            **deformnode_pts_dict,
            **smplnode_pts_dict,
        }

        # NOTE: Some gaussian classes may be empty (because no points for initialization)
        #       We will delete these classes from the model_config and models
        empty_classes = []

        # collect models
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]

            empty = False
            if class_name in ("Background", "Road"):
                # ------ initialize gaussians ------
                init_cfg = model_cfg.get("init")
                strict_mask_only = bool(init_cfg.get("strict_mask_only", class_name == "Road"))
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    candidate_indices = None
                    if class_name == "Background" and "Road" in self.gaussian_classes:
                        road_keep_indices = dataset.get_lidar_indices_from_mask_region(mask_attr="road_masks")
                        if road_keep_indices.numel() > 0:
                            road_keep_mask = torch.zeros(
                                dataset.lidar_source.num_points,
                                dtype=torch.bool,
                                device=road_keep_indices.device,
                            )
                            road_keep_mask[road_keep_indices] = True
                            candidate_indices = torch.nonzero(~road_keep_mask, as_tuple=False).squeeze(-1)
                            logger.info(
                                "Background LiDAR sampling excludes %d road-mask points; %d candidates remain",
                                int(road_keep_indices.numel()),
                                int(candidate_indices.numel()),
                            )
                            if candidate_indices.numel() == 0:
                                logger.warning("No non-road LiDAR candidates for Background; falling back to full LiDAR")
                                candidate_indices = None

                    sampled_pts, sampled_color, sampled_time = (
                        dataset.get_lidar_samples(
                            **init_cfg.from_lidar,
                            candidate_indices=candidate_indices,
                            device=self.device,
                        )
                    )
                else:
                    sampled_pts, sampled_color, sampled_time = (
                        torch.empty(0, 3).to(self.device),
                        torch.empty(0, 3).to(self.device),
                        None,
                    )

                if class_name == "Road":
                    requested_num_samples = sampled_pts.shape[0]
                    if init_cfg.get("from_lidar", None) is not None:
                        requested_num_samples = int(init_cfg.from_lidar.get("num_samples", requested_num_samples))
                    road_mask_pts, road_mask_colors = dataset.get_lidar_points_from_mask_region(
                        mask_attr="road_masks",
                        num_samples=requested_num_samples,
                        return_color=True,
                        device=self.device,
                    )
                    sampled_pts = road_mask_pts
                    sampled_color = road_mask_colors
                    if strict_mask_only:
                        logger.info(
                            "Road strict_mask_only enabled: using only road-mask LiDAR seeds (%d pts)",
                            int(road_mask_pts.shape[0]),
                        )
                    elif road_mask_pts.shape[0] < 3 and init_cfg.get("from_lidar", None) is not None:
                        logger.warning(
                            "Road mask LiDAR seed is too small (%d pts), falling back to generic LiDAR seeds",
                            int(road_mask_pts.shape[0]),
                        )
                        sampled_pts, sampled_color, _ = dataset.get_lidar_samples(
                            **init_cfg.from_lidar,
                            candidate_indices=None,
                            device=self.device,
                        )
                    elif road_mask_pts.shape[0] < 3:
                        logger.warning(
                            "Road mask LiDAR seed is too small (%d pts) and no from_lidar fallback is configured",
                            int(road_mask_pts.shape[0]),
                        )

                    # Gaussian init needs >=3 points for nearest-neighbor scale estimation.
                    if sampled_pts.shape[0] > 0 and sampled_pts.shape[0] < 3:
                        repeat_count = 3 - sampled_pts.shape[0]
                        sampled_pts = torch.cat([sampled_pts, sampled_pts[:1].repeat(repeat_count, 1)], dim=0)
                        sampled_color = torch.cat([sampled_color, sampled_color[:1].repeat(repeat_count, 1)], dim=0)

                road_lidar_pts = sampled_pts if class_name == "Road" else None

                random_pts = []
                num_near_pts = init_cfg.get("near_randoms", 0)
                if (
                    num_near_pts > 0
                ):  # uniformly sample points inside the scene's sphere
                    num_near_pts *= (
                        3  # since some invisible points will be filtered out
                    )
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get("far_randoms", 0)
                if (
                    num_far_pts > 0
                ):  # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(
                        uniform_sample_sphere(num_far_pts, self.device, inverse=True)
                    )

                if class_name == "Road" and strict_mask_only and (num_near_pts + num_far_pts > 0):
                    logger.info("Road strict_mask_only enabled: skipping near/far random seed points")
                elif num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0)
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]

                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat(
                        [
                            sampled_color,
                            torch.rand(
                                valid_pts.shape,
                            ).to(self.device),
                        ],
                        dim=0,
                    )

                processed_init_pts = dataset.filter_pts_in_boxes(
                    seed_pts=sampled_pts,
                    seed_colors=sampled_color,
                    valid_instances_dict=allnode_pts_dict,
                )

                if processed_init_pts["pts"].shape[0] < 3:
                    empty = True
                    logger.warning(
                        "%s init has too few points after filtering (%d), skipping class",
                        class_name,
                        int(processed_init_pts["pts"].shape[0]),
                    )
                    continue

                model.create_from_pcd(
                    init_means=processed_init_pts["pts"],
                    init_colors=processed_init_pts["colors"],
                )

                if class_name == "Road" and hasattr(model, "set_surface_normal_lock"):
                    requested_num_samples = 200000
                    if init_cfg.get("from_lidar", None) is not None:
                        requested_num_samples = int(init_cfg.from_lidar.get("num_samples", requested_num_samples))
                    road_normal_pts = dataset.get_lidar_points_from_mask_region(
                        mask_attr="road_masks",
                        num_samples=min(requested_num_samples, 200000),
                        device=self.device,
                    )
                    if road_normal_pts is None or road_normal_pts.shape[0] < 3:
                        road_normal_pts = road_lidar_pts
                    road_normal = self._estimate_ground_normal_from_lidar(road_normal_pts)
                    if road_normal is not None:
                        model.set_surface_normal_lock(road_normal)
                        logger.info(
                            "Applied Road surface-normal lock from %d road-mask LiDAR points: %s",
                            int(road_normal_pts.shape[0]) if road_normal_pts is not None else 0,
                            road_normal.detach().cpu().tolist(),
                        )

            if class_name == "RigidNodes":
                empty = self.safe_init_models(
                    model=model, instance_pts_dict=rigidnode_pts_dict
                )

            if class_name == "DeformableNodes":
                empty = self.safe_init_models(
                    model=model, instance_pts_dict=deformnode_pts_dict
                )

            if class_name == "SMPLNodes":
                empty = self.safe_init_models(
                    model=model, instance_pts_dict=smplnode_pts_dict
                )

            if empty:
                empty_classes.append(class_name)
                logger.warning(
                    f"No points for {class_name} found, will remove the model"
                )
            else:
                logger.info(f"Initialized {class_name} gaussians")

        if len(empty_classes) > 0:
            for class_name in empty_classes:
                del self.models[class_name]
                del self.model_config[class_name]
                del self.gaussian_classes[class_name]
                logger.warning(f"Model for {class_name} is removed")

        self.log_gaussian_count(prefix="Gaussian counts after dataset initialization")
        logger.info(f"Initialized gaussians from pcd")

    def forward(
        self,
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )

        # for evaluation
        for model in self.models.values():
            if hasattr(model, "in_test_set"):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, "set_cur_frame"):
                model.set_cur_frame(self.cur_frame)

        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view,
        )
        gs = self.collect_gaussians(
            cam=processed_cam, image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, render_fn = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get("radius_clip", 0.0),
        )

        # render sky
        sky_model = self.models["Sky"]
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])

        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]),
            image_infos,
        )

        if self.training and "Background" in self.gaussian_classes and "sky_masks" in image_infos:
            gaussian_mask = self.pts_labels == self.gaussian_classes["Background"]
            bg_rgb, bg_depth, bg_opacity = render_fn(gaussian_mask)
            outputs["Background_rgb"] = self.affine_transformation(bg_rgb, image_infos)
            outputs["Background_opacity"] = bg_opacity
            outputs["Background_depth"] = bg_depth

        if self.training and "Road" in self.gaussian_classes and "road_masks" in image_infos:
            gaussian_mask = self.pts_labels == self.gaussian_classes["Road"]
            road_rgb, road_depth, road_opacity = render_fn(gaussian_mask)
            outputs["Road_rgb"] = self.affine_transformation(road_rgb, image_infos)
            outputs["Road_opacity"] = road_opacity
            outputs["Road_depth"] = road_depth

        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                    outputs[class_name + "_rgb"] = self.affine_transformation(
                        sep_rgb, image_infos
                    )
                    outputs[class_name + "_opacity"] = sep_opacity
                    outputs[class_name + "_depth"] = sep_depth

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                if "Road" in self.gaussian_classes:
                    gaussian_mask = gaussian_mask & (self.pts_labels != self.gaussian_classes["Road"])
                sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                outputs["Dynamic_rgb"] = self.affine_transformation(
                    sep_rgb, image_infos
                )
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth

        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)

        return loss_dict

    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)

        return metric_dict

    def get_rigid_info(self, frame_idx: int = None):
        """Get rigid node information"""
        if frame_idx is None:
            frame_idx = self.cur_frame.item()

        if "RigidNodes" not in self.models:
            return None

        rigid_model = self.models["RigidNodes"]

        # Set current frame
        self.cur_frame = torch.tensor(frame_idx, device=self.device)
        if hasattr(rigid_model, "set_cur_frame"):
            rigid_model.set_cur_frame(self.cur_frame)
        elif hasattr(rigid_model, "cur_frame"):
            rigid_model.cur_frame = self.cur_frame

        # Create dummy camera
        dummy_cam = dataclass_camera(
            camtoworlds=torch.eye(4, device=self.device)[None],
            camtoworlds_gt=torch.eye(4, device=self.device)[None],
            Ks=torch.eye(3, device=self.device)[None],
            H=1,
            W=1,
        )

        gs = rigid_model.get_gaussians(dummy_cam)
        if gs is None:
            return None

        result = {
            "positions": gs["_means"].detach(),
            "instance_ids": (
                rigid_model.point_ids[..., 0]
                if hasattr(rigid_model, "point_ids")
                else None
            ),
        }

        # Get transform information
        if hasattr(rigid_model, "instances_quats") and hasattr(
            rigid_model, "instances_trans"
        ):
            quats = rigid_model.instances_quats[frame_idx]  # [num_instances, 4]
            trans = rigid_model.instances_trans[frame_idx]  # [num_instances, 3]

            if hasattr(rigid_model, "quat_act"):
                quats = rigid_model.quat_act(quats)

            # 使用我们自己写的函数
            rot_matrices = quat_to_rotmat(quats)

            result["transforms"] = {
                "rotations": rot_matrices.detach(),
                "translations": trans.detach(),
                "quaternions": quats.detach(),
            }

        return result

    def get_smpl_info(self, frame_idx: int = None):
        """Get SMPL node information"""
        if frame_idx is None:
            frame_idx = self.cur_frame.item()

        if "SMPLNodes" not in self.models:
            return None

        smpl_model = self.models["SMPLNodes"]

        # Set current frame
        self.cur_frame = torch.tensor(frame_idx, device=self.device)
        if hasattr(smpl_model, "set_cur_frame"):
            smpl_model.set_cur_frame(self.cur_frame)

        # Create dummy camera
        dummy_cam = dataclass_camera(
            camtoworlds=torch.eye(4, device=self.device)[None],
            camtoworlds_gt=torch.eye(4, device=self.device)[None],
            Ks=torch.eye(3, device=self.device)[None],
            H=1,
            W=1,
        )

        gs = smpl_model.get_gaussians(dummy_cam)
        if gs is None:
            return None

        result = {
            "positions": gs["_means"].detach(),
            "instance_ids": (
                smpl_model.point_ids[..., 0]
                if hasattr(smpl_model, "point_ids")
                else None
            ),
        }

        # 获取SMPL特有的参数
        if hasattr(smpl_model, "smpl_qauts") and hasattr(smpl_model, "instances_trans"):
            smpl_quats = smpl_model.smpl_qauts[frame_idx]  # [num_instances, 23, 4]
            global_quats = smpl_model.instances_quats[
                frame_idx
            ]  # [num_instances, 1, 4]
            trans = smpl_model.instances_trans[frame_idx]  # [num_instances, 3]

            if hasattr(smpl_model, "quat_act"):
                smpl_quats = smpl_model.quat_act(smpl_quats)
                global_quats = smpl_model.quat_act(global_quats)

            result["transforms"] = {
                "smpl_quats": smpl_quats.detach(),
                "global_quats": global_quats.detach(),
                "translations": trans.detach(),
            }

        return result
