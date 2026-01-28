import numpy as np

# ego_to_cam = np.array(
#     [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0], [1.0, 0.0, 0.0, 0]]
# )

# # Rotation matrix R
# R = np.array(
#     [
#         [7.533745e-03, -9.999714e-01, -6.166020e-04],
#         [1.480249e-02, 7.280733e-04, -9.998902e-01],
#         [9.998621e-01, 7.523790e-03, 1.480755e-02],
#     ]
# )

# Rotation matrix R
R = np.array(
    [
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ]
)

# Translation vector T
T = np.array([0.0, 0.1, 0.0])

# Build 4x4 homogeneous transformation matrix
transform_matrix = np.eye(4)  # Initialize identity matrix
transform_matrix[:3, :3] = R  # Fill rotation part
transform_matrix[:3, 3] = T  # Fill translation part

transform_matrix = np.linalg.inv(transform_matrix)
print("transform_matrix:")
print(transform_matrix)

cam_list = ["SUB_RGB_1", "SUB_RGB_2"]
cam_configs = {
    "SUB_RGB_1": {
        "BLUEPRINT": "sensor.camera.rgb",
        "ATTRIBUTE": {"image_size_x": 1920, "image_size_y": 1080, "fov": 90},
        "TRANSFORM": {"location": [0, 0.1, 0.0], "rotation": [0, 90, -90]},
    },
    "SUB_RGB_2": {
        "BLUEPRINT": "sensor.camera.rgb",
        "ATTRIBUTE": {"image_size_x": 1920, "image_size_y": 1080, "fov": 90},
        "TRANSFORM": {"location": [0, -0.1, 0.0], "rotation": [0, 90, -90]},
    },
}
for i, cam_name in enumerate(cam_list):
    cam_config = cam_configs[cam_name]

    location = cam_config["TRANSFORM"]["location"]
    rotation = cam_config["TRANSFORM"]["rotation"]
    # Convert euler angles to rotation matrix (ZYX order)
    rx, ry, rz = np.radians(rotation)
    # Build rotation matrices for each axis (right-hand coordinate system, ZYX order)
    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )

    Ry = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )

    R = np.linalg.inv(Rx @ Ry @ Rz)
    print(R)
    R = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]])

    # Build 4x4 transformation matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :4] = R
    extrinsic_matrix[:3, 3] = location
    # print(extrinsic_matrix)
    # 转换为相机坐标系（从世界到相机）
    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
    print(extrinsic_matrix)
