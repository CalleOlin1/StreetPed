"""
This code aims to only train the road mesh by using road masks, lidar and rgb images

The basic algorithm is as follows:
- 1 Aggregate point cloud using road masks and lidar data
    - (Code is available for this when initialising the point cloud for the road class for 3dgs training)
    (Log birds eye view of the point cloud)
- 2 Create a 3d mesh using the point cloud
    - Here we assume no overlap along the up axis
    - Need to use a reasonable amount of polys, can probably be fairly low amount of polys
    (Log mesh as .pth)
- 3 Create an image buffer for the road mesh
    - Use 4096x4096 resolution (to begin) and we can map pixels to the mesh simply using the x and y axis with a scaling factor
- 4 Project rays using rgb images and cam position onto the mesh
    - Only project road areas (according to road mask)
    - Here we fill the image buffer using the rgb data from images
    (Log image buffer as an image)
    (Log mesh with the image buffer as a texture)
    (Log example render from one of the camera positions of the mesh)

CLI args:
Which scene to train on, for example "data/paralane/processed/scene_000_clip_000"
Output folder for project, default "output"
Project name, ie "road_mesh_training"
Run name, ie "run_000"
So this logs as "output/road_mesh_training/run_000"
"""
