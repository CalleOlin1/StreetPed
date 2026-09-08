Format Specification
Overview
Our dataset contains 25 scenes, and each scene includes 3 clips. The file structure within
a scene is as follows:
Copy
Images
In each clip directory, there are two image folders: 'images' and 'foreground masks'.
The 'images' folder contains the original images captured by our five cameras. Every 
five images belonging to the same frame are grouped together in a folder named with
the corresponding timestamp, and the images within a frame are named according to
the camera names. It should look like this:
Copy
├── scene_0
│ ├── clip_0
│ │ ├── foreground_mask 
│ │ ├── images
│ │ ├── lidars
│ │ ├── sparse
│ │ └── video.mp4
│ ├── clip_1
│ │ ├── foreground_mask 
│ │ ├── images
│ │ ├── lidars
│ │ ├── sparse
│ │ └── video.mp4
│ └── clip_2
│ ├── foreground_mask 
│ ├── images
│ ├── lidars
│ ├── sparse
│ └── video.mp4
├── scene_1
...
●
1 / 3
However, we are currently only releasing the images from the front camera, so each
frame directory contains only one image.
Sensitive text and personal information in the original images have been blurred.
The 'foreground_mask' folders contain the foreground masks of the original images.
You may need these masks to isolate foreground objects when performing NVS tasks,
as these objects can move and may not be consistent across frames and clips.
Poses & lidars
Copy
In the "sparse" directory, we provide the following informations:
/mnt_gx/ziqian_data/paralane_publish/0315/scene_0/clip_0/images
├── 1718346704099615
│ ├── CAMERA_FRONT.png
│ ├── CAMERA_PANO_FRONT.png
│ ├── CAMERA_PANO_LEFT.png
│ ├── CAMERA_PANO_BACK.png
│ └── CAMERA_PANO_RIGHT.png
├── 1718346704199621
│ ├── CAMERA_FRONT.png
│ ├── CAMERA_PANO_FRONT.png
│ ├── CAMERA_PANO_LEFT.png
│ ├── CAMERA_PANO_BACK.png
│ └── CAMERA_PANO_RIGHT.png
├── 1718346704299617
...
●
●
/sparse
`-- 0
 |-- cameras.txt
 |-- images.txt
 |-- lidar_poses.txt
 |-- merged_lidar_pcd.ply
 `-- visual_points.ply
● SfM infos:
'cameras.txt' & 'images.txt': These files contain the camera intrinsic parameters 
and image poses, all provided in COLMAP format (more details can be found
here). Similar to the images, we only provide information about the front camera in
these files.
○
2 / 3
'visual_points.ply': This file contains the point cloud generated through Structure
from Motion (SfM), which is also referred to as triangulated visual landmarks.
○
● lidars
'merged_lidar_pcd.ply': This is the global point cloud that we stitched together
from laser point clouds taken every 5 frames. Compared to 'visual_points.ply', it is 
denser but has a more limited range.
'lidar_poses.txt': This file contains the poses of individual lidar frames in relation to
the global coordinate system. The transformation is written in the order of Qx, Qy,
Qz, Qw, X, Y, Z.
In the 'lidars' folder, we provide individual lidar frames, with the foreground point
clouds removed.
○
○
○
3 / 3
