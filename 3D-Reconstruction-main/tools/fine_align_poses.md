The goal of this script is to align the camera poses to ensure proper alignment with the GT view.
The current implementation tries to minimize distances between point clouds, however does not account for RGB renders.
This new implementation tries to finetune camera poses to minimize the RGB difference between the rendered images and the GT images.
Inputs:
- Camera traj estimate of GT (slightly different from the GT traj)
- GT images
- Gaussian/ mesh model of the scene
Outputs:
- A new traj estimate that is more aligned with camera poses of the GT images
Method:
- Render the scene from the estimated camera poses
- Compare the rendered images with the GT images
- Generate n random perturbations of distance d rotation r of camera poses
- Render the scene from the perturbed camera poses
- Compare the rendered images with the GT images
- Keep the perturbation with lowest RGB difference
- Decrease d
- Decrease r
- Repeat until convergence or max iterations reached
