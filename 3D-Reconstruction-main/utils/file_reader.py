import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models.video_utils import compute_psnr  # Ensure correct function import
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import imageio
import shutil
from skimage.metrics import structural_similarity as ssim


def create_comparison_video(npy_dir, png_dir, view, output_path="comparison.mp4", fps=30, exclude_rows=0):
    """
    Generate three-image comparison video
    
    Args:
        npy_dir: .npy file directory
        png_dir: .png file directory
        output_path: Output video path
        fps: Video frame rate
        exclude_rows: Number of bottom rows to exclude
    """
    # Create temporary directory
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate all comparison images
    frame_paths = []
    for i in tqdm(range(150), desc="Generating comparison images"):
        npy_path = os.path.join(npy_dir, f"new_frame{i:04d}.npy")
        png_path = os.path.join(png_dir, f"{i:06d}_new_view_{view}.png")
        
        if not (os.path.exists(npy_path) and os.path.exists(png_path)):
            continue
            
        # Generate comparison image and save
        fig = create_combined_figure(npy_path, png_path, exclude_rows)
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)
    
    # Generate video
    with imageio.get_writer(output_path, fps=fps) as writer:
        for path in tqdm(frame_paths, desc="Generating video"):
            image = imageio.imread(path)
            writer.append_data(image)
    
    print(f"Video saved to: {output_path}")

    shutil.rmtree(temp_dir)
    print(f"Temporary files cleaned up: {temp_dir}")


def create_combined_figure(npy_path, png_path, exclude_rows):
    """
    Create a Figure object for three-image comparison
    """
    # Load data
    render = np.load(npy_path)
    gt = np.array(Image.open(png_path).convert('RGB'))
    gt = np.array(Image.fromarray(gt).resize((render.shape[1], render.shape[0]))) / 255.0
    
    # Crop
    H, W = render.shape[:2]
    render = render[:H-exclude_rows]
    gt = gt[:H-exclude_rows]
    
    # Calculate difference
    diff = np.abs(render - gt).mean(axis=-1)
    
    # Create canvas
    fig = plt.figure(figsize=(18, 6), dpi=100)
    
    # Rendered result
    plt.subplot(131)
    plt.imshow(np.clip(render, 0, 1))
    plt.title("Rendered")
    
    # Ground truth image
    plt.subplot(132)
    plt.imshow(gt)
    plt.title("Ground Truth")
    
    # Difference map
    plt.subplot(133)
    plt.imshow(diff, cmap='jet', vmin=0, vmax=0.3)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Difference Map")
    
    # Unified settings
    for ax in fig.axes:
        ax.axis('off')
        ax.axhline(y=H-exclude_rows-1, color='r', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig

def batch_calculate_ssim(npy_dir, png_dir, view, output_csv="ssim_results.csv", exclude_rows=30):
    """
    Batch calculate SSIM (fix empty output issue)
    """
    start_frame, end_frame = get_frame_range(npy_dir, png_dir, view)
    
    file_pairs = []
    for i in range(start_frame, end_frame+1):
        npy_name = f"new_frame{i:04d}.npy"
        png_name = f"{i:06d}_new_view_{view}.png"
        npy_path = os.path.join(npy_dir, npy_name)
        png_path = os.path.join(png_dir, png_name)
        
        if os.path.exists(npy_path) and os.path.exists(png_path):
            file_pairs.append((npy_path, png_path, i))
        else:
            print(f"File missing: {npy_name} or {png_name}")
            print(f"File missing: {npy_path} or {png_path}")

    results = []
    
    for npy_path, png_path, frame_id in tqdm(file_pairs, desc="Calculating SSIM"):
        result = {
            "frame_id": frame_id,
            "ssim": None,  # Initialize default value
            "npy_path": npy_path,
            "png_path": png_path,
            "error": None
        }
        try:
            # Data loading
            render = np.load(npy_path)
            img = Image.open(png_path).convert('RGB')
            img = img.resize((render.shape[1], render.shape[0]))
            gt = np.array(img)
            
            # Preprocessing
            H, W = render.shape[:2]
            render_crop = (np.clip(render[:H-exclude_rows], 0, 1) * 255).astype(np.uint8)
            gt_crop = gt[:H-exclude_rows]
            
            # Calculate SSIM
            result["ssim"] = ssim(gt_crop, render_crop, 
                                channel_axis=2, 
                                data_range=255)
            
        except Exception as e:
            result["error"] = str(e)
            print(f"Frame {frame_id} error: {str(e)}")
        
        results.append(result)

    # Ensure columns exist
    df = pd.DataFrame(results, columns=["frame_id", "ssim", "npy_path", "png_path", "error"])
    
    # Save results
    df.to_csv(output_csv, index=False)
    
    # Statistics
    valid_ssim = df[df['ssim'].notnull()]['ssim']
    print(f"\nValid frames: {len(valid_ssim)}")
    if not valid_ssim.empty:
        print(f"Average SSIM: {valid_ssim.mean():.4f}")
        print(f"Highest SSIM: {valid_ssim.max():.4f}")
        print(f"Lowest SSIM: {valid_ssim.min():.4f}")
    else:
        print("No valid SSIM data")
    
    return df

def get_frame_range(npy_dir, png_dir, view):
    """Dynamically get the valid frame range"""
    # Get npy file list
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    npy_frames = {int(f[9:13]) for f in npy_files if f.startswith('new_frame')}
    
    # Get png file list
    png_files = [f for f in os.listdir(png_dir) if f.endswith(f'_new_view_{view}.png')]
    png_frames = {int(f[:6]) for f in png_files}  # Assuming filename format is 000000_new_view_X.png
    
    # Take intersection and sort
    common_frames = sorted(npy_frames & png_frames)
    
    if not common_frames:
        raise ValueError(f"No matching frames in directories {npy_dir} and {png_dir}")
    
    return min(common_frames), max(common_frames)


def batch_calculate_psnr(npy_dir, png_dir, view, output_csv="psnr_results.csv", exclude_rows=30):
    """
    Batch calculate PSNR
    
    Args:
        npy_dir: .npy file directory (rendered results)
        png_dir: .png file directory (ground truth images)
        output_csv: Result save path
        exclude_rows: Number of bottom rows to exclude
    """
    start_frame, end_frame = get_frame_range(npy_dir, png_dir, view)
    
    file_pairs = []
    for i in range(start_frame, end_frame+1):
        npy_name = f"new_frame{i:04d}.npy"
        png_name = f"{i:06d}_new_view_{view}.png" # Note filename format difference
        npy_path = os.path.join(npy_dir, npy_name)
        png_path = os.path.join(png_dir, png_name)
        
        if os.path.exists(npy_path) and os.path.exists(png_path):
            file_pairs.append((npy_path, png_path, i))
        else:
            print(f"File missing: {npy_name} or {png_name}")
            print(f"File missing: {npy_path} or {png_path}")

    # Prepare result storage
    results = []
    
    # Batch processing
    for npy_path, png_path, frame_id in tqdm(file_pairs, desc="Calculating PSNR"):
        try:
            # Load data
            render = np.load(npy_path)  # [H, W, 3]
            img = Image.open(png_path).convert('RGB')
            img = img.resize((render.shape[1], render.shape[0]))
            gt = np.array(img).astype(np.float32) / 255.0
            
            # Crop bottom
            H, W = render.shape[:2]
            render_crop = render[:H-exclude_rows]
            gt_crop = gt[:H-exclude_rows]
            
            # Calculate PSNR
            psnr = compute_psnr(
                torch.from_numpy(render_crop).permute(2,0,1),
                torch.from_numpy(gt_crop).permute(2,0,1)
            )
            
            results.append({
                "frame_id": frame_id,
                "psnr": psnr,
                "npy_path": npy_path,
                "png_path": png_path
            })
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            results.append({
                "frame_id": frame_id,
                "psnr": None,
                "error": str(e)
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Print statistics
    valid_psnr = df[df['psnr'].notnull()]['psnr']
    print(f"\nProcessing complete! Valid frames: {len(valid_psnr)}")
    print(f"Average PSNR: {valid_psnr.mean():.2f} dB")
    print(f"Highest PSNR: {valid_psnr.max():.2f} dB")
    print(f"Lowest PSNR: {valid_psnr.min():.2f} dB")
    
    return df

def plot_comparison(npy_path, png_path, exclude_rows=0, save_path=None):
    """
    Visualize comparison (excluding specified bottom rows)
    """
    # Load data
    render = np.load(npy_path)
    gt = np.array(Image.open(png_path).convert('RGB'))
    
    # Resize and crop
    gt = np.array(Image.fromarray(gt).resize((render.shape[1], render.shape[0])))
    H, W = render.shape[0], render.shape[1]
    
    render_cropped = render[:H-exclude_rows, :, :]
    gt_cropped = gt[:H-exclude_rows, :, :] / 255.0
    
    # Create canvas
    plt.figure(figsize=(18, 6))
    
    # Rendered result (cropped)
    plt.subplot(131)
    plt.imshow(np.clip(render_cropped, 0, 1))
    plt.title(f"Rendered (Cropped)\nShape: {render_cropped.shape}")
    
    # Ground truth image (cropped)
    plt.subplot(132)
    plt.imshow(gt_cropped)
    plt.title(f"Ground Truth (Cropped)\nShape: {gt_cropped.shape}")
    
    # Difference map (cropped)
    diff = np.abs(render_cropped - gt_cropped)
    plt.subplot(133)
    im = plt.imshow(diff.mean(axis=-1), cmap='jet', vmin=0, vmax=0.3)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Difference (Cropped)\nMean: {diff.mean():.4f}")
    
    # Draw crop line
    for ax in plt.gcf().axes:
        ax.axhline(y=H-exclude_rows-1, color='r', linestyle='--', linewidth=2, alpha=0.7)
        ax.axis('off')
    
    # # Save or display
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     print(f"Comparison image saved to: {save_path}")
    plt.show()

def calculate_custom_psnr(npy_path, png_path):
    """
    Calculate PSNR between custom .npy file and .png image
    
    Args:
        npy_path: .npy file path (rendered output)
        png_path: .png file path (ground truth image)
    
    Returns:
        psnr_value: PSNR value (dB)
    """
    # Load .npy data
    render_data = np.load(npy_path)  # Shape should be [H, W, 3]
    
    # Process rendered data
    if render_data.dtype == np.float32:
        render_rgb = np.clip(render_data, 0.0, 1.0)  # Clamp to [0,1] range
    else:
        raise ValueError(f"Unsupported .npy data type: {render_data.dtype}")

    # Load and process PNG image
    png_img = Image.open(png_path)
    if png_img.mode == 'RGBA':
        png_img = png_img.convert('RGB')
    
    # Resize to match
    if png_img.size != (render_rgb.shape[1], render_rgb.shape[0]):
        png_img = png_img.resize((render_rgb.shape[1], render_rgb.shape[0]))
    
    png_array = np.array(png_img).astype(np.float32) / 255.0  # [H, W, 3]

    # Convert to Tensor
    tensor_render = torch.from_numpy(render_rgb).permute(2, 0, 1)  # [3, H, W]
    tensor_png = torch.from_numpy(png_array).permute(2, 0, 1)      # [3, H, W]

    # Calculate PSNR
    return compute_psnr(tensor_render, tensor_png)


def calculate_psnr(img1_path, npz_path, dataset_key='rgbs', target_index=0, exclude_rows=0, visualize=True):
    """
    Enhanced PSNR calculation function with visualization and cropping support
    
    Args:
        img1_path: PNG image path 
        npz_path: NPZ file path
        dataset_key: Key name for RGB data in NPZ file, default is 'rgbs'
        target_index: Image index to compare in NPZ data, default is 0
        exclude_rows: Number of bottom rows to exclude, default is 0
        visualize: Whether to generate visualization comparison, default is True
    
    Returns:
        psnr_value: PSNR value (dB)
    """
    # Load and preprocess data
    npz_data = np.load(npz_path)
    rgb_npz = npz_data[dataset_key]
    
    # Check index validity
    if target_index >= rgb_npz.shape[0]:
        raise IndexError(f"Index {target_index} out of range (total samples: {rgb_npz.shape[0]})")
    
    # Extract target image
    render = rgb_npz[target_index]  # [H, W, 3]
    if render.dtype == np.uint8:
        render = render.astype(np.float32) / 255.0
    else:
        render = np.clip(render, 0.0, 1.0)
    
    # Load and process PNG
    img_pil = Image.open(img1_path).convert('RGB')
    img_pil = img_pil.resize((render.shape[1], render.shape[0]))
    gt = np.array(img_pil).astype(np.float32) / 255.0
    
    # Crop bottom rows
    H, W = render.shape[:2]
    render_cropped = render[:H-exclude_rows]
    gt_cropped = gt[:H-exclude_rows]
    
    # Calculate PSNR
    tensor_render = torch.from_numpy(render_cropped).permute(2, 0, 1)
    tensor_gt = torch.from_numpy(gt_cropped).permute(2, 0, 1)
    psnr = compute_psnr(tensor_render, tensor_gt)
    
    # Visualization
    if visualize:
        plt.figure(figsize=(18, 6))
        
        # Rendered result
        plt.subplot(131)
        plt.imshow(render_cropped)
        plt.title(f"Rendered (Cropped)\nPSNR: {psnr:.2f}dB")
        
        # Ground truth image
        plt.subplot(132)
        plt.imshow(gt_cropped)
        plt.title("Ground Truth")
        
        # Difference map
        plt.subplot(133)
        diff = np.abs(render_cropped - gt_cropped).mean(axis=-1)
        plt.imshow(diff, cmap='jet', vmin=0, vmax=0.2)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"Difference Map\nMax: {diff.max():.2f}")
        
        for ax in plt.gcf().axes:
            ax.axhline(y=H-exclude_rows-1, color='r', linestyle='--', alpha=0.6)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return psnr

def calculate_masked_metrics(npy_dir, png_dir, view, mask_dir, output_csv="masked_metrics.csv", exclude_rows=0, visualize=False, vis_dir=None):
    """
    Calculate PSNR and SSIM for masked regions
    """
    # Add visualization directory check at the beginning of the function
    if visualize and not vis_dir:
        raise ValueError("vis_dir parameter must be specified when visualization is enabled")
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    start_frame, end_frame = get_frame_range(npy_dir, png_dir, view)
    
    file_triplets = []
    for i in range(start_frame, end_frame+1):
        npy_name = f"new_frame{i:04d}.npy"
        png_name = f"{i:06d}_new_view_{view}.png"
        mask_name = f"{i:06d}_nonrigid_{view}.png"
        
        npy_path = os.path.join(npy_dir, npy_name)
        png_path = os.path.join(png_dir, png_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        if all(os.path.exists(p) for p in [npy_path, png_path, mask_path]):
            file_triplets.append((npy_path, png_path, mask_path, i))
        else:
            print(f"File missing: {npy_name}, {png_name} or {mask_name}")
            print(f"File missing:\n - NPY path: {npy_path}\n - PNG path: {png_path}\n - Mask path: {mask_path}\n")

    results = []
    
    for npy_path, png_path, mask_path, frame_id in tqdm(file_triplets, desc="Processing progress"):
        result = {"frame_id": frame_id, "psnr": None, "ssim": None, "error": None}
        try:
            # Load data
            render = np.load(npy_path)  # [H, W, 3] float32
            gt = np.array(Image.open(png_path).convert('RGB')) / 255.0  # [H, W, 3] float32
            mask = np.array(Image.open(mask_path).convert('L'))  # [H, W] uint8
            
            # Unify dimensions
            H, W = render.shape[:2]
            gt = np.array(Image.fromarray((gt*255).astype(np.uint8)).resize((W, H))) / 255.0
            mask = np.array(Image.fromarray(mask).resize((W, H))) > 128  # Binarize
            
            # Apply mask
            masked_render = render * mask[..., None]  # [H, W, 3]
            masked_gt = gt * mask[..., None]
            
            # Crop bottom
            masked_render = masked_render[:H-exclude_rows]
            masked_gt = masked_gt[:H-exclude_rows]
            mask = mask[:H-exclude_rows]
            
            # Extract masked region pixels
            mask_flat = mask.flatten()
            render_pixels = masked_render.reshape(-1, 3)[mask_flat]
            gt_pixels = masked_gt.reshape(-1, 3)[mask_flat]
            
            # Convert to Tensor and calculate
            tensor_render = torch.from_numpy(render_pixels).permute(1,0)
            tensor_gt = torch.from_numpy(gt_pixels).permute(1,0)
            result["psnr"] = compute_psnr(tensor_render, tensor_gt)

            # Calculate SSIM (masked region only)
            # Convert data types
            render_uint8 = (np.clip(masked_render, 0, 1) * 255).astype(np.uint8)  # Add this line
            gt_uint8 = (masked_gt * 255).astype(np.uint8)  # Add this line
            
            # Create image block containing only masked region
            y, x = np.where(mask)
            if len(y) == 0 or len(x) == 0:  # Add empty mask check
                raise ValueError("Mask region is empty")
                
            min_y, max_y = np.min(y), np.max(y)
            min_x, max_x = np.min(x), np.max(x)
            
            # Extract ROI region
            render_roi = render_uint8[min_y:max_y+1, min_x:max_x+1]
            gt_roi = gt_uint8[min_y:max_y+1, min_x:max_x+1]
            mask_roi = mask[min_y:max_y+1, min_x:max_x+1]
            
            # Add dimension validation
            if render_roi.shape != gt_roi.shape:
                raise ValueError(f"ROI dimensions mismatch: render {render_roi.shape} vs gt {gt_roi.shape}")
            
            result["ssim"] = ssim(gt_roi, render_roi,
                                channel_axis=2,
                                data_range=255,
                                win_size=11,
                                use_sample_covariance=False,
                                mask=mask_roi)
            
            # Visualization section
            if visualize:
                fig = plt.figure(figsize=(18, 6), dpi=100)
                
                # Rendered result
                plt.subplot(131)
                plt.imshow(np.clip(masked_render, 0, 1))
                plt.title(f"Masked Render\nPSNR: {result['psnr']:.2f}dB")
                
                # Ground truth image
                plt.subplot(132)
                plt.imshow(masked_gt)
                plt.title("Masked Ground Truth")
                
                # Difference map
                plt.subplot(133)
                diff = np.abs(masked_render - masked_gt).mean(axis=-1)
                plt.imshow(diff, cmap='jet', vmin=0, vmax=0.3)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title(f"Difference Map\nSSIM: {result['ssim']:.4f}")
                
                # # Draw mask boundary
                # for ax in fig.axes:
                #     y, x = np.where(mask)
                #     if len(y) > 0 and len(x) > 0:
                #         rect = plt.Rectangle((x.min(), y.min()), 
                #                            x.max()-x.min(), 
                #                            y.max()-y.min(),
                #                            fill=False, 
                #                            edgecolor='lime', 
                #                            linewidth=2)
                #         ax.add_patch(rect)
                #     ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"frame_{frame_id:04d}_masked_comparison.png"), bbox_inches='tight')
                plt.close()


        except Exception as e:
            result["error"] = str(e)
            print(f"Frame {frame_id} error: {str(e)}")
        
        results.append(result)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Print statistics
    valid_psnr = df['psnr'].dropna()
    valid_ssim = df['ssim'].dropna()
    
    # Filter outliers (PSNR > 100dB or SSIM > 0.999 are considered anomalies)
    filtered_psnr = valid_psnr[valid_psnr < 100]
    filtered_ssim = valid_ssim[valid_ssim < 0.999]
    
    print("\nPSNR Statistics:")
    print(f"Total valid frames: {len(valid_psnr)}")
    print(f"Filtered frames: {len(filtered_psnr)} (excluded {len(valid_psnr)-len(filtered_psnr)} outliers)")
    if not filtered_psnr.empty:
        print(f"Average: {filtered_psnr.mean():.2f} dB")
        print(f"Range: [{filtered_psnr.min():.2f}, {filtered_psnr.max():.2f}]")
    
    print("\nSSIM Statistics:")
    print(f"Total valid frames: {len(valid_ssim)}")
    print(f"Filtered frames: {len(filtered_ssim)} (excluded {len(valid_ssim)-len(filtered_ssim)} outliers)")
    if not filtered_ssim.empty:
        print(f"Average: {filtered_ssim.mean():.4f}")
        print(f"Range: [{filtered_ssim.min():.4f}, {filtered_ssim.max():.4f}]")
    
    return df

def batch_calculate_masked_metrics(npy_base_dir, png_base_dir, mask_base_dir, output_dir="masked_metrics", exclude_rows=30, views=range(10)):
    """
    Batch calculate masked metrics for multiple views
    
    Args:
        npy_base_dir: Parent directory of raw_fixed_offset folder
        png_base_dir: Parent directory of new_view image directory
        mask_base_dir: Mask files root directory
        output_dir: Results output directory
        views: List of view numbers to process (default 0-9)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for view in views:
        print(f"\nProcessing masked view {view}...")
        # Build paths
        npy_dir = os.path.join(npy_base_dir, f"raw_fixed_offset_{view+1}")
        png_dir = os.path.join(png_base_dir, "new_view")
        mask_dir = os.path.join(mask_base_dir, "nonrigid")  # Adjust according to actual directory structure
        
        # Calculate masked metrics
        results = calculate_masked_metrics(
            npy_dir=npy_dir,
            png_dir=png_dir,
            mask_dir=mask_dir,
            output_csv=os.path.join(output_dir, f"masked_metrics_view_{view}.csv"),
            exclude_rows=exclude_rows,
            visualize=False,  # Recommended to disable visualization for batch processing
            view=view
        )

def create_masked_video(npy_dir, png_dir, mask_dir, output_path="masked_comparison.mp4", fps=30, exclude_rows=0):
    """
    Generate masked region comparison video
    Args:
        output_path: Output video path
        fps: Video frame rate
    """
    # Create temporary directory
    temp_dir = os.path.join(os.path.dirname(output_path), "masked_temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate all comparison images
    frame_paths = []
    for i in tqdm(range(150), desc="Generating masked comparison images"):
        npy_path = os.path.join(npy_dir, f"new_frame{i:04d}.npy")
        png_path = os.path.join(png_dir, f"{i:06d}_new_view_3.png")
        mask_path = os.path.join(mask_dir, f"{i:06d}_rigid_3.png")
        
        if not all(os.path.exists(p) for p in [npy_path, png_path, mask_path]):
            continue
            
        # Generate comparison image
        fig = plot_masked_comparison(npy_path, png_path, mask_path, exclude_rows)
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)
    
    # Generate video
    with imageio.get_writer(output_path, fps=fps) as writer:
        for path in tqdm(sorted(frame_paths), desc="Generating video"):
            image = imageio.imread(path)
            writer.append_data(image)
    
    # Clean up temporary files
    shutil.rmtree(temp_dir)
    print(f"\nVideo saved to: {output_path}")
    print(f"Temporary files cleaned: {temp_dir}")

def plot_masked_comparison(npy_path, png_path, mask_path, exclude_rows=0):
    """
    Generate single frame masked comparison image
    """
    # Load data
    render = np.load(npy_path)
    gt = np.array(Image.open(png_path).convert('RGB')) / 255.0
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # Preprocessing
    H, W = render.shape[:2]
    gt = np.array(Image.fromarray((gt*255).astype(np.uint8)).resize((W, H))) / 255.0
    mask = np.array(Image.fromarray(mask).resize((W, H))) > 128
    
    # Apply mask
    masked_render = render * mask[..., None]
    masked_gt = gt * mask[..., None]
    
    # Crop bottom
    masked_render = masked_render[:H-exclude_rows]
    masked_gt = masked_gt[:H-exclude_rows]
    mask = mask[:H-exclude_rows]
    
    # Create canvas
    fig = plt.figure(figsize=(18, 6), dpi=100)
    
    # 渲染结果
    plt.subplot(131)
    plt.imshow(np.clip(masked_render, 0, 1))
    plt.title("Masked Render")
    
    # 真实图像
    plt.subplot(132)
    plt.imshow(masked_gt)
    plt.title("Masked Ground Truth")
    
    # 差异图
    plt.subplot(133)
    diff = np.abs(masked_render - masked_gt).mean(axis=-1)
    plt.imshow(diff, cmap='jet', vmin=0, vmax=0.3)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Difference Map")
    
    # 绘制掩码边界
    # y, x = np.where(mask)
    # if len(y) > 0 and len(x) > 0:
    #     for ax in fig.axes:
    #         rect = plt.Rectangle((x.min(), y.min()), 
    #                            x.max()-x.min(), 
    #                            y.max()-y.min(),
    #                            fill=False, 
    #                            edgecolor='lime',
    #                            linewidth=1)
    #         ax.add_patch(rect)
    
    for ax in fig.axes:
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def batch_calculate_metrics(npy_base_dir, png_base_dir, output_dir="metrics", exclude_rows=30, views=range(10)):
    """
    Batch calculate PSNR and SSIM for multiple views
    
    Args:
        npy_base_dir: Parent directory of raw_fixed_offset folder
        png_base_dir: Parent directory of new_view image directory
        output_dir: Results output directory
        views: List of view numbers to process (default 0-9)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for view in views:
        print(f"\nProcessing view {view}...")
        npy_dir = os.path.join(npy_base_dir, f"raw_fixed_offset_{view+1}")
        png_dir = os.path.join(png_base_dir, f"new_view")
        
        # Calculate PSNR
        psnr_df = batch_calculate_psnr(
            npy_dir=npy_dir,
            png_dir=png_dir,
            output_csv=os.path.join(output_dir, f"psnr_view_{view}.csv"),
            exclude_rows=exclude_rows, 
            view=view
        )
        
        # Calculate SSIM
        ssim_df = batch_calculate_ssim(
            npy_dir=npy_dir,
            png_dir=png_dir,
            output_csv=os.path.join(output_dir, f"ssim_view_{view}.csv"),
            exclude_rows=exclude_rows , 
            view=view
        )



class ResultsReader:
    def __init__(self, file_path):
        """
        Initialize results reader, supports HDF5 and NPZ formats
        :param file_path: File path (.h5/.hdf5/.npz)
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        self.data = None
        self.file_type = None
        self._open_file()

    def _open_file(self):
        """Open file and identify format"""
        suffix = self.file_path.suffix.lower()
        
        if suffix in ['.h5', '.hdf5']:
            self.file_type = 'h5'
            self.data = h5py.File(self.file_path, 'r')
            print(f"Successfully loaded HDF5 file: {self.file_path.name}")
        elif suffix == '.npz':
            self.file_type = 'npz'
            self.data = np.load(self.file_path, allow_pickle=True)
            print(f"Successfully loaded NPZ file: {self.file_path.name}")
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        # Unified data access interface
        self.datasets = list(self.data.keys()) if self.file_type == 'h5' else list(self.data.files)
        print(f"Datasets contained: {self.datasets}")

    def get_dataset_info(self):
        """Get information about all datasets"""
        info = {}
        for name in self.datasets:
            if self.file_type == 'h5':
                dataset = self.data[name]
                info[name] = {
                    'shape': dataset.shape,
                    'dtype': dataset.dtype,
                    'size': dataset.size
                }
            else:  # npz
                arr = self.data[name]
                info[name] = {
                    'shape': arr.shape,
                    'dtype': arr.dtype,
                    'size': arr.size
                }
        return info

    def visualize_rgbs(self, max_images=5):
        """Visualize RGB image data"""
        if 'rgbs' not in self.datasets:
            print("Warning: No rgbs dataset in file")
            return

        rgbs = self.data['rgbs'][()] if self.file_type == 'h5' else self.data['rgbs']
        
        # Handle different storage formats
        if rgbs.ndim == 5:  # (N, T, H, W, C)
            rgbs = rgbs[:,0]  # Take first timestep
        
        print(f"Found {rgbs.shape[0]} RGB images")
        
        plt.figure(figsize=(15, 5))
        for i in range(min(max_images, rgbs.shape[0])):
            plt.subplot(1, min(max_images, rgbs.shape[0]), i+1)
            plt.imshow(rgbs[i])
            plt.title(f"Frame {i}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save_images(self, dataset_name, output_dir):
        """Save images from specified dataset to directory"""
        if dataset_name not in self.datasets:
            print(f"Error: Dataset {dataset_name} does not exist")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images = self.data[dataset_name][()] if self.file_type == 'h5' else self.data[dataset_name]
        
        # Handle different time dimensions
        if images.ndim == 5:  # (N, T, H, W, C)
            images = images[:,0]  # Take first timestep
        
        print(f"Saving {images.shape[0]} images to {output_path}...")
        
        for i in range(images.shape[0]):
            img = images[i]
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
            plt.imsave(output_path / f"{dataset_name}_{i:04d}.png", img)
        
        print("Save complete!")

    def get_metrics(self):
        """Get evaluation metrics"""
        metrics = {}
        for name in ['psnr', 'ssim', 'lpips']:
            if name in self.datasets:
                metrics[name] = self.data[name][()] if self.file_type == 'h5' else self.data[name].item()
        return metrics

    def close(self):
        """Close file"""
        if self.data:
            if self.file_type == 'h5':
                self.data.close()
            elif self.file_type == 'npz':
                self.data.close()
            print("File closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 使用示例
if __name__ == "__main__":
    import argparse

    root_path = '/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti'
    data_path = 'training_20250508_103023_DynamicObjectCrossing_1'
    npy_folder = f"/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/{data_path}/videos/novel_30000/raw_fixed_offset_1"
    png_folder = f"/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20/{data_path}/new_view"
    
    save_path = f'{root_path}/{data_path}/metrics'

    # batch_calculate_metrics(
    #     npy_base_dir="/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/training_20250508_103023_DynamicObjectCrossing_1/videos_eval/novel_30000",
    #     png_base_dir="/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20/training_20250508_103023_DynamicObjectCrossing_1",
    #     output_dir=f"{save_path}/all_views_metrics",
    #     views=range(10)  # 0-9
    # )

    # mask_dir = "/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20/training_20250505_160554_FollowLeadingVehicleWithObstacle_1/new_mask/rigid"
    # results = calculate_masked_metrics(
    #     npy_dir=npy_folder,
    #     png_dir=png_folder,
    #     mask_dir=mask_dir,
    #     output_csv=f"{save_path}/masked_metrics.csv",
    #     exclude_rows=30
    # )

    # batch_calculate_masked_metrics(
    #     npy_base_dir="/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/training_20250508_103023_DynamicObjectCrossing_1/videos_eval/novel_30000",
    #     png_base_dir="/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20/training_20250508_103023_DynamicObjectCrossing_1",
    #     mask_base_dir="/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data/custom_kitti/raw/2025_02_20/training_20250508_103023_DynamicObjectCrossing_1/new_mask",  # Mask files root directory
    #     output_dir=f"{save_path}/all_masked_metrics",
    #     views=range(5)
    # )

    # create_masked_video(
    #     npy_dir=npy_folder,
    #     png_dir=png_folder,
    #     mask_dir=mask_dir,
    #     output_path=f"{save_path}/masked_comparison.mp4",
    #     fps=15,
    #     exclude_rows=30
    # )

    create_comparison_video(
    npy_dir=npy_folder,
    png_dir=png_folder,
    view=1,
    output_path=f"{save_path}/comparison_video.mp4",
    fps=10,
    exclude_rows=0
    )
    
    png_path = '/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data_generator/data/training_20250424_113506_SignalizedJunctionLeftTurn_5/image/000019_camera_0.png'
    npz_path = '/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/training_20250424_113506_SignalizedJunctionLeftTurn_5/render_results/full_set_trimmed_20250429_135411.npz'

    psnr_value = calculate_psnr(png_path, npz_path,target_index=0,
    exclude_rows=0,
    visualize=True)
    print(f"PSNR: {psnr_value:.2f} dB")

    # 使用示例
    npy_file = "/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/output/Kitti/dataset=Kitti/training_20250424_113506_SignalizedJunctionLeftTurn_5/videos_eval/novel_30000/raw_rgb/new_frame0130.npy"
    png_file = "/home/yzhan536/thesis/src/Master-Theis-Volvo-3D-Reconstruction/data_generator/data/training_20250424_113506_SignalizedJunctionLeftTurn_5/new_view/000130_new_view_3.png"

    psnr = calculate_custom_psnr(npy_file, png_file)
    print(f"PSNR between render and GT: {psnr:.2f} dB")

    plot_comparison(npy_file, png_file)
    
    parser = argparse.ArgumentParser(description='HDF5 Results File Analysis Tool')
    parser.add_argument('file', help='HDF5 file path')
    parser.add_argument('--dataset', help='Specify dataset name to operate on')
    parser.add_argument('--save', help='Save images to specified directory')
    args = parser.parse_args()

    try:
        with ResultsReader(args.file) as reader:
            # Display basic information
            print("\nFile Information:")
            print("="*40)
            for name, info in reader.get_dataset_info().items():
                print(f"{name}:")
                print(f"  |- Shape: {info['shape']}")
                print(f"  |- Type: {info['dtype']}")
                print(f"  |- Size: {info['size']:,}")
            
            # Display evaluation metrics
            print("\nEvaluation Metrics:")
            print("="*40)
            metrics = reader.get_metrics()
            for k, v in metrics.items():
                print(f"{k.upper()}: {v:.4f}")

            # Visualize RGB images
            if args.dataset == 'rgbs':
                reader.visualize_rgbs()
                
            # Save images
            if args.save and args.dataset:
                reader.save_images(args.dataset, args.save)
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")