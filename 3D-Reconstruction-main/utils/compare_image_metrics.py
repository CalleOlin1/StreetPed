import argparse

import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image(path):
    """Load image as RGB float32 array in [0, 1]."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def image_to_lpips_tensor(image):
    """
    Convert HWC RGB image in [0, 1] to LPIPS input:
    NCHW tensor in [-1, 1].
    """
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def main():
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, and LPIPS between two images."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to the target/reference image.",
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to the predicted image.",
    )
    parser.add_argument(
        "--net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone (default: alex).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, otherwise cpu).",
    )

    args = parser.parse_args()

    target = load_image(args.target)
    pred = load_image(args.pred)

    if target.shape != pred.shape:
        raise ValueError(
            f"Image dimensions do not match: "
            f"target={target.shape}, pred={pred.shape}"
        )

    # PSNR
    psnr = peak_signal_noise_ratio(
        target,
        pred,
        data_range=1.0,
    )

    # SSIM
    ssim = structural_similarity(
        target,
        pred,
        channel_axis=2,
        data_range=1.0,
    )

    # LPIPS
    loss_fn = lpips.LPIPS(net=args.net).to(args.device)
    loss_fn.eval()

    target_tensor = image_to_lpips_tensor(target).to(args.device)
    pred_tensor = image_to_lpips_tensor(pred).to(args.device)

    with torch.no_grad():
        lpips_value = loss_fn(target_tensor, pred_tensor).item()

    print(f"PSNR : {psnr:.6f} dB")
    print(f"SSIM : {ssim:.6f}")
    print(f"LPIPS: {lpips_value:.6f}")


if __name__ == "__main__":
    main()
