import subprocess
import io
from PIL import Image
import numpy as np
import threading

def print_stderr(pipe):
    for line in iter(pipe.readline, b''):
        print("Worker:", line.decode().rstrip())

def wait_for_ready(proc):
    # Used to ensure other process has succesfully started before calling it
    line = proc.stdout.readline().decode().strip()

    if line != "READY":
        stderr = proc.stderr.read().decode()
        raise RuntimeError(
            f"Difix worker failed to start.\n"
            f"Received: {line}\n"
            f"stderr:\n{stderr}"
        )

def pad_to_multiple_of_8(img):
    w, h = img.size
    new_w = (w + 7) // 8 * 8
    new_h = (h + 7) // 8 * 8
    pad_w = new_w - w
    pad_h = new_h - h

    img_np = np.array(img)

    pad_config = [(0, pad_h), (0, pad_w)]
    if img_np.ndim == 3:
        pad_config.append((0, 0))

    padded = np.pad(
        img_np,
        pad_config,
        mode="reflect"
    )

    return Image.fromarray(padded), w, h

def difix_repair_batch(input_images, reference_images, **kwargs):
    """
    Sends lists of images to difix_diffuse_list.py running in another Conda environment
    and returns a list of repaired images.

    Args:
        input_images: list of PIL.Image.Image to repair
        reference_images: list of PIL.Image.Image as references
        kwargs: Optional overrides for difix_env, script_path, working_dir
    Returns:
        List of PIL.Image.Image repaired images
    """
    import sys
    difix_env = kwargs.get("difix_env", "difix")
    script_path = kwargs.get("script_path", "/home/hstromgr/Documents/Github/Difix3D/src/difix_diffuse_list.py")
    working_dir = kwargs.get("working_dir", "/home/hstromgr/Documents/Github/Difix3D/src")

    def serialize_image(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def send_image_list(pipe, images):
        # Write 4 bytes for the number of images
        pipe.write(len(images).to_bytes(4, "big"))
        for img in images:
            img_bytes = serialize_image(img)
            pipe.write(len(img_bytes).to_bytes(4, "big"))
            pipe.write(img_bytes)
        pipe.flush()

    def read_exact(pipe, n):
        buf = b""
        while len(buf) < n:
            chunk = pipe.read(n - len(buf))
            if not chunk:
                raise EOFError("Unexpected EOF while reading from difix process")
            buf += chunk
        return buf

    def read_image(pipe):
        length_bytes = read_exact(pipe, 4)
        length = int.from_bytes(length_bytes, "big")
        img_data = read_exact(pipe, length)
        img = Image.open(io.BytesIO(img_data))
        img.load()
        return img

    def read_image_list(pipe):
        num_bytes = read_exact(pipe, 4)
        num_images = int.from_bytes(num_bytes, "big")
        images = []
        for _ in range(num_images):
            images.append(read_image(pipe))
        return images

    # Pad all images to multiple of 8
    padded_inputs = []
    orig_sizes = []
    for img in input_images:
        padded, w, h = pad_to_multiple_of_8(img)
        padded_inputs.append(padded)
        orig_sizes.append((w, h))
    padded_refs = []
    for img in reference_images:
        padded, _, _ = pad_to_multiple_of_8(img)
        padded_refs.append(padded)

    # Launch difix_diffuse_list.py in the other Conda environment
    proc = subprocess.Popen(
        ["conda", "run", "--no-capture-output", "-n", difix_env, "python", "-u", script_path],
        cwd=working_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for READY handshake
    line = proc.stdout.readline().decode().strip()
    if line != "READY":
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"Difix worker failed to start.\nReceived: {line}\nstderr:\n{stderr}")

    threading.Thread(target=print_stderr, args=(proc.stderr,), daemon=True).start()

    # Send reference images and input images
    send_image_list(proc.stdin, padded_refs)
    send_image_list(proc.stdin, padded_inputs)

    # Read repaired images
    repaired_images = read_image_list(proc.stdout)

    # Close and wait for process
    proc.stdin.close()
    proc.wait()

    # Crop repaired images back to original size
    cropped = [img.crop((0, 0, w, h)) for img, (w, h) in zip(repaired_images, orig_sizes)]
    return cropped

def difix_repair(
    input_image: Image.Image,
    reference_image: Image.Image,
    difix_env="difix",
    script_path="/home/hstromgr/Documents/Github/Difix3D/src/difix_diffuse.py",
    working_dir="/home/hstromgr/Documents/Github/Difix3D/src"
) -> Image.Image:
    """
    Sends two images to difix_diffuse.py running in another Conda environment
    and returns the repaired image.

    NOTE: Other users must change script_path and working_dir to match their setup.
    """

    def serialize_image(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def send_image(pipe, img_bytes: bytes):
        if pipe.closed:
            raise RuntimeError("difix subprocess stdin closed")
        pipe.write(len(img_bytes).to_bytes(4, "big"))
        pipe.write(img_bytes)
        pipe.flush()

    def read_exact(pipe, n):
        buf = b""
        while len(buf) < n:
            chunk = pipe.read(n - len(buf))
            if not chunk:
                raise EOFError("Unexpected EOF while reading from difix process")
            buf += chunk
        return buf

    def read_image(pipe) -> Image.Image:
        length_bytes = read_exact(pipe, 4)
        length = int.from_bytes(length_bytes, "big")
        img_data = read_exact(pipe, length)
        img = Image.open(io.BytesIO(img_data))
        img.load()
        return img


    # Pad images
    input_image, target_width, target_height = pad_to_multiple_of_8(input_image)
    reference_image, _, _ = pad_to_multiple_of_8(reference_image)

    input_bytes = serialize_image(input_image)
    ref_bytes = serialize_image(reference_image)

    # Launch difix_diffuse.py in the other Conda environment
    proc = subprocess.Popen(
        ["conda", "run", "--no-capture-output", "-n", difix_env, "python", "-u", script_path],
        cwd=working_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    wait_for_ready(proc)

    threading.Thread(target=print_stderr, args=(proc.stderr,), daemon=True).start()

    # Send images
    send_image(proc.stdin, input_bytes)
    send_image(proc.stdin, ref_bytes)

    # Read processed image
    output_image = read_image(proc.stdout)
    print(f"Received repaired image of size: {output_image.size}")

    # Close and wait for process
    proc.stdin.close()
    proc.wait()

    # Print stderr after process exits
    stderr_output = proc.stderr.read().decode()
    if stderr_output.strip():
        print("Difix stderr:", stderr_output)

    # Crop back to original size
    output_image = output_image.crop((0, 0, target_width, target_height))

    # Verify output image size matches input
    print(f"Cropped image to original size: {output_image.size}")
    return output_image