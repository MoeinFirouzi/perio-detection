from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os


def image_resize(
    source_path: str,
    dist_path: str,
    h_size: int,
    w_size: int,
):
    """
    Resizes all images in the source directory and saves them to the
    destination directory.

    Args:
        source_path (str): Path to the directory containing the source images.
        dist_path (str): Path to the directory to save resized images.
        h_size (int): Desired height of the resized images.
        w_size (int): Desired width of the resized images.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_names = []

    # Get all valid image files
    for dirpath, dirnames, filenames in os.walk(source_path, topdown=True):
        image_names.extend(
            [f for f in filenames if Path(f).suffix.lower() in
             valid_extensions]
        )

    # Create destination path if it doesn't exist
    Path(dist_path).mkdir(parents=True, exist_ok=True)

    for image in tqdm(image_names, desc="Resizing images"):
        try:
            image_file = Image.open(Path(source_path) / image)
            image_file = image_file.resize((w_size, h_size))
            # Save resized image
            image_file.save(Path(dist_path) / image)

        except Exception as e:
            print(f"Error processing {image}: {e}")
