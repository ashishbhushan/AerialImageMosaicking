"""Stage 1: Image loading and preprocessing."""

import cv2
from pathlib import Path


def load_images(images_dir: str, max_width: int = 800, n: int = None, start: int = 0) -> tuple[list, list]:
    """Load, resize, and preprocess images from a directory.

    Returns:
        color_imgs: list of BGR images (for final output)
        gray_imgs:  list of grayscale + blurred images (for feature detection)
    """
    images_dir = Path(images_dir)
    paths = sorted(images_dir.glob("*.JPG")) + sorted(images_dir.glob("*.jpg"))

    if not paths:
        raise FileNotFoundError(f"No JPG images found in {images_dir}")

    paths = paths[start:]
    if n is not None:
        paths = paths[:n]

    color_imgs, gray_imgs = [], []

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: could not read {p}, skipping")
            continue

        # Resize so longest side == max_width (preserves aspect ratio)
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_width:
            scale = max_width / longest
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        color_imgs.append(img)
        gray_imgs.append(gray)

    print(f"Loaded {len(color_imgs)} images from {images_dir}")
    return color_imgs, gray_imgs
