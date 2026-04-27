"""Stage 1: Image loading, preprocessing, and GPS-based ordering.

Two loading modes:
  load_images()           – sorted by filename (original behaviour)
  load_images_gps_sorted() – sorted by GPS proximity (greedy nearest-neighbour)

GPS ordering is critical for real-world deployments where drone images may
arrive in arbitrary order from the file system.  We read GPS coordinates from
EXIF metadata using Pillow (falls back to filename sort if EXIF is absent).
"""

import cv2
import numpy as np
from pathlib import Path


# ── EXIF GPS extraction ────────────────────────────────────────────────────────

def _dms_to_decimal(dms, ref: str) -> float | None:
    """Convert degrees-minutes-seconds tuple to decimal degrees."""
    try:
        d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
        val = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            val = -val
        return val
    except Exception:
        return None


def _get_gps(path: Path) -> tuple[float, float] | None:
    """Return (lat, lon) decimal degrees from EXIF, or None if unavailable."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        with Image.open(path) as im:
            raw_exif = im._getexif()
        if raw_exif is None:
            return None

        gps_raw = None
        for tag_id, val in raw_exif.items():
            if TAGS.get(tag_id) == "GPSInfo":
                gps_raw = val
                break
        if gps_raw is None:
            return None

        gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
        lat = _dms_to_decimal(gps.get("GPSLatitude"),  gps.get("GPSLatitudeRef",  "N"))
        lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E"))
        if lat is None or lon is None:
            return None
        return lat, lon

    except Exception:
        return None


def _gps_sort(paths: list[Path]) -> list[Path]:
    """Greedy nearest-neighbour tour starting from the first image.

    Reads GPS from each image's EXIF.  Falls back to filename order if GPS is
    unavailable for the majority of images.
    """
    coords = {}
    for p in paths:
        gps = _get_gps(p)
        if gps is not None:
            coords[p] = gps

    if len(coords) < len(paths) // 2:
        print("  GPS ordering: insufficient EXIF data, using filename order")
        return paths

    print(f"  GPS ordering: {len(coords)}/{len(paths)} images have GPS")

    # Greedy nearest-neighbour from the first path that has GPS
    remaining = [p for p in paths if p in coords]
    if not remaining:
        return paths

    ordered = [remaining.pop(0)]
    while remaining:
        last_lat, last_lon = coords[ordered[-1]]
        # Find closest remaining image
        best, best_dist = None, float("inf")
        for p in remaining:
            lat, lon = coords[p]
            # Approximate Euclidean distance in degrees (good enough for short flights)
            dist = (lat - last_lat) ** 2 + (lon - last_lon) ** 2
            if dist < best_dist:
                best_dist, best = dist, p
        ordered.append(best)
        remaining.remove(best)

    # Append any paths without GPS at the end
    without_gps = [p for p in paths if p not in coords]
    ordered.extend(without_gps)
    return ordered


# ── Core loader ───────────────────────────────────────────────────────────────

def _resize_and_preprocess(img: np.ndarray, max_width: int) -> tuple:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_width:
        scale = max_width / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, gray


def load_images(
    images_dir: str,
    max_width: int = 800,
    n: int | None = None,
    start: int = 0,
    gps_sort: bool = False,
) -> tuple[list, list]:
    """Load, resize, and preprocess images from a directory.

    Parameters
    ----------
    images_dir : path to folder containing JPG images
    max_width  : resize so longest dimension ≤ this value
    n          : maximum number of images to load (None = all)
    start      : skip this many images from the front of the sorted list
    gps_sort   : if True, sort by GPS proximity instead of filename

    Returns
    -------
    color_imgs : list of BGR images (for output)
    gray_imgs  : list of grayscale + blurred images (for feature detection)
    """
    images_dir = Path(images_dir)
    paths = sorted(images_dir.glob("*.JPG")) + sorted(images_dir.glob("*.jpg"))

    if not paths:
        raise FileNotFoundError(f"No JPG images found in {images_dir}")

    paths = paths[start:]
    if n is not None:
        paths = paths[:n]

    if gps_sort:
        paths = _gps_sort(paths)

    color_imgs, gray_imgs = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: could not read {p}, skipping")
            continue
        img, gray = _resize_and_preprocess(img, max_width)
        color_imgs.append(img)
        gray_imgs.append(gray)

    sort_label = "GPS" if gps_sort else "filename"
    print(f"Loaded {len(color_imgs)} images from {images_dir} (sorted by {sort_label})")
    return color_imgs, gray_imgs


def load_images_gps_sorted(
    images_dir: str,
    max_width: int = 800,
    n: int | None = None,
    start: int = 0,
) -> tuple[list, list]:
    """Convenience wrapper: load_images with GPS sorting enabled."""
    return load_images(images_dir, max_width=max_width, n=n, start=start, gps_sort=True)
