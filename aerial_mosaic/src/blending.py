"""Stage 6: Seam blending using distance-transform-based weighted averaging."""

import cv2
import numpy as np


def blend_images(warped_imgs: list, canvas_shape: tuple) -> np.ndarray:
    """Distance-transform weighted blend of all warped images.

    For each pixel, each contributing image's weight is proportional to its
    distance-to-nearest-black-border (interior pixels get higher weight),
    so overlap zones fade smoothly rather than producing hard seams.
    """
    h, w = canvas_shape[:2]
    accum = np.zeros((h, w, 3), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)

    for warped in warped_imgs:
        if warped is None:
            continue

        mask = (warped.sum(axis=2) > 0).astype(np.uint8)
        if mask.sum() == 0:
            continue

        # Distance transform: each pixel = distance to nearest zero (border)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float64)

        accum += warped.astype(np.float64) * dist[:, :, np.newaxis]
        weight_sum += dist

    # Avoid divide-by-zero in empty regions
    valid = weight_sum > 0
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[valid] = np.clip(
        accum[valid] / weight_sum[valid, np.newaxis], 0, 255
    ).astype(np.uint8)

    return result
