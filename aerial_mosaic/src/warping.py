"""Stage 5: Image warping and canvas assembly."""

import cv2
import numpy as np


def compute_canvas_size(imgs: list, H_chain: list) -> tuple[int, int, np.ndarray]:
    """Project all image corners through their homographies to find the mosaic bounds.

    Returns:
        canvas_w, canvas_h: canvas dimensions in pixels
        offset:             3x3 translation matrix to shift all transforms so no
                            image falls off the left/top edge of the canvas
    """
    all_corners = []
    h, w = imgs[0].shape[:2]
    corners_local = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    for i, H in enumerate(H_chain):
        if H is None:
            continue
        hi, wi = imgs[i].shape[:2]
        c = np.float32([[0, 0], [wi, 0], [wi, hi], [0, hi]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(c, H)
        all_corners.append(warped)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = all_corners[:, 0, 0].min(), all_corners[:, 0, 1].min()
    x_max, y_max = all_corners[:, 0, 0].max(), all_corners[:, 0, 1].max()

    # Clamp to a sane maximum canvas size to guard against degenerate homographies
    MAX_DIM = 20000
    canvas_w = min(int(np.ceil(x_max - x_min)), MAX_DIM)
    canvas_h = min(int(np.ceil(y_max - y_min)), MAX_DIM)

    tx, ty = -x_min, -y_min
    offset = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    return canvas_w, canvas_h, offset


def warp_images(imgs: list, H_chain: list) -> tuple[np.ndarray, list]:
    """Warp all images onto a shared canvas.

    Returns:
        canvas:       BGR mosaic (naive overwrite blending)
        warped_imgs:  list of individually warped BGR images (same canvas size,
                      black background) — used for blending
    """
    canvas_w, canvas_h, offset = compute_canvas_size(imgs, H_chain)
    print(f"  Canvas size: {canvas_w} x {canvas_h}")

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    warped_imgs = []

    for i, (img, H) in enumerate(zip(imgs, H_chain)):
        if H is None:
            warped_imgs.append(np.zeros_like(canvas))
            continue

        H_shifted = offset @ H
        warped = cv2.warpPerspective(img, H_shifted, (canvas_w, canvas_h))
        warped_imgs.append(warped)

        # Naive paste: later images overwrite earlier ones
        mask = warped.any(axis=2)
        canvas[mask] = warped[mask]

    return canvas, warped_imgs
