"""Stage 6: Seam blending.

Two methods are provided:

  blend_images()     – original distance-transform weighted averaging (fast)
  blend_multiband()  – multi-band Laplacian pyramid blending (best quality)

Multi-band blending (Burt & Adelson, 1983):
  • Decomposes each image into a Laplacian pyramid (band-pass layers).
  • Decomposes the weight map into a Gaussian pyramid (low-pass layers).
  • Blends each pyramid level independently:
      L_blend[k] = Σ_i  L_image_i[k]  *  G_weight_i[k]
  • Reconstructs the final mosaic by collapsing the blended pyramid.

The key insight: coarse levels are blended over a wide spatial zone
(smooth colour transitions), while fine levels use sharp boundaries
(crisp texture detail).  This eliminates visible seams without blurring.
"""

import cv2
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _weight_maps(warped_imgs: list) -> tuple[list, np.ndarray]:
    """Distance-transform weight per image + total weight map."""
    weights = []
    h, w = warped_imgs[0].shape[:2]
    total = np.zeros((h, w), dtype=np.float64)

    for img in warped_imgs:
        if img is None or img.sum() == 0:
            weights.append(None)
            continue
        mask = (img.sum(axis=2) > 0).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float64)
        weights.append(dist)
        total += dist

    return weights, total


def _gauss_pyramid(img: np.ndarray, levels: int) -> list:
    pyr = [img.astype(np.float64)]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _lap_pyramid(gauss: list) -> list:
    lap = []
    for i in range(len(gauss) - 1):
        g = gauss[i]
        up = cv2.pyrUp(gauss[i + 1], dstsize=(g.shape[1], g.shape[0]))
        lap.append(g - up)
    lap.append(gauss[-1].copy())
    return lap


def _collapse(lap: list) -> np.ndarray:
    img = lap[-1].copy()
    for lvl in reversed(lap[:-1]):
        img = cv2.pyrUp(img, dstsize=(lvl.shape[1], lvl.shape[0]))
        img += lvl
    return img


# ── Method 1: distance-transform weighted averaging (original) ─────────────────

def blend_images(warped_imgs: list, canvas_shape: tuple) -> np.ndarray:
    """Distance-transform weighted blend of all warped images.

    For each pixel, each contributing image's weight is proportional to its
    distance to the nearest black border (interior pixels get higher weight),
    so overlap zones fade smoothly rather than producing hard seams.
    """
    h, w = canvas_shape[:2]
    accum      = np.zeros((h, w, 3), dtype=np.float64)
    weight_sum = np.zeros((h, w),    dtype=np.float64)

    for warped in warped_imgs:
        if warped is None:
            continue

        mask = (warped.sum(axis=2) > 0).astype(np.uint8)
        if mask.sum() == 0:
            continue

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float64)
        accum      += warped.astype(np.float64) * dist[:, :, np.newaxis]
        weight_sum += dist

    valid  = weight_sum > 0
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[valid] = np.clip(
        accum[valid] / weight_sum[valid, np.newaxis], 0, 255
    ).astype(np.uint8)

    return result


# ── Method 2: multi-band Laplacian pyramid blending ───────────────────────────

def blend_multiband(
    warped_imgs: list,
    canvas_shape: tuple,
    levels: int = 5,
) -> np.ndarray:
    """Multi-band Laplacian pyramid blending (Burt & Adelson 1983).

    Produces seamless mosaics by blending each spatial frequency band at the
    appropriate scale — wide transition zones for colour, sharp boundaries for
    texture detail.

    Parameters
    ----------
    warped_imgs  : list of BGR warped images on a shared canvas
    canvas_shape : (h, w[, 3]) shape of the canvas
    levels       : pyramid depth (5–6 works well; auto-capped to image size)
    """
    h, w = canvas_shape[:2]

    # Cap levels so the coarsest layer is at least 8×8
    max_levels = int(np.floor(np.log2(min(h, w)))) - 2
    levels = max(1, min(levels, max_levels))

    raw_weights, total_w = _weight_maps(warped_imgs)

    # Normalize weights so they sum to 1 at each pixel
    safe_total = np.where(total_w > 0, total_w, 1.0)
    norm_weights = [
        (wt / safe_total if wt is not None else None) for wt in raw_weights
    ]

    # Accumulate blended Laplacian pyramid (per channel)
    blended_lap = None  # list of 3 channel pyramids, each a list of level arrays

    for img, nw in zip(warped_imgs, norm_weights):
        if img is None or nw is None:
            continue

        # Per-channel Laplacian pyramid
        img_f = img.astype(np.float64)
        img_lap = []
        for c in range(3):
            g = _gauss_pyramid(img_f[:, :, c], levels)
            img_lap.append(_lap_pyramid(g))

        # Weight Gaussian pyramid
        w_gauss = _gauss_pyramid(nw, levels)

        if blended_lap is None:
            blended_lap = [
                [np.zeros_like(img_lap[c][k]) for k in range(levels)]
                for c in range(3)
            ]

        for c in range(3):
            for k in range(levels):
                lk = img_lap[c][k]
                wk = w_gauss[k]

                # Sizes may differ by 1px due to pyrDown rounding — fix with resize
                if lk.shape[:2] != blended_lap[c][k].shape[:2]:
                    lk = cv2.resize(lk, (blended_lap[c][k].shape[1],
                                         blended_lap[c][k].shape[0]))
                if wk.shape[:2] != blended_lap[c][k].shape[:2]:
                    wk = cv2.resize(wk, (blended_lap[c][k].shape[1],
                                         blended_lap[c][k].shape[0]))

                blended_lap[c][k] += lk * wk

    if blended_lap is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Collapse pyramid and assemble channels
    channels = [_collapse(blended_lap[c]) for c in range(3)]
    result = np.stack(channels, axis=2)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Zero out pixels with no image coverage
    final = np.zeros((h, w, 3), dtype=np.uint8)
    coverage = total_w > 0
    final[coverage] = result[coverage]

    return final
