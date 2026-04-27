"""Exposure compensation: normalize per-image gains before blending.

Aerial images taken at different angles and times have inconsistent exposures
that produce a patchwork look in the final mosaic.  This module solves for a
per-image scalar gain g_i that minimises the sum of squared intensity
differences in every overlap region:

    min  Σ_{(i,j)} N_ij * (g_i * μ_ij^i  -  g_j * μ_ij^j)²

Setting ∂/∂g_i = 0 yields a symmetric linear system A g = 0 with the
constraint g_ref = 1 (reference image unchanged).  We solve it with
numpy.linalg.solve after anchoring the reference row.
"""

import cv2
import numpy as np


def _grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return img.astype(np.float64)


def compensate_exposure(
    warped_imgs: list,
    ref_idx: int | None = None,
    min_overlap_px: int = 200,
) -> tuple[list, np.ndarray]:
    """Compute and apply per-image gain corrections.

    Parameters
    ----------
    warped_imgs   : list of BGR warped images (black background = no coverage)
    ref_idx       : index of the reference image (gain fixed to 1.0)
    min_overlap_px: minimum overlap pixels required to use a pair as a constraint

    Returns
    -------
    adjusted_imgs : list of exposure-corrected images (same shape as inputs)
    gains         : float64 array of length N with the applied gain per image
    """
    n = len(warped_imgs)
    if ref_idx is None:
        ref_idx = n // 2

    grays = [_grayscale(img) for img in warped_imgs]
    masks = [(g > 0).astype(np.uint8) for g in grays]

    # Build the normal equations A g = b
    A = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            overlap = masks[i] & masks[j]
            N = int(overlap.sum())
            if N < min_overlap_px:
                continue

            mu_i = grays[i][overlap > 0].mean()
            mu_j = grays[j][overlap > 0].mean()
            if mu_i < 1e-4 or mu_j < 1e-4:
                continue

            # Contribution to equations for i and j
            A[i, i] += N * mu_i * mu_i
            A[i, j] -= N * mu_i * mu_j
            A[j, i] -= N * mu_j * mu_i
            A[j, j] += N * mu_j * mu_j

    # Anchor reference image: replace its row with the identity constraint g_ref = 1
    A[ref_idx, :] = 0.0
    A[ref_idx, ref_idx] = 1.0
    b = np.zeros(n, dtype=np.float64)
    b[ref_idx] = 1.0

    try:
        gains = np.linalg.solve(A, b)
        # Guard against extreme corrections (clip to ±1 stop = [0.5, 2.0])
        gains = np.clip(gains, 0.5, 2.0)
    except np.linalg.LinAlgError:
        print("  Exposure compensation: solver failed, using unit gains")
        gains = np.ones(n, dtype=np.float64)

    # Apply gains
    adjusted = []
    for img, g in zip(warped_imgs, gains):
        if img is None or img.sum() == 0:
            adjusted.append(img)
        else:
            adj = np.clip(img.astype(np.float64) * g, 0, 255).astype(np.uint8)
            # Preserve the zero (no-coverage) mask so warping borders stay black
            zero_mask = img.sum(axis=2) == 0
            adj[zero_mask] = 0
            adjusted.append(adj)

    print(
        f"  Exposure gains: min={gains.min():.3f}  max={gains.max():.3f}  "
        f"range={gains.max()-gains.min():.3f}"
    )
    return adjusted, gains
