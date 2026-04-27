"""Custom RANSAC homography estimation and Direct Linear Transform (DLT) from scratch.

Implements the full classical pipeline without calling cv2.findHomography:
  1. Hartley normalization for numerical stability
  2. DLT to solve for H from a minimal 4-point sample
  3. Symmetric reprojection error as the inlier criterion
  4. Adaptive iteration budget based on observed inlier ratio
  5. Final refit on the full inlier set
"""

import numpy as np


# ── Normalization ──────────────────────────────────────────────────────────────

def _normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: translate centroid to origin, scale RMS dist to √2.

    Returns (normalized_pts, T) where T is the 3x3 normalization matrix.
    Denormalize after solving: H_orig = T2^{-1} @ H_norm @ T1
    """
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    rms = np.sqrt((centered ** 2).sum(axis=1).mean())
    scale = np.sqrt(2) / max(rms, 1e-8)

    T = np.array(
        [[scale, 0,     -scale * centroid[0]],
         [0,     scale, -scale * centroid[1]],
         [0,     0,      1                  ]],
        dtype=np.float64,
    )
    ones = np.ones((len(pts), 1), dtype=np.float64)
    pts_n = (T @ np.hstack([pts, ones]).T).T[:, :2]
    return pts_n, T


# ── DLT ───────────────────────────────────────────────────────────────────────

def dlt_homography(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Direct Linear Transform: estimate H such that p2 ≈ H @ p1 (homogeneous).

    Constructs the 2n×9 linear system A h = 0 from the cross-product constraint
    and solves via SVD.  Hartley normalization is applied before and reversed
    after to condition the system.

    p1, p2 : (N, 2) float64 arrays, N ≥ 4
    Returns : 3×3 homography matrix (H[2,2] = 1)
    """
    p1, p2 = np.asarray(p1, np.float64), np.asarray(p2, np.float64)
    p1_n, T1 = _normalize_points(p1)
    p2_n, T2 = _normalize_points(p2)

    n = len(p1_n)
    A = np.zeros((2 * n, 9), dtype=np.float64)

    for i in range(n):
        x,  y  = p1_n[i]
        xp, yp = p2_n[i]
        # Row 2i:   xp = H @ p1 in x
        # Row 2i+1: yp = H @ p1 in y
        A[2 * i]     = [-x, -y, -1,  0,  0,  0, xp * x, xp * y, xp]
        A[2 * i + 1] = [ 0,  0,  0, -x, -y, -1, yp * x, yp * y, yp]

    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T2) @ H_n @ T1
    if abs(H[2, 2]) > 1e-10:
        H /= H[2, 2]
    return H


# ── Reprojection error ─────────────────────────────────────────────────────────

def symmetric_reprojection_error(
    H: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    """Symmetric transfer error: (forward + backward) / 2, in pixels.

    Forward : project p1 through H and measure distance from p2.
    Backward: project p2 through H^{-1} and measure distance from p1.
    More robust than one-sided error because it penalises degenerate H.
    """
    n = len(p1)
    ones = np.ones((n, 1), dtype=np.float64)
    p1_h = np.hstack([p1, ones])
    p2_h = np.hstack([p2, ones])

    # Forward
    proj = (H @ p1_h.T).T
    w = np.clip(proj[:, 2:], 1e-8, None)
    err_fwd = np.linalg.norm(proj[:, :2] / w - p2, axis=1)

    # Backward
    try:
        H_inv = np.linalg.inv(H)
        proj_b = (H_inv @ p2_h.T).T
        wb = np.clip(proj_b[:, 2:], 1e-8, None)
        err_bwd = np.linalg.norm(proj_b[:, :2] / wb - p1, axis=1)
    except np.linalg.LinAlgError:
        err_bwd = err_fwd

    return (err_fwd + err_bwd) * 0.5


# ── RANSAC ────────────────────────────────────────────────────────────────────

def ransac_homography(
    p1: np.ndarray,
    p2: np.ndarray,
    thresh: float = 5.0,
    confidence: float = 0.99,
    max_iters: int = 2000,
    seed: int = 42,
) -> tuple:
    """RANSAC homography estimation implemented entirely from scratch.

    Finds H such that p2 ≈ H @ p1 using the minimal 4-point sample.

    Algorithm:
      1. Randomly sample 4 correspondences.
      2. Estimate H via DLT with Hartley normalization.
      3. Classify inliers: symmetric reprojection error < thresh.
      4. Update adaptive iteration budget: N = log(1-p)/log(1-e^4).
      5. After convergence, refit H on all inliers for maximum accuracy.

    Returns
    -------
    H    : 3×3 ndarray or None if estimation failed
    mask : uint8 ndarray of shape (n, 1), 1 = inlier, 0 = outlier
    """
    p1 = np.asarray(p1, np.float64)
    p2 = np.asarray(p2, np.float64)
    n = len(p1)

    if n < 4:
        return None, np.zeros((n, 1), dtype=np.uint8)

    rng        = np.random.default_rng(seed)
    best_H     = None
    best_mask  = np.zeros(n, dtype=bool)
    best_count = 0
    iters      = max_iters
    i          = 0

    while i < iters:
        idx = rng.choice(n, 4, replace=False)

        try:
            H_cand = dlt_homography(p1[idx], p2[idx])
        except Exception:
            i += 1
            continue

        if np.any(np.isnan(H_cand)) or np.any(np.isinf(H_cand)):
            i += 1
            continue

        errs   = symmetric_reprojection_error(H_cand, p1, p2)
        inlier = errs < thresh
        count  = int(inlier.sum())

        if count > best_count:
            best_count = count
            best_mask  = inlier
            best_H     = H_cand

            # Adaptive budget: how many iterations to guarantee confidence p
            ratio = count / n
            if 0 < ratio < 1:
                n_needed = int(
                    np.log(1.0 - confidence) / np.log(1.0 - ratio ** 4 + 1e-12)
                )
                iters = min(max(n_needed, i + 1), max_iters)

        i += 1

    # Refit on full inlier set for maximum accuracy
    if best_count >= 4:
        try:
            best_H    = dlt_homography(p1[best_mask], p2[best_mask])
            errs      = symmetric_reprojection_error(best_H, p1, p2)
            best_mask = errs < thresh
        except Exception:
            pass

    mask_out = best_mask.astype(np.uint8).reshape(-1, 1)
    return (best_H if best_count >= 4 else None), mask_out


# ── Drop-in replacement for homography._compute_H ────────────────────────────

def compute_H_custom(
    kps_src: list,
    kps_dst: list,
    matches: list,
    thresh: float = 5.0,
    min_inliers: int = 10,
) -> tuple:
    """Custom RANSAC wrapper matching the interface of homography._compute_H.

    Computes H mapping image_dst → image_src (same convention as OpenCV
    findHomography(dst_pts, src_pts)).

    Returns (H, mask) on success or (None, inlier_count_int) on failure —
    the same mixed return type as _compute_H so callers need no changes.
    """
    if len(matches) < 4:
        return None, 0

    src_pts = np.float64([kps_src[m.queryIdx].pt for m in matches])
    dst_pts = np.float64([kps_dst[m.trainIdx].pt for m in matches])

    # H maps dst → src  (same convention as cv2.findHomography(dst, src))
    H, mask = ransac_homography(dst_pts, src_pts, thresh=thresh)

    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < min_inliers:
        return None, inliers

    return H, mask
