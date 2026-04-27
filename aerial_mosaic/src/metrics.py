"""Quality metrics for mosaic evaluation.

Provides SSIM, PSNR, and seam-gradient discontinuity score — all computed
purely with NumPy/OpenCV so no extra dependencies are required.
"""

import cv2
import numpy as np


# ── SSIM ──────────────────────────────────────────────────────────────────────

def _ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM for a single-channel float64 image pair."""
    K1, K2, L = 0.01, 0.03, 255.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a2  = mu_a * mu_a
    mu_b2  = mu_b * mu_b
    mu_ab  = mu_a * mu_b

    sig_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)

    ssim_map = np.where(den > 0, num / den, 0.0)
    return float(ssim_map.mean())


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean SSIM across all channels, optionally restricted to a mask region.

    img1, img2 : uint8 BGR images of the same shape
    mask       : uint8 binary mask (1 = pixel to include); None = whole image
    Returns    : SSIM in [-1, 1] (1 = identical)
    """
    if img1.shape != img2.shape:
        return float("nan")

    if mask is not None and mask.sum() < 100:
        return float("nan")

    if mask is not None:
        ys, xs = np.where(mask > 0)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        img1 = img1[y0:y1, x0:x1]
        img2 = img2[y0:y1, x0:x1]

    a = img1.astype(np.float64)
    b = img2.astype(np.float64)

    if a.ndim == 2:
        return _ssim_channel(a, b)

    scores = [_ssim_channel(a[:, :, c], b[:, :, c]) for c in range(a.shape[2])]
    return float(np.mean(scores))


# ── PSNR ──────────────────────────────────────────────────────────────────────

def compute_psnr(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Peak Signal-to-Noise Ratio in dB, optionally restricted to mask.

    Returns inf if images are identical, nan if mask is too small.
    """
    if img1.shape != img2.shape:
        return float("nan")

    a = img1.astype(np.float64)
    b = img2.astype(np.float64)

    if mask is not None:
        if mask.sum() < 10:
            return float("nan")
        a = a[mask > 0]
        b = b[mask > 0]

    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


# ── Seam gradient discontinuity ───────────────────────────────────────────────

def seam_gradient_score(warped_imgs: list, blended: np.ndarray) -> float:
    """Measure blending quality via gradient discontinuity at seam boundaries.

    At every pixel that sits on the boundary between two overlapping images,
    we compute the gradient magnitude in the blended result.  Smooth blending
    means low gradients at boundaries; hard seams produce large spikes.

    Returns the mean boundary gradient magnitude (lower = better blending).
    """
    h, w = blended.shape[:2]
    gray_blended = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Gradient magnitude of the blended mosaic
    gx = cv2.Sobel(gray_blended, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blended, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Build seam map: pixels covered by ≥ 2 images are "seam candidates"
    coverage = np.zeros((h, w), dtype=np.uint8)
    for img in warped_imgs:
        if img is None:
            continue
        mask = (img.sum(axis=2) > 0).astype(np.uint8)
        coverage += mask

    seam_mask = coverage >= 2

    if seam_mask.sum() == 0:
        return float("nan")

    return float(grad_mag[seam_mask].mean())


# ── Overlap SSIM (between pairs) ─────────────────────────────────────────────

def overlap_ssim_pairs(warped_imgs: list) -> list[float]:
    """SSIM in the overlap region for each consecutive image pair.

    Low SSIM in the overlap indicates poor alignment or large exposure difference.
    """
    scores = []
    for i in range(len(warped_imgs) - 1):
        a, b = warped_imgs[i], warped_imgs[i + 1]
        if a is None or b is None:
            scores.append(float("nan"))
            continue
        mask_a = (a.sum(axis=2) > 0).astype(np.uint8)
        mask_b = (b.sum(axis=2) > 0).astype(np.uint8)
        overlap = mask_a & mask_b
        scores.append(compute_ssim(a, b, mask=overlap))
    return scores


# ── Summary dict ─────────────────────────────────────────────────────────────

def compute_all_metrics(
    warped_imgs: list,
    canvas: np.ndarray,
    blended: np.ndarray,
) -> dict:
    """Compute the full set of quality metrics and return as a flat dict."""
    overlap_scores = overlap_ssim_pairs(warped_imgs)
    valid = [s for s in overlap_scores if not np.isnan(s)]
    mean_overlap_ssim = float(np.mean(valid)) if valid else float("nan")

    # Build combined coverage mask
    h, w = canvas.shape[:2]
    coverage = np.zeros((h, w), dtype=np.uint8)
    for img in warped_imgs:
        if img is not None:
            coverage |= (img.sum(axis=2) > 0).astype(np.uint8)

    ssim_blend_vs_naive = compute_ssim(canvas, blended, mask=coverage)
    psnr_blend_vs_naive = compute_psnr(canvas, blended, mask=coverage)
    seam_score          = seam_gradient_score(warped_imgs, blended)

    return {
        "mean_overlap_ssim":    mean_overlap_ssim,
        "ssim_blended_vs_naive": ssim_blend_vs_naive,
        "psnr_blended_vs_naive": psnr_blend_vs_naive,
        "seam_gradient_score":  seam_score,
        "overlap_ssim_pairs":   overlap_scores,
    }
