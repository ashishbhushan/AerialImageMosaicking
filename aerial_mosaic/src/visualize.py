"""All visualization and figure generation for the report."""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def _save(fig, name: str):
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# Figure 1 — sample input images
def fig1_sample_inputs(color_imgs: list, n: int = 4):
    n = min(n, len(color_imgs))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, color_imgs[:n]):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    fig.suptitle("Figure 1: Sample Input Drone Images", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig1_sample_inputs.png")


# Figure 2 — SIFT keypoints on 2-3 images
def fig2_keypoints(color_imgs: list, all_kps: list, n: int = 3):
    n = min(n, len(color_imgs))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, kps in zip(axes, color_imgs[:n], all_kps[:n]):
        vis = cv2.drawKeypoints(
            img, kps, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{len(kps)} keypoints", fontsize=11)
        ax.axis("off")
    fig.suptitle("Figure 2: SIFT Keypoints", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig2_keypoints.png")


# Figure 3 — raw feature matches between first pair (before ratio test)
def fig3_raw_matches(color_imgs: list, all_kps: list, all_descs: list):
    import cv2
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw = flann.knnMatch(all_descs[0], all_descs[1], k=2)
    all_matches = [m for m, _ in raw]

    vis = cv2.drawMatches(
        color_imgs[0], all_kps[0],
        color_imgs[1], all_kps[1],
        all_matches[:80], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Figure 3: Raw Feature Matches (first 80 of {len(all_matches)})", fontsize=13)
    ax.axis("off")
    _save(fig, "fig3_raw_matches.png")


# Figure 4 — inlier vs outlier matches after RANSAC
def fig4_inliers_outliers(color_imgs: list, all_kps: list, matches_list: list, masks: list):
    good = matches_list[0]
    mask = masks[0]
    if mask is None:
        print("  fig4: no mask for pair 0, skipping")
        return

    inliers  = [m for m, flag in zip(good, mask.ravel()) if flag]
    outliers = [m for m, flag in zip(good, mask.ravel()) if not flag]

    vis_in = cv2.drawMatches(
        color_imgs[0], all_kps[0],
        color_imgs[1], all_kps[1],
        inliers, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    vis_out = cv2.drawMatches(
        color_imgs[0], all_kps[0],
        color_imgs[1], all_kps[1],
        outliers, None,
        matchColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.imshow(cv2.cvtColor(vis_in,  cv2.COLOR_BGR2RGB)); ax1.set_title(f"Inliers ({len(inliers)})",  color="green"); ax1.axis("off")
    ax2.imshow(cv2.cvtColor(vis_out, cv2.COLOR_BGR2RGB)); ax2.set_title(f"Outliers ({len(outliers)})", color="red");   ax2.axis("off")
    fig.suptitle("Figure 4: RANSAC Inliers (green) vs Outliers (red)", fontsize=13)
    plt.tight_layout()
    _save(fig, "fig4_inliers_outliers.png")


# Figure 5 — individual warped images semi-transparent on canvas
def fig5_warped_overlay(warped_imgs: list, n: int = 6):
    n = min(n, len(warped_imgs))
    h, w = warped_imgs[0].shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.float64)
    count   = np.zeros((h, w),    dtype=np.float64)

    for warped in warped_imgs[:n]:
        mask = warped.any(axis=2).astype(np.float64)
        overlay += warped.astype(np.float64) * mask[:, :, np.newaxis]
        count   += mask

    valid = count > 0
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[valid] = np.clip(overlay[valid] / count[valid, np.newaxis], 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Figure 5: First {n} Warped Images Overlaid (semi-transparent)", fontsize=12)
    ax.axis("off")
    _save(fig, "fig5_warped_overlay.png")


# Figure 6 — naive mosaic (no blending)
def fig6_naive_mosaic(canvas: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title("Figure 6: Naive Mosaic (no blending)", fontsize=13)
    ax.axis("off")
    _save(fig, "fig6_naive_mosaic.png")


# Figure 7 — blended mosaic (hero image)
def fig7_blended_mosaic(blended: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    ax.set_title("Figure 7: Final Mosaic with Distance-Transform Blending", fontsize=13)
    ax.axis("off")
    _save(fig, "fig7_blended_mosaic.png")
    # Also save as full-res JPG for download
    cv2.imwrite(str(OUTPUTS / "final_mosaic.jpg"), blended)


# Figure 8 — side-by-side blending comparison
def fig8_comparison(canvas: np.ndarray, blended: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    ax1.imshow(cv2.cvtColor(canvas,  cv2.COLOR_BGR2RGB)); ax1.set_title("No blending",        fontsize=13); ax1.axis("off")
    ax2.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)); ax2.set_title("Distance-T blending", fontsize=13); ax2.axis("off")
    fig.suptitle("Figure 8: Blending Comparison", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig8_comparison.png")


def generate_all_figures(
    color_imgs, all_kps, all_descs, matches_list, masks, warped_imgs, canvas, blended
):
    print("Generating report figures...")
    fig1_sample_inputs(color_imgs)
    fig2_keypoints(color_imgs, all_kps)
    fig3_raw_matches(color_imgs, all_kps, all_descs)
    fig4_inliers_outliers(color_imgs, all_kps, matches_list, masks)
    fig5_warped_overlay(warped_imgs)
    fig6_naive_mosaic(canvas)
    fig7_blended_mosaic(blended)
    fig8_comparison(canvas, blended)
    print("All figures saved to outputs/")
