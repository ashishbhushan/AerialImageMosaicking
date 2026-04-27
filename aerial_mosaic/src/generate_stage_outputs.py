"""
Generate visual outputs at every pipeline stage.
Run: python src/generate_stage_outputs.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from loader import load_images
from features import detect_and_describe
from matching import match_consecutive_pairs
from homography import estimate_homographies, chain_homographies
from warping import warp_images
from blending import blend_images

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs" / "stages"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DATA_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "odm_data_aukerman" / "images")
N_IMAGES = 6   # enough to see stitching without being slow
START    = 2


def save(fig, name):
    p = OUTPUTS / name
    fig.savefig(str(p), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ── STAGE 1 ──────────────────────────────────────────────────────────────────
print("\n[Stage 1] Loading images...")
color_imgs, gray_imgs = load_images(DATA_DIR, n=N_IMAGES, start=START)

# 1a: color images
fig, axes = plt.subplots(1, len(color_imgs), figsize=(4 * len(color_imgs), 4))
fig.suptitle("Stage 1a — Color Images (resized to 800px, BGR→RGB)", fontsize=13)
for ax, img in zip(axes, color_imgs):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
plt.tight_layout()
save(fig, "stage1a_color_images.png")

# 1b: grayscale + blurred (what SIFT actually sees)
fig, axes = plt.subplots(1, len(gray_imgs), figsize=(4 * len(gray_imgs), 4))
fig.suptitle("Stage 1b — Grayscale + Gaussian Blur (5×5) — input to SIFT", fontsize=13)
for ax, g in zip(axes, gray_imgs):
    ax.imshow(g, cmap="gray"); ax.axis("off")
plt.tight_layout()
save(fig, "stage1b_grayscale_blurred.png")

# 1c: side-by-side color vs gray for image 0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Stage 1c — Image 0: Color vs Grayscale+Blur", fontsize=13)
ax1.imshow(cv2.cvtColor(color_imgs[0], cv2.COLOR_BGR2RGB)); ax1.set_title("Color (BGR)"); ax1.axis("off")
ax2.imshow(gray_imgs[0], cmap="gray"); ax2.set_title("Grayscale + Gaussian Blur"); ax2.axis("off")
plt.tight_layout()
save(fig, "stage1c_color_vs_gray.png")


# ── STAGE 2 ──────────────────────────────────────────────────────────────────
print("\n[Stage 2] Detecting SIFT features...")
all_kps, all_descs = detect_and_describe(gray_imgs)

# 2a: keypoints with rich visualization (scale + orientation rings)
fig, axes = plt.subplots(1, len(color_imgs), figsize=(5 * len(color_imgs), 5))
fig.suptitle("Stage 2a — SIFT Keypoints (circle=scale, line=orientation)", fontsize=13)
for ax, img, kps in zip(axes, color_imgs, all_kps):
    vis = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{len(kps)} keypoints", fontsize=10); ax.axis("off")
plt.tight_layout()
save(fig, "stage2a_sift_keypoints_all.png")

# 2b: descriptor heatmap — show what the 128-d descriptor vector looks like
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Stage 2b — SIFT Descriptor vectors (first 20 keypoints, each is a 128-d vector)", fontsize=12)
for ax, descs, i in zip(axes, all_descs[:3], range(3)):
    ax.imshow(descs[:20], aspect="auto", cmap="hot")
    ax.set_xlabel("Descriptor dimension (0–127)"); ax.set_ylabel("Keypoint index")
    ax.set_title(f"Image {i}")
plt.tight_layout()
save(fig, "stage2b_descriptor_heatmap.png")


# ── STAGE 3 ──────────────────────────────────────────────────────────────────
print("\n[Stage 3] Matching features...")
matches_list = match_consecutive_pairs(all_descs)

# 3a: raw matches (before ratio test) for pair 0-1
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
raw_all = flann.knnMatch(all_descs[0], all_descs[1], k=2)
raw_matches = [m for m, _ in raw_all]
vis_raw = cv2.drawMatches(
    color_imgs[0], all_kps[0], color_imgs[1], all_kps[1],
    raw_matches[:60], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle(f"Stage 3a — RAW matches pair(0,1): {len(raw_matches)} total (first 60 shown)\nMany are WRONG — lines cross chaotically", fontsize=12)
ax.imshow(cv2.cvtColor(vis_raw, cv2.COLOR_BGR2RGB)); ax.axis("off")
save(fig, "stage3a_raw_matches_before_ratio.png")

# 3b: filtered matches (after Lowe's ratio test 0.75) for pair 0-1
good = matches_list[0]
vis_good = cv2.drawMatches(
    color_imgs[0], all_kps[0], color_imgs[1], all_kps[1],
    good[:60], None, matchColor=(0, 200, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle(f"Stage 3b — After Lowe's ratio test (threshold=0.75): {len(good)} kept (first 60 shown)\nLines are more consistent — fewer wrong matches", fontsize=12)
ax.imshow(cv2.cvtColor(vis_good, cv2.COLOR_BGR2RGB)); ax.axis("off")
save(fig, "stage3b_filtered_matches_after_ratio.png")

# 3c: before vs after count bar chart for all pairs
raw_counts, good_counts = [], []
for i in range(len(all_descs) - 1):
    raw_i = flann.knnMatch(all_descs[i], all_descs[i + 1], k=2)
    raw_counts.append(len(raw_i))
    good_counts.append(len(matches_list[i]))

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(raw_counts))
ax.bar(x - 0.2, raw_counts, 0.4, label="Raw (before ratio test)", color="salmon")
ax.bar(x + 0.2, good_counts, 0.4, label="After ratio test", color="steelblue")
ax.set_xlabel("Pair index"); ax.set_ylabel("Match count")
ax.set_title("Stage 3c — Match counts: Raw vs After Lowe's Ratio Test")
ax.set_xticks(x); ax.set_xticklabels([f"({i},{i+1})" for i in range(len(raw_counts))])
ax.legend(); plt.tight_layout()
save(fig, "stage3c_match_count_comparison.png")


# ── STAGE 4 ──────────────────────────────────────────────────────────────────
print("\n[Stage 4] Estimating homographies...")
H_pairs, masks = estimate_homographies(all_kps, matches_list)
H_chain = chain_homographies(H_pairs)

# 4a: inliers vs outliers for EACH pair
n_pairs = len(matches_list)
fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4 * n_pairs))
fig.suptitle("Stage 4a — RANSAC: Inliers (green) vs Outliers (red) for every pair", fontsize=13)
for row, (i, good_m, mask) in enumerate(zip(range(n_pairs), matches_list, masks)):
    if mask is None:
        for ax in axes[row]:
            ax.text(0.5, 0.5, f"Pair ({i},{i+1}): RANSAC failed", ha="center"); ax.axis("off")
        continue
    inliers  = [m for m, f in zip(good_m, mask.ravel()) if f]
    outliers = [m for m, f in zip(good_m, mask.ravel()) if not f]
    vis_in = cv2.drawMatches(color_imgs[i], all_kps[i], color_imgs[i+1], all_kps[i+1],
                              inliers, None, matchColor=(0,220,0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis_out = cv2.drawMatches(color_imgs[i], all_kps[i], color_imgs[i+1], all_kps[i+1],
                               outliers, None, matchColor=(0,0,220),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes[row][0].imshow(cv2.cvtColor(vis_in, cv2.COLOR_BGR2RGB))
    axes[row][0].set_title(f"Pair ({i},{i+1}) — Inliers: {len(inliers)}", color="green", fontsize=10)
    axes[row][0].axis("off")
    axes[row][1].imshow(cv2.cvtColor(vis_out, cv2.COLOR_BGR2RGB))
    axes[row][1].set_title(f"Pair ({i},{i+1}) — Outliers: {len(outliers)}", color="red", fontsize=10)
    axes[row][1].axis("off")
plt.tight_layout()
save(fig, "stage4a_ransac_all_pairs.png")

# 4b: homography matrices as text table
valid_pairs = [(i, H) for i, H in enumerate(H_pairs) if H is not None]
fig, axes = plt.subplots(1, len(valid_pairs), figsize=(5 * len(valid_pairs), 4))
if len(valid_pairs) == 1:
    axes = [axes]
fig.suptitle("Stage 4b — Homography Matrices H[i→i+1]  (3×3, row 2 = [0,0,1] means near-planar)", fontsize=11)
for ax, (i, H) in zip(axes, valid_pairs):
    ax.axis("off")
    rows = [f"{H[r,0]:+.4f}  {H[r,1]:+.4f}  {H[r,2]:+.5f}" for r in range(3)]
    text = "\n".join(rows)
    ax.text(0.5, 0.5, f"H({i}→{i+1}):\n\n{text}", ha="center", va="center",
            fontfamily="monospace", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title(f"Pair ({i},{i+1})", fontsize=10)
plt.tight_layout()
save(fig, "stage4b_homography_matrices.png")

# 4c: inlier ratio bar chart
inlier_ratios = []
for good_m, mask in zip(matches_list, masks):
    if mask is not None and len(good_m) > 0:
        inlier_ratios.append(mask.sum() / len(good_m))
    else:
        inlier_ratios.append(0.0)
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["green" if r > 0.5 else "orange" if r > 0.2 else "red" for r in inlier_ratios]
ax.bar([f"({i},{i+1})" for i in range(len(inlier_ratios))], inlier_ratios, color=colors)
ax.axhline(0.5, color="gray", linestyle="--", label="50% threshold")
ax.set_ylabel("Inlier ratio"); ax.set_title("Stage 4c — RANSAC Inlier Ratio per pair\n(green=good, orange=marginal, red=bad)")
ax.legend(); plt.tight_layout()
save(fig, "stage4c_inlier_ratios.png")


# ── STAGE 5 ──────────────────────────────────────────────────────────────────
print("\n[Stage 5] Warping images...")
canvas, warped_imgs = warp_images(color_imgs, H_chain)

# 5a: each warped image individually
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Stage 5a — Individual Warped Images on Shared Canvas\n(black = empty canvas space)", fontsize=13)
for ax, warped, i in zip(axes.flat, warped_imgs, range(len(warped_imgs))):
    ax.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Image {i} warped", fontsize=10); ax.axis("off")
for ax in axes.flat[len(warped_imgs):]:
    ax.axis("off")
plt.tight_layout()
save(fig, "stage5a_individual_warped_images.png")

# 5b: building up the canvas step-by-step (cumulative paste)
h, w = canvas.shape[:2]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Stage 5b — Canvas Built Up Step-by-Step (naive overwrite)", fontsize=13)
cumulative = np.zeros((h, w, 3), dtype=np.uint8)
for ax, warped, i in zip(axes.flat, warped_imgs, range(len(warped_imgs))):
    mask = warped.any(axis=2)
    cumulative[mask] = warped[mask]
    ax.imshow(cv2.cvtColor(cumulative.copy(), cv2.COLOR_BGR2RGB))
    ax.set_title(f"After pasting image {i}", fontsize=10); ax.axis("off")
for ax in axes.flat[len(warped_imgs):]:
    ax.axis("off")
plt.tight_layout()
save(fig, "stage5b_canvas_buildup.png")

# 5c: final naive canvas
fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
ax.set_title("Stage 5c — Final Naive Canvas (no blending)\nHard seams visible at image boundaries", fontsize=12)
ax.axis("off"); plt.tight_layout()
save(fig, "stage5c_naive_canvas.png")


# ── STAGE 6 ──────────────────────────────────────────────────────────────────
print("\n[Stage 6] Blending seams...")
blended = blend_images(warped_imgs, canvas.shape)

# 6a: distance transform weights for first 3 images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Stage 6a — Distance Transform Weight Maps\n(brighter = higher weight = pixel is far from image border → more reliable)", fontsize=12)
for ax, warped, i in zip(axes, warped_imgs[:3], range(3)):
    mask = (warped.sum(axis=2) > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = (dist / dist.max() * 255).astype(np.uint8) if dist.max() > 0 else dist.astype(np.uint8)
    ax.imshow(dist_norm, cmap="hot")
    ax.set_title(f"Image {i} weight map", fontsize=10); ax.axis("off")
plt.tight_layout()
save(fig, "stage6a_distance_transform_weights.png")

# 6b: overlap count map — shows where images pile up
overlap_count = np.zeros(canvas.shape[:2], dtype=np.int32)
for warped in warped_imgs:
    overlap_count += (warped.any(axis=2)).astype(np.int32)
fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(overlap_count, cmap="plasma")
plt.colorbar(im, ax=ax, label="Number of images covering this pixel")
ax.set_title("Stage 6b — Overlap Count Map\n(yellow/white = 2+ images overlap here → blending needed)", fontsize=12)
ax.axis("off"); plt.tight_layout()
save(fig, "stage6b_overlap_count_map.png")

# 6c: final blended result
fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
ax.set_title("Stage 6c — Final Blended Mosaic (distance-transform weighted average)", fontsize=12)
ax.axis("off"); plt.tight_layout()
save(fig, "stage6c_final_blended.png")

# 6d: zoom-in seam comparison (crop overlap region)
cy, cx = h // 2, w // 2
crop_h, crop_w = min(300, h // 3), min(400, w // 3)
y1, y2 = max(0, cy - crop_h), min(h, cy + crop_h)
x1, x2 = max(0, cx - crop_w), min(w, cx + crop_w)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Stage 6d — Zoomed Seam: Naive vs Blended (center crop)", fontsize=13)
ax1.imshow(cv2.cvtColor(canvas[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
ax1.set_title("Naive — hard seam visible"); ax1.axis("off")
ax2.imshow(cv2.cvtColor(blended[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
ax2.set_title("Blended — smooth transition"); ax2.axis("off")
plt.tight_layout()
save(fig, "stage6d_seam_zoom_comparison.png")

print(f"\nDone. All stage outputs saved to:\n  {OUTPUTS}")
