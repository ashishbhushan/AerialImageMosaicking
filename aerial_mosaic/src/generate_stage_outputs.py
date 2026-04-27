"""
Generate visual outputs at every pipeline stage for all 3 datasets.
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
from homography import estimate_homographies, chain_homographies, build_chain_with_fallback
from warping import warp_images
from blending import blend_images

ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS = {
    "aukerman": {
        "dir":          str(ROOT / "data" / "odm_data_aukerman" / "images"),
        "start":        2,    # skip first 2 (missing frame gap)
        "n":            6,
        "max_skip":     1,
        "ransac_thresh": 5.0,
        "min_inliers":  10,
        "label":        "Aukerman (fields)",
    },
    "bellus": {
        "dir":          str(ROOT / "data" / "odm_data_bellus" / "images"),
        "start":        12,   # clean strip — matches main.py config
        "n":            8,
        "max_skip":     1,
        "ransac_thresh": 8.0, # looser threshold for repetitive tree texture
        "min_inliers":  6,
        "label":        "Bellus (urban)",
    },
    "pacifica": {
        "dir":          str(ROOT / "data" / "odm_data_pacifica" / "images"),
        "start":        0,
        "n":            12,
        "max_skip":     11,
        "ransac_thresh": 5.0,
        "min_inliers":  10,
        "label":        "Pacifica (coastal)",
    },
}


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(str(p), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {p.name}")


def run_dataset(name, cfg):
    label    = cfg["label"]
    out_dir  = Path(__file__).resolve().parent.parent / "outputs" / "stages" / name
    n        = cfg["n"]
    use_fallback = cfg["max_skip"] > 1

    print(f"\n{'='*60}")
    print(f"  Dataset: {label}  ({n} images, start={cfg['start']})")
    print(f"{'='*60}")

    # ── Stage 1 ──────────────────────────────────────────────
    print("  [Stage 1] Loading...")
    color_imgs, gray_imgs = load_images(cfg["dir"], n=n, start=cfg["start"])

    fig, axes = plt.subplots(1, len(color_imgs), figsize=(4 * len(color_imgs), 4))
    fig.suptitle(f"[{label}] Stage 1a — Color Images (resized to 800px)", fontsize=12)
    for ax, img in zip(axes, color_imgs):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
    plt.tight_layout(); save(fig, out_dir, "stage1a_color_images.png")

    fig, axes = plt.subplots(1, len(gray_imgs), figsize=(4 * len(gray_imgs), 4))
    fig.suptitle(f"[{label}] Stage 1b — Grayscale + Gaussian Blur (input to SIFT)", fontsize=12)
    for ax, g in zip(axes, gray_imgs):
        ax.imshow(g, cmap="gray"); ax.axis("off")
    plt.tight_layout(); save(fig, out_dir, "stage1b_grayscale_blurred.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"[{label}] Stage 1c — Image 0: Color vs Grayscale+Blur", fontsize=12)
    ax1.imshow(cv2.cvtColor(color_imgs[0], cv2.COLOR_BGR2RGB)); ax1.set_title("Color"); ax1.axis("off")
    ax2.imshow(gray_imgs[0], cmap="gray"); ax2.set_title("Grayscale + Gaussian Blur"); ax2.axis("off")
    plt.tight_layout(); save(fig, out_dir, "stage1c_color_vs_gray.png")

    # ── Stage 2 ──────────────────────────────────────────────
    print("  [Stage 2] SIFT features...")
    all_kps, all_descs = detect_and_describe(gray_imgs)

    fig, axes = plt.subplots(1, len(color_imgs), figsize=(5 * len(color_imgs), 5))
    fig.suptitle(f"[{label}] Stage 2a — SIFT Keypoints (circle=scale, line=orientation)", fontsize=12)
    for ax, img, kps in zip(axes, color_imgs, all_kps):
        vis = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{len(kps)} kps", fontsize=9); ax.axis("off")
    plt.tight_layout(); save(fig, out_dir, "stage2a_sift_keypoints_all.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"[{label}] Stage 2b — SIFT Descriptor vectors (first 20 kps, 128-d each)", fontsize=11)
    for ax, descs, i in zip(axes, all_descs[:3], range(3)):
        ax.imshow(descs[:20], aspect="auto", cmap="hot")
        ax.set_xlabel("Dimension (0–127)"); ax.set_ylabel("Keypoint"); ax.set_title(f"Image {i}")
    plt.tight_layout(); save(fig, out_dir, "stage2b_descriptor_heatmap.png")

    # ── Stage 3 ──────────────────────────────────────────────
    print("  [Stage 3] Matching...")
    matches_list = match_consecutive_pairs(all_descs)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw_all = flann.knnMatch(all_descs[0], all_descs[1], k=2)
    raw_matches = [m for m, _ in raw_all]
    vis_raw = cv2.drawMatches(color_imgs[0], all_kps[0], color_imgs[1], all_kps[1],
                               raw_matches[:60], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"[{label}] Stage 3a — RAW matches pair(0,1): {len(raw_matches)} total (first 60)\nMany WRONG — lines cross chaotically", fontsize=11)
    ax.imshow(cv2.cvtColor(vis_raw, cv2.COLOR_BGR2RGB)); ax.axis("off")
    save(fig, out_dir, "stage3a_raw_matches_before_ratio.png")

    good = matches_list[0]
    vis_good = cv2.drawMatches(color_imgs[0], all_kps[0], color_imgs[1], all_kps[1],
                                good[:60], None, matchColor=(0, 200, 0),
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"[{label}] Stage 3b — After Lowe's ratio test (0.75): {len(good)} kept (first 60)\nLines more consistent — fewer wrong matches", fontsize=11)
    ax.imshow(cv2.cvtColor(vis_good, cv2.COLOR_BGR2RGB)); ax.axis("off")
    save(fig, out_dir, "stage3b_filtered_matches_after_ratio.png")

    raw_counts, good_counts = [], []
    for i in range(len(all_descs) - 1):
        r = flann.knnMatch(all_descs[i], all_descs[i + 1], k=2)
        raw_counts.append(len(r))
        good_counts.append(len(matches_list[i]))
    fig, ax = plt.subplots(figsize=(max(8, len(raw_counts) * 1.5), 4))
    x = np.arange(len(raw_counts))
    ax.bar(x - 0.2, raw_counts,  0.4, label="Raw (before ratio test)", color="salmon")
    ax.bar(x + 0.2, good_counts, 0.4, label="After ratio test",        color="steelblue")
    ax.set_xlabel("Pair"); ax.set_ylabel("Match count")
    ax.set_title(f"[{label}] Stage 3c — Match counts: Raw vs After Ratio Test")
    ax.set_xticks(x); ax.set_xticklabels([f"({i},{i+1})" for i in range(len(raw_counts))])
    ax.legend(); plt.tight_layout(); save(fig, out_dir, "stage3c_match_count_comparison.png")

    # ── Stage 4 ──────────────────────────────────────────────
    print("  [Stage 4] Homography...")
    ransac_thresh = cfg["ransac_thresh"]
    min_inliers   = cfg["min_inliers"]
    if use_fallback:
        H_chain, matches_list, masks = build_chain_with_fallback(
            all_kps, all_descs, max_skip=cfg["max_skip"],
            ransac_thresh=ransac_thresh, min_inliers=min_inliers)
        H_pairs = [None] * (len(color_imgs) - 1)
    else:
        H_pairs, masks = estimate_homographies(all_kps, matches_list, ransac_thresh, min_inliers)
        H_chain = chain_homographies(H_pairs, ref_idx=len(color_imgs) // 2)

    n_pairs = len(matches_list)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4 * n_pairs))
    if n_pairs == 1:
        axes = [axes]
    fig.suptitle(f"[{label}] Stage 4a — RANSAC: Inliers (green) vs Outliers (red)", fontsize=12)
    for row, (i, good_m, mask) in enumerate(zip(range(n_pairs), matches_list, masks)):
        if mask is None:
            for ax in axes[row]:
                ax.text(0.5, 0.5, f"Pair ({i},{i+1}): RANSAC failed", ha="center"); ax.axis("off")
            continue
        inliers  = [m for m, f in zip(good_m, mask.ravel()) if f]
        outliers = [m for m, f in zip(good_m, mask.ravel()) if not f]
        vis_in  = cv2.drawMatches(color_imgs[i], all_kps[i], color_imgs[i+1], all_kps[i+1],
                                   inliers, None, matchColor=(0,220,0),
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        vis_out = cv2.drawMatches(color_imgs[i], all_kps[i], color_imgs[i+1], all_kps[i+1],
                                   outliers, None, matchColor=(0,0,220),
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        axes[row][0].imshow(cv2.cvtColor(vis_in,  cv2.COLOR_BGR2RGB))
        axes[row][0].set_title(f"Pair ({i},{i+1}) Inliers: {len(inliers)}", color="green", fontsize=9); axes[row][0].axis("off")
        axes[row][1].imshow(cv2.cvtColor(vis_out, cv2.COLOR_BGR2RGB))
        axes[row][1].set_title(f"Pair ({i},{i+1}) Outliers: {len(outliers)}", color="red",   fontsize=9); axes[row][1].axis("off")
    plt.tight_layout(); save(fig, out_dir, "stage4a_ransac_all_pairs.png")

    inlier_ratios = [
        (mask.sum() / max(len(m), 1)) if mask is not None else 0.0
        for m, mask in zip(matches_list, masks)
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(inlier_ratios) * 1.5), 4))
    colors = ["green" if r > 0.5 else "orange" if r > 0.2 else "red" for r in inlier_ratios]
    ax.bar([f"({i},{i+1})" for i in range(len(inlier_ratios))], inlier_ratios, color=colors)
    ax.axhline(0.5, color="gray", linestyle="--", label="50% line")
    ax.set_ylabel("Inlier ratio"); ax.set_ylim(0, 1)
    ax.set_title(f"[{label}] Stage 4b — RANSAC Inlier Ratio per pair")
    ax.legend(); plt.tight_layout(); save(fig, out_dir, "stage4b_inlier_ratios.png")

    # ── Stage 5 ──────────────────────────────────────────────
    print("  [Stage 5] Warping...")
    canvas, warped_imgs = warp_images(color_imgs, H_chain)
    h, w = canvas.shape[:2]

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(f"[{label}] Stage 5a — Individual Warped Images on Shared Canvas", fontsize=12)
    for ax, warped, i in zip(axes.flat, warped_imgs, range(len(warped_imgs))):
        ax.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Image {i}", fontsize=9); ax.axis("off")
    for ax in axes.flat[len(warped_imgs):]:
        ax.axis("off")
    plt.tight_layout(); save(fig, out_dir, "stage5a_individual_warped.png")

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(f"[{label}] Stage 5b — Canvas Built Up Step-by-Step (naive overwrite)", fontsize=12)
    cumulative = np.zeros((h, w, 3), dtype=np.uint8)
    for ax, warped, i in zip(axes.flat, warped_imgs, range(len(warped_imgs))):
        mask = warped.any(axis=2)
        cumulative[mask] = warped[mask]
        ax.imshow(cv2.cvtColor(cumulative.copy(), cv2.COLOR_BGR2RGB))
        ax.set_title(f"After image {i}", fontsize=9); ax.axis("off")
    for ax in axes.flat[len(warped_imgs):]:
        ax.axis("off")
    plt.tight_layout(); save(fig, out_dir, "stage5b_canvas_buildup.png")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(f"[{label}] Stage 5c — Naive Canvas (no blending) — hard seams visible", fontsize=11)
    ax.axis("off"); plt.tight_layout(); save(fig, out_dir, "stage5c_naive_canvas.png")

    # ── Stage 6 ──────────────────────────────────────────────
    print("  [Stage 6] Blending...")
    blended = blend_images(warped_imgs, canvas.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"[{label}] Stage 6a — Distance Transform Weight Maps\n(bright=high weight=far from border=more reliable)", fontsize=11)
    for ax, warped, i in zip(axes, warped_imgs[:3], range(3)):
        mask = (warped.sum(axis=2) > 0).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dn = (dist / dist.max() * 255).astype(np.uint8) if dist.max() > 0 else dist.astype(np.uint8)
        ax.imshow(dn, cmap="hot"); ax.set_title(f"Image {i}"); ax.axis("off")
    plt.tight_layout(); save(fig, out_dir, "stage6a_distance_transform_weights.png")

    overlap = np.zeros(canvas.shape[:2], dtype=np.int32)
    for warped in warped_imgs:
        overlap += warped.any(axis=2).astype(np.int32)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(overlap, cmap="plasma")
    plt.colorbar(im, ax=ax, label="Images covering this pixel")
    ax.set_title(f"[{label}] Stage 6b — Overlap Count Map\n(yellow=4 images overlap → blending needed)", fontsize=11)
    ax.axis("off"); plt.tight_layout(); save(fig, out_dir, "stage6b_overlap_count_map.png")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    ax.set_title(f"[{label}] Stage 6c — Final Blended Mosaic", fontsize=12)
    ax.axis("off"); plt.tight_layout(); save(fig, out_dir, "stage6c_final_blended.png")

    cy, cx = h // 2, w // 2
    ch, cw = min(300, h // 3), min(400, w // 3)
    y1, y2 = max(0, cy - ch), min(h, cy + ch)
    x1, x2 = max(0, cx - cw), min(w, cx + cw)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"[{label}] Stage 6d — Zoomed Seam: Naive vs Blended", fontsize=12)
    ax1.imshow(cv2.cvtColor(canvas[y1:y2, x1:x2],  cv2.COLOR_BGR2RGB)); ax1.set_title("Naive — hard seam"); ax1.axis("off")
    ax2.imshow(cv2.cvtColor(blended[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)); ax2.set_title("Blended — smooth");  ax2.axis("off")
    plt.tight_layout(); save(fig, out_dir, "stage6d_seam_zoom_comparison.png")

    print(f"  Done -> outputs/stages/{name}/")


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        run_dataset(name, cfg)
    print("\nAll datasets complete.")
