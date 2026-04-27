"""All visualization and figure generation for the report."""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def _save(fig, name: str):
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Figure 1: sample inputs ───────────────────────────────────────────────────

def fig1_sample_inputs(color_imgs: list, n: int = 4):
    n = min(n, len(color_imgs))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, color_imgs[:n]):
        ax.imshow(_rgb(img))
        ax.axis("off")
    fig.suptitle("Figure 1: Sample Input Drone Images", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig1_sample_inputs.png")


# ── Figure 2: SIFT keypoints ──────────────────────────────────────────────────

def fig2_keypoints(color_imgs: list, all_kps: list, n: int = 3):
    n = min(n, len(color_imgs))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, kps in zip(axes, color_imgs[:n], all_kps[:n]):
        vis = cv2.drawKeypoints(img, kps, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(_rgb(vis))
        ax.set_title(f"{len(kps)} keypoints", fontsize=11)
        ax.axis("off")
    fig.suptitle("Figure 2: SIFT Keypoints", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig2_keypoints.png")


# ── Figure 3: raw matches ─────────────────────────────────────────────────────

def fig3_raw_matches(color_imgs: list, all_kps: list, all_descs: list):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw   = flann.knnMatch(all_descs[0], all_descs[1], k=2)
    all_m = [m for m, _ in raw]
    vis = cv2.drawMatches(
        color_imgs[0], all_kps[0], color_imgs[1], all_kps[1],
        all_m[:80], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(_rgb(vis))
    ax.set_title(f"Figure 3: Raw Feature Matches (first 80 of {len(all_m)})", fontsize=13)
    ax.axis("off")
    _save(fig, "fig3_raw_matches.png")


# ── Figure 4: RANSAC inliers vs outliers ─────────────────────────────────────

def fig4_inliers_outliers(color_imgs, all_kps, matches_list, masks):
    good = matches_list[0]
    mask = masks[0]
    if mask is None:
        print("  fig4: no mask for pair 0, skipping")
        return
    inliers  = [m for m, f in zip(good, mask.ravel()) if f]
    outliers = [m for m, f in zip(good, mask.ravel()) if not f]
    vis_in  = cv2.drawMatches(color_imgs[0], all_kps[0], color_imgs[1], all_kps[1],
                               inliers, None, matchColor=(0, 255, 0),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis_out = cv2.drawMatches(color_imgs[0], all_kps[0], color_imgs[1], all_kps[1],
                               outliers, None, matchColor=(0, 0, 255),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.imshow(_rgb(vis_in));  ax1.set_title(f"Inliers ({len(inliers)})",  color="green"); ax1.axis("off")
    ax2.imshow(_rgb(vis_out)); ax2.set_title(f"Outliers ({len(outliers)})", color="red");   ax2.axis("off")
    fig.suptitle("Figure 4: RANSAC Inliers (green) vs Outliers (red)", fontsize=13)
    plt.tight_layout()
    _save(fig, "fig4_inliers_outliers.png")


# ── Figure 5: warped overlay ──────────────────────────────────────────────────

def fig5_warped_overlay(warped_imgs: list, n: int = 6):
    n = min(n, len(warped_imgs))
    h, w = warped_imgs[0].shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.float64)
    count   = np.zeros((h, w),    dtype=np.float64)
    for warped in warped_imgs[:n]:
        mask = warped.any(axis=2).astype(np.float64)
        overlay += warped.astype(np.float64) * mask[:, :, np.newaxis]
        count   += mask
    valid  = count > 0
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[valid] = np.clip(overlay[valid] / count[valid, np.newaxis], 0, 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(_rgb(result))
    ax.set_title(f"Figure 5: First {n} Warped Images Overlaid", fontsize=12)
    ax.axis("off")
    _save(fig, "fig5_warped_overlay.png")


# ── Figure 6: naive mosaic ────────────────────────────────────────────────────

def fig6_naive_mosaic(canvas: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(_rgb(canvas))
    ax.set_title("Figure 6: Naive Mosaic (no blending)", fontsize=13)
    ax.axis("off")
    _save(fig, "fig6_naive_mosaic.png")


# ── Figure 7: blended mosaic ──────────────────────────────────────────────────

def fig7_blended_mosaic(blended: np.ndarray, method: str = "Distance-Transform"):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(_rgb(blended))
    ax.set_title(f"Figure 7: Final Mosaic — {method} Blending", fontsize=13)
    ax.axis("off")
    _save(fig, "fig7_blended_mosaic.png")
    cv2.imwrite(str(OUTPUTS / "final_mosaic.jpg"), blended)


# ── Figure 8: naive vs blended comparison ────────────────────────────────────

def fig8_comparison(canvas: np.ndarray, blended: np.ndarray, blend_label: str = "Distance-T"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    ax1.imshow(_rgb(canvas));  ax1.set_title("No blending",      fontsize=13); ax1.axis("off")
    ax2.imshow(_rgb(blended)); ax2.set_title(f"{blend_label} blending", fontsize=13); ax2.axis("off")
    fig.suptitle("Figure 8: Blending Comparison", fontsize=14)
    plt.tight_layout()
    _save(fig, "fig8_comparison.png")


# ── Figure 9: pipeline metrics table ─────────────────────────────────────────

def _apply_table_style(t, n_cols):
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.auto_set_column_width(list(range(n_cols)))
    for (r, _), cell in t._cells.items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if r % 2 == 0 else "white")


def fig9_metrics_table(all_kps, raw_counts, matches_list, masks, reproj_errors, timings):
    n_imgs  = len(all_kps)
    n_pairs = len(matches_list)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(13, max(12, (n_imgs + 2 + n_pairs + 1 + 9) * 0.38 + 2.5)),
        gridspec_kw={"height_ratios": [n_imgs + 2, n_pairs + 1, 9]},
    )
    fig.suptitle("Figure 9: Pipeline Evaluation Metrics", fontsize=14, fontweight="bold")

    # Table 1 — feature detection
    ax1.axis("off")
    ax1.set_title("Feature Detection", fontsize=11, fontweight="bold", pad=4)
    kp_rows = [[str(i), f"{len(all_kps[i]):,}"] for i in range(n_imgs)]
    avg_kp  = sum(len(k) for k in all_kps) / n_imgs
    kp_rows.append(["Average", f"{avg_kp:,.0f}"])
    t1 = ax1.table(cellText=kp_rows,
                   colLabels=["Image", "Keypoints Detected"],
                   loc="center", cellLoc="center")
    _apply_table_style(t1, 2)
    for j in range(2):
        t1[len(kp_rows), j].set_text_props(fontweight="bold")

    # Table 2 — matching & homography
    ax2.axis("off")
    ax2.set_title("Feature Matching & RANSAC Homography", fontsize=11, fontweight="bold", pad=4)
    match_rows = []
    for i in range(n_pairs):
        mask   = masks[i]
        n_inl  = int(mask.sum()) if mask is not None else 0
        ratio  = n_inl / max(len(matches_list[i]), 1)
        re     = reproj_errors[i] if i < len(reproj_errors) else float("nan")
        re_str = f"{re:.2f}" if not np.isnan(re) else "—"
        rc     = str(raw_counts[i]) if i < len(raw_counts) else "—"
        match_rows.append([f"({i},{i+1})", rc, str(len(matches_list[i])),
                           str(n_inl), f"{ratio:.2f}", re_str])
    t2 = ax2.table(
        cellText=match_rows,
        colLabels=["Pair", "Raw\nMatches", "After Ratio\nTest",
                   "RANSAC\nInliers", "Inlier\nRatio", "Reproj.\nError (px)"],
        loc="center", cellLoc="center",
    )
    _apply_table_style(t2, 6)
    for i, row in enumerate(match_rows, start=1):
        ratio = float(row[4])
        color = "#d4edda" if ratio >= 0.7 else "#fff3cd" if ratio >= 0.4 else "#f8d7da"
        t2[i, 4].set_facecolor(color)

    # Table 3 — runtime
    ax3.axis("off")
    ax3.set_title("Runtime Breakdown", fontsize=11, fontweight="bold", pad=4)
    stage_map = {
        "1_load":       "Stage 1 — Image Loading & Preprocessing",
        "2_features":   "Stage 2 — Feature Detection",
        "3_match":      "Stage 3 — Feature Matching",
        "4_homography": "Stage 4 — RANSAC Homography Estimation",
        "5_warp":       "Stage 5 — Image Warping",
        "6_exposure":   "Stage 6a — Exposure Compensation",
        "6_blend":      "Stage 6b — Seam Blending",
        "7_visualize":  "Stage 7 — Figure Generation",
    }
    total_t = timings.get("total", 1.0)
    rt_rows = []
    for key, label in stage_map.items():
        if key in timings:
            t = timings[key]
            rt_rows.append([label, f"{t:.2f}", f"{100 * t / total_t:.1f}%"])
    rt_rows.append(["TOTAL", f"{total_t:.2f}", "100%"])
    t3 = ax3.table(cellText=rt_rows,
                   colLabels=["Stage", "Time (s)", "% of Total"],
                   loc="center", cellLoc="center")
    _apply_table_style(t3, 3)
    for j in range(3):
        t3[len(rt_rows), j].set_text_props(fontweight="bold")

    plt.tight_layout()
    _save(fig, "fig9_metrics_table.png")


# ── Figure 10: SIFT vs ORB vs AKAZE comparison ───────────────────────────────

def fig10_detector_comparison(color_imgs: list, comparison: dict):
    """Three-way detector comparison: keypoints, matches, speed.

    comparison : dict from features.compare_detectors()
    """
    detectors = [d for d in ("SIFT", "ORB", "AKAZE") if d in comparison]
    colors    = {"SIFT": "#2196F3", "ORB": "#4CAF50", "AKAZE": "#FF9800"}
    n_imgs    = len(color_imgs)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Figure 10: Feature Detector Comparison — SIFT vs ORB vs AKAZE\n"
        "Application: Aerial Image Mosaicking for Precision Agriculture & Disaster Mapping",
        fontsize=13, fontweight="bold",
    )

    # Row 1: keypoint visualisations for each detector
    for col, det in enumerate(detectors):
        ax = fig.add_subplot(3, len(detectors), col + 1)
        kps = comparison[det]["all_kps"][0]
        vis = cv2.drawKeypoints(
            color_imgs[0], kps, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        ax.imshow(_rgb(vis))
        ax.set_title(
            f"{det}\n{len(kps)} keypoints  "
            f"({comparison[det]['detect_time']:.2f}s detect)",
            fontsize=11, color=colors[det], fontweight="bold",
        )
        ax.axis("off")

    # Row 2: per-image keypoint count bar chart
    ax_kp = fig.add_subplot(3, 1, 2)
    x = np.arange(n_imgs)
    width = 0.25
    for k, det in enumerate(detectors):
        counts = comparison[det]["kps"]
        ax_kp.bar(x + k * width, counts, width, label=det, color=colors[det], alpha=0.85)
    ax_kp.set_xlabel("Image index"); ax_kp.set_ylabel("Keypoints")
    ax_kp.set_title("Keypoints per Image", fontsize=11)
    ax_kp.legend(); ax_kp.set_xticks(x + width)
    ax_kp.set_xticklabels([str(i) for i in range(n_imgs)], fontsize=8)
    ax_kp.grid(axis="y", alpha=0.3)

    # Row 3: summary bars (avg matches, speed)
    ax_sum = fig.add_subplot(3, 2, 5)
    avg_m  = [comparison[d]["avg_matches"] for d in detectors]
    bars   = ax_sum.bar(detectors, avg_m, color=[colors[d] for d in detectors], alpha=0.85)
    ax_sum.set_ylabel("Avg matches per pair"); ax_sum.set_title("Match Count (after ratio test)")
    for bar, val in zip(bars, avg_m):
        ax_sum.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax_sum.grid(axis="y", alpha=0.3)

    ax_spd = fig.add_subplot(3, 2, 6)
    times  = [comparison[d]["total_time"] for d in detectors]
    bars2  = ax_spd.bar(detectors, times, color=[colors[d] for d in detectors], alpha=0.85)
    ax_spd.set_ylabel("Total time (s)"); ax_spd.set_title("Speed (detect + match)")
    for bar, val in zip(bars2, times):
        ax_spd.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}s", ha="center", fontsize=10, fontweight="bold")
    ax_spd.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig10_detector_comparison.png")


# ── Figure 11: multi-band vs distance-transform blending ─────────────────────

def fig11_blending_comparison(
    canvas: np.ndarray,
    dist_blended: np.ndarray,
    multiband_blended: np.ndarray,
):
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    titles = [
        "Naive (overwrite)",
        "Distance-Transform Blending",
        "Multi-Band Laplacian Blending",
    ]
    for ax, img, title in zip(axes, [canvas, dist_blended, multiband_blended], titles):
        ax.imshow(_rgb(img)); ax.set_title(title, fontsize=12); ax.axis("off")

    fig.suptitle(
        "Figure 11: Blending Method Comparison\n"
        "Multi-band blending eliminates colour seams while preserving fine texture detail",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "fig11_blending_comparison.png")


# ── Figure 12: quality metrics summary ───────────────────────────────────────

def fig12_quality_metrics(quality_metrics: dict, exposure_gains: np.ndarray | None = None):
    """Visual summary of SSIM, PSNR, seam gradient score, and exposure gains."""
    has_gains = exposure_gains is not None and len(exposure_gains) > 1

    ncols = 2 if has_gains else 2
    fig, axes = plt.subplots(1, ncols + 1, figsize=(6 * (ncols + 1), 5))
    fig.suptitle(
        "Figure 12: Quantitative Quality Metrics",
        fontsize=14, fontweight="bold",
    )

    # Panel A: overlap SSIM per pair
    ax = axes[0]
    scores = quality_metrics.get("overlap_ssim_pairs", [])
    x = np.arange(len(scores))
    colors = ["#4CAF50" if (not np.isnan(s) and s >= 0.7) else
              "#FF9800" if (not np.isnan(s) and s >= 0.4) else "#f44336"
              for s in scores]
    ax.bar(x, [s if not np.isnan(s) else 0 for s in scores], color=colors, alpha=0.85)
    ax.set_xlabel("Consecutive pair"); ax.set_ylabel("SSIM")
    ax.set_title("Overlap SSIM per Pair\n(green ≥ 0.7 = good alignment)")
    ax.axhline(0.7, color="green",  ls="--", lw=1, alpha=0.6, label="0.7 threshold")
    ax.axhline(0.4, color="orange", ls="--", lw=1, alpha=0.6, label="0.4 threshold")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)

    # Panel B: summary metric table
    ax2 = axes[1]
    ax2.axis("off")
    rows = [
        ["Mean overlap SSIM",        f"{quality_metrics.get('mean_overlap_ssim', float('nan')):.3f}"],
        ["SSIM: blended vs naive",   f"{quality_metrics.get('ssim_blended_vs_naive', float('nan')):.3f}"],
        ["PSNR: blended vs naive",   f"{quality_metrics.get('psnr_blended_vs_naive', float('nan')):.1f} dB"],
        ["Seam gradient score",      f"{quality_metrics.get('seam_gradient_score', float('nan')):.2f}"],
    ]
    t = ax2.table(cellText=rows,
                  colLabels=["Metric", "Value"],
                  loc="center", cellLoc="center")
    _apply_table_style(t, 2)
    ax2.set_title("Quality Summary", fontsize=11)

    # Panel C: exposure gains
    if has_gains:
        ax3 = axes[2]
        ax3.bar(np.arange(len(exposure_gains)), exposure_gains, color="#9C27B0", alpha=0.8)
        ax3.axhline(1.0, color="black", ls="--", lw=1)
        ax3.set_xlabel("Image index"); ax3.set_ylabel("Gain factor")
        ax3.set_title("Exposure Compensation Gains\n(1.0 = no correction)")
        ax3.grid(axis="y", alpha=0.3)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    _save(fig, "fig12_quality_metrics.png")


# ── Master entry point ────────────────────────────────────────────────────────

def generate_all_figures(
    color_imgs, all_kps, all_descs,
    matches_list, masks,
    warped_imgs, canvas, blended,
    raw_counts=None,
    reproj_errors=None,
    timings=None,
    multiband_blended=None,
    detector_comparison=None,
    quality_metrics=None,
    exposure_gains=None,
    blend_method: str = "Distance-Transform",
):
    print("Generating report figures...")

    fig1_sample_inputs(color_imgs)
    fig2_keypoints(color_imgs, all_kps)
    fig3_raw_matches(color_imgs, all_kps, all_descs)
    fig4_inliers_outliers(color_imgs, all_kps, matches_list, masks)
    fig5_warped_overlay(warped_imgs)
    fig6_naive_mosaic(canvas)
    fig7_blended_mosaic(blended, method=blend_method)
    fig8_comparison(canvas, blended, blend_label=blend_method)

    if raw_counts is not None and reproj_errors is not None and timings is not None:
        fig9_metrics_table(all_kps, raw_counts, matches_list, masks, reproj_errors, timings)

    if detector_comparison is not None:
        fig10_detector_comparison(color_imgs, detector_comparison)

    if multiband_blended is not None:
        dist_b = blended if blend_method != "Multi-Band" else canvas
        fig11_blending_comparison(canvas, dist_b, multiband_blended)

    if quality_metrics is not None:
        fig12_quality_metrics(quality_metrics, exposure_gains)

    print("All figures saved to outputs/")
