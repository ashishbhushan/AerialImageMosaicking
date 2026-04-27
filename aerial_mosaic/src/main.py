"""Entry point: runs the full aerial image mosaicking pipeline."""

import time
import sys
from pathlib import Path

# Allow running as `python src/main.py` from the project root
sys.path.insert(0, str(Path(__file__).parent))

from loader import load_images
from features import detect_and_describe
from matching import match_consecutive_pairs
from homography import estimate_homographies, chain_homographies
from warping import warp_images
from blending import blend_images
from visualize import generate_all_figures

# ── Configuration ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS = {
    "aukerman": {
        "dir":          str(_PROJECT_ROOT / "data" / "odm_data_aukerman" / "images"),
        "start":        2,    # skip first 2 (DSC00231 is missing, causing a gap)
        "n":            20,
        "ransac_thresh": 5.0,
        "min_inliers":  10,
    },
    "bellus": {
        "dir":          str(_PROJECT_ROOT / "data" / "odm_data_bellus" / "images"),
        "start":        12,   # start at the clean strip (images 12-19 have strong overlap)
        "n":            8,
        "ransac_thresh": 8.0, # looser threshold — repetitive tree texture means noisier matches
        "min_inliers":  6,
    },
}

ACTIVE_DATASET = "aukerman"   # change to "bellus" to run on the second dataset
MAX_WIDTH = 800
# ───────────────────────────────────────────────────────────────────────────────


def run_pipeline(data_dir=None, n=None, start=None, max_width=MAX_WIDTH):
    cfg = DATASETS[ACTIVE_DATASET]
    if data_dir is None:
        data_dir = cfg["dir"]
    if n is None:
        n = cfg["n"]
    if start is None:
        start = cfg["start"]

    timings = {}
    total_start = time.time()

    print(f"\nDataset: {ACTIVE_DATASET} | images {start} .. {start+n-1}")

    # Stage 1: Load
    print("\n[Stage 1] Loading images...")
    t0 = time.time()
    color_imgs, gray_imgs = load_images(data_dir, max_width=max_width, n=n, start=start)
    timings["1_load"] = time.time() - t0

    # Stage 2: SIFT
    print("\n[Stage 2] Detecting SIFT features...")
    t0 = time.time()
    all_kps, all_descs = detect_and_describe(gray_imgs)
    timings["2_sift"] = time.time() - t0
    avg_kps = sum(len(k) for k in all_kps) / len(all_kps)
    print(f"  Average keypoints per image: {avg_kps:.0f}")

    # Stage 3: Matching
    print("\n[Stage 3] Matching features...")
    t0 = time.time()
    matches_list = match_consecutive_pairs(all_descs)
    timings["3_match"] = time.time() - t0

    # Stage 4: Homography
    print("\n[Stage 4] Estimating homographies...")
    t0 = time.time()
    ransac_thresh = cfg.get("ransac_thresh", 5.0)
    min_inliers   = cfg.get("min_inliers", 10)
    H_pairs, masks = estimate_homographies(all_kps, matches_list, ransac_thresh, min_inliers)
    H_chain = chain_homographies(H_pairs, ref_idx=len(color_imgs) // 2)
    timings["4_homography"] = time.time() - t0

    # Evaluation: inlier ratios
    for i, (m_list, mask) in enumerate(zip(matches_list, masks)):
        if mask is not None:
            ratio = mask.sum() / max(len(m_list), 1)
            print(f"  Pair ({i},{i+1}): inlier ratio {ratio:.2f}")

    # Stage 5: Warping
    print("\n[Stage 5] Warping images...")
    t0 = time.time()
    canvas, warped_imgs = warp_images(color_imgs, H_chain)
    timings["5_warp"] = time.time() - t0

    # Stage 6: Blending
    print("\n[Stage 6] Blending seams...")
    t0 = time.time()
    blended = blend_images(warped_imgs, canvas.shape)
    timings["6_blend"] = time.time() - t0

    # Visualizations
    print("\n[Stage 7] Generating report figures...")
    t0 = time.time()
    generate_all_figures(
        color_imgs, all_kps, all_descs,
        matches_list, masks,
        warped_imgs, canvas, blended
    )
    timings["7_visualize"] = time.time() - t0

    timings["total"] = time.time() - total_start

    # Runtime table
    print("\n=== Runtime Breakdown ===")
    for stage, secs in timings.items():
        print(f"  {stage:<18} {secs:.2f}s")

    print(f"\nDone. Final mosaic: outputs/final_mosaic.jpg")
    return blended, canvas, all_kps, matches_list, masks


if __name__ == "__main__":
    run_pipeline()
