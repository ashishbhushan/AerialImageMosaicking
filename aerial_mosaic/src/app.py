"""Gradio UI for the aerial image mosaicking pipeline."""

import sys
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from loader import load_images
from features import detect_and_describe
from homography import build_chain_with_fallback
from warping import warp_images
from blending import blend_images

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Per-dataset config:
#   dir       – path to images folder
#   start     – skip this many images from the front of the sorted list
#   max_n     – hard cap (dataset size or usable strip length)
#   default_n – slider default
#   max_skip  – how far back to bridge when a consecutive pair fails
_DATASET_DEFAULTS = {
    "Aukerman (fields)": {
        "dir":          str(_PROJECT_ROOT / "data" / "odm_data_aukerman" / "images"),
        "start":        2,
        "max_n":        25,
        "default_n":    15,
        "max_skip":     3,
        "ransac_thresh": 5.0,
        "min_inliers":  10,
    },
    "Bellus (urban)": {
        "dir":          str(_PROJECT_ROOT / "data" / "odm_data_bellus" / "images"),
        "start":        0,
        "max_n":        20,
        "default_n":    12,
        "max_skip":     3,
        "ransac_thresh": 8.0,  # looser — repetitive tree texture causes noisy matches
        "min_inliers":  6,
    },
    "Pacifica (coastal)": {
        "dir":          str(_PROJECT_ROOT / "data" / "odm_data_pacifica" / "images"),
        "start":        0,
        "max_n":        12,
        "default_n":    12,
        "max_skip":     11,    # images are out of sequential order — look at all pairs
        "ransac_thresh": 5.0,
        "min_inliers":  10,
    },
}


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _find_best_pair(matches_list, masks):
    """Return the index of the pair with the most RANSAC inliers."""
    best_i, best_count = 0, 0
    for i, (m, mask) in enumerate(zip(matches_list, masks)):
        count = int(mask.sum()) if mask is not None else len(m)
        if count > best_count:
            best_count, best_i = count, i
    return best_i


def update_slider(dataset: str):
    """Called when the dataset dropdown changes — adjust the slider range."""
    cfg = _DATASET_DEFAULTS[dataset]
    return gr.update(maximum=cfg["max_n"], value=cfg["default_n"])


def run_pipeline_ui(n_images: int, dataset: str):
    cfg = _DATASET_DEFAULTS[dataset]
    data_dir = cfg["dir"]
    start    = cfg["start"]
    max_skip = cfg["max_skip"]

    # Stages 1-2
    color_imgs, gray_imgs = load_images(data_dir, n=int(n_images), start=int(start))
    all_kps, all_descs = detect_and_describe(gray_imgs)

    # Stages 3-4: fallback bridging handles both strip transitions and out-of-order datasets
    H_chain, matches_list, masks = build_chain_with_fallback(
        all_kps, all_descs, max_skip=max_skip,
        ransac_thresh=cfg["ransac_thresh"], min_inliers=cfg["min_inliers"],
    )

    # Panel A: SIFT keypoints on first image
    kp_vis = cv2.drawKeypoints(
        color_imgs[0], all_kps[0], None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # Panel B: pick the pair with the most inliers for the RANSAC visualization
    best = _find_best_pair(matches_list, masks)
    good = matches_list[best]
    mask = masks[best]
    if mask is not None:
        inliers  = [m for m, f in zip(good, mask.ravel()) if f]
        outliers = [m for m, f in zip(good, mask.ravel()) if not f]
    else:
        inliers, outliers = good, []

    img_a, img_b = color_imgs[best], color_imgs[best + 1]
    kps_a, kps_b = all_kps[best], all_kps[best + 1]

    match_vis = cv2.drawMatches(
        img_a, kps_a, img_b, kps_b,
        inliers, None, matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    match_vis = cv2.drawMatches(
        img_a, kps_a, img_b, kps_b,
        outliers, match_vis, matchColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.putText(
        match_vis,
        f"Best pair ({best},{best+1})  |  {len(inliers)} inliers  {len(outliers)} outliers",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # Stages 5-6
    canvas, warped_imgs = warp_images(color_imgs, H_chain)
    blended = blend_images(warped_imgs, canvas.shape)

    # Save for download
    out_path = str(_PROJECT_ROOT / "aerial_mosaic" / "outputs" / "final_mosaic.jpg")
    Path(out_path).parent.mkdir(exist_ok=True)
    cv2.imwrite(out_path, blended)

    return _to_rgb(kp_vis), _to_rgb(match_vis), _to_rgb(canvas), _to_rgb(blended), out_path


with gr.Blocks(title="Aerial Image Mosaicking") as demo:
    gr.Markdown("# Aerial Image Mosaicking Pipeline\nCS 415 — Computer Vision I, UIC Spring 2026")

    with gr.Row():
        dataset  = gr.Dropdown(
            list(_DATASET_DEFAULTS.keys()),
            value="Aukerman (fields)",
            label="Dataset",
        )
        n_images = gr.Slider(5, 25, value=15, step=1, label="Number of images")

    dataset.change(fn=update_slider, inputs=dataset, outputs=n_images)

    run_btn = gr.Button("Run Pipeline", variant="primary")

    with gr.Row():
        out_kp    = gr.Image(label="SIFT Keypoints (image 0)")
        out_match = gr.Image(label="RANSAC — Best pair: inliers (green) / outliers (red)")

    with gr.Row():
        out_naive   = gr.Image(label="Naive Mosaic (no blending)")
        out_blended = gr.Image(label="Final Blended Mosaic")

    out_file = gr.File(label="Download final mosaic (JPG)")

    run_btn.click(
        fn=run_pipeline_ui,
        inputs=[n_images, dataset],
        outputs=[out_kp, out_match, out_naive, out_blended, out_file],
    )

if __name__ == "__main__":
    demo.launch()
