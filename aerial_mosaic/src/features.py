"""Stage 2: SIFT feature detection and description."""

import cv2
import numpy as np


def detect_and_describe(gray_imgs: list) -> tuple[list, list]:
    """Run SIFT on each grayscale image.

    Returns:
        all_kps:   list of keypoint lists, one per image
        all_descs: list of descriptor arrays (N x 128 float32), one per image
    """
    sift = cv2.SIFT_create()
    all_kps, all_descs = [], []

    for i, gray in enumerate(gray_imgs):
        kps, descs = sift.detectAndCompute(gray, None)
        all_kps.append(kps)
        all_descs.append(descs)
        print(f"  Image {i}: {len(kps)} keypoints detected")

    return all_kps, all_descs
