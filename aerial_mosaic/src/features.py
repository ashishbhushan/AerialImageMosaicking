"""Stage 2: Feature detection and description.

Three detectors are implemented for comparative study:

  SIFT  – Scale-Invariant Feature Transform (Lowe 2004)
          128-D float descriptor, rotation & scale invariant, matched with FLANN.
          Gold standard for accuracy; patented (now expired).

  ORB   – Oriented FAST + Rotated BRIEF (Rublee 2011)
          256-bit binary descriptor, very fast, patent-free.
          Good for real-time applications; weaker on large scale changes.

  AKAZE – Accelerated KAZE (Alcantarilla 2012)
          486-bit binary descriptor (MLDB), nonlinear scale-space.
          Better than ORB on texture-rich aerial imagery; faster than SIFT.

compare_detectors() runs all three on the same image set and returns a
timing + keypoint + match-count table for the research evaluation section.
"""

import time

import cv2
import numpy as np


# ── FLANN matcher (for float descriptors: SIFT) ───────────────────────────────
_FLANN_PARAMS  = dict(algorithm=1, trees=5)
_SEARCH_PARAMS = dict(checks=50)


def _ratio_test(raw: list, ratio: float) -> list:
    return [m for m, n in raw if m.distance < ratio * n.distance]


# ── SIFT ─────────────────────────────────────────────────────────────────────

def detect_and_describe(
    gray_imgs: list,
    n_features: int = 0,
) -> tuple[list, list]:
    """SIFT feature detection and description (float128 descriptor).

    Returns
    -------
    all_kps   : list[list[cv2.KeyPoint]]
    all_descs : list[np.ndarray]  shape (N, 128)
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    all_kps, all_descs = [], []

    for i, gray in enumerate(gray_imgs):
        kps, descs = sift.detectAndCompute(gray, None)
        all_kps.append(kps)
        all_descs.append(descs)
        print(f"  Image {i}: {len(kps)} SIFT keypoints")

    return all_kps, all_descs


def match_sift_pair(descs_i, descs_j, ratio: float = 0.75) -> list:
    if descs_i is None or descs_j is None:
        return []
    flann = cv2.FlannBasedMatcher(_FLANN_PARAMS, _SEARCH_PARAMS)
    raw = flann.knnMatch(descs_i, descs_j, k=2)
    return _ratio_test(raw, ratio)


def match_consecutive_pairs(all_descs: list, ratio: float = 0.75) -> list:
    """SIFT consecutive-pair matching (FLANN + Lowe ratio test)."""
    matches_list = []
    flann = cv2.FlannBasedMatcher(_FLANN_PARAMS, _SEARCH_PARAMS)

    for i in range(len(all_descs) - 1):
        if all_descs[i] is None or all_descs[i + 1] is None:
            matches_list.append([])
            continue
        raw = flann.knnMatch(all_descs[i], all_descs[i + 1], k=2)
        good = _ratio_test(raw, ratio)
        matches_list.append(good)
        print(f"  Pair ({i},{i+1}): {len(raw)} raw → {len(good)} after ratio test")

    return matches_list


def match_pair(all_descs: list, i: int, j: int, ratio: float = 0.75) -> list:
    """Match SIFT descriptors between any pair (i, j)."""
    return match_sift_pair(all_descs[i], all_descs[j], ratio)


def count_raw_matches(all_descs: list) -> list:
    flann = cv2.FlannBasedMatcher(_FLANN_PARAMS, _SEARCH_PARAMS)
    counts = []
    for i in range(len(all_descs) - 1):
        if all_descs[i] is None or all_descs[i + 1] is None:
            counts.append(0)
            continue
        raw = flann.knnMatch(all_descs[i], all_descs[i + 1], k=2)
        counts.append(len(raw))
    return counts


# ── ORB ──────────────────────────────────────────────────────────────────────

def detect_and_describe_orb(
    gray_imgs: list,
    n_features: int = 4000,
) -> tuple[list, list]:
    """ORB feature detection and description (binary 256-bit descriptor).

    Returns
    -------
    all_kps   : list[list[cv2.KeyPoint]]
    all_descs : list[np.ndarray]  shape (N, 32)  uint8
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    all_kps, all_descs = [], []

    for i, gray in enumerate(gray_imgs):
        kps, descs = orb.detectAndCompute(gray, None)
        all_kps.append(kps)
        all_descs.append(descs)
        print(f"  Image {i}: {len(kps)} ORB keypoints")

    return all_kps, all_descs


def match_consecutive_pairs_orb(all_descs: list, ratio: float = 0.75) -> list:
    """ORB consecutive-pair matching (BFMatcher Hamming + ratio test)."""
    matches_list = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    for i in range(len(all_descs) - 1):
        di, dj = all_descs[i], all_descs[i + 1]
        if di is None or dj is None or len(di) < 2 or len(dj) < 2:
            matches_list.append([])
            continue
        raw = bf.knnMatch(di, dj, k=2)
        good = _ratio_test(raw, ratio)
        matches_list.append(good)
        print(f"  Pair ({i},{i+1}): {len(raw)} raw → {len(good)} after ratio test (ORB)")

    return matches_list


def match_pair_orb(all_descs: list, i: int, j: int, ratio: float = 0.75) -> list:
    di, dj = all_descs[i], all_descs[j]
    if di is None or dj is None or len(di) < 2 or len(dj) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(di, dj, k=2)
    return _ratio_test(raw, ratio)


# ── AKAZE ─────────────────────────────────────────────────────────────────────

def detect_and_describe_akaze(
    gray_imgs: list,
    threshold: float = 0.001,
) -> tuple[list, list]:
    """AKAZE feature detection and description (binary MLDB 486-bit descriptor).

    AKAZE operates in a nonlinear scale-space (unlike SIFT's Gaussian
    scale-space), making it more robust to noise and repetitive texture —
    exactly the challenges present in aerial agricultural imagery.

    Returns
    -------
    all_kps   : list[list[cv2.KeyPoint]]
    all_descs : list[np.ndarray]  shape (N, 61)  uint8
    """
    akaze = cv2.AKAZE_create(threshold=threshold)
    all_kps, all_descs = [], []

    for i, gray in enumerate(gray_imgs):
        kps, descs = akaze.detectAndCompute(gray, None)
        all_kps.append(kps)
        all_descs.append(descs)
        print(f"  Image {i}: {len(kps)} AKAZE keypoints")

    return all_kps, all_descs


def match_consecutive_pairs_akaze(all_descs: list, ratio: float = 0.75) -> list:
    """AKAZE consecutive-pair matching (BFMatcher Hamming + ratio test)."""
    matches_list = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    for i in range(len(all_descs) - 1):
        di, dj = all_descs[i], all_descs[i + 1]
        if di is None or dj is None or len(di) < 2 or len(dj) < 2:
            matches_list.append([])
            continue
        raw = bf.knnMatch(di, dj, k=2)
        good = _ratio_test(raw, ratio)
        matches_list.append(good)
        print(f"  Pair ({i},{i+1}): {len(raw)} raw → {len(good)} after ratio test (AKAZE)")

    return matches_list


def match_pair_akaze(all_descs: list, i: int, j: int, ratio: float = 0.75) -> list:
    di, dj = all_descs[i], all_descs[j]
    if di is None or dj is None or len(di) < 2 or len(dj) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(di, dj, k=2)
    return _ratio_test(raw, ratio)


# ── Unified dispatcher ────────────────────────────────────────────────────────

DETECTOR_NAMES = ("SIFT", "ORB", "AKAZE")


def detect_and_describe_by_name(
    gray_imgs: list,
    detector: str = "SIFT",
) -> tuple[list, list]:
    """Detect and describe using the named detector ('SIFT', 'ORB', 'AKAZE')."""
    d = detector.upper()
    if d == "SIFT":
        return detect_and_describe(gray_imgs)
    if d == "ORB":
        return detect_and_describe_orb(gray_imgs)
    if d == "AKAZE":
        return detect_and_describe_akaze(gray_imgs)
    raise ValueError(f"Unknown detector: {detector!r}. Choose from {DETECTOR_NAMES}")


def match_pair_by_name(
    all_descs: list,
    i: int,
    j: int,
    detector: str = "SIFT",
    ratio: float = 0.75,
) -> list:
    d = detector.upper()
    if d == "SIFT":
        return match_pair(all_descs, i, j, ratio)
    if d == "ORB":
        return match_pair_orb(all_descs, i, j, ratio)
    if d == "AKAZE":
        return match_pair_akaze(all_descs, i, j, ratio)
    raise ValueError(f"Unknown detector: {detector!r}")


def match_consecutive_by_name(
    all_descs: list,
    detector: str = "SIFT",
    ratio: float = 0.75,
) -> list:
    d = detector.upper()
    if d == "SIFT":
        return match_consecutive_pairs(all_descs, ratio)
    if d == "ORB":
        return match_consecutive_pairs_orb(all_descs, ratio)
    if d == "AKAZE":
        return match_consecutive_pairs_akaze(all_descs, ratio)
    raise ValueError(f"Unknown detector: {detector!r}")


# ── Three-way comparative benchmark ──────────────────────────────────────────

def compare_detectors(gray_imgs: list, ratio: float = 0.75) -> dict:
    """Run SIFT, ORB, and AKAZE on the same image set and return a comparison table.

    Returns a dict with keys 'SIFT', 'ORB', 'AKAZE', each mapping to:
      {
        'kps':          list of per-image keypoint counts
        'avg_kps':      float
        'matches':      list of per-consecutive-pair match counts
        'avg_matches':  float
        'detect_time':  seconds for detection across all images
        'match_time':   seconds for consecutive-pair matching
        'total_time':   detect + match
        'all_kps':      raw keypoints (for visualization)
        'all_descs':    raw descriptors
      }
    """
    results = {}

    configs = [
        ("SIFT",  detect_and_describe,       match_consecutive_pairs),
        ("ORB",   detect_and_describe_orb,   match_consecutive_pairs_orb),
        ("AKAZE", detect_and_describe_akaze, match_consecutive_pairs_akaze),
    ]

    for name, detect_fn, match_fn in configs:
        print(f"\n  [{name}] detecting...")
        t0 = time.perf_counter()
        kps, descs = detect_fn(gray_imgs)
        detect_time = time.perf_counter() - t0

        print(f"  [{name}] matching...")
        t0 = time.perf_counter()
        matches_list = match_fn(descs, ratio)
        match_time = time.perf_counter() - t0

        kp_counts = [len(k) for k in kps]
        m_counts  = [len(m) for m in matches_list]

        results[name] = {
            "kps":          kp_counts,
            "avg_kps":      float(np.mean(kp_counts)),
            "matches":      m_counts,
            "avg_matches":  float(np.mean(m_counts)),
            "detect_time":  detect_time,
            "match_time":   match_time,
            "total_time":   detect_time + match_time,
            "all_kps":      kps,
            "all_descs":    descs,
        }

        print(
            f"  [{name}] avg_kps={results[name]['avg_kps']:.0f}  "
            f"avg_matches={results[name]['avg_matches']:.0f}  "
            f"time={results[name]['total_time']:.2f}s"
        )

    return results
