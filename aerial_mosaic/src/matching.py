"""Stage 3: Feature matching with FLANN and Lowe's ratio test."""

import cv2

# FLANN parameters for SIFT (uses KD-tree)
_FLANN_PARAMS = dict(algorithm=1, trees=5)
_SEARCH_PARAMS = dict(checks=50)


def _ratio_test(raw_matches: list, ratio: float) -> list:
    return [m for m, n in raw_matches if m.distance < ratio * n.distance]


def match_pair(all_descs: list, i: int, j: int, ratio: float = 0.75) -> list:
    """Match descriptors between images i and j (any pair, not just consecutive)."""
    if all_descs[i] is None or all_descs[j] is None:
        return []
    flann = cv2.FlannBasedMatcher(_FLANN_PARAMS, _SEARCH_PARAMS)
    raw = flann.knnMatch(all_descs[i], all_descs[j], k=2)
    return _ratio_test(raw, ratio)


def match_consecutive_pairs(all_descs: list, ratio: float = 0.75) -> list[list]:
    """Match descriptors between each consecutive image pair.

    Returns:
        matches_list: list of length N-1; each entry is a list of cv2.DMatch
                      objects that passed the ratio test.
    """
    flann = cv2.FlannBasedMatcher(_FLANN_PARAMS, _SEARCH_PARAMS)
    matches_list = []

    for i in range(len(all_descs) - 1):
        if all_descs[i] is None or all_descs[i + 1] is None:
            matches_list.append([])
            print(f"  Pair ({i},{i+1}): skipped (no descriptors)")
            continue
        raw = flann.knnMatch(all_descs[i], all_descs[i + 1], k=2)
        good = _ratio_test(raw, ratio)
        matches_list.append(good)
        print(f"  Pair ({i},{i+1}): {len(raw)} raw -> {len(good)} after ratio test")

    return matches_list
