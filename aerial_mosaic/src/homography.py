"""Stage 4: RANSAC homography estimation and transform chaining."""

import cv2
import numpy as np

MIN_INLIERS = 10  # reject homographies with fewer inliers than this


def _compute_H(kps_src: list, kps_dst: list, matches: list, ransac_thresh: float):
    """Compute H (dst->src) from a match list. Returns (H, inlier_count) or (None, 0)."""
    if len(matches) < 4:
        return None, 0
    src_pts = np.float32([kps_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
    inliers = int(mask.sum()) if mask is not None else 0
    return (H, mask) if (H is not None and inliers >= MIN_INLIERS) else (None, inliers)


def estimate_homographies(
    all_kps: list, matches_list: list, ransac_thresh: float = 5.0
) -> tuple[list, list]:
    """Compute pairwise homographies H[i] mapping image i+1 -> image i via RANSAC.

    Returns:
        H_pairs:  list of N-1 homography matrices (or None if estimation failed)
        masks:    list of N-1 inlier masks (uint8 arrays)
    """
    H_pairs, masks = [], []

    for i, matches in enumerate(matches_list):
        H, result = _compute_H(all_kps[i], all_kps[i + 1], matches, ransac_thresh)
        if isinstance(result, int):
            inliers, mask = result, None
        else:
            mask = result
            inliers = int(mask.sum()) if mask is not None else 0

        print(f"  Pair ({i},{i+1}): {len(matches)} matches -> {inliers} inliers", end="")
        if H is None:
            print(f" -> rejected (< {MIN_INLIERS})")
        else:
            print()

        H_pairs.append(H)
        masks.append(mask)

    return H_pairs, masks


def chain_homographies(H_pairs: list, ref_idx: int = 0) -> list:
    """Chain pairwise homographies so every image maps to the reference frame.

    H_chain[i] is the transform that warps image i onto the canvas of image ref_idx.
    Images unreachable through valid pairs get H_chain[i] = None and are skipped.
    """
    n = len(H_pairs) + 1
    H_chain = [None] * n
    H_chain[ref_idx] = np.eye(3, dtype=np.float64)

    # Forward: H_pairs[i] maps image i+1 -> image i
    for i in range(ref_idx, n - 1):
        if H_pairs[i] is None or H_chain[i] is None:
            H_chain[i + 1] = None
        else:
            H_chain[i + 1] = H_chain[i] @ H_pairs[i]

    # Backward
    for i in range(ref_idx - 1, -1, -1):
        if H_pairs[i] is None or H_chain[i + 1] is None:
            H_chain[i] = None
        else:
            H_chain[i] = H_chain[i + 1] @ np.linalg.inv(H_pairs[i])

    valid = sum(1 for h in H_chain if h is not None)
    print(f"  Chain: {valid}/{n} images reachable from reference {ref_idx}")
    return H_chain


def build_chain_with_fallback(
    all_kps: list, all_descs: list, max_skip: int = 3,
    ratio: float = 0.75, ransac_thresh: float = 5.0
) -> tuple[list, list, list]:
    """Build homography chain with skip-N bridging for datasets with strip transitions.

    When consecutive pair (i, i+1) fails RANSAC, tries to match image i directly
    with i+2, i+3 ... up to i+max_skip to jump over a flight-strip boundary.

    Returns:
        H_chain:      list of N transforms (None where unreachable)
        matches_list: list of N-1 match lists (consecutive pairs, for visualization)
        masks:        list of N-1 RANSAC masks (consecutive pairs, for visualization)
    """
    from matching import match_pair, match_consecutive_pairs

    n = len(all_descs)

    # Always compute consecutive matches for visualization purposes
    print("  Computing consecutive matches (for visualization)...")
    matches_list = match_consecutive_pairs(all_descs, ratio)
    H_pairs_consec, masks_consec = estimate_homographies(all_kps, matches_list, ransac_thresh)

    # Now build the chain with fallback bridging
    H_chain = [None] * n
    H_chain[0] = np.eye(3, dtype=np.float64)

    for j in range(1, n):
        bridged = False
        for skip in range(1, max_skip + 1):
            src = j - skip
            if src < 0:
                break
            if H_chain[src] is None:
                continue  # source itself is unreachable

            if skip == 1 and H_pairs_consec[src] is not None:
                # Already computed consecutive pair
                H_chain[j] = H_chain[src] @ H_pairs_consec[src]
                bridged = True
                break
            else:
                # Try a direct match between src and j
                print(f"  Bridging: trying pair ({src},{j})...")
                matches_bridge = match_pair(all_descs, src, j, ratio)
                H_bridge, result = _compute_H(all_kps[src], all_kps[j], matches_bridge, ransac_thresh)
                if isinstance(result, int):
                    inliers = result
                else:
                    inliers = int(result.sum()) if result is not None else 0
                print(f"    ({src},{j}): {len(matches_bridge)} matches -> {inliers} inliers", end="")
                if H_bridge is not None:
                    print(" -> accepted (bridge)")
                    H_chain[j] = H_chain[src] @ H_bridge
                    bridged = True
                    break
                else:
                    print(" -> rejected")

        if not bridged:
            H_chain[j] = None

    valid = sum(1 for h in H_chain if h is not None)
    print(f"  Chain (with fallback): {valid}/{n} images reachable")
    return H_chain, matches_list, masks_consec
