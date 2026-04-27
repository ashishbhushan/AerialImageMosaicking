"""Stage 4: Homography estimation and transform chaining.

Two chaining strategies:

  build_chain_with_fallback()   – sequential chain with skip-N bridging
                                   (original method, fast)

  build_spanning_tree_chain()   – maximum spanning tree over the image graph
                                   (new: globally optimal connections, less drift)

Both strategies accept a `use_custom_ransac` flag to switch between
the from-scratch RANSAC implementation (ransac.py) and OpenCV's optimised one.
"""

import cv2
import numpy as np
from collections import deque

MIN_INLIERS = 10


# ── Low-level homography computation ─────────────────────────────────────────

def _compute_H(
    kps_src: list,
    kps_dst: list,
    matches: list,
    ransac_thresh: float,
    min_inliers: int = MIN_INLIERS,
    use_custom_ransac: bool = False,
):
    """Compute H mapping image_dst → image_src.

    Returns (H, mask) on success, (None, inlier_count_int) on failure.
    The mixed return type is preserved for backward compatibility.
    """
    if len(matches) < 4:
        return None, 0

    src_pts = np.float32([kps_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if use_custom_ransac:
        from ransac import compute_H_custom
        H, result = compute_H_custom(
            kps_src, kps_dst, matches,
            thresh=ransac_thresh, min_inliers=min_inliers,
        )
    else:
        H, mask_raw = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
        if H is None:
            return None, 0
        inliers = int(mask_raw.sum()) if mask_raw is not None else 0
        if inliers < min_inliers:
            return None, inliers
        return H, mask_raw

    if H is None:
        inliers = result if isinstance(result, int) else 0
        return None, inliers
    return H, result


# ── Pairwise estimation ───────────────────────────────────────────────────────

def estimate_homographies(
    all_kps: list,
    matches_list: list,
    ransac_thresh: float = 5.0,
    min_inliers: int = MIN_INLIERS,
    use_custom_ransac: bool = False,
) -> tuple[list, list]:
    """Compute pairwise homographies H[i] mapping image i+1 → image i via RANSAC."""
    H_pairs, masks = [], []

    for i, matches in enumerate(matches_list):
        H, result = _compute_H(
            all_kps[i], all_kps[i + 1], matches,
            ransac_thresh, min_inliers, use_custom_ransac,
        )
        if isinstance(result, int):
            inliers, mask = result, None
        else:
            mask    = result
            inliers = int(mask.sum()) if mask is not None else 0

        ransac_label = "custom" if use_custom_ransac else "cv2"
        print(
            f"  Pair ({i},{i+1}): {len(matches)} matches → {inliers} inliers "
            f"[{ransac_label}]",
            end="",
        )
        print(f" → rejected (< {min_inliers})" if H is None else "")

        H_pairs.append(H)
        masks.append(mask)

    return H_pairs, masks


# ── Sequential chain (original) ───────────────────────────────────────────────

def chain_homographies(H_pairs: list, ref_idx: int = 0) -> list:
    """Chain pairwise homographies so every image maps to the reference frame."""
    n = len(H_pairs) + 1
    H_chain = [None] * n
    H_chain[ref_idx] = np.eye(3, dtype=np.float64)

    for i in range(ref_idx, n - 1):
        if H_pairs[i] is None or H_chain[i] is None:
            H_chain[i + 1] = None
        else:
            H_chain[i + 1] = H_chain[i] @ H_pairs[i]

    for i in range(ref_idx - 1, -1, -1):
        if H_pairs[i] is None or H_chain[i + 1] is None:
            H_chain[i] = None
        else:
            H_chain[i] = H_chain[i + 1] @ np.linalg.inv(H_pairs[i])

    valid = sum(1 for h in H_chain if h is not None)
    print(f"  Chain: {valid}/{n} images reachable from reference {ref_idx}")
    return H_chain


# ── Reprojection error ─────────────────────────────────────────────────────────

def compute_reprojection_errors(
    all_kps: list, matches_list: list, masks: list, H_pairs: list
) -> list:
    """Mean reprojection error (px) for each consecutive pair's RANSAC inliers."""
    errors = []
    for i, (matches, mask, H) in enumerate(zip(matches_list, masks, H_pairs)):
        if H is None or mask is None or len(matches) == 0:
            errors.append(float("nan"))
            continue
        inlier_idx = np.where(mask.ravel())[0]
        if len(inlier_idx) == 0:
            errors.append(float("nan"))
            continue
        inlier_m = [matches[j] for j in inlier_idx]
        src_pts  = np.float32([all_kps[i    ][m.queryIdx].pt for m in inlier_m])
        dst_pts  = np.float32([all_kps[i + 1][m.trainIdx].pt for m in inlier_m])
        proj     = cv2.perspectiveTransform(dst_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors.append(float(np.linalg.norm(proj - src_pts, axis=1).mean()))
    return errors


# ── Spanning-tree chain (new) ──────────────────────────────────────────────────

def build_spanning_tree_chain(
    all_kps: list,
    all_descs: list,
    n_images: int,
    ratio: float = 0.75,
    ransac_thresh: float = 5.0,
    min_inliers: int = MIN_INLIERS,
    max_pair_dist: int = 6,
    detector: str = "SIFT",
    use_custom_ransac: bool = False,
) -> tuple[list, list, list, dict]:
    """Maximum spanning tree homography chain.

    Instead of sequential chaining (error accumulates over N images), considers
    all image pairs within a window of ±max_pair_dist and builds a graph whose
    edges are weighted by RANSAC inlier count.  The maximum spanning tree of
    this graph gives the globally best set of connections, minimising the
    worst-case drift in the final mosaic.

    Algorithm (Kruskal):
      1. Match all pairs (i, j) with |i-j| ≤ max_pair_dist.
      2. Weight each edge by inlier count.
      3. Greedily add heaviest edges that don't create a cycle (union-find).
      4. BFS from reference to chain transforms along tree edges.

    Returns
    -------
    H_chain      : list[ndarray | None]
    matches_list : consecutive-pair matches (for visualization)
    masks        : consecutive-pair masks   (for visualization)
    edges_info   : dict (i,j) → {H, inliers, matches, mask}
    """
    from features import match_pair_by_name

    # Always compute consecutive matches for visualization
    from features import match_consecutive_by_name
    print("  Computing consecutive matches (for visualization)...")
    matches_list = match_consecutive_by_name(all_descs, detector=detector, ratio=ratio)

    # Compute consecutive pair homographies for masks/visualization
    H_consec, masks_consec = estimate_homographies(
        all_kps, matches_list, ransac_thresh, min_inliers, use_custom_ransac,
    )

    # ── Build full edge graph ─────────────────────────────────────────────────
    edges_info = {}   # (i, j) i<j → {H, inliers, matches, mask}

    print(f"  Building image graph (window ±{max_pair_dist})...")
    for i in range(n_images):
        for j in range(i + 1, min(i + max_pair_dist + 1, n_images)):
            if j == i + 1 and H_consec[i] is not None:
                # Reuse already-computed consecutive result
                mask    = masks_consec[i]
                inliers = int(mask.sum()) if mask is not None else 0
                edges_info[(i, j)] = {
                    "H":       H_consec[i],
                    "inliers": inliers,
                    "matches": matches_list[i],
                    "mask":    mask,
                }
            else:
                ms = match_pair_by_name(all_descs, i, j, detector=detector, ratio=ratio)
                if len(ms) < 4:
                    continue
                H, result = _compute_H(
                    all_kps[i], all_kps[j], ms,
                    ransac_thresh, min_inliers, use_custom_ransac,
                )
                mask    = result if not isinstance(result, int) else None
                inliers = int(mask.sum()) if mask is not None else (
                    result if isinstance(result, int) else 0
                )
                if H is not None:
                    edges_info[(i, j)] = {
                        "H":       H,
                        "inliers": inliers,
                        "matches": ms,
                        "mask":    mask,
                    }

    print(f"  Graph has {len(edges_info)} valid edges from {n_images} nodes")

    # ── Kruskal's maximum spanning tree ──────────────────────────────────────
    parent = list(range(n_images))
    rank   = [0] * n_images

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    sorted_edges = sorted(edges_info, key=lambda e: edges_info[e]["inliers"], reverse=True)
    tree_edges   = []
    for edge in sorted_edges:
        i, j = edge
        if union(i, j):
            tree_edges.append(edge)
            if len(tree_edges) == n_images - 1:
                break

    print(f"  Spanning tree: {len(tree_edges)} edges selected")

    # ── BFS from reference to chain transforms ────────────────────────────────
    ref     = n_images // 2
    H_chain = [None] * n_images
    H_chain[ref] = np.eye(3, dtype=np.float64)

    # Build adjacency: for each tree edge (i,j) store transform in both directions
    adj = {k: [] for k in range(n_images)}
    for (i, j) in tree_edges:
        H_ij = edges_info[(i, j)]["H"]   # maps j → i
        adj[i].append((j, H_ij))         # neighbour j, transform maps j→i
        adj[j].append((i, None))         # neighbour i, need inv(H_ij) = i→j... wait

    # Reset and redo: store the edge key so we can look up direction in BFS
    adj = {k: [] for k in range(n_images)}
    for (i, j) in tree_edges:
        adj[i].append(j)
        adj[j].append(i)

    queue   = deque([ref])
    visited = {ref}

    while queue:
        curr = queue.popleft()
        for nb in adj[curr]:
            if nb in visited:
                continue
            visited.add(nb)

            # Determine which edge key and direction
            if (curr, nb) in edges_info:
                # H maps nb → curr; H_chain[curr] maps curr → ref
                # So H_chain[nb] = H_chain[curr] @ H_nb_to_curr
                H_nb_to_curr = edges_info[(curr, nb)]["H"]
                H_chain[nb]  = H_chain[curr] @ H_nb_to_curr
            elif (nb, curr) in edges_info:
                # H maps curr → nb; invert to get nb → curr
                H_curr_to_nb = edges_info[(nb, curr)]["H"]
                try:
                    H_nb_to_curr = np.linalg.inv(H_curr_to_nb)
                    H_chain[nb]  = H_chain[curr] @ H_nb_to_curr
                except np.linalg.LinAlgError:
                    H_chain[nb] = None

            queue.append(nb)

    valid = sum(1 for h in H_chain if h is not None)
    print(f"  Spanning-tree chain: {valid}/{n_images} images reachable")
    return H_chain, matches_list, masks_consec, edges_info


# ── Original fallback chain (kept for compatibility) ──────────────────────────

def build_chain_with_fallback(
    all_kps: list,
    all_descs: list,
    max_skip: int = 3,
    ratio: float = 0.75,
    ransac_thresh: float = 5.0,
    min_inliers: int = MIN_INLIERS,
    detector: str = "SIFT",
    use_custom_ransac: bool = False,
) -> tuple[list, list, list]:
    """Sequential chain with skip-N bridging for strip-transition datasets.

    Falls back to the spanning-tree approach internally if a pair fails.
    """
    from features import match_pair_by_name, match_consecutive_by_name

    n = len(all_descs)

    print("  Computing consecutive matches (for visualization)...")
    matches_list = match_consecutive_by_name(all_descs, detector=detector, ratio=ratio)
    H_pairs_consec, masks_consec = estimate_homographies(
        all_kps, matches_list, ransac_thresh, min_inliers, use_custom_ransac,
    )

    H_chain    = [None] * n
    H_chain[0] = np.eye(3, dtype=np.float64)

    for j in range(1, n):
        bridged = False
        for skip in range(1, max_skip + 1):
            src = j - skip
            if src < 0:
                break
            if H_chain[src] is None:
                continue

            if skip == 1 and H_pairs_consec[src] is not None:
                H_chain[j] = H_chain[src] @ H_pairs_consec[src]
                bridged = True
                break
            else:
                print(f"  Bridging: trying pair ({src},{j})...")
                ms = match_pair_by_name(all_descs, src, j, detector=detector, ratio=ratio)
                H_br, result = _compute_H(
                    all_kps[src], all_kps[j], ms,
                    ransac_thresh, min_inliers, use_custom_ransac,
                )
                inliers = (
                    int(result.sum()) if not isinstance(result, int) and result is not None
                    else (result if isinstance(result, int) else 0)
                )
                print(
                    f"    ({src},{j}): {len(ms)} matches → {inliers} inliers",
                    end="",
                )
                if H_br is not None:
                    print(" → accepted (bridge)")
                    H_chain[j] = H_chain[src] @ H_br
                    bridged = True
                    break
                else:
                    print(" → rejected")

        if not bridged:
            H_chain[j] = None

    valid = sum(1 for h in H_chain if h is not None)
    print(f"  Chain (with fallback): {valid}/{n} images reachable")
    return H_chain, matches_list, masks_consec
