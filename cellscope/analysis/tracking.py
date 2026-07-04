"""Cell tracking across time.

Milestone 1 uses nearest-neighbour centroid linking between consecutive frames
with a maximum-distance gate, solved optimally per frame pair with the Hungarian
algorithm (``scipy.optimize.linear_sum_assignment``). Each cell receives a
persistent track ID that is stable across the whole time course.

Swap for btrack / TrackMate-style linking with division handling later; the
``track_centroids`` signature is the contract.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def track_centroids(
    centroids_per_frame: list[np.ndarray],
    max_distance: float = 30.0,
) -> tuple[list[np.ndarray], dict[int, np.ndarray]]:
    """Link detections across frames into persistent tracks.

    Parameters
    ----------
    centroids_per_frame : list (length T) of ``(k_t, 2)`` arrays of ``(y, x)``.
        Row ``i`` in frame ``t`` corresponds to label value ``i + 1`` in that
        frame's label image.
    max_distance : a detection in frame t+1 can only inherit a track from frame t
        if their centroids are within this many pixels.

    Returns
    -------
    assignment : list (length T) of int arrays. ``assignment[t][i]`` is the track
        ID of the detection at row ``i`` of frame ``t``.
    tracks : dict mapping track ID -> ``(n, 3)`` array of ``(frame, y, x)`` rows,
        ordered by frame.
    """
    n_frames = len(centroids_per_frame)
    assignment: list[np.ndarray] = [np.zeros(0, dtype=np.int32) for _ in range(n_frames)]
    tracks: dict[int, list[tuple[int, float, float]]] = {}
    if n_frames == 0:
        return assignment, {}

    next_track = 1

    def register(track_id: int, frame: int, y: float, x: float) -> None:
        tracks.setdefault(track_id, []).append((frame, float(y), float(x)))

    # Seed every detection in frame 0 with a fresh track.
    first = np.asarray(centroids_per_frame[0]).reshape(-1, 2)
    a0 = np.zeros(first.shape[0], dtype=np.int32)
    for i in range(first.shape[0]):
        a0[i] = next_track
        register(next_track, 0, first[i, 0], first[i, 1])
        next_track += 1
    assignment[0] = a0

    for t in range(1, n_frames):
        prev = np.asarray(centroids_per_frame[t - 1]).reshape(-1, 2)
        cur = np.asarray(centroids_per_frame[t]).reshape(-1, 2)
        a_prev = assignment[t - 1]
        a_cur = np.full(cur.shape[0], -1, dtype=np.int32)

        if cur.shape[0] and prev.shape[0]:
            dist = cdist(cur, prev)  # (n_cur, n_prev)
            cost = dist.copy()
            cost[dist > max_distance] = 1e6  # forbid links beyond the gate
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if dist[r, c] <= max_distance:
                    a_cur[r] = a_prev[c]

        for i in range(cur.shape[0]):
            if a_cur[i] < 0:
                a_cur[i] = next_track
                next_track += 1
            register(int(a_cur[i]), t, cur[i, 0], cur[i, 1])
        assignment[t] = a_cur

    tracks_arr = {tid: np.asarray(pts, dtype=np.float64) for tid, pts in tracks.items()}
    return assignment, tracks_arr
