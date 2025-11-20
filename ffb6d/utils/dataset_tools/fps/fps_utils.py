import numpy as np


def farthest_point_sampling(pts, sn, init_center=False):
    """Farthest point sampling implemented with NumPy.

    Args:
        pts (np.ndarray): Input points of shape (N, 3).
        sn (int): Number of samples to select.
        init_center (bool): If True, start from the point closest to the centroid.

    Returns:
        np.ndarray: Sampled points of shape (sn, 3).
    """

    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must have shape (N, 3)")

    pn = pts.shape[0]
    if pn == 0 or sn <= 0:
        return np.empty((0, 3), dtype=np.float32)

    sn = min(sn, pn)
    idxs = np.zeros(sn, dtype=np.int64)

    if init_center:
        center = pts.mean(axis=0)
        start_idx = np.linalg.norm(pts - center, axis=1).argmin()
    else:
        start_idx = np.random.randint(0, pn)

    idxs[0] = start_idx
    min_dist = np.full(pn, np.inf, dtype=np.float32)

    for i in range(1, sn):
        diff = pts - pts[idxs[i - 1]]
        dist = np.sum(diff * diff, axis=1)
        min_dist = np.minimum(min_dist, dist)
        idxs[i] = np.argmax(min_dist)

    return pts[idxs]
