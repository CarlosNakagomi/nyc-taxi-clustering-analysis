"""Extra metrics: Hopkins, Dunn, Meila’s Variation of Information (nats)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.cluster.hierarchy import fcluster, linkage


def hopkins_statistic(X: np.ndarray, m: int | None = None, random_state: int = 123) -> float:  # noqa: E501
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 3 or X.ndim != 2:
        return float("nan")
    m = int(m or min(500, max(20, n // 200)))
    m = min(m, n - 1)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, m, replace=False)
    U = np.empty(m, dtype=np.float64)
    for t, i in enumerate(idx):
        d = np.linalg.norm(X - X[i], axis=1)
        d[i] = np.inf
        U[t] = float(d.min())
    mins, maxs = X.min(0), X.max(0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    unif = rng.random((m, X.shape[1])) * span + mins
    dmat = pairwise_distances(unif, X, metric="euclidean")
    W = dmat.min(axis=1)
    su, sw = U.sum(), W.sum()
    if su + sw < 1e-12:
        return float("nan")
    return float(sw / (sw + su))


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    d = pairwise_distances(X, metric="euclidean")
    u = np.unique(labels[labels >= 0])
    clusters = [np.where(labels == a)[0] for a in u]
    if len(clusters) < 2:
        return 0.0
    d_min = np.inf
    for i in range(len(clusters) - 1):
        for j in range(i + 1, len(clusters)):
            a, b = clusters[i], clusters[j]
            d_min = min(d_min, d[np.ix_(a, b)].min())
    d_max = 0.0
    for a in clusters:
        if len(a) < 2:
            continue
        d_max = max(d_max, d[np.ix_(a, a)].max())
    if d_max < 1e-15 or not np.isfinite(d_min):
        return 0.0
    return float(d_min / d_max)


def meila_vi(a: np.ndarray, b: np.ndarray) -> float:
    """
    Meila’s Variation of Information = 2*H(AB) - H(A) - H(B) (nats, joint over label pairs).  # noqa: E501
    """
    a = np.asarray(a, dtype=int).ravel()
    b = np.asarray(b, dtype=int).ravel()
    n = len(a)
    if n < 1 or b.shape[0] != n:
        return 0.0
    ctab = _contingency2d(a, b) / n
    ha, hb = -np.nansum(ctab.sum(1) * _safe_log(ctab.sum(1))), -np.nansum(
        ctab.sum(0) * _safe_log(ctab.sum(0))  # noqa: E501
    )
    hab = -np.nansum(ctab * _safe_log(ctab))
    return float(2.0 * hab - ha - hb) if np.isfinite(hab) else 0.0


def _contingency2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = int(a.max() + 1), int(b.max() + 1)
    t = np.zeros((na, nb), dtype=np.float64)
    for i in range(len(a)):
        t[a[i], b[i]] += 1.0
    return t


def _safe_log(p: np.ndarray) -> np.ndarray:  # noqa: E501
    out = np.zeros_like(p, dtype=np.float64)
    m = p > 0
    out[m] = np.log(p[m])
    return out


def hclust_silhouette_k4(X: np.ndarray, random_state: int = 123, max_rows: int = 500) -> tuple[float, int]:  # noqa: E501
    rng = np.random.default_rng(random_state)
    n = min(X.shape[0], max_rows)
    idx = rng.choice(X.shape[0], n, replace=False)
    Xs = X[idx]
    z = linkage(Xs, method="ward", metric="euclidean")
    lab = fcluster(z, 4, criterion="maxclust")
    return float(silhouette_score(Xs, lab, metric="euclidean")), n
