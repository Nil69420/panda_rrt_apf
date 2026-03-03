"""Numba JIT-accelerated kernels for performance-critical inner loops.

Provides compiled implementations of brute-force nearest-neighbour search,
path-length computation, smoothness-cost evaluation, and RRT* ball-radius
near-neighbour search.  Falls back to equivalent pure-NumPy versions when
Numba is not installed.

When Numba **is** available the ``parallel=True`` variants use ``prange``
to distribute independent loop iterations across CPU threads, which
helps on trees larger than a few hundred nodes.

Usage::

    from panda_rrt.computations.jit_kernels import nearest_node_idx, path_length
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False
    prange = range                       # fallback: plain range

    def njit(*args, **kwargs):
        """Identity decorator when Numba is unavailable."""
        if args and callable(args[0]):
            return args[0]

        def _wrap(fn):
            return fn
        return _wrap


# ── Nearest-neighbour search ───────────────────────


@njit(cache=True)
def nearest_node_idx(nodes_q, n_nodes, q):
    """Return the index in *nodes_q[0:n_nodes]* closest to *q* (L2).

    Runs as a compiled loop when Numba is available, giving a 10--50x
    speed-up over the equivalent Python list-comprehension.

    Parameters
    ----------
    nodes_q : ndarray, shape (capacity, dof)
        Pre-allocated matrix of node configurations.
    n_nodes : int
        Number of valid (occupied) rows.
    q : ndarray, shape (dof,)
        Query configuration.

    Returns
    -------
    int
        Row index of the closest node.
    """
    min_dist_sq = np.inf
    min_idx = 0
    dof = q.shape[0]
    for i in range(n_nodes):
        d_sq = 0.0
        for j in range(dof):
            diff = nodes_q[i, j] - q[j]
            d_sq += diff * diff
        if d_sq < min_dist_sq:
            min_dist_sq = d_sq
            min_idx = i
    return min_idx


# ── Ball-radius near-neighbour search (RRT*) ──────


@njit(parallel=True, cache=True)
def _near_flags(nodes_q, n_nodes, q, radius_sq):
    """Boolean mask: ``True`` for nodes within *sqrt(radius_sq)* of *q*.

    Uses ``prange`` so each row is checked on a separate thread.
    """
    dof = q.shape[0]
    flags = np.zeros(n_nodes, dtype=np.bool_)
    for i in prange(n_nodes):
        d_sq = 0.0
        for j in range(dof):
            diff = nodes_q[i, j] - q[j]
            d_sq += diff * diff
        if d_sq <= radius_sq:
            flags[i] = True
    return flags


def near_indices(nodes_q, n_nodes, q, radius):
    """Return a 1-D int array of node indices within *radius* of *q*.

    Delegates to a parallel JIT kernel for the distance check and
    then extracts matching indices with ``np.where``.

    Parameters
    ----------
    nodes_q : ndarray, shape (capacity, dof)
    n_nodes : int
    q : ndarray, shape (dof,)
    radius : float

    Returns
    -------
    ndarray of int
    """
    flags = _near_flags(nodes_q, n_nodes, q, radius * radius)
    return np.where(flags)[0]


# ── Squared-distance utility ──────────────────────


@njit(cache=True)
def dist_sq(a, b):
    """Squared Euclidean distance between two 1-D arrays."""
    total = 0.0
    for j in range(a.shape[0]):
        diff = a[j] - b[j]
        total += diff * diff
    return total


# ── Path metrics ───────────────────────────────────


@njit(cache=True)
def path_length(path):
    r"""Total Euclidean arc-length of a joint-space path.

    .. math::
        L = \sum_{i=0}^{n-2} \|q_{i+1} - q_i\|_2

    Parameters
    ----------
    path : ndarray, shape (n, dof)

    Returns
    -------
    float
    """
    total = 0.0
    n = path.shape[0]
    dof = path.shape[1]
    for i in range(n - 1):
        d_sq = 0.0
        for j in range(dof):
            diff = path[i + 1, j] - path[i, j]
            d_sq += diff * diff
        total += np.sqrt(d_sq)
    return total


@njit(cache=True)
def smoothness_cost(path):
    r"""Finite-difference acceleration energy of a path.

    .. math::
        J_{\text{smooth}} = \sum_{i=1}^{n-2}
            \|q_{i+1} - 2\,q_i + q_{i-1}\|^2

    Parameters
    ----------
    path : ndarray, shape (n, dof)

    Returns
    -------
    float
    """
    total = 0.0
    n = path.shape[0]
    dof = path.shape[1]
    for i in range(1, n - 1):
        for j in range(dof):
            accel = path[i + 1, j] - 2.0 * path[i, j] + path[i - 1, j]
            total += accel * accel
    return total


# ── Batch smoothness gradient (optimizer) ──────────


@njit(parallel=True, cache=True)
def smooth_gradient_all(path):
    r"""Smoothness gradient for every interior waypoint of *path*.

    For interior index *i* (1 <= i <= n-2) the gradient is:

    .. math::
        \nabla_i J = 2\,a_{i-1} - 4\,a_i + 2\,a_{i+1}

    where :math:`a_k = q_{k+1} - 2q_k + q_{k-1}`.  Boundary terms
    (:math:`a_{i-2}` or :math:`a_{i+2}` outside the path) are clamped
    to zero.

    Uses ``prange`` so each interior waypoint is computed on a separate
    thread.

    Parameters
    ----------
    path : ndarray, shape (n, dof)
        Waypoint matrix.

    Returns
    -------
    ndarray, shape (n - 2, dof)
        One gradient row per interior waypoint.
    """
    n = path.shape[0]
    dof = path.shape[1]
    grads = np.zeros((n - 2, dof))
    for idx in prange(n - 2):
        i = idx + 1                       # actual path index
        for j in range(dof):
            a_i = path[i + 1, j] - 2.0 * path[i, j] + path[i - 1, j]
            grad_j = -4.0 * a_i
            if i >= 2:
                a_prev = path[i, j] - 2.0 * path[i - 1, j] + path[i - 2, j]
                grad_j += 2.0 * a_prev
            if i <= n - 3:
                a_next = path[i + 2, j] - 2.0 * path[i + 1, j] + path[i, j]
                grad_j += 2.0 * a_next
            grads[idx, j] = grad_j
    return grads


# ── C-space APF attractive gradient ───────────────


@njit(cache=True)
def attractive_gradient_jit(q, q_goal, k_att, d_thresh):
    """Piecewise attractive gradient (quadratic inside, conic outside).

    Parameters
    ----------
    q, q_goal : ndarray, shape (dof,)
    k_att : float
    d_thresh : float

    Returns
    -------
    ndarray, shape (dof,)
    """
    dof = q.shape[0]
    d_sq = 0.0
    for j in range(dof):
        diff = q[j] - q_goal[j]
        d_sq += diff * diff
    d = np.sqrt(d_sq)
    if d < 1e-12:
        return np.zeros(dof)
    out = np.empty(dof)
    if d <= d_thresh:
        for j in range(dof):
            out[j] = k_att * (q[j] - q_goal[j])
    else:
        scale = k_att * d_thresh / d
        for j in range(dof):
            out[j] = scale * (q[j] - q_goal[j])
    return out


# ── Steering ───────────────────────────────────────


@njit(cache=True)
def steer(q_near, q_target, step_size):
    """Move from *q_near* toward *q_target*, capped at *step_size*.

    Parameters
    ----------
    q_near, q_target : ndarray, shape (dof,)
    step_size : float

    Returns
    -------
    ndarray, shape (dof,)
    """
    diff = q_target - q_near
    d_sq = 0.0
    for j in range(diff.shape[0]):
        d_sq += diff[j] * diff[j]
    dist = np.sqrt(d_sq)
    if dist < step_size:
        return q_target.copy()
    return q_near + (diff / dist) * step_size


@njit(cache=True)
def in_limits(q, lower, upper):
    """Check whether every element of *q* lies in [lower, upper]."""
    for j in range(q.shape[0]):
        if q[j] < lower[j] or q[j] > upper[j]:
            return False
    return True
