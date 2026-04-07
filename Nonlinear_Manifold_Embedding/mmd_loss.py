"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
def mmd_loss(x, y, bandwidths=np.array([0.01, 0.1, 1.0, 2.0, 5.0, 10.0])):
    """
    Unbiased MMD loss with multiple RBF kernels (logsumexp aggregation).
    
    Parameters
    ----------
    x : np.ndarray, shape (n, d)
    y : np.ndarray, shape (m, d)
    ls : np.ndarray or None
        Optional 1D array of kernel coefficients gamma.
        If None, they are computed from the pooled pairwise distances.

    Returns
    -------
    float
        Scalar MMD loss value.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n, m = x.shape[0], y.shape[0]

    if n < 2 or m < 2:
        raise ValueError("x and y must each contain at least 2 samples.")

    # Compute ls if not provided
    length_scales = 1.0 / bandwidths**2 

    # Gram matrices
    xx = x @ x.T
    yy = y @ y.T
    xy = x @ y.T

    rx = np.diag(xx)[None, :]   # (1, n)
    ry = np.diag(yy)[None, :]   # (1, m)

    dist_xx = np.maximum(rx.T + rx - 2.0 * xx, 0.0)   # (n, n)
    dist_yy = np.maximum(ry.T + ry - 2.0 * yy, 0.0)   # (m, m)
    dist_xy = np.maximum(rx.T + ry - 2.0 * xy, 0.0)   # (n, m)

    gamma = 0.5 * length_scales[:, None, None]                         # (S, 1, 1)

    k_xx = np.exp(-gamma * dist_xx[None, :, :])       # (S, n, n)
    k_yy = np.exp(-gamma * dist_yy[None, :, :])       # (S, m, m)
    k_xy = np.exp(-gamma * dist_xy[None, :, :])       # (S, n, m)

    # Remove diagonals for unbiased estimates
    k_xx_diag = np.trace(k_xx, axis1=1, axis2=2)
    k_yy_diag = np.trace(k_yy, axis1=1, axis2=2)

    k_xx_sum = (k_xx.sum(axis=(1, 2)) - k_xx_diag) / (n * (n - 1))
    k_yy_sum = (k_yy.sum(axis=(1, 2)) - k_yy_diag) / (m * (m - 1))
    k_xy_mean = k_xy.mean(axis=(1, 2))

    K = (k_xx_sum + k_yy_sum - 2.0 * k_xy_mean) #* 1e3  # (S,)

    # Stable logsumexp
    K_max = np.max(K)
    return K_max + np.log(np.sum(np.exp(K - K_max)))
