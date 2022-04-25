# implement alignment methods for cross-subjects/sessions
from typing import Dict, Optional
import numpy as np
import scipy.linalg as linalg


def np_compute_transform_mat(
    X: np.ndarray, use_log: bool = False, inv: bool = True
) -> np.ndarray:
    """
    compute transform matrix (inv) sqrt of mean of covariances over trials

    Parameters
    ---------------------
    X   np.ndarray
        trials' data [bsz, channels, times]
    use_log     bool
        use log-euclidean metric or euclidean metric (default)
    inv     bool
        compute inverse of matrix (R^-1)

    Return
    ---------------------
    R   np.ndarray
        transform matrix [1, channels, channels]
    """
    assert X.ndim == 3, f"invalid input's ndim {X.ndim} isn't equal 3"

    Xm = X - X.mean(axis=2, keepdims=True)
    C = Xm @ Xm.transpose((0, 2, 1)) / (Xm.shape[2] - 1)  # [bsz, channels, channels]
    if use_log:  # log-euclid metric
        Cm = linalg.expm(np.mean(np.stack([linalg.logm(Ci) for Ci in C]), axis=0))
    else:  # euclid metric
        Cm = np.mean(C, axis=0)

    R = linalg.sqrtm(Cm)
    if inv:
        R = linalg.inv(R)
        if np.iscomplexobj(R):
            R = np.real(R).astype(np.float32)

    return R[np.newaxis]


def np_euclidean_alignment(X: np.ndarray, R: Optional[np.ndarray] = None) -> np.ndarray:
    """
    align subject's trials data domain
    after transform, X @ X.T = I

    Parameters
    ---------------------
    X   np.ndarray
        trials' data [bsz, channels, times]

    R   np.ndarray
        transform matrix [1, channels, channels]

    Return
    ---------------------
    X   np.ndarray
        aligned trials' data
    """
    if R is None:
        R = np_compute_transform_mat(X)

    return R @ X  # R_s^(-1/2) @ x


def np_compute_target_transform_mats(
    X: np.ndarray, Y: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    compute target domain's transform matrices for each class

    Parameters
    ---------------------
    X   np.ndarray
        target's trials data [bsz, n_channels, times]
    Y   np.ndarray
        target's trials label [bsz,]

    Return
    ---------------------
    Rs  dict
        dict of each class' transform matrix
    """
    Rs = dict()
    for i in np.unique(Y):
        idx = np.where(Y == i)[0]
        Rs[i] = np_compute_transform_mat(X[idx], inv=False)

    return Rs


def np_label_alignment(
    X: np.ndarray, Y: np.ndarray, tgtRs: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    align src subject's trials data domain to tgt subject's trials data domain FOR EACH CLASS

    Parameters
    ---------------------
    X   np.ndarray
        trials' data [bsz, channels, times]
    Y   np.ndarray
        trials' label [bsz,]
    tgtRs   list
        list of transform matrix [1, channels, channels] for each class of tgt's domain
        for all i in Y, i < len(tgtRs)

    Return
    ---------------------
    X   np.ndarray
        aligned trials' data
    """
    for i in np.unique(Y):
        idx = np.where(Y == i)[0]
        x = X[idx]
        X[idx] = (
            tgtRs[i] @ np_compute_transform_mat(x) @ x
        )  # R_t^(1/2) @ R_s^(-1/2) @ x

    return X
