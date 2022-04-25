# implement alignment methods for cross-subjects/sessions
from typing import List
import numpy as np
import scipy.linalg as linalg


def compute_transform_mat(
    X: np.ndarray, use_log: bool = True, inv: bool = True
) -> np.ndarray:
    """
    compute transform matrix (inv) sqrt of mean of covariances over trials

    Parameters
    ---------------------
    X   np.ndarray
        trials' data [bsz, channels, times]
    use_log     bool
        use log-euclidean metric or euclidean metric
    inv     bool
        compute inverse of matrix (R^-1)

    Return
    ---------------------
    R   np.ndarray
        transform matrix [1, channels, channels]
    """
    assert X.ndim == 3, f"invalid input's ndim {X.ndim} isn't equal 3"

    C = X @ X.transpose((0, 2, 1))  # [bsz, channels, channels]
    if use_log:
        C = linalg.expm(np.mean(np.stack([linalg.logm(Ci) for Ci in C], axis=0))
    else:
        C = np.mean(C, axis=0)

    if inv:
        R = linalg.inv(linalg.sqrtm(C))
        if np.iscomplexobj(R):
            R = np.real(R).astype(np.float32)
    else:
        R = linalg.sqrtm(C)

    return R[np.newaxis]


def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """
    align subject's trials data domain
    after transform, X @ X.T = I

    Parameters
    ---------------------
    X   np.ndarray
        trials' data [bsz, channels, times]

    Return
    ---------------------
    X   np.ndarray
        aligned trials' data
    """
    return compute_transform_mat(X) @ X


def compute_target_transform_mats(X: np.ndarray, Y: np.ndarray) -> List[np.ndarray]:
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
    Rs  list
        list of transform matrix
    """
    Rs = list()
    for i in np.unique(Y):
        idx = np.where(Y == i)[0]
        Rs.append(compute_transform_mat(X[idx], inv=False))

    return Rs


def label_alignment(
    X: np.ndarray, Y: np.ndarray, tgtRs: List[np.ndarray]
) -> np.ndarray:
    """
    align src subject's trials data domain to tgt subject's trials data domain FOR EACH CLASS
    after transform, Xc @ Xc.T = I

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
        X[idx] = tgtRs[i] @ compute_transform_mat(x) @ x

    return X
