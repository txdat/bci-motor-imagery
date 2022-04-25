import numpy as np
from scipy import stats


def np_standardize(x: np.ndarray, robust: bool = False) -> np.ndarray:
    if robust:
        mean = np.median(x, axis=-1, keepdims=True)
        std = np.expand_dims(
            stats.median_abs_deviation(x, axis=-1, scale=1.0 / 1.4628),
            axis=-1,
        )
    else:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)

    x -= mean
    x /= std

    return x


def np_covariance(x: np.ndarray) -> np.ndarray:
    x -= x.mean(axis=-1, keepdims=True)
    x = (x @ np.swapaxes(x, -1, -2)) / (x.shape[-1] - 1)

    return x


def np_correlation(x: np.ndarray) -> np.ndarray:
    x = np_covariance(x)

    diag = np.expand_dims(np.diagonal(x, axis1=-1, axis2=-2), axis=-1)
    x /= np.sqrt(diag @ np.swapaxes(diag, -1, -2))

    return x
