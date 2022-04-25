from typing import Tuple, List

import numpy as np
from scipy.signal import butter, filtfilt
from mne.decoding import CSP
from mne.filter import filter_data
from sklearn.base import BaseEstimator, TransformerMixin


def butter_bandpass_filter(x, low, high, sfreq, order):
    b, a = butter(order, [low, high], btype="bandpass", fs=sfreq)
    x = filtfilt(b, a, x)

    return x


class FBCSP(BaseEstimator, TransformerMixin):
    """
    based on https://github.com/IoBT-VISTEC/Decoding-EEG-during-AO-MI-ME/blob/master/pysitstand/model.py#L11
    """

    def __init__(
        self,
        filters: List[Tuple[int, ...]],
        filter_order: int,
        sfreq: float,
        **csp_params
    ):
        self.filters = filters
        self.filter_order = filter_order
        self.sfreq = sfreq

        self.filt_csp = [CSP(**csp_params) for _ in range(len(filters))]

    def fit(self, X: np.ndarray, y: np.ndarray):
        for i, (low, high) in enumerate(self.filters):
            # Xi = butter_bandpass_filter(X, low, high, self.sfreq, self.filter_order)
            Xi = filter_data(
                X, sfreq=self.sfreq, l_freq=low, h_freq=high, verbose=False
            )
            self.filt_csp[i].fit(X=Xi, y=y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xout = list()
        for i, (low, high) in enumerate(self.filters):
            # Xi = butter_bandpass_filter(X, low, high, self.sfreq, self.filter_order)
            Xi = filter_data(
                X, sfreq=self.sfreq, l_freq=low, h_freq=high, verbose=False
            )
            Xout.append(self.filt_csp[i].transform(X=Xi))  # [n_samples, n_components]

        Xout = np.stack(Xout).transpose((1, 2, 0)).reshape((X.shape[0], -1))

        return Xout


if __name__ == "__main__":
    fbcsp = FBCSP(
        filters=[(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30)],
        filter_order=2,
        sfreq=128.0,
        n_components=4,
        reg=None,
        log=True,
        norm_trace=False,
    )
