import random
import math
import numpy as np
from scipy.interpolate import CubicSpline


def jittering(x, sigma=0.1):
    return x + np.random.normal(0, sigma, size=x.shape)


def scaling(x, sigma=0.1):
    return x * np.random.normal(1, sigma, size=(x.shape[0], 1))


def rotation(x):
    ch, _ = x.shape
    ch_inds = np.arange(ch)
    np.random.shuffle(ch_inds)

    return x[ch_inds] * np.random.choice([-1, 1], size=(ch, 1))


def permutation(x, num_segs, equal_segs=False):
    _, tlen = x.shape
    if equal_segs:
        seg_inds = np.linspace(0, tlen, num_segs + 1).astype(int)

    else:
        seg_inds = np.zeros((num_segs + 1,), dtype=int)
        seg_inds[-1] = tlen
        inds = np.random.randint(1, tlen, size=(num_segs - 1))
        seg_inds[1:-1] = np.sort(inds)

    perm_inds = np.arange(num_segs)
    np.random.shuffle(perm_inds)

    x = np.concatenate([x[:, seg_inds[i] : seg_inds[i + 1]] for i in perm_inds], axis=1)

    return x


def magnitude_warping(x, sigma=0.2, knot=4):
    ch, tlen = x.shape
    steps = np.arange(tlen)

    warp_steps = np.linspace(0, tlen - 1, knot + 2)

    for i in range(ch):
        s = CubicSpline(warp_steps, np.random.normal(1, sigma, size=warp_steps.shape))(
            steps
        )
        x[i] *= s

    return x


def time_warping(x, sigma=0.2, knot=4):
    ch, tlen = x.shape
    steps = np.arange(tlen)

    warp_steps = np.linspace(0, tlen - 1, knot + 2)

    for i in range(ch):
        t = CubicSpline(
            warp_steps, warp_steps * np.random.normal(1, sigma, size=warp_steps.shape)
        )(steps)
        t *= (tlen - 1) / t[-1]
        x[i] = np.interp(steps, t, x[i])

    return x


def window_slicing(x, reduce_ratio=0.9):
    ch, tlen = x.shape
    target_len = math.ceil(reduce_ratio * tlen)

    start = random.randint(0, tlen - target_len)
    end = start + target_len

    for i in range(ch):
        x[i] = np.interp(
            np.linspace(0, target_len, tlen), np.arange(target_len), x[i, start:end]
        )

    return x


def window_warping(x, window_ratio, scale):
    ch, tlen = x.shape
    window_len = math.ceil(window_ratio * tlen)

    window_start = random.randint(0, tlen - window_len)
    window_end = window_start + window_len

    for i in range(ch):
        fseg = x[i, :window_start]
        lseg = x[i, window_end:]
        wseg = x[i, window_start:window_end]
        wseg = np.interp(
            np.linspace(0, window_len - 1, int(scale * window_len)),
            np.arange(window_len),
            wseg,
        )
        seg = np.concatenate((fseg, wseg, lseg), axis=0)
        x[i] = np.interp(np.arange(tlen), np.linspace(0, tlen - 1, len(seg)), seg)

    return x
