from typing import Type, Optional

import numpy as np
import torch
import torch.utils.data as td
from braindecode.classifier import EEGClassifier
from braindecode.augmentation import Mixup


class ClassBasedMixup(Mixup):
    def get_params(self, *batch):
        X, y = batch[:2]
        device = X.device
        batch_size, _, _ = X.shape

        if self.alpha > 0:
            if self.beta_per_sample:
                lam = torch.as_tensor(
                    self.rng.beta(self.alpha, self.alpha, batch_size)
                ).to(device)
            else:
                lam = torch.ones(batch_size).to(device)
                lam *= self.rng.beta(self.alpha, self.alpha)
        else:
            lam = torch.ones(batch_size).to(device)

        # do permutation within class
        idx_perm = torch.arange(batch_size)
        for c in y.unique():
            c_idx = torch.where(y == c)[0]
            c_perm = torch.randperm(c_idx.size(0))
            idx_perm[c_idx] = c_idx[c_perm]

        return {
            "lam": lam,
            "idx_perm": idx_perm,
        }


class BalancedMixupDataLoader(object):
    def __init__(
        self,
        dataset: td.Dataset,
        loader_cls: Type[td.DataLoader],
        *args,
        instance_sampler: Optional[td.Sampler] = None,
        class_sampler: Optional[td.Sampler] = None,
        alpha: float = 0.1,
        beta_per_sample: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        kwargs.pop("shuffle", None)
        self.instance_loader = loader_cls(
            dataset,
            *args,
            sampler=instance_sampler,
            shuffle=instance_sampler is None,
            **kwargs,
        )
        self.class_loader = loader_cls(
            dataset,
            *args,
            sampler=class_sampler,
            shuffle=class_sampler is None,
            **kwargs,
        )

        self.instance_loader_iter = iter(self.instance_loader)
        self.class_loader_iter = iter(self.class_loader)

        self.alpha = alpha
        self.beta_per_sample = beta_per_sample
        self.rng = np.random.RandomState(seed=seed)

    def __len__(self):
        return min(len(self.instance_loader), len(self.class_loader))

    def __iter__(self):
        for _ in range(len(self)):
            xi, yi, _ = self.instance_loader_iter.next()  # return indexes?
            xc, yc, _ = self.class_loader_iter.next()

            bsz = xi.size(0)
            device = xi.device

            if self.beta_per_sample:
                lam = (
                    torch.tensor(self.rng.beta(self.alpha, 1.0, bsz)).float().to(device)
                )
            else:
                lam = torch.ones(bsz).float().to(device) * self.rng.beta(
                    self.alpha, 1.0
                )

            _lam = lam.reshape(-1, 1, 1)

            # https://github.com/agaldran/balanced_mixup/blob/main/train_lt_mxp.py#L137
            x = _lam * xc + (1.0 - _lam) * xi

            yield x, (yc, yi, lam)


class ThrowAwayIndexLoader(object):
    def __init__(self, net, loader, is_regression):
        self.net = net
        self.loader = loader
        self.last_i = None
        self.is_regression = is_regression

    def __iter__(
        self,
    ):
        normal_iter = self.loader.__iter__()
        for batch in normal_iter:
            if len(batch) == 3:
                x, y, i = batch
                # Store for scoring callbacks
                self.net._last_window_inds_ = i
            else:
                x, y = batch

            yield x, y


class MixupEEGClassifier(EEGClassifier):
    def get_iterator(self, dataset, training=False, drop_index=True):
        iterator = super(EEGClassifier, self).get_iterator(dataset, training=training)
        if drop_index:
            return ThrowAwayIndexLoader(self, iterator, is_regression=False)
        else:
            return iterator
