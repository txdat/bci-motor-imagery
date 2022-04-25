import os
from typing import Tuple, Optional

import numpy as np
from scipy import stats
import torch
import torch.utils.data as td
from sklearn.utils.class_weight import compute_class_weight


class ImbalancedDataSampler(td.Sampler):
    def __init__(self, targets: np.ndarray):  # [len(data),]
        self.num_samples = len(targets)

        count = np.bincount(targets)
        self.weight = torch.tensor(1.0 / count[targets]).float()

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        return (
            int(i)
            for i in torch.multinomial(self.weight, self.num_samples, replacement=True)
        )


class EEGDataset(td.Dataset):
    STD_SCALE = 1.0 / 1.4628

    def __init__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        is_training: bool = True,
        use_augmentation: bool = True,
        use_normalization: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.X = X
        self.Y = Y or [-1] * len(X)

        self.is_training = is_training
        self.use_augmentation = use_augmentation
        self.use_normalization = use_normalization

        self.rng = np.random.RandomState(seed=seed)

    def __len__(self):
        return len(self.X)

    @property
    def class_weight(self) -> np.ndarray:
        return compute_class_weight("balanced", classes=np.unique(self.Y), y=self.Y)

    def augmentation(self, x):
        if self.rng.uniform() < 0.5:  # gaussian noise
            x += self.rng.normal(
                0.0, 0.2 * np.std(x, axis=-1, keepdims=True), size=x.shape
            )

        return x

    def normalization(self, x):
        mean = np.median(x, axis=-1, keepdims=True)
        std = np.expand_dims(
            stats.median_abs_deviation(x, axis=-1, scale=self.STD_SCALE),
            axis=-1,
        )
        # mean = np.mean(x, axis=-1, keepdims=True)
        # std = np.std(x, axis=-1, keepdims=True)

        x -= mean
        x /= std

        return x

    def __getitem__(self, item):
        x = self.X[item]
        y = self.Y[item]

        if self.use_augmentation:
            x = self.augmentation(x)

        if self.use_normalization:
            x = self.normalization(x)

        return x, y

    @staticmethod
    def batchify(batch):
        x, y = zip(*batch)

        x = torch.tensor(np.stack(x)).float()
        y = torch.tensor(np.array(y, dtype=int)).long()

        return x, y


def get_eeg_data_loader(
    eeg_data: Tuple[np.ndarray, Optional[np.ndarray]],
    is_training: bool = True,
    use_augmentation: bool = True,
    use_normalization: bool = True,
    seed: int = 42,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    use_imbalanced_sampler: bool = False,
    **kwargs
) -> Tuple[td.Dataset, td.DataLoader]:
    eeg_ds = EEGDataset(
        *eeg_data,
        is_training=is_training,
        use_augmentation=use_augmentation,
        use_normalization=use_normalization,
        seed=seed,
    )

    sampler = None
    if num_samples is not None:
        sampler = td.RandomSampler(eeg_ds, replacement=True, num_samples=num_samples)
    elif use_imbalanced_sampler:
        sampler = ImbalancedDataSampler(eeg_ds.Y)

    loader = td.DataLoader(
        eeg_ds,
        batch_size=batch_size,
        shuffle=is_training and sampler is None,
        sampler=sampler,
        num_workers=kwargs.pop("num_workers", os.cpu_count()),
        collate_fn=EEGDataset.batchify,
        pin_memory=True,
        **kwargs,
    )

    return eeg_ds, loader
