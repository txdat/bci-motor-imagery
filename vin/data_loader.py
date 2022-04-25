import os
from typing import List, Union, Tuple, Optional

import numpy as np
from scipy import stats
import torch
import torch.utils.data as td
from braindecode.datasets import BaseConcatDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder


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


class SubjectSampler(td.Sampler):
    def __init__(
        self, subjects: np.ndarray, is_training: bool = True, batch_size: int = 1
    ):  # [len(data),]
        values, counts = np.unique(subjects, return_counts=True)
        counts = counts // batch_size + 1  # num of sampling times
        self.num_samples = counts.sum() * batch_size

        self.indices = {v: np.where(subjects == v)[0] for v in values}

        self.subject_indices = list()
        for i, v in enumerate(values):
            self.subject_indices.extend([v] * counts[i])
        self.subject_indices = np.array(self.subject_indices, dtype=int)

        self.is_training = is_training
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.is_training:
            np.random.shuffle(self.subject_indices)

            indices = list()
            for v in self.subject_indices:
                indices.extend(
                    np.random.choice(
                        self.indices[v], self.batch_size, replace=True
                    ).tolist()
                )

            return iter(indices)

        else:
            indices = list()
            indices_idx = {k: 0 for k in self.indices.keys()}
            for v in self.subject_indices:
                vi = list()
                while len(vi) < self.batch_size:
                    vi.append(self.indices[v][indices_idx[v]])
                    indices_idx[v] = (indices_idx[v] + 1) % len(self.indices[v])

                indices.extend(vi)

            return iter(indices)


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
        **kwargs,
    ):
        super().__init__()

        assert X.ndim == 3, f"invalid data shape {X.ndim} != 3"

        self.X = X  # [B,C,T]
        self.Y = Y  # [B,]

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
        y = self.Y[item] if self.Y is not None else -1

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


class MultipleLoader(object):
    def __init__(self, list_of_loader: List[td.DataLoader]):
        self.list_of_loader = list_of_loader
        self.list_of_loader_iter = [iter(loader) for loader in list_of_loader]

        self.n_steps = max([len(loader) for loader in list_of_loader])

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        for _ in range(self.n_steps):
            batch = list()
            for i, loader_iter in enumerate(self.list_of_loader_iter):
                try:
                    batch.append(loader_iter.next())

                except StopIteration:
                    self.list_of_loader_iter[i] = iter(self.list_of_loader[i])

                    batch.append(self.list_of_loader_iter[i].next())

            yield batch


def get_eeg_data_loader(
    eeg_data: Union[
        BaseConcatDataset, Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
    ],
    is_training: bool = True,
    use_augmentation: bool = True,
    use_normalization: bool = True,
    seed: int = 42,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    use_imbalanced_sampler: bool = False,
    use_subject_sampler: bool = False,
    **kwargs,
) -> Tuple[td.Dataset, td.DataLoader]:
    if isinstance(eeg_data, BaseConcatDataset):
        X = list()
        Y = list()
        S = list()
        for ds in eeg_data.datasets:
            x = ds.windows.get_data()
            X.append(x)
            Y.extend(ds.y)
            S.extend([ds.description["subject"]] * len(x))

        X = np.concatenate(X, axis=0)
        Y = np.array(Y, dtype=int)
        S = LabelEncoder().fit_transform(S)

    else:
        X, Y, S = eeg_data

    eeg_ds = EEGDataset(
        X,
        Y,
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
    elif use_subject_sampler and S is not None:
        sampler = SubjectSampler(S, is_training=is_training, batch_size=batch_size)

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
