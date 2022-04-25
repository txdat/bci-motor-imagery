import os
from itertools import combinations, product
from typing import List, Union, Tuple, Optional

import numpy as np
from scipy import stats
import torch
import torch.utils.data as td
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader as GraphDataLoader
from braindecode.datasets import BaseConcatDataset
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


class GraphEEGDataset(td.Dataset):
    STD_SCALE = 1.0 / 1.4628

    def __init__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        is_training: bool = True,
        use_augmentation: bool = True,
        use_normalization: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        assert X.ndim == 3, f"invalid data shape {X.ndim} != 3"

        self.X = X  # [B,C,T]
        self.Y = Y  # [B,]

        # TODO: implement topk edges
        self.top_k = (
            (top_k + 1) if top_k is not None else None
        )  # ignore self-connection

        self.n_channels = X.shape[1]

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

    def compute_edge_weight(self, x):  # w >= 0
        # pearson correlation coefficients
        x = x - x.mean(axis=-1, keepdims=True)
        x = (x @ np.swapaxes(x, -1, -2)) / (x.shape[-1] - 1)  # [B,C,C]

        diag = np.expand_dims(np.diagonal(x, axis1=-1, axis2=-2), axis=-1)  # [B,C,1]
        x /= np.sqrt(diag @ np.swapaxes(diag, -1, -2))  # [B,C,C] \in [-1, 1]
        x /= 2.0
        x += 0.5

        return x

    def __getitem__(self, item):
        x = self.X[item]
        y = self.Y[item] if self.Y is not None else -1

        if self.use_augmentation:
            x = self.augmentation(x)

        if self.use_normalization:
            x = self.normalization(x)

        edge_weight = self.compute_edge_weight(x)  # [C,C]
        if self.top_k is not None:
            indices = edge_weight.argsort(axis=1)[:, -self.top_k :]
            edge_index = set()
            for i, j in product(range(self.n_channels), range(self.top_k)):
                if (i, indices[i, j]) in edge_index:
                    continue
                edge_index.add((i, indices[i, j]))
                if indices[i, j] != i:
                    edge_index.add((indices[i, j], i))
            edge_index = np.array(list(edge_index), dtype=int).T
            edge_weight = edge_weight[edge_index[0], edge_index[1]]  # [nE]

        else:
            # without self-connection
            # edge_index = np.array(
            #     list(combinations(range(self.n_channels), r=2)), dtype=int
            # )
            # edge_index = np.hstack((edge_index, edge_index[:, ::-1])).T  # [2,nE]
            # with self-connection
            edge_index = np.array(
                list(product(range(self.n_channels), range(self.n_channels))), dtype=int
            ).T  # [2,nE]
            edge_weight = edge_weight[edge_index[0], edge_index[1]]  # [nE,]

        return GraphData(
            x=torch.tensor(x).float(),
            edge_index=torch.tensor(edge_index).long(),
            y=torch.tensor([y]).long(),
            edge_weight=torch.tensor(edge_weight).float(),
        )


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


def get_graph_eeg_data_loader(
    eeg_data: Union[BaseConcatDataset, Tuple[np.ndarray, Optional[np.ndarray]]],
    top_k: Optional[int] = None,
    is_training: bool = True,
    use_augmentation: bool = True,
    use_normalization: bool = True,
    seed: int = 42,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    use_imbalanced_sampler: bool = False,
    **kwargs,
) -> Tuple[td.Dataset, td.DataLoader]:
    if isinstance(eeg_data, BaseConcatDataset):
        X = list()
        Y = list()
        for ds in eeg_data.datasets:
            X.append(ds.windows.get_data())
            Y.extend(ds.y)

        X = np.concatenate(X, axis=0)
        Y = np.array(Y, dtype=int)

    else:
        X, Y = eeg_data

    eeg_ds = GraphEEGDataset(
        X,
        Y,
        top_k=top_k,
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

    loader = GraphDataLoader(
        eeg_ds,
        batch_size=batch_size,
        shuffle=is_training and sampler is None,
        sampler=sampler,
        num_workers=kwargs.pop("num_workers", os.cpu_count()),
        pin_memory=True,
        **kwargs,
    )

    return eeg_ds, loader
