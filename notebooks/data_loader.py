from typing import Optional

import numpy as np
import torch
import torch.utils.data as td
from braindecode.datasets import BaseConcatDataset


class EEGDataset(td.Dataset):
    def __init__(self, ds: BaseConcatDataset, split: str = "train"):
        super().__init__()

        self.X = list()
        self.Y = list()
        ds_info = ds.description
        for i in ds_info[ds_info["split"] == split].index:
            self.X.append(ds.datasets[i].windows.get_data())
            self.Y.append(np.array(ds.datasets[i].y, dtype=int))

        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    @staticmethod
    def batchify(batch):
        x, y = zip(*batch)
        x = torch.tensor(x).float()
        y = torch.tensor(y).long()

        return x, y


def eeg_data_loader(
    ds: BaseConcatDataset,
    split: str = "train",
    bsz: int = 1,
    is_training: bool = True,
    sampler: Optional[td.Sampler] = None,
    **kwargs
):
    return td.DataLoader(
        EEGDataset(ds, split),
        batch_size=bsz,
        shuffle=is_training and sampler is not None,
        sampler=sampler,
        collate_fn=EEGDataset.batchify,
        pin_memory=True,
        **kwargs,
    )
