import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class ConvT(nn.Sequential):
    def __init__(
        self,
        in_chans,
        F1=8,
        kernel_length=32,
        pool_mode="mean",
        drop_prob=0.5,
    ):
        super().__init__()

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.add_module(
            "conv_dw",
            nn.Conv2d(
                in_chans,
                in_chans,
                kernel_size=(1, kernel_length),
                stride=(1, 1),
                padding=(0, kernel_length // 2),
                groups=in_chans,
                bias=False,
            ),
        )
        self.add_module(
            "conv_pw",
            nn.Conv2d(
                in_chans,
                in_chans * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=in_chans,
                bias=False,
            ),
        )
        self.add_module("bnorm", nn.BatchNorm2d(in_chans * F1))
        self.add_module("elu", nn.ELU(inplace=True))
        self.add_module("pool", pool_class(kernel_size=(1, 4), stride=(1, 4)))
        self.add_module("drop", nn.Dropout(p=drop_prob))


class MB_EEGNet_GNN(nn.Module):
    def __init__(
        self,
        n_classes,
        in_chans,
        in_samples,
        in_depth=1,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        F3=16,
        kernel_lengths=(32,),
        third_kernel_size=(8, 4),
        drop_prob=0.5,
    ):
        super().__init__()

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        # CNN
        self.conv_1s = nn.ModuleList()
        for kernel_length in kernel_lengths:
            self.conv_1s.append(
                ConvT(
                    in_chans,
                    F1,
                    kernel_length,
                    pool_mode,
                    drop_prob,
                )
            )

        self.conv_2 = nn.Sequential()
        self.conv_2.add_module(
            "conv_dw",
            nn.Conv2d(
                in_chans * F1,
                in_chans * F1,
                kernel_size=(1, 16),
                stride=(1, 1),
                padding=(0, 8),
                groups=in_chans * F1,
                bias=False,
            ),
        )
        self.conv_2.add_module(
            "conv_pw",
            nn.Conv2d(
                in_chans * F1,
                in_chans * F2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=in_chans,
                bias=False,
            ),
        )
        self.conv_2.add_module("bnorm", nn.BatchNorm2d(in_chans * F2))
        self.conv_2.add_module("elu", nn.ELU(inplace=True))
        self.conv_2.add_module("pool", pool_class(kernel_size=(1, 8), stride=(1, 8)))
        self.conv_2.add_module("drop", nn.Dropout(p=drop_prob))

        with torch.no_grad():
            x = torch.zeros((1, in_chans, 1, in_samples), dtype=torch.float32)
            x = torch.stack([conv(x) for conv in self.conv_1s]).sum(dim=0)
            x = self.conv_2(x)
            x = x.cpu().data.numpy()

        self.conv_3 = nn.Sequential()
        self.conv_3.add_module(
            "conv_dw",
            nn.Conv2d(
                in_chans * F2,
                in_chans * F2,
                kernel_size=(1, x.shape[3]),
                stride=(1, 1),
                padding=(0, 0),
                groups=in_chans * F2,
                bias=False,
            ),
        )
        self.conv_3.add_module(
            "conv_pw",
            nn.Conv2d(
                in_chans * F2,
                in_chans * F3,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=in_chans,
                bias=False,
            ),
        )
        self.conv_3.add_module("bnorm", nn.BatchNorm2d(in_chans * F3))
        self.conv_3.add_module("elu", nn.ELU(inplace=True))

        # Classifier
        # self.clf = nn.Linear(F3, n_classes, bias=True)

        self.in_chans = in_chans
        self.in_samples = in_samples
        self.F3 = F3

    def forward(self, x):
        """
        x: [B*C,T]
        """
        x = x.reshape(-1, self.in_chans, 1, self.in_samples).contiguous()
        x = torch.stack(([conv(x) for conv in self.conv_1s])).sum(dim=0)
        x = self.conv_2(x)
        x = self.conv_3(x)  # [B,C*F3,1,1]
        x = x.reshape(-1, self.in_chans, self.F3).reshape(-1, self.F3).contiguous()

        # Classifier
        # x = self.clf(x)

        return x


class GraphEEGClassifier(nn.Module):
    def __init__(self, F3, n_classes, drop_prob=0.5):
        super().__init__()

        # GCN
        self.gconv_1 = gnn.Sequential(
            "x, edge_index, edge_weight",
            [
                (nn.Dropout(p=drop_prob), "x -> x"),
                (
                    gnn.GCNConv(
                        F3, F3, add_self_loops=False, normalize=True, bias=False
                    ),
                    "x, edge_index, edge_weight -> x",
                    # gnn.GATv2Conv(
                    #     F3,
                    #     F3,
                    #     heads=1,
                    #     dropout=0.0,
                    #     add_self_loops=False,
                    #     bias=False,
                    # ),
                    # "x, edge_index -> x",
                ),
                nn.BatchNorm1d(F3),
                nn.ELU(inplace=True),
            ],
        )

        self.gconv_2 = gnn.Sequential(
            "x, edge_index, edge_weight",
            [
                (nn.Dropout(p=drop_prob), "x -> x"),
                (
                    gnn.GCNConv(
                        F3, F3, add_self_loops=False, normalize=True, bias=False
                    ),
                    "x, edge_index, edge_weight -> x",
                    # gnn.GATv2Conv(
                    #     F3,
                    #     F3,
                    #     heads=1,
                    #     dropout=0.0,
                    #     add_self_loops=False,
                    #     bias=False,
                    # ),
                    # "x, edge_index -> x",
                ),
                nn.BatchNorm1d(F3),
                nn.ELU(inplace=True),
            ],
        )

        # Classifier
        self.drop = nn.Dropout(p=drop_prob)

        self.ln = nn.Linear(F3, n_classes, bias=True)

    def forward(self, x, edge_index, edge_weight, batch):
        # GCN
        x = self.gconv_1(x, edge_index, edge_weight)
        x = self.gconv_2(x, edge_index, edge_weight)
        x = gnn.global_mean_pool(x, batch)  # [B,F3]

        # Classifier
        x = self.drop(x)
        x = self.ln(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary
    from itertools import combinations
    import numpy as np
    import pickle as pkl
    import sys

    sys.path.append("../vin")
    from graph_data_loader import get_graph_eeg_data_loader  # type: ignore

    model = MB_EEGNet_GNN(
        n_classes=4,
        in_chans=28,
        in_samples=256,
        in_depth=1,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        F3=16,
        kernel_lengths=(32,),
        third_kernel_size=(8, 4),
        drop_prob=0.5,
    )
    clf = GraphEEGClassifier(16, 4, drop_prob=0.5)

    with open("/home/txdat/Downloads/PHY_1_tgt.pkl", mode="rb") as f:
        Xtrain_tgt, Ytrain_tgt, Xvalid_tgt, Yvalid_tgt, Xtest_tgt, Ytest_tgt = pkl.load(
            f
        )

    _, valid_tgt_ds_loader = get_graph_eeg_data_loader(
        eeg_data=(Xvalid_tgt, Yvalid_tgt),
        top_k=None,
        is_training=False,
        use_augmentation=False,
        use_normalization=True,
        batch_size=8,
        num_samples=None,
        use_imbalanced_sampler=False,
    )

    for batch in valid_tgt_ds_loader:
        x = model(batch.x)
        x = clf(x, batch.edge_index, batch.edge_weight, batch.batch)
        print(x.size())
        break
