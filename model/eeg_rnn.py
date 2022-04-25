from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WSConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm: Optional[float] = None, **kwargs):
        self.max_norm = max_norm
        super(WSConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        std, mean = torch.std_mean(self.weight.data, dim=(1, 2, 3), keepdim=True)
        self.weight.data = (self.weight.data - mean) / (std + 1e-5)
        # self.weight.data = self.weight.data - mean

        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

        return super(WSConv2d, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class TemporalNorm(nn.Module):
    def __init__(self, in_chans, F1):
        super().__init__()

        # self.norm = BCNorm(
        #     in_chans * F1, num_groups=in_chans, momentum=0.01, estimate=True
        # )
        # self.norm = nn.BatchNorm2d(in_chans * F1, momentum=0.01, affine=True, eps=1e-3)
        self.norm = nn.GroupNorm(num_groups=in_chans, num_channels=in_chans * F1)
        # self.norm = nn.InstanceNorm2d(in_chans * F1, momentum=0.01, affine=True, eps=1e-3)

        # self.norm = nn.GroupNorm(num_groups=in_chans, num_channels=in_chans)

        self.in_chans = in_chans
        self.F1 = F1

    def forward(self, x):  # [B,F1,C,T]
        x = (
            x.permute(0, 2, 1, 3)
            .reshape(x.size(0), self.in_chans * self.F1, 1, -1)
            .contiguous()
        )
        x = self.norm(x)
        x = (
            x.reshape(x.size(0), self.in_chans, self.F1, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # x = x.permute(0, 2, 1, 3).contiguous()
        # x = self.norm(x)
        # x = x.permute(0, 2, 1, 3).contiguous()

        return x


class ConvTS(nn.Sequential):
    def __init__(
        self,
        in_chans,
        in_depth,
        F1=8,
        D=2,
        kernel_length=64,
        pool_mode="mean",
        drop_prob=0.25,
    ):
        super().__init__()

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.add_module(
            "conv_temporal",
            # nn.Conv2d(
            WSConv2d(
                in_depth,
                F1,
                kernel_size=(1, kernel_length),
                stride=(1, 1),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
        )
        self.add_module(
            "bnorm_temporal",
            # nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            # nn.InstanceNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            # nn.GroupNorm(num_groups=8, num_channels=F1),
            # BatchInstanceNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            # BCNorm(F1, num_groups=8, momentum=0.01, eps=1e-5, estimate=True),
            TemporalNorm(in_chans, F1),
        )
        self.add_module(
            "conv_spatial",
            # Conv2dWithConstraint(
            WSConv2d(
                F1,
                F1 * D,
                max_norm=1.0,
                kernel_size=(in_chans, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=F1,
                bias=False,
            ),
        )
        self.add_module(
            "bnorm_1",
            # nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            # nn.InstanceNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.GroupNorm(num_groups=8, num_channels=F1 * D),
            # BatchInstanceNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            # BCNorm(F1 * D, num_groups=8, momentum=0.01, eps=1e-5, estimate=True),
        )
        # self.add_module("elu_1", nn.ELU(inplace=True))
        self.add_module("elu_1", nn.Mish())
        self.add_module("pool_1", pool_class(kernel_size=(1, 4), stride=(1, 4)))
        self.add_module("drop_1", nn.Dropout(p=drop_prob))


class MB_EEGRnn(nn.Module):
    def __init__(
        self,
        n_classes,
        in_chans,
        in_samples,
        in_depth=1,
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        num_layers=1,
        kernel_lengths=(64,),
        drop_prob=0.5,
    ):
        super().__init__()

        self.conv_1s = nn.ModuleList()
        for kernel_length in kernel_lengths:
            self.conv_1s.append(
                ConvTS(
                    in_chans,
                    in_depth,
                    F1,
                    D,
                    kernel_length,
                    pool_mode,
                    drop_prob,
                )
            )

        self.gru = nn.GRU(
            F1 * D,
            F2,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            dropout=drop_prob,
            bidirectional=True,
        )

    def forward(self, x):  # x: [B,C,T]
        if x.ndim == 3:
            x = x.unsqueeze(1)  # x: [B,1,C,T]

        x = torch.stack([conv(x) for conv in self.conv_1s]).sum(
            dim=0
        )  # x: [B,F1*D,1,T//*]
        x = x.squeeze(2)

        x = x.permute(2, 0, 1)  # [L,B,F1*D]
        x, h = self.gru(x)  # x: [L,B,F2*2], h: [2*n_layers,B,F2]

        # attention
        # get forward/backward of last layer's hidden
        h = torch.cat((h[-2], h[-1]), dim=1)  # [B,F2*2]
        h = h.unsqueeze(2)

        x = x.permute(1, 0, 2)
        a = torch.matmul(x, h)  # a: [B,L,1]
        a = F.softmax(a, dim=1)

        x = x * a  # x: [B,L,F2*2]
        x = x.sum(dim=1)  # [B,F2*2]

        return x


if __name__ == "__main__":
    model = MB_EEGRnn(
        n_classes=5,
        in_chans=32,
        in_samples=256,
        in_depth=1,
        pool_mode="mean",
        F1=8,
        D=4,
        F2=32,
        num_layers=1,
    )

    print(model)
