import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def __repr__(self):
        return f"{super().__repr__()}, max_norm={self.max_norm}"

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )  # [out,in,H,W]
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def __repr__(self):
        return f"{super().__repr__()}, max_norm={self.max_norm}"

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )  # [out,in]
        return super(LinearWithConstraint, self).forward(x)


def WNConv2d(*args, **kwargs):
    kwargs["bias"] = True

    # weight_norm initialization issue
    # https://github.com/pytorch/pytorch/issues/28594#issuecomment-1149882811
    module = weight_norm(nn.Conv2d(*args, **kwargs))
    module.weight = module.weight_v.detach()

    return module


def WNLinear(*args, **kwargs):
    kwargs["bias"] = True

    module = weight_norm(nn.Linear(*args, **kwargs))
    module.weight = module.weight_v.detach()

    return module


class TemporalSpatialConv(nn.Sequential):
    def __init__(
        self,
        in_chans,
        in_depth=1,
        F=8,
        D=2,
        kernel_length=64,
        drop_prob=0.5,
        bn_mm=0.01,
        bn_track=True,
    ):
        modules = [
            nn.Conv2d(
                in_depth,
                F,
                kernel_size=(1, kernel_length),
                stride=(1, 1),
                padding=(0, kernel_length // 2),
                bias=not bn_track,
            ),
            nn.BatchNorm2d(
                F, momentum=bn_mm, affine=bn_track, track_running_stats=bn_track
            ),
            Conv2dWithConstraint(
                F,
                F * D,
                max_norm=1.0,
                kernel_size=(in_chans, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=F,
                bias=not bn_track,
            ),
            nn.BatchNorm2d(
                F * D, momentum=bn_mm, affine=bn_track, track_running_stats=bn_track
            ),
            nn.ELU(inplace=True),
        ]

        super().__init__(*modules)


class DilationConv(nn.Module):
    def __init__(
        self, F, kernel_length=4, dilation=1, drop_prob=0.5, bn_mm=0.01, bn_track=True
    ):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                F,
                F,
                kernel_size=(1, kernel_length),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, dilation),
                groups=F,
                bias=not bn_track,
            ),
            nn.BatchNorm2d(
                F, momentum=bn_mm, affine=bn_track, track_running_stats=bn_track
            ),
            nn.ELU(inplace=True),
            nn.Dropout(p=drop_prob),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                F,
                F,
                kernel_size=(1, kernel_length),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, dilation),
                groups=F,
                bias=not bn_track,
            ),
            nn.BatchNorm2d(
                F, momentum=bn_mm, affine=bn_track, track_running_stats=bn_track
            ),
            nn.ELU(inplace=True),
            nn.Dropout(p=drop_prob),
        )

        self.left_padding = (kernel_length - 1) * dilation

    def forward(self, x):  # x: [NF1T]
        x0 = x

        x = F.pad(x, (self.left_padding, 0, 0, 0))
        x = self.conv_1(x)
        x = F.pad(x, (self.left_padding, 0, 0, 0))
        x = self.conv_2(x)

        x = x + x0

        return x


class EEG_ITNet(nn.Module):
    def __init__(
        self,
        in_chans,
        in_samples,
        in_depth=1,
        Fs=(8,),
        Ds=(2,),
        F2=16,
        F3=16,
        kernel_lengths=(64,),
        dilation_kernel_length=4,
        n_dilation_layers=3,
        drop_prob=0.5,
        bn_mm=0.01,
        bn_track=True,
    ):
        super().__init__()

        # -------temporal-spatial block----------
        self.temporal_spatial_convs = nn.ModuleList()
        for F, D, kernel_length in zip(Fs, Ds, kernel_lengths):
            self.temporal_spatial_convs.append(
                TemporalSpatialConv(
                    in_chans,
                    in_depth,
                    F,
                    D,
                    kernel_length,
                    drop_prob,
                    bn_mm,
                    bn_track,
                )
            )

        self.pool_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), ceil_mode=False),
            nn.Dropout(p=drop_prob),
        )

        F = sum(F * D for F, D in zip(Fs, Ds))

        # -------dilation block------------------
        self.dilation_conv = nn.Sequential(
            *(
                DilationConv(
                    F, dilation_kernel_length, 2**i, drop_prob, bn_mm, bn_track
                )
                for i in range(n_dilation_layers)
            )
        )

        # -------conv2 (reduction) block-----------------
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                F,
                F2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=not bn_track,
            ),
            nn.BatchNorm2d(
                F2, momentum=bn_mm, affine=bn_track, track_running_stats=bn_track
            ),
            nn.ELU(inplace=True),
        )

        self.pool_2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), ceil_mode=False),
            nn.Dropout(p=drop_prob),
        )

        # # -------conv3 block---------------------
        # with torch.no_grad():
        #     x = torch.zeros((1, in_depth, in_chans, in_samples), dtype=torch.float32)
        #     x = torch.cat([conv(x) for conv in self.temporal_spatial_convs], dim=1)
        #     x = self.pool_1(x)
        #     x = self.dilation_conv(x)
        #     x = self.conv_2(x)
        #     x = self.pool_2(x)
        #     final_kernel_h, final_kernel_w = x.size(2), x.size(3)

        # self.conv_3 = nn.Sequential(
        #     nn.Conv2d(
        #     # Conv2dWithConstraint(
        #         F2,
        #         F2,
        #         # max_norm=1.0,
        #         kernel_size=(final_kernel_h, final_kernel_w),
        #         stride=(1, 1),
        #         padding=(0, 0),
        #         groups=F2,
        #         bias=not bn_track,
        #     ),  # -> [NF11]
        #     nn.Conv2d(
        #     # Conv2dWithConstraint(
        #         F2,
        #         F3,
        #         # max_norm=1.0,
        #         kernel_size=(1, 1),
        #         stride=(1, 1),
        #         padding=(0, 0),
        #         bias=not bn_track,
        #     ),
        #     nn.BatchNorm2d(
        #         F3,
        #         momentum=bn_mm,
        #         affine=bn_track,
        #         track_running_stats=bn_track,
        #     ),
        # )

        # # -------linear layer--------------------
        # with torch.no_grad():
        #     x = torch.zeros((1, in_depth, in_chans, in_samples), dtype=torch.float32)
        #     x = torch.cat([conv(x) for conv in self.temporal_spatial_convs], dim=1)
        #     x = self.pool_1(x)
        #     x = self.dilation_conv(x)
        #     x = self.conv_2(x)
        #     x = self.pool_2(x)
        #     x = torch.flatten(x, start_dim=1)
        #     flatten_size = x.size(1)

        # self.linear = nn.Sequential(
        #     # nn.Linear(flatten_size, F3, bias=not bn_track),
        #     LinearWithConstraint(flatten_size, F3, max_norm=1.0, bias=not bn_track),
        #     nn.ELU(inplace=True),
        #     nn.BatchNorm1d(
        #         F3, momentum=bn_mm, affine=bn_track, track_running_stats=bn_track
        #     ),
        #     nn.Dropout(p=drop_prob),
        # )

    def forward(self, x):  # x: [NCT]
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = torch.cat(
            [conv(x) for conv in self.temporal_spatial_convs], dim=1
        )  # [NF1T]
        x = self.pool_1(x)
        x = self.dilation_conv(x)  # [NF1T]
        x = self.conv_2(x)  # [NF1T]
        x = self.pool_2(x)
        # x = self.conv_3(x)
        x = torch.flatten(x, start_dim=1)  # [B,F3]
        # x = torch.flatten(x, start_dim=1)
        # x = self.linear(x)

        return x


if __name__ == "__main__":
    module = EEG_ITNet(
        in_chans=32,
        in_samples=256,
        in_depth=1,
        Fs=(8, 4, 2),
        Ds=(2, 2, 2),
        F2=28,
        F3=64,
        kernel_lengths=(64, 32, 16),
        dilation_kernel_length=6,
        n_dilation_layers=3,
        drop_prob=0.4,
        bn_mm=0.01,
        bn_track=True,
    )
    print(module)

    x = torch.rand(8, 32, 256).float()
    print(module(x).size())
