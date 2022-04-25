from typing import Optional
import torch
import torch.nn as nn


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


class ChannelWiseNorm(nn.Module):
    def __init__(self, in_chans, in_depth, F1):
        super().__init__()

        # transpose to [B,C*F1,1,T]
        # self.norm = nn.BatchNorm2d(in_chans * F1, momentum=0.01, affine=True, eps=1e-3)
        # self.norm = nn.InstanceNorm2d(in_chans * F1, momentum=0.01, affine=True, eps=1e-3)
        self.norm = nn.GroupNorm(num_groups=in_chans, num_channels=in_chans * F1)

        # transpose to [B,C,F1,T]
        # self.norm = nn.GroupNorm(num_groups=in_chans, num_channels=in_chans)

        self.in_chans = in_chans
        self.in_depth = in_depth
        self.F1 = F1

    def forward(self, x):  # [B,F1,C,T]
        # transpose to [B,C*F1,1,T]
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

        # transpose to [B,C,F1,T]
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
                in_depth * F1,
                kernel_size=(1, kernel_length),
                stride=(1, 1),
                padding=(0, kernel_length // 2),
                groups=in_depth,
                bias=False,
            ),
        )
        self.add_module(
            "norm_temporal",
            # nn.BatchNorm2d(in_depth * F1, momentum=0.01, affine=True, eps=1e-3),
            # nn.InstanceNorm2d(in_depth * F1, momentum=0.01, affine=True, eps=1e-3),
            nn.GroupNorm(num_groups=in_depth, num_channels=in_depth * F1),
            # ChannelWiseNorm(in_chans, in_depth, F1),
        )
        self.add_module(
            "conv_spatial",
            # Conv2dWithConstraint(
            WSConv2d(
                in_depth * F1,
                in_depth * F1 * D,
                max_norm=1.0,
                kernel_size=(in_chans, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=in_depth * F1,
                bias=False,
            ),
        )
        self.add_module(
            "norm_1",
            # nn.BatchNorm2d(in_depth * F1 * D, momentum=0.01, affine=True, eps=1e-3),
            # nn.InstanceNorm2d(in_depth * F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.GroupNorm(num_groups=in_depth * F1, num_channels=in_depth * F1 * D),
        )
        # self.add_module("act_1", nn.ELU(inplace=True))
        self.add_module("act_1", nn.Mish())
        self.add_module("pool_1", pool_class(kernel_size=(1, 4), stride=(1, 4)))
        self.add_module("drop_1", nn.Dropout(p=drop_prob))


class MB_EEGNet(nn.Module):
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
        kernel_lengths=(64,),
        third_kernel_size=(8, 4),
        drop_prob=0.5,
    ):
        super().__init__()

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

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

        F1 *= in_depth
        self.conv_2 = nn.Sequential()
        self.conv_2.add_module(
            "conv_separable_depth",
            nn.Conv2d(
                F1 * D,
                F1 * D,
                kernel_size=(1, 16),
                stride=(1, 1),
                padding=(0, 8),
                groups=F1 * D,
                bias=False,
            ),
        )
        self.conv_2.add_module(
            "conv_separable_point",
            # nn.Conv2d(
            WSConv2d(
                F1 * D,
                F2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
        )
        self.conv_2.add_module(
            "norm_2",
            # nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            # nn.InstanceNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.GroupNorm(num_groups=8, num_channels=F2),
        )
        # self.conv_2.add_module("act_2", nn.ELU(inplace=True))
        self.conv_2.add_module("act_2", nn.Mish())
        self.conv_2.add_module("pool_2", pool_class(kernel_size=(1, 8), stride=(1, 8)))
        self.conv_2.add_module("drop_2", nn.Dropout(p=drop_prob))

        # with torch.no_grad():
        #     x = torch.zeros((1, in_depth, in_chans, in_samples), dtype=torch.float32)
        #     x = torch.stack([conv(x) for conv in self.conv_1s]).sum(dim=0)
        #     x = self.conv_2(x)
        #     x = x.cpu().data.numpy()

        self.conv_3 = nn.Sequential()
        self.conv_3.add_module(
            "conv_separable_depth",
            nn.Conv2d(
                F2,
                F2,
                # kernel_size=(x.shape[2], x.shape[3]),
                kernel_size=(1, 4),
                stride=(1, 1),
                # padding=(0, 0),
                padding=(0, 2),
                groups=F2,
                bias=False,
            ),
        )
        self.conv_3.add_module(
            "conv_separable_point",
            # nn.Conv2d(
            WSConv2d(
                F2,
                F3,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
        )
        self.conv_3.add_module(
            "norm_3",
            # nn.BatchNorm2d(F3, momentum=0.01, affine=True, eps=1e-3),
            # nn.InstanceNorm2d(F3, momentum=0.01, affine=True, eps=1e-3),
            nn.GroupNorm(num_groups=8, num_channels=F3),
        )
        # self.conv_3.add_module("act_3", nn.ELU(inplace=True))
        self.conv_3.add_module("act_3", nn.Mish())

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        # self.drop = nn.Dropout(p=drop_prob)
        # self.clf = nn.Linear(F3, n_classes, bias=True)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B1CT]

        x = torch.stack([conv(x) for conv in self.conv_1s]).sum(dim=0)
        x = self.conv_2(x)
        x = self.conv_3(x)

        # x = torch.flatten(x, start_dim=1)  # [B,F3]
        x = torch.flatten(self.pool(x), start_dim=1)
        # x = self.drop(x)
        # x = self.clf(x)

        return x


class EEGClassifier(nn.Module):
    def __init__(self, F3, n_classes, drop_prob=0.5):
        super().__init__()

        self.drop = nn.Dropout(p=drop_prob)
        self.ln = nn.Linear(F3, n_classes, bias=True)
        # self.ln = LinearWithConstraint(F3, n_classes, max_norm=0.5, bias=True)

    def forward(self, x):
        x = self.drop(x)
        x = self.ln(x)

        return x
