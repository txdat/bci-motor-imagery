from typing import Optional
import torch
import torch.nn as nn


class WSConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm: Optional[float] = None, **kwargs):
        self.max_norm = max_norm
        super(WSConv2d, self).__init__(*args, **kwargs)

    def __repr__(self):
        if self.max_norm is not None:
            return f"{super().__repr__()}, max_norm={self.max_norm}"

        return super().__repr__()

    def forward(self, x):
        std, mean = torch.std_mean(
            self.weight.data, dim=(1, 2, 3), keepdim=True
        )  # [out,in,H,W]
        self.weight.data = (self.weight.data - mean) / (std + 1e-5)

        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

        return super(WSConv2d, self).forward(x)


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


class ConvTranspose2dWithConstraint(nn.ConvTranspose2d):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super(ConvTranspose2dWithConstraint, self).__init__(*args, **kwargs)

    def __repr__(self):
        return f"{super().__repr__()}, max_norm={self.max_norm}"

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )  # [out,in,H,W]
        return super(ConvTranspose2dWithConstraint, self).forward(x)


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


class TSConv(nn.Sequential):
    def __init__(
        self,
        in_chans,
        in_depth,
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
                F,
                momentum=bn_mm,
                affine=bn_track,
                track_running_stats=bn_track,
            ),
            nn.Dropout(p=drop_prob),
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
            nn.ELU(inplace=True),
            nn.BatchNorm2d(
                F * D,
                momentum=bn_mm,
                affine=bn_track,
                track_running_stats=bn_track,
            ),
            nn.Dropout(p=drop_prob),
        ]

        super().__init__(*modules)


class EEGNetEncoder(nn.Module):
    def __init__(
        self,
        in_chans,
        in_samples,
        in_depth=1,
        pool_mode="mean",
        Fs=(8,),
        Ds=(2,),
        F2=16,
        F3=16,
        kernel_lengths=(64,),
        drop_prob=0.5,
        bn_mm=0.01,
        bn_track=True,
    ):
        super().__init__()

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        # conv1 blocks--------------------------
        self.conv_1s = nn.ModuleList()
        for F, D, kernel_length in zip(Fs, Ds, kernel_lengths):
            self.conv_1s.append(
                TSConv(
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
            pool_class(kernel_size=(1, 4), stride=(1, 4), ceil_mode=False),
            # nn.Dropout(p=drop_prob),
        )

        F = sum(F * D for F, D in zip(Fs, Ds))

        # conv2 block--------------------------
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                F,
                F,
                kernel_size=(1, 16),
                stride=(1, 1),
                padding=(0, 8),
                groups=F,
                bias=not bn_track,
            ),
            nn.Conv2d(
                F,
                F2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=not bn_track,
            ),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(
                F2,
                momentum=bn_mm,
                affine=bn_track,
                track_running_stats=bn_track,
            ),
            nn.Dropout(p=drop_prob),
        )

        self.pool_2 = nn.Sequential(
            pool_class(kernel_size=(1, 8), stride=(1, 8), ceil_mode=False),
            # nn.Dropout(p=drop_prob),
        )

        # conv3 block--------------------------
        with torch.no_grad():
            x = torch.zeros((1, in_depth, in_chans, in_samples), dtype=torch.float32)
            x = torch.cat([conv(x) for conv in self.conv_1s], dim=1)
            x = self.pool_1(x)
            x = self.conv_2(x)
            x = self.pool_2(x)
            final_kernel_h, final_kernel_w = x.size(2), x.size(3)

        self.conv_3 = nn.Sequential(
            # nn.Conv2d(
            Conv2dWithConstraint(
                F2,
                F2,
                max_norm=1.0,
                kernel_size=(final_kernel_h, final_kernel_w),
                stride=(1, 1),
                padding=(0, 0),
                groups=F2,
                bias=not bn_track,
            ),  # -> [NF11]
            # nn.Conv2d(
            Conv2dWithConstraint(
                F2,
                F3,
                max_norm=1.0,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=not bn_track,
            ),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(
                F3,
                momentum=bn_mm,
                affine=bn_track,
                track_running_stats=bn_track,
            ),
            nn.Dropout(p=drop_prob),
        )

        # # linear layer-------------------------
        # with torch.no_grad():
        #     x = torch.zeros((1, in_depth, in_chans, in_samples), dtype=torch.float32)
        #     x = torch.cat([conv(x) for conv in self.conv_1s], dim=1)
        #     x = self.pool_1(x)
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

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B1CT]

        x = torch.cat([conv(x) for conv in self.conv_1s], dim=1)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = torch.flatten(x, start_dim=1)  # [B,F3]
        # x = torch.flatten(x, start_dim=1)
        # x = self.linear(x)

        return x


class EEGNetDecoder(nn.Module):
    def __init__(
        self,
        in_chans,
        in_samples,
        in_depth=1,
        Fz=16,
        F=8,
        D=2,
        F2=16,
        F3=16,
        kernel_length=64,
        drop_prob=0.5,
        bn_mm=0.01,
        bn_track=True,
    ):
        super().__init__()

        # # linear layer---------------------------
        # self.linear = nn.Sequential(
        #     nn.Linear(Fz, F2 * (in_samples // 32), bias=not bn_track),
        #     nn.ELU(inplace=True),
        #     nn.BatchNorm1d(
        #         F2 * (in_samples // 32),
        #         momentum=bn_mm,
        #         affine=bn_track,
        #         track_running_stats=bn_track,
        #     ),
        #     nn.Dropout(p=drop_prob),
        # )

        # convtranspose3 block-----------------
        self.conv_trans_3 = nn.Sequential(
            nn.ConvTranspose2d(
                Fz,
                Fz,
                kernel_size=(1, in_samples // 32),
                stride=(1, 1),
                padding=(0, 0),
                groups=Fz,
                bias=not bn_track,
            ),
            nn.Conv2d(
                Fz,
                F2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=not bn_track,
            ),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(
                F2,
                momentum=bn_mm,
                affine=bn_track,
                track_running_stats=bn_track,
            ),
            nn.Dropout(p=drop_prob),
        )

        # convtranspose2 block-----------------
        self.conv_trans_2 = nn.Sequential(
            nn.ConvTranspose2d(
                F2,
                F2,
                kernel_size=(1, 16),
                stride=(1, 8),
                padding=(0, 4),
                groups=F2,
                bias=not bn_track,
            ),
            nn.Conv2d(
                F2,
                F * D,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=not bn_track,
            ),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(
                F * D,
                momentum=bn_mm,
                affine=bn_track,
                track_running_stats=bn_track,
            ),
            nn.Dropout(p=drop_prob),
        )

        # convtranspose1 block-----------------
        self.conv_trans_1 = nn.Sequential(
            nn.ConvTranspose2d(
                F * D,
                F * D,
                kernel_size=(1, kernel_length),
                stride=(1, 4),
                padding=(0, (kernel_length - 4) // 2),
                groups=F * D,
                bias=not bn_track,
            ),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(
                F * D,
                momentum=bn_mm,
                affine=bn_track,
                track_running_stats=bn_track,
            ),
            nn.Dropout(p=drop_prob),
            nn.Conv2d(
                # Conv2dWithConstraint(
                F * D,
                in_depth * in_chans,
                # max_norm=1.0,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
            ),
        )

        self.F2 = F2
        self.in_chans = in_chans
        self.in_depth = in_depth

    def forward(self, x):  # x: [B,Fz]
        # x = self.linear(x)  # [B,F2*(in_samples//32)]
        # x = x.reshape(x.size(0), self.F2, 1, -1).contiguous()  # [B,F2,1,in_samples//32]
        x = x.unsqueeze(2).unsqueeze(3)  # [B,Fz,1,1]
        x = self.conv_trans_3(x)  # [B,F2,1,in_samples//32]
        x = self.conv_trans_2(x)  # [B,F1*D,1,in_samples//4]
        x = self.conv_trans_1(x)  # [B,in_depth*in_chans,1,in_samples]
        x = x.reshape(
            x.size(0), self.in_depth, self.in_chans, x.size(3)
        ).contiguous()  # [B,in_depth,in_chans,in_samples]

        return x


class EEGClassifier(nn.Module):
    def __init__(self, Fz, n_classes, drop_prob=0.5, max_norm=None):
        super().__init__()

        # self.drop = nn.Dropout(p=drop_prob)
        if max_norm is None:
            self.linear = nn.Linear(Fz, n_classes, bias=True)
        else:
            self.linear = LinearWithConstraint(
                Fz, n_classes, max_norm=max_norm, bias=True
            )

    def forward(self, x):
        # x = self.drop(x)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    n_channels = 32
    n_samples = 256

    K = 64
    F1 = 8
    D = 2
    F2 = 28
    F3 = 64
    Fz = 16
    drop_prob = 0.5

    encoder = EEGNetEncoder(
        in_chans=n_channels,
        in_samples=n_samples,
        in_depth=1,
        pool_mode="mean",
        Fs=(F1, F1 // 2, F1 // 4),
        Ds=(D, D, D),
        F2=F2,
        F3=F3,
        kernel_lengths=(K, K // 2, K // 4),
        drop_prob=drop_prob,
        bn_mm=0.01,
        bn_track=True,
    )

    decoder = EEGNetDecoder(
        in_chans=n_channels,
        in_samples=n_samples,
        in_depth=1,
        Fz=Fz,
        F=F1,
        D=D,
        F2=F2,
        F3=F3,
        kernel_length=K,
        drop_prob=drop_prob,
        bn_mm=0.01,
        bn_track=True,
    )

    print(encoder)
    print(decoder)

    vae_mean = nn.Linear(F3, Fz, bias=True)
    vae_logvar = nn.Linear(F3, Fz, bias=True)

    x = torch.rand(8, n_channels, n_samples).float()

    e = encoder(x)
    z_mean = vae_mean(e)
    z_logvar = vae_logvar(e)

    std = torch.exp(0.5 * z_logvar)
    z = z_mean + torch.randn_like(std) * std

    x0 = decoder(z).squeeze(1)
    print(x0.size())
