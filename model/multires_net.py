import torch
import torch.nn as nn
import torch.nn.functional as F

from util import Conv2d


class MultiResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        res_channels,
        base_kernel_size,
        stride=1,
        res_scales=1,
        drop=0.25,
        dilation=True,
        use_inputs=True,
        use_sep_conv=False,
    ):
        super().__init__()

        self.res_convs = nn.ModuleList()
        for i in range(res_scales):
            if dilation:
                dil = 2 ** i
                kernel_size = base_kernel_size

            else:
                dil = 1
                kernel_size = base_kernel_size * 2 ** i

            self.res_convs.append(
                Conv2d(
                    in_channels,
                    res_channels,
                    kernel_size=(1, kernel_size + 1 - kernel_size % 2),
                    stride=(1, stride),
                    padding=(0, kernel_size // 2 * dil),
                    dilation=(1, dil),
                    bias=False,
                    use_sep_conv=use_sep_conv,
                    use_norm=True,
                    use_act=True,
                )
            )

        if use_inputs:
            self.identity_conv = (
                Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=(1, 1),
                    stride=(1, stride),
                    padding=(0, 0),
                    bias=False,
                    use_sep_conv=False,
                    use_norm=True,
                    use_act=True,
                )
                if stride > 1
                else nn.Identity()
            )

        else:
            self.identity_conv = None

        self.reduction_conv = Conv2d(
            in_channels * int(use_inputs) + res_channels * res_scales,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
            use_sep_conv=False,
            use_norm=True,
            use_act=True,
        )

    def forward(self, x: torch.Tensor):
        if self.identity_conv is not None:
            x = torch.cat(
                [self.identity_conv(x)] + [conv(x) for conv in self.res_convs],
                dim=1,
            )

        else:
            x = torch.cat([conv(x) for conv in self.res_convs], dim=1)

        x = self.reduction_conv(x)

        return x


class MultiResNet(nn.Module):
    def __init__(
        self,
        n_classes,
        n_channels,
        n_samples,
        sfreq,
        ft,
        fs,
        res_scales=3,
        embedding_size=64,
        drop=0.25,
        use_sep_conv=False,
    ):
        super().__init__()

        assert (
            sfreq % 32 == 0 and sfreq // 32 >= 1
        ), f"sampling frequency {sfreq} is too small"
        assert (
            n_samples >= sfreq and n_samples // 64 >= 1
        ), f"sample duration {n_samples / sfreq}s is too small"

        self.backbone = nn.Sequential(
            MultiResBlock(
                1,
                ft,
                ft,
                base_kernel_size=sfreq // 4,
                stride=1,
                res_scales=res_scales,
                drop=drop,
                dilation=True,
                use_inputs=False,
                use_sep_conv=use_sep_conv,
            ),
            Conv2d(
                ft,
                fs,
                kernel_size=(n_channels, 1),
                stride=(n_channels, 1),
                padding=(0, 0),
                bias=False,
                use_sep_conv=use_sep_conv,
                use_norm=True,
                use_act=True,
            ),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),
            nn.Dropout(p=drop),
            MultiResBlock(
                fs,
                fs,
                fs,
                base_kernel_size=sfreq // 8,
                stride=1,
                res_scales=res_scales,
                drop=drop,
                dilation=True,
                use_inputs=True,
                use_sep_conv=use_sep_conv,
            ),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),
            nn.Dropout(p=drop),
            MultiResBlock(
                fs,
                fs,
                fs,
                base_kernel_size=sfreq // 32,
                stride=1,
                res_scales=res_scales,
                drop=drop,
                dilation=True,
                use_inputs=True,
                use_sep_conv=use_sep_conv,
            ),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),
            nn.Dropout(p=drop),
        )

        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples).float()
            x = self.backbone(x)
            reduction_kernel_size = (x.size(2), x.size(3))  # [N,*,*,T//64]

        self.reduction_conv = Conv2d(
            fs,
            embedding_size,
            kernel_size=reduction_kernel_size,
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
            use_sep_conv=use_sep_conv,
            use_norm=True,
            use_act=False,
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(embedding_size, n_classes, bias=True),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)  # [B,1,C,T]
        x = self.backbone(x)
        x = self.reduction_conv(x)
        x = self.flatten(x)
        x = self.linear(x)

        x = F.log_softmax(x, dim=1)

        return x


if __name__ == "__main__":
    model = MultiResNet(
        n_classes=7,
        n_channels=11,
        n_samples=128,
        sfreq=128,
        ft=8,
        fs=16,
        use_sep_conv=False,
    )
    # print(model)
    #
    # print(sum(p.numel() for p in model.parameters()))
    # for name, p in model.named_parameters():
    #     print(f"{name}: {p.size()}")

    x = torch.rand(8, 11, 128).float()
    print(model(x))
