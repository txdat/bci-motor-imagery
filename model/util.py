import torch
import torch.nn as nn


def init_weights(
    module: nn.Module,
    xavier_init: bool = True,
    gain: float = 1.0,
    nonlinearity: str = "leaky_relu",
):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            if xavier_init:
                nn.init.xavier_uniform_(m.weight, gain=gain)
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.0)

        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, val=1.0)
            nn.init.constant_(m.bias, val=0.0)


def static_batchnorm(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.track_running_stats = False
            m.affine = False


def square(x: torch.Tensor) -> torch.Tensor:
    return x * x


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def covariance(x: torch.Tensor) -> torch.Tensor:
    # x: [bsz, *, ch, t]
    xm = x - x.mean(dim=-1, keepdim=True)
    return torch.matmul(xm, xm.transpose(-1, -2)) / (x.size(-1) - 1)


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class SafeLog(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return safe_log(x, self.eps)


class Covariance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return covariance(x)


class ConstraintConv2d(nn.Conv2d):
    def __init__(self, *args, maxnorm=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.maxnorm = maxnorm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.maxnorm
        )

        return super().forward(x)


class Conv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        *args,
        use_constraint_conv=False,
        use_sep_conv=False,
        use_norm=False,
        use_act=False,
        **kwargs
    ):
        super().__init__()

        conv_class = [nn.Conv2d, ConstraintConv2d][int(use_constraint_conv)]

        if use_sep_conv:
            kwargs.pop("groups", None)
            self.add_module(
                "dw_conv",
                conv_class(
                    in_channels, in_channels, *args, groups=in_channels, **kwargs
                ),
            )

            if use_norm:
                self.add_module("dw_norm", nn.BatchNorm2d(in_channels))

            for k in ["kernel_size", "stride", "padding", "groups", "bias"]:
                kwargs.pop(k, None)
            self.add_module(
                "pw_conv",
                conv_class(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    groups=1,
                    bias=False,
                    **kwargs,
                ),
            )

            if use_norm:
                self.add_module("pw_norm", nn.BatchNorm2d(out_channels))

        else:
            self.add_module(
                "conv", conv_class(in_channels, out_channels, *args, **kwargs)
            )

            if use_norm:
                self.add_module("norm", nn.BatchNorm2d(out_channels))

        if use_act:
            self.add_module("act", nn.ELU(inplace=True))


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class SqueezeExcite(nn.Module):
    def __init__(self):
        super().__init__()
