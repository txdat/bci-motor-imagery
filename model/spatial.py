import torch
import torch.nn as nn

from .util import Conv2d


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class SymmetricSpatialBlock(nn.Module):
    """
    Symmetric left-right spatial features
    https://arxiv.org/abs/2004.02965

    inputs' channels:
        [left's channels,
         center's channels,
         right's channels]
    with left/right's channels are symmetric
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_channels,  # electrodes
        n_center_channels=0,  # electrodes
        **kwargs,
    ):
        assert (n_channels - n_center_channels) % 2 == 0, "not symmetric electrodes"

        super().__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=(n_channels, 1),
            stride=(n_channels, 1),
            padding=(0, 0),
            bias=False,
            **kwargs,
        )

        n_lr_channels = (n_channels - n_center_channels) // 2
        self.lr_conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=(n_lr_channels, 1),
            stride=(n_lr_channels + n_center_channels, 1),  # skip center's channels
            padding=(0, 0),
            bias=False,
            **kwargs,
        )

    def forward(self, x):
        x = torch.cat([self.conv(x), self.lr_conv(x)], dim=2)  # [N*CT] -> [N*3T]
        return x


class DynamicSpatialBlock(nn.Module):
    """
    Dynamic spatial filtering (n_in_channels -> n_out_channels)
    https://arxiv.org/abs/2105.12916

    # TODO: check working with normalized EEG?
    """

    def __init__(self, n_in_channels, n_out_channels, logcov=True, soft_thresh=0.1):
        super().__init__()

        inp = (n_in_channels * (n_in_channels + 1) // 2) if logcov else n_in_channels
        out = n_out_channels * n_in_channels + n_out_channels
        hid = out // 8

        self.linear = nn.Sequential(
            nn.Linear(inp, hid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out, bias=True),
            # LinearWithConstraint(hid, out, max_norm=1.0, bias=True),
        )

        self.inds = torch.triu_indices(
            n_in_channels, n_in_channels
        )  # upper, [2,n_in*(n_in+1)/2]

        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.logcov = logcov
        self.soft_thresh = soft_thresh

    def forward(self, x):  # x: [B,C,T]
        # x is not NORMALIZED

        if self.logcov:
            # compute log-covariance
            # log(C) = Ulog(V)U.T
            xm = x - x.mean(dim=2, keepdim=True)
            cov = torch.matmul(xm, xm.transpose(2, 1)) / (xm.size(2) - 1)

            e, v = torch.linalg.eigh(cov, UPLO="U")
            e = torch.log(torch.clamp(e, min=1e-10))
            log_cov = torch.matmul(
                torch.matmul(v, e.diag_embed()), v.transpose(2, 1)
            )  # [B,C,C]

            feats = log_cov[:, self.inds[0], self.inds[1]]  # [B,n_in*(n_in+1)/2]

        else:
            # compute log-variance
            feats = torch.log(torch.var(x, dim=2, unbiased=True))
            feats[torch.isneginf(feats)] = 0

        feats = self.linear(feats)  # [B,n_out*n_in+n_out]

        w = feats[:, : -self.n_out_channels].reshape(
            -1, self.n_out_channels, self.n_in_channels
        )  # [B,n_out,n_in]
        b = feats[:, -self.n_out_channels :].reshape(
            -1, self.n_out_channels, 1
        )  # [B,n_out,1]

        if self.logcov and self.soft_thresh > 0:
            # apply soft-thresholding in [-thr,thr]
            w = torch.clamp(w - self.soft_thresh, min=0) + torch.clamp(
                w + self.soft_thresh, max=0
            )

        x = (x - x.mean(dim=2, keepdim=True)) / x.std(dim=2, keepdim=True)
        x = torch.matmul(w, x) + b

        return x
