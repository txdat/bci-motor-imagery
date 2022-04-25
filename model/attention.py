import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSelfAttn(nn.Module):
    def __init__(self, F):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, F, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
        )
        self.conv2 = nn.Conv2d(
            1, F, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # [B,1,C,T]
        b, _, c, t = x.size()
        x1 = self.conv1(x).permute(0, 2, 1, 3).reshape(b, c, -1)  # [B,C,F*T]
        x2 = self.conv2(x).permute(0, 1, 3, 2).reshape(b, -1, c)  # [B,F*T,C]

        e = torch.matmul(x1, x2)  # [B,C,C]
        e_max, _ = e.max(dim=2, keepdim=True)
        e_min, _ = e.min(dim=2, keepdim=True)
        e = (e - e_min) / (e_max - e_min + 1e-8)
        e = F.softmax(e, dim=2).unsqueeze(1)  # [B,1,C,C]

        x = self.gamma * torch.matmul(e, x) + x

        return x


class TimeSelfAttn(nn.Module):
    def __init__(self, F):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, F, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
        )
        self.conv2 = nn.Conv2d(
            1, F, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # [B,1,C,T]
        b, _, c, t = x.size()
        x1 = self.conv1(x).permute(0, 3, 1, 2).reshape(b, t, -1)  # [B,T,F*C]
        x2 = self.conv2(x).reshape(b, -1, t)  # [B,F*C,T]

        e = torch.matmul(x1, x2)  # [B,T,T]
        e_max, _ = e.max(dim=2, keepdim=True)
        e_min, _ = e.min(dim=2, keepdim=True)
        e = (e - e_min) / (e_max - e_min + 1e-8)
        e = F.softmax(e, dim=2).unsqueeze(1)  # [B,1,T,T]

        x = self.gamma * torch.matmul(x, e.transpose(-2, -1)) + x

        return x


class SelfAttn(nn.Module):
    """https://sci-hub.se/10.3389/fnins.2020.587520"""

    def __init__(self, Fc, Ft):
        super().__init__()

        self.ca = ChannelSelfAttn(Fc)
        self.ta = TimeSelfAttn(Ft)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B,1,C,T]
        x = torch.cat((x, self.ca(x), self.ta(x)), dim=1)

        return x
