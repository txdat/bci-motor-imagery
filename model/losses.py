from typing import Type, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    smooth: float = 0,
    reduce: bool = True,
) -> torch.Tensor:
    # input (log_prob): [bsz, num_classes], target (classes): [bsz]
    n_classes = input.size(1)
    target = F.one_hot(target, num_classes=n_classes).float()
    sm_target = (1.0 - smooth) * target + smooth / n_classes

    if weight is not None:
        if reduce:
            return -(sm_target * weight * input).sum() / (target * weight).sum()

        return -(sm_target * weight * input).sum(dim=1)

    else:
        if reduce:
            return -(sm_target * input).sum(dim=1).mean()

        return -(sm_target * input).sum(dim=1)


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    smooth: float = 0,
    gamma: float = 2.0,
    reduce: bool = True,
) -> torch.Tensor:
    # https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
    # not working well with smooth

    # input (log_prob): [bsz, num_classes], target (classes): [bsz]
    n_classes = input.size(1)
    sm_gamma = (
        F.one_hot(target, num_classes=n_classes).float() * gamma
    )  # disable (1-p)^g on negative indices
    weighted_input = torch.pow(1.0 - torch.exp(input), sm_gamma) * input

    return nll_loss(weighted_input, target, weight=weight, smooth=smooth, reduce=reduce)


class NllLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        smooth: float = 0,
        reduce: bool = True,
    ):
        super().__init__()

        self.register_buffer(
            "weight", weight.unsqueeze(0) if weight is not None else None
        )  # [1, n_classes]
        self.weight: Optional[torch.Tensor]

        self.smooth = smooth
        self.reduce = reduce

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nll_loss(
            input,
            target,
            weight=self.weight,
            smooth=self.smooth,
            reduce=self.reduce,
        )


class FocalLoss(NllLoss):
    """
    class-weighted focal loss
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        smooth: float = 0,
        gamma: float = 2.0,
        reduce: bool = True,
    ):
        super().__init__(weight, smooth, reduce)

        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(
            input,
            target,
            weight=self.weight,
            smooth=self.smooth,
            gamma=self.gamma,
            reduce=self.reduce,
        )


class MixUpLoss(nn.Module):
    """
    braindecode's mixup criterion
    """

    def __init__(self, criterion_cls: Type, **criterion_params):
        super().__init__()

        self.criterion = criterion_cls(**criterion_params)
        setattr(self.criterion, "reduce", False)  # disable batch reduction

    def forward(
        self, input: torch.Tensor, target: Union[List[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if type(target) == list:  # training
            target_a, target_b, lam = target  # [B,]...

            loss_a = self.criterion(input, target_a)
            loss_b = self.criterion(input, target_b)

            loss = lam * loss_a + (1.0 - lam) * loss_b
            loss = loss.mean()

        else:  # valid/test
            loss = self.criterion(input, target).mean()  # diff. weighted

        return loss


if __name__ == "__main__":
    n = 128
    c = 10
    x0 = torch.rand(n, c).float()
    x = F.log_softmax(x0, dim=-1)
    y = torch.randint(0, c, (n,)).long()
    y1 = torch.randint(0, c, (n,)).long()
    lam = torch.rand(n).float()
    w = torch.rand(c).float()
    sm = 0.1

    print(torch.allclose(nll_loss(x, y), F.nll_loss(x, y)))
    print(torch.allclose(nll_loss(x, y, weight=w), F.nll_loss(x, y, weight=w)))
    print(
        torch.allclose(nll_loss(x, y, reduce=False), F.nll_loss(x, y, reduction="none"))
    )
    print(
        torch.allclose(
            nll_loss(x, y, weight=w, reduce=False),
            F.nll_loss(x, y, weight=w, reduction="none"),
        )
    )
    print(
        torch.allclose(
            nll_loss(x, y, smooth=sm), F.cross_entropy(x0, y, label_smoothing=sm)
        )
    )
    print(
        torch.allclose(
            nll_loss(x, y, smooth=sm, reduce=False),
            F.cross_entropy(x0, y, label_smoothing=sm, reduction="none"),
        )
    )
    print(
        torch.allclose(
            nll_loss(x, y, weight=w, smooth=sm),
            F.cross_entropy(x0, y, weight=w, label_smoothing=sm),
        )
    )
    print(
        torch.allclose(
            nll_loss(x, y, weight=w, smooth=sm, reduce=False),
            F.cross_entropy(x0, y, weight=w, label_smoothing=sm, reduction="none"),
        )
    )
