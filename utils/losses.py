from collections.abc import Callable

import torch


def _fp16_recon_loss(x_r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(torch.abs(x_r - x), dim=tuple(range(1, x_r.dim()))))


def _mocca_recon_loss(x_r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum((x_r - x) ** 2, dim=tuple(range(1, x_r.dim()))))


DISTS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = dict(
    l1=_fp16_recon_loss,
    l2=_mocca_recon_loss,
)
