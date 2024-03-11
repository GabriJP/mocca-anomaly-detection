import logging
import random
from collections.abc import Callable
from functools import partial
from typing import ParamSpec

import numpy as np
import torch
from torch import nn

from models.shanghaitech_base_model import DownsampleBlock
from models.shanghaitech_base_model import TemporallySharedFullyConnection
from models.shanghaitech_base_model import UpsampleBlock

P = ParamSpec("P")


def _initialize_module(
    module: nn.Module,
    func: Callable[[torch.Tensor], torch.Tensor] | Callable[[torch.Tensor, P.args, P.kwargs], torch.Tensor],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    if not isinstance(module, (nn.Conv3d, TemporallySharedFullyConnection, nn.LSTM, DownsampleBlock, UpsampleBlock)):
        return

    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        func(module.weight, *args, **kwargs)

    if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
        func(module.bias, *args, **kwargs)


def init_weights_xavier_uniform(module: nn.Module) -> None:
    if not isinstance(module, (nn.Conv3d, TemporallySharedFullyConnection, nn.LSTM, DownsampleBlock, UpsampleBlock)):
        return

    gain = 1.0
    if hasattr(module, "weight") and isinstance(module.weight, torch.nn.ReLU):
        gain = nn.init.calculate_gain("relu")
        logging.info("Using RELU initializer")

    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.dim() > 1:
        nn.init.xavier_uniform_(module.weight, gain=gain)

    if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor) and module.bias.dim() > 1:
        nn.init.xavier_uniform_(module.bias, gain=gain)


initializers: dict[str, Callable[[nn.Module], None]] = dict(
    zeros=partial(_initialize_module, func=nn.init.zeros_),
    ones=partial(_initialize_module, func=nn.init.ones_),
    xavier_uniform=init_weights_xavier_uniform,
    none=lambda _: None,
)


def set_seeds(seed: int) -> None:
    """Set all seeds.

    Parameters
    ----------
    seed : int
        Seed

    """
    # Set the seed only if the user specified it
    np.seterr(all="raise")
    if seed == -1:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
