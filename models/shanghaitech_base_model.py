from functools import reduce
from operator import mul
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn


class BaseModule(nn.Module):
    """
    Implements the basic module.
    All other modules inherit from this one
    """

    def load_w(self, checkpoint_path: str) -> None:
        """
        Loads a checkpoint into the state_dict.
        :param checkpoint_path: the checkpoint file to be loaded.
        """
        self.load_state_dict(torch.load(checkpoint_path))

    def __repr__(self) -> str:
        """
        String representation
        """
        good_old = repr(super())

        # return good_old + '\n' + addition
        return good_old

    @property
    def n_parameters(self) -> int:
        """
        Number of parameters of the model.
        """
        n_parameters = 0
        for p in self.parameters():
            if hasattr(p, "mask"):
                n_parameters += int(torch.sum(p.mask).item())
            else:
                n_parameters += reduce(mul, p.shape)
        return int(n_parameters)


class MaskedConv3d(BaseModule, nn.Conv3d):
    """
    Implements a Masked Convolution 3D.
    This is a 3D Convolution that cannot access future frames.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", self.weight.data.clone())
        _, _, kT, kH, kW = self.weight.size()
        self.mask.fill_(1)  # type: ignore
        self.mask[:, :, kT // 2 + 1 :] = 0  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.
        :param x: the input tensor.
        :return: the output tensor as result of the convolution.
        """
        self.weight.data *= self.mask
        return super().forward(x)


class TemporallySharedFullyConnection(BaseModule):
    """
    Implements a temporally-shared fully connection.
    Processes a time series of feature vectors and performs
    the same linear projection to all of them.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Class constructor.
        :param in_features: number of input features.
        :param out_features: number of output features.
        :param bias: whether or not to add bias.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # the layer to be applied at each timestep
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.
        :param x: layer input. Has shape=(batchsize, seq_len, in_features).
        :return: layer output. Has shape=(batchsize, seq_len, out_features)
        """
        b, t, d = x.size()

        output = []
        for i in range(0, t):
            # apply dense layer
            output.append(self.linear(x[:, i, :]))

        return torch.stack(output, 1)


def residual_op(
    x: torch.Tensor, functions: List[nn.Module], bns: List[Union[nn.Module, None]], activation_fn: nn.Module
) -> torch.Tensor:
    """
    Implements a global residual operation.
    :param x: the input tensor.
    :param functions: a list of functions (nn.Modules).
    :param bns: a list of optional batch-norm layers.
    :param activation_fn: the activation to be applied.
    :return: the output of the residual operation.
    """
    f1, f2, f3 = functions
    bn1, bn2, bn3 = bns

    assert len(functions) == len(bns) == 3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)  # type: ignore

    # A-branch
    ha = x
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)

    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)

    # B-branch
    hb = x
    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)

    # Residual connection
    out = ha + hb
    return activation_fn(out)


class BaseBlock(BaseModule):
    """Base class for all blocks."""

    def __init__(
        self, channel_in: int, channel_out: int, activation_fn: nn.Module, use_bn: bool = True, use_bias: bool = True
    ) -> None:
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super().__init__()

        assert not (use_bn and use_bias), "Using bias=True with batch_normalization is forbidden."

        self._channel_in = channel_in
        self._channel_out = channel_out
        self._activation_fn = activation_fn
        self._use_bn = use_bn
        self._bias = use_bias

    def get_bn(self) -> Optional[nn.Module]:
        """
        Returns batch norm layers, if needed.
        :return: batch norm layers or None
        """
        return nn.BatchNorm3d(num_features=self._channel_out) if self._use_bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract forward function. Not implemented.
        """
        raise NotImplementedError


class DownsampleBlock(BaseBlock):
    """Implements a Downsampling block for videos (Fig. 1ii)."""

    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        activation_fn: nn.Module,
        stride: Tuple[int, int, int],
        use_bn: bool = True,
        use_bias: bool = False,
    ) -> None:
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param stride: the stride to be applied to downsample feature maps.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super().__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
        self.stride = stride

        # Convolutions
        self.conv1a = MaskedConv3d(
            in_channels=channel_in, out_channels=channel_out, kernel_size=3, padding=1, stride=stride, bias=use_bias
        )
        self.conv1b = MaskedConv3d(
            in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1, bias=use_bias
        )
        self.conv2a = nn.Conv3d(
            in_channels=channel_in, out_channels=channel_out, kernel_size=1, padding=0, stride=stride, bias=use_bias
        )

        # Batch Normalization layers
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn,
        )


class UpsampleBlock(BaseBlock):
    """Implements a Upsampling block for videos (Fig. 1ii)."""

    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        activation_fn: nn.Module,
        stride: Tuple[int, int, int],
        output_padding: Tuple[int, int, int],
        use_bn: bool = True,
        use_bias: bool = False,
    ) -> None:
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param stride: the stride to be applied to upsample feature maps.
        :param output_padding: the padding to be added applied output feature maps.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super().__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
        self.stride = stride
        self.output_padding = output_padding

        # Convolutions
        self.conv1a = nn.ConvTranspose3d(
            channel_in,
            channel_out,
            kernel_size=5,
            padding=2,
            stride=stride,
            output_padding=output_padding,
            bias=use_bias,
        )
        self.conv1b = nn.Conv3d(
            in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1, bias=use_bias
        )
        self.conv2a = nn.ConvTranspose3d(
            channel_in,
            channel_out,
            kernel_size=5,
            padding=2,
            stride=stride,
            output_padding=output_padding,
            bias=use_bias,
        )

        # Batch Normalization layers
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn,
        )
