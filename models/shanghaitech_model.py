from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shanghaitech_base_model import BaseModule
from .shanghaitech_base_model import DownsampleBlock
from .shanghaitech_base_model import TemporallySharedFullyConnection
from .shanghaitech_base_model import UpsampleBlock


class Selector(BaseModule):
    def __init__(self, idx: int) -> None:
        super().__init__()
        """
        sizes = [[ch, time , h, w], ...]
        """
        self.idx = idx
        self.sizes = [[8, 16, 128, 256], [16, 16, 64, 128], [32, 8, 32, 64], [64, 8, 16, 32], [64, 4, 8, 16]]
        mid_features_size = 256
        # self.adaptive = nn.AdaptiveMaxPool3d(output_size=(None,16,16))
        self.cv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=self.sizes[idx][0],
                out_channels=self.sizes[idx][0] * 2,
                kernel_size=3,
                padding=1,
                stride=(1, 2, 2),
            ),
            nn.BatchNorm3d(num_features=self.sizes[idx][0] * 2),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.sizes[idx][0] * 2, out_channels=self.sizes[idx][0] * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm3d(num_features=self.sizes[idx][0] * 4),
            nn.ReLU(),
            nn.Conv3d(in_channels=self.sizes[idx][0] * 4, out_channels=self.sizes[idx][0] * 4, kernel_size=1),
        )

        self.fc = nn.Sequential(
            TemporallySharedFullyConnection(
                in_features=(self.sizes[self.idx][0] * 4 * self.sizes[self.idx][2] // 2 * self.sizes[self.idx][3] // 2),
                out_features=mid_features_size,
                bias=True,
            ),
            nn.BatchNorm1d(self.sizes[self.idx][1]),
            nn.ReLU(),
            TemporallySharedFullyConnection(
                in_features=mid_features_size, out_features=self.sizes[self.idx][0], bias=True
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        _, t, _, _ = self.sizes[self.idx]
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, t, (self.sizes[self.idx][0] * 4 * self.sizes[self.idx][2] // 2 * self.sizes[self.idx][3] // 2))
        x = self.fc(x)
        return x


def build_lstm(input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool) -> nn.LSTM:
    return nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=True,
        dropout=dropout,
        bidirectional=bidirectional,
    )


def shape_lstm_input(o: torch.Tensor) -> torch.Tensor:
    # batch, channel, height, width
    o = o.permute(0, 2, 1, 3, 4)
    kernel_size = (1, o.shape[-2], o.shape[-1])
    o = F.avg_pool3d(o, kernel_size).squeeze() if o.ndimension() > 3 else o
    # batch, time, channel
    return o if o.ndim > 2 else o.unsqueeze(0)


class ShanghaiTechEncoder(BaseModule):
    """
    ShanghaiTech model encoder.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        code_length: int,
        load_lstm: bool,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        use_selectors: bool,
    ) -> None:
        """
        Class constructor:
        :param input_shape: the shape of UCSD Ped2 samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super().__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.load_lstm = load_lstm
        self.use_selectors = use_selectors

        c, t, h, w = input_shape

        activation_fn = nn.LeakyReLU()

        # Convolutional network
        # self.conv = nn.Sequential(
        self.conv_1 = DownsampleBlock(channel_in=c, channel_out=8, activation_fn=activation_fn, stride=(1, 2, 2))
        self.conv_2 = DownsampleBlock(channel_in=8, channel_out=16, activation_fn=activation_fn, stride=(1, 2, 2))
        self.conv_3 = DownsampleBlock(channel_in=16, channel_out=32, activation_fn=activation_fn, stride=(2, 2, 2))
        self.conv_4 = DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn, stride=(1, 2, 2))
        self.conv_5 = DownsampleBlock(channel_in=64, channel_out=64, activation_fn=activation_fn, stride=(2, 2, 2))
        if load_lstm:
            self.lstm_1 = build_lstm(8, hidden_size, num_layers, dropout, bidirectional)
            self.lstm_2 = build_lstm(16, hidden_size, num_layers, dropout, bidirectional)
            self.lstm_3 = build_lstm(32, hidden_size, num_layers, dropout, bidirectional)
            self.lstm_4 = build_lstm(64, hidden_size, num_layers, dropout, bidirectional)
            self.lstm_5 = build_lstm(64, hidden_size, num_layers, dropout, bidirectional)
        # )

        # Features selector models (MLPs)
        if self.use_selectors:
            self.sel1 = Selector(0)
            self.sel2 = Selector(1)
            self.sel3 = Selector(2)
            self.sel4 = Selector(3)
            self.sel5 = Selector(4)

        self.deepest_shape = (64, t // 4, h // 32, w // 32)

        # FC network
        dc, dt, dh, dw = self.deepest_shape
        # self.tdl = nn.Sequential(
        self.tdl_1 = TemporallySharedFullyConnection(in_features=(dc * dh * dw), out_features=512)
        self.tanh = nn.Tanh()
        self.tdl_2 = TemporallySharedFullyConnection(in_features=512, out_features=code_length)
        self.sigmoid = nn.Sigmoid()
        if load_lstm:
            self.lstm_tdl_1 = build_lstm(512, hidden_size, num_layers, dropout, bidirectional)
            self.lstm_tdl_2 = build_lstm(code_length, hidden_size, num_layers, dropout, bidirectional)
        # )

    @staticmethod
    def get_names() -> List[str]:
        d_lstm_names = [f"conv_lstm_o_{i}" for i in range(5)]
        d_lstm_names.extend(f"tdl_lstm_o_{i}" for i in range(2))
        return d_lstm_names

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward propagation.
        :param x: the input batch of patches.
        :return: the batch of latent vectors.
        """
        h = x
        # h = self.conv(h)
        o1 = self.conv_1(h)
        o2 = self.conv_2(o1)
        o3 = self.conv_3(o2)
        o4 = self.conv_4(o3)
        o5 = self.conv_5(o4)

        # Reshape for fully connected subnetwork (flatten)
        c, t, height, width = self.deepest_shape
        h = torch.transpose(o5, 1, 2).contiguous()
        h = h.view(-1, t, (c * height * width))
        # o = self.tdl(h)
        o_tdl_1 = self.tdl_1(h)
        o_tdl_1_t = self.tanh(o_tdl_1)
        o_tdl_2 = self.tdl_2(o_tdl_1_t)
        o_tdl_2_s = self.sigmoid(o_tdl_2)

        if not self.load_lstm:
            return o_tdl_2_s

        if self.use_selectors:
            o1_lstm, _ = self.lstm_1(self.sel1(o1))
            o2_lstm, _ = self.lstm_2(self.sel2(o2))
            o3_lstm, _ = self.lstm_3(self.sel3(o3))
            o4_lstm, _ = self.lstm_4(self.sel4(o4))
            o5_lstm, _ = self.lstm_5(self.sel5(o5))
        else:
            o1_lstm, _ = self.lstm_1(shape_lstm_input(o1))
            o2_lstm, _ = self.lstm_2(shape_lstm_input(o2))
            o3_lstm, _ = self.lstm_3(shape_lstm_input(o3))
            o4_lstm, _ = self.lstm_4(shape_lstm_input(o4))
            o5_lstm, _ = self.lstm_5(shape_lstm_input(o5))

        o1_tdl_lstm, _ = self.lstm_tdl_1(o_tdl_1_t)
        o2_tdl_lstm, _ = self.lstm_tdl_2(o_tdl_2_s)

        conv_lstms = [o1_lstm[:, -1], o2_lstm[:, -1], o3_lstm[:, -1], o4_lstm[:, -1], o5_lstm[:, -1]]
        tdl_lstms = [o1_tdl_lstm[:, -1], o2_tdl_lstm[:, -1]]

        d_lstms = {f"conv_lstm_o_{i}": conv_lstm for i, conv_lstm in enumerate(conv_lstms)}
        d_lstms.update((f"tdl_lstm_o_{i}", tdl_lstm) for i, tdl_lstm in enumerate(tdl_lstms))
        return o_tdl_2_s, d_lstms


class ShanghaiTechDecoder(BaseModule):
    """
    ShanghaiTech model decoder.
    """

    def __init__(
        self, code_length: int, deepest_shape: Tuple[int, int, int, int], output_shape: Tuple[int, int, int, int]
    ) -> None:
        """
        Class constructor.
        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of UCSD Ped2 samples.
        """
        super().__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        dc, dt, dh, dw = deepest_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.tdl = nn.Sequential(
            TemporallySharedFullyConnection(in_features=code_length, out_features=512),
            nn.Tanh(),
            TemporallySharedFullyConnection(in_features=512, out_features=(dc * dh * dw)),
            activation_fn,
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(
                channel_in=dc, channel_out=64, activation_fn=activation_fn, stride=(2, 2, 2), output_padding=(1, 1, 1)
            ),
            UpsampleBlock(
                channel_in=64, channel_out=32, activation_fn=activation_fn, stride=(1, 2, 2), output_padding=(0, 1, 1)
            ),
            UpsampleBlock(
                channel_in=32, channel_out=16, activation_fn=activation_fn, stride=(2, 2, 2), output_padding=(1, 1, 1)
            ),
            UpsampleBlock(
                channel_in=16, channel_out=8, activation_fn=activation_fn, stride=(1, 2, 2), output_padding=(0, 1, 1)
            ),
            UpsampleBlock(
                channel_in=8, channel_out=8, activation_fn=activation_fn, stride=(1, 2, 2), output_padding=(0, 1, 1)
            ),
            nn.Conv3d(in_channels=8, out_channels=output_shape[0], kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.tdl(h)

        # Reshape to encoder's deepest convolutional shape
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(len(h), *self.deepest_shape)

        h = self.conv(h)
        o = h

        return o


class ShanghaiTech(BaseModule):
    """
    Model for ShanghaiTech video anomaly detection.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        code_length: int,
        load_lstm: bool = False,
        hidden_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        use_selectors: bool = False,
    ) -> None:
        """
        Class constructor.
        :param input_shape: the shape of UCSD Ped2 samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super().__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.load_lstm = load_lstm
        # Build encoder
        self.encoder = ShanghaiTechEncoder(
            input_shape=input_shape,
            code_length=code_length,
            load_lstm=load_lstm,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_selectors=use_selectors,
        )

        # Build decoder
        self.decoder = ShanghaiTechDecoder(
            code_length=code_length, deepest_shape=self.encoder.deepest_shape, output_shape=input_shape
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward propagation.
        :param x: the input batch of patches.
        :return: a tuple of torch.Tensors holding reconstructions, latent vectors and CPD estimates.
        """
        # Produce representations
        if self.load_lstm:
            z, d_lstms = self.encoder(x)
            x_r = self.decoder(z)
            x_r = x_r.view(-1, *self.input_shape)
            return x_r, z, d_lstms
        else:
            z = self.encoder(x)
            x_r = self.decoder(z)
            x_r = x_r.view(-1, *self.input_shape)
            return x_r, z
