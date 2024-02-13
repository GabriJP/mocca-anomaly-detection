from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from torch.utils.data import default_collate

T_NET_DTYPE = torch.float32
NP_NET_DTYPE = np.float32
OP_DTYPE = np.float64

U8_A = npt.NDArray[np.uint8]
NET_A = npt.NDArray[NP_NET_DTYPE]
OP_A = npt.NDArray[OP_DTYPE]


class DatasetBase(Dataset[torch.Tensor], ABC):
    """
    Base class for all datasets.
    """

    @abstractmethod
    def test(self, video_id: str, *args: Any) -> None:
        """
        Sets the dataset in test mode.
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int, int]:
        """
        Returns the shape of examples.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of examples.
        """
        pass

    @abstractmethod
    def __getitem__(self, i: int) -> torch.Tensor:
        """
        Provides the i-th example.
        """
        pass


class VideoAnomalyDetectionDataset(DatasetBase, ABC):
    """
    Base class for all video anomaly detection datasets.
    """

    @property
    def collate_fn(self) -> Callable[[list[Any]], Any]:
        return default_collate

    @property
    @abstractmethod
    def test_videos(self) -> list[str]:
        """
        Returns all test video ids.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of examples.
        """
        pass

    @property
    def raw_shape(self) -> tuple[int, int, int, int]:
        """
        Workaround!
        """
        return self.shape

    @abstractmethod
    def __getitem__(self, i: int) -> torch.Tensor:
        """
        Provides the i-th example.
        """
        pass

    @abstractmethod
    def load_test_sequence_gt(self, video_id: str) -> npt.NDArray[np.uint8]:
        """
        Loads the groundtruth of a test video in memory.
        :param video_id: the id of the test video for which the groundtruth has to be loaded.
        :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
        """
        pass


class ToFloatTensor3D:
    """Convert videos to FloatTensors"""

    def __init__(self, normalize: bool = True) -> None:
        self._normalize = normalize

    def __call__(self, sample: Sequence[npt.NDArray[np.uint8]] | npt.NDArray[np.uint8]) -> torch.Tensor:
        if len(sample) == 3:
            x, _, _ = sample
        else:
            x = sample

        # swap color axis because
        # numpy image: T x H x W x C
        x = x.transpose((3, 0, 1, 2))
        # Y = Y.transpose(3, 0, 1, 2)

        if self._normalize:
            x = x / 255.0

        return torch.from_numpy(x.astype(NP_NET_DTYPE))
