import random
from functools import lru_cache
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import numpy
import numpy as np
import skimage.io as io
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from tqdm import tqdm

from .base import ToFloatTensor3D
from .base import VideoAnomalyDetectionDataset
from .shanghaitech_test import ShanghaiTechTestHandler


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class ShanghaiTechDataHolder:
    """
    ShanghaiTech data holder class

    Parameters
    ----------
    root : str
        root folder of ShanghaiTech dataset
    clip_length : int
        number of frames that form a clip
    stride : int
        for creating a clip what should be the size of sliding window
    """

    def __init__(
        self,
        root: Path,
        seed: int,
        clip_length: int = 16,
        stride: int = 1,
    ) -> None:
        self.root: Path = root
        self.clip_length = clip_length
        self.stride = stride
        self.shape = (3, clip_length, 256, 512)
        self.train_dir = root / "training" / "nobackground_frames_resized"
        # Transform
        self.transform = transforms.Compose([ToFloatTensor3D(normalize=True)])
        self.seed = seed

    def get_test_data(self) -> VideoAnomalyDetectionDataset:
        """Load test dataset

        Returns
        -------
        ShanghaiTech : Dataset
            Custom dataset to handle ShanghaiTech data

        """
        return ShanghaiTechTestHandler(self.root)

    def get_train_data(self) -> "MySHANGHAI":
        """Load train dataset

        Parameters
        ----------
        """

        # Load all ids
        self.train_ids = self.load_train_ids()
        # Create clips with given clip_length and stride
        self.train_clips = self.create_clips(
            self.train_dir, self.train_ids, clip_length=self.clip_length, stride=self.stride
        )
        return MySHANGHAI(self.train_clips, self.transform, clip_length=self.clip_length)

    def get_loaders(
        self, batch_size: int, shuffle_train: bool = True, pin_memory: bool = False, num_workers: int = 0
    ) -> Tuple[DataLoader[Tuple[torch.Tensor, int]], DataLoader[torch.Tensor]]:
        """Returns MVtec dataloaders

        Parameters
        ----------
        batch_size : int
            Size of the batch to
        shuffle_train : bool
            If True, shuffles the training dataset
        pin_memory : bool
            If True, pin memeory
        num_workers : int
            Number of dataloader workers

        Returns
        -------
        loaders : DataLoader
            Train and test data loaders

        """
        g = torch.Generator()
        if self.seed != -1:
            g.manual_seed(self.seed)
        train_loader = DataLoader(
            dataset=self.get_train_data(),
            batch_size=batch_size,
            shuffle=shuffle_train,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        g = torch.Generator()
        if self.seed != -1:
            g.manual_seed(self.seed)
        test_loader = DataLoader(
            dataset=self.get_test_data(),
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return train_loader, test_loader

    @lru_cache(maxsize=None)
    def load_train_ids(self) -> Tuple[str, ...]:
        """
        Loads the set of all train video ids.
        :return: The list of train ids.
        """
        return tuple(sorted(d.name for d in self.train_dir.iterdir() if d.is_dir()))

    @staticmethod
    def create_clips(dir_path: Path, ids: Tuple[str, ...], clip_length: int = 16, stride: int = 1) -> np.ndarray:
        """
        Gets frame directory and ids of the directories in the frame dir
        Creates clips which consist of number of clip_length at each clip.
        Clips are created in a sliding window fashion. Default window slide is 1
        but stride controls the window slide
        Example: for default parameters first clip is [001.jpg, 002.jpg, ...,016.jpg]
        second clip would be [002.jpg, 003.jpg, ..., 017.jpg]
        If read_target is True then it will try to read from test directory
        If read_target is False then it will populate the array with all zeros
        :return: clips:: numpy array with (num_clips,clip_length) shape
                 ground_truths:: numpy array with (num_clips,clip_length) shape
        """
        clips: List[List[Path]] = list()
        print(f"Creating clips for {dir_path} dataset with length {clip_length}...")
        for idx in tqdm(ids):
            frames = sorted(x for x in (dir_path / idx).iterdir() if x.suffix == ".jpg")
            # Slide the window with stride to collect clips
            for window in range(0, len(frames) - clip_length + 1, stride):
                clips.append(frames[window : window + clip_length])
        return np.array(clips)


class MySHANGHAI(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, clips: np.ndarray, transform: Optional[Compose] = None, clip_length: int = 16) -> None:
        self.clips = clips
        self.transform = transform
        self.shape = (3, clip_length, 256, 512)

    def __len__(self) -> int:
        return len(self.clips)

    def load(self, index: int) -> np.ndarray:
        return np.stack([np.uint8(io.imread(img_path)) for img_path in self.clips[index]])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
            targets are all 0 target
        """
        # index = int(torch.randint(0, len(self.clips), size=(1,)).item())
        sample = self.load(index)
        sample_t = self.transform(sample) if self.transform else torch.from_numpy(sample)
        return sample_t, index
