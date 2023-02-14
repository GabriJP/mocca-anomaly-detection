from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import numpy.typing as npt
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

    def __init__(self, root: Path, clip_length: int = 16, stride: int = 1) -> None:
        self.root: Path = root
        self.clip_length = clip_length
        self.stride = stride
        self.shape = (3, clip_length, 256, 512)
        self.train_dir = root / "training" / "nobackground_frames_resized"
        # Transform
        self.transform = transforms.Compose([ToFloatTensor3D(normalize=True)])

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
        train_loader = DataLoader(
            dataset=self.get_train_data(),
            batch_size=batch_size,
            shuffle=shuffle_train,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            dataset=self.get_test_data(), batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers
        )
        return train_loader, test_loader

    def load_train_ids(self) -> List[str]:
        """
        Loads the set of all train video ids.
        :return: The list of train ids.
        """
        return sorted(d.name for d in self.train_dir.iterdir() if d.is_dir())

    @staticmethod
    def create_clips(dir_path: Path, ids: List[str], clip_length: int = 16, stride: int = 1) -> npt.NDArray[np.str_]:
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
        clips = []
        print(f"Creating clips for {dir_path} dataset with length {clip_length}...")
        for idx in tqdm(ids):
            frames = sorted(x for x in (dir_path / idx).iterdir() if x.stem == ".jpg")
            num_frames = len(frames)
            # Slide the window with stride to collect clips
            for window in range(0, num_frames - clip_length + 1, stride):
                clips.append(frames[window : window + clip_length])
        return np.array(clips)


class MySHANGHAI(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, clips: npt.NDArray[np.str_], transform: Optional[Compose] = None, clip_length: int = 16):
        self.clips = clips
        self.transform = transform
        self.shape = (3, clip_length, 256, 512)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
            targets are all 0 target
        """
        index_ = int(torch.randint(0, len(self.clips), size=(1,)).item())
        sample = np.stack([np.uint8(io.imread(img_path)) for img_path in self.clips[index_]])
        sample_t = self.transform(sample) if self.transform else torch.from_numpy(sample)
        return sample_t, index_


def get_target_label_idx(labels: npt.NDArray[np.uint8], targets: Sequence[int]) -> List[int]:
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.Tensor, scale: str = "l2") -> torch.Tensor:
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ("l1", "l2")

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    x_scale = torch.mean(torch.abs(x)) if scale == "l1" else torch.sqrt(torch.sum(x**2)) / n_features

    x /= x_scale

    return x
