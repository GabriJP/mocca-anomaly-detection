from pathlib import Path

from .shanghaitech import ContinuousShanghaiTechDataHolder
from .shanghaitech import ShanghaiTechDataHolder

AVAILABLE_DATASETS = ("cifar10", "ShanghaiTech", "MVTec_Anomaly")


class DataManager:
    """ "Init class to manage and load data"""

    def __init__(
        self,
        dataset_name: str,
        data_path: Path,
        normal_class: int,
        seed: int,
        clip_length: int = 16,
        only_test: bool = False,
    ):
        """Init the DataManager

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        data_path : Path
            Path to the dataset
        normal_class : int
            Index of the normal class
        clip_length: int
            Number of video frames in each clip (ShanghaiTech only)
        only_test : bool
            True if we are in test model, False otherwise

        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.normal_class = normal_class
        self.seed = seed
        self.clip_length = clip_length
        self.only_test = only_test

        # Immediately check if the data are available
        self.__check_dataset()

    def __check_dataset(self) -> None:
        """Checks if the required dataset is available"""
        assert self.dataset_name in AVAILABLE_DATASETS, f"{self.dataset_name} dataset is not available"
        assert self.data_path.exists(), f"{self.dataset_name} dataset is available but not found at: \n{self.data_path}"

    def get_data_holder(self) -> ShanghaiTechDataHolder:
        """Returns the data holder for the required dataset

        Rerurns
        -------
        MVTec_DataHolder : MVTec_DataHolder
            Class to handle datasets

        """

        return ShanghaiTechDataHolder(self.data_path, self.seed, clip_length=self.clip_length)


class ContinuousDataManager(DataManager):
    def get_data_holder(self) -> ShanghaiTechDataHolder:
        return ContinuousShanghaiTechDataHolder(self.data_path, self.seed, clip_length=self.clip_length)
