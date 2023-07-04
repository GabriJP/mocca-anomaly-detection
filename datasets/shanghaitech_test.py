from functools import cached_property
from pathlib import Path
from shutil import rmtree
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt
import skimage.io as io
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .base import ToFloatTensor3D
from .base import VideoAnomalyDetectionDataset


class ShanghaiTechTestHandler(VideoAnomalyDetectionDataset):
    def __init__(self, path: Path) -> None:
        """
        Class constructor.
        :param path: The folder in which ShanghaiTech is stored.
        """
        super().__init__()
        self.path = path
        # Test directory
        self.test_dir = path / "testing"
        # Transform
        self.transform = transforms.Compose([ToFloatTensor3D(normalize=True)])
        # Other utilities
        self.cur_len = 0
        self.cur_video_id: str
        self.cur_video_frames: npt.NDArray[np.uint8]
        self.cur_video_gt: npt.NDArray[np.uint8]

    @cached_property
    def test_ids(self) -> List[str]:
        """
        Loads the set of all test video ids.
        :return: The list of test ids.
        """
        return sorted(p.stem for p in (Path(self.test_dir) / "test_frame_mask").iterdir() if p.suffix == ".npy")

    def load_test_sequence_frames(self, video_id: str) -> npt.NDArray[np.uint8]:
        """
        Loads a test video in memory.
        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """
        sequence_dir = self.test_dir / "nobackground_frames_resized" / video_id
        img_list = sorted(p for p in sequence_dir.iterdir() if p.suffix == ".jpg")
        # print(f"Creating clips for {sequence_dir} dataset with length {t}...")
        return np.stack([np.uint8(io.imread(img_path)) for img_path in img_list])

    def load_test_sequence_gt(self, video_id: str) -> npt.NDArray[np.uint8]:
        """
        Loads the groundtruth of a test video in memory.
        :param video_id: the id of the test video for which the groundtruth has to be loaded.
        :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
        """
        return np.load(str(self.test_dir / "test_frame_mask" / f"{video_id}.npy"))

    def test(self, video_id: str, *_: Any) -> None:
        """
        Sets the dataset in test mode.
        :param video_id: the id of the video to test.
        """
        c, t, h, w = self.shape
        self.cur_video_id = video_id
        self.cur_video_frames = self.load_test_sequence_frames(video_id)
        self.cur_video_gt = self.load_test_sequence_gt(video_id)
        self.cur_len = len(self.cur_video_frames) - t + 1

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """
        Returns the shape of examples being fed to the model.
        """
        return 3, 16, 256, 512

    @property
    def test_videos(self) -> List[str]:
        """
        Returns all available test videos.
        """
        return self.test_ids

    def __len__(self) -> int:
        """
        Returns the number of examples.
        """
        return self.cur_len

    def __getitem__(self, i: int) -> torch.Tensor:
        """
        Provides the i-th example.
        """
        c, t, h, w = self.shape
        clip = self.cur_video_frames[i : i + t]
        sample = clip
        # Apply transform
        return self.transform(sample) if self.transform else torch.from_numpy(sample)

    def __repr__(self) -> str:
        return f"ShanghaiTech (video id = {self.cur_video_id})"


class ResultsAccumulator:
    """
    Accumulates results in a buffer for a sliding window
    results computation. Employed to get frame-level scores
    from clip-level scores.
    ` In order to recover the anomaly score of each
    frame, we compute the mean score of all clips in which it
    appears`
    """

    def __init__(self, nb_frames_per_clip: int) -> None:
        """
        Class constructor.
        :param nb_frames_per_clip: the number of frames each clip holds.
        """

        # These buffers rotate.
        self.buffer: npt.NDArray[np.float32] = np.zeros(shape=(nb_frames_per_clip,), dtype=np.float32)
        self.counts = np.zeros(shape=(nb_frames_per_clip,))

    def push(self, score: float) -> None:
        """
        Pushes the score of a clip into the buffer.
        :param score: the score of a clip
        """

        # Update buffer and counts
        self.buffer += score
        self.counts += 1

    def get_next(self) -> float:
        """
        Gets the next frame (the first in the buffer) score,
        computed as the mean of the clips in which it appeared,
        and rolls the buffers.
        :return: the averaged score of the frame exiting the buffer.
        """

        # Return first in buffer
        ret = self.buffer[0] / self.counts[0]

        # Roll time backwards
        self.buffer = np.roll(self.buffer, shift=-1)
        self.counts = np.roll(self.counts, shift=-1)

        # Zero out final frame (next to be filled)
        self.buffer[-1] = 0
        self.counts[-1] = 0

        return ret

    @property
    def results_left(self) -> int:
        """
        Returns the number of frames still in the buffer.
        """
        return int(np.sum(self.counts != 0))


class VideoAnomalyDetectionResultHelper:
    """
    Performs tests for video anomaly detection datasets (UCSD Ped2 or Shanghaitech).
    """

    def __init__(
        self,
        dataset: VideoAnomalyDetectionDataset,
        model: nn.Module,
        R: Dict[str, torch.Tensor],
        boundary: str,
        device: str,
        end_to_end_training: bool,
        debug: bool,
        output_file: Optional[Path],
    ) -> None:
        """
        Class constructor.
        :param dataset: dataset class.
        :param model: pytorch model to evaluate.
        :param output_file: text file where to save results.
        """
        self.dataset = dataset
        self.model = model
        self.keys = list(R.keys())
        self.R = R
        self.boundary = boundary
        self.device = device
        self.end_to_end_training = end_to_end_training
        self.debug = debug
        self.output_file = output_file

    def _get_scores(self, d_lstm: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Eval novelty scores
        dists = {k: torch.sum(d_lstm[k] ** 2, dim=1) for k in self.keys}
        scores = {k: torch.zeros((dist.shape[0],), device=self.device) for k, dist in dists.items()}
        overall_score = torch.zeros((dists[self.keys[0]].shape[0],), device=self.device)
        for k, dist in dists.items():
            if self.boundary == "soft":
                scores[k] += dist - self.R[k] ** 2
                overall_score += dist - self.R[k] ** 2
            else:
                scores[k] += dist
                overall_score += dist
        # Every value over the number of keys
        return {k: v / len(self.keys) for k, v in scores.items()}, overall_score / len(self.keys)

    @torch.no_grad()
    def test_video_anomaly_detection(self, view: bool = False) -> Tuple[npt.NDArray[np.float64], List[float]]:
        """
        Actually performs tests.
        """
        self.model.eval().to(self.device)

        c, t, h, w = self.dataset.raw_shape

        # Prepare a table to show results
        vad_table = self.empty_table

        # Set up container for anomaly scores from all test videos
        # oc: one class
        # rc: reconstruction
        # as: overall anomaly score
        global_oc = list()
        global_rc = list()
        global_as = list()
        global_as_by_layer: Dict[str, List[npt.NDArray[np.float64]]] = {k: list() for k in self.keys}
        global_y = list()
        global_y_by_layer: Dict[str, List[npt.NDArray[np.uint8]]] = {k: list() for k in self.keys}

        # Get accumulators
        ra_rc = ResultsAccumulator(nb_frames_per_clip=t)
        ra_oc = ResultsAccumulator(nb_frames_per_clip=t)
        ra_oc_by_layer = {k: ResultsAccumulator(nb_frames_per_clip=t) for k in self.keys}
        print(self.dataset.test_videos)

        # Start iteration over test videos
        for cl_idx, video_id in tqdm(
            enumerate(self.dataset.test_videos, start=1), total=len(self.dataset.test_videos), desc="Test on Video"
        ):
            view_root_path = Path.home() / "Escritorio" / "view" / "weights_name" / "dataset_name" / f"{cl_idx}"
            rmtree(view_root_path, ignore_errors=True)
            view_root_path.mkdir(parents=True)
            # Run the test
            self.dataset.test(video_id)
            loader = DataLoader(self.dataset, collate_fn=self.dataset.collate_fn)

            # Build score containers
            sample_rc = np.zeros(shape=(len(loader) + t - 1,))
            sample_oc = np.zeros(shape=(len(loader) + t - 1,))
            sample_oc_by_layer = {k: np.zeros(shape=(len(loader) + t - 1,)) for k in self.keys}
            sample_y = self.dataset.load_test_sequence_gt(video_id)

            view_img: npt.NDArray[np.uint8] = np.full((256, 512 * 2 + 5, 3), 255, dtype=np.uint8)
            for i, x in tqdm(
                enumerate(loader), total=len(loader), desc=f"Computing scores for {self.dataset}", leave=False
            ):
                # x.shape = [1, 3, 16, 256, 512]
                if view:
                    view_img[:, :512, :] = (np.transpose(x.numpy()[0], (1, 2, 3, 0))[-2] * 255).astype(
                        np.uint8, casting="unsafe"
                    )
                x = x.to(self.device)

                if self.end_to_end_training:
                    x_r, _, d_lstm = self.model(x)
                    if view:
                        view_img[:, -512:, :] = (np.transpose(x_r.cpu().numpy()[0], (1, 2, 3, 0))[-2] * 255).astype(
                            np.uint8, casting="unsafe"
                        )
                    cv2.imwrite(str(view_root_path / f"{i:03d}.png"), view_img)
                    recon_loss = torch.sum((x_r - x) ** 2, dim=tuple(range(1, x_r.dim())))
                else:
                    _, d_lstm = self.model(x)
                    recon_loss = torch.tensor([0.0])

                # Eval one class score for current clip
                oc_loss_by_layer, oc_overall_loss = self._get_scores(d_lstm)

                # Feed results accumulators
                ra_rc.push(recon_loss.item())
                sample_rc[i] = ra_rc.get_next()
                ra_oc.push(oc_overall_loss.item())
                sample_oc[i] = ra_oc.get_next()

                for k in self.keys:
                    if k in ("tdl_lstm_o_0", "tdl_lstm_o_1"):
                        continue
                    ra_oc_by_layer[k].push(oc_loss_by_layer[k].item())
                    sample_oc_by_layer[k][i] = ra_oc_by_layer[k].get_next()

            # Get last results layer by layer
            for k in self.keys:
                if k in ("tdl_lstm_o_0", "tdl_lstm_o_1"):
                    continue

                while ra_oc_by_layer[k].results_left != 0:
                    index = -ra_oc_by_layer[k].results_left
                    sample_oc_by_layer[k][index] = ra_oc_by_layer[k].get_next()

                min_, ptp = sample_oc_by_layer[k].min(), sample_oc_by_layer[k].ptp()

                # Computes the normalized novelty score given likelihood scores, reconstruction scores
                # and normalization coefficients (Eq. 9-10).
                sample_ns = (sample_oc_by_layer[k] - min_) / ptp

                # Update global scores (used for global metrics)
                global_as_by_layer[k].append(sample_ns)
                global_y_by_layer[k].append(sample_y)

                try:
                    # Compute AUROC for this video
                    vad_table.add_row([k, video_id, roc_auc_score(sample_y, sample_ns), 0.0, 0.0])
                except ValueError as e:
                    print(f"Exception {e}: continue")
                    # This happens for sequences in which all frames are abnormal
                    # Skipping this row in the table (the sequence will still count for global metrics)
                    continue

            # Get last results
            while ra_oc.results_left != 0:
                index = -ra_oc.results_left
                sample_oc[index] = ra_oc.get_next()
                sample_rc[index] = ra_rc.get_next()

            min_oc, ptp_oc, min_rc, ptp_rc = sample_oc.min(), sample_oc.ptp(), sample_rc.min(), sample_rc.ptp()

            # Computes the normalized novelty score given likelihood scores, reconstruction scores
            # and normalization coefficients (Eq. 9-10).
            sample_oc = (sample_oc - min_oc) / ptp_oc
            sample_rc = (sample_rc - min_rc) / ptp_rc if ptp_rc > 0 else np.zeros_like(sample_rc)
            sample_as = sample_oc + sample_rc

            if view:
                np.save(str(view_root_path / "sample_oc"), sample_oc)
                np.save(str(view_root_path / "sample_rc"), sample_rc)
                np.save(str(view_root_path / "sample_as"), sample_as)
                np.save(str(view_root_path / "sample_y"), sample_y)

            # Update global scores (used for global metrics)
            global_oc.append(sample_oc)
            global_rc.append(sample_rc)
            global_as.append(sample_as)
            global_y.append(sample_y)

            try:
                # Compute AUROC for this video
                vad_table.add_row(
                    [
                        "Overall",
                        video_id,
                        roc_auc_score(sample_y, sample_oc),  # one class metric
                        roc_auc_score(sample_y, sample_rc),  # reconstruction metric
                        roc_auc_score(sample_y, sample_as),  # anomaly score
                    ]
                )
            except ValueError as e:
                print(f"Exception {e}: continue")
                # This happens for sequences in which all frames are abnormal
                # Skipping this row in the table (the sequence will still count for global metrics)
                continue

            if self.debug:
                break

        # Compute global AUROC and print table
        for k in self.keys:
            if k in ("tdl_lstm_o_0", "tdl_lstm_o_1"):
                continue
            vad_table.add_row(
                [
                    k,
                    "avg",
                    roc_auc_score(np.concatenate(global_y_by_layer[k]), np.concatenate(global_as_by_layer[k])),
                    0.0,
                    0.0,
                ]
            )

        # Compute global AUROC and print table
        global_y_conc = np.concatenate(global_y)
        global_oc_conc = np.concatenate(global_oc)
        global_metrics = [
            roc_auc_score(global_y_conc, global_oc_conc),  # one class metric
            roc_auc_score(global_y_conc, np.concatenate(global_rc)),  # reconstruction metric
            roc_auc_score(global_y_conc, np.concatenate(global_as)),  # anomaly score
        ]

        vad_table.add_row(["Overall", "avg", *global_metrics])
        print(vad_table)

        # Save table
        if self.output_file is not None:
            with self.output_file.open("w") as f:
                f.write(str(vad_table))

        return global_oc_conc, global_metrics

    @property
    def empty_table(self) -> PrettyTable:
        """
        Sets up a nice ascii-art table to hold results.
        This table is suitable for the video anomaly detection setting.
        :return: table to be filled with auroc metrics.
        """
        table = PrettyTable()
        table.field_names = ["Layer key", "VIDEO-ID", "OC metric", "Recon metric", "AUROC-AS"]
        table.float_format = "0.3"
        return table
