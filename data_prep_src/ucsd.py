from itertools import product
from pathlib import Path
from shutil import rmtree
from typing import Final
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt

from .utils import load_video
from .utils import n_subpaths
from .utils import process_background_cpu
from .utils import process_background_gpu
from .utils import relative_symlink
from .utils import remove_background
from .utils import save_video

U8_NDTYPE = npt.NDArray[np.uint8]


UCSD_NAMES: Final[Tuple[str, str]] = ("UCSDped1", "UCSDped2")


def process_ucsd_gt(data_root: Path) -> None:
    ped12_fm_path = data_root / "UCSDped12" / "testing" / "test_frame_mask"
    ped12_pm_path = data_root / "UCSDped12" / "testing" / "test_pixel_mask"
    rmtree(ped12_fm_path, ignore_errors=True)
    rmtree(ped12_pm_path, ignore_errors=True)
    ped12_fm_path.mkdir(parents=True)
    ped12_pm_path.mkdir(parents=True)

    for current_ucsd_name in UCSD_NAMES:
        current_ped_fm_path = data_root / current_ucsd_name / "testing" / "test_frame_mask"
        current_ped_pm_path = data_root / current_ucsd_name / "testing" / "test_pixel_mask"
        rmtree(current_ped_fm_path, ignore_errors=True)
        rmtree(current_ped_pm_path, ignore_errors=True)
        current_ped_fm_path.mkdir(parents=True)
        current_ped_pm_path.mkdir(parents=True)

        dot_m_path = data_root / "pre" / current_ucsd_name / "Test" / f"{current_ucsd_name}.m"
        dot_m_lines = dot_m_path.read_text().splitlines()[1:]
        test_clips = sorted(p for p in dot_m_path.parent.iterdir() if p.is_dir() and not p.name.endswith("_gt"))
        clip_path: Path
        clip_gt_line: str
        for clip_path, clip_gt_line in zip(test_clips, dot_m_lines):
            n_frames = n_subpaths(clip_path, lambda p: p.suffix == ".tif")
            clip_labels: npt.NDArray[np.uint8] = np.zeros(n_frames, dtype=np.uint8)
            ranges = clip_gt_line[clip_gt_line.rfind("[") + 1 : clip_gt_line.rfind("]")].split(",")
            for positive_range in ranges:
                positive_range = positive_range.strip()
                start, end = positive_range.split(":")
                clip_labels[int(start) : int(end)] = 1

            np_path = current_ped_fm_path / f"{clip_path.name}.npy"
            np.save(np_path, clip_labels)
            relative_symlink(ped12_fm_path / f"P{current_ucsd_name[-1]}_{clip_path.name}.npy", np_path)

        for gt_path in dot_m_path.parent.iterdir():
            if not gt_path.is_dir() or not gt_path.name.endswith("_gt"):
                continue
            video: U8_NDTYPE = np.empty((n_subpaths(gt_path, lambda p: p.suffix == ".bmp"), 256, 512), dtype=np.uint8)
            for i, bmp_path in enumerate(sorted(p for p in gt_path.iterdir() if p.suffix == ".bmp")):
                img = cv2.imread(str(bmp_path), cv2.IMREAD_UNCHANGED)
                cv2.resize(img, (512, 256), dst=video[i, ...], interpolation=cv2.INTER_NEAREST_EXACT)

            np.save(current_ped_pm_path / f"{gt_path.name[:-3]}.npy", video)
            relative_symlink(
                ped12_pm_path / f"P{current_ucsd_name[-1]}_{gt_path.name[:-3]}.npy",
                current_ped_pm_path / f"{gt_path.name[:-3]}.npy",
            )


def process_ucsd(data_root: Path, use_cuda: bool) -> None:
    ucsd_12 = data_root / "UCSDped12"
    rmtree(ucsd_12, ignore_errors=True)

    for ucsd_name, (pre_phase, out_phase) in product(UCSD_NAMES, (("Train", "training"), ("Test", "testing"))):
        pre_training_path = data_root / "pre" / ucsd_name / pre_phase
        out_training_path = data_root / ucsd_name / out_phase
        out_training_path12 = ucsd_12 / out_phase

        frames_path = out_training_path / "frames"
        no_bg_path = out_training_path / "nobackground_frames_resized"
        frames_path12 = out_training_path12 / "frames"
        no_bg_path12 = out_training_path12 / "nobackground_frames_resized"
        rmtree(frames_path.parent, ignore_errors=True)
        frames_path.mkdir(parents=True, exist_ok=True)
        frames_path12.mkdir(parents=True, exist_ok=True)
        no_bg_path12.mkdir(parents=True, exist_ok=True)

        for train_clip_path in pre_training_path.iterdir():
            if train_clip_path.name.endswith("_gt") or not train_clip_path.is_dir():
                continue
            relative_symlink(frames_path / train_clip_path.name, train_clip_path)
            relative_symlink(frames_path12 / f"P{ucsd_name[-1]}_{train_clip_path.name}", train_clip_path)

            no_bg_clip_path = no_bg_path / train_clip_path.name
            no_bg_clip_path.mkdir(parents=True, exist_ok=True)
            (no_bg_path12 / f"P{ucsd_name[-1]}_{train_clip_path.name}").symlink_to(
                Path("../" * (len(no_bg_clip_path.parents) - 1) / no_bg_clip_path)
            )

            video = load_video(train_clip_path / "%03d.tif")

            bg = process_background_gpu(video) if use_cuda else process_background_cpu(video)
            wo_bg = remove_background(video, bg, 10)
            save_video(wo_bg, no_bg_clip_path)

    process_ucsd_gt(data_root)
