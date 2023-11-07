from pathlib import Path
from shutil import rmtree

import numpy as np
import numpy.typing as npt

from .utils import load_video
from .utils import process_background_cpu
from .utils import process_background_gpu
from .utils import relative_symlink
from .utils import remove_background
from .utils import save_video

U8_NDTYPE = npt.NDArray[np.uint8]


def process_shang_train(data_root: Path, use_cuda: bool) -> None:
    videos_path = data_root / "pre" / "shanghaitech" / "training" / "videos"
    training_path = data_root / "shanghaitech" / "complete" / "training"
    rmtree(training_path, ignore_errors=True)
    training_path.mkdir(parents=True)
    relative_symlink(training_path / "videos", videos_path)
    nobg_videos_path = training_path / "nobackground_frames_resized"
    nobg_videos_path.mkdir(parents=True)
    for video_path in videos_path.iterdir():
        video = load_video(video_path)
        bg = process_background_gpu(video) if use_cuda else process_background_cpu(video)
        wo_bg = remove_background(video, bg, 10)
        save_video(wo_bg, nobg_videos_path / video_path.stem)


def process_shang_test(data_root: Path, use_cuda: bool) -> None:
    frames_path = data_root / "pre" / "shanghaitech" / "testing" / "frames"
    testing_path = data_root / "shanghaitech" / "complete" / "testing"
    rmtree(testing_path, ignore_errors=True)
    testing_path.mkdir(parents=True)
    relative_symlink(testing_path / "videos", frames_path)
    nobg_videos_path = testing_path / "nobackground_frames_resized"
    nobg_videos_path.mkdir(parents=True)
    for video_path in frames_path.iterdir():
        video = load_video(video_path / "%03d.jpg")
        bg = process_background_gpu(video) if use_cuda else process_background_cpu(video)
        wo_bg = remove_background(video, bg, 10)
        save_video(wo_bg, nobg_videos_path / video_path.stem)

    relative_symlink(testing_path / "test_frame_mask", frames_path.parent / "test_frame_mask")
    relative_symlink(testing_path / "test_pixel_mask", frames_path.parent / "test_pixel_mask")


def process_shang(data_root: Path, use_cuda: bool) -> None:
    process_shang_train(data_root, use_cuda)
    process_shang_test(data_root, use_cuda)
