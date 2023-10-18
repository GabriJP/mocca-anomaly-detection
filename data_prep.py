from collections import defaultdict
from itertools import product
from pathlib import Path
from shutil import rmtree

import click
import cv2
import numpy as np
import numpy.typing as npt

U8_NDTYPE = npt.NDArray[np.uint8]


@click.group()
def tui() -> None:
    pass


def _process_background_gpu(video: U8_NDTYPE) -> U8_NDTYPE:
    img_gpu = cv2.cuda_GpuMat(video.shape[1:], cv2.CV_8U)

    # noinspection PyUnresolvedReferences
    mog = cv2.cuda.createBackgroundSubtractorMOG2(history=video.shape[0], detectShadows=False)
    stream = cv2.cuda.Stream_Null()

    for frame in video:
        img_gpu.upload(frame)
        mog.apply(img_gpu, -1, stream)

    mog.getBackgroundImage(stream, img_gpu)
    return img_gpu.download()


def _process_background_cpu(video: U8_NDTYPE) -> U8_NDTYPE:
    # noinspection PyUnresolvedReferences
    mog = cv2.bgsegm.createBackgroundSubtractorCNT()
    for frame in video:
        mog.apply(frame)

    bg: U8_NDTYPE = mog.getBackgroundImage()
    if video.ndim == bg.ndim:  # Video is grayscale and bg is not
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    return bg


def remove_background(video: U8_NDTYPE, background: U8_NDTYPE, threshold: float) -> U8_NDTYPE:
    cv2.blur(background, (3, 3), dst=background)
    no_bg_video = np.empty_like(video)
    tmp_array = np.empty_like(video, shape=video.shape[1:])
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i, frame in enumerate(video):
        cv2.absdiff(frame, background, dst=tmp_array)
        cv2.blur(tmp_array, (3, 3), dst=tmp_array)
        mask: U8_NDTYPE = (tmp_array > threshold).astype(np.uint8)
        mask = cv2.erode(mask, dilate_kernel, iterations=2)
        cv2.medianBlur(mask, 5, dst=mask)
        mask = cv2.dilate(mask, dilate_kernel, iterations=5)

        np.multiply(frame, mask, out=no_bg_video[i, :, :])

    return no_bg_video


ucsd_names = ("UCSDped1", "UCSDped2")


def relative_symlink(from_path: Path, to_path: Path) -> None:
    from_path.parent.mkdir(parents=True, exist_ok=True)
    from_path.symlink_to(Path("../" * (len(from_path.parents) - 1) / to_path))


def _process_ucsd_gt(data_root: Path) -> None:
    ped12_path = data_root / "UCSDped12" / "testing" / "test_frame_mask"
    rmtree(ped12_path, ignore_errors=True)
    ped12_path.mkdir(parents=True)

    for current_ucsd_name in ucsd_names:
        current_ped_frame_mask_path = data_root / current_ucsd_name / "testing" / "test_frame_mask"
        rmtree(current_ped_frame_mask_path, ignore_errors=True)
        current_ped_frame_mask_path.mkdir(parents=True)

        dot_m_path = data_root / "pre" / current_ucsd_name / "Test" / f"{current_ucsd_name}.m"
        dot_m_lines = dot_m_path.read_text().splitlines()[1:]
        test_clips = sorted(p for p in dot_m_path.parent.iterdir() if p.is_dir() and not p.name.endswith("_gt"))
        clip_path: Path
        clip_gt_line: str
        for clip_path, clip_gt_line in zip(test_clips, dot_m_lines):
            n_frames = sum(1 for p in clip_path.iterdir() if p.suffix == ".tif")
            clip_labels = np.zeros(n_frames, dtype=np.uint8)
            ranges = clip_gt_line[clip_gt_line.rfind("[") + 1 : clip_gt_line.rfind("]")].split(",")
            for positive_range in ranges:
                positive_range = positive_range.strip()
                start, end = positive_range.split(":")
                clip_labels[int(start) : int(end)] = 1

            np_path = current_ped_frame_mask_path / f"{clip_path.name}.npy"
            np.save(np_path, clip_labels)
            relative_symlink(ped12_path / f"P{current_ucsd_name[-1]}_{clip_path.name}.npy", np_path)


def _process_ucsd(data_root: Path, use_cuda: bool) -> None:
    ucsd_12 = data_root / "UCSDped12"
    rmtree(ucsd_12, ignore_errors=True)

    for ucsd_name, (pre_phase, out_phase) in product(ucsd_names, (("Train", "training"), ("Test", "testing"))):
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

            img_paths = [p for p in train_clip_path.iterdir() if p.suffix == ".tif"]
            imgs: U8_NDTYPE = np.empty((len(img_paths), 256, 512), dtype=np.uint8)
            for i, img_path in enumerate(sorted(img_paths)):
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                cv2.resize(img, (512, 256), dst=imgs[i, :, :], interpolation=cv2.INTER_CUBIC)

            bg = _process_background_gpu(imgs) if use_cuda else _process_background_cpu(imgs)
            wo_bg = remove_background(imgs, bg, 10)

            for i, frame in enumerate(wo_bg):
                cv2.imwrite(str(no_bg_clip_path / f"{i:03d}.jpg"), frame, (cv2.IMWRITE_JPEG_QUALITY, 100))

    _process_ucsd_gt(data_root)


def _process_shang(data_root: Path, use_cuda: bool) -> None:
    pass


@tui.command()
@click.option("--data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
@click.option("--cuda", is_flag=True)
def process_ucsd(data_root: Path, cuda: bool) -> None:
    _process_ucsd(data_root, cuda)


@tui.command()
@click.option("--data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
@click.option("--cuda", is_flag=True)
def process_shanghai(data_root: Path, cuda: bool) -> None:
    _process_shang(data_root, cuda)


if __name__ == "__main__":
    # tui()
    _process_ucsd(Path("data2"), False)
