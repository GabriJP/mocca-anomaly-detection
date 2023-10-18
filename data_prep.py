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


def _process_ucsd(data_root: Path, use_cuda: bool) -> None:
    ucsd_12 = data_root / "UCSDped12"
    rmtree(ucsd_12, ignore_errors=True)
    for ucsd_name, (pre_phase, out_phase) in product(
        ("UCSDped1", "UCSDped2"), (("Train", "training"), ("Test", "testing"))
    ):
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
            (frames_path / train_clip_path.name).symlink_to(
                Path("../" * (len(train_clip_path.parents) - 1) / train_clip_path)
            )
            (frames_path12 / f"P{ucsd_name[-1]}_{train_clip_path.name}").symlink_to(
                Path("../" * (len(train_clip_path.parents) - 1) / train_clip_path)
            )

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
    _process_ucsd(Path("data"), False)
