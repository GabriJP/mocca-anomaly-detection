from pathlib import Path
from shutil import rmtree

import click
import cv2
import numpy as np
import requests


def _process_background_gpu(video: np.ndarray):
    img_gpu = cv2.cuda_GpuMat(video[0].shape, cv2.CV_8U)

    mog = cv2.cuda.createBackgroundSubtractorMOG2()
    stream = cv2.cuda.Stream_Null()

    for frame in video:
        img_gpu.upload(frame)
        mog.apply(img_gpu, -1, stream)

    mog.getBackgroundImage(stream, img_gpu)
    return img_gpu.download()


def _process_background_cpu(video: np.ndarray):
    mog = cv2.createBackgroundSubtractorMOG2()
    for frame in video:
        mog.apply(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    bg = mog.getBackgroundImage()
    return cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)


def _process_ucsd(data_root: Path, ucsd_name: str, use_cuda: bool) -> None:
    pre_training_path = data_root / "pre" / ucsd_name / "Train"
    out_training_path = data_root / ucsd_name / "training"

    frames_path = out_training_path / "frames"
    no_bg_path = out_training_path / "nobackground_frames_resized"
    rmtree(frames_path.parent)
    frames_path.mkdir(parents=True, exist_ok=True)

    for train_clip_path in pre_training_path.iterdir():
        if train_clip_path.name.endswith("_gt") or not train_clip_path.is_dir():
            continue
        frames_path.with_name(train_clip_path.name).symlink_to(
            Path("../" * (len(train_clip_path.parents) - 1) / train_clip_path)
        )

        no_bg_clip_path = no_bg_path / train_clip_path.name
        no_bg_clip_path.mkdir(parents=True, exist_ok=True)

        imgs = np.empty((200, 256, 512), dtype=np.uint8)
        for i, img_path in enumerate(sorted(p for p in train_clip_path.iterdir() if not p.name.startswith("."))):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            imgs[i, :, :] = cv2.resize(img, (512, 256), interpolation=cv2.INTER_CUBIC)

        bg = _process_background_gpu(imgs) if use_cuda else _process_background_cpu(imgs)


@click.command()
@click.option("--data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
def process_ucsd(data_root: Path) -> None:
    _process_ucsd(data_root, "UCSDped1")
    _process_ucsd(data_root, "UCSDped2")


@click.command()
@click.option("--data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
def process_shanghai(data_root: Path) -> None:
    download_path = data_root / "raw"
    download_path.mkdir(parents=True, exist_ok=True)
    with requests.get("http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz", stream=True) as r, (
        download_path / "UCSD_Anomaly_Dataset.tar.gz"
    ).open("wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


if __name__ == "__main__":
    _process_ucsd(Path("data"), "UCSDped1", True)
