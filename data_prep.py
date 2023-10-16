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
    img_gpu = cv2.cuda_GpuMat(video[0].shape, cv2.CV_8U)

    # noinspection PyUnresolvedReferences
    mog = cv2.cuda.createBackgroundSubtractorMOG2(history=200, detectShadows=False)
    stream = cv2.cuda.Stream_Null()

    for frame in video:
        img_gpu.upload(frame)
        mog.apply(img_gpu, -1, stream)

    mog.getBackgroundImage(stream, img_gpu)
    return img_gpu.download()


def _process_background_cpu(video: U8_NDTYPE) -> U8_NDTYPE:
    mog = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=False)
    for frame in video:
        mog.apply(frame)

    bg = mog.getBackgroundImage()
    return bg


def remove_background(video: U8_NDTYPE, background: U8_NDTYPE, threshold: float) -> U8_NDTYPE:
    diff_array = np.empty_like(video)
    tmp_array = np.empty_like(video, shape=video.shape[1:])
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i, frame in enumerate(video):
        cv2.absdiff(frame, background, tmp_array)
        mask: U8_NDTYPE = (tmp_array > threshold).astype(np.uint8)
        mask = cv2.dilate(mask, dilate_kernel, iterations=5)

        diff_array[i, :, :] = frame * mask

    return diff_array


def _process_ucsd(data_root: Path, ucsd_name: str, use_cuda: bool) -> None:
    for pre_phase, out_phase in (("Train", "training"), ("Test", "testing")):
        pre_training_path = data_root / "pre" / ucsd_name / pre_phase
        out_training_path = data_root / ucsd_name / out_phase

        frames_path = out_training_path / "frames"
        no_bg_path = out_training_path / "nobackground_frames_resized"
        rmtree(frames_path.parent, ignore_errors=True)
        frames_path.mkdir(parents=True, exist_ok=True)

        for train_clip_path in pre_training_path.iterdir():
            if train_clip_path.name.endswith("_gt") or not train_clip_path.is_dir():
                continue
            (frames_path / train_clip_path.name).symlink_to(
                Path("../" * (len(train_clip_path.parents) - 1) / train_clip_path)
            )

            no_bg_clip_path = no_bg_path / train_clip_path.name
            no_bg_clip_path.mkdir(parents=True, exist_ok=True)

            img_paths = [p for p in train_clip_path.iterdir() if p.suffix == ".tif"]
            imgs: U8_NDTYPE = np.empty((len(img_paths), 256, 512), dtype=np.uint8)
            for i, img_path in enumerate(sorted(img_paths)):
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                imgs[i, :, :] = cv2.resize(img, (512, 256), interpolation=cv2.INTER_CUBIC)

            bg = _process_background_gpu(imgs) if use_cuda else _process_background_cpu(imgs)
            wo_bg = remove_background(imgs, bg, 50)

            for i, frame in enumerate(wo_bg):
                cv2.imwrite(str(no_bg_clip_path / f"{i:03d}.jpg"), frame, (cv2.IMWRITE_JPEG_QUALITY, 100))


@tui.command()
@click.option("--data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
@click.option("--cuda", is_flag=True)
def process_ucsd(data_root: Path, cuda: bool) -> None:
    _process_ucsd(data_root, "UCSDped1", cuda)
    _process_ucsd(data_root, "UCSDped2", cuda)


@tui.command()
@click.option("--data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
@click.option("--cuda", is_flag=True)
def process_shanghai(data_root: Path, cuda: bool) -> None:
    raise NotImplementedError


@tui.command()
@click.option("--data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
def show_no_bg(data_root: Path) -> None:
    cv2.namedWindow("Frames", cv2.WINDOW_GUI_EXPANDED)
    for npy_path in data_root.glob("**/*.npy"):
        wo_bg = np.load(npy_path)
        for frame in wo_bg:
            cv2.imshow("Frames", frame)
            k = cv2.waitKey(100)
            if k == ord("q"):
                break
            if k == ord("e"):
                return


if __name__ == "__main__":
    # tui()
    _process_ucsd(Path("data"), "UCSDped1", True)
    _process_ucsd(Path("data"), "UCSDped2", True)
