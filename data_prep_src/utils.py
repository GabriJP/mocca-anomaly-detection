from collections.abc import Callable
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

U8_NDTYPE = npt.NDArray[np.uint8]


def show_video(video: U8_NDTYPE, delay: int = 300) -> None:
    cv2.namedWindow("Window", cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_AUTOSIZE)
    # cv2.setWindowProperty("Window", cv2.WND_PROP_, cv2.WINDOW_FULLSCREEN)
    for frame in video:
        cv2.imshow("Window", frame)
        k = cv2.waitKey(delay)
        if k == ord("x"):
            exit(0)
        elif k == ord("q"):
            return
    cv2.destroyAllWindows()


def process_background_cpu(video: U8_NDTYPE) -> U8_NDTYPE:
    # noinspection PyUnresolvedReferences
    mog = cv2.bgsegm.createBackgroundSubtractorCNT()

    frame: U8_NDTYPE = np.empty(video.shape[1:-1], dtype=np.uint8)
    for bgr_frame in video:
        cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY, dst=frame)
        mog.apply(frame)

    return mog.getBackgroundImage()


def process_background_gpu(video: U8_NDTYPE) -> U8_NDTYPE:
    img_gpu = cv2.cuda_GpuMat(video.shape[1:], cv2.CV_8U)

    # noinspection PyUnresolvedReferences
    mog = cv2.cuda.createBackgroundSubtractorMOG2(history=video.shape[0], detectShadows=False)
    stream = cv2.cuda.Stream_Null()

    for frame in video:
        img_gpu.upload(frame)
        mog.apply(img_gpu, -1, stream)

    mog.getBackgroundImage(stream, img_gpu)
    return img_gpu.download()


def load_video(video_path: Path, interpolation: int = cv2.INTER_CUBIC) -> U8_NDTYPE:
    cap = cv2.VideoCapture(str(video_path))
    try:
        h, w, n = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        video: U8_NDTYPE = np.empty((n, 256, 512, 3), dtype=np.uint8)
        frame: U8_NDTYPE = np.empty((h, w, 3), dtype=np.uint8)
        i = 0
        for i in range(len(video)):
            print("Reading", video_path, i)
            ret, _ = cap.read(frame)
            if not ret:
                i -= 1
                break
            cv2.resize(frame, (512, 256), dst=video[i, ...], interpolation=interpolation)
        return video[: i + 1]
    finally:
        cap.release()


def save_video(video: U8_NDTYPE, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(video):
        cv2.imwrite(str(save_path / f"{i:03d}.jpg"), frame, (cv2.IMWRITE_JPEG_QUALITY, 100))


def remove_background(video: U8_NDTYPE, background: U8_NDTYPE, threshold: float) -> U8_NDTYPE:
    cv2.blur(background, (3, 3), dst=background)
    no_bg_video = np.empty_like(video)
    tmp_shape = video.shape[1:] if len(video.shape) == 3 else video.shape[1:-1]
    tmp_array = np.empty_like(video, shape=tmp_shape)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i, bgr_frame in enumerate(video):
        frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        cv2.absdiff(frame, background, dst=tmp_array)
        cv2.blur(tmp_array, (3, 3), dst=tmp_array)
        mask: U8_NDTYPE = (tmp_array > threshold).astype(np.uint8)
        mask = cv2.erode(mask, dilate_kernel, iterations=2)
        cv2.medianBlur(mask, 5, dst=mask)
        mask = cv2.dilate(mask, dilate_kernel, iterations=5)

        if len(bgr_frame.shape) == 3:
            mask = np.stack((mask,) * 3, axis=-1)

        np.multiply(bgr_frame, mask, out=no_bg_video[i, ...])

    return no_bg_video


def n_subpaths(path: Path, count_filter: Optional[Callable[[Path], bool]] = (lambda _: True)) -> int:
    return sum(1 for _ in filter(count_filter, path.iterdir()))


def relative_symlink(from_path: Path, to_path: Path) -> None:
    from_path.parent.mkdir(parents=True, exist_ok=True)
    from_path.symlink_to(Path("../" * (len(from_path.parents) - 1) / to_path))


def copy_path_include_prefix(source: Path, dst_path: Path, include_prefix: str) -> None:
    for p in source.iterdir():
        if p.name.startswith(include_prefix):
            relative_symlink(dst_path / p.name, p)
            continue

        if p.is_dir():
            copy_path_include_prefix(p, dst_path / p.name, include_prefix)


def copy_path_exclude_prefix(source: Path, dst_path: Path, exclude_prefix: str) -> None:
    for p in source.iterdir():
        if p.name.startswith(exclude_prefix):
            continue

        if p.is_dir():
            if all(f.is_file() and "_" not in f.name for f in p.iterdir()):
                relative_symlink(dst_path / p.name, p)
            else:
                copy_path_exclude_prefix(p, dst_path / p.name, exclude_prefix)
        elif p.is_file():
            relative_symlink(dst_path / p.name, p)
        else:
            raise ValueError
