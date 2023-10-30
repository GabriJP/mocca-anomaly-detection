from itertools import product
from pathlib import Path
from shutil import rmtree
from typing import Callable
from typing import Optional

import click
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


def _process_background_cpu(video: U8_NDTYPE) -> U8_NDTYPE:
    # noinspection PyUnresolvedReferences
    mog = cv2.bgsegm.createBackgroundSubtractorCNT()

    frame: U8_NDTYPE = np.empty(video.shape[1:-1], dtype=np.uint8)
    for bgr_frame in video:
        cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY, dst=frame)
        mog.apply(frame)

    return mog.getBackgroundImage()


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


ucsd_names = ("UCSDped1", "UCSDped2")


def _process_ucsd_gt(data_root: Path) -> None:
    ped12_fm_path = data_root / "UCSDped12" / "testing" / "test_frame_mask"
    ped12_pm_path = data_root / "UCSDped12" / "testing" / "test_pixel_mask"
    rmtree(ped12_fm_path, ignore_errors=True)
    rmtree(ped12_pm_path, ignore_errors=True)
    ped12_fm_path.mkdir(parents=True)
    ped12_pm_path.mkdir(parents=True)

    for current_ucsd_name in ucsd_names:
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
            video = load_video(gt_path / "%03d.bmp", interpolation=cv2.INTER_NEAREST_EXACT)

            np.save(current_ped_pm_path / f"{gt_path.name[:-3]}.npy", video)
            relative_symlink(
                ped12_pm_path / f"P{current_ucsd_name[-1]}_{gt_path.name[:-3]}.npy",
                current_ped_pm_path / f"{gt_path.name[:-3]}.npy",
            )


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

            video = load_video(train_clip_path / "%03d.tif")

            bg = _process_background_gpu(video) if use_cuda else _process_background_cpu(video)
            wo_bg = remove_background(video, bg, 10)
            save_video(wo_bg, no_bg_clip_path)

    _process_ucsd_gt(data_root)


def _process_shang_train(data_root: Path, use_cuda: bool) -> None:
    videos_path = data_root / "pre" / "shanghaitech" / "training" / "videos"
    training_path = data_root / "shanghaitech" / "complete" / "training"
    rmtree(training_path, ignore_errors=True)
    training_path.mkdir(parents=True)
    relative_symlink(training_path / "videos", videos_path)
    nobg_videos_path = training_path / "nobackground_frames_resized"
    nobg_videos_path.mkdir(parents=True)
    for video_path in videos_path.iterdir():
        video = load_video(video_path)
        bg = _process_background_gpu(video) if use_cuda else _process_background_cpu(video)
        wo_bg = remove_background(video, bg, 10)
        save_video(wo_bg, nobg_videos_path / video_path.stem)


def _process_shang_test(data_root: Path, use_cuda: bool) -> None:
    frames_path = data_root / "pre" / "shanghaitech" / "testing" / "frames"
    testing_path = data_root / "shanghaitech" / "complete" / "testing"
    rmtree(testing_path, ignore_errors=True)
    testing_path.mkdir(parents=True)
    relative_symlink(testing_path / "videos", frames_path)
    nobg_videos_path = testing_path / "nobackground_frames_resized"
    nobg_videos_path.mkdir(parents=True)
    for video_path in frames_path.iterdir():
        video = load_video(video_path / "%03d.jpg")
        bg = _process_background_gpu(video) if use_cuda else _process_background_cpu(video)
        wo_bg = remove_background(video, bg, 10)
        save_video(wo_bg, nobg_videos_path / video_path.stem)

    relative_symlink(testing_path / "test_frame_mask", frames_path.parent / "test_frame_mask")
    relative_symlink(testing_path / "test_pixel_mask", frames_path.parent / "test_pixel_mask")


def _process_shang(data_root: Path, use_cuda: bool) -> None:
    _process_shang_train(data_root, use_cuda)
    _process_shang_test(data_root, use_cuda)


@click.group()
def tui() -> None:
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


@tui.command()
@click.option("--data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
@click.option("--cuda", is_flag=True)
@click.pass_context
def process_all(ctx: click.Context, data_root: Path, cuda: bool) -> None:
    ctx.invoke(process_ucsd, data_root=data_root, cuda=cuda)
    ctx.invoke(process_shanghai, data_root=data_root, cuda=cuda)


if __name__ == "__main__":
    # tui()
    _process_shang_test(Path("data"), False)
