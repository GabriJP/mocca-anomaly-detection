from pathlib import Path
from shutil import rmtree

from .utils import copy_path_exclude_prefix
from .utils import copy_path_include_prefix
from .utils import load_video
from .utils import process_background_cpu
from .utils import process_background_gpu
from .utils import relative_symlink
from .utils import remove_background
from .utils import save_video


def separated_shang(shang_path: Path) -> None:
    separated_path = shang_path / "separated"
    rmtree(separated_path, ignore_errors=True)
    nfs = shang_path / "complete" / "training" / "nobackground_frames_resized"
    for path in nfs.iterdir():
        output_path = separated_path / f"shang{path.name[:2]}" / "training" / "nobackground_frames_resized" / path.name
        relative_symlink(output_path, path)

    for current_shang in range(1, 14):
        copy_path_include_prefix(
            shang_path / "complete" / "testing",
            separated_path / f"shang{current_shang:02d}" / "testing",
            f"{current_shang:02d}_",
        )


def one_out_shang(shang_path: Path) -> None:
    one_out_path = shang_path / "one_out"
    rmtree(one_out_path, ignore_errors=True)

    # Training
    nfs = shang_path / "complete" / "training" / "nobackground_frames_resized"
    all_shangs = {f"{i:02d}" for i in range(1, 14)}
    for path in nfs.iterdir():
        for current_shang in all_shangs - {path.name[:2]}:
            output_path = (
                one_out_path / f"shang{current_shang}" / "training" / "nobackground_frames_resized" / path.name
            )
            relative_symlink(output_path, path)

    # Testing
    for current_shang in all_shangs:
        copy_path_exclude_prefix(
            shang_path / "complete" / "testing",
            one_out_path / f"shang{current_shang}" / "testing",
            exclude_prefix=current_shang,
        )


def avo_shang(shang_path: Path) -> None:
    separated_path = shang_path / "avo"
    rmtree(separated_path, ignore_errors=True)
    nfs = shang_path / "complete" / "training" / "nobackground_frames_resized"
    for current_shang in range(1, 14):
        output_path = separated_path / f"shang{current_shang:02d}" / "training" / "nobackground_frames_resized"
        relative_symlink(output_path, nfs)

        copy_path_include_prefix(
            shang_path / "complete" / "testing",
            separated_path / f"shang{current_shang:02d}" / "testing",
            f"{current_shang:02d}_",
        )


def continuous_shang(shang_path: Path, *, partitions: int = 2) -> None:
    continuous_path = shang_path / f"continuous_{partitions}"
    rmtree(continuous_path, ignore_errors=True)
    separated = shang_path / "separated"

    separated_shangs = sorted(p for p in separated.iterdir())

    for current_partition in range(partitions):
        for current_sepshang in separated_shangs[current_partition::partitions]:
            current_contshang_path = continuous_path / str(current_partition) / current_sepshang.name
            relative_symlink(current_contshang_path / "training", current_sepshang / "training")
            relative_symlink(current_contshang_path / "testing", shang_path / "complete" / "testing")


def generate_all_subsets(shang_path: Path) -> None:
    separated_shang(shang_path)
    one_out_shang(shang_path)
    avo_shang(shang_path)
    continuous_shang(shang_path)
    continuous_shang(shang_path, partitions=2)
    continuous_shang(shang_path, partitions=3)
    continuous_shang(shang_path, partitions=4)


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
    generate_all_subsets(data_root)
