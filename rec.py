import logging
import sys
from multiprocessing.pool import Pool
from os import cpu_count
from pathlib import Path
from typing import Tuple

import click
import cv2
import numpy as np
import numpy.typing as npt
import torch

from datasets.data_manager import DataManager
from datasets.shanghaitech_test import VideoAnomalyDetectionResultHelper
from models.shanghaitech_model import ShanghaiTech
from models.shanghaitech_model import ShanghaiTechEncoder
from utils import extract_arguments_from_checkpoint
from utils import set_seeds


@click.group()
def cli() -> None:
    pass


@cli.command("test_network", context_settings=dict(show_default=True))
@click.option("-s", "--seed", type=int, default=-1, help="Random seed")
@click.option("--disable_cuda", is_flag=True, help="Do not use cuda even if available")
@click.option("-dl", "--disable-logging", is_flag=True, help="Disable logging")
@click.option("-db", "--debug", is_flag=True, help="Debug mode")
# Model config
@click.option(
    "-ck",
    "--model-ckp",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Model checkpoint",
)
# Data
@click.option(
    "-dp",
    "--data-path",
    type=click.Path(file_okay=False, path_type=Path),
    default="./ShanghaiTech",
    help="Dataset main path",
)
@click.option("--view", is_flag=True, help="Save output to desktop")
@click.option("--compile_net", is_flag=True)
def test_network(
    seed: int,
    disable_cuda: bool,
    disable_logging: bool,
    debug: bool,
    model_ckp: Path,
    data_path: Path,
    view: bool,
    compile_net: bool,
) -> None:
    # Set seed
    set_seeds(seed)

    if disable_logging:
        logging.disable(level=logging.INFO)

    # Init logger & print training/warm-up summary
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )

    device = "cuda" if not disable_cuda and torch.cuda.is_available() else "cpu"

    # Init DataHolder class
    data_holder = DataManager(
        dataset_name="ShanghaiTech", data_path=data_path, normal_class=-1, seed=seed, only_test=True
    ).get_data_holder()

    (
        code_length,
        batch_size,
        boundary,
        use_selectors,
        idx_list_enc_ilist,
        load_lstm,
        hidden_size,
        num_layers,
        dropout,
        bidirectional,
        dataset_name,
        train_type,
    ) = extract_arguments_from_checkpoint(model_ckp)

    # Init dataset
    dataset = data_holder.get_test_data()
    model_cls = ShanghaiTech if train_type == "train_end_to_end" else ShanghaiTechEncoder
    net: torch.nn.Module = model_cls(
        data_holder.shape, code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional, use_selectors
    )
    if disable_cuda:
        st_dict = torch.load(model_ckp, map_location="cpu")
    else:
        st_dict = torch.load(model_ckp)

    if compile_net:
        torch.set_float32_matmul_precision("high")
        net = torch.compile(net, dynamic=False)  # type: ignore
    load_state_dict_warn = net.load_state_dict(st_dict["net_state_dict"], strict=False)
    logging.warning(f"Missing keys when loading state_dict: {load_state_dict_warn.missing_keys}")
    logging.warning(f"Unexpected keys when loading state_dict: {load_state_dict_warn.unexpected_keys}")
    logging.info(f"Loaded model from: {model_ckp}")

    # Initialize test helper for processing each video seperately
    # It prints the result to the loaded checkpoint directory
    helper = VideoAnomalyDetectionResultHelper(
        dataset=dataset,
        model=net,
        R=st_dict["R"],
        boundary=boundary,
        device=device,
        end_to_end_training=train_type == "train_end_to_end",
        debug=debug,
        output_file=model_ckp.parent / "shanghaitech_test_results.txt",
    )
    # TEST
    helper.test_video_anomaly_detection(view=view, view_data=(model_ckp.stem, dataset_name))
    logging.info("Test finished")


def _label_path(data_path: Path) -> None:
    files = sorted(p for p in data_path.iterdir() if p.suffix == ".png")
    y_trues = np.load(str(data_path / "sample_y.npy"))
    y_preds = np.load(str(data_path / "sample_as.npy"))
    for file, y_true, y_pred in zip(files[2:], y_trues[2:], y_preds[2:]):
        img = cv2.imread(str(file))
        img[:, 512 : 512 + 5, :] = 0
        img[:128, 512 : 512 + 5, 2 if y_true else 1] = 255
        img[128:, 512 : 512 + 5, 2 if y_pred > 1 else 1] = 255
        cv2.imwrite(str(file), img)


@cli.command("label_path")
@click.argument("data_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def label_path(data_path: Path) -> None:
    _label_path(data_path)


@cli.command("label_paths")
@click.argument("data_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def label_paths(data_path: Path) -> None:
    with Pool() as pool:
        pool.map_async(_label_path, data_path.iterdir())
        pool.close()
        pool.join()


def _color_for(value: float) -> Tuple[int, int, int]:
    color: npt.NDArray[np.uint8] = np.array((0.0, value * 255, (1 - value) * 255)).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def _plot_labels(data_path: Path) -> None:
    colors = (0, 0, 255), (0, 255, 0)
    _label_path(data_path)
    y_trues = np.load(str(data_path / "sample_y.npy"))
    y_preds = (np.load(str(data_path / "sample_as.npy")) > 1).astype(np.uint8)
    y_rc = np.load(str(data_path / "sample_rc.npy"))
    y_oc = np.load(str(data_path / "sample_oc.npy"))

    col_len, col_sep = 10, 5
    row_len, row_sep = 10, 50

    height, width = 4 * (row_len + row_sep) - row_sep, len(y_trues) * (col_len + col_sep) - col_sep
    img: npt.NDArray[np.uint8] = np.zeros((height, width, 3), dtype=np.uint8)

    for i, (i1, i2, i3, i4) in enumerate(zip(y_trues, y_preds, y_rc, y_oc)):
        x1 = i * (col_len + col_sep)
        x2 = x1 + col_len
        y1 = row_len + row_sep
        cv2.rectangle(
            img,
            (x1, 0),
            (x2, row_len),
            color=colors[i1],
            thickness=cv2.FILLED,
        )
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y1 + row_len),
            color=colors[i2],
            thickness=cv2.FILLED,
        )
        cv2.rectangle(
            img,
            (x1, y1 * 2),
            (x2, y1 * 2 + row_len),
            color=_color_for(i3),
            thickness=cv2.FILLED,
        )
        cv2.rectangle(
            img,
            (x1, y1 * 3),
            (x2, y1 * 3 + row_len),
            color=_color_for(i4),
            thickness=cv2.FILLED,
        )
    cv2.imwrite(str(data_path / "labels.png"), img)


@cli.command("plot_labels")
@click.argument("data_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def plot_labels(data_path: Path) -> None:
    with Pool() as pool:
        pool.map_async(_plot_labels, data_path.iterdir(), error_callback=lambda x: print(x, file=sys.stderr))
        pool.close()
        pool.join()


if __name__ == "__main__":
    cli()
