import logging
from os import cpu_count
from pathlib import Path

import click
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
@click.option(
    "--n_workers",
    type=click.IntRange(0),
    default=cpu_count(),
    help="Number of workers for data loading. 0 means that the data will be loaded in the main process.",
)
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
def test_network(
    seed: int,
    n_workers: int,
    disable_cuda: bool,
    disable_logging: bool,
    debug: bool,
    model_ckp: Path,
    data_path: Path,
    view: bool,
) -> None:
    # Set seed
    set_seeds(seed)

    if disable_logging:
        logging.disable(level=logging.INFO)

    # Init logger & print training/warm-up summary
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler("./training.log"), logging.StreamHandler()],
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

    # Print data infos
    logging.info("Dataset info:")
    logging.info(f"\n\n\t\t\t\tBatch size    : {batch_size}")

    # Init dataset
    dataset = data_holder.get_test_data()
    model_cls = ShanghaiTech if train_type == "train_end_to_end" else ShanghaiTechEncoder
    net = model_cls(
        data_holder.shape, code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional, use_selectors
    )
    if disable_cuda:
        st_dict = torch.load(model_ckp, map_location="cpu")
    else:
        st_dict = torch.load(model_ckp)

    net.load_state_dict(st_dict["net_state_dict"], strict=False)
    logging.info(f"Loaded model from: {model_ckp}")
    logging.info(
        f"Start test with params:"
        f"\n\t\t\t\tDataset        : {dataset_name}"
        f"\n\t\t\t\tCode length    : {code_length}"
        f"\n\t\t\t\tEnc layer list : {idx_list_enc_ilist}"
        f"\n\t\t\t\tBoundary       : {boundary}"
        f"\n\t\t\t\tUse Selectors  : {use_selectors}"
        f"\n\t\t\t\tBatch size     : {batch_size}"
        f"\n\t\t\t\tN workers      : {n_workers}"
        f"\n\t\t\t\tLoad LSTMs     : {load_lstm}"
        f"\n\t\t\t\tHidden size    : {hidden_size}"
        f"\n\t\t\t\tNum layers     : {num_layers}"
        f"\n\t\t\t\tBidirectional  : {bidirectional}"
        f"\n\t\t\t\tDropout prob   : {dropout}"
    )

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
    helper.test_video_anomaly_detection(view=view)
    logging.info("Test finished")


if __name__ == "__main__":
    cli()
