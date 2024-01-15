import logging
import time
from contextlib import ExitStack
from dataclasses import asdict
from os import cpu_count
from pathlib import Path
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import click
import torch
import wandb

from datasets.data_manager import DataManager
from main_shanghaitech2 import MoccaClient
from models.shanghaitech_model import ShanghaiTech
from utils import EarlyStoppingDM
from utils import RunConfig
from utils import set_seeds
from utils import wandb_logger


@click.command("cli", context_settings=dict(show_default=True))
@click.option("-s", "--seed", type=int, default=-1, help="Random seed")
@click.option(
    "--n_workers",
    type=click.IntRange(0),
    default=cpu_count(),
    help="Number of workers for data loading. 0 means that the data will be loaded in the main process.",
)
@click.option("--output_path", type=click.Path(file_okay=False, path_type=Path), default="./output")
@click.option("-dl", "--disable-logging", is_flag=True, help="Disable logging")
# Model config
@click.option("-zl", "--code-length", default=2048, type=int, help="Code length")
# Optimizer config
@click.option("-lr", "--learning-rate", type=float, default=1.0e-4, help="Learning rate")
@click.option("-wd", "--weight-decay", type=float, default=0.5e-6, help="Learning rate")
# Data
@click.option(
    "-dp",
    "--data-path",
    type=click.Path(file_okay=False, path_type=Path),
    default="./ShanghaiTech",
    help="Dataset main path",
)
@click.option("-cl", "--clip-length", type=int, default=16, help="Clip length")
# Training config
# LSTMs
@click.option("-ll", "--load-lstm", is_flag=True, help="Load LSTMs")
@click.option("-bdl", "--bidirectional", is_flag=True, help="Bidirectional LSTMs")
@click.option("-hs", "--hidden-size", type=int, default=100, help="Hidden size")
@click.option("-nl", "--num-layers", type=int, default=1, help="Number of LSTMs layers")
@click.option("-drp", "--dropout", type=float, default=0.0, help="Dropout probability")
# Autoencoder
@click.option("-bs", "--batch-size", type=int, default=4, help="Batch size")
@click.option("-bd", "--boundary", type=click.Choice(("hard", "soft")), default="soft", help="Boundary")
@click.option("-ile", "--idx-list-enc", type=str, default="6", help="List of indices of model encoder")
@click.option("-e", "--epochs", type=int, default=1, help="Training epochs")
@click.option("-nu", "--nu", type=float, default=0.1)
@click.option("--fp16", is_flag=True)
@click.option("--dist", type=click.Choice(["l1", "l2"]), default="l2")
@click.option("--wandb_group", type=str, default=None)
@click.option("--wandb_prefix", type=str, default=None)
@click.option("--compile_net", is_flag=True)
@click.option("--es_initial_patience_epochs", type=click.IntRange(0), default=1, help="Early stopping initial patience")
@click.option("--rolling_factor", type=click.IntRange(2), default=20, help="Early stopping rolling window size")
@click.option("--es_patience", type=click.IntRange(1), default=100, help="Early stopping patience")
@click.option("--view", is_flag=True)
@click.option("--test-chk", type=str, default="30", help="Comma-separated of epochs. Checkpoints for test")
def main(
    seed: int,
    n_workers: int,
    output_path: Path,
    disable_logging: bool,
    # Model config,
    code_length: int,
    # Optimizer config,
    learning_rate: float,
    weight_decay: float,
    # Data,
    data_path: Path,
    clip_length: int,
    # Training config,
    # LSTMs,
    load_lstm: bool,
    bidirectional: bool,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    # Autoencoder,
    batch_size: int,
    boundary: str,
    idx_list_enc: str,
    epochs: int,
    nu: float,
    fp16: bool,
    dist: str,
    wandb_group: Optional[str],
    wandb_prefix: Optional[str],
    compile_net: bool,
    es_initial_patience_epochs: int,
    rolling_factor: int,
    es_patience: int,
    view: bool,
    test_chk: str,
) -> None:
    idx_list_enc_ilist: Tuple[int, ...] = tuple(int(a) for a in idx_list_enc.split(","))
    test_chk_split = test_chk.split(",")
    test_chk_set: Set[int] = {int(a) for a in test_chk_split if not a.startswith("%")}
    test_chk_remainder: List[int] = list(int(a[1:]) for a in test_chk_split if a.startswith("%"))
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

    rc = RunConfig(
        n_workers,
        output_path,
        code_length,
        learning_rate,
        weight_decay,
        data_path,
        clip_length,
        load_lstm,
        bidirectional,
        hidden_size,
        num_layers,
        dropout,
        batch_size,
        boundary,
        idx_list_enc_ilist,
        nu,
        fp16,
        compile_net,
        dist,
    )

    data_holders = {
        path.name[5:]: DataManager(
            dataset_name="ShanghaiTech", data_path=path, normal_class=-1, seed=seed, clip_length=clip_length
        ).get_data_holder()
        for path in data_path.iterdir()
    }
    with ExitStack() as stack:
        runs = {
            i: wandb.init(
                project="mocca", entity="gabijp", group=wandb_group, name=f"{wandb_prefix}_{i}", config=asdict(rc)
            )
            for i in data_holders
        }
        for run in runs.values():
            if not isinstance(run, wandb.sdk.wandb_run.Run):
                raise ValueError
            stack.push(run)  # type: ignore

        one_data_holder = next(iter(data_holders.values()))

        net = ShanghaiTech(
            one_data_holder.shape, code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional
        )

        if compile_net:
            torch.set_float32_matmul_precision("high")
            net = torch.compile(net)  # type: ignore
        wandb.watch(net)
        rc.epochs = 1
        rc.warm_up_n_epochs = 0

        train_loader, _ = one_data_holder.get_loaders(
            batch_size=rc.batch_size, shuffle_train=True, pin_memory=True, num_workers=rc.n_workers
        )

        es = EarlyStoppingDM(
            initial_patience=len(train_loader) * es_initial_patience_epochs,
            rolling_factor=rolling_factor,
            es_patience=es_patience,
        )

        mc = MoccaClient(net, one_data_holder, rc, es, view=view, view_data=("noname", data_path.name))

        initial_time = time.perf_counter()

        if len(test_chk_remainder):
            test_chk_set.union(set(range(1, epochs, test_chk_remainder[0])))
        i = 0
        for i in range(epochs):
            mc.fit()
            for j, run in runs.items():
                if isinstance(run, wandb.sdk.wandb_run.Run):
                    wandb.run = run
                mc.data_holder = data_holders[j]
                if es.early_stop:
                    mc.evaluate()
                    break
                if i in test_chk_set:
                    mc.evaluate()

        for j, run in runs.items():
            if isinstance(run, wandb.sdk.wandb_run.Run):
                wandb.run = run
            mc.data_holder = data_holders[j]
            mc.evaluate()

        logging.getLogger().info(
            f"Fitted in {i + 1} epochs requiring {time.perf_counter() - initial_time:.02f} seconds"
        )
        wandb_logger.save_model(dict(net_state_dict=net.state_dict(), R=mc.R))


if __name__ == "__main__":
    main()
