#!/usr/bin/env python3
import logging
import time
from dataclasses import asdict
from os import cpu_count
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Set
from typing import Tuple

import click
import torch
import wandb

from datasets.data_manager import DataManager
from main_shanghaitech2 import MoccaClient
from models.shanghaitech_model import ShanghaiTech
from utils import get_out_dir2 as get_out_dir
from utils import load_model
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
@click.option("--wandb_group", type=str, default=None)
@click.option("--compile_net", is_flag=True)
@click.option("--test-chk", type=str, default="30", help="Comma-separated of epochs. Checkpoints for test")
@click.option("--debug", is_flag=True)
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
    wandb_group: Optional[str],
    compile_net: bool,
    test_chk: str,
    debug: bool,
) -> None:
    idx_list_enc_ilist: Tuple[int, ...] = tuple(int(a) for a in idx_list_enc.split(","))
    test_chk_split = test_chk.split(",")
    test_chk_set: Set[int] = set(map(int, test_chk_split))
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
        fp16=False,
        compile=compile_net,
        dist="l2",
        debug=debug,
    )

    data_holders = {
        path.name[5:]: DataManager(
            dataset_name="ShanghaiTech", data_path=path, normal_class=-1, seed=seed, clip_length=rc.clip_length
        ).get_data_holder()
        for path in sorted(rc.data_path.iterdir())
    }
    wandb.init(project="mocca", entity="gabijp", group=wandb_group, name="train_run", config=asdict(rc))

    one_data_holder = next(iter(data_holders.values()))

    net: ShanghaiTech = ShanghaiTech(
        one_data_holder.shape, code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional
    )

    torch.set_float32_matmul_precision("high")
    net = torch.compile(net, dynamic=False, disable=not compile_net)  # type: ignore
    wandb.watch(net)
    rc.epochs = 1
    rc.warm_up_n_epochs = 0

    train_loader, _ = one_data_holder.get_loaders(
        batch_size=rc.batch_size, shuffle_train=True, pin_memory=True, num_workers=rc.n_workers
    )

    mc = MoccaClient(net, one_data_holder, rc)

    initial_time = time.perf_counter()

    i = 0
    for i in range(epochs):
        mc.fit()
        if rc.debug and i:
            break

    test_chk_set.add(i)
    out_dir, _ = get_out_dir(rc)
    checkpoints: Dict[int, Path] = {int(path.name.split("_")[2]): path for path in out_dir.iterdir()}
    for name, data_holder in data_holders.items():
        wandb.init(
            project="mocca",
            entity="gabijp",
            group=wandb_group,
            name=f"shang{name}",
            config=asdict(rc),
            reinit=True,
            job_type="test",
        )
        for j in sorted(test_chk_set):
            if not checkpoints[j].exists():
                continue
            model = load_model(checkpoints[j])
            mc.net.load_state_dict(model["net_state_dict"])
            mc.R = model["R"]
            if not isinstance(model["config"], RunConfig):
                raise ValueError
            mc.rc = model["config"]
            mc.epoch = j
            mc.data_holder = data_holders[name]
            mc.evaluate()

    logging.getLogger().info(f"Fitted in {i + 1} epochs requiring {time.perf_counter() - initial_time:.02f} seconds")
    wandb_logger.save_model(dict(net_state_dict=net.state_dict(), R=mc.R))


if __name__ == "__main__":
    main()
