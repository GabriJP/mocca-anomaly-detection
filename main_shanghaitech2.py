#!/usr/bin/env python3
import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from os import cpu_count
from pathlib import Path
from typing import ParamSpec

import click
import torch
import torch.nn as nn
import wandb

from datasets import ShanghaiTechDataHolder
from datasets import VideoAnomalyDetectionResultHelper
from datasets.data_manager import DataManager
from models.shanghaitech_base_model import DownsampleBlock
from models.shanghaitech_base_model import TemporallySharedFullyConnection
from models.shanghaitech_base_model import UpsampleBlock
from models.shanghaitech_model import ShanghaiTech
from trainers import train
from utils import EarlyStoppingDM
from utils import get_out_dir2 as get_out_dir
from utils import load_model
from utils import RunConfig
from utils import set_seeds
from utils import wandb_logger

device = "cuda"


P = ParamSpec("P")


def _initialize_module(
    module: nn.Module,
    func: Callable[[torch.Tensor], torch.Tensor] | Callable[[torch.Tensor, P.args, P.kwargs], torch.Tensor],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    if not isinstance(module, (nn.Conv3d, TemporallySharedFullyConnection, nn.LSTM, DownsampleBlock, UpsampleBlock)):
        return

    if module.weight is not None:
        func(module.weight, *args, **kwargs)
    else:
        logging.warning(f"Not initializing weights for module {module} of class name {module.__class__.__name__}")

    if isinstance(module.bias, torch.Tensor):
        func(module.bias, *args, **kwargs)


def init_weights_xavier_uniform(m: nn.Module) -> None:
    _initialize_module(m, nn.init.xavier_uniform_)


initializers: dict[str, Callable[[nn.Module], None]] = dict(
    zeros=partial(_initialize_module, func=nn.init.zeros_),
    ones=partial(_initialize_module, func=nn.init.ones_),
    xavier_uniform=init_weights_xavier_uniform,
    none=lambda _: None,
)


class MoccaClient:
    def __init__(
        self,
        net: ShanghaiTech,
        data_holder: ShanghaiTechDataHolder,
        rc: RunConfig,
        es: EarlyStoppingDM | None = None,
        view: bool = False,
        view_data: tuple[str, str] = ("weights_name", "dataset_name"),
    ) -> None:
        super().__init__()
        self.net = net.to(device)
        self.data_holder = data_holder
        self.rc = rc
        self.es = es
        self.view = view
        self.view_data = view_data
        self.R: dict[str, torch.Tensor] = dict()
        self.epoch = -self.rc.epochs

    def fit(self) -> None:
        self.epoch += self.rc.epochs
        train_loader, _ = self.data_holder.get_loaders(
            batch_size=self.rc.batch_size, shuffle_train=True, pin_memory=True, num_workers=self.rc.n_workers
        )
        out_dir, _ = get_out_dir(self.rc)
        net_checkpoint = train(
            self.net,
            train_loader,
            out_dir,
            device,
            None,
            self.rc,
            self.R,
            0.0,
            self.es,
            self.epoch,
        )

        torch_dict = load_model(net_checkpoint)
        self.R = torch_dict["R"]

    def evaluate(self) -> None:
        helper = VideoAnomalyDetectionResultHelper(
            dataset=self.data_holder.get_test_data(),
            model=self.net,
            R=self.R,
            boundary=self.rc.boundary,
            device=device,
            end_to_end_training=True,
            debug=self.rc.debug,
            output_file=None,
            dist=self.rc.dist,
        )
        _, global_metrics = helper.test_video_anomaly_detection(view=self.view, view_data=self.view_data)
        wandb_logger.log_test(dict(zip(("oc_metric", "recon_metric", "anomaly_score"), global_metrics)), self.epoch)


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
@click.option("--initialization", type=click.Choice(list(initializers)), default="none")
@click.option("--wandb_group", type=str, default=None)
@click.option("--wandb_name", type=str, default=None)
@click.option("--compile_net", is_flag=True)
@click.option("--es_initial_patience_epochs", type=click.IntRange(0), default=1, help="Early stopping initial patience")
@click.option("--rolling_factor", type=click.IntRange(2), default=20, help="Early stopping rolling window size")
@click.option("--es_patience", type=click.IntRange(1), default=100, help="Early stopping patience")
@click.option("--view", is_flag=True)
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
    fp16: bool,
    dist: str,
    initialization: str,
    wandb_group: str | None,
    wandb_name: str | None,
    compile_net: bool,
    es_initial_patience_epochs: int,
    rolling_factor: int,
    es_patience: int,
    view: bool,
    test_chk: str,
    debug: bool,
) -> None:
    idx_list_enc_ilist: tuple[int, ...] = tuple(int(a) for a in idx_list_enc.split(","))
    test_chk_split = test_chk.split(",")
    test_chk_set: set[int] = {int(a) for a in test_chk_split if not a.startswith("%")}
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
        debug=debug,
    )

    wandb.init(project="mocca", entity="gabijp", group=wandb_group, name=wandb_name, config=asdict(rc))
    wandb_logger.add_epoch_metrics(("test.oc_metric", "test.recon_metric", "test.anomaly_score"))
    data_holder = DataManager(
        dataset_name="ShanghaiTech", data_path=data_path, normal_class=-1, seed=seed, clip_length=clip_length
    ).get_data_holder()
    net: ShanghaiTech = ShanghaiTech(
        data_holder.shape, code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional
    )
    net.apply(initializers[initialization])
    torch.set_float32_matmul_precision("high")
    net = torch.compile(net, dynamic=False, disable=not compile_net)  # type: ignore
    wandb.watch(net)
    rc.epochs = 1
    rc.warm_up_n_epochs = 0

    train_loader, _ = data_holder.get_loaders(
        batch_size=rc.batch_size, shuffle_train=True, pin_memory=True, num_workers=rc.n_workers
    )

    es = EarlyStoppingDM(
        initial_patience=len(train_loader) * es_initial_patience_epochs,
        rolling_factor=rolling_factor,
        es_patience=es_patience,
    )

    mc = MoccaClient(net, data_holder, rc, es, view=view, view_data=(wandb_name or "noname", data_path.name))

    initial_time = time.perf_counter()

    i = 0
    for i in range(epochs):
        mc.fit()
        if es.early_stop:
            break

        if i in test_chk_set:
            mc.evaluate()

        if rc.debug and i:
            break

    mc.evaluate()

    logging.getLogger().info(f"Fitted in {i + 1} epochs requiring {time.perf_counter() - initial_time:.02f} seconds")
    # wandb_logger.save_model(dict(net_state_dict=net.state_dict(), R=mc.R))


if __name__ == "__main__":
    main()
