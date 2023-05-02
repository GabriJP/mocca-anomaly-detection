import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple

import click
import torch

import wandb
from client import get_out_dir
from datasets import ShanghaiTechDataHolder
from datasets import VideoAnomalyDetectionResultHelper
from datasets.data_manager import DataManager
from models.shanghaitech_model import ShanghaiTech
from trainers import train
from utils import EarlyStoppingDM
from utils import RunConfig
from utils import set_seeds

device = "cuda"


class MoccaClient:
    def __init__(
        self,
        net: ShanghaiTech,
        data_holder: ShanghaiTechDataHolder,
        rc: RunConfig,
        es: Optional[EarlyStoppingDM] = None,
    ) -> None:
        super().__init__()
        self.net = net.to(device)
        self.data_holder = data_holder
        self.rc = rc
        self.es = es
        self.R: Dict[str, torch.Tensor] = dict()

    def fit(self) -> None:
        train_loader, _ = self.data_holder.get_loaders(
            batch_size=self.rc.batch_size, shuffle_train=True, pin_memory=True
        )
        out_dir, tmp = get_out_dir(self.rc)
        net_checkpoint = train(self.net, train_loader, out_dir, device, None, self.rc, self.R, 0.0, self.es)

        torch_dict = torch.load(net_checkpoint)
        self.R = torch_dict["R"]

    def evaluate(self) -> None:
        helper = VideoAnomalyDetectionResultHelper(
            dataset=self.data_holder.get_test_data(),
            model=self.net,
            R=self.R,
            boundary=self.rc.boundary,
            device=device,
            end_to_end_training=True,
            debug=False,
            output_file=None,
        )
        _, global_metrics = helper.test_video_anomaly_detection()
        wandb.log(dict(test=dict(zip(("oc_metric", "recon_metric", "anomaly_score"), global_metrics))))


@click.command("cli", context_settings=dict(show_default=True))
@click.option("-s", "--seed", type=int, default=-1, help="Random seed")
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
@click.option("--wandb_name", type=str, default=None)
@click.option("--compile_net", is_flag=True)
def main(
    seed: int,
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
    wandb_name: Optional[str],
    compile_net: bool,
) -> None:
    idx_list_enc_ilist: Tuple[int, ...] = tuple(int(a) for a in idx_list_enc.split(","))
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
        output_path,
        code_length,
        learning_rate,
        weight_decay,
        data_path,
        clip_length,
        load_lstm,
        hidden_size,
        num_layers,
        dropout,
        batch_size,
        boundary,
        idx_list_enc_ilist,
        nu,
    )

    wandb.init(project="mocca", entity="gabijp", group=wandb_group, name=wandb_name, config=asdict(rc))

    data_holder = DataManager(
        dataset_name="ShanghaiTech", data_path=data_path, normal_class=-1, clip_length=clip_length
    ).get_data_holder()
    net = ShanghaiTech(data_holder.shape, code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional)
    wandb.watch(net)
    if compile_net:
        torch.compile(net)
    rc.epochs = 1
    rc.warm_up_n_epochs = 0

    es = EarlyStoppingDM(initial_patience=100, rolling_factor=5)

    mc = MoccaClient(net, data_holder, rc, es)

    for i in range(epochs):
        mc.fit()
        mc.evaluate()
        if es.es:
            break


if __name__ == "__main__":
    main()
