import logging
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import click
import flwr as fl
import torch
from flwr.common import Config
from flwr.common import NDArrays
from flwr.server import SimpleClientManager
from flwr.server.strategy import FedProx

import wandb
from datasets import DataManager
from datasets import ShanghaiTechDataHolder
from datasets import VideoAnomalyDetectionResultHelper
from models import ShanghaiTech
from trainers import get_keys
from trainers import train
from utils import EarlyStopServer
from utils import RunConfig
from utils import set_seeds
from utils import wandb_logger

device = "cuda"


def get_out_dir(rc: RunConfig) -> Tuple[Path, str]:
    tmp_name = (
        f"train-mn_ShanghaiTech-cl_{rc.code_length}-bs_{rc.batch_size}-nu_{rc.nu}-lr_{rc.learning_rate}-"
        f"bd_{rc.boundary}-sl_False-ile_{'.'.join(map(str, rc.idx_list_enc))}-lstm_{rc.load_lstm}-bidir_False-"
        f"hs_{rc.hidden_size}-nl_{rc.num_layers}-dp_{rc.dropout}"
    )
    out_dir = (rc.output_path / "ShanghaiTech" / "train_end_to_end" / tmp_name).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir, tmp_name


class MoccaClient(fl.client.NumPyClient):
    def __init__(self, net: ShanghaiTech, data_holder: ShanghaiTechDataHolder, rc: RunConfig) -> None:
        super().__init__()
        self.net = net.to(device)
        self.data_holder = data_holder
        self.rc = rc
        self.R: Dict[str, torch.Tensor] = dict()

    def get_parameters(self, config: Config) -> NDArrays:
        if not len(self.R):
            self.R = {k: torch.tensor(0.0, device=device) for k in get_keys(self.rc.idx_list_enc)}

        return [val.cpu().numpy() for val in self.net.state_dict().values()] + [
            val.cpu().numpy() for val in self.R.values()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

        rs_list = parameters[len(state_dict) :]

        keys = list(self.R.keys()) if len(self.R) else get_keys(self.rc.idx_list_enc)

        if len(keys) != len(rs_list):
            raise ValueError("Keys, cs and rs differ in quantity")

        for k, rv in zip(keys, rs_list):
            self.R[k] = torch.tensor(rv, device=device)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Config]:
        self.rc.epochs = int(config["epochs"])
        self.rc.warm_up_n_epochs = int(config["warm_up_n_epochs"])
        self.set_parameters(parameters)
        train_loader, _ = self.data_holder.get_loaders(
            batch_size=self.rc.batch_size, shuffle_train=True, pin_memory=True
        )
        out_dir, tmp = get_out_dir(self.rc)
        net_checkpoint = train(
            self.net,
            train_loader,
            out_dir,
            device,
            None,
            self.rc,
            self.R,
            float(config.get("proximal_mu", 0.0)),
        )

        torch_dict = torch.load(net_checkpoint)
        self.R = torch_dict["R"]
        return self.get_parameters(config=dict()), len(train_loader) * self.rc.epochs, dict()

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Config]:
        self.set_parameters(parameters)
        dataset = self.data_holder.get_test_data()
        helper = VideoAnomalyDetectionResultHelper(
            dataset=dataset,
            model=self.net,
            R=self.R,
            boundary=self.rc.boundary,
            device=device,
            end_to_end_training=True,
            debug=False,
            output_file=None,
        )
        global_oc, global_metrics = helper.test_video_anomaly_detection()
        global_metrics_dict: Config = dict(zip(("oc_metric", "recon_metric", "anomaly_score"), global_metrics))
        wandb_logger.log_test(global_metrics_dict)
        return float(global_oc.mean()), len(global_oc), global_metrics_dict


@click.group()
def cli() -> None:
    pass


@cli.command(context_settings=dict(show_default=True))
@click.argument("server_address", type=str, default="150.214.203.248:8080")
@click.option("--output_path", type=click.Path(file_okay=False, path_type=Path), default=Path("./output"))
@click.option("--code-length", type=click.IntRange(1), default=1024, help="Code length")
@click.option("--learning-rate", type=click.FloatRange(0, 1), default=1.0e-4, help="Learning rate")
@click.option("--weight-decay", type=click.FloatRange(0, 1), default=0.5e-6, help="Learning rate")
@click.option(
    "--data-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/UCSDped2"),
    help="Dataset main path",
)
@click.option("--clip-length", type=click.IntRange(1), default=16, help="Clip length")
@click.option("--load-lstm", is_flag=True, help="Load LSTMs")
@click.option("--bidirectional", is_flag=True, help="Bidirectional LSTMs")
@click.option("--hidden-size", type=click.IntRange(1), default=100, help="Hidden size")
@click.option("--num-layers", type=click.IntRange(1), default=1, help="Number of LSTMs layers")
@click.option("--dropout", type=click.FloatRange(0, 1), default=0.0, help="Dropout probability")
@click.option("--batch-size", type=click.IntRange(1), default=4, help="Batch size")
@click.option("--boundary", type=click.Choice(["soft", "hard"]), default="soft", help="Boundary")
@click.option("--idx-list-enc", type=str, default="6", help="List of indices of model encoder")
@click.option("--nu", type=click.FloatRange(0, 1), default=0.1)
@click.option("--wandb_group", type=str, default=None)
@click.option("--wandb_name", type=str, default=None)
@click.option("--seed", type=int, default=-1)
def client(
    server_address: str,
    output_path: Path,
    code_length: int,
    learning_rate: float,
    weight_decay: float,
    data_path: Path,
    clip_length: int,
    load_lstm: bool,
    bidirectional: bool,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    boundary: str,
    idx_list_enc: str,
    nu: float,
    wandb_group: Optional[str],
    wandb_name: Optional[str],
    seed: int,
) -> None:
    idx_list_enc_ilist: Tuple[int, ...] = tuple(int(a) for a in idx_list_enc.split(","))
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
    set_seeds(seed)
    # Init logger & print training/warm-up summary
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler("./training.log"), logging.StreamHandler()],
    )

    wandb.init(project="mocca", entity="gabijp", group=wandb_group, name=wandb_name, config=asdict(rc))
    data_holder = DataManager(
        dataset_name="ShanghaiTech", data_path=data_path, normal_class=-1, clip_length=clip_length
    ).get_data_holder()
    net = ShanghaiTech(data_holder.shape, code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional)
    mc = MoccaClient(net, data_holder, rc)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=mc,
        grpc_max_message_length=1024**3,  # 1 GB
        root_certificates=Path("ca.crt").read_bytes(),
    )
    wandb_logger.save_model(dict(net_state_dict=net.state_dict(), R=mc.R), name="last_model")


def create_fit_config_fn(epochs: int, warm_up_n_epochs: int) -> Callable[[int], Config]:
    def inner(_: int) -> Config:
        return dict(epochs=epochs, warm_up_n_epochs=warm_up_n_epochs)

    return inner


@cli.command(context_settings=dict(show_default=True))
@click.option("--num_rounds", type=click.IntRange(1), default=5)
@click.option("--epochs", type=click.IntRange(1), default=5)
@click.option("--warm_up_n_epochs", type=click.IntRange(0), default=0)
@click.option("--proximal_mu", type=click.FloatRange(0, 1), default=1.0)
@click.option("--patience", type=click.IntRange(0), default=None)
@click.option("--min_delta_pct", type=click.FloatRange(0, 1), default=None)
@click.option("--min_fit_clients", type=click.IntRange(2), default=2)
@click.option("--min_evaluate_clients", type=click.IntRange(2), default=2)
@click.option("--min_available_clients", type=click.IntRange(2), default=2)
def server(
    num_rounds: int,
    epochs: int,
    warm_up_n_epochs: int,
    proximal_mu: float,
    patience: Optional[int],
    min_delta_pct: Optional[float],
    min_fit_clients: int,
    min_evaluate_clients: int,
    min_available_clients: int,
) -> None:
    strategy = FedProx(
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=create_fit_config_fn(epochs, warm_up_n_epochs),
        proximal_mu=proximal_mu,
    )
    fl_server = (
        None
        if patience is None or min_delta_pct is None
        else EarlyStopServer(
            client_manager=SimpleClientManager(), strategy=strategy, patience=patience, min_delta_pct=min_delta_pct
        )
    )
    certificates_path = Path.home() / "certs"
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        server=fl_server,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        grpc_max_message_length=1024**3,
        certificates=(
            (certificates_path / "ca.crt").read_bytes(),
            (certificates_path / "server.pem").read_bytes(),
            (certificates_path / "server.key").read_bytes(),
        ),
    )


if __name__ == "__main__":
    cli()
