import fcntl
import logging
import pickle
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
from os import cpu_count
from pathlib import Path
from time import perf_counter
from time import sleep
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import click
import flwr as fl
import torch
import wandb
from flwr.common import Config
from flwr.common import NDArrays
from flwr.common import Scalar
from flwr.server import SimpleClientManager
from flwr.server.strategy import FedProx

from datasets import ContinuousDataManager
from datasets import ContinuousShanghaiTechDataHolder
from datasets import DataManager
from datasets import ShanghaiTechDataHolder
from datasets import VideoAnomalyDetectionResultHelper
from models import ShanghaiTech
from trainers import get_keys
from trainers import train
from utils import EarlyStopServer
from utils import get_out_dir2 as get_out_dir
from utils import load_model
from utils import RunConfig
from utils import set_seeds
from utils import wandb_logger

wanted_device = "cuda"


class MoccaClient(fl.client.NumPyClient):
    def __init__(self, net: ShanghaiTech, data_holder: ShanghaiTechDataHolder, rc: RunConfig) -> None:
        super().__init__()
        self.net = net.to(wanted_device)
        self.data_holder: Union[ShanghaiTechDataHolder, ContinuousShanghaiTechDataHolder] = data_holder
        self.rc = rc
        self.R: Dict[str, torch.Tensor] = dict()

    def get_parameters(self, config: Config) -> NDArrays:
        if not len(self.R):
            self.R = {k: torch.tensor(0.0, device=wanted_device) for k in get_keys(self.rc.idx_list_enc)}

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
            self.R[k] = torch.tensor(rv, device=wanted_device)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Config]:
        self.rc.epochs = int(config["epochs"])
        self.rc.warm_up_n_epochs = int(config["warm_up_n_epochs"])
        self.set_parameters(parameters)
        train_loader, _ = self.data_holder.get_loaders(
            batch_size=self.rc.batch_size, shuffle_train=True, pin_memory=True, num_workers=self.rc.n_workers
        )
        out_dir, tmp = get_out_dir(self.rc)
        net_checkpoint = train(
            self.net,
            train_loader,
            out_dir,
            wanted_device,
            None,
            self.rc,
            self.R,
            float(config.get("proximal_mu", 0.0)),
        )

        torch_dict = load_model(net_checkpoint)
        self.R = torch_dict["R"]
        return self.get_parameters(config=dict()), len(train_loader) * self.rc.epochs, dict()

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Config]:
        if self.data_holder.root.name.endswith("13"):
            return 0.0, 0, dict(oc_metric=0.0, recon_metric=0.0, anomaly_score=0.0)
        self.set_parameters(parameters)
        dataset = self.data_holder.get_test_data()
        helper = VideoAnomalyDetectionResultHelper(
            dataset=dataset,
            model=self.net,
            R=self.R,
            boundary=self.rc.boundary,
            device=wanted_device,
            end_to_end_training=True,
            debug=False,
            output_file=None,
            dist=self.rc.dist,
        )
        global_oc, global_metrics = helper.test_video_anomaly_detection()
        global_metrics_dict: Config = dict(zip(("oc_metric", "recon_metric", "anomaly_score"), global_metrics))
        wandb_logger.log_test(global_metrics_dict)
        return float(global_oc.mean()), len(global_oc), global_metrics_dict


class ParallelClient(MoccaClient):
    def __init__(self, net: ShanghaiTech, data_holder: ShanghaiTechDataHolder, rc: RunConfig) -> None:
        super().__init__(net, data_holder, rc)
        self.current_device = torch.device("cpu")
        self.to_cpu()
        self.is_locked = False

    def to_device(self, device: torch.device) -> None:
        self.net.to(device)
        self.R = {k: v.to(device) for k, v in self.R.items()}
        self.current_device = device

    def to_cpu(self) -> None:
        self.to_device(torch.device("cpu"))

    def to_target_device(self) -> None:
        self.to_device(torch.device(wanted_device))

    def set_parameters(self, parameters: NDArrays) -> None:
        with self.execution_exclusive_context():
            super().set_parameters(parameters)
            self.to_device(self.current_device)

    @contextmanager
    def execution_exclusive_context(self, *, to_target_device: bool = False) -> Iterator[None]:
        if self.is_locked:
            logging.info("Lock skipped")
            yield
            logging.info("Unlock skipped")
            return
        with open(__file__) as fd:
            start = float("inf")
            try:
                logging.info("Locking")
                fcntl.flock(fd, fcntl.LOCK_EX)
                start = perf_counter()
                self.is_locked = True
                logging.info("Locked")
                if to_target_device:
                    self.to_target_device()
                yield
            finally:
                self.to_cpu()
                torch.cuda.empty_cache()
                sleep(5)
                logging.info(f"Unlocking after {perf_counter() - start:02f} seconds")
                self.is_locked = False
                fcntl.flock(fd, fcntl.LOCK_UN)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Config]:
        with self.execution_exclusive_context(to_target_device=True):
            return super().fit(parameters, config)

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Config]:
        with self.execution_exclusive_context(to_target_device=True):
            return super().evaluate(parameters, config)


class ContinuousClient(ParallelClient):
    def __init__(self, net: ShanghaiTech, data_holder: ContinuousShanghaiTechDataHolder, rc: RunConfig) -> None:
        super().__init__(net, data_holder, rc)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Config]:
        if not isinstance(self.data_holder, ContinuousShanghaiTechDataHolder):
            raise ValueError
        if config.get("reset", False):
            self.data_holder.reset()
        return super().fit(parameters, config)


@click.group()
def cli() -> None:
    pass


@cli.command(context_settings=dict(show_default=True))
@click.argument("server_address", type=str, default="xavier:8080")
@click.option(
    "--n_workers",
    type=click.IntRange(0),
    default=cpu_count(),
    help="Number of workers for data loading. 0 means that the data will be loaded in the main process.",
)
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
@click.option("--compile_net", is_flag=True)
@click.option("--parallel", is_flag=True, help="Use Parallel client so only one execution is running at any given time")
@click.option("--continuous", is_flag=True, help="Use Continuous Data Manager")
def client(
    n_workers: int,
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
    compile_net: bool,
    parallel: bool,
    continuous: bool,
) -> None:
    idx_list_enc_ilist: Tuple[int, ...] = tuple(int(a) for a in idx_list_enc.split(","))
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
    )
    set_seeds(seed)
    # Init logger & print training/warm-up summary
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler("./training.log"), logging.StreamHandler()],
    )

    wandb.init(project="mocca", entity="gabijp", group=wandb_group, name=wandb_name, config=asdict(rc))
    data_holder_cl: Type[DataManager] = ContinuousDataManager if continuous else DataManager
    data_holder = data_holder_cl(
        dataset_name="ShanghaiTech", data_path=data_path, normal_class=-1, seed=seed, clip_length=clip_length
    ).get_data_holder()
    net: ShanghaiTech = ShanghaiTech(
        data_holder.shape, code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional
    )
    torch.set_float32_matmul_precision("high")
    net = torch.compile(net, dynamic=False, disable=not compile_net)  # type: ignore
    client_class: Type[MoccaClient]
    if continuous:
        client_class = ContinuousClient
    elif parallel:
        client_class = ParallelClient
    else:
        client_class = MoccaClient
    mc = client_class(net, data_holder, rc)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=mc,
        grpc_max_message_length=1024**3,  # 1 GiB
        root_certificates=Path("ca.crt").read_bytes(),
    )
    wandb_logger.save_model(dict(net_state_dict=net.state_dict(), R=mc.R), name="last_model")


def create_fit_config_fn(epochs: int, warm_up_n_epochs: int) -> Callable[[int], Config]:
    def inner(_: int) -> Config:
        return dict(epochs=epochs, warm_up_n_epochs=warm_up_n_epochs)

    return inner


def get_evaluate_fn(
    wandb_group: str, test_checkpoint: int, dist: str, compile_net: bool, data_path: Path
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]]:
    wandb.init(project="mocca", entity="gabijp", group=wandb_group, name="server")
    idx_list_enc = (3, 4, 5, 6)
    data_holder = DataManager(
        dataset_name="ShanghaiTech",
        data_path=data_path,
        normal_class=-1,
        seed=-1,
        clip_length=16,
    ).get_data_holder()
    net = ShanghaiTech(
        data_holder.shape,
        code_length=512,
        load_lstm=True,
        hidden_size=100,
        num_layers=1,
        dropout=0.3,
        bidirectional=True,
    )
    if compile_net:
        torch.set_float32_matmul_precision("high")
        net = torch.compile(net)  # type: ignore
    R = {k: torch.tensor(0.0, device=wanted_device) for k in get_keys(idx_list_enc)}

    def centralized_evaluation(
        server_round: int, parameters: NDArrays, _: Dict[str, Scalar]
    ) -> Tuple[float, Dict[str, Scalar]]:
        if server_round % test_checkpoint != 0:
            wandb_logger.manual_step()
            return 0.0, dict()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        rs_list = parameters[len(state_dict) :]

        keys = list(R.keys()) if len(R) else get_keys(idx_list_enc)

        if len(keys) != len(rs_list):
            raise ValueError("Keys, cs and rs differ in quantity")

        for k, rv in zip(keys, rs_list):
            R[k] = torch.tensor(rv, device=wanted_device)

        dataset = data_holder.get_test_data()
        helper = VideoAnomalyDetectionResultHelper(
            dataset=dataset,
            model=net,
            R=R,
            boundary="soft",
            device=wanted_device,
            end_to_end_training=True,
            debug=False,
            output_file=None,
            dist=dist,
        )
        global_oc, global_metrics = helper.test_video_anomaly_detection()
        global_metrics_dict: Config = dict(zip(("oc_metric", "recon_metric", "anomaly_score"), global_metrics))
        wandb_logger.log_test(global_metrics_dict)
        return float(global_oc.mean()), global_metrics_dict

    return centralized_evaluation


@cli.command(context_settings=dict(show_default=True))
@click.argument("data_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--port", type=click.IntRange(0, 65_535), default=8080)
@click.option("--num_rounds", type=click.IntRange(1), default=5)
@click.option("--epochs", type=click.IntRange(1), default=5)
@click.option("--warm_up_n_epochs", type=click.IntRange(0), default=0)
@click.option("--proximal_mu", type=click.FloatRange(0, 1), default=1.0)
@click.option("--patience", type=click.IntRange(0), default=None)
@click.option("--min_delta_pct", type=click.FloatRange(0, 1), default=None)
@click.option("--min_fit_clients", type=click.IntRange(2), default=2)
@click.option("--min_evaluate_clients", type=click.IntRange(0), default=2)
@click.option("--min_available_clients", type=click.IntRange(2), default=2)
@click.option("--dist", type=click.Choice(["l1", "l2"]), default="l2")
@click.option("--wandb_group", type=str, default=None)
@click.option("--test_checkpoint", type=click.IntRange(1), default=1)
@click.option("--compile_net", is_flag=True)
def server(
    data_path: Path,
    port: int,
    num_rounds: int,
    epochs: int,
    warm_up_n_epochs: int,
    proximal_mu: float,
    patience: Optional[int],
    min_delta_pct: Optional[float],
    min_fit_clients: int,
    min_evaluate_clients: int,
    min_available_clients: int,
    dist: str,
    wandb_group: Optional[str],
    test_checkpoint: int,
    compile_net: bool,
) -> None:
    strategy = FedProx(
        fraction_fit=0.0,
        fraction_evaluate=0.0 if min_evaluate_clients == 0 else 1e-5,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=wandb_group and get_evaluate_fn(wandb_group, test_checkpoint, dist, compile_net, data_path),
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
    hist = fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        server=fl_server,
        config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=172_800.0),  # Timeout==2 days
        strategy=strategy,
        grpc_max_message_length=1024**3,  # 1 GiB
        certificates=(
            (certificates_path / "ca.crt").read_bytes(),
            (certificates_path / "server.pem").read_bytes(),
            (certificates_path / "server.key").read_bytes(),
        ),
    )
    with Path(f"server_hist_{int(time.time())}.pckl").open("wb") as fd:
        pickle.dump(hist, fd)


if __name__ == "__main__":
    cli()
