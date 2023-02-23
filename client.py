from collections import OrderedDict
from dataclasses import asdict
from itertools import count
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Tuple

import click
import flwr as fl
import torch
import wandb
from flwr.common import Config
from flwr.common import NDArrays

from datasets import DataManager
from datasets import ShanghaiTechDataHolder
from datasets import VideoAnomalyDetectionResultHelper
from models import ShanghaiTech
from trainers import init_center_c
from trainers import train
from utils import RunConfig

device = "cuda"


def get_out_dir(rc: RunConfig) -> Tuple[Path, str]:
    tmp_name = (
        f"train-mn_ShanghaiTech-cl_{rc.code_length}-bs_{rc.batch_size}-nu_{rc.nu}-lr_{rc.learning_rate}-"
        f"bd_{rc.boundary}-sl_False-ile_{'.'.join(map(str, rc.idx_list_enc))}-lstm_{rc.load_lstm}-bidir_False-"
        f"hs_{rc.hidden_size}-nl_{rc.num_layers}-dp_{rc.dropout}"
    )
    out_dir = rc.output_path / "ShanghaiTech" / "train_end_to_end" / tmp_name

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir, tmp_name


def get_log_func() -> Callable[[Dict[str, float]], None]:
    counter = count()

    def log_func(data: Dict[str, float]) -> None:
        wandb.log(data, step=next(counter))

    return log_func


class MoccaClient(fl.client.NumPyClient):
    def __init__(self, net: ShanghaiTech, data_holder: ShanghaiTechDataHolder, rc: RunConfig) -> None:
        super().__init__()
        self.net = net.to(device)
        self.data_holder = data_holder
        self.rc = rc
        self.c: Dict[str, torch.Tensor] = dict()
        self.R: Dict[str, torch.Tensor] = dict()
        self.log_func = get_log_func()

    def get_parameters(self, config: Config) -> NDArrays:
        if not len(self.c) or not len(self.R):
            train_loader, _ = self.data_holder.get_loaders(
                batch_size=self.rc.batch_size, shuffle_train=True, pin_memory=True
            )
            self.c, keys = init_center_c(train_loader, self.net, self.rc.idx_list_enc, device, True, False)
            self.R = {k: torch.tensor(0.0, device=device) for k in keys}

        return (
            [val.cpu().numpy() for _, val in self.net.state_dict().items()]
            + [val.cpu().numpy() for _, val in self.c.items()]
            + [val.cpu().numpy() for _, val in self.R.items()]
        )

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

        c_r = parameters[len(state_dict) :]
        if len(c_r) % 2:
            raise ValueError("Not an even number of remaining tensors")

        if not len(self.c) is None:
            train_loader, _ = self.data_holder.get_loaders(
                batch_size=self.rc.batch_size, shuffle_train=True, pin_memory=True
            )
            _, keys = init_center_c(train_loader, self.net, self.rc.idx_list_enc, device, True, False)
        else:
            keys = list(self.c.keys())

        cs_list = c_r[: len(c_r) // 2]
        rs_list = c_r[len(c_r) // 2 :]

        if len(keys) != len(cs_list) != len(rs_list):
            raise ValueError("Keys, cs and rs differ in quantity")

        for k, cv, rv in zip(keys, cs_list, rs_list):
            self.c[k] = torch.tensor(cv, device=device)
            self.R[k] = torch.tensor(rv, device=device)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Config]:
        self.rc.epochs = int(config["epochs"])
        self.rc.batch_size = int(config["batch_size"])
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
            self.c,
            self.R,
            float(config.get("proximal_mu", 0.0)),
            log_func=self.log_func,
        )

        torch_dict = torch.load(net_checkpoint)
        self.R = torch_dict["R"]
        self.c = torch_dict["c"]
        return self.get_parameters(config=dict()), len(train_loader) * self.rc.epochs, dict()

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Config]:
        self.set_parameters(parameters)
        dataset = self.data_holder.get_test_data()
        helper = VideoAnomalyDetectionResultHelper(
            dataset=dataset,
            model=self.net,
            c=self.c,
            R=self.R,
            boundary=self.rc.boundary,
            device=device,
            end_to_end_training=True,
            debug=False,
            output_file=None,
        )
        global_oc, global_metrics = helper.test_video_anomaly_detection()
        return (
            float(global_oc.mean()),
            len(global_oc),
            dict(zip(("oc_metric", "recon_metric", "anomaly_score"), global_metrics)),
        )


@click.command(context_settings=dict(show_default=True))
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
@click.option("--hidden-size", type=click.IntRange(1), default=100, help="Hidden size")
@click.option("--num-layers", type=click.IntRange(1), default=1, help="Number of LSTMs layers")
@click.option("--dropout", type=click.FloatRange(0, 1), default=0.0, help="Dropout probability")
@click.option("--batch-size", type=click.IntRange(1), default=4, help="Batch size")
@click.option("--boundary", type=click.Choice(["soft", "hard"]), default="soft", help="Boundary")
@click.option(
    "--idx_list_enc", type=click.IntRange(1), multiple=True, default=[6], help="List of indices of model encoder"
)
@click.option("--nu", type=click.FloatRange(0, 1), default=0.1)
@click.option("--wandb_group", type=str, default=None)
def cli(
    server_address: str,
    output_path: Path,
    code_length: int,
    learning_rate: float,
    weight_decay: float,
    data_path: Path,
    clip_length: int,
    load_lstm: bool,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    boundary: str,
    idx_list_enc: Tuple[int],
    nu: float,
    wandb_group: str,
) -> None:
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
        idx_list_enc,
        nu,
    )
    wandb.init(project="mocca", entity="gabijp", group=wandb_group, config=asdict(rc))
    data_holder = DataManager(
        dataset_name="ShanghaiTech", data_path=data_path, normal_class=-1, clip_length=clip_length
    ).get_data_holder()
    net = ShanghaiTech(data_holder.shape, code_length, load_lstm, hidden_size, num_layers, dropout)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=MoccaClient(net, data_holder, rc),
        grpc_max_message_length=1024**3,  # 1 GB
        root_certificates=Path("ca.crt").read_bytes(),
    )


if __name__ == "__main__":
    cli()
