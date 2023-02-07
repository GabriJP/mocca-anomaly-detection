from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import click
import flwr as fl
import torch
from flwr.common import Config
from flwr.common import NDArrays
from tensorboardX import SummaryWriter

from datasets.data_manager import DataManager
from datasets.shanghaitech import ShanghaiTech_DataHolder
from models.shanghaitech_model import ShanghaiTech
from trainers.trainer_shanghaitech import train

device = "cuda"


@dataclass
class RunConfig:
    output_path: Path
    code_length: int
    learning_rate: float
    weight_decay: float
    data_path: Path
    clip_length: int
    load_lstm: bool
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    boundary: str
    idx_list_enc: tuple[int, ...]
    nu: float
    optimizer: str = "adam"
    lr_milestones: tuple[int, ...] = tuple()
    end_to_end_training: bool = True
    debug: bool = False
    warm_up_n_epochs: int = 0
    epochs: int = 0
    log_frequency: int = 5


def get_out_dir(rc: RunConfig) -> tuple[Path, str]:
    tmp_name = (
        f"train-mn_ShanghaiTech-cl_{rc.code_length}-bs_{rc.batch_size}-nu_{rc.nu}-lr_{rc.learning_rate}-"
        f"bd_{rc.boundary}-sl_False-ile_{'.'.join(map(str, rc.idx_list_enc))}-lstm_{rc.load_lstm}-bidir_False-"
        f"hs_{rc.hidden_size}-nl_{rc.num_layers}-dp_{rc.dropout}"
    )
    out_dir = rc.output_path / "ShanghaiTech" / "train_end_to_end" / tmp_name

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir, tmp_name


class MoccaClient(fl.client.NumPyClient):
    def __init__(self, net: ShanghaiTech, data_holder: ShanghaiTech_DataHolder, rc: RunConfig) -> None:
        super().__init__()
        self.net = net
        self.data_holder = data_holder
        self.rc = rc

    def get_parameters(self, config: Config) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, Config]:
        self.rc.epochs = int(config["epochs"])
        self.set_parameters(parameters)
        train_loader, _ = self.data_holder.get_loaders(
            batch_size=int(config["batch_size"]), shuffle_train=True, pin_memory=True
        )
        out_dir, tmp = get_out_dir(self.rc)
        with SummaryWriter(str(self.rc.output_path / "ShanghaiTech" / "tb_runs_train_end_to_end" / tmp)) as tb_writer:
            train(self.net, train_loader, str(out_dir), tb_writer, device, None, self.rc)

        return self.get_parameters(config={}), 0, {}

    def evaluate(self, parameters: NDArrays, config: Config) -> tuple[float, int, Config]:
        raise NotImplementedError
        # self.set_parameters(parameters)
        # loss, accuracy = test(self.net, testloader)
        # return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


@click.command()
@click.argument("server_address", type=str, default="150.214.203.248:8080")
@click.option("--output_path", type=click.Path(file_okay=False, path_type=Path), default=Path("./output"))
@click.option("--code-length", default=1024, type=int, help="Code length")
@click.option("--learning-rate", type=float, default=1.0e-4, help="Learning rate")
@click.option("--weight-decay", type=float, default=0.5e-6, help="Learning rate")
@click.option(
    "--data-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/UCSDped2"),
    help="Dataset main path",
)
@click.option("--clip-length", type=int, default=16, help="Clip length")
@click.option("--load-lstm", is_flag=True, help="Load LSTMs")
@click.option("--hidden-size", type=int, default=100, help="Hidden size")
@click.option("--num-layers", type=int, default=1, help="Number of LSTMs layers")
@click.option("--dropout", type=float, default=0.0, help="Dropout probability")
@click.option("--batch-size", type=int, default=4, help="Batch size")
@click.option("--boundary", type=click.Choice(["soft", "hard"]), default="soft", help="Boundary")
@click.option("--idx_list_enc", type=int, multiple=True, default=[6], help="List of indices of model encoder")
@click.option("--nu", type=float, default=0.0)
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
    idx_list_enc: tuple[int],
    nu: float,
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
    data_holder = DataManager(
        dataset_name="ShanghaiTech", data_path=str(data_path), normal_class=-1, clip_length=clip_length
    ).get_data_holder()
    net = ShanghaiTech(data_holder.shape, code_length, load_lstm, hidden_size, num_layers, dropout)
    fl.client.start_numpy_client(server_address=server_address, client=MoccaClient(net, data_holder, rc))


if __name__ == "__main__":
    cli()
