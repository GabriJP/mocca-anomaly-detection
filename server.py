from pathlib import Path
from typing import Callable
from typing import Optional

import click
import flwr as fl
from flwr.common import Config
from flwr.server import SimpleClientManager
from flwr.server.strategy import FedProx

from utils import EarlyStopServer


def create_fit_config_fn(epochs: int, warm_up_n_epochs: int, batch_size: int) -> Callable[[int], Config]:
    def inner(_: int) -> Config:
        return dict(epochs=epochs, warm_up_n_epochs=warm_up_n_epochs, batch_size=batch_size)

    return inner


@click.command()
@click.option("--num_rounds", type=click.IntRange(1), default=5)
@click.option("--epochs", type=click.IntRange(1), default=5)
@click.option("--warm_up_n_epochs", type=click.IntRange(0), default=0)
@click.option("--batch_size", type=click.IntRange(1), default=5)
@click.option("--proximal_mu", type=click.FloatRange(0, 1), default=1.0)
@click.option("--patience", type=click.IntRange(0), default=None)
@click.option("--min_delta_pct", type=click.FloatRange(0, 1), default=None)
def cli(
    num_rounds: int,
    epochs: int,
    warm_up_n_epochs: int,
    batch_size: int,
    proximal_mu: float,
    patience: Optional[int],
    min_delta_pct: Optional[float],
) -> None:
    strategy = FedProx(
        on_fit_config_fn=create_fit_config_fn(epochs, warm_up_n_epochs, batch_size), proximal_mu=proximal_mu
    )
    server = (
        None
        if patience is None or min_delta_pct is None
        else EarlyStopServer(
            client_manager=SimpleClientManager(), strategy=strategy, patience=patience, min_delta_pct=min_delta_pct
        )
    )
    certificates_path = Path.home() / "certs"
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        server=server,
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
