from collections import Counter
from pathlib import Path
from typing import Counter as tCounter

import click
import numpy as np
import numpy.typing as npt
from prettytable import PrettyTable

from data_prep_src import process_shang
from data_prep_src import process_ucsd as src_process_ucsd

U8_NDTYPE = npt.NDArray[np.uint8]


@click.group()
def tui() -> None:
    pass


@tui.command()
@click.argument("data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
@click.option("--cuda", is_flag=True)
def process_ucsd(data_root: Path, cuda: bool) -> None:
    src_process_ucsd(data_root, cuda)


@tui.command()
@click.argument("data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
@click.option("--cuda", is_flag=True)
def process_shanghai(data_root: Path, cuda: bool) -> None:
    process_shang(data_root, cuda)


@tui.command()
@click.argument("data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
def count_classes(data_root: Path) -> None:
    table = PrettyTable()
    table.field_names = ["Path", "Normal count", "Anomaly count", "Unbalance pct"]
    table.sortby = "Unbalance pct"
    table.float_format = "0.2"
    for p in data_root.glob("**/*.npy"):
        a: npt.NDArray[np.uint8] = np.load(p)
        if a.ndim > 1:
            continue
        c: tCounter[int] = Counter(a.tolist())
        table.add_row([p, c[0], c[1], (c[1] - c[0]) / (c[1] + c[0])])
    print(table)


@tui.command()
@click.argument("data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
@click.option("--cuda", is_flag=True)
@click.pass_context
def process_all(ctx: click.Context, data_root: Path, cuda: bool) -> None:
    ctx.invoke(process_ucsd, data_root=data_root, cuda=cuda)
    ctx.invoke(process_shanghai, data_root=data_root, cuda=cuda)


if __name__ == "__main__":
    tui()
    # _process_ucsd(Path("data"), False)
