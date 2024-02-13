from collections import Counter
from collections import Counter as tCounter
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from prettytable import PrettyTable

from data_prep_src import process_shang
from data_prep_src import process_ucsd as src_process_ucsd
from data_prep_src.shang import generate_all_subsets

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
def process_shanghai_subsets(data_root: Path) -> None:
    generate_all_subsets(data_root)


@tui.command()
@click.argument("data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
def count_classes(data_root: Path) -> None:
    table = PrettyTable()
    table.field_names = ["Path", "Normal count", "Anomaly count", "Unbalance pct"]
    table.sortby = "Unbalance pct"
    table.float_format = "0.2"
    pcts: list[float] = list()
    for p in data_root.glob("**/*.npy"):
        a: npt.NDArray[np.uint8] = np.load(p, mmap_mode="r")
        if a.ndim > 1:
            continue
        c: tCounter[int] = Counter(a.tolist())
        pct = (c[1] - c[0]) / (c[1] + c[0])
        table.add_row([p, c[0], c[1], pct])
        pcts.append(pct)
    print(table)
    plt.hist(pcts, range=(-1.0, 1.0))
    plt.show()


@tui.command()
@click.argument("data_root", type=click.Path(file_okay=False, path_type=Path), default=Path("./data"))
@click.option("--cuda", is_flag=True)
@click.pass_context
def process_all(ctx: click.Context, data_root: Path, cuda: bool) -> None:
    ctx.invoke(process_ucsd, data_root=data_root, cuda=cuda)
    ctx.invoke(process_shanghai, data_root=data_root, cuda=cuda)
    ctx.invoke(process_shanghai_subsets, data_root=data_root)


if __name__ == "__main__":
    tui()
    # _process_ucsd(Path("data"), False)
