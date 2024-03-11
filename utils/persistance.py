from pathlib import Path
from typing import Any
from typing import TypedDict

import torch

from .configs import FullRunConfig
from .configs import RunConfig


class TorchDict(TypedDict):
    net_state_dict: dict[str, Any]
    R: dict[str, torch.Tensor]
    config: FullRunConfig | RunConfig


def get_out_dir(rc: FullRunConfig, pretrain: bool, aelr: float, dset_name: str = "cifar10") -> tuple[Path, str]:
    """Creates training output dir

    Parameters
    ----------

    rc :
        Arguments
    pretrain : bool
        True if pretrain the model
    aelr : float
        Full AutoEncoder learning rate
    dset_name : str
        Dataset name
        ................................................................

    Returns
    -------
    out_dir : str
        Path to output folder
    tmp : str
        String containing infos about the current experiment setup

    """
    output_path = Path(rc.output_path)
    if dset_name == "ShanghaiTech":
        if pretrain:
            tmp = f"pretrain-mn_{dset_name}-cl_{rc.code_length}-lr_{rc.ae_learning_rate}"
            out_dir = output_path / dset_name / "pretrain" / tmp
        else:
            tmp = (
                f"train-mn_{dset_name}-cl_{rc.code_length}-bs_{rc.batch_size}-nu_{rc.nu}-lr_{rc.learning_rate}-"
                f"bd_{rc.boundary}-sl_{rc.use_selectors}-ile_{'.'.join(map(str, rc.idx_list_enc))}-"
                f"lstm_{rc.load_lstm}-bidir_{rc.bidirectional}-hs_{rc.hidden_size}-nl_{rc.num_layers}-"
                f"dp_{rc.dropout}"
            )
            out_dir = output_path / dset_name / "train" / tmp
            if rc.end_to_end_training:
                out_dir = output_path / dset_name / "train_end_to_end" / tmp
    else:
        if pretrain:
            tmp = (
                f"pretrain-mn_{dset_name}-nc_{rc.normal_class}-cl_{rc.code_length}-lr_{rc.ae_learning_rate}-"
                f"awd_{rc.ae_weight_decay}"
            )
            out_dir = output_path / dset_name / str(rc.normal_class) / "pretrain" / tmp

        else:
            tmp = (
                f"train-mn_{dset_name}-nc_{rc.normal_class}-cl_{rc.code_length}-bs_{rc.batch_size}-nu_{rc.nu}-"
                f"lr_{rc.learning_rate}-wd_{rc.weight_decay}-bd_{rc.boundary}-alr_{aelr}-sl_{rc.use_selectors}-"
                f"ep_{rc.epochs}-ile_{'.'.join(map(str, rc.idx_list_enc))}"
            )
            out_dir = output_path / dset_name / str(rc.normal_class) / "train" / tmp

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir, tmp


def get_out_dir2(rc: RunConfig) -> tuple[Path, str]:
    tmp_name = (
        f"train-mn_ShanghaiTech-cl_{rc.code_length}-bs_{rc.batch_size}-nu_{rc.nu}-lr_{rc.learning_rate}-"
        f"bd_{rc.boundary}-sl_False-ile_{'.'.join(map(str, rc.idx_list_enc))}-lstm_{rc.load_lstm}-"
        f"bidir_{rc.bidirectional}-hs_{rc.hidden_size}-nl_{rc.num_layers}-dp_{rc.dropout}"
    )
    out_dir = (rc.output_path / "ShanghaiTech" / "train_end_to_end" / tmp_name).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir, tmp_name


def save_model(
    path: Path,
    net: torch.nn.Module,
    r: dict[str, torch.Tensor],
    config: FullRunConfig | RunConfig,
) -> None:
    torch.save(dict(net_state_dict=net.state_dict(), R=r, config=config), path)


def load_model(path: Path, **load_kwargs: Any) -> TorchDict:
    return torch.load(path, **load_kwargs)
