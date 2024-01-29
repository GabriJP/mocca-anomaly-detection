import logging
import random
import timeit
from collections import deque
from dataclasses import dataclass
from logging import INFO
from pathlib import Path
from typing import Any
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypedDict
from typing import Union

import flwr
import numpy as np
import numpy.typing as npt
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

WANDB_DATA = Dict[str, Union[float, int, bool]]


def _fp16_recon_loss(x_r: torch.Tensor, x: torch.Tensor, *, test: bool) -> torch.Tensor:
    recon_loss = torch.sum(torch.abs(x_r - x), dim=tuple(range(1, x_r.dim())))
    if test:
        return recon_loss
    return torch.mean(recon_loss)


def _mocca_recon_loss(x_r: torch.Tensor, x: torch.Tensor, *, test: bool) -> torch.Tensor:
    recon_loss = torch.sum((x_r - x) ** 2, dim=tuple(range(1, x_r.dim())))
    if test:
        return recon_loss
    return torch.mean(recon_loss)


DISTS = dict(
    l1=_fp16_recon_loss,
    l2=_mocca_recon_loss,
)


class WandbLogger:
    def __init__(self) -> None:
        self.data: Dict[str, WANDB_DATA] = dict()
        self.artifacts: Dict[str, wandb.Artifact] = dict()
        self.step = 0
        self.epoch = 0

    def manual_step(self) -> None:
        if len(self.data):
            self._log()
        else:
            self.step += 1

    def _log(self) -> None:
        wandb.log(self.data, step=self.step, commit=True)
        self.step += 1
        self.data.clear()

    def log_train(self, data: WANDB_DATA, *, key: str = "train") -> None:
        if key in self.data:
            self._log()
        self.data[key] = data

    def log_test(self, data: WANDB_DATA, *, key: str = "test") -> None:
        if self.epoch == 0:
            for metric in data:
                wandb.define_metric(metric, step_metric="epoch", goal="maximize")
        self.log_train(dict(**data, epoch=self.epoch), key=key)
        self._log()

    @staticmethod
    def save_model(save_dict: Dict[str, Any], name: str = "model") -> None:
        if wandb.run is None:
            raise ValueError
        torch.save(save_dict, Path(wandb.run.dir) / f"{name}.pt")


wandb_logger = WandbLogger()


class EarlyStopServer(flwr.server.Server):
    def __init__(
        self,
        *,
        client_manager: Any,
        strategy: Optional[flwr.server.strategy.Strategy] = None,
        patience: int = 0,
        min_delta_pct: float = 0.0,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.patience = patience
        self.min_delta_pct = min_delta_pct
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta_pct):
            self.counter += 1

        return self.counter >= self.patience

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> flwr.server.History:
        """Run federated averaging for a number of rounds."""
        history = flwr.server.History()

        # Initialize parameters
        flwr.common.logger.log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        flwr.common.logger.log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            flwr.common.logger.log(INFO, "initial parameters (loss, other metrics): %s, %s", res[0], res[1])
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        flwr.common.logger.log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                flwr.common.logger.log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(server_round=current_round, metrics=evaluate_metrics_fed)
                if self.early_stop(loss_fed):
                    flwr.common.logger.log(INFO, "FL early stop")
                    break

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        flwr.common.logger.log(INFO, "FL finished in %s", elapsed)
        return history


class EarlyStoppingDM:
    def __init__(self, initial_patience: int, rolling_factor: int, es_patience: int) -> None:
        self.initial_patience = initial_patience
        self.rolling_factor = rolling_factor
        self.step = -1
        self.losses: Deque[float] = deque(maxlen=rolling_factor)
        self.losses.append(0.0)
        self.prev_mean = 0.0
        self.early_stops: Deque[float] = deque([False] * es_patience, maxlen=es_patience)
        self.es = False

    def log_loss(self, new_loss: float) -> Dict[str, float]:
        self.step += 1
        self.losses.append(new_loss)

        losses: npt.NDArray[np.float64] = np.sort(np.array(self.losses, dtype=np.float64))
        current_mean = float(np.mean(np.sort(losses)))
        current_std = float(np.std(losses, ddof=1))
        current_pend = self.prev_mean - current_mean

        self.prev_mean = current_mean
        self.early_stops.append(current_std > current_pend)

        self.early_stop = all(self.early_stops)

        return dict(mean=current_mean, std=current_std, pend=current_pend)

    @property
    def early_stop(self) -> bool:
        return self.es

    @early_stop.setter
    def early_stop(self, es: bool) -> None:
        self.es = self.es or (self.step >= max(self.initial_patience, self.rolling_factor) and es)


@dataclass
class FullRunConfig:
    seed: int
    n_workers: int
    output_path: Path
    log_frequency: int
    disable_logging: bool
    debug: bool
    # Model config
    code_length: int
    model_ckp: Optional[Path]
    # Optimizer config
    optimizer: str
    ae_learning_rate: float
    learning_rate: float
    ae_weight_decay: float
    weight_decay: float
    ae_lr_milestones: List[int]
    lr_milestones: List[int]
    # Data
    data_path: Path
    clip_length: int
    # Training config
    # LSTMs
    load_lstm: bool
    bidirectional: bool
    hidden_size: int
    num_layers: int
    dropout: float
    # Autoencoder
    end_to_end_training: bool
    warm_up_n_epochs: int
    use_selectors: bool
    batch_accumulation: int
    train: bool
    test: bool
    train_best_conf: bool
    batch_size: int
    boundary: str
    idx_list_enc: List[int]
    epochs: int
    ae_epochs: int
    nu: float
    fp16: bool
    compile: bool
    dist: str
    normal_class: int = -1


@dataclass
class RunConfig:
    n_workers: int
    output_path: Path
    code_length: int
    learning_rate: float
    weight_decay: float
    data_path: Path
    clip_length: int
    load_lstm: bool
    bidirectional: bool
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    boundary: str
    idx_list_enc: Tuple[int, ...]
    nu: float
    fp16: bool
    compile: bool
    dist: str
    optimizer: str = "adam"
    lr_milestones: Tuple[int, ...] = tuple()
    end_to_end_training: bool = True
    debug: bool = False
    warm_up_n_epochs: int = 0
    epochs: int = 0
    log_frequency: int = 1


def get_out_dir(rc: FullRunConfig, pretrain: bool, aelr: float, dset_name: str = "cifar10") -> Tuple[Path, str]:
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


def get_out_dir2(rc: RunConfig) -> Tuple[Path, str]:
    tmp_name = (
        f"train-mn_ShanghaiTech-cl_{rc.code_length}-bs_{rc.batch_size}-nu_{rc.nu}-lr_{rc.learning_rate}-"
        f"bd_{rc.boundary}-sl_False-ile_{'.'.join(map(str, rc.idx_list_enc))}-lstm_{rc.load_lstm}-"
        f"bidir_{rc.bidirectional}-hs_{rc.hidden_size}-nl_{rc.num_layers}-dp_{rc.dropout}"
    )
    out_dir = (rc.output_path / "ShanghaiTech" / "train_end_to_end" / tmp_name).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir, tmp_name


def set_seeds(seed: int) -> None:
    """Set all seeds.

    Parameters
    ----------
    seed : int
        Seed

    """
    # Set the seed only if the user specified it
    np.seterr(all="raise")
    if seed == -1:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def purge_params(encoder_net: nn.Module, ae_net_cehckpoint: str) -> None:
    """Load Encoder preatrained weights from the full AutoEncoder.
    After the pretraining phase, we don't need the full AutoEncoder parameters, we only need the Encoder

    Parameters
    ----------
    encoder_net :
        The Encoder network
    ae_net_cehckpoint : str
        Path to full AutoEncoder checkpoint

    """
    # Load the full AutoEncoder checkpoint dict
    ae_net_dict = torch.load(ae_net_cehckpoint, map_location=lambda storage, loc: storage)["ae_state_dict"]

    # Load encoder weight from autoencoder
    net_dict = encoder_net.state_dict()

    # Filter out decoder network keys
    st_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}

    # Overwrite values in the existing state_dict
    net_dict.update(st_dict)

    # Load the new state_dict
    encoder_net.load_state_dict(net_dict, strict=True)


class TorchDict(TypedDict):
    net_state_dict: Dict[str, Any]
    R: Dict[str, torch.Tensor]
    config: Union[FullRunConfig, RunConfig]


def save_model(
    path: Path,
    net: torch.nn.Module,
    r: Dict[str, torch.Tensor],
    config: Union[FullRunConfig, RunConfig],
) -> None:
    torch.save(dict(net_state_dict=net.state_dict(), R=r, config=config), path)


def load_model(path: Path, **load_kwargs: Any) -> TorchDict:
    return torch.load(path, **load_kwargs)


def eval_spheres_centers(
    train_loader: DataLoader[Tuple[torch.Tensor, int]],
    encoder_net: torch.nn.Module,
    ae_net_cehckpoint: str,
    use_selectors: bool,
    device: str,
    debug: bool,
) -> Dict[str, torch.Tensor]:
    """Eval the centers of the hyperspheres at each chosen layer.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader for trainin data
    encoder_net : torch.nn.Module
        Encoder network
    ae_net_cehckpoint : str
        Checkpoint of the full AutoEncoder
    use_selectors : bool
        True if we want to use selector models
    device : str
        Device on which run the computations
    debug : bool
        Activate debug mode

    Returns
    -------
    dict : dictionary
        Dictionary with k='layer name'; v='features vector representing hypersphere center'

    """
    logger = logging.getLogger()

    centers_files = ae_net_cehckpoint[:-4] + f"_w_centers_{use_selectors}.pth"

    # If centers are found, then load and return
    if Path(centers_files).exists():
        logger.info("Found hyperspheres centers")
        ae_net_ckp = torch.load(centers_files, map_location=lambda storage, loc: storage)

        centers = {k: v.to(device) for k, v in ae_net_ckp["centers"].items()}
    else:
        logger.info("Hyperspheres centers not found... evaluating...")
        centers_ = init_center_c(train_loader=train_loader, encoder_net=encoder_net, device=device, debug=debug)

        logger.info("Hyperspheres centers evaluated!!!")
        new_ckp = ae_net_cehckpoint.split(".pth")[0] + f"_w_centers_{use_selectors}.pth"

        logger.info(f"New AE dict saved at: {new_ckp}!!!")
        centers = {k: v for k, v in centers_.items()}

        torch.save({"ae_state_dict": torch.load(ae_net_cehckpoint)["ae_state_dict"], "centers": centers}, new_ckp)

    return centers


@torch.no_grad()
def init_center_c(
    train_loader: DataLoader[Tuple[torch.Tensor, int]],
    encoder_net: torch.nn.Module,
    device: str,
    debug: bool,
    eps: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Initialize hypersphere center as the mean from an initial forward pass on the data."""
    n_samples = 0

    encoder_net.eval().to(device)

    for idx, (data, _) in enumerate(
        tqdm(train_loader, desc="Init hyperspheres centeres", total=len(train_loader), leave=False)
    ):
        if debug and idx == 5:
            break

        data = data.to(device)
        n_samples += data.shape[0]

        zipped = encoder_net(data)

        if idx == 0:
            c = {item[0]: torch.zeros_like(item[1][-1], device=device) for item in zipped}

        for item in zipped:
            c[item[0]] += torch.sum(item[1], dim=0)

    for k in c.keys():
        c[k] = c[k] / n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[k][(abs(c[k]) < eps) & (c[k] < 0)] = -eps
        c[k][(abs(c[k]) < eps) & (c[k] > 0)] = eps

    return c
