from collections.abc import Iterable
from pathlib import Path
from typing import Any
from typing import Union

import torch
import wandb

WANDB_DATA = dict[str, Union[float, int, bool]]


class WandbLogger:
    def __init__(self) -> None:
        self.data: dict[str, float | int | bool | WANDB_DATA] = dict()
        self.artifacts: dict[str, wandb.Artifact] = dict()
        self.step = 0

    @staticmethod
    def add_epoch_metrics(epoch_metrics: Iterable[str] = ()) -> None:
        epoch_metric = wandb.define_metric("epoch", hidden=True)
        for metric in epoch_metrics:
            wandb.define_metric(metric, step_metric=epoch_metric.name, goal="maximize")

    def manual_step(self) -> None:
        if not self._log():
            self.step += 1

    def _log(self) -> bool:
        if not self.data:
            return False
        wandb.log(self.data, step=self.step, commit=True)
        self.step += 1
        self.data.clear()
        return True

    def log_train(self, data: WANDB_DATA, *, key: str = "train") -> None:
        if key in self.data:
            self._log()
        self.data[key] = data

    def log_test(self, data: WANDB_DATA, epoch: int, *, key: str = "test") -> None:
        self.data["epoch"] = epoch
        self.log_train(data, key=key)
        self._log()

    @staticmethod
    def save_model(save_dict: dict[str, Any], name: str = "model") -> None:
        if wandb.run is None:
            raise ValueError
        torch.save(save_dict, Path(wandb.run.dir) / f"{name}.pt")


wandb_logger = WandbLogger()
