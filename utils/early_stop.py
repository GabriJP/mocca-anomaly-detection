import timeit
from collections import deque
from logging import INFO
from typing import Any
from typing import Deque

import flwr
import numpy as np
import numpy.typing as npt


class EarlyStopServer(flwr.server.Server):
    def __init__(
        self,
        *,
        client_manager: Any,
        strategy: flwr.server.strategy.Strategy | None = None,
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
    def fit(self, num_rounds: int, timeout: float | None) -> flwr.server.History:
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

    def log_loss(self, new_loss: float) -> dict[str, float]:
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
