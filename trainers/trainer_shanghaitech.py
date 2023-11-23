import logging
import time
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.shanghaitech_model import ShanghaiTechEncoder
from utils import EarlyStoppingDM
from utils import FullRunConfig
from utils import RunConfig
from utils import wandb_logger


def train(
    net: nn.Module,
    train_loader: DataLoader[Tuple[torch.Tensor, int]],
    out_dir: Path,
    device: str,
    ae_net_checkpoint: Optional[Path],
    rc: Union[FullRunConfig, RunConfig],
    r: Dict[str, torch.Tensor],
    mu: float = 0.0,
    es: Optional[EarlyStoppingDM] = None,
) -> Path:
    logger = logging.getLogger()

    global_params = list(net.parameters())

    idx_list_enc = [int(i) for i in rc.idx_list_enc]

    # Set device for network
    net = net.to(device)

    # Set optimizer
    optimizer = (
        Adam(net.parameters(), lr=rc.learning_rate, weight_decay=rc.weight_decay, eps=1e-4)
        if rc.optimizer == "adam"
        else SGD(net.parameters(), lr=rc.learning_rate, weight_decay=rc.weight_decay, momentum=0.9)
    )

    # Set learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=rc.lr_milestones, gamma=0.1)

    # Initialize hypersphere center c
    if len(r):
        keys = list(r.keys())
    else:
        logger.info("Evaluating hypersphere centers...")
        keys = get_keys(idx_list_enc)
        logger.info(f"Keys: {keys}")
        logger.info("Done!")

        r = {k: torch.tensor(0.0, device=device) for k in keys}

    # Training
    logger.info("Starting training...")
    warm_up_n_epochs = rc.warm_up_n_epochs
    net.train()
    scaler = torch.cuda.amp.GradScaler()

    best_loss = 1e12
    net_checkpoint = Path()
    es_data = dict()
    for epoch in range(1 if rc.debug else rc.epochs):
        one_class_loss = 0.0
        recon_loss = 0.0
        objective_loss = 0.0
        n_batches = 0
        d_from_c = {k: 0.0 for k in keys}

        # Zero the network parameter gradients
        optimizer.zero_grad()

        for idx, (data, _) in tqdm(
            enumerate(train_loader, 1), desc=f"Training epoch: {epoch + 1}", total=len(train_loader)
        ):
            if rc.debug and idx == 2:
                break

            n_batches += 1
            data = data.to(device)

            # Update network parameters via backpropagation: forward + backward + optimize
            with torch.autocast(device_type=device, enabled=False):
                x_r, _, d_lstms = net(data)
                recon_loss_ = torch.mean(torch.sum(torch.abs(x_r - data), dim=tuple(range(1, x_r.dim()))))
                dist, one_class_loss_ = eval_ad_loss(d_lstms, r, rc.nu, rc.boundary, device)
                objective_loss_ = one_class_loss_ + recon_loss_
                if torch.isinf(objective_loss_):
                    objective_loss_.fill_(torch.finfo(torch.float16).max)

                if es is not None:
                    es_data = es.log_loss(objective_loss_.item())

                if mu > 0:
                    proximal_term = sum(
                        (local_weights - global_weights).norm(2)
                        for local_weights, global_weights in zip(net.parameters(), global_params)
                    )
                    objective_loss_ += mu / 2 * proximal_term

                for k in keys:
                    d_from_c[k] += torch.mean(dist[k]).item()

            scaler.scale(objective_loss_).backward()
            # if (idx + 1) % 5 == 0 or (idx + 1 == len(train_loader)):
            if True:
                # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                # Zero the network parameter gradients
                optimizer.zero_grad()

            one_class_loss += one_class_loss_.item()
            recon_loss += recon_loss_.item()
            objective_loss += objective_loss_.item()

            # if idx % (len(train_loader) // rc.log_frequency) == 0:
            logger.debug(
                f"TRAIN at epoch: {epoch} [{idx}]/[{len(train_loader)}] ==> "
                f"\n\t\t\t\tReconstr Loss : {recon_loss / n_batches:.4f}"
                f"\n\t\t\t\tOne class Loss: {one_class_loss / n_batches:.4f}"
                f"\n\t\t\t\tObjective Loss: {objective_loss / n_batches:.4f}"
            )
            log_data = dict(
                recon_loss=recon_loss / n_batches,
                one_class_loss=one_class_loss / n_batches,
                objective_loss=objective_loss / n_batches,
            )
            for k in keys:
                logger.info(f"[{k}] -- Radius: {r[k]:.4f} - " f"Dist from sphere centr: {d_from_c[k] / n_batches:.4f}")
                log_data[f"radius_{k}"] = float(r[k].data.cpu().numpy())
                log_data[f"distance_c_sphere_{k}"] = d_from_c[k] / n_batches
            wandb_logger.log_train(log_data)
            wandb_logger.log_train(es_data, key="es")

            # Update hypersphere radius R on mini-batch distances
            if rc.boundary != "soft" or epoch < warm_up_n_epochs:
                continue
            for k in r.keys():
                r[k].data = torch.tensor(
                    np.quantile(np.sqrt(dist[k].clone().data.cpu().numpy()), 1 - rc.nu), device=device
                )

        if epoch in rc.lr_milestones:
            logger.info(f"  LR scheduler: new learning rate is {float(scheduler.get_lr()):g}")

        time_ = time.time() if ae_net_checkpoint is None else ae_net_checkpoint.name.split("_")[-1].split(".p")[0]
        net_checkpoint = out_dir / f"net_ckp_{epoch}_{time_}.pth"
        torch.save(dict(net_state_dict=net.state_dict(), R=r), net_checkpoint)
        logger.info(f"Saved model at: {net_checkpoint}")
        if objective_loss < best_loss or epoch == 0:
            best_loss = objective_loss
            wandb_logger.save_model(dict(net_state_dict=net.state_dict(), R=r), name="best_model")
        if es is not None and es.early_stop:
            logger.info("Early stopping")
            break

    logger.info("Finished training.")

    return net_checkpoint


def get_keys(idx_list_enc: Sequence[int]) -> List[str]:
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    names = ShanghaiTechEncoder.get_names()
    return [names[i] for i in idx_list_enc]


def eval_ad_loss(
    d_lstms: Dict[str, torch.Tensor], R: Dict[str, torch.Tensor], nu: float, boundary: str, device: str
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    dist = dict()
    loss = torch.tensor(1.0, device=device)

    for k in R.keys():
        dist[k] = torch.sum(d_lstms[k] ** 2, dim=-1)

        if boundary == "soft":
            scores = dist[k] - R[k] ** 2
            loss += R[k] ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss += torch.mean(dist[k])

    return dist, loss
