import argparse
import logging
import os
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from client import RunConfig


def pretrain(
    ae_net: nn.Module,
    train_loader: DataLoader,
    out_dir: str,
    tb_writer: SummaryWriter,
    device: str,
    args: argparse.Namespace,
) -> str:
    logger = logging.getLogger()

    ae_net = ae_net.train().to(device)

    # Set optimizer
    optimizer: torch.optim.Optimizer
    if args.optimizer == "adam":
        optimizer = Adam(ae_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(ae_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    scheduler = MultiStepLR(optimizer, milestones=args.ae_lr_milestones, gamma=0.1)

    ae_epochs = 1 if args.debug else args.ae_epochs
    it_t = 0
    logger.info("Start Pretraining the autoencoder...")
    ae_net_checkpoint = ""
    for epoch in range(ae_epochs):
        recon_loss = 0.0
        n_batches = 0
        for idx, (data, _) in enumerate(tqdm(train_loader), 1):
            if args.debug and idx == 2:
                break

            data = data.to(device)
            optimizer.zero_grad()
            x_r = ae_net(data)[0]
            recon_loss_ = torch.mean(torch.sum((x_r - data) ** 2, dim=tuple(range(1, x_r.dim()))))
            recon_loss_.backward()
            optimizer.step()

            recon_loss += recon_loss_.item()
            n_batches += 1

            if idx % (len(train_loader) // args.log_frequency) == 0:
                logger.info(
                    f"PreTrain at epoch: {epoch + 1} [{idx}]/[{len(train_loader)}] ==> "
                    f"Recon Loss: {recon_loss / idx:.4f}"
                )
                tb_writer.add_scalar("pretrain/recon_loss", recon_loss / idx, it_t)
                it_t += 1

        scheduler.step()
        if epoch in args.ae_lr_milestones:
            logger.info(f"  LR scheduler: new learning rate is {float(scheduler.get_lr()):g}")

        ae_net_checkpoint = os.path.join(out_dir, f"ae_ckp_epoch_{epoch}_{time.time()}.pth")
        torch.save(dict(ae_state_dict=ae_net.state_dict()), ae_net_checkpoint)

    logger.info("Finished pretraining.")
    logger.info(f"Saved autoencoder at: {ae_net_checkpoint}")

    return ae_net_checkpoint


def train(
    net: nn.Module,
    train_loader: DataLoader,
    out_dir: str,
    tb_writer: SummaryWriter,
    device: Union[str, torch.device],
    ae_net_checkpoint: Optional[str],
    args: Union[argparse.Namespace, RunConfig],
    c: Optional[Dict[str, torch.Tensor]] = None,
    R: Optional[Dict[str, torch.Tensor]] = None,
) -> str:
    logger = logging.getLogger()

    idx_list_enc = [int(i) for i in args.idx_list_enc]

    # Set device for network
    net = net.to(device)

    # Set optimizer
    optimizer: torch.optim.Optimizer
    if args.optimizer == "adam":
        optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    # Set learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    # Initialize hypersphere center c
    if c is not None and R is not None:
        keys = list(c.keys())
    else:
        logger.info("Evaluating hypersphere centers...")
        c, keys = init_center_c(train_loader, net, idx_list_enc, device, args.end_to_end_training, args.debug)
        logger.info(f"Keys: {keys}")
        logger.info("Done!")

        R = {k: torch.tensor(0.0, device=device) for k in keys}

    # Training
    logger.info("Starting training...")
    warm_up_n_epochs = args.warm_up_n_epochs
    net.train()
    it_t = 0

    best_loss = 1e12
    epochs = 1 if args.debug else args.epochs
    net_checkpoint = ""
    for epoch in range(epochs):
        one_class_loss = 0.0
        recon_loss = 0.0
        objective_loss = 0.0
        n_batches = 0
        d_from_c = {k: 0.0 for k in keys}

        for idx, (data, _) in enumerate(
            tqdm(train_loader, total=len(train_loader), desc=f"Training epoch: {epoch + 1}"), 1
        ):
            if args.debug and idx == 2:
                break

            n_batches += 1
            data = data.to(device)

            # Zero the network parameter gradients
            optimizer.zero_grad()

            # Update network parameters via backpropagation: forward + backward + optimize
            if args.end_to_end_training:
                x_r, _, d_lstms = net(data)
                recon_loss_ = torch.mean(torch.sum((x_r - data) ** 2, dim=tuple(range(1, x_r.dim()))))
            else:
                _, d_lstms = net(data)
                recon_loss_ = torch.tensor([0.0], device=device)

            dist, one_class_loss_ = eval_ad_loss(d_lstms, c, R, args.nu, args.boundary)
            objective_loss_ = one_class_loss_ + recon_loss_

            for k in keys:
                d_from_c[k] += torch.mean(dist[k]).item()

            objective_loss_.backward()
            optimizer.step()

            one_class_loss += one_class_loss_.item()
            recon_loss += recon_loss_.item()
            objective_loss += objective_loss_.item()

            if idx % (len(train_loader) // args.log_frequency) == 0:
                logger.info(
                    f"TRAIN at epoch: {epoch} [{idx}]/[{len(train_loader)}] ==> "
                    f"\n\t\t\t\tReconstr Loss : {recon_loss / n_batches:.4f}"
                    f"\n\t\t\t\tOne class Loss: {one_class_loss / n_batches:.4f}"
                    f"\n\t\t\t\tObjective Loss: {objective_loss / n_batches:.4f}"
                )
                tb_writer.add_scalar("train/recon_loss", recon_loss / n_batches, it_t)
                tb_writer.add_scalar("train/one_class_loss", one_class_loss / n_batches, it_t)
                tb_writer.add_scalar("train/objective_loss", objective_loss / n_batches, it_t)
                for k in keys:
                    logger.info(
                        f"[{k}] -- Radius: {R[k]:.4f} - " f"Dist from sphere centr: {d_from_c[k] / n_batches:.4f}"
                    )
                    tb_writer.add_scalar(f"train/radius_{k}", R[k], it_t)
                    tb_writer.add_scalar(f"train/distance_c_sphere_{k}", d_from_c[k] / n_batches, it_t)
                    it_t += 1

            # Update hypersphere radius R on mini-batch distances
            if args.boundary != "soft" or epoch < warm_up_n_epochs:
                continue
            for k in R.keys():
                R[k].data = torch.tensor(
                    np.quantile(np.sqrt(dist[k].clone().data.cpu().numpy()), 1 - args.nu), device=device
                )

        scheduler.step()
        if epoch in args.lr_milestones:
            logger.info(f"  LR scheduler: new learning rate is {float(scheduler.get_lr()):g}")

        time_ = time.time() if ae_net_checkpoint is None else ae_net_checkpoint.split("_")[-1].split(".p")[0]
        net_checkpoint = os.path.join(out_dir, f"net_ckp_{epoch}_{time_}.pth")
        torch.save(dict(net_state_dict=net.state_dict(), R=R, c=c), net_checkpoint)
        logger.info(f"Saved model at: {net_checkpoint}")
        if objective_loss < best_loss or epoch == 0:
            best_loss = objective_loss
            best_model_checkpoint = os.path.join(out_dir, f"net_ckp_best_model_{time_}.pth")
            torch.save(dict(net_state_dict=net.state_dict(), R=R, c=c), best_model_checkpoint)

    logger.info("Finished training.")

    return net_checkpoint


@torch.no_grad()
def init_center_c(
    train_loader: DataLoader,
    net: nn.Module,
    idx_list_enc: Sequence[int],
    device: Union[str, torch.device],
    debug: bool,
    eps: float = 0.1,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    net.eval()

    data, _ = next(iter(train_loader))
    d_lstms = net(data.to(device))[-1]

    keys = []
    c = {}
    for en, k in enumerate(list(d_lstms.keys())):
        if en in idx_list_enc:
            keys.append(k)
            c[k] = torch.zeros_like(d_lstms[k][-1], device=device)

    for idx, (data, _) in enumerate(
        tqdm(train_loader, desc="init hyperspheres centeres", total=len(train_loader), leave=False)
    ):
        if debug and idx == 2:
            break
        # get the inputs of the batch
        n_samples += data.shape[0]
        d_lstms = net(data.to(device))[-1]
        for k in keys:
            c[k] += torch.sum(d_lstms[k], dim=0)

    for k in keys:
        c[k] = c[k] / n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[k][(abs(c[k]) < eps) & (c[k] < 0)] = -eps
        c[k][(abs(c[k]) < eps) & (c[k] > 0)] = eps

    return c, keys


def eval_ad_loss(
    d_lstms: Dict[str, torch.Tensor], c: Dict[str, torch.Tensor], R: Dict[str, torch.Tensor], nu: float, boundary: str
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    dist = {}
    loss = torch.tensor(1.0)

    for k in c.keys():
        dist[k] = torch.sum((d_lstms[k] - c[k].unsqueeze(0)) ** 2, dim=-1)

        if boundary == "soft":
            scores = dist[k] - R[k] ** 2
            loss += R[k] ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss += torch.mean(dist[k])

    return dist, loss
