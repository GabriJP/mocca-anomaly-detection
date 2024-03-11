import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# def purge_params(encoder_net: nn.Module, ae_net_cehckpoint: str) -> None:
#     """Load Encoder preatrained weights from the full AutoEncoder.
#     After the pretraining phase, we don't need the full AutoEncoder parameters, we only need the Encoder
#
#     Parameters
#     ----------
#     encoder_net :
#         The Encoder network
#     ae_net_cehckpoint : str
#         Path to full AutoEncoder checkpoint
#
#     """
#     # Load the full AutoEncoder checkpoint dict
#     ae_net_dict = torch.load(ae_net_cehckpoint, map_location=lambda storage, loc: storage)["ae_state_dict"]
#
#     # Load encoder weight from autoencoder
#     net_dict = encoder_net.state_dict()
#
#     # Filter out decoder network keys
#     st_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
#
#     # Overwrite values in the existing state_dict
#     net_dict.update(st_dict)
#
#     # Load the new state_dict
#     encoder_net.load_state_dict(net_dict, strict=True)


def eval_spheres_centers(
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    encoder_net: torch.nn.Module,
    ae_net_cehckpoint: str,
    use_selectors: bool,
    device: str,
    debug: bool,
) -> dict[str, torch.Tensor]:
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
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    encoder_net: torch.nn.Module,
    device: str,
    debug: bool,
    eps: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Initialize hypersphere center as the mean from an initial forward pass on the data."""
    n_samples = 0

    encoder_net.eval().to(device)
    data: torch.Tensor
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
