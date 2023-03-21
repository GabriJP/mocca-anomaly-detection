import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List
from typing import Optional

import click
import torch

import wandb
from datasets.data_manager import DataManager
from datasets.shanghaitech_test import VideoAnomalyDetectionResultHelper
from models.shanghaitech_model import ShanghaiTech
from models.shanghaitech_model import ShanghaiTechEncoder
from trainers.trainer_shanghaitech import train as sh_train
from utils import extract_arguments_from_checkpoint
from utils import FullRunConfig
from utils import get_out_dir
from utils import set_seeds


@click.command("cli", context_settings=dict(show_default=True))
@click.option("-s", "--seed", type=int, default=-1, help="Random seed")
@click.option(
    "--n_workers",
    type=int,
    default=8,
    help="Number of workers for data loading. 0 means that the data will be loaded in the main process.",
)
@click.option("--output_path", type=click.Path(file_okay=False, path_type=Path), default="./output")
@click.option("-lf", "--log-frequency", type=int, default=5, help="Log frequency")
@click.option("-dl", "--disable-logging", is_flag=True, help="Disable logging")
@click.option("-db", "--debug", is_flag=True, help="Debug mode")
# Model config
@click.option("-zl", "--code-length", default=2048, type=int, help="Code length")
@click.option(
    "-ck",
    "--model-ckp",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Model checkpoint",
)
# Optimizer config
@click.option("-opt", "--optimizer", type=click.Choice(("adam", "sgd")), default="adam", help="Optimizer")
@click.option("-alr", "--ae-learning-rate", type=float, default=1.0e-4, help="Warm up learning rate")
@click.option("-lr", "--learning-rate", type=float, default=1.0e-4, help="Learning rate")
@click.option("-awd", "--ae-weight-decay", type=float, default=0.5e-6, help="Warm up learning rate")
@click.option("-wd", "--weight-decay", type=float, default=0.5e-6, help="Learning rate")
@click.option("-aml", "--ae-lr-milestones", type=int, multiple=True, help="Pretrain milestone")
@click.option("-ml", "--lr-milestones", type=int, multiple=True, help="Training milestone")
# Data
@click.option(
    "-dp",
    "--data-path",
    type=click.Path(file_okay=False, path_type=Path),
    default="./ShanghaiTech",
    help="Dataset main path",
)
@click.option("-cl", "--clip-length", type=int, default=16, help="Clip length")
# Training config
# LSTMs
@click.option("-ll", "--load-lstm", is_flag=True, help="Load LSTMs")
@click.option("-bdl", "--bidirectional", is_flag=True, help="Bidirectional LSTMs")
@click.option("-hs", "--hidden-size", type=int, default=100, help="Hidden size")
@click.option("-nl", "--num-layers", type=int, default=1, help="Number of LSTMs layers")
@click.option("-drp", "--dropout", type=float, default=0.0, help="Dropout probability")
# Autoencoder
@click.option(
    "-ee",
    "--end-to-end-training",
    is_flag=True,
    help="End-to-End training of the autoencoder",
)
@click.option("-we", "--warm_up_n_epochs", type=int, default=5, help="Warm up epochs")
@click.option("-use", "--use-selectors", is_flag=True, help="Use features selector")
@click.option("-ba", "--batch-accumulation", type=int, default=-1, help="Batch accumulation")
@click.option("-ptr", "--pretrain", is_flag=True, help="Pretrain model")
@click.option("-tr", "--train", is_flag=True, help="Train model")
@click.option("-tt", "--test", is_flag=True, help="Test model")
@click.option("-tbc", "--train-best-conf", is_flag=True, help="Train best configurations")
@click.option("-bs", "--batch-size", type=int, default=4, help="Batch size")
@click.option("-bd", "--boundary", type=click.Choice(("hard", "soft")), default="soft", help="Boundary")
@click.option("-ile", "--idx-list-enc", type=int, multiple=True, default=[6], help="List of indices of model encoder")
@click.option("-e", "--epochs", type=int, default=1, help="Training epochs")
@click.option("-ae", "--ae-epochs", type=int, default=1, help="Warmp up epochs")
@click.option("-nu", "--nu", type=float, default=0.1)
@click.option("--wandb_group", type=str, default=None)
@click.option("--wandb_name", type=str, default=None)
def main(
    seed: int,
    n_workers: int,
    output_path: Path,
    log_frequency: int,
    disable_logging: bool,
    debug: bool,
    # Model config,
    code_length: int,
    model_ckp: Optional[Path],
    # Optimizer config,
    optimizer: str,
    ae_learning_rate: float,
    learning_rate: float,
    ae_weight_decay: float,
    weight_decay: float,
    ae_lr_milestones: List[int],
    lr_milestones: List[int],
    # Data,
    data_path: Path,
    clip_length: int,
    # Training config,
    # LSTMs,
    load_lstm: bool,
    bidirectional: bool,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    # Autoencoder,
    end_to_end_training: bool,
    warm_up_n_epochs: int,
    use_selectors: bool,
    batch_accumulation: int,
    pretrain: bool,
    train: bool,
    test: bool,
    train_best_conf: bool,
    batch_size: int,
    boundary: str,
    idx_list_enc: List[int],
    epochs: int,
    ae_epochs: int,
    nu: float,
    wandb_group: Optional[None],
    wandb_name: Optional[None],
) -> None:
    rc = FullRunConfig(
        seed,
        n_workers,
        output_path,
        log_frequency,
        disable_logging,
        debug,
        code_length,
        model_ckp,
        optimizer,
        ae_learning_rate,
        learning_rate,
        ae_weight_decay,
        weight_decay,
        ae_lr_milestones,
        lr_milestones,
        data_path,
        clip_length,
        load_lstm,
        bidirectional,
        hidden_size,
        num_layers,
        dropout,
        end_to_end_training,
        warm_up_n_epochs,
        use_selectors,
        batch_accumulation,
        pretrain,
        train,
        test,
        train_best_conf,
        batch_size,
        boundary,
        idx_list_enc,
        epochs,
        ae_epochs,
        nu,
    )
    # Set seed
    set_seeds(seed)

    if disable_logging:
        logging.disable(level=logging.INFO)

    # Init logger & print training/warm-up summary
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler("./training.log"), logging.StreamHandler()],
    )

    wandb.init(project="mocca", entity="gabijp", group=wandb_group, name=wandb_name, config=asdict(rc))

    if not any([train, pretrain, end_to_end_training]) and model_ckp is None:
        logging.error("CANNOT TEST MODEL WITHOUT A VALID CHECKPOINT")
        raise ValueError("CANNOT TEST MODEL WITHOUT A VALID CHECKPOINT")

    logging.info(
        "Start run with params:\n"
        f"\n\t\t\t\tEnd to end training : {end_to_end_training}"
        f"\n\t\t\t\tPretrain model      : {pretrain}"
        f"\n\t\t\t\tTrain model         : {train}"
        f"\n\t\t\t\tTest model          : {test}"
        f"\n\t\t\t\tBatch size          : {batch_size}\n"
        f"\n\t\t\t\tAutoEncoder Pretraining"
        f"\n\t\t\t\tPretrain epochs     : {ae_epochs}"
        f"\n\t\t\t\tAE-Learning rate    : {ae_learning_rate}"
        f"\n\t\t\t\tAE-milestones       : {ae_lr_milestones}"
        f"\n\t\t\t\tAE-Weight decay     : {ae_weight_decay}\n"
        f"\n\t\t\t\tEncoder Training"
        f"\n\t\t\t\tClip length         : {clip_length}"
        f"\n\t\t\t\tBoundary            : {boundary}"
        f"\n\t\t\t\tTrain epochs        : {epochs}"
        f"\n\t\t\t\tLearning rate       : {learning_rate}"
        f"\n\t\t\t\tMilestones          : {lr_milestones}"
        f"\n\t\t\t\tUse selectors       : {use_selectors}"
        f"\n\t\t\t\tWeight decay        : {weight_decay}"
        f"\n\t\t\t\tCode length         : {code_length}"
        f"\n\t\t\t\tNu                  : {nu}"
        f"\n\t\t\t\tEncoder list        : {idx_list_enc}\n"
        f"\n\t\t\t\tLSTMs"
        f"\n\t\t\t\tLoad LSTMs          : {load_lstm}"
        f"\n\t\t\t\tBidirectional       : {bidirectional}"
        f"\n\t\t\t\tHidden size         : {hidden_size}"
        f"\n\t\t\t\tNumber of layers    : {num_layers}"
        f"\n\t\t\t\tDropout prob        : {dropout}\n"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init DataHolder class
    data_holder = DataManager(
        dataset_name="ShanghaiTech", data_path=data_path, normal_class=-1, only_test=test
    ).get_data_holder()

    # Load data
    train_loader, _ = data_holder.get_loaders(
        batch_size=batch_size, shuffle_train=True, pin_memory=device == "cuda", num_workers=n_workers
    )
    # Print data infos
    only_test = test and not train and not pretrain
    logging.info("Dataset info:")
    logging.info("\n" f"\n\t\t\t\tBatch size    : {batch_size}")
    if not only_test:
        logging.info(
            f"TRAIN:"
            f"\n\t\t\t\tNumber of clips  : {len(train_loader.dataset)}"  # type: ignore
            f"\n\t\t\t\tNumber of batches : {len(train_loader.dataset) // batch_size}"  # type: ignore
        )

    #
    # Train the AUTOENCODER on the RECONSTRUCTION task and then train only the #
    # ENCODER on the ONE CLASS OBJECTIVE #
    #
    net_checkpoint: Optional[Path] = None
    if train and not end_to_end_training:
        if net_checkpoint is None:
            if model_ckp is None:
                logging.info("CANNOT TRAIN MODEL WITHOUT A VALID CHECKPOINT")
                sys.exit(0)
            net_checkpoint = model_ckp

        aelr = float(net_checkpoint.parent.name.split("-")[4].split("_")[-1])

        out_dir, tmp = get_out_dir(rc, pretrain=False, aelr=aelr, dset_name="ShanghaiTech")

        # Init Encoder
        net: torch.nn.Module = ShanghaiTechEncoder(
            data_holder.shape,
            code_length,
            load_lstm,
            hidden_size,
            num_layers,
            dropout,
            bidirectional,
            use_selectors,
        )

        # Load encoder weight from autoencoder
        net_dict = net.state_dict()
        logging.info(f"Loading encoder from: {net_checkpoint}")
        ae_net_dict = torch.load(net_checkpoint, map_location=lambda storage, loc: storage)["ae_state_dict"]

        # Filter out decoder network keys
        st_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(st_dict)
        # Load the new state_dict
        net.load_state_dict(net_dict)

        # TRAIN
        net_checkpoint = sh_train(net, train_loader, out_dir, device, net_checkpoint, rc)

    #
    #

    #
    # Train the AUTOENCODER on the combined objective: #
    # RECONSTRUCTION + ONE CLASS #
    #
    if end_to_end_training:
        out_dir, tmp = get_out_dir(rc, pretrain=False, aelr=int(learning_rate), dset_name="ShanghaiTech")

        # Init AutoEncoder
        ae_net = ShanghaiTech(
            data_holder.shape,
            code_length,
            load_lstm,
            hidden_size,
            num_layers,
            dropout,
            bidirectional,
            use_selectors,
        )
        # End to end TRAIN
        net_checkpoint = sh_train(ae_net, train_loader, out_dir, device, None, rc)
    #
    #

    #
    # Model test #
    #
    if test:
        if net_checkpoint is None:
            net_checkpoint = model_ckp

        if net_checkpoint is None:
            raise ValueError
        (
            code_length,
            batch_size,
            boundary,
            use_selectors,
            idx_list_enc,
            load_lstm,
            hidden_size,
            num_layers,
            dropout,
            bidirectional,
            dataset_name,
            train_type,
        ) = extract_arguments_from_checkpoint(net_checkpoint)

        # Init dataset
        dataset = data_holder.get_test_data()
        net = (
            ShanghaiTech(
                data_holder.shape,
                code_length,
                load_lstm,
                hidden_size,
                num_layers,
                dropout,
                bidirectional,
                use_selectors,
            )
            if train_type == "train_end_to_end"
            else ShanghaiTechEncoder(
                dataset.shape, code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional, use_selectors
            )
        )
        st_dict = torch.load(net_checkpoint)

        net.load_state_dict(st_dict["net_state_dict"])
        logging.info(f"Loaded model from: {net_checkpoint}")
        logging.info(
            f"Start test with params:"
            f"\n\t\t\t\tDataset        : {dataset_name}"
            f"\n\t\t\t\tCode length    : {code_length}"
            f"\n\t\t\t\tEnc layer list : {idx_list_enc}"
            f"\n\t\t\t\tBoundary       : {boundary}"
            f"\n\t\t\t\tUse Selectors  : {use_selectors}"
            f"\n\t\t\t\tBatch size     : {batch_size}"
            f"\n\t\t\t\tN workers      : {n_workers}"
            f"\n\t\t\t\tLoad LSTMs     : {load_lstm}"
            f"\n\t\t\t\tHidden size    : {hidden_size}"
            f"\n\t\t\t\tNum layers     : {num_layers}"
            f"\n\t\t\t\tBidirectional  : {bidirectional}"
            f"\n\t\t\t\tDropout prob   : {dropout}"
        )

        # Initialize test helper for processing each video seperately
        # It prints the result to the loaded checkpoint directory
        helper = VideoAnomalyDetectionResultHelper(
            dataset=dataset,
            model=net,
            R=st_dict["R"],
            boundary=boundary,
            device=device,
            end_to_end_training=True if train_type == "train_end_to_end" else False,
            debug=debug,
            output_file=net_checkpoint.parent / "shanghaitech_test_results.txt",
        )
        # TEST
        helper.test_video_anomaly_detection()
        print("Test finished")
    #
    #


if __name__ == "__main__":
    main()
