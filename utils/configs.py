from dataclasses import dataclass
from pathlib import Path


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
    model_ckp: Path | None
    # Optimizer config
    optimizer: str
    ae_learning_rate: float
    learning_rate: float
    ae_weight_decay: float
    weight_decay: float
    ae_lr_milestones: list[int]
    lr_milestones: list[int]
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
    idx_list_enc: list[int]
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
    idx_list_enc: tuple[int, ...]
    nu: float
    fp16: bool
    compile: bool
    dist: str
    optimizer: str = "adam"
    lr_milestones: tuple[int, ...] = tuple()
    end_to_end_training: bool = True
    debug: bool = False
    warm_up_n_epochs: int = 0
    epochs: int = 0
    log_frequency: int = 1
