from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MIMEConfig:
    """Configuration for a standard MIME run."""

    # Data and protocol
    root: str = "./data"
    prior_ratio: float = 0.10
    train_ratio: float = 0.60
    products_max_nodes: int = 200_000
    eval_scope: str = "visible"  # "visible" or "full"

    # Device and reproducibility
    device: Optional[str] = None
    deterministic: bool = True

    # Model
    hidden_dim: int = 128
    dropout: float = 0.50
    victim_hidden_dim: int = 16
    victim_epochs: int = 200
    victim_lr: float = 1e-2
    surrogate_lr: float = 1e-3
    weight_decay: float = 5e-4

    # DGI bootstrapping
    dgi_epochs: int = 200
    dgi_lr: float = 1e-3

    # Query schedule
    initial_batch_size: int = 5
    batch_size: int = 5
    eps_min: float = 0.20
    eps_max: float = 0.50
    pool_factor: int = 10

    # Surrogate refinement
    epochs_per_round: int = 100
    final_epochs: int = 200
    lap_lambda: float = 5e-4
    edge_drop_p: float = 0.40
    label_smoothing: float = 0.05

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"
