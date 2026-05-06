import random
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def budget_from_c(num_classes: int, c_multiplier: int) -> int:
    return int(num_classes * c_multiplier)


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.0
    pred = logits[mask].argmax(dim=-1)
    return float((pred == labels[mask]).float().mean().item())


@torch.no_grad()
def masked_fidelity(surrogate_logits: torch.Tensor, victim_logits: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.0
    s_pred = surrogate_logits[mask].argmax(dim=-1)
    v_pred = victim_logits[mask].argmax(dim=-1)
    return float((s_pred == v_pred).float().mean().item())


def gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def as_serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out
