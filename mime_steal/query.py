import math
from typing import Optional

import torch


def normalize_embeddings(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return z / z.norm(p=2, dim=1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def kcenter_select(
    candidates: torch.Tensor,
    k: int,
    embeddings: torch.Tensor,
    anchors: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Farthest-first k-center selection in normalized representation space."""
    if k <= 0 or candidates.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=embeddings.device)
    candidates = candidates.unique()
    if candidates.numel() <= k:
        return candidates

    z = normalize_embeddings(embeddings)
    selected = []

    if anchors is not None and anchors.numel() > 0:
        anchors = anchors.unique()
        min_dist = torch.cdist(z[candidates], z[anchors]).min(dim=1).values
    else:
        center = z[candidates].mean(dim=0, keepdim=True)
        min_dist = torch.cdist(z[candidates], center).view(-1)

    for _ in range(k):
        local_pos = int(torch.argmax(min_dist).item())
        new_node = candidates[local_pos]
        selected.append(new_node)
        new_dist = torch.cdist(z[candidates], z[new_node].view(1, -1)).view(-1)
        min_dist = torch.minimum(min_dist, new_dist)
        min_dist[local_pos] = -1.0

    return torch.stack(selected).long()


@torch.no_grad()
def entropy_scores(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


def exploration_ratio(num_queried: int, budget: int, eps_min: float, eps_max: float) -> float:
    if budget <= 0:
        return eps_min
    raw = 1.0 - float(num_queried) / float(budget)
    return float(max(eps_min, min(eps_max, raw)))


def batch_split(q_t: int, epsilon_t: float) -> tuple[int, int]:
    r_t = int(math.floor(epsilon_t * q_t))
    s_t = int(q_t - r_t)
    return r_t, s_t
