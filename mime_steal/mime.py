import time
from typing import Any, Dict, Optional

import torch

from .config import MIMEConfig
from .data import build_visible_graph, load_dataset, make_split
from .models import GCN
from .query import batch_split, entropy_scores, exploration_ratio, kcenter_select
from .train import evaluate, get_embeddings, pretrain_dgi, query_oracle, train_surrogate, train_victim
from .utils import budget_from_c, gpu_memory_mb, set_seed


@torch.no_grad()
def _select_next_batch(
    surrogate: GCN,
    embeddings: torch.Tensor,
    visible_x: torch.Tensor,
    visible_edge_index: torch.Tensor,
    queried_mask: torch.Tensor,
    budget: int,
    config: MIMEConfig,
) -> torch.Tensor:
    num_queried = int(queried_mask.sum().item())
    remaining = int(budget - num_queried)
    q_t = min(config.batch_size, remaining)
    if q_t <= 0:
        return torch.empty(0, dtype=torch.long, device=visible_x.device)

    epsilon_t = exploration_ratio(num_queried, budget, config.eps_min, config.eps_max)
    r_t, s_t = batch_split(q_t, epsilon_t)
    all_nodes = torch.arange(visible_x.size(0), device=visible_x.device)
    unqueried = all_nodes[~queried_mask]
    anchors = all_nodes[queried_mask]

    r_t = min(r_t, int(unqueried.numel()))
    R_t = kcenter_select(unqueried, r_t, embeddings, anchors=anchors)

    candidate_mask = ~queried_mask.clone()
    if R_t.numel() > 0:
        candidate_mask[R_t] = False
    candidates = all_nodes[candidate_mask]

    if s_t <= 0 or candidates.numel() == 0:
        return R_t.unique()

    surrogate.eval()
    logits = surrogate(visible_x, visible_edge_index)
    scores = entropy_scores(logits)
    pool_size = min(int(candidates.numel()), max(1, config.pool_factor * s_t))
    pool = candidates[torch.topk(scores[candidates], k=pool_size).indices]
    anchor_for_div = torch.cat([anchors, R_t]).unique() if R_t.numel() > 0 else anchors
    S_t = kcenter_select(pool, min(s_t, int(pool.numel())), embeddings, anchors=anchor_for_div)
    return torch.cat([R_t, S_t]).unique()


def run_mime(
    dataset: str,
    c: int,
    seed: int = 42,
    config: Optional[MIMEConfig] = None,
) -> Dict[str, Any]:
    """Run standard MIME on one dataset, one C-budget, and one seed.

    Parameters
    ----------
    dataset:
        Dataset name, e.g., "CoCS", "CoraFull", "Arxiv", "Products",
        "Amazon-ratings", or "Squirrel".
    c:
        Query budget multiplier. The total budget is c * number_of_classes.
    seed:
        Random seed.
    config:
        Optional MIMEConfig object.
    """
    config = config or MIMEConfig()
    device = config.resolved_device()
    set_seed(seed, deterministic=config.deterministic)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t_start = time.time()
    data, alias = load_dataset(dataset, root=config.root, products_max_nodes=config.products_max_nodes)
    data = make_split(data, seed=seed, train_ratio=config.train_ratio, prior_ratio=config.prior_ratio)
    data = data.to(device)
    visible = build_visible_graph(data, device)
    budget = min(budget_from_c(data.num_classes, c), int(visible.x.size(0)))

    t_victim = time.time()
    victim = train_victim(data, config, device)
    victim_time = time.time() - t_victim

    surrogate = GCN(data.num_node_features, config.hidden_dim, data.num_classes, config.dropout).to(device)

    t_pre = time.time()
    surrogate = pretrain_dgi(surrogate, visible, config)
    embeddings = get_embeddings(surrogate, visible)
    pretrain_time = time.time() - t_pre

    queried_mask = torch.zeros(visible.x.size(0), dtype=torch.bool, device=device)
    queried_y = torch.full((visible.x.size(0),), -1, dtype=torch.long, device=device)

    query_time = 0.0
    train_time = 0.0
    rounds = 0

    # Phase I: label-free k-center cold start.
    init_k = min(config.initial_batch_size, budget)
    if init_k > 0:
        all_nodes = torch.arange(visible.x.size(0), device=device)
        init_idx = kcenter_select(all_nodes, init_k, embeddings, anchors=None)
        t_query = time.time()
        labels = query_oracle(victim, data, visible.global_nodes[init_idx], device)
        query_time += time.time() - t_query
        queried_y[init_idx] = labels
        queried_mask[init_idx] = True

        t_train = time.time()
        surrogate = train_surrogate(surrogate, visible, queried_mask, queried_y, config, config.epochs_per_round)
        train_time += time.time() - t_train
        rounds += 1

    # Phase II and III: entropy-diversity selection followed by surrogate refinement.
    while int(queried_mask.sum().item()) < budget:
        batch_idx = _select_next_batch(
            surrogate,
            embeddings,
            visible.x,
            visible.edge_index,
            queried_mask,
            budget,
            config,
        )
        if batch_idx.numel() == 0:
            break
        # Avoid accidental duplicates after unique operations.
        batch_idx = batch_idx[~queried_mask[batch_idx]]
        if batch_idx.numel() == 0:
            break

        t_query = time.time()
        labels = query_oracle(victim, data, visible.global_nodes[batch_idx], device)
        query_time += time.time() - t_query
        queried_y[batch_idx] = labels
        queried_mask[batch_idx] = True

        t_train = time.time()
        surrogate = train_surrogate(surrogate, visible, queried_mask, queried_y, config, config.epochs_per_round)
        train_time += time.time() - t_train
        rounds += 1

    t_final = time.time()
    surrogate = train_surrogate(surrogate, visible, queried_mask, queried_y, config, config.final_epochs)
    final_time = time.time() - t_final
    train_time += final_time

    metrics = evaluate(surrogate, victim, data, visible, queried_mask, config, device)
    total_time = time.time() - t_start

    return {
        "dataset": alias,
        "seed": int(seed),
        "c": int(c),
        "budget": int(budget),
        "num_classes": int(data.num_classes),
        "num_nodes": int(data.num_nodes),
        "visible_nodes": int(visible.x.size(0)),
        "queried_nodes": int(queried_mask.sum().item()),
        "eval_scope": config.eval_scope,
        "eval_nodes": int(metrics["eval_nodes"]),
        "accuracy": float(metrics["accuracy"]),
        "fidelity": float(metrics["fidelity"]),
        "victim_time": float(victim_time),
        "pretrain_time": float(pretrain_time),
        "query_time": float(query_time),
        "train_time": float(train_time),
        "final_time": float(final_time),
        "total_time": float(total_time),
        "rounds": int(rounds),
        "memory_mb": float(gpu_memory_mb()),
    }
