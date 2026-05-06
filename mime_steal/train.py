import copy
from typing import Dict

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from .config import MIMEConfig
from .data import VisibleGraph
from .models import DGI, GCN
from .utils import masked_accuracy, masked_fidelity


def train_victim(data, config: MIMEConfig, device: str) -> GCN:
    model = GCN(
        data.num_node_features,
        config.victim_hidden_dim,
        data.num_classes,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.victim_lr, weight_decay=0.0)
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)

    for _ in range(config.victim_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    return model


def pretrain_dgi(model: GCN, visible: VisibleGraph, config: MIMEConfig) -> GCN:
    dgi = DGI(model, config.hidden_dim).to(visible.x.device)
    optimizer = torch.optim.Adam(dgi.parameters(), lr=config.dgi_lr, weight_decay=config.weight_decay)
    for _ in range(config.dgi_epochs):
        dgi.train()
        optimizer.zero_grad()
        loss = dgi(visible.x, visible.edge_index)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def get_embeddings(model: GCN, visible: VisibleGraph) -> torch.Tensor:
    model.eval()
    _, embeddings = model(visible.x, visible.edge_index, return_embed=True)
    return embeddings.detach()


def laplacian_penalty(logits: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    probs = logits.softmax(dim=-1)
    src, dst = edge_index
    return (probs[src] - probs[dst]).pow(2).sum(dim=1).mean()


def train_surrogate(
    model: GCN,
    visible: VisibleGraph,
    queried_mask: torch.Tensor,
    queried_y: torch.Tensor,
    config: MIMEConfig,
    epochs: int,
) -> GCN:
    idx = queried_mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0 or epochs <= 0:
        return model
    optimizer = torch.optim.Adam(model.parameters(), lr=config.surrogate_lr, weight_decay=config.weight_decay)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        if config.edge_drop_p > 0 and visible.edge_index.numel() > 0:
            edge_train, _ = dropout_edge(visible.edge_index, p=config.edge_drop_p, training=True)
        else:
            edge_train = visible.edge_index
        logits = model(visible.x, edge_train)
        ce = F.cross_entropy(logits[idx], queried_y[idx], label_smoothing=config.label_smoothing)
        lap = laplacian_penalty(logits, edge_train)
        loss = ce + config.lap_lambda * lap
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def query_oracle(victim: GCN, data, global_nodes: torch.Tensor, device: str) -> torch.Tensor:
    victim.eval()
    logits = victim(data.x.to(device), data.edge_index.to(device))
    return logits.argmax(dim=-1)[global_nodes]


@torch.no_grad()
def evaluate(
    surrogate: GCN,
    victim: GCN,
    data,
    visible: VisibleGraph,
    queried_mask: torch.Tensor,
    config: MIMEConfig,
    device: str,
) -> Dict[str, float]:
    surrogate.eval()
    victim.eval()

    if config.eval_scope == "visible":
        s_logits = surrogate(visible.x, visible.edge_index)
        v_full = victim(data.x.to(device), data.edge_index.to(device))
        v_logits = v_full[visible.global_nodes]
        labels = data.y.to(device)[visible.global_nodes]
        eval_mask = ~queried_mask
        return {
            "accuracy": masked_accuracy(s_logits, labels, eval_mask),
            "fidelity": masked_fidelity(s_logits, v_logits, eval_mask),
            "eval_nodes": int(eval_mask.sum().item()),
        }

    if config.eval_scope == "full":
        # Optional compatibility mode for protocols that evaluate the learned weights on the full graph.
        full_surrogate = GCN(data.num_node_features, config.hidden_dim, data.num_classes, config.dropout).to(device)
        full_surrogate.load_state_dict(copy.deepcopy(surrogate.state_dict()))
        s_logits = full_surrogate(data.x.to(device), data.edge_index.to(device))
        v_logits = victim(data.x.to(device), data.edge_index.to(device))
        test_mask = data.test_mask.to(device)
        return {
            "accuracy": masked_accuracy(s_logits, data.y.to(device), test_mask),
            "fidelity": masked_fidelity(s_logits, v_logits, test_mask),
            "eval_nodes": int(test_mask.sum().item()),
        }

    raise ValueError("config.eval_scope must be either 'visible' or 'full'.")
