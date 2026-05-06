from dataclasses import dataclass
from typing import Tuple

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, CoraFull, HeterophilousGraphDataset, WikipediaNetwork
from torch_geometric.utils import subgraph, to_undirected

try:
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:  # pragma: no cover
    PygNodePropPredDataset = None


@dataclass
class VisibleGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    global_nodes: torch.Tensor


def _patch_torch_load_for_pyg():
    """Return original torch.load after installing a compatibility wrapper."""
    original_load = torch.load

    def safe_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = safe_load
    return original_load


def sanitize_labels(data: Data) -> Data:
    y = data.y
    if y.dim() > 1 and y.size(1) == 1:
        y = y.squeeze(1)
    y = y.long()
    values = torch.unique(y)
    values, _ = torch.sort(values)
    y_new = y.clone()
    for new_label, old_label in enumerate(values.tolist()):
        y_new[y == old_label] = new_label
    data.y = y_new
    data.num_classes = int(values.numel())
    return data


def load_dataset(name: str, root: str = "./data", products_max_nodes: int = 200_000) -> Tuple[Data, str]:
    """Load one supported node-classification dataset.

    Products uses the front-200K induced subgraph by OGB node order.
    """
    name_l = name.lower().replace("_", "-")
    original_load = _patch_torch_load_for_pyg()
    try:
        if "product" in name_l:
            if PygNodePropPredDataset is None:
                raise ImportError("Please install ogb to use ogbn-products.")
            ds = PygNodePropPredDataset(name="ogbn-products", root=root, transform=T.ToUndirected())
            data = ds[0]
            if data.num_nodes > products_max_nodes:
                node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                node_mask[:products_max_nodes] = True
                data = data.subgraph(node_mask)
                data.num_nodes = products_max_nodes
            alias = "Products"
        elif "arxiv" in name_l or "arvix" in name_l:
            if PygNodePropPredDataset is None:
                raise ImportError("Please install ogb to use ogbn-arxiv.")
            ds = PygNodePropPredDataset(name="ogbn-arxiv", root=root, transform=T.ToUndirected())
            data = ds[0]
            alias = "Arxiv"
        elif "cora" in name_l and "full" in name_l:
            data = CoraFull(root)[0]
            alias = "CoraFull"
        elif "amazon-ratings" in name_l or "amazonratings" in name_l:
            data = HeterophilousGraphDataset(root=root, name="Amazon-ratings")[0]
            alias = "Amazon-ratings"
        elif "squirrel" in name_l:
            data = WikipediaNetwork(root=root, name="squirrel", geom_gcn_preprocess=False, transform=T.ToUndirected())[0]
            alias = "Squirrel"
        elif "cocs" in name_l or ("coauthor" in name_l and "cs" in name_l):
            data = Coauthor(root, "CS")[0]
            alias = "CoCS"
        elif "amazon" in name_l or "amzc" in name_l:
            data = Amazon(root, "Computers")[0]
            alias = "AmazonC"
        else:
            raise ValueError(f"Unsupported dataset: {name}")
    finally:
        torch.load = original_load

    data = sanitize_labels(data)
    if getattr(data, "edge_index", None) is not None and data.edge_index.numel() > 0:
        data.edge_index = to_undirected(data.edge_index)
    return data, alias


def make_split(data: Data, seed: int, train_ratio: float = 0.60, prior_ratio: float = 0.10) -> Data:
    """Create victim train / test split and a visible prior subset sampled from train nodes."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes, generator=generator)
    train_size = int(train_ratio * num_nodes)
    visible_size = min(max(1, int(prior_ratio * num_nodes)), train_size)

    train_idx = perm[:train_size]
    test_idx = perm[train_size:]
    visible_idx = train_idx[torch.randperm(train_size, generator=generator)[:visible_size]]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_idx] = True
    data.visible_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.visible_mask[visible_idx] = True
    return data


def build_visible_graph(data: Data, device: str) -> VisibleGraph:
    global_nodes = data.visible_mask.nonzero(as_tuple=False).view(-1).to(device)
    edge_index, _ = subgraph(global_nodes, data.edge_index.to(device), relabel_nodes=True)
    edge_index = to_undirected(edge_index)
    return VisibleGraph(x=data.x.to(device)[global_nodes], edge_index=edge_index, global_nodes=global_nodes)
