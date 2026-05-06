import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_embed: bool = False):
        embed = self.conv1(x, edge_index)
        h = F.relu(embed)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)
        if return_embed:
            return out, embed
        return out


class DGI(nn.Module):
    """Deep Graph Infomax wrapper using the first GCN layer as representation."""

    def __init__(self, encoder: GCN, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        nn.init.xavier_uniform_(self.discriminator.weight)

    @staticmethod
    def readout(h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(h.mean(dim=0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        _, h_pos = self.encoder(x, edge_index, return_embed=True)
        perm = torch.randperm(x.size(0), device=x.device)
        _, h_neg = self.encoder(x[perm], edge_index, return_embed=True)
        summary = self.readout(h_pos).expand_as(h_pos)
        pos_loss = self.loss_fn(
            self.discriminator(h_pos, summary),
            torch.ones(h_pos.size(0), 1, device=x.device),
        )
        neg_loss = self.loss_fn(
            self.discriminator(h_neg, summary),
            torch.zeros(h_pos.size(0), 1, device=x.device),
        )
        return pos_loss + neg_loss
