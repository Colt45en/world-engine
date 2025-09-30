import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RelGraphEncoderTG(nn.Module):
    """
    Typed dependency graph encoder using torch-geometric's RGCNConv.
    Works on a flattened [B*N, d] node matrix with a global batched graph.
    - edge_index: Long[2, E] over flattened nodes
    - edge_type:  Long[E]    with values in [0, n_rels-1]
    """

    def __init__(self, d_model: int, n_rels: int, num_layers: int = 2, num_bases: int = None,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        num_bases = num_bases or min(32, n_rels)

        for _ in range(num_layers):
            self.layers.append(RGCNConv(d_model, d_model, num_relations=n_rels, num_bases=num_bases))
            self.norms.append(nn.LayerNorm(d_model))

    def forward(self, x_flat, edge_index, edge_type, valid_mask_flat=None):
        """
        x_flat:          [B*N, d]
        edge_index:      [2, E]
        edge_type:       [E]
        valid_mask_flat: [B*N] boolean; True for real tokens, False for pads
        """
        if valid_mask_flat is not None:
            x_flat = x_flat.masked_fill(~valid_mask_flat.unsqueeze(-1), 0.0)

        h = x_flat
        for conv, ln in zip(self.layers, self.norms):
            m = conv(h, edge_index, edge_type)
            m = self.drop(F.gelu(m))
            h = ln(h + m)

        if valid_mask_flat is not None:
            h = h.masked_fill(~valid_mask_flat.unsqueeze(-1), 0.0)
        return h
