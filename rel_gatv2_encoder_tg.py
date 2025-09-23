import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class RelGATv2EncoderTG(nn.Module):
    """
    Typed dependency encoder with GATv2.
    - x_flat:     [B*N, d]
    - edge_index: [2, E]  (global indices over the flattened batch)
    - edge_type:  [E]     (ints in [0, n_rels-1])
    We embed edge types -> edge_attr and pass to GATv2Conv via edge_dim.
    """
    def __init__(self, d_model: int, n_rels: int, num_layers: int = 2,
                 heads: int = 4, dropout: float = 0.1, rel_dim: int = 64):
        super().__init__()
        self.rel_emb = nn.Embedding(n_rels, rel_dim)
        self.layers = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.drop   = nn.Dropout(dropout)

        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    in_channels=d_model,
                    out_channels=d_model,      # concat=False -> out dim = d_model
                    heads=heads,
                    dropout=dropout,
                    edge_dim=rel_dim,
                    concat=False               # keep residual-friendly dims
                )
            )
            self.norms.append(nn.LayerNorm(d_model))

    def forward(self, x_flat, edge_index, edge_type, valid_mask_flat=None):
        if valid_mask_flat is not None:
            x_flat = x_flat.masked_fill(~valid_mask_flat.unsqueeze(-1), 0.0)

        h = x_flat
        eattr = self.rel_emb(edge_type)  # [E, rel_dim]

        for conv, ln in zip(self.layers, self.norms):
            m = conv(h, edge_index, eattr)   # [B*N, d]
            m = self.drop(F.gelu(m))
            h = ln(h + m)

        if valid_mask_flat is not None:
            h = h.masked_fill(~valid_mask_flat.unsqueeze(-1), 0.0)
        return h