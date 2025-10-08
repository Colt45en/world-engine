from __future__ import annotations

"""Tier-4 combined utilities + encoders for typed dependency graphs.

This module exposes:

- ``build_batched_edges``: robust, device-safe global edge builder
- ``RelGraphEncoderTG``:   RGCN-based encoder with residual + LayerNorm
- ``RelGATv2EncoderTG``:  GATv2-based encoder with typed edge attrs
- Helpful mask/flatten utilities and a quick self-test (when run directly)

The implementation gracefully handles optional dependencies and is safe to drop
straight into Tier-4 IDE or training pipelines.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, RGCNConv
except Exception as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required. Install it via:\n"
        "  pip install torch-geometric\n"
        "and the matching PyTorch scatter/sparse wheels for your platform."
    ) from e

try:  # Optional coalescing if torch_sparse is present (pragma: no cover)
    from torch_sparse import coalesce as _coalesce
except Exception:  # pragma: no cover
    _coalesce = None


@dataclass
class GraphBatch:
    """Container for a flattened batched graph."""

    edge_index: torch.Tensor  # Long[2, E]
    edge_type: torch.Tensor  # Long[E]
    num_nodes: int
    offsets: Optional[torch.Tensor] = None  # Long[B], prefix sums of lengths


def mask_from_lengths(
    lengths: Sequence[int],
    max_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a boolean mask ``[B, max_len]`` where True marks valid tokens."""

    lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device)
    batch = lengths_t.numel()
    max_tokens = (
        int(max_len)
        if max_len is not None
        else int(lengths_t.max().item()) if batch > 0 else 0
    )
    if max_tokens == 0:
        return torch.zeros((batch, 0), dtype=torch.bool, device=device)
    arange = torch.arange(max_tokens, device=device).unsqueeze(0).expand(batch, max_tokens)
    return arange < lengths_t.unsqueeze(1)


def flatten_padded(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten ``[B, N, D]`` -> ``[B*N, D]`` and ``[B, N]`` -> ``[B*N]`` while
    preserving zeros on pads.
    """

    batch, steps = mask.shape
    x_flat = x.reshape(batch * steps, *x.shape[2:])
    mask_flat = mask.reshape(batch * steps)
    return x_flat, mask_flat


def _maybe_coalesce(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove duplicates and sort if ``torch_sparse.coalesce`` is available."""

    if _coalesce is None:
        return edge_index, edge_type

    src, dst = edge_index
    triple = torch.stack([src, dst, edge_type], dim=0)
    values = torch.ones(triple.size(1), device=triple.device, dtype=torch.float32)
    triple_out, _ = _coalesce(triple, values, m=num_nodes, n=num_nodes, op="sum")
    src_out, dst_out, type_out = triple_out[0], triple_out[1], triple_out[2]
    return torch.stack([src_out, dst_out], dim=0), type_out


def build_batched_edges(
    batch_edges: List[Tuple[torch.Tensor, torch.Tensor]],
    lengths: Sequence[int],
    *,
    add_self_loops: bool = False,
    undirected: bool = False,
    device: Optional[torch.device] = None,
    coalesce: bool = True,
) -> GraphBatch:
    """Combine per-sentence local dependency edges into a single global graph."""

    if len(batch_edges) != len(lengths):
        raise ValueError(
            f"batch_edges size ({len(batch_edges)}) != lengths size ({len(lengths)})"
        )

    if device is None and batch_edges:
        device = batch_edges[0][0].device

    lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device)
    batch = lengths_t.numel()
    num_nodes = int(lengths_t.sum().item())

    offsets = torch.zeros(batch, dtype=torch.long, device=device)
    if batch > 1:
        offsets[1:] = torch.cumsum(lengths_t[:-1], dim=0)

    all_src: List[torch.Tensor] = []
    all_dst: List[torch.Tensor] = []
    all_typ: List[torch.Tensor] = []

    for idx, (edge_index_b, edge_type_b) in enumerate(batch_edges):
        if edge_index_b.numel() == 0:
            continue
        off = offsets[idx]
        if edge_index_b.dtype != torch.long:
            edge_index_b = edge_index_b.long()
        if edge_type_b.dtype != torch.long:
            edge_type_b = edge_type_b.long()
        if edge_index_b.device != device:
            edge_index_b = edge_index_b.to(device)
        if edge_type_b.device != device:
            edge_type_b = edge_type_b.to(device)

        edge_index_global = edge_index_b + off
        all_src.append(edge_index_global[0])
        all_dst.append(edge_index_global[1])
        all_typ.append(edge_type_b)

        if undirected:
            all_src.append(edge_index_global[1])
            all_dst.append(edge_index_global[0])
            all_typ.append(edge_type_b)

    if add_self_loops and num_nodes > 0:
        nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
        all_src.append(nodes)
        all_dst.append(nodes)
        all_typ.append(torch.zeros(num_nodes, device=device, dtype=torch.long))

    if not all_src:
        empty_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_type = torch.zeros((0,), dtype=torch.long, device=device)
        return GraphBatch(empty_index, empty_type, num_nodes, offsets)

    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
    edge_type = torch.cat(all_typ)

    if coalesce:
        edge_index, edge_type = _maybe_coalesce(edge_index, edge_type, num_nodes)

    if undirected and coalesce and add_self_loops:
        pass

    return GraphBatch(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes, offsets=offsets)


class RelGraphEncoderTG(nn.Module):
    """Typed dependency encoder using PyG's ``RGCNConv``."""

    def __init__(
        self,
        d_model: int,
        n_rels: int,
        *,
        num_layers: int = 2,
        num_bases: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        bases = num_bases or min(32, n_rels)

        for _ in range(num_layers):
            self.layers.append(
                RGCNConv(d_model, d_model, num_relations=n_rels, num_bases=bases)
            )
            self.norms.append(nn.LayerNorm(d_model))

    def forward(
        self,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        valid_mask_flat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_mask_flat is not None:
            x_flat = x_flat.masked_fill(~valid_mask_flat.unsqueeze(-1), 0.0)

        hidden = x_flat
        for conv, norm in zip(self.layers, self.norms):
            message = conv(hidden, edge_index, edge_type)
            message = self.drop(F.gelu(message))
            hidden = norm(hidden + message)

        if valid_mask_flat is not None:
            hidden = hidden.masked_fill(~valid_mask_flat.unsqueeze(-1), 0.0)
        return hidden


class RelGATv2EncoderTG(nn.Module):
    """Typed dependency encoder built on PyG's ``GATv2Conv``."""

    def __init__(
        self,
        d_model: int,
        n_rels: int,
        *,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        rel_dim: int = 64,
    ) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(n_rels, rel_dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    in_channels=d_model,
                    out_channels=d_model,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=rel_dim,
                    concat=False,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))

    def forward(
        self,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        valid_mask_flat: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if valid_mask_flat is not None:
            x_flat = x_flat.masked_fill(~valid_mask_flat.unsqueeze(-1), 0.0)

        hidden = x_flat
        edge_attr = self.rel_emb(edge_type)
        attn_info: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        for idx, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            if return_attention and idx == len(self.layers) - 1:
                message, (edge_idx_out, alpha) = conv(
                    hidden,
                    edge_index,
                    edge_attr,
                    return_attention_weights=True,
                )
                attn_info = (edge_idx_out, alpha)
            else:
                message = conv(hidden, edge_index, edge_attr)
            message = self.drop(F.gelu(message))
            hidden = norm(hidden + message)

        if valid_mask_flat is not None:
            hidden = hidden.masked_fill(~valid_mask_flat.unsqueeze(-1), 0.0)

        if return_attention:
            return hidden, attn_info
        return hidden


if __name__ == "__main__":  # pragma: no cover
    torch.manual_seed(7)

    lengths = [4, 3]
    edges0 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    types0 = torch.tensor([1, 2, 2], dtype=torch.long)
    edges1 = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    types1 = torch.tensor([0, 1], dtype=torch.long)

    graph = build_batched_edges(
        [(edges0, types0), (edges1, types1)],
        lengths,
        undirected=True,
        add_self_loops=True,
        coalesce=True,
    )
    print("edge_index:", graph.edge_index.shape)
    print("edge_type:", graph.edge_type.shape)
    print("num_nodes:", graph.num_nodes)

    batch, max_len, hidden_dim = 2, max(lengths), 16
    node_states = torch.randn(batch, max_len, hidden_dim)
    mask = mask_from_lengths(lengths, device=node_states.device)
    node_states_flat, mask_flat = flatten_padded(node_states, mask)

    rgcn = RelGraphEncoderTG(d_model=hidden_dim, n_rels=5, num_layers=2)
    out_rgcn = rgcn(node_states_flat, graph.edge_index, graph.edge_type, mask_flat)
    print("RGCN out:", out_rgcn.shape)

    gat = RelGATv2EncoderTG(d_model=hidden_dim, n_rels=5, num_layers=2, heads=2, rel_dim=8)
    out_gat, attn = gat(
        node_states_flat,
        graph.edge_index,
        graph.edge_type,
        mask_flat,
        return_attention=True,
    )
    print("GAT out:", out_gat.shape)
    if attn is not None:
        edge_idx_out, alpha = attn
        print("attention index shape:", edge_idx_out.shape)
        print("attention weights shape:", alpha.shape)
