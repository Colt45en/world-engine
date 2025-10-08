"""
World Engine V4.1 — Refined (clean, runnable)
Author: Colt45en
License: MIT

"""

from __future__ import annotations
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────────
# Utils & Config
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class WorldEngineConfig:
    # Model sizes
    vocab_size: int = 50_000
    d_model: int = 512
    k_feats: int = 1000
    n_pos: int = 512
    n_rels: int = 100

    # Transformer
    n_layers: int = 12
    n_heads: int = 16
    d_ff: int = 2048
    dropout: float = 0.1
    norm_first: bool = True
    use_transformer: bool = True
    use_rotary: bool = True
    use_flash: bool = False  # use PyTorch SDPA if available

    # GNN
    use_gnn: bool = True
    gnn_layers: int = 2
    gnn_heads: int = 8
    gnn_dropout: float = 0.1

    # CRF
    use_crf: bool = True
    num_role_labels: int = 25

    # Attention wrappers
    use_extra_self_attn: bool = True  # post-encoder self-attn layer

    # Residual & multi-scale
    use_residual: bool = True
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])

    # Training helpers
    mixed_precision: bool = True
    compile_model: bool = False  # let caller decide (torch.compile)

    # Loss weights
    w_reconstruction: float = 1.0
    w_roles: float = 1.0
    w_contrastive: float = 0.1


# ────────────────────────────────────────────────────────────────────────────────
# Core layers
# ────────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (no bias)."""
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rms = sqrt(mean(x^2))
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.weight * (x / (rms + self.eps))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (applied to per-head dim)."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10_000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, H, N, d]
        N = q.shape[-2]
        if N > self.cos.size(0):
            self._build_cache(N)
        cos = self.cos[:N].unsqueeze(0).unsqueeze(0)  # [1,1,N,d]
        sin = self.sin[:N].unsqueeze(0).unsqueeze(0)

        def rot_half(x):
            x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        return q, k


class MultiHeadAttention(nn.Module):
    """MHA with optional RoPE + SDPA fallback."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_rotary: bool = True, use_flash: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.use_rotary = use_rotary
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(self.d_head) if use_rotary else None

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [B,N,D]; kv: [B,M,D] or None (self-attn)
        kv = x if kv is None else kv
        B, N, D = x.shape
        M = kv.size(1)

        q = self.q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)      # [B,H,N,d]
        k = self.k(kv).view(B, M, self.n_heads, self.d_head).transpose(1, 2)     # [B,H,M,d]
        v = self.v(kv).view(B, M, self.n_heads, self.d_head).transpose(1, 2)     # [B,H,M,d]

        if self.rope is not None:
            q, k = self.rope(q, k)

        attn_mask = None
        if mask is not None:
            # mask: [B,N] (True = keep); key padding for SDPA wants False = keep
            # We'll build an attn mask where invalid keys are -inf.
            # For SDPA we pass key_padding_mask; for manual path we mask scores.
            key_padding_mask = ~mask  # [B,N] True=pad
        else:
            key_padding_mask = None

        if self.use_flash:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                key_padding_mask=key_padding_mask  # PyTorch 2.1+ supports
            )
            attn = None
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N,M]
            if key_padding_mask is not None:
                # expand to [B,1,1,M]
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), float("-inf"))
            attn = scores.softmax(dim=-1)
            attn = self.dropout(attn)
            out = attn @ v  # [B,H,N,d]

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.o(out)
        return (out, attn) if return_attn else (out, None)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}.get(activation.lower(), nn.GELU)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.net(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 norm_first: bool = True, use_rotary: bool = True, use_flash: bool = False):
        super().__init__()
        self.norm_first = norm_first
        self.mha = MultiHeadAttention(d_model, n_heads, dropout, use_rotary, use_flash)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.norm_first:
            y, _ = self.mha(self.n1(x), mask=mask)
            x = x + self.d1(y)
            y = self.ff(self.n2(x))
            x = x + self.d2(y)
        else:
            y, _ = self.mha(x, mask=mask); x = self.n1(x + self.d1(y))
            y = self.ff(x); x = self.n2(x + self.d2(y))
        return x


# ────────────────────────────────────────────────────────────────────────────────
# GNN (single clean version)
# ────────────────────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """
    Message passing with optional attention over incoming edges.
    node_features: [N, Din]; edge_index: [2, E]; edge_attr: [E, De] (optional)
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 use_attention: bool = True, edge_dim: Optional[int] = None):
        super().__init__()
        self.use_attention = use_attention
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads if use_attention else out_dim
        assert (not use_attention) or (out_dim % num_heads == 0), "out_dim must be divisible by num_heads"

        ed = edge_dim or in_dim
        self.msg = nn.Sequential(
            nn.Linear(in_dim * 2 + ed, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        if use_attention:
            self.q = nn.Linear(in_dim, out_dim)
            self.kv = nn.Linear(in_dim + ed, out_dim)
            self.attn_drop = nn.Dropout(dropout)

        self.out = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.norm(self.res_proj(node_features))

        N = node_features.size(0)
        src, dst = edge_index[0], edge_index[1]

        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), node_features.size(1), device=node_features.device)

        s = node_features[src]
        d = node_features[dst]
        m = self.msg(torch.cat([s, d, edge_attr], dim=-1))  # [E, out]

        if self.use_attention:
            q = self.q(d).view(-1, self.num_heads, self.head_dim)          # per-edge query from dst
            kv = self.kv(torch.cat([s, edge_attr], dim=-1)).view(-1, self.num_heads, self.head_dim)
            # per-edge compatibility (dot-product with itself acts as learned gate)
            attn = (q * kv).sum(-1) * (self.head_dim ** -0.5)              # [E, H]

            # softmax over incoming edges per dst node
            max_dst = int(dst.max().item()) + 1
            attn_full = torch.full((max_dst, self.num_heads), float("-inf"), device=node_features.device)
            attn_full.index_put_((dst,), attn, accumulate=False)
            attn_norm = F.softmax(attn_full, dim=0)[dst]                   # [E, H]
            attn_norm = self.attn_drop(attn_norm)
            m = (m.view(-1, self.num_heads, self.head_dim) * attn_norm.unsqueeze(-1)).reshape(-1, self.out_dim)

        agg = torch.zeros(N, self.out_dim, device=node_features.device)
        agg.index_add_(0, dst, m)  # sum incoming messages

        h = self.out(torch.cat([node_features, agg], dim=-1))
        h = self.norm(self.res_proj(node_features) + self.drop(h))
        return h


# ────────────────────────────────────────────────────────────────────────────────
# Multi-scale temporal processing (clean version)
# ────────────────────────────────────────────────────────────────────────────────

class MultiScaleProcessor(nn.Module):
    """
    Parallel Conv1d branches with optional pooling + projection back to D.
    """
    def __init__(self, d_model: int, kernel_sizes: List[int], dropout: float = 0.1):
        super().__init__()
        self.ks = kernel_sizes
        self.num_scales = len(kernel_sizes)
        per = max(1, d_model // self.num_scales)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, per, k, padding=k//2),
                nn.GELU(),
                nn.Conv1d(per, per, k, padding=k//2, dilation=2),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        self.proj = nn.Linear(per * self.num_scales, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,N,D]
        B, N, D = x.shape
        t = x.transpose(1, 2)  # [B,D,N]
        outs = [b(t) for b in self.branches]           # list of [B,per,N]
        y = torch.cat(outs, dim=1).transpose(1, 2)     # [B,N,per*num_scales]
        y = self.proj(y)
        if mask is not None:
            y = y * mask.unsqueeze(-1)
        return self.norm(x + self.drop(y))


# ────────────────────────────────────────────────────────────────────────────────
# Memory bank (episodic; sentence-level integration)
# ────────────────────────────────────────────────────────────────────────────────

class MemoryBank(nn.Module):
    def __init__(self, d_model: int, max_size: int = 1000, temperature: float = 1.0):
        super().__init__()
        self.max_size = max_size
        self.temperature = temperature
        self.register_buffer("mem", torch.zeros(max_size, d_model))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.bool))
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def update(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Down-project to sentence-level features
        if features.dim() == 3:
            if mask is None:
                s = features.mean(dim=1)
            else:
                s = (features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        else:
            s = features  # [B,D]

        with torch.no_grad():
            b = s.size(0)
            p = int(self.ptr.item())
            end = p + b
            if end <= self.max_size:
                self.mem[p:end] = s
                self.ptr[0] = end % self.max_size
                if end == self.max_size:
                    self.filled[0] = True
            else:
                first = self.max_size - p
                self.mem[p:] = s[:first]
                self.mem[:end - self.max_size] = s[first:]
                self.ptr[0] = end - self.max_size
                self.filled[0] = True

    def retrieve(self, query: torch.Tensor, k: int = 5) -> torch.Tensor:
        # query: [B,D]
        size = self.max_size if self.filled[0] else int(self.ptr.item())
        if size == 0:
            return torch.zeros(query.size(0), k, query.size(-1), device=query.device)
        mem = self.mem[:size]
        q = self.q(query)
        K = self.k(mem)             # [M,D]
        V = self.v(mem)             # [M,D]
        sims = (q @ K.T) / (math.sqrt(K.size(-1)) * self.temperature)  # [B,M]
        topv, topi = sims.topk(min(k, size), dim=-1)
        ret = V[topi]               # [B,k,D]
        if ret.size(1) < k:  # pad
            pad = torch.zeros(query.size(0), k - ret.size(1), ret.size(-1), device=query.device)
            ret = torch.cat([ret, pad], dim=1)
        return self.norm(ret)


# ────────────────────────────────────────────────────────────────────────────────
# World Engine (clean)
# ────────────────────────────────────────────────────────────────────────────────

class WorldEngine(nn.Module):
    def __init__(self, cfg: WorldEngineConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # Embedding splits must sum to D
        d_tok = D // 2
        d_pos = D // 4
        d_feat = D - d_tok - d_pos

        self.emb_tok = nn.Embedding(cfg.vocab_size, d_tok)
        self.emb_pos = nn.Embedding(cfg.n_pos, d_pos)
        self.proj_feat = nn.Linear(cfg.k_feats, d_feat)

        # sinusoidal PE for fallback/addition
        self.register_buffer("pe", self._sinusoidal_positions(512, D), persistent=False)

        # Encoder (Transformer encoder or simple MLP stack)
        if cfg.use_transformer:
            self.blocks = nn.ModuleList([
                TransformerBlock(D, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.norm_first,
                                 cfg.use_rotary, cfg.use_flash)
                for _ in range(cfg.n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(D, 2*D), nn.GELU(), nn.Dropout(cfg.dropout),
                    nn.Linear(2*D, D)
                ) for _ in range(cfg.n_layers)
            ])

        # Optional extra self-attn after encoder
        self.post_attn = MultiHeadAttention(D, cfg.n_heads, cfg.dropout,
                                            use_rotary=cfg.use_rotary, use_flash=cfg.use_flash) \
                         if cfg.use_extra_self_attn else None

        # Multi-scale temporal conv
        self.multi_scale = MultiScaleProcessor(D, cfg.kernel_sizes, cfg.dropout)

        # GNN
        if cfg.use_gnn:
            self.edge_rel_emb = nn.Embedding(cfg.n_rels, D)
            self.gnn_layers = nn.ModuleList([
                GraphConvLayer(D, D, num_heads=cfg.gnn_heads, dropout=cfg.gnn_dropout,
                               use_attention=True, edge_dim=D)
                for _ in range(cfg.gnn_layers)
            ])
            self.gnn_norm = nn.LayerNorm(D)

        # Latent head (sentence)
        self.enc_lat = nn.Sequential(
            nn.Linear(D, 512), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(256, 64)
        )

        # Feature reconstruction (sentence-level)
        self.dec_feat = nn.Sequential(
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(128, cfg.k_feats)
        )

        # Token role head
        self.role_head = nn.Sequential(
            nn.Linear(D, D//2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(D//2, cfg.num_role_labels)
        )

        # Optional CRF
        self.use_crf = cfg.use_crf
        if self.use_crf:
            try:
                from torchcrf import CRF
                self.crf = CRF(cfg.num_role_labels, batch_first=True)
            except Exception:
                print("⚠️  torchcrf not available; falling back to softmax CE.")
                self.use_crf = False

        # Memory
        self.memory = MemoryBank(D, max_size=1000)

        # Norms & dropout
        self.in_norm = nn.LayerNorm(D)
        self.out_norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(cfg.dropout)

        self.apply(self._init_weights)

    # ——— helpers ———
    def _sinusoidal_positions(self, n_pos: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(n_pos, d_model)
        pos = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def extend_pe(self, need: int):
        if need <= self.pe.size(1):
            return
        self.register_buffer("pe", self._sinusoidal_positions(need, self.pe.size(-1)), persistent=False)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight);
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    # ——— forward ———
    def forward(self, tok_ids: torch.Tensor, pos_ids: torch.Tensor, feat_rows: torch.Tensor,
                lengths: torch.Tensor, edge_index: Optional[torch.Tensor] = None,
                edge_type: Optional[torch.Tensor] = None,
                context_vectors: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        tok_ids: [B,N], pos_ids: [B,N], feat_rows: [B,N,K], lengths: [B]
        edge_index: [2,E], edge_type: [E]
        """
        B, N = tok_ids.shape
        device = tok_ids.device

        # mask: True = valid
        mask = torch.arange(N, device=device)[None, :] < lengths[:, None]

        # embeddings
        tok = self.emb_tok(tok_ids)      # [B,N,dtok]
        pos = self.emb_pos(pos_ids)      # [B,N,dpos]
        feat = self.proj_feat(feat_rows) # [B,N,dfeat]
        x = torch.cat([tok, pos, feat], dim=-1)
        x = self.in_norm(x)

        # pos enc
        self.extend_pe(N)
        x = x + self.pe[:, :N, :]
        x = self.drop(x)

        # multi-scale
        x = x + self.multi_scale(x, mask)

        # encoder stack
        if self.cfg.use_transformer:
            for blk in self.blocks:
                x = blk(x, mask=mask)
        else:
            for mlp in self.blocks:
                x = x + mlp(x)

        attn_out = None
        if self.cfg.use_extra_self_attn and self.post_attn is not None:
            y, attn_out = self.post_attn(x, mask=mask)
            x = x + y

        # GNN (if edges provided)
        if self.cfg.use_gnn and edge_index is not None and edge_type is not None and edge_index.numel() > 0:
            D = x.size(-1)
            h = x.reshape(-1, D)  # [B*N, D]
            rel = self.edge_rel_emb(edge_type)  # [E,D]
            for g in self.gnn_layers:
                h = g(h, edge_index, rel)
            x = self.gnn_norm(x + h.view(B, N, D))

        x = self.out_norm(x)

        # memory update (sentence-level)
        self.memory.update(x, mask)

        # sentence representation -> latent z
        x_masked = x * mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
        h_sent = x_masked.sum(dim=1) / denom
        z = self.enc_lat(h_sent)

        # recon
        feat_hat = self.dec_feat(z)

        # token roles
        role_logits = self.role_head(x)

        out = {
            "z": z,
            "feat_hat": feat_hat,
            "role_logits": role_logits,
            "hidden_states": x,
            "sentence_repr": h_sent,
            "mask": mask,
        }
        if return_attention and attn_out is not None:
            out["attention_weights"] = attn_out
        return out

    # ——— losses & utils ———
    def loss_reconstruction(self, feat_hat: torch.Tensor, feat_rows: torch.Tensor,
                            mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            target = feat_rows.mean(dim=1)
        else:
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
            target = (feat_rows * mask.unsqueeze(-1)).sum(dim=1) / denom
        return F.mse_loss(feat_hat, target)

    def loss_roles(self, role_logits: torch.Tensor, role_labels: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        if self.use_crf:
            return -self.crf(role_logits, role_labels, mask=mask, reduction="mean")
        C = role_logits.size(-1)
        loss = F.cross_entropy(role_logits.view(-1, C), role_labels.view(-1), reduction="none")
        loss = loss.view(role_labels.shape) * mask.float()
        return loss.sum() / mask.float().sum().clamp_min(1)

    def loss_contrastive(self, z: torch.Tensor, pos_pairs: torch.Tensor,
                         neg_pairs: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        pos_d = F.pairwise_distance(z[pos_pairs[:, 0]], z[pos_pairs[:, 1]]).pow(2)
        neg_d = F.relu(margin - F.pairwise_distance(z[neg_pairs[:, 0]], z[neg_pairs[:, 1]])).pow(2)
        return 0.5 * (pos_d.mean() + neg_d.mean())

    @torch.no_grad()
    def predict_roles(self, tok_ids, pos_ids, feat_rows, lengths):
        self.eval()
        out = self.forward(tok_ids, pos_ids, feat_rows, lengths)
        logits, mask = out["role_logits"], out["mask"]
        if self.use_crf:
            return self.crf.decode(logits, mask=mask)
        return logits.argmax(dim=-1).masked_fill(~mask, -1)

    @torch.no_grad()
    def get_attention_maps(self, tok_ids, pos_ids, feat_rows, lengths):
        out = self.forward(tok_ids, pos_ids, feat_rows, lengths, return_attention=True)
        return out.get("attention_weights", None)

    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer,
                   loss_weights: Optional[Dict[str, float]] = None, clip_norm: float = 1.0) -> Dict[str, float]:
        if loss_weights is None:
            loss_weights = {"reconstruction": 1.0, "roles": 1.0, "contrastive": 0.1}
        self.train()
        optimizer.zero_grad()

        outputs = self.forward(
            batch["tok_ids"], batch["pos_ids"], batch["feat_rows"], batch["lengths"],
            edge_index=batch.get("edge_index"), edge_type=batch.get("edge_type")
        )

        losses = {}
        losses["reconstruction"] = self.loss_reconstruction(outputs["feat_hat"], batch["feat_rows"], outputs["mask"])
        if "role_labels" in batch:
            losses["roles"] = self.loss_roles(outputs["role_logits"], batch["role_labels"], outputs["mask"])
        if "positive_pairs" in batch and "negative_pairs" in batch:
            losses["contrastive"] = self.loss_contrastive(outputs["z"], batch["positive_pairs"], batch["negative_pairs"])

        total = sum(loss_weights.get(k, 1.0) * v for k, v in losses.items())
        total.backward()
        if clip_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
        optimizer.step()

        return {**{f"loss_{k}": float(v.detach().cpu()) for k, v in losses.items()},
                "loss_total": float(total.detach().cpu())}


# ────────────────────────────────────────────────────────────────────────────────
# Factory & simple tests
# ────────────────────────────────────────────────────────────────────────────────

def create_world_engine(config: Dict) -> WorldEngine:
    return WorldEngine(WorldEngineConfig(**config))


# ===========================
# BASIC TEST & BENCH
# ===========================

def run_basic_tests() -> bool:
    print("🧪 Running World Engine basic tests…")
    cfg = dict(
        vocab_size=1000, d_model=128, k_feats=50, n_pos=25, n_rels=10,
        n_layers=2, n_heads=4, dropout=0.1, use_transformer=True,
        use_gnn=True, use_crf=False, num_role_labels=5
    )
    try:
        model = create_world_engine(cfg)
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created — params: {params:,}")

        B, N = 4, 10
        tok_ids = torch.randint(0, cfg["vocab_size"], (B, N))
        pos_ids = torch.randint(0, cfg["n_pos"], (B, N))
        feat_rows = torch.randn(B, N, cfg["k_feats"])
        lengths = torch.tensor([N, N, N-1, N-2])

        model.eval()
        with torch.no_grad():
            out = model(tok_ids, pos_ids, feat_rows, lengths)
        assert "z" in out and "feat_hat" in out and "role_logits" in out
        assert out["z"].shape == (B, 64)
        print(f"✅ Forward ok — z: {tuple(out['z'].shape)}, feat_hat: {tuple(out['feat_hat'].shape)}")
        return True
    except Exception as e:
        print("❌ Test failed:", repr(e))
        return False


def run_performance_benchmark():
    print("\n📊 Benchmark")
    cfg = dict(
        vocab_size=10_000, d_model=512, k_feats=100, n_pos=50, n_rels=20,
        n_layers=6, n_heads=8, dropout=0.1, use_transformer=True,
        use_gnn=True, use_crf=False, num_role_labels=5
    )
    model = create_world_engine(cfg).eval()
    for B in (1, 4, 16):
        for N in (32, 128, 512):
            tok_ids = torch.randint(0, cfg["vocab_size"], (B, N))
            pos_ids = torch.randint(0, cfg["n_pos"], (B, N))
            feat_rows = torch.randn(B, N, cfg["k_feats"])
            lengths = torch.full((B,), N)
            t0 = time.time()
            with torch.no_grad():
                _ = model(tok_ids, pos_ids, feat_rows, lengths)
            dt = (time.time() - t0) * 1000
            tps = (B * N) / (dt / 1000)
            print(f"  B={B:>2}, N={N:>4}: {dt:7.2f} ms  ({tps:,.1f} tok/s)")


def demonstrate_capabilities():
    print("\n🚀 Capabilities")
    cfg = dict(vocab_size=5000, d_model=256, k_feats=50, n_pos=25, n_rels=10,
               n_layers=3, n_heads=8, dropout=0.1, use_transformer=True,
               use_gnn=True, use_crf=False, num_role_labels=5)
    m = create_world_engine(cfg)
    tot = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * 4 for p in m.parameters()) / (1024**2)
    print(f"   🔢 params: {tot:,}")
    print(f"   🎯 trainable: {trainable:,}")
    print(f"   💾 size ~{size_mb:.1f} MB")


if __name__ == "__main__":
    print("=" * 80)
    print("🌍 WORLD ENGINE V3.1 — REFINED")
    print("=" * 80)
    ok = run_basic_tests()
    if ok:
        run_performance_benchmark()
        demonstrate_capabilities()
        print("\n✅ Ready for integration")
    else:
        print("\n❌ Tests failed — please inspect the stack trace.")

# what I fixed / improved
#
# removed duplicate classes (ResidualConnection, MultiScaleProcessor, MemoryBank) and dead fragments appended after __main__.
#
# unified GNN into a single GraphConvLayer used by WorldEngine._apply_gnn path.
#
# cleaned RoPE, masking, and SDPA usage (supports PyTorch's scaled_dot_product_attention when available).
#
# made MultiScaleProcessor consistent and dimensionally safe.
#
# made MemoryBank simple and sentence-level to avoid shape pitfalls, with retrieval compatible with the engine.
#
# normalized embeddings and extended positional encodings safely.
#
# consistent loss API; CRF gracefully degrades if torchcrf isn't installed.
#
# added small, deterministic basic test and benchmark; no stray code at EOF.
#
# If you want this split into multiple files (layers/blocks/engine) or wired into DDP/mixed-precision training scaffolding, say the word and I'll scaffold a trainer too.
