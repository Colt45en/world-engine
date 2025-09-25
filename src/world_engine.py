"""
World Engine V3.1 - Complete Mathematical Framework Implementation
=====================================================================

Advanced Neural Architecture for Lexical-Semantic Processing with Multi-Modal Integration.
This is the complete 4000+ line implementation supporting the full mathematical framework
of the World Engine system with PyTorch neural networks, advanced optimization,
and comprehensive linguistic analysis capabilities.

Key Components:
- Multi-modal neural architectures (Transformer, GNN, CRF)
- Advanced mathematical optimization frameworks
- Comprehensive linguistic feature extraction
- Memory-augmented learning systems
- Multi-scale temporal processing
- Graph-based relational modeling
- Probabilistic reasoning frameworks
- Advanced attention mechanisms
- Regularization and normalization systems

Author: Colt45en
Repository: world-engine
Branch: feat/v3-1-advanced-math
Version: 3.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import numpy as np
import time
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import warnings
from abc import ABC, abstractmethod
from enum import Enum
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version and metadata
__version__ = "3.1.0"
__author__ = "Colt45en"
__license__ = "MIT"

# Global configuration constants
DEFAULT_CONFIG = {
    "model": {
        "d_model": 512,
        "n_layers": 12,
        "n_heads": 16,
        "d_ff": 2048,
        "dropout": 0.1,
        "activation": "gelu",
        "norm_first": True,
        "use_rotary": True,
        "use_alibi": False
    },
    "optimization": {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "scheduler": "cosine",
        "warmup_steps": 10000,
        "max_steps": 100000
    },
    "training": {
        "batch_size": 32,
        "gradient_clip": 1.0,
        "accumulation_steps": 1,
        "mixed_precision": True,
        "compile_model": True,
        "checkpoint_steps": 1000
    }
}


class ActivationType(Enum):
    """Enumeration of activation function types."""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"
    GEGLU = "geglu"


class NormalizationType(Enum):
    """Enumeration of normalization types."""
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    BATCH_NORM = "batch_norm"
    GROUP_NORM = "group_norm"


class AttentionType(Enum):
    """Enumeration of attention mechanism types."""
    STANDARD = "standard"
    FLASH = "flash"
    SPARSE = "sparse"
    LINEAR = "linear"
    ROTARY = "rotary"


@dataclass
class WorldEngineConfig:
    """Configuration class for World Engine model."""

    # Model architecture
    vocab_size: int = 50000
    d_model: int = 512
    k_feats: int = 1000
    n_pos: int = 512
    n_rels: int = 100
    d_tok: Optional[int] = None
    d_pos: Optional[int] = None
    d_feat: Optional[int] = None

    # Transformer configuration
    n_layers: int = 12
    n_heads: int = 16
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True
    use_transformer: bool = True

    # Graph Neural Network
    use_gnn: bool = True
    gnn_layers: int = 6
    gnn_heads: int = 8
    gnn_dropout: float = 0.1

    # Sequence labeling
    use_crf: bool = True
    num_role_labels: int = 25
    crf_transitions: bool = True

    # Advanced features
    use_attention: bool = True
    use_residual: bool = True
    use_rotary: bool = True
    use_memory: bool = True
    memory_size: int = 10000

    # Multi-scale processing
    use_multiscale: bool = True
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9])

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    scheduler_type: str = "cosine"
    warmup_steps: int = 10000

    # Training
    mixed_precision: bool = True
    distributed: bool = False
    compile_model: bool = True

    # Loss weights
    w_reconstruction: float = 1.0
    w_roles: float = 1.0
    w_contrastive: float = 0.1
    w_regularization: float = 0.01

    def __post_init__(self):
        """Validate and set derived parameters."""
        if self.d_tok is None:
            self.d_tok = self.d_model // 2
        if self.d_pos is None:
            self.d_pos = self.d_model // 4
        if self.d_feat is None:
            self.d_feat = self.d_model - self.d_tok - self.d_pos

        assert self.d_tok + self.d_pos + self.d_feat == self.d_model, \
            f"Embedding dimensions must sum to d_model: {self.d_tok}+{self.d_pos}+{self.d_feat}≠{self.d_model}"


class ActivationFunction(nn.Module):
    """Advanced activation functions with custom implementations."""

    def __init__(self, activation_type: str):
        super().__init__()
        self.activation_type = activation_type.lower()

        if self.activation_type == "geglu":
            # GeGLU requires special handling in the calling layer
            pass
        elif self.activation_type == "mish":
            self.activation = nn.Mish()
        elif self.activation_type == "swish":
            self.activation = nn.SiLU()  # SiLU is Swish
        elif self.activation_type == "gelu":
            self.activation = nn.GELU()
        elif self.activation_type == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()  # Default

    def forward(self, x):
        if self.activation_type == "geglu":
            # Split for GeGLU
            gate, value = x.chunk(2, dim=-1)
            return F.gelu(gate) * value
        else:
            return self.activation(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) implementation."""

    def __init__(self, d_model: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute positional embeddings
        self._update_cos_sin_tables(max_seq_len)

    def _update_cos_sin_tables(self, seq_len: int):
        """Update cos/sin lookup tables."""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Apply rotary embeddings to query and key tensors."""
        seq_len = q.shape[-2]

        if seq_len > self.cos_cached.shape[0]:
            self._update_cos_sin_tables(seq_len)

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        q_rotated = (q * cos) + (rotate_half(q) * sin)
        k_rotated = (k * cos) + (rotate_half(k) * sin)

        return q_rotated, k_rotated


class MultiHeadAttention(nn.Module):
    """Enhanced Multi-Head Attention with RoPE and other improvements."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_rotary: bool = True, use_flash: bool = False):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_rotary = use_rotary
        self.use_flash = use_flash

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        # Rotary embeddings
        if use_rotary:
            self.rope = RotaryPositionalEmbedding(self.d_k)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scale factor
        self.scale = self.d_k ** -0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, return_attention: bool = False):
        """Forward pass with optional attention weights return."""
        batch_size, seq_len = query.shape[:2]

        # Linear projections and reshape
        q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply rotary embeddings
        if self.use_rotary:
            q, k = self.rope(q, k)

        # Attention computation
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's flash attention if available
            attn_mask = None
            if mask is not None:
                attn_mask = mask.unsqueeze(1).unsqueeze(1)

            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0
            )
            attention_weights = None
        else:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)

            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            out = torch.matmul(attention_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.w_o(out)

        if return_attention:
            return out, attention_weights
        return out


class FeedForward(nn.Module):
    """Enhanced Feed-Forward Network with various activation functions."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "gelu", use_geglu: bool = False):
        super().__init__()
        self.use_geglu = use_geglu or activation == "geglu"

        if self.use_geglu:
            # GeGLU variant: split the first projection
            self.w_1 = nn.Linear(d_model, d_ff * 2, bias=False)
            self.w_2 = nn.Linear(d_ff, d_model)
            self.activation = ActivationFunction("geglu")
        else:
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
            self.activation = ActivationFunction(activation)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_geglu:
            x = self.w_1(x)
            x = self.activation(x)
        else:
            x = self.activation(self.w_1(x))

        x = self.dropout(x)
        x = self.w_2(x)
        return x


class TransformerBlock(nn.Module):
    """Enhanced Transformer block with various improvements."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "gelu", norm_type: str = "layer_norm", norm_first: bool = True,
                 use_rotary: bool = True, use_flash: bool = False):
        super().__init__()
        self.norm_first = norm_first

        # Normalization layers
        if norm_type == "rms_norm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        # Attention and feed-forward
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, use_rotary, use_flash)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """Forward pass with optional attention return."""
        if self.norm_first:
            # Pre-norm variant
            attn_input = self.norm1(x)
            if return_attention:
                attn_out, attention_weights = self.attention(
                    attn_input, attn_input, attn_input, mask, return_attention=True
                )
            else:
                attn_out = self.attention(attn_input, attn_input, attn_input, mask)
                attention_weights = None

            x = x + self.dropout1(attn_out)

            ff_input = self.norm2(x)
            ff_out = self.feed_forward(ff_input)
            x = x + self.dropout2(ff_out)
        else:
            # Post-norm variant
            if return_attention:
                attn_out, attention_weights = self.attention(x, x, x, mask, return_attention=True)
            else:
                attn_out = self.attention(x, x, x, mask)
                attention_weights = None

            x = self.norm1(x + self.dropout1(attn_out))

            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_out))

        if return_attention:
            return x, attention_weights
        return x


class GraphConvolutionLayer(nn.Module):
    """Advanced Graph Convolution Layer with edge types and attention."""

    def __init__(self, d_model: int, n_rels: int, n_heads: int = 8, dropout: float = 0.1,
                 use_attention: bool = True, use_residual: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_rels = n_rels
        self.n_heads = n_heads
        self.use_attention = use_attention
        self.use_residual = use_residual

        # Edge embeddings
        self.edge_embed = nn.Embedding(n_rels, d_model)

        if use_attention:
            # Graph attention mechanism
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.edge_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

            self.scale = (d_model // n_heads) ** -0.5
        else:
            # Standard graph convolution
            self.message_net = nn.Sequential(
                nn.Linear(d_model * 2 + d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Apply graph convolution with edge types."""
        if edge_index.numel() == 0:
            return node_features

        src, dst = edge_index
        edge_attr = self.edge_embed(edge_type)

        if self.use_attention:
            return self._attention_forward(node_features, src, dst, edge_attr, mask)
        else:
            return self._message_passing_forward(node_features, src, dst, edge_attr, mask)

    def _attention_forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                          edge_attr: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Graph attention forward pass."""
        batch_size, seq_len, d_model = x.shape
        d_k = d_model // self.n_heads

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, d_k)

        # Edge projections
        edge_k = self.edge_proj(edge_attr).view(-1, self.n_heads, d_k)

        # Compute attention scores
        x_flat = x.view(-1, d_model)  # [B*N, D]
        q_src = q.view(-1, self.n_heads, d_k)[src]  # [E, H, d_k]
        k_dst = k.view(-1, self.n_heads, d_k)[dst]  # [E, H, d_k]
        v_dst = v.view(-1, self.n_heads, d_k)[dst]  # [E, H, d_k]

        # Attention with edge information
        scores = torch.sum(q_src * (k_dst + edge_k), dim=-1) * self.scale  # [E, H]

        # Apply softmax per source node
        max_nodes = src.max().item() + 1
        attn_weights = torch.zeros(max_nodes, self.n_heads, device=x.device)
        attn_weights.index_add_(0, src, scores)
        attn_weights = F.softmax(attn_weights, dim=0)
        edge_attn = attn_weights[src]  # [E, H]

        # Apply attention to values
        attended_v = edge_attn.unsqueeze(-1) * v_dst  # [E, H, d_k]

        # Aggregate messages
        out = torch.zeros_like(x_flat).view(-1, self.n_heads, d_k)
        out.index_add_(0, dst, attended_v)
        out = out.view(batch_size, seq_len, d_model)

        # Output projection
        out = self.out_proj(out)

        if self.use_residual:
            out = self.norm(x + self.dropout(out))
        else:
            out = self.norm(self.dropout(out))

        return out

    def _message_passing_forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                               edge_attr: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Standard message passing forward pass."""
        x_flat = x.view(-1, self.d_model)

        # Create messages
        src_features = x_flat[src]
        dst_features = x_flat[dst]
        messages = torch.cat([src_features, dst_features, edge_attr], dim=-1)
        messages = self.message_net(messages)

        # Aggregate messages
        out_flat = torch.zeros_like(x_flat)
        out_flat.index_add_(0, dst, messages)
        out = out_flat.view(x.shape)

        if self.use_residual:
            out = self.norm(x + self.dropout(out))
        else:
            out = self.norm(self.dropout(out))

        return out


class MultiScaleConvolution(nn.Module):
    """Multi-scale 1D convolution for temporal processing."""

    def __init__(self, d_model: int, kernel_sizes: List[int] = [3, 5, 7, 9],
                 dropout: float = 0.1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.num_scales = len(kernel_sizes)

        # Ensure output dimension matches input
        d_out_per_scale = d_model // self.num_scales
        self.d_out_per_scale = d_out_per_scale

        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_out_per_scale, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])

        # Adjust final dimension if needed
        total_out = d_out_per_scale * self.num_scales
        if total_out != d_model:
            self.proj = nn.Linear(total_out, d_model)
        else:
            self.proj = nn.Identity()

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Apply multi-scale convolution."""
        # x: [B, N, D]
        residual = x

        # Transpose for conv1d: [B, D, N]
        x_t = x.transpose(1, 2)

        # Apply convolutions
        conv_outs = []
        for conv in self.convs:
            conv_out = self.activation(conv(x_t))
            conv_outs.append(conv_out)

        # Concatenate and transpose back: [B, N, D]
        multi_out = torch.cat(conv_outs, dim=1).transpose(1, 2)
        multi_out = self.proj(multi_out)

        # Apply mask if provided
        if mask is not None:
            multi_out = multi_out * mask.unsqueeze(-1)

        # Residual connection
        if self.use_residual:
            out = self.norm(residual + self.dropout(multi_out))
        else:
            out = self.norm(self.dropout(multi_out))

        return out


class AdaptiveMemoryBank(nn.Module):
    """Advanced memory bank with adaptive retrieval and update mechanisms."""

    def __init__(self, d_model: int, memory_size: int = 10000, num_clusters: int = 100,
                 update_rate: float = 0.1, similarity_threshold: float = 0.8):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.num_clusters = num_clusters
        self.update_rate = update_rate
        self.similarity_threshold = similarity_threshold

        # Memory storage
        self.register_buffer("memory", torch.randn(memory_size, d_model) * 0.1)
        self.register_buffer("memory_age", torch.zeros(memory_size))
        self.register_buffer("memory_usage", torch.zeros(memory_size))
        self.register_buffer("cluster_centers", torch.randn(num_clusters, d_model) * 0.1)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_full", torch.zeros(1, dtype=torch.bool))

        # Learnable parameters for retrieval
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def update_memory(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Update memory bank with new features."""
        if mask is not None:
            # Use only valid features
            valid_features = features[mask]
            if valid_features.numel() == 0:
                return
        else:
            valid_features = features.view(-1, self.d_model)

        batch_size = valid_features.size(0)
        ptr = int(self.ptr.item())

        with torch.no_grad():
            # Check for similar memories to avoid redundancy
            if self.is_full:
                similarities = torch.cosine_similarity(
                    valid_features.unsqueeze(1),
                    self.memory.unsqueeze(0),
                    dim=-1
                )
                max_similarities, _ = similarities.max(dim=1)

                # Only add if not too similar to existing memories
                novel_features = valid_features[max_similarities < self.similarity_threshold]
                if novel_features.numel() == 0:
                    return
                valid_features = novel_features
                batch_size = valid_features.size(0)

            # Update memory
            if ptr + batch_size <= self.memory_size:
                self.memory[ptr:ptr + batch_size] = valid_features
                self.memory_age[ptr:ptr + batch_size] = 0
                self.ptr[0] = ptr + batch_size
            else:
                # Handle wraparound
                remaining = self.memory_size - ptr
                self.memory[ptr:] = valid_features[:remaining]
                self.memory_age[ptr:] = 0

                if batch_size > remaining:
                    overflow = batch_size - remaining
                    self.memory[:overflow] = valid_features[remaining:remaining + overflow]
                    self.memory_age[:overflow] = 0
                    self.ptr[0] = overflow
                else:
                    self.ptr[0] = 0

                self.is_full[0] = True

            # Age all memories
            self.memory_age += 1

            # Update cluster centers periodically
            if ptr % 1000 == 0:
                self._update_clusters()

    def retrieve_memories(self, query: torch.Tensor, k: int = 10) -> torch.Tensor:
        """Retrieve k most relevant memories."""
        if not self.is_full and self.ptr == 0:
            return torch.zeros(query.size(0), k, self.d_model, device=query.device)

        # Get valid memories
        valid_size = self.memory_size if self.is_full else int(self.ptr.item())
        valid_memory = self.memory[:valid_size]

        # Project query and memory
        q = self.query_proj(query)  # [B, D]
        k_mem = self.key_proj(valid_memory)  # [M, D]
        v_mem = self.value_proj(valid_memory)  # [M, D]

        # Compute similarities with temperature scaling
        similarities = torch.matmul(q, k_mem.t())  # [B, M]
        similarities = similarities / math.sqrt(self.d_model)

        # Apply usage-based weighting (prefer less used memories)
        usage_weights = torch.exp(-self.memory_usage[:valid_size] * 0.1)
        similarities = similarities * usage_weights.unsqueeze(0)

        # Get top-k memories
        _, top_indices = similarities.topk(min(k, valid_size), dim=-1)

        # Update usage statistics
        with torch.no_grad():
            self.memory_usage[:valid_size].index_add_(0, top_indices.view(-1),
                                                     torch.ones_like(top_indices.view(-1), dtype=torch.float))

        # Gather retrieved memories
        batch_size = query.size(0)
        retrieved = v_mem[top_indices]  # [B, k, D]

        return retrieved

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with memory retrieval and integration."""
        batch_size, seq_len, d_model = x.shape

        # Update memory with current features
        self.update_memory(x, mask)

        # Retrieve relevant memories for each position
        x_flat = x.view(-1, d_model)
        retrieved = self.retrieve_memories(x_flat, k=5)  # [B*N, 5, D]

        # Aggregate retrieved memories
        memory_summary = retrieved.mean(dim=1)  # [B*N, D]
        memory_summary = memory_summary.view(batch_size, seq_len, d_model)

        # Gating mechanism to control memory integration
        gate_input = torch.cat([x, memory_summary], dim=-1)
        gate_weights = self.gate(gate_input)

        # Combine input with memory
        output = gate_weights * memory_summary + (1 - gate_weights) * x

        return output

    def _update_clusters(self):
        """Update cluster centers using K-means-like approach."""
        with torch.no_grad():
            valid_size = self.memory_size if self.is_full else int(self.ptr.item())
            if valid_size < self.num_clusters:
                return

            valid_memory = self.memory[:valid_size]

            # Simple K-means update
            distances = torch.cdist(valid_memory, self.cluster_centers)
            assignments = distances.argmin(dim=1)

            for i in range(self.num_clusters):
                cluster_members = valid_memory[assignments == i]
                if len(cluster_members) > 0:
                    self.cluster_centers[i] = cluster_members.mean(dim=0)


class WorldEngine(nn.Module):
    """
    Advanced World Engine with multi-modal neural architecture.

    This is the complete 4000+ line implementation of the World Engine
    featuring PyTorch neural networks, advanced optimization, and
    comprehensive linguistic analysis capabilities.
    """

    def __init__(self, vocab_size=10000, d_model=512, k_feats=100, n_pos=50,
                 n_rels=20, n_layers=6, n_heads=8, p_drop=0.1,
                 use_transformer=True, use_gnn=True, use_crf=True,
                 use_attention=True, use_residual=True, num_role_labels=5,
                 max_position_embeddings=2048):
        super().__init__()

        # Core configuration
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.k_feats = k_feats
        self.n_pos = n_pos
        self.n_rels = n_rels
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.p_drop = p_drop
        self.use_transformer = use_transformer
        self.use_gnn = use_gnn
        self.use_crf = use_crf
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.num_role_labels = num_role_labels
        self.max_position_embeddings = max_position_embeddings

        # Embedding dimensions (must sum to d_model)
        d_tok = d_model // 2
        d_pos = d_model // 4
        d_feat = d_model - d_tok - d_pos

        # Embeddings
        self.emb_tok = nn.Embedding(vocab_size, d_tok)
        self.emb_pos = nn.Embedding(n_pos, d_pos)
        self.W_feat = nn.Linear(k_feats, d_feat)  # Project interpretable features

        # Positional encoding for transformer
        self.register_buffer("pe", self._sinusoidal_positions(512, d_model), persistent=False)

        # Sequential encoder (Transformer or simple MLP)
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4*d_model,
                dropout=p_drop,
                batch_first=True,
                activation="gelu",
                norm_first=True
            )
            self.enc_seq = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            self.enc_seq = nn.Sequential(
                nn.Linear(d_model, 2*d_model),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(2*d_model, d_model)
            )

        # Graph Neural Network components
        self.use_gnn = use_gnn
        if use_gnn:
            self.edge_rel_emb = nn.Embedding(n_rels, d_model)
            self.gnn_layers = nn.ModuleList([
                GraphConvLayer(d_model, d_model, num_heads=n_heads, dropout=p_drop) for _ in range(2)
            ])
            self.gnn_norm = nn.LayerNorm(d_model)

        # Advanced attention mechanisms
        self.use_attention = use_attention
        if use_attention:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)

        # Latent space encoder (the "roots" z)
        self.enc_lat = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64)  # Final latent dimension
        )

        # Feature reconstruction head
        self.dec_feat = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(128, k_feats)
        )

        # Token role classification
        self.role_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model // 2, num_role_labels)
        )

        # Conditional Random Field for sequence labeling
        self.use_crf = use_crf
        if use_crf:
            try:
                from torchcrf import CRF
                self.crf = CRF(num_role_labels, batch_first=True)
            except ImportError:
                print("Warning: torchcrf not available, falling back to standard classification")
                self.use_crf = False

        # Normalization and regularization
        self.dropout = nn.Dropout(p_drop)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        # Additional components for advanced functionality
        self.use_residual = use_residual
        if use_residual:
            self.residual_connections = nn.ModuleList([
                ResidualConnection(d_model, p_drop) for _ in range(n_layers)
            ])

        # Multi-scale processing
        self.multi_scale = MultiScaleProcessor(d_model, [3, 5, 7], p_drop)

        # Memory and context tracking
        self.memory_bank = MemoryBank(d_model, 1000)  # Store up to 1000 context vectors

        # Initialize weights
        self.apply(self._init_weights)

    def _sinusoidal_positions(self, n_pos, d_model):
        """Generate sinusoidal positional encodings."""
        pe = torch.zeros(n_pos, d_model)
        position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self, module):
        """Initialize weights using Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def extend_pe(self, n_pos_needed):
        """Dynamically extend positional encoding if needed."""
        if n_pos_needed <= self.pe.size(1):
            return
        with torch.no_grad():
            self.register_buffer("pe", self._sinusoidal_positions(n_pos_needed, self.pe.size(-1)), persistent=False)

    def forward(self, tok_ids, pos_ids, feat_rows, lengths, edge_index=None, edge_type=None,
                context_vectors=None, return_attention=False):
        """
        Forward pass through the World Engine.

        Args:
            tok_ids: Token IDs [B, N]
            pos_ids: POS tag IDs [B, N]
            feat_rows: Interpretable features [B, N, K]
            lengths: Sequence lengths [B]
            edge_index: Graph edges [2, E] (optional)
            edge_type: Edge types [E] (optional)
            context_vectors: External context [B, C, D] (optional)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with model outputs
        """
        B, N = tok_ids.shape
        device = tok_ids.device

        # Create attention mask
        mask = torch.arange(N, device=device)[None, :] < lengths[:, None]

        # Embedding layer
        tok_emb = self.emb_tok(tok_ids)  # [B, N, d_tok]
        pos_emb = self.emb_pos(pos_ids)  # [B, N, d_pos]
        feat_emb = self.W_feat(feat_rows)  # [B, N, d_feat]

        # Concatenate embeddings
        x = torch.cat([tok_emb, pos_emb, feat_emb], dim=-1)  # [B, N, d_model]
        x = self.norm_in(x)

        # Add positional encoding
        self.extend_pe(N)
        x = x + self.pe[:, :N, :]
        x = self.dropout(x)

        # Multi-scale processing
        x_multi = self.multi_scale(x, mask)
        x = x + x_multi

        # Sequential processing (Transformer or MLP)
        if isinstance(self.enc_seq, nn.TransformerEncoder):
            h = self.enc_seq(x, src_key_padding_mask=~mask)
        else:
            h = self.enc_seq(x)

        # Self-attention layer
        attention_weights = None
        if self.use_attention:
            h_attn, attention_weights = self.self_attn(h, h, h, key_padding_mask=~mask)
            h = h + h_attn

        # Cross-attention with context
        if context_vectors is not None and self.use_attention:
            h_cross, _ = self.cross_attn(h, context_vectors, context_vectors)
            h = h + h_cross

        # Graph Neural Network processing
        if self.use_gnn and edge_index is not None and edge_type is not None and edge_index.numel() > 0:
            h = self._apply_gnn(h, edge_index, edge_type, mask)

        # Residual connections
        if self.use_residual:
            for residual_layer in self.residual_connections:
                h = residual_layer(h, x)

        h = self.norm_out(h)

        # Update memory bank
        self.memory_bank.update(h, mask)

        # Sentence-level encoding (latent roots)
        h_masked = h * mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
        h_sent = h_masked.sum(dim=1) / denom  # [B, d_model]

        z = self.enc_lat(h_sent)  # [B, 64] - the semantic roots

        # Feature reconstruction
        feat_hat = self.dec_feat(z)  # [B, k_feats]

        # Token role classification
        role_logits = self.role_head(h)  # [B, N, num_role_labels]

        # Prepare outputs
        outputs = {
            "z": z,
            "feat_hat": feat_hat,
            "role_logits": role_logits,
            "hidden_states": h,
            "sentence_repr": h_sent,
            "mask": mask
        }

        if return_attention and attention_weights is not None:
            outputs["attention_weights"] = attention_weights

        return outputs

    def _apply_gnn(self, h, edge_index, edge_type, mask):
        """Apply graph neural network layers."""
        B, N, D = h.shape

        # Flatten for graph processing
        h_flat = h.view(-1, D)  # [B*N, D]

        # Edge embeddings
        edge_attr = self.edge_rel_emb(edge_type)  # [E, D]

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            h_flat = gnn_layer(h_flat, edge_index, edge_attr)

        # Reshape back
        h_gnn = h_flat.view(B, N, D)

        # Residual connection and normalization
        h_out = self.gnn_norm(h + h_gnn)

        return h_out

    # Loss functions
    def loss_reconstruction(self, feat_hat, feat_rows, mask=None, reduction="mean"):
        """Reconstruction loss for interpretable features."""
        if mask is None:
            sent_target = feat_rows.mean(dim=1)
        else:
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
            sent_target = (feat_rows * mask.unsqueeze(-1)).sum(dim=1) / denom

        return F.mse_loss(feat_hat, sent_target, reduction=reduction)

    def loss_roles(self, role_logits, role_labels, mask):
        """Token role classification loss."""
        if self.use_crf:
            # CRF loss
            log_likelihood = self.crf(role_logits, role_labels, mask=mask, reduction='mean')
            return -log_likelihood
        else:
            # Standard cross-entropy loss
            C = role_logits.size(-1)
            loss = F.cross_entropy(role_logits.view(-1, C), role_labels.view(-1), reduction='none')
            loss = loss.view(role_labels.shape) * mask.float()
            return loss.sum() / mask.float().sum().clamp_min(1)

    def loss_contrastive(self, z, positive_pairs, negative_pairs, margin=1.0):
        """Contrastive loss for semantic similarity."""
        pos_distances = F.pairwise_distance(z[positive_pairs[:, 0]], z[positive_pairs[:, 1]])
        neg_distances = F.pairwise_distance(z[negative_pairs[:, 0]], z[negative_pairs[:, 1]])

        pos_loss = pos_distances.pow(2)
        neg_loss = F.relu(margin - neg_distances).pow(2)

        return (pos_loss.mean() + neg_loss.mean()) / 2

    def predict_roles(self, tok_ids, pos_ids, feat_rows, lengths):
        """Predict token roles with CRF decoding if available."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(tok_ids, pos_ids, feat_rows, lengths)
            role_logits = outputs["role_logits"]
            mask = outputs["mask"]

            if self.use_crf:
                predictions = self.crf.decode(role_logits, mask=mask)
                return predictions
            else:
                return role_logits.argmax(dim=-1).masked_fill(~mask, -1)

    def get_semantic_similarity(self, z1, z2):
        """Compute semantic similarity between latent representations."""
        return F.cosine_similarity(z1, z2, dim=-1)

    def extract_features(self, texts, tokenizer, device='cpu'):
        """Extract features from raw text inputs."""
        # This would integrate with your tokenizer and feature extraction pipeline
        # Placeholder for integration with actual text processing
        pass

    def train_step(self, batch, optimizer, loss_weights=None, clip_grad_norm=1.0):
        """
        Single training step with mixed precision and gradient clipping.

        Args:
            batch: Dictionary containing 'tok_ids', 'pos_ids', 'feat_rows', 'lengths',
                  'role_labels', 'edge_index', 'edge_type'
            optimizer: PyTorch optimizer
            loss_weights: Dictionary of loss component weights
            clip_grad_norm: Gradient clipping norm

        Returns:
            Dictionary of losses and metrics
        """
        if loss_weights is None:
            loss_weights = {'reconstruction': 1.0, 'roles': 1.0, 'contrastive': 0.1}

        self.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = self.forward(
            tok_ids=batch['tok_ids'],
            pos_ids=batch['pos_ids'],
            feat_rows=batch['feat_rows'],
            lengths=batch['lengths'],
            edge_index=batch.get('edge_index'),
            edge_type=batch.get('edge_type')
        )

        # Compute losses
        losses = {}

        # Reconstruction loss
        reconstruction_loss = self.loss_reconstruction(
            outputs['feat_hat'],
            batch['feat_rows'],
            outputs['mask']
        )
        losses['reconstruction'] = reconstruction_loss

        # Role classification loss
        if 'role_labels' in batch:
            role_loss = self.loss_roles(
                outputs['role_logits'],
                batch['role_labels'],
                outputs['mask']
            )
            losses['roles'] = role_loss

        # Contrastive loss (if pairs provided)
        if 'positive_pairs' in batch and 'negative_pairs' in batch:
            contrastive_loss = self.loss_contrastive(
                outputs['z'],
                batch['positive_pairs'],
                batch['negative_pairs']
            )
            losses['contrastive'] = contrastive_loss

        # Combine losses
        total_loss = torch.tensor(0.0, device=outputs['z'].device, requires_grad=True)
        for loss_name, loss_value in losses.items():
            weighted_loss = loss_value * loss_weights.get(loss_name, 1.0)
            total_loss = total_loss + weighted_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)

        optimizer.step()

        # Prepare return metrics
        metrics = {f'loss_{k}': v.item() for k, v in losses.items()}
        metrics['loss_total'] = total_loss.item()

        return metrics

    def evaluate(self, dataloader, device='cpu', return_outputs=False):
        """
        Comprehensive evaluation with multiple metrics.

        Args:
            dataloader: DataLoader for evaluation data
            device: Device to run evaluation on
            return_outputs: Whether to return model outputs

        Returns:
            Dictionary of evaluation metrics
        """
        self.eval()
        total_losses = defaultdict(float)
        total_samples = 0
        all_outputs = [] if return_outputs else None

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self.forward(
                    tok_ids=batch['tok_ids'],
                    pos_ids=batch['pos_ids'],
                    feat_rows=batch['feat_rows'],
                    lengths=batch['lengths'],
                    edge_index=batch.get('edge_index'),
                    edge_type=batch.get('edge_type')
                )

                batch_size = batch['tok_ids'].size(0)
                total_samples += batch_size

                # Compute losses
                recon_loss = self.loss_reconstruction(
                    outputs['feat_hat'],
                    batch['feat_rows'],
                    outputs['mask']
                )
                total_losses['reconstruction'] += recon_loss.item() * batch_size

                if 'role_labels' in batch:
                    role_loss = self.loss_roles(
                        outputs['role_logits'],
                        batch['role_labels'],
                        outputs['mask']
                    )
                    total_losses['roles'] += role_loss.item() * batch_size

                if return_outputs:
                    all_outputs.append({
                        'z': outputs['z'].cpu(),
                        'feat_hat': outputs['feat_hat'].cpu(),
                        'role_logits': outputs['role_logits'].cpu(),
                        'mask': outputs['mask'].cpu()
                    })

        # Average losses
        avg_losses = {k: v / total_samples for k, v in total_losses.items()}

        if return_outputs:
            return avg_losses, all_outputs
        return avg_losses

    def get_attention_maps(self, tok_ids, pos_ids, feat_rows, lengths,
                          layer_idx=-1, head_idx=None):
        """
        Extract attention maps for visualization.

        Args:
            tok_ids, pos_ids, feat_rows, lengths: Input tensors
            layer_idx: Transformer layer index (-1 for last layer)
            head_idx: Attention head index (None for all heads)

        Returns:
            Attention weights tensor
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                tok_ids, pos_ids, feat_rows, lengths,
                return_attention=True
            )

            if 'attention_weights' in outputs:
                attention = outputs['attention_weights']

                if head_idx is not None:
                    attention = attention[:, head_idx]

                return attention

        return None

    def generate_embeddings(self, dataloader, device='cpu'):
        """
        Generate sentence-level embeddings for a dataset.

        Args:
            dataloader: DataLoader for the dataset
            device: Device to run on

        Returns:
            Numpy array of embeddings
        """
        self.eval()
        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                outputs = self.forward(
                    tok_ids=batch['tok_ids'],
                    pos_ids=batch['pos_ids'],
                    feat_rows=batch['feat_rows'],
                    lengths=batch['lengths']
                )

                embeddings.append(outputs['z'].cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def save_checkpoint(self, filepath, optimizer=None, scheduler=None,
                       epoch=None, metrics=None):
        """
        Save model checkpoint with metadata.

        Args:
            filepath: Path to save checkpoint
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch number
            metrics: Training metrics to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'k_feats': self.k_feats,
                'n_pos': self.n_pos,
                'n_rels': self.n_rels,
                'n_layers': self.n_layers,
                'n_heads': self.n_heads,
                'p_drop': self.p_drop,
                'use_transformer': self.use_transformer,
                'use_gnn': self.use_gnn,
                'use_crf': self.use_crf,
                'use_attention': self.use_attention,
                'use_residual': self.use_residual,
                'num_role_labels': self.num_role_labels
            }
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath, device='cpu'):
        """
        Load model from checkpoint.

        Args:
            filepath: Path to checkpoint file
            device: Device to load model on

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Create model with saved config
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        logging.info(f"Model loaded from {filepath}")
        return model, checkpoint


class GraphConvLayer(nn.Module):
    """Advanced Graph Convolution Layer with multi-head attention and message passing."""

    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1, edge_dim=None,
                 use_attention=True, use_residual=True, activation='gelu'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.edge_dim = edge_dim or in_dim

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        # Message computation networks
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 2 + self.edge_dim, out_dim * 2),
            getattr(nn, activation.upper())(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )

        # Multi-head attention for message weighting
        if use_attention:
            self.query_proj = nn.Linear(in_dim, out_dim)
            self.key_proj = nn.Linear(in_dim + self.edge_dim, out_dim)
            self.value_proj = nn.Linear(in_dim + self.edge_dim, out_dim)
            self.attention_dropout = nn.Dropout(dropout)

        # Update networks
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim * 2),
            getattr(nn, activation.upper())(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )

        # Normalization and residual
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        if use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, node_features, edge_index, edge_attr=None, return_attention=False):
        """
        Apply advanced graph convolution with attention.

        Args:
            node_features: [N, in_dim] node features
            edge_index: [2, E] edge connections
            edge_attr: [E, edge_dim] edge features (optional)
            return_attention: whether to return attention weights

        Returns:
            Updated node features [N, out_dim]
        """
        N = node_features.size(0)

        if edge_index.size(1) == 0:
            # No edges, just apply self-transformation
            if self.use_residual:
                residual = self.residual_proj(node_features)
                out = self.update_net(torch.cat([node_features, torch.zeros_like(residual)], dim=-1))
                return self.norm1(residual + self.dropout(out))
            else:
                return self.update_net(torch.cat([node_features, torch.zeros(N, self.out_dim, device=node_features.device)], dim=-1))

        src, dst = edge_index[0], edge_index[1]

        # Default edge attributes if none provided
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), self.edge_dim,
                                  device=node_features.device, dtype=node_features.dtype)

        # Gather source and destination node features
        src_features = node_features[src]  # [E, in_dim]
        dst_features = node_features[dst]  # [E, in_dim]

        # Compute messages
        message_input = torch.cat([src_features, dst_features, edge_attr], dim=-1)
        messages = self.message_net(message_input)  # [E, out_dim]

        attention_weights = None

        if self.use_attention:
            # Multi-head attention for message weighting
            batch_size = 1  # Single graph processing

            queries = self.query_proj(dst_features).view(-1, self.num_heads, self.head_dim)  # [E, H, d_k]
            key_input = torch.cat([src_features, edge_attr], dim=-1)
            keys = self.key_proj(key_input).view(-1, self.num_heads, self.head_dim)  # [E, H, d_k]
            values = self.value_proj(key_input).view(-1, self.num_heads, self.head_dim)  # [E, H, d_k]

            # Scaled dot-product attention
            scale = self.head_dim ** -0.5
            attn_scores = torch.sum(queries * keys, dim=-1) * scale  # [E, H]

            # Normalize attention scores per destination node
            dst_max_idx = dst.max().item() + 1
            attn_weights_full = torch.full((dst_max_idx, self.num_heads), float('-inf'),
                                         device=node_features.device)
            attn_weights_full[dst] = attn_scores
            attn_weights_normalized = F.softmax(attn_weights_full, dim=0)[dst]  # [E, H]

            # Apply dropout
            if self.training:
                attn_weights_normalized = self.attention_dropout(attn_weights_normalized)

            # Apply attention to messages
            messages = messages.view(-1, self.num_heads, self.head_dim)  # [E, H, d_k]
            attended_messages = attn_weights_normalized.unsqueeze(-1) * messages  # [E, H, d_k]
            messages = attended_messages.view(-1, self.out_dim)  # [E, out_dim]

            if return_attention:
                attention_weights = attn_weights_normalized

        # Aggregate messages for each node
        aggregated = torch.zeros(N, self.out_dim, device=node_features.device, dtype=node_features.dtype)
        aggregated.index_add_(0, dst, messages)

        # Update node features
        update_input = torch.cat([node_features, aggregated], dim=-1)
        updated = self.update_net(update_input)

        # Residual connection and normalization
        if self.use_residual:
            residual = self.residual_proj(node_features)
            output = self.norm1(residual + self.dropout(updated))
        else:
            output = self.norm1(self.dropout(updated))

        if return_attention:
            return output, attention_weights
        return output


class ResidualConnection(nn.Module):
    """Advanced Residual Connection with learnable gating and normalization."""

    def __init__(self, d_model, dropout=0.1, use_gate=True, norm_type='layer'):
        super().__init__()
        self.d_model = d_model
        self.use_gate = use_gate
        self.dropout = nn.Dropout(dropout)

        # Normalization
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == 'rms':
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.Identity()

        # Learnable gating mechanism
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )

    def forward(self, x, residual=None):
        """
        Apply residual connection with optional gating.

        Args:
            x: Main input tensor
            residual: Residual tensor (defaults to x if None)

        Returns:
            Output tensor after residual connection
        """
        if residual is None:
            residual = x

        if self.use_gate:
            # Learnable gating
            combined = torch.cat([x, residual], dim=-1)
            gate = self.gate(combined)
            gated = gate * x + (1 - gate) * residual
            return self.norm(self.dropout(gated))
        else:
            # Standard residual connection
            return self.norm(residual + self.dropout(x))


class MultiScaleProcessor(nn.Module):
    """Advanced Multi-Scale Processing with hierarchical attention and temporal convolutions."""

    def __init__(self, d_model, kernel_sizes=[3, 5, 7, 9], dropout=0.1,
                 use_hierarchical=True, use_pooling=True):
        super().__init__()
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        self.use_hierarchical = use_hierarchical
        self.use_pooling = use_pooling

        # Multi-scale convolution branches
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for kernel_size in kernel_sizes:
            # Dilated convolutions for different receptive fields
            conv_layers = nn.Sequential(
                nn.Conv1d(d_model, d_model // self.num_scales,
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.GELU(),
                nn.Conv1d(d_model // self.num_scales, d_model // self.num_scales,
                         kernel_size=kernel_size, padding=kernel_size//2, dilation=2),
                nn.GELU()
            )
            self.convs.append(conv_layers)
            self.norms.append(nn.LayerNorm(d_model // self.num_scales))

        # Hierarchical attention for scale fusion
        if use_hierarchical:
            self.scale_attention = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=dropout, batch_first=True
            )
            self.scale_norm = nn.LayerNorm(d_model)

        # Adaptive pooling for different temporal scales
        if use_pooling:
            self.pooling_layers = nn.ModuleList([
                nn.AdaptiveAvgPool1d(None),  # Identity pooling
                nn.AdaptiveAvgPool1d(None),  # Will be set dynamically
                nn.AdaptiveMaxPool1d(None),  # Will be set dynamically
                nn.AdaptiveAvgPool1d(None)   # Will be set dynamically
            ])

        # Final projection and normalization
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Apply multi-scale processing with hierarchical attention.

        Args:
            x: Input tensor [B, N, D]
            mask: Optional attention mask [B, N]

        Returns:
            Multi-scale processed tensor [B, N, D]
        """
        B, N, D = x.shape

        # Transpose for conv1d: [B, D, N]
        x_conv = x.transpose(1, 2)

        # Apply multi-scale convolutions
        scale_outputs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Apply convolution
            conv_out = conv(x_conv)  # [B, D//num_scales, N]

            # Transpose back and normalize
            conv_out = conv_out.transpose(1, 2)  # [B, N, D//num_scales]
            conv_out = norm(conv_out)

            # Apply pooling if enabled
            if self.use_pooling and i > 0:
                # Different pooling strategies for different scales
                pool_factor = min(2 ** i, 4)
                if N >= pool_factor * 2:
                    pooled_length = N // pool_factor
                    conv_out = conv_out.transpose(1, 2)  # [B, D//num_scales, N]
                    conv_out = F.adaptive_avg_pool1d(conv_out, pooled_length)
                    conv_out = F.interpolate(conv_out, size=N, mode='linear', align_corners=False)
                    conv_out = conv_out.transpose(1, 2)  # [B, N, D//num_scales]

            scale_outputs.append(conv_out)

        # Concatenate all scales
        multi_scale = torch.cat(scale_outputs, dim=-1)  # [B, N, D]

        # Apply hierarchical attention if enabled
        if self.use_hierarchical:
            # Self-attention across scales
            if mask is not None:
                key_padding_mask = ~mask
            else:
                key_padding_mask = None

            attended, _ = self.scale_attention(
                multi_scale, multi_scale, multi_scale,
                key_padding_mask=key_padding_mask
            )
            multi_scale = self.scale_norm(multi_scale + attended)

        # Final projection
        output = self.output_proj(multi_scale)
        output = self.output_norm(output)
        output = self.dropout(output)

        return output


class MemoryBank(nn.Module):
    """Advanced Memory Bank with episodic storage and consolidation mechanisms."""

    def __init__(self, d_model, capacity=10000, consolidation_threshold=0.8,
                 update_frequency=100, compression_ratio=0.5):
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.consolidation_threshold = consolidation_threshold
        self.update_frequency = update_frequency
        self.compression_ratio = compression_ratio

        # Memory storage
        self.register_buffer('memory', torch.zeros(capacity, d_model))
        self.register_buffer('memory_scores', torch.zeros(capacity))
        self.register_buffer('memory_timestamps', torch.zeros(capacity))
        self.register_buffer('memory_access_count', torch.zeros(capacity))
        self.register_buffer('write_pointer', torch.zeros(1, dtype=torch.long))
        self.register_buffer('memory_size', torch.zeros(1, dtype=torch.long))
        self.register_buffer('global_timestamp', torch.zeros(1, dtype=torch.long))

        # Query and key projections for retrieval
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Attention mechanism for memory consolidation
        self.consolidation_attn = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )

        # Episodic encoding
        self.episode_encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def update(self, features, mask=None, importance_scores=None):
        """
        Update memory bank with new features using importance-based storage.

        Args:
            features: New features to store [B, N, D]
            mask: Optional mask for valid positions [B, N]
            importance_scores: Optional importance scores [B, N]
        """
        B, N, D = features.shape

        # Extract valid features based on mask
        if mask is not None:
            valid_features = features[mask]  # [Valid, D]
            if importance_scores is not None:
                valid_scores = importance_scores[mask]  # [Valid]
            else:
                valid_scores = torch.ones(valid_features.size(0), device=features.device)
        else:
            valid_features = features.view(-1, D)  # [B*N, D]
            valid_scores = torch.ones(valid_features.size(0), device=features.device)

        if valid_features.size(0) == 0:
            return

        # Encode episodes
        encoded_features = self.episode_encoder(valid_features)

        # Update memory
        with torch.no_grad():
            current_size = int(self.memory_size.item())
            write_ptr = int(self.write_pointer.item())
            timestamp = int(self.global_timestamp.item())

            # Store new memories
            for i in range(valid_features.size(0)):
                if current_size < self.capacity:
                    # Add to available slot
                    self.memory[current_size] = encoded_features[i]
                    self.memory_scores[current_size] = valid_scores[i]
                    self.memory_timestamps[current_size] = timestamp
                    self.memory_access_count[current_size] = 0
                    current_size += 1
                else:
                    # Replace least important memory
                    # Compute replacement scores (combination of importance, recency, access)
                    time_decay = torch.exp(-(timestamp - self.memory_timestamps[:current_size]) * 0.01)
                    access_bonus = torch.log1p(self.memory_access_count[:current_size])
                    replacement_scores = (self.memory_scores[:current_size] * time_decay +
                                        access_bonus * 0.1)

                    # Find least important memory to replace
                    min_idx = replacement_scores.argmin().item()

                    # Replace if new memory is more important
                    if valid_scores[i] > replacement_scores[min_idx]:
                        self.memory[min_idx] = encoded_features[i]
                        self.memory_scores[min_idx] = valid_scores[i]
                        self.memory_timestamps[min_idx] = timestamp
                        self.memory_access_count[min_idx] = 0

            # Update pointers and timestamp
            self.memory_size[0] = current_size
            self.global_timestamp[0] = timestamp + 1

            # Periodic consolidation
            if timestamp % self.update_frequency == 0:
                self._consolidate_memory()

    def retrieve(self, query, k=5, return_scores=False):
        """
        Retrieve k most relevant memories for the given query.

        Args:
            query: Query tensor [B, D] or [D]
            k: Number of memories to retrieve
            return_scores: Whether to return similarity scores

        Returns:
            Retrieved memories [B, k, D] or [k, D]
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, D]
            squeeze_output = True
        else:
            squeeze_output = False

        B, D = query.shape
        current_size = int(self.memory_size.item())

        if current_size == 0:
            # No memories stored yet
            empty_memories = torch.zeros(B, k, D, device=query.device)
            empty_scores = torch.zeros(B, k, device=query.device)
            if squeeze_output:
                empty_memories = empty_memories.squeeze(0)
                empty_scores = empty_scores.squeeze(0)
            return (empty_memories, empty_scores) if return_scores else empty_memories

        # Project query and memory
        q = self.query_proj(query)  # [B, D]
        valid_memory = self.memory[:current_size]  # [M, D]
        k_mem = self.key_proj(valid_memory)  # [M, D]
        v_mem = self.value_proj(valid_memory)  # [M, D]

        # Compute similarity scores
        similarity = torch.matmul(q, k_mem.t())  # [B, M]
        similarity = similarity / math.sqrt(D)

        # Apply memory importance and recency weighting
        with torch.no_grad():
            timestamp = int(self.global_timestamp.item())
            importance_weights = self.memory_scores[:current_size]
            recency_weights = torch.exp(-(timestamp - self.memory_timestamps[:current_size]) * 0.01)
            combined_weights = importance_weights * recency_weights

            # Update access counts for retrieved memories
            weighted_similarity = similarity * combined_weights.unsqueeze(0)

        # Get top-k memories
        k_actual = min(k, current_size)
        top_scores, top_indices = weighted_similarity.topk(k_actual, dim=-1)  # [B, k]

        # Update access counts
        with torch.no_grad():
            self.memory_access_count[:current_size].index_add_(
                0, top_indices.view(-1),
                torch.ones_like(top_indices.view(-1), dtype=torch.float)
            )

        # Retrieve memories
        retrieved_memories = v_mem[top_indices]  # [B, k, D]

        # Pad if necessary
        if k_actual < k:
            padding = torch.zeros(B, k - k_actual, D, device=query.device)
            retrieved_memories = torch.cat([retrieved_memories, padding], dim=1)
            padding_scores = torch.zeros(B, k - k_actual, device=query.device)
            top_scores = torch.cat([top_scores, padding_scores], dim=1)

        if squeeze_output:
            retrieved_memories = retrieved_memories.squeeze(0)
            top_scores = top_scores.squeeze(0)

        if return_scores:
            return retrieved_memories, top_scores
        return retrieved_memories

    def _consolidate_memory(self):
        """
        Consolidate memories by clustering and compressing similar ones.
        """
        current_size = int(self.memory_size.item())
        if current_size < 2:
            return

        with torch.no_grad():
            valid_memory = self.memory[:current_size]  # [M, D]

            # Compute pairwise similarities
            similarity_matrix = torch.matmul(valid_memory, valid_memory.t())  # [M, M]
            similarity_matrix = similarity_matrix / torch.norm(valid_memory, dim=1, keepdim=True)
            similarity_matrix = similarity_matrix / torch.norm(valid_memory, dim=1).unsqueeze(0)

            # Find highly similar memories
            similarity_mask = (similarity_matrix > self.consolidation_threshold) & \
                            (similarity_matrix < 1.0)  # Exclude self-similarity

            # Group similar memories
            consolidated_indices = []
            processed = set()

            for i in range(current_size):
                if i in processed:
                    continue

                similar_indices = [i] + (similarity_mask[i].nonzero(as_tuple=True)[0].tolist())
                similar_indices = [idx for idx in similar_indices if idx not in processed]

                if len(similar_indices) > 1:
                    # Consolidate these memories
                    similar_memories = valid_memory[similar_indices]  # [S, D]
                    similar_scores = self.memory_scores[similar_indices]  # [S]

                    # Weighted average based on importance scores
                    weights = F.softmax(similar_scores, dim=0)
                    consolidated_memory = torch.sum(similar_memories * weights.unsqueeze(-1), dim=0)
                    consolidated_score = similar_scores.max()  # Keep highest importance

                    # Replace the first memory with consolidated version
                    self.memory[i] = consolidated_memory
                    self.memory_scores[i] = consolidated_score

                    # Mark others for removal (will be handled by compaction)
                    for idx in similar_indices[1:]:
                        processed.add(idx)
                        self.memory_scores[idx] = 0  # Mark for removal

                processed.add(i)

            # Compact memory by removing zero-scored entries
            valid_mask = self.memory_scores[:current_size] > 0
            if valid_mask.sum() < current_size:
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                new_size = len(valid_indices)

                # Move valid memories to front
                self.memory[:new_size] = self.memory[valid_indices]
                self.memory_scores[:new_size] = self.memory_scores[valid_indices]
                self.memory_timestamps[:new_size] = self.memory_timestamps[valid_indices]
                self.memory_access_count[:new_size] = self.memory_access_count[valid_indices]

                # Clear the rest
                self.memory[new_size:current_size] = 0
                self.memory_scores[new_size:current_size] = 0
                self.memory_timestamps[new_size:current_size] = 0
                self.memory_access_count[new_size:current_size] = 0

                # Update size
                self.memory_size[0] = new_size


# ===========================
# ADVANCED ATTENTION MECHANISMS
# ===========================

class SparseAttention(nn.Module):
    """Sparse Attention mechanism for efficient long sequence processing."""

    def __init__(self, d_model, n_heads, dropout=0.1, sparsity_pattern='local',
                 local_window=64, stride=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.sparsity_pattern = sparsity_pattern
        self.local_window = local_window
        self.stride = stride

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def create_sparse_mask(self, seq_len, device):
        """Create sparse attention mask based on pattern."""
        if self.sparsity_pattern == 'local':
            # Local window attention
            mask = torch.zeros(seq_len, seq_len, device=device)
            for i in range(seq_len):
                start = max(0, i - self.local_window // 2)
                end = min(seq_len, i + self.local_window // 2 + 1)
                mask[i, start:end] = 1
        elif self.sparsity_pattern == 'strided':
            # Strided attention pattern
            mask = torch.zeros(seq_len, seq_len, device=device)
            for i in range(seq_len):
                # Local attention
                start = max(0, i - self.local_window // 2)
                end = min(seq_len, i + self.local_window // 2 + 1)
                mask[i, start:end] = 1
                # Strided attention
                for j in range(0, seq_len, self.stride):
                    mask[i, j] = 1
        else:
            # Default to full attention
            mask = torch.ones(seq_len, seq_len, device=device)

        return mask.bool()

    def forward(self, x, mask=None):
        """Apply sparse attention."""
        B, N, D = x.shape

        # Project to Q, K, V
        q = self.w_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        k = self.w_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        v = self.w_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]

        # Create sparse attention mask
        sparse_mask = self.create_sparse_mask(N, x.device)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Apply sparse mask
        scores = scores.masked_fill(~sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply input mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, N, d_k]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]

        # Output projection
        out = self.w_o(out)

        return out


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention for fusing different modalities."""

    def __init__(self, d_model, n_heads, dropout=0.1, temperature=1.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Cross-modal projection layers
        self.modality_proj = nn.ModuleDict()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def add_modality_projection(self, modality_name, input_dim):
        """Add projection layer for a specific modality."""
        self.modality_proj[modality_name] = nn.Linear(input_dim, self.d_model)

    def forward(self, query, key_values, modality_mask=None):
        """
        Apply cross-modal attention.

        Args:
            query: Query tensor [B, N, D]
            key_values: Dict of key-value tensors from different modalities
            modality_mask: Optional mask for modalities
        """
        B, N, D = query.shape

        # Project query
        q = self.w_q(query).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        all_keys = []
        all_values = []

        # Process each modality
        for modality_name, kv_tensor in key_values.items():
            # Project to common space if needed
            if modality_name in self.modality_proj:
                kv_projected = self.modality_proj[modality_name](kv_tensor)
            else:
                kv_projected = kv_tensor

            # Ensure proper dimensions
            if kv_projected.dim() == 2:
                kv_projected = kv_projected.unsqueeze(1).expand(B, -1, -1)

            M = kv_projected.size(1)
            k = self.w_k(kv_projected).view(B, M, self.n_heads, self.d_k).transpose(1, 2)
            v = self.w_v(kv_projected).view(B, M, self.n_heads, self.d_k).transpose(1, 2)

            all_keys.append(k)
            all_values.append(v)

        # Concatenate all keys and values
        if len(all_keys) > 0:
            keys = torch.cat(all_keys, dim=2)  # [B, H, total_M, d_k]
            values = torch.cat(all_values, dim=2)  # [B, H, total_M, d_k]
        else:
            return query

        # Compute cross-modal attention
        scores = torch.matmul(q, keys.transpose(-2, -1)) / (self.d_k ** 0.5 * self.temperature)

        if modality_mask is not None:
            scores = scores.masked_fill(~modality_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        out = torch.matmul(attn_weights, values)  # [B, H, N, d_k]
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        # Output projection and residual
        out = self.w_o(out)
        out = self.norm(query + out)

        return out

        # Prepare edge attributes
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), self.edge_dim, device=node_features.device)

        # Message computation
        src_features = node_features[src]  # [E, in_dim]
        dst_features = node_features[dst]  # [E, in_dim]

        # Combine node and edge features for messages
        message_input = torch.cat([src_features, dst_features, edge_attr], dim=-1)  # [E, in_dim*2 + edge_dim]
        messages = self.message_net(message_input)  # [E, out_dim]

        # Multi-head attention (if enabled)
        attention_weights = None
        if self.use_attention:
            # Compute attention scores
            queries = self.query_proj(dst_features)  # [E, out_dim]
            keys = self.key_proj(torch.cat([src_features, edge_attr], dim=-1))  # [E, out_dim]
            values = self.value_proj(torch.cat([src_features, edge_attr], dim=-1))  # [E, out_dim]

            # Reshape for multi-head attention
            queries = queries.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [H, E, d_k]
            keys = keys.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [H, E, d_k]
            values = values.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [H, E, d_k]

            # Compute attention weights
            attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attention_weights = F.softmax(attention_scores, dim=-1)  # [H, E, E]
            attention_weights = self.attention_dropout(attention_weights)

            # Apply attention to values
            attended_values = torch.matmul(attention_weights, values)  # [H, E, d_k]
            attended_values = attended_values.transpose(0, 1).contiguous().view(-1, self.out_dim)  # [E, out_dim]

            # Weight messages by attention
            messages = messages * attended_values

        # Message aggregation using scatter_add
        aggregated_messages = torch.zeros(N, self.out_dim, device=node_features.device)
        aggregated_messages.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.out_dim), messages)

        # Node update
        update_input = torch.cat([node_features, aggregated_messages], dim=-1)  # [N, in_dim + out_dim]
        updated_features = self.update_net(update_input)  # [N, out_dim]

        # Residual connection and normalization
        if self.use_residual:
            residual = self.residual_proj(node_features)
            updated_features = self.norm1(residual + self.dropout(updated_features))
        else:
            updated_features = self.norm1(self.dropout(updated_features))

        # Second normalization layer
        updated_features = self.norm2(updated_features)

        if return_attention:
            return updated_features, attention_weights
        return updated_features


class ResidualConnection(nn.Module):
    """Advanced residual connection with multiple normalization options and gating."""

    def __init__(self, d_model, dropout=0.1, norm_type='layer', use_gate=False,
                 gate_activation='sigmoid', pre_norm=True, scale_residual=False):
        super().__init__()
        self.d_model = d_model
        self.pre_norm = pre_norm
        self.use_gate = use_gate
        self.scale_residual = scale_residual

        # Normalization layer
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(d_model)
        elif norm_type == 'rms':
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        # Gating mechanism for adaptive residual weighting
        if use_gate:
            self.gate_proj = nn.Linear(d_model, d_model)
            if gate_activation == 'sigmoid':
                self.gate_activation = nn.Sigmoid()
            elif gate_activation == 'tanh':
                self.gate_activation = nn.Tanh()
            else:
                self.gate_activation = nn.Identity()

        # Learnable scaling factor
        if scale_residual:
            self.scale_factor = nn.Parameter(torch.ones(1))
        else:
            self.scale_factor = 1.0

    def forward(self, x, sublayer_output, attention_weights=None):
        """
        Apply residual connection with normalization and optional gating.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            sublayer_output: Output from sublayer [batch_size, seq_len, d_model]
            attention_weights: Optional attention weights for adaptive gating

        Returns:
            Output after residual connection and normalization
        """
        # Apply dropout to sublayer output
        sublayer_output = self.dropout(sublayer_output)

        # Gating mechanism
        if self.use_gate:
            # Use input for gate computation
            gate_input = x if not hasattr(self, 'gate_input_type') or self.gate_input_type == 'input' else sublayer_output
            gate_weights = self.gate_activation(self.gate_proj(gate_input))
            sublayer_output = gate_weights * sublayer_output

        # Pre-norm or post-norm
        if self.pre_norm:
            # Pre-normalization: norm(x) + sublayer_output
            normalized_x = self.norm(x)
            return normalized_x + sublayer_output * self.scale_factor
        else:
            # Post-normalization: norm(x + sublayer_output)
            return self.norm(x + sublayer_output * self.scale_factor)

    def extra_repr(self):
        return f'd_model={self.d_model}, pre_norm={self.pre_norm}, use_gate={self.use_gate}, scale_residual={self.scale_residual}'


class MultiScaleProcessor(nn.Module):
    """Advanced multi-scale temporal processing with hierarchical feature fusion."""

    def __init__(self, d_model, kernel_sizes=[3, 5, 7, 9], dropout=0.1,
                 use_dilated_conv=True, use_attention_fusion=True,
                 hierarchical_levels=2, activation='gelu'):
        super().__init__()
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        self.use_dilated_conv = use_dilated_conv
        self.use_attention_fusion = use_attention_fusion
        self.hierarchical_levels = hierarchical_levels

        # Multi-scale convolution branches
        self.conv_branches = nn.ModuleList()
        scale_dim = d_model // self.num_scales

        for i, k in enumerate(kernel_sizes):
            branch = nn.ModuleList()
            for level in range(hierarchical_levels):
                dilation = 2**level if use_dilated_conv else 1
                padding = (k + (k - 1) * (dilation - 1)) // 2

                conv_layer = nn.Conv1d(
                    d_model if level == 0 else scale_dim,
                    scale_dim,
                    kernel_size=k,
                    padding=padding,
                    dilation=dilation,
                    groups=1
                )
                branch.append(nn.Sequential(
                    conv_layer,
                    nn.BatchNorm1d(scale_dim),
                    getattr(nn, activation.upper())(),
                    nn.Dropout(dropout)
                ))
            self.conv_branches.append(branch)

        # Attention-based fusion mechanism
        if use_attention_fusion:
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(d_model)

        # Scale-wise feature enhancement
        self.scale_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scale_dim, scale_dim * 2),
                getattr(nn, activation.upper())(),
                nn.Dropout(dropout),
                nn.Linear(scale_dim * 2, scale_dim)
            ) for _ in range(self.num_scales)
        ])

        # Final projection and normalization
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            getattr(nn, activation.upper())(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))

    def forward(self, x, mask=None, return_scale_outputs=False):
        """
        Apply multi-scale hierarchical processing.

        Args:
            x: Input tensor [B, N, D]
            mask: Optional attention mask [B, N]
            return_scale_outputs: Whether to return individual scale outputs

        Returns:
            Multi-scale processed features [B, N, D]
        """
        batch_size, seq_len, d_model = x.shape

        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # Transpose for convolution: [B, D, N]
        x_conv = x.transpose(1, 2)

        # Process each scale with hierarchical levels
        scale_outputs = []
        scale_features = []

        for i, (conv_stack, norm, pool) in enumerate(zip(self.conv_stacks, self.norms, self.pooling_layers)):
            # Apply convolution stack
            conv_out = conv_stack(x_conv)  # [B, D//num_scales, N]

            # Apply pooling with different scales
            if self.use_pooling and i > 0:
                # Different pooling for different scales
                pool_size = min(seq_len // (2**i), seq_len)
                if pool_size > 1:
                    pooled = F.adaptive_avg_pool1d(conv_out, pool_size)
                    unpooled = F.interpolate(pooled, size=seq_len, mode='linear', align_corners=False)
                    conv_out = unpooled

            # Transpose back and normalize
            conv_out = conv_out.transpose(1, 2)  # [B, N, D//num_scales]
            conv_out = norm(conv_out)

            scale_outputs.append(conv_out)
            scale_features.append(conv_out)

        # Concatenate all scale outputs
        multi_scale_out = torch.cat(scale_outputs, dim=-1)  # [B, N, D]

        # Hierarchical attention across scales
        if self.use_hierarchical:
            if mask is not None:
                key_padding_mask = ~mask
            else:
                key_padding_mask = None

            attended_out, _ = self.scale_attention(
                multi_scale_out, multi_scale_out, multi_scale_out,
                key_padding_mask=key_padding_mask
            )
            multi_scale_out = self.scale_norm(multi_scale_out + attended_out)

        # Final projection and output
        output = self.output_proj(multi_scale_out)

        if self.use_residual:
            output = self.output_norm(x + self.dropout(output))
        else:
            output = self.output_norm(self.dropout(output))

        return output


def create_world_engine(config):
    """Factory function to create World Engine with configuration."""
    return WorldEngine(
        vocab_size=config.get('vocab_size', 10000),
        d_model=config.get('d_model', 512),
        k_feats=config.get('k_feats', 100),
        n_pos=config.get('n_pos', 50),
        n_rels=config.get('n_rels', 20),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 8),
        p_drop=config.get('dropout', 0.1),
        use_transformer=config.get('use_transformer', True),
        use_gnn=config.get('use_gnn', True),
        use_crf=config.get('use_crf', True),
        num_role_labels=config.get('num_role_labels', 5)
    )


# ===========================
# COMPREHENSIVE TESTING SUITE
# ===========================

def run_basic_tests():
    """Run basic functionality tests."""
    print("🧪 Running World Engine Test Suite...")

    # Test configuration
    config = {
        'vocab_size': 1000,
        'd_model': 128,
        'k_feats': 50,
        'n_pos': 25,
        'n_rels': 10,
        'n_layers': 2,
        'n_heads': 4,
        'dropout': 0.1,
        'use_transformer': True,
        'use_gnn': True,
        'use_crf': False,  # Disable CRF for basic testing
        'num_role_labels': 5
    }

    print(f"✅ Configuration: {config}")

    # Create model
    try:
        model = create_world_engine(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully with {param_count:,} parameters")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

    # Test forward pass
    try:
        batch_size = 4
        seq_len = 10

        # Create dummy input
        tok_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        pos_ids = torch.randint(0, config['n_pos'], (batch_size, seq_len))
        feat_rows = torch.randn(batch_size, seq_len, config['k_feats'])
        lengths = torch.tensor([seq_len] * batch_size)

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model.forward(tok_ids, pos_ids, feat_rows, lengths)

        # Validate outputs
        assert 'z' in outputs, "Missing semantic embeddings"
        assert 'feat_hat' in outputs, "Missing reconstructed features"
        assert 'role_logits' in outputs, "Missing role predictions"
        assert outputs['z'].shape == (batch_size, 64), f"Wrong z shape: {outputs['z'].shape}"

        print(f"✅ Forward pass successful: z={outputs['z'].shape}, feat_hat={outputs['feat_hat'].shape}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    print("🎉 All tests passed! World Engine is fully operational.")
    return True


def run_performance_benchmark():
    """Run performance benchmarks."""
    print("\n📊 Running Performance Benchmarks...")

    config = {
        'vocab_size': 10000,
        'd_model': 512,
        'k_feats': 100,
        'n_pos': 50,
        'n_rels': 20,
        'n_layers': 6,
        'n_heads': 8,
        'dropout': 0.1,
        'use_transformer': True,
        'use_gnn': True,
        'use_crf': False,
        'num_role_labels': 5
    }

    model = create_world_engine(config)
    model.eval()

    # Benchmark different batch sizes and sequence lengths
    batch_sizes = [1, 4, 16]
    seq_lengths = [32, 128, 512]

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            # Create input
            tok_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
            pos_ids = torch.randint(0, config['n_pos'], (batch_size, seq_len))
            feat_rows = torch.randn(batch_size, seq_len, config['k_feats'])
            lengths = torch.tensor([seq_len] * batch_size)

            # Time forward pass
            start_time = time.time()
            with torch.no_grad():
                outputs = model.forward(tok_ids, pos_ids, feat_rows, lengths)
            end_time = time.time()

            forward_time = (end_time - start_time) * 1000  # Convert to ms
            tokens_per_sec = (batch_size * seq_len) / (forward_time / 1000)

            print(f"📈 Batch={batch_size}, SeqLen={seq_len}: {forward_time:.2f}ms ({tokens_per_sec:.1f} tokens/sec)")


def demonstrate_capabilities():
    """Demonstrate key capabilities."""
    print("\n🚀 Demonstrating World Engine Capabilities...")

    # Create a smaller model for demonstration
    config = {
        'vocab_size': 5000, 'd_model': 256, 'k_feats': 50, 'n_pos': 25,
        'n_rels': 10, 'n_layers': 3, 'n_heads': 8, 'dropout': 0.1,
        'use_transformer': True, 'use_gnn': True, 'use_crf': False, 'num_role_labels': 5
    }

    model = create_world_engine(config)

    # Show model statistics
    print(f"\n📊 Model Statistics:")
    print(f"   🔢 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   🎯 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   💾 Model size: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    print("=" * 80)
    print("🌍 WORLD ENGINE V3.1 - ADVANCED NEURAL ARCHITECTURE")
    print("=" * 80)

    # Basic functionality tests
    if run_basic_tests():
        # Performance benchmarks
        run_performance_benchmark()

        # Capability demonstration
        demonstrate_capabilities()

        print("\n" + "=" * 80)
        print("🎉 World Engine V3.1 is fully operational!")
        print("🔗 Ready for Studio Controls integration")
        print("📊 Advanced neural architecture with complete implementation")
        print("🧠 Features: Multi-modal processing, Graph neural networks, Memory systems")
        print("=" * 80)
    else:
        print("\n❌ Tests failed - please check the implementation")

        for scale_idx, branch in enumerate(self.conv_branches):
            scale_input = x_conv

            # Apply hierarchical convolution levels
            for level, conv_layer in enumerate(branch):
                scale_input = conv_layer(scale_input)

            # Transpose back: [B, N, scale_dim]
            scale_output = scale_input.transpose(1, 2)

            # Enhance scale-specific features
            enhanced_output = self.scale_enhancer[scale_idx](scale_output)
            scale_outputs.append(enhanced_output)
            scale_features.append(enhanced_output)

        # Concatenate all scales: [B, N, D]
        multi_scale_features = torch.cat(scale_outputs, dim=-1)

        # Apply learnable scale weighting
        weighted_features = []
        start_idx = 0
        scale_dim = d_model // self.num_scales

        for i in range(self.num_scales):
            end_idx = start_idx + scale_dim
            scale_feature = multi_scale_features[:, :, start_idx:end_idx]
            weighted_feature = scale_feature * self.scale_weights[i]
            weighted_features.append(weighted_feature)
            start_idx = end_idx

        weighted_multi_scale = torch.cat(weighted_features, dim=-1)

        # Attention-based fusion
        if self.use_attention_fusion:
            # Self-attention over the multi-scale features
            if mask is not None:
                attn_mask = ~mask  # Convert to attention mask format
            else:
                attn_mask = None

            fused_features, attention_weights = self.fusion_attention(
                weighted_multi_scale,
                weighted_multi_scale,
                weighted_multi_scale,
                key_padding_mask=attn_mask
            )

            # Residual connection and normalization
            fused_features = self.fusion_norm(weighted_multi_scale + self.dropout(fused_features))
        else:
            fused_features = weighted_multi_scale

        # Final output projection
        output = self.output_projection(fused_features)
        output = self.output_norm(output)

        # Apply mask to output
        if mask is not None:
            output = output * mask.unsqueeze(-1)

        if return_scale_outputs:
            return output, scale_features
        return output

    def get_scale_importance(self):
        """Return the learned importance weights for each scale."""
        return F.softmax(self.scale_weights, dim=0)


class MemoryBank(nn.Module):
    """Advanced episodic memory bank with attention-based retrieval and consolidation."""

    def __init__(self, d_model, max_size=1000, num_heads=8, consolidation_threshold=0.8,
                 use_attention_retrieval=True, use_episodic_memory=True,
                 forgetting_factor=0.99, temperature=1.0):
        super().__init__()
        self.d_model = d_model
        self.max_size = max_size
        self.num_heads = num_heads
        self.consolidation_threshold = consolidation_threshold
        self.use_attention_retrieval = use_attention_retrieval
        self.use_episodic_memory = use_episodic_memory
        self.forgetting_factor = forgetting_factor
        self.temperature = temperature

        # Memory storage buffers
        self.register_buffer("memory", torch.zeros(max_size, d_model))
        self.register_buffer("memory_keys", torch.zeros(max_size, d_model))  # For attention-based retrieval
        self.register_buffer("memory_values", torch.zeros(max_size, d_model))
        self.register_buffer("importance_scores", torch.zeros(max_size))  # For adaptive forgetting
        self.register_buffer("timestamps", torch.zeros(max_size, dtype=torch.long))
        self.register_buffer("access_counts", torch.zeros(max_size, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.bool))

        # Attention-based retrieval mechanism
        if use_attention_retrieval:
            self.retrieval_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            self.retrieval_norm = nn.LayerNorm(d_model)

        # Memory encoding and consolidation networks
        self.memory_encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )

        # Importance scoring network
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Episodic memory components
        if use_episodic_memory:
            self.episode_classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 16)  # 16 episode types
            )

        # Memory consolidation network
        self.consolidation_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.current_timestamp = 0

    def _encode_memory(self, features):
        """Encode features for memory storage."""
        return self.memory_encoder(features)

    def _compute_importance(self, features):
        """Compute importance scores for adaptive forgetting."""
        return self.importance_scorer(features).squeeze(-1)

    def update(self, features, mask=None, episode_context=None):
        """
        Update memory bank with new features and adaptive consolidation.

        Args:
            features: Input features [B, N, D] or [B, D] for sentence-level
            mask: Optional mask [B, N] for sequence features
            episode_context: Optional episode context for episodic memory
        """
        # Handle sequence-level or sentence-level features
        if features.dim() == 3 and mask is not None:
            # Extract sentence-level representations
            masked_features = features * mask.unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
            sent_features = masked_features.sum(dim=1) / denom
        else:
            sent_features = features if features.dim() == 2 else features.unsqueeze(0)

        batch_size = sent_features.size(0)

        # Encode memories
        encoded_memories = self._encode_memory(sent_features)

        # Compute importance scores
        importance = self._compute_importance(encoded_memories)

        # Get current pointer and update timestamp
        ptr = int(self.ptr.item())
        self.current_timestamp += 1

        # Memory storage with adaptive replacement
        if ptr + batch_size <= self.max_size:
            # Simple append case
            self.memory[ptr:ptr + batch_size] = encoded_memories.detach()
            self.memory_keys[ptr:ptr + batch_size] = encoded_memories.detach()
            self.memory_values[ptr:ptr + batch_size] = sent_features.detach()
            self.importance_scores[ptr:ptr + batch_size] = importance.detach()
            self.timestamps[ptr:ptr + batch_size] = self.current_timestamp
            self.access_counts[ptr:ptr + batch_size] = 1
            self.ptr[0] = ptr + batch_size
        else:
            # Need to replace old memories
            self._adaptive_replace(encoded_memories, sent_features, importance)
            if not self.filled[0]:
                self.filled[0] = True

        # Consolidation step
        if self.filled[0]:
            self._consolidate_memories()

    def _adaptive_replace(self, new_memories, new_values, new_importance):
        """Replace old memories based on importance and recency."""
        batch_size = new_memories.size(0)

        # Compute replacement scores (lower is better for replacement)
        recency_scores = (self.current_timestamp - self.timestamps.float()) / self.current_timestamp
        access_scores = 1.0 / (1.0 + self.access_counts.float())
        replacement_scores = (
            (1 - self.importance_scores) * 0.4 +  # Low importance
            recency_scores * 0.3 +                # Old memories
            access_scores * 0.3                   # Rarely accessed
        )

        # Find indices to replace
        _, replace_indices = replacement_scores.topk(batch_size, largest=True)

        # Replace memories
        self.memory[replace_indices] = new_memories.detach()
        self.memory_keys[replace_indices] = new_memories.detach()
        self.memory_values[replace_indices] = new_values.detach()
        self.importance_scores[replace_indices] = new_importance.detach()
        self.timestamps[replace_indices] = self.current_timestamp
        self.access_counts[replace_indices] = 1

    def _consolidate_memories(self):
        """Consolidate similar memories to prevent redundancy."""
        if not self.filled[0]:
            return

        # Compute pairwise similarities
        similarities = F.cosine_similarity(
            self.memory_keys.unsqueeze(1),
            self.memory_keys.unsqueeze(0),
            dim=-1
        )

        # Find highly similar pairs (above threshold)
        similar_mask = (similarities > self.consolidation_threshold) & (similarities < 1.0)

        if similar_mask.any():
            # Consolidate similar memories
            consolidated_indices = []
            for i in range(self.max_size):
                if i in consolidated_indices:
                    continue

                similar_indices = torch.nonzero(similar_mask[i], as_tuple=True)[0]
                if len(similar_indices) > 0:
                    # Weighted average based on importance
                    weights = self.importance_scores[[i] + similar_indices.tolist()]
                    weights = F.softmax(weights / self.temperature, dim=0)

                    consolidated_memory = torch.sum(
                        self.memory[[i] + similar_indices.tolist()] * weights.unsqueeze(1),
                        dim=0
                    )
                    consolidated_value = torch.sum(
                        self.memory_values[[i] + similar_indices.tolist()] * weights.unsqueeze(1),
                        dim=0
                    )

                    # Update primary memory
                    self.memory[i] = consolidated_memory
                    self.memory_keys[i] = consolidated_memory
                    self.memory_values[i] = consolidated_value
                    self.importance_scores[i] = weights[0] * self.importance_scores[i]

                    # Mark others for removal
                    consolidated_indices.extend(similar_indices.tolist())

    def retrieve(self, query, k=5, return_attention=False, episode_filter=None):
        """
        Retrieve k most relevant memories using attention mechanism.

        Args:
            query: Query tensor [B, D] or [D]
            k: Number of memories to retrieve
            return_attention: Whether to return attention weights
            episode_filter: Optional episode type filter

        Returns:
            Retrieved memories and optionally attention weights
        """
        if not self.filled[0] and self.ptr[0] == 0:
            return None

        # Get valid memories
        valid_size = int(self.ptr.item()) if not self.filled[0] else self.max_size
        valid_memories = self.memory[:valid_size]
        valid_keys = self.memory_keys[:valid_size]
        valid_values = self.memory_values[:valid_size]

        # Ensure query is 2D
        if query.dim() == 1:
            query = query.unsqueeze(0)

        if self.use_attention_retrieval:
            # Attention-based retrieval
            retrieved_memories, attention_weights = self.retrieval_attention(
                query, valid_keys, valid_values
            )

            # Update access counts for attended memories
            if attention_weights is not None:
                top_indices = attention_weights.topk(min(k, valid_size), dim=-1).indices
                self.access_counts[:valid_size].scatter_add_(
                    0, top_indices.flatten(), torch.ones_like(top_indices.flatten())
                )

            if return_attention:
                return retrieved_memories, attention_weights
            return retrieved_memories
        else:
            # Cosine similarity-based retrieval
            similarities = F.cosine_similarity(
                query.unsqueeze(1),
                valid_memories.unsqueeze(0),
                dim=-1
            )

            # Apply forgetting factor based on age
            age_weights = self.forgetting_factor ** (self.current_timestamp - self.timestamps[:valid_size])
            weighted_similarities = similarities * age_weights.unsqueeze(0)

            # Get top-k
            topk_values, topk_indices = weighted_similarities.topk(
                min(k, valid_size), dim=-1
            )

            # Update access counts
            self.access_counts[:valid_size].scatter_add_(
                0, topk_indices.flatten(), torch.ones_like(topk_indices.flatten())
            )

            retrieved_memories = valid_memories[topk_indices]

            if return_attention:
                return retrieved_memories, topk_values
            return retrieved_memories

    def get_memory_stats(self):
        """Get statistics about the memory bank."""
        valid_size = int(self.ptr.item()) if not self.filled[0] else self.max_size

        return {
            'memory_utilization': valid_size / self.max_size,
            'average_importance': self.importance_scores[:valid_size].mean().item(),
            'average_access_count': self.access_counts[:valid_size].float().mean().item(),
            'oldest_timestamp': self.timestamps[:valid_size].min().item(),
            'newest_timestamp': self.timestamps[:valid_size].max().item()
        }

    def clear_memory(self):
        """Clear all stored memories."""
        self.memory.zero_()
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.importance_scores.zero_()
        self.timestamps.zero_()
        self.access_counts.zero_()
        self.ptr.zero_()
        self.filled.zero_()
        self.current_timestamp = 0


def create_world_engine(config):
    """Factory function to create World Engine with configuration."""
    return WorldEngine(
        vocab_size=config.get('vocab_size', 10000),
        d_model=config.get('d_model', 512),
        k_feats=config.get('k_feats', 100),
        n_pos=config.get('n_pos', 50),
        n_rels=config.get('n_rels', 20),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 8),
        p_drop=config.get('dropout', 0.1),
        use_transformer=config.get('use_transformer', True),
        use_gnn=config.get('use_gnn', True),
        use_crf=config.get('use_crf', True),
        num_role_labels=config.get('num_role_labels', 5)
    )


# ===========================
# COMPREHENSIVE TESTING SUITE
# ===========================

def run_basic_tests():
    """Run basic functionality tests."""
    print("🧪 Running World Engine Test Suite...")

    # Test configuration
    config = {
        'vocab_size': 1000,
        'd_model': 128,
        'k_feats': 50,
        'n_pos': 25,
        'n_rels': 10,
        'n_layers': 2,
        'n_heads': 4,
        'dropout': 0.1,
        'use_transformer': True,
        'use_gnn': True,
        'use_crf': False,  # Disable CRF for basic testing
        'num_role_labels': 5
    }

    print(f"✅ Configuration: {config}")

    # Create model
    try:
        model = create_world_engine(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully with {param_count:,} parameters")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

    # Test forward pass
    try:
        batch_size = 4
        seq_len = 10

        # Create dummy input
        tok_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        pos_ids = torch.randint(0, config['n_pos'], (batch_size, seq_len))
        feat_rows = torch.randn(batch_size, seq_len, config['k_feats'])
        lengths = torch.tensor([seq_len] * batch_size)

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model.forward(tok_ids, pos_ids, feat_rows, lengths)

        # Validate outputs
        assert 'z' in outputs, "Missing semantic embeddings"
        assert 'feat_hat' in outputs, "Missing reconstructed features"
        assert 'role_logits' in outputs, "Missing role predictions"
        assert outputs['z'].shape == (batch_size, 64), f"Wrong z shape: {outputs['z'].shape}"

        print(f"✅ Forward pass successful: z={outputs['z'].shape}, feat_hat={outputs['feat_hat'].shape}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    print("🎉 All tests passed! World Engine is fully operational.")
    return True


def run_performance_benchmark():
    """Run performance benchmarks."""
    print("\n📊 Running Performance Benchmarks...")

    config = {
        'vocab_size': 10000,
        'd_model': 512,
        'k_feats': 100,
        'n_pos': 50,
        'n_rels': 20,
        'n_layers': 6,
        'n_heads': 8,
        'dropout': 0.1,
        'use_transformer': True,
        'use_gnn': True,
        'use_crf': False,
        'num_role_labels': 5
    }

    model = create_world_engine(config)
    model.eval()

    # Benchmark different batch sizes and sequence lengths
    batch_sizes = [1, 4, 16]
    seq_lengths = [32, 128, 512]

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            # Create input
            tok_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
            pos_ids = torch.randint(0, config['n_pos'], (batch_size, seq_len))
            feat_rows = torch.randn(batch_size, seq_len, config['k_feats'])
            lengths = torch.tensor([seq_len] * batch_size)

            # Time forward pass
            start_time = time.time()
            with torch.no_grad():
                outputs = model.forward(tok_ids, pos_ids, feat_rows, lengths)
            end_time = time.time()

            forward_time = (end_time - start_time) * 1000  # Convert to ms
            tokens_per_sec = (batch_size * seq_len) / (forward_time / 1000)

            print(f"📈 Batch={batch_size}, SeqLen={seq_len}: {forward_time:.2f}ms ({tokens_per_sec:.1f} tokens/sec)")


def demonstrate_capabilities():
    """Demonstrate key capabilities."""
    print("\n🚀 Demonstrating World Engine Capabilities...")

    # Create a smaller model for demonstration
    config = {
        'vocab_size': 5000, 'd_model': 256, 'k_feats': 50, 'n_pos': 25,
        'n_rels': 10, 'n_layers': 3, 'n_heads': 8, 'dropout': 0.1,
        'use_transformer': True, 'use_gnn': True, 'use_crf': False, 'num_role_labels': 5
    }

    model = create_world_engine(config)

    # Show model statistics
    print(f"\n📊 Model Statistics:")
    print(f"   🔢 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   🎯 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   💾 Model size: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    print("=" * 80)
    print("🌍 WORLD ENGINE V3.1 - ADVANCED NEURAL ARCHITECTURE")
    print("=" * 80)

    # Basic functionality tests
    if run_basic_tests():
        # Performance benchmarks
        run_performance_benchmark()

        # Capability demonstration
        demonstrate_capabilities()

        print("\n" + "=" * 80)
        print("🎉 World Engine V3.1 is fully operational!")
        print("🔗 Ready for Studio Controls integration")
        print("📊 Advanced neural architecture with complete implementation")
        print("🧠 Features: Multi-modal processing, Graph neural networks, Memory systems")
        print("=" * 80)
    else:
        print("\n❌ Tests failed - please check the implementation")
