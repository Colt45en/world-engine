"""Compatibility shim keeping top-level imports stable after relocation."""

from src.python import tier4_graph_encoders as _impl
from src.python.tier4_graph_encoders import *  # noqa: F401,F403

__all__ = [
    "GraphBatch",
    "mask_from_lengths",
    "flatten_padded",
    "build_batched_edges",
    "RelGraphEncoderTG",
    "RelGATv2EncoderTG",
]


def __getattr__(name: str):
    return getattr(_impl, name)
