"""Embedding-based neighbor expansion for scales."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import numpy as np

__all__ = ["EmbeddingExpander"]


class EmbeddingExpander:
    """Expands a vocabulary using cosine-similarity neighbors."""

    def __init__(self, embeddings: Dict[str, np.ndarray]):
        self.emb: Dict[str, np.ndarray] = {}
        for key, vec in embeddings.items():
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr)) + 1e-12
            self.emb[key] = arr / norm

    def nearest(self, query: str, k: int = 10, min_sim: float = 0.0) -> List[Tuple[str, float]]:
        if query not in self.emb:
            return []
        q = self.emb[query]
        sims: List[Tuple[str, float]] = []
        for word, vec in self.emb.items():
            if word == query:
                continue
            score = float(np.dot(q, vec))
            if score >= min_sim:
                sims.append((word, score))
        sims.sort(key=lambda item: -item[1])
        return sims[:k]

    def expand(self, seeds: Iterable[str], k_each: int = 5, min_sim: float = 0.3) -> Dict[str, List[Tuple[str, float]]]:
        out: Dict[str, List[Tuple[str, float]]] = {}
        for seed in seeds:
            out[seed] = self.nearest(seed, k=k_each, min_sim=min_sim)
        return out
