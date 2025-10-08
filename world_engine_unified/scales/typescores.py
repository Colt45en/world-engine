"""Heuristic type-level scoring utilities."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import numpy as np

from .graph import SynonymGraph
from .isotonic import IsotonicCalibrator
from .seeds import SeedManager

__all__ = ["ScoreConfig", "TypeScorer"]


@dataclass
class ScoreConfig:
    y_min: float = -1.0
    y_max: float = 1.0
    syn_smooth_steps: int = 2
    syn_strength: float = 0.5
    proj_iters: int = 1000
    proj_tol: float = 1e-9


class TypeScorer:
    """Dependency-light scorer that enforces lexical ordering heuristics."""

    def __init__(
        self,
        seed_mgr: SeedManager,
        graph: SynonymGraph,
        calib: IsotonicCalibrator,
        cfg: ScoreConfig | None = None,
    ) -> None:
        self.seed_mgr = seed_mgr
        self.graph = graph
        self.calib = calib
        self.cfg = cfg or ScoreConfig()

    # ------------------------------------------------------------------
    def _vocab(self) -> List[str]:
        vocab: Set[str] = set(self.seed_mgr.seeds.keys())
        vocab.update([w for pair in self.seed_mgr.constraints for w in pair])
        vocab.update(self.graph.nodes())
        return sorted(vocab)

    @staticmethod
    def _topo_from_pairs(pairs: Sequence[Tuple[str, str]]) -> List[str]:
        adj: Dict[str, List[str]] = defaultdict(list)
        indeg: Dict[str, int] = defaultdict(int)
        nodes: Set[str] = set()
        for a, b in pairs:
            nodes.add(a)
            nodes.add(b)
            adj[a].append(b)
            indeg[b] += 1
            indeg.setdefault(a, 0)
        queue = deque([node for node in nodes if indeg[node] == 0])
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in adj.get(node, []):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        for node in nodes:
            if node not in order:
                order.append(node)
        return order

    def _project_pairs(self, scores: Dict[str, float]) -> Dict[str, float]:
        adjusted = dict(scores)
        pairs = [pair for pair in self.seed_mgr.constraints if pair[0] in adjusted and pair[1] in adjusted]
        if not pairs:
            return adjusted
        for _ in range(self.cfg.proj_iters):
            max_violation = 0.0
            for a, b in pairs:
                if adjusted[a] > adjusted[b]:
                    avg = 0.5 * (adjusted[a] + adjusted[b])
                    adjusted[a] = adjusted[b] = avg
                    max_violation = max(max_violation, adjusted[a] - adjusted[b])
            if max_violation <= self.cfg.proj_tol:
                break
        lo, hi = self.cfg.y_min, self.cfg.y_max
        for key in adjusted:
            adjusted[key] = float(np.clip(adjusted[key], lo, hi))
        return adjusted

    def _syn_smooth(self, scores: Dict[str, float]) -> Dict[str, float]:
        smoothed = dict(scores)
        for _ in range(self.cfg.syn_smooth_steps):
            delta: Dict[str, float] = defaultdict(float)
            weight: Dict[str, float] = defaultdict(float)
            for edge in self.graph.syn:
                a, b, w = edge.u, edge.v, edge.w
                if a in smoothed and b in smoothed and w > 0:
                    avg = 0.5 * (smoothed[a] + smoothed[b])
                    delta[a] += w * (avg - smoothed[a])
                    delta[b] += w * (avg - smoothed[b])
                    weight[a] += w
                    weight[b] += w
            for key, d in delta.items():
                if weight[key] > 0:
                    smoothed[key] += self.cfg.syn_strength * (d / weight[key])
        return smoothed

    # ------------------------------------------------------------------
    def compute(self) -> Dict[str, float]:
        vocab = self._vocab()
        if not vocab:
            return {}
        lo, hi = self.cfg.y_min, self.cfg.y_max
        if self.seed_mgr.seeds:
            base = sum(self.seed_mgr.seeds.values()) / len(self.seed_mgr.seeds)
        else:
            base = 0.0
        scores: Dict[str, float] = {word: float(base) for word in vocab}
        scores.update({w: float(np.clip(v, lo, hi)) for w, v in self.seed_mgr.seeds.items()})

        scores = self._syn_smooth(scores)
        scores = self._project_pairs(scores)
        topo = self._topo_from_pairs(self.seed_mgr.constraints)
        scores = self.calib.fit_transform(topo, scores)
        for key in scores:
            scores[key] = float(np.clip(scores[key], lo, hi))
        return scores
