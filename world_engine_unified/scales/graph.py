"""Synonym/antonym graph utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

__all__ = ["SynonymGraph", "GraphEdge"]


@dataclass
class GraphEdge:
    u: str
    v: str
    w: float
    kind: str  # "syn" or "ant"


class SynonymGraph:
    """Lightweight synonym/antonym graph structure."""

    def __init__(self) -> None:
        self.syn: List[GraphEdge] = []
        self.ant: List[GraphEdge] = []
        self._nodes: Set[str] = set()

    # ------------------------------------------------------------------
    def add_synonym(self, u: str, v: str, weight: float = 1.0) -> None:
        w = float(max(0.0, weight))
        self.syn.append(GraphEdge(u, v, w, "syn"))
        self._nodes.update([u, v])

    def add_antonym(self, u: str, v: str, weight: float = 1.0) -> None:
        w = float(max(0.0, weight))
        self.ant.append(GraphEdge(u, v, w, "ant"))
        self._nodes.update([u, v])

    # ------------------------------------------------------------------
    def nodes(self) -> Set[str]:
        return set(self._nodes)

    def neighbors(self, word: str, kind: str | None = None) -> List[Tuple[str, float, str]]:
        out: List[Tuple[str, float, str]] = []
        if kind in (None, "syn"):
            for edge in self.syn:
                if edge.u == word:
                    out.append((edge.v, edge.w, "syn"))
                elif edge.v == word:
                    out.append((edge.u, edge.w, "syn"))
        if kind in (None, "ant"):
            for edge in self.ant:
                if edge.u == word:
                    out.append((edge.v, edge.w, "ant"))
                elif edge.v == word:
                    out.append((edge.u, edge.w, "ant"))
        return out
