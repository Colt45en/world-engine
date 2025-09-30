"""Seed management and lexical constants for World Engine scales."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

__all__ = [
    "SeedManager",
    "DEFAULT_SEEDS",
    "DEFAULT_CONSTRAINTS",
    "INTENSIFIERS",
    "REVERSERS_LIMITERS",
    "NEGATORS",
    "DIRECTION_LEMMAS",
]


# Default seed examples based on the original project corpus.
DEFAULT_SEEDS: Dict[str, float] = {
    "terrible": -0.8,
    "bad": -0.5,
    "poor": -0.3,
    "neutral": 0.0,
    "good": 0.3,
    "great": 0.6,
    "excellent": 0.8,
    "amazing": 0.9,
}

DEFAULT_CONSTRAINTS: List[Tuple[str, str]] = [
    ("terrible", "bad"),
    ("bad", "poor"),
    ("poor", "neutral"),
    ("neutral", "good"),
    ("good", "great"),
    ("great", "excellent"),
    ("excellent", "amazing"),
]

# Lexical constants reused by the context rules/scorer pipeline.
INTENSIFIERS: Dict[str, float] = {
    "very": 1.5,
    "extremely": 1.8,
    "incredibly": 1.9,
    "super": 1.4,
    "quite": 1.2,
    "rather": 1.15,
    "slightly": 0.8,
    "somewhat": 0.9,
    "barely": 0.6,
    "hardly": 0.65,
}

REVERSERS_LIMITERS: Dict[str, float] = {
    "only": 0.85,
    "just": 0.9,
    "merely": 0.85,
    "almost": 0.75,
    "nearly": 0.75,
}

NEGATORS: set[str] = {"not", "never", "no", "hardly", "scarcely", "barely"}

DIRECTION_LEMMAS: Dict[str, float] = {
    "up": 1.0,
    "increase": 1.0,
    "boost": 1.25,
    "down": -1.0,
    "decrease": -1.0,
    "reduce": -1.1,
}


class SeedManager:
    """Manages semantic seeds and pairwise ordering constraints."""

    def __init__(self, seed_file: Optional[str] = None):
        self.seeds: Dict[str, float] = {}
        self.constraints: List[Tuple[str, str]] = []
        self.graph_cache: Dict[str, Dict] = {}

        if seed_file:
            self.load_seeds(seed_file)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add_seed(self, word: str, value: float) -> None:
        self.seeds[word] = float(value)
        self._invalidate_cache()

    def add_constraint(self, word1: str, word2: str) -> None:
        constraint = (word1, word2)
        if constraint not in self.constraints:
            self.constraints.append(constraint)
            self._invalidate_cache()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_seed_value(self, word: str) -> Optional[float]:
        return self.seeds.get(word)

    def is_seed(self, word: str) -> bool:
        return word in self.seeds

    def get_constraints_for_word(self, word: str) -> List[Tuple[str, str]]:
        return [c for c in self.constraints if word in c]

    def validate_constraints(self) -> List[str]:
        errors: List[str] = []
        for word1, word2 in self.constraints:
            val1 = self.seeds.get(word1)
            val2 = self.seeds.get(word2)
            if val1 is not None and val2 is not None and val1 >= val2:
                errors.append(
                    f"Constraint violation: {word1} ({val1}) >= {word2} ({val2})"
                )
        return errors

    def get_seed_range(self) -> Tuple[float, float]:
        if not self.seeds:
            return (0.0, 1.0)
        values = list(self.seeds.values())
        return (min(values), max(values))

    def get_stats(self) -> Dict[str, Any]:
        if not self.seeds:
            return {"num_seeds": 0, "num_constraints": len(self.constraints)}
        values = list(self.seeds.values())
        return {
            "num_seeds": len(self.seeds),
            "num_constraints": len(self.constraints),
            "value_range": (min(values), max(values)),
            "mean_value": sum(values) / len(values),
            "constraint_violations": len(self.validate_constraints()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load_seeds(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.seeds = {k: float(v) for k, v in data.get("seeds", {}).items()}
        self.constraints = [tuple(c) for c in data.get("constraints", [])]
        self._invalidate_cache()

    def save_seeds(self, filepath: str) -> None:
        data = {
            "seeds": self.seeds,
            "constraints": self.constraints,
            "metadata": {
                "num_seeds": len(self.seeds),
                "num_constraints": len(self.constraints),
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    def _invalidate_cache(self) -> None:
        self.graph_cache.clear()
