"""
Seed Manager - Handles tiny hand seeds and pairwise order constraints.

Manages the foundational semantic relationships that bootstrap the entire system.
"""

import json
from typing import Dict, List, Tuple, Set
from pathlib import Path


class SeedManager:
    """Manages semantic seeds and pairwise ordering constraints."""

    def __init__(self, seed_file: str = None):
        self.seeds: Dict[str, float] = {}
        self.constraints: List[Tuple[str, str]] = []  # (word1, word2) where word1 < word2
        self.graph_cache: Dict = {}

        if seed_file:
            self.load_seeds(seed_file)

    def add_seed(self, word: str, value: float):
        """Add a hand-labeled seed with its semantic value."""
        self.seeds[word] = value
        self._invalidate_cache()

    def add_constraint(self, word1: str, word2: str):
        """Add ordering constraint: word1 < word2 semantically."""
        constraint = (word1, word2)
        if constraint not in self.constraints:
            self.constraints.append(constraint)
            self._invalidate_cache()

    def get_seed_value(self, word: str) -> float:
        """Get the seed value for a word, or None if not a seed."""
        return self.seeds.get(word)

    def is_seed(self, word: str) -> bool:
        """Check if word is a hand-labeled seed."""
        return word in self.seeds

    def get_constraints_for_word(self, word: str) -> List[Tuple[str, str]]:
        """Get all constraints involving the given word."""
        return [c for c in self.constraints if word in c]

    def validate_constraints(self) -> List[str]:
        """Check for constraint violations and return list of errors."""
        errors = []

        for word1, word2 in self.constraints:
            val1 = self.seeds.get(word1)
            val2 = self.seeds.get(word2)

            if val1 is not None and val2 is not None:
                if val1 >= val2:
                    errors.append(f"Constraint violation: {word1} ({val1}) >= {word2} ({val2})")

        return errors

    def load_seeds(self, filepath: str):
        """Load seeds and constraints from JSON file."""
        path = Path(filepath)
        if not path.exists():
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.seeds = data.get('seeds', {})
        self.constraints = [tuple(c) for c in data.get('constraints', [])]
        self._invalidate_cache()

    def save_seeds(self, filepath: str):
        """Save seeds and constraints to JSON file."""
        data = {
            'seeds': self.seeds,
            'constraints': self.constraints,
            'metadata': {
                'num_seeds': len(self.seeds),
                'num_constraints': len(self.constraints)
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _invalidate_cache(self):
        """Clear any cached computations when seeds/constraints change."""
        self.graph_cache.clear()

    def get_seed_range(self) -> Tuple[float, float]:
        """Get the min/max range of seed values."""
        if not self.seeds:
            return (0.0, 1.0)

        values = list(self.seeds.values())
        return (min(values), max(values))

    def get_stats(self) -> Dict:
        """Get statistics about the seed collection."""
        if not self.seeds:
            return {'num_seeds': 0, 'num_constraints': len(self.constraints)}

        values = list(self.seeds.values())
        return {
            'num_seeds': len(self.seeds),
            'num_constraints': len(self.constraints),
            'value_range': (min(values), max(values)),
            'mean_value': sum(values) / len(values),
            'constraint_violations': len(self.validate_constraints())
        }


# Default seed examples based on your project
DEFAULT_SEEDS = {
    'terrible': -0.8,
    'bad': -0.5,
    'poor': -0.3,
    'neutral': 0.0,
    'good': 0.3,
    'great': 0.6,
    'excellent': 0.8,
    'amazing': 0.9
}

DEFAULT_CONSTRAINTS = [
    ('terrible', 'bad'),
    ('bad', 'poor'),
    ('poor', 'neutral'),
    ('neutral', 'good'),
    ('good', 'great'),
    ('great', 'excellent'),
    ('excellent', 'amazing')
]
