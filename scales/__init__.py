"""
Scales module - Handles semantic scaling and ordering operations.

This module provides tools for:
- Seed-based word relationship mapping
- Synonym/antonym graph construction
- Isotonic regression for order-preserving calibration
- Embedding-based neighbor expansion
- Type-level lexicon vector generation
"""

from .seeds import SeedManager
from .graph import SynonymGraph
from .isotonic import IsotonicCalibrator
from .embeddings import EmbeddingExpander
from .typescores import TypeScorer

__all__ = ['SeedManager', 'SynonymGraph', 'IsotonicCalibrator', 'EmbeddingExpander', 'TypeScorer']
