"""World Engine Unified Package

A unified lexicon processing system combining mathematical morpheme analysis,
semantic scaling, and advanced NLP processing.
"""

__version__ = "3.1.0"
__author__ = "World Engine Team"

from .scales.seeds import SeedManager, DEFAULT_SEEDS, DEFAULT_CONSTRAINTS

__all__ = [
    'SeedManager',
    'DEFAULT_SEEDS',
    'DEFAULT_CONSTRAINTS'
]
