"""
World Engine - Unified Lexicon Processing System

A comprehensive system for lexicon analysis, semantic scaling, and interactive exploration.
Combines Python-based processing with web-based interfaces for maximum flexibility.
"""

__version__ = "0.1.0"
__author__ = "World Engine Project"

from .scales import seeds, graph, isotonic, embeddings, typescores
from .context import parser, rules, scorer
from .api import service

__all__ = [
    'seeds', 'graph', 'isotonic', 'embeddings', 'typescores',
    'parser', 'rules', 'scorer', 'service'
]
