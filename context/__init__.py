"""
Context module - Handles natural language processing and contextual analysis.

This module provides:
- Text parsing and linguistic analysis
- Rule-based processing for negation, intensifiers, etc.
- Sense disambiguation and sarcasm detection
- Domain adaptation capabilities
- Token-level scoring with explanations
"""

from .parser import TextParser
from .rules import RuleProcessor
from .scorer import ContextScorer

__all__ = ['TextParser', 'RuleProcessor', 'ContextScorer']
