"""
API module - FastAPI service endpoints for the World Engine.

Provides REST endpoints for:
- Word scoring and analysis
- Token-level processing
- Scale comparisons
- Lexicon queries
"""

from .service import create_app, WordEngineAPI

__all__ = ['create_app', 'WordEngineAPI']
