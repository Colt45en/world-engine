"""
FastAPI Service - REST endpoints for the World Engine.

Provides HTTP API access to lexicon processing, scoring, and analysis capabilities.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path

from ..scales.seeds import SeedManager
from ..context.parser import TextParser


class WordRequest(BaseModel):
    """Request model for word analysis."""
    word: str
    context: Optional[str] = None


class TokenRequest(BaseModel):
    """Request model for token analysis."""
    text: str
    options: Optional[Dict[str, Any]] = {}


class ScaleRequest(BaseModel):
    """Request model for scale operations."""
    word1: str
    word2: str
    scale_type: Optional[str] = "semantic"


class WordEngineAPI:
    """Core API class for World Engine operations."""

    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / "config"
        self.seed_manager = SeedManager()
        self.text_parser = TextParser()
        self._load_configuration()

    def _load_configuration(self):
        """Load configuration files."""
        # Load seeds if available
        seed_file = self.config_dir / "seeds.json"
        if seed_file.exists():
            self.seed_manager.load_seeds(str(seed_file))

    def score_word(self, word: str, context: str = None) -> Dict[str, Any]:
        """Score a word with optional context."""
        # Check if it's a seed word
        seed_value = self.seed_manager.get_seed_value(word)

        result = {
            "word": word,
            "is_seed": seed_value is not None,
            "seed_value": seed_value,
            "context_provided": context is not None
        }

        # If context provided, parse it
        if context:
            parsed = self.text_parser.parse(context)
            # Find the word in context
            word_tokens = [t for t in parsed.tokens if t.lemma.lower() == word.lower()]

            if word_tokens:
                token = word_tokens[0]
                result.update({
                    "pos": token.pos,
                    "dependency": token.dep,
                    "head": token.head_text,
                    "children": token.children
                })

        return result

    def score_token(self, text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze and score tokens in text."""
        if options is None:
            options = {}

        parsed = self.text_parser.parse(text)

        token_scores = []
        for token in parsed.tokens:
            if token.is_alpha and not token.is_stop:
                seed_value = self.seed_manager.get_seed_value(token.lemma.lower())

                token_scores.append({
                    "text": token.text,
                    "lemma": token.lemma,
                    "pos": token.pos,
                    "is_seed": seed_value is not None,
                    "seed_value": seed_value,
                    "dependency": token.dep
                })

        return {
            "text": text,
            "tokens": token_scores,
            "entities": parsed.entities,
            "noun_chunks": parsed.noun_chunks,
            "summary": {
                "total_tokens": len(parsed.tokens),
                "scored_tokens": len(token_scores),
                "entities_found": len(parsed.entities)
            }
        }

    def scale_between(self, word1: str, word2: str, scale_type: str = "semantic") -> Dict[str, Any]:
        """Compare two words on a scale."""
        val1 = self.seed_manager.get_seed_value(word1)
        val2 = self.seed_manager.get_seed_value(word2)

        result = {
            "word1": word1,
            "word2": word2,
            "scale_type": scale_type,
            "word1_value": val1,
            "word2_value": val2
        }

        if val1 is not None and val2 is not None:
            result.update({
                "difference": val2 - val1,
                "comparison": "equal" if val1 == val2 else ("word1_higher" if val1 > val2 else "word2_higher")
            })

        return result


def create_app(config_dir: str = None) -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="World Engine API",
        description="Lexicon processing and semantic analysis API",
        version="0.1.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize API
    api = WorldEngineAPI(config_dir)

    # Mount static files (for web interfaces)
    web_dir = Path(__file__).parent.parent / "web"
    if web_dir.exists():
        app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

    @app.get("/")
    async def root():
        """API root endpoint."""
        return {
            "message": "World Engine API",
            "version": "0.1.0",
            "endpoints": [
                "/score_word",
                "/score_token",
                "/scale_between",
                "/health",
                "/web",
                "/studio"
            ],
            "interfaces": {
                "studio": "/web/studio.html",
                "lexicon": "/web/worldengine.html",
                "api_docs": "/docs"
            }
        }

    @app.get("/studio")
    async def studio_redirect():
        """Redirect to the studio interface."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/web/studio.html")    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "seed_count": len(api.seed_manager.seeds),
            "parser_model": api.text_parser.model_name
        }

    @app.post("/score_word")
    async def score_word(request: WordRequest):
        """Score a single word with optional context."""
        try:
            result = api.score_word(request.word, request.context)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/score_token")
    async def score_token(request: TokenRequest):
        """Analyze and score tokens in text."""
        try:
            result = api.score_token(request.text, request.options)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/scale_between")
    async def scale_between(request: ScaleRequest):
        """Compare two words on a semantic scale."""
        try:
            result = api.scale_between(request.word1, request.word2, request.scale_type)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/seeds")
    async def get_seeds():
        """Get all seed words and their values."""
        return {
            "seeds": api.seed_manager.seeds,
            "constraints": api.seed_manager.constraints,
            "stats": api.seed_manager.get_stats()
        }

    return app


# For running directly
if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
