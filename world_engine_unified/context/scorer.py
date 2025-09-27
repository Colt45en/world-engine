"""Token-level scoring utilities combining seeds and context rules."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .parser import TextParser, Token
from .rules import ContextSignals, compute_context_signals
from scales.seeds import SeedManager

__all__ = ["TokenScore", "TokenScorer"]


@dataclass(frozen=True)
class TokenScore:
    token: str
    lemma: str
    base_value: Optional[float]
    adjusted_value: float
    negated: bool
    intensity_multiplier: float
    direction_bias: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TokenScorer:
    """Scores tokens by combining seed values with context heuristics."""

    def __init__(self, seed_manager: SeedManager, parser: Optional[TextParser] = None) -> None:
        self.seed_manager = seed_manager
        self.parser = parser or TextParser()

    def _window(self, tokens: List[Token], index: int, radius: int = 2) -> List[Token]:
        lo = max(0, index - radius)
        hi = min(len(tokens), index + radius + 1)
        window = tokens[lo:index] + tokens[index + 1 : hi]
        return window

    def score_text(self, text: str) -> Dict[str, Any]:
        parsed = self.parser.parse(text)
        scores: List[TokenScore] = []
        for idx, tok in enumerate(parsed.tokens):
            if not tok.is_alpha:
                continue
            lemma = tok.lemma.lower()
            base = self.seed_manager.get_seed_value(lemma)
            if base is None:
                # Use neutral baseline when no seed value present
                base = 0.0
            window = self._window(parsed.tokens, idx)
            signals: ContextSignals = compute_context_signals(tok, window)

            adjusted = base * signals.intensity_multiplier
            if signals.negated:
                adjusted = -adjusted
            adjusted += signals.direction_bias * 0.05  # small directional nudge

            scores.append(
                TokenScore(
                    token=tok.text,
                    lemma=lemma,
                    base_value=base,
                    adjusted_value=adjusted,
                    negated=signals.negated,
                    intensity_multiplier=signals.intensity_multiplier,
                    direction_bias=signals.direction_bias,
                )
            )

        return {
            "text": text,
            "tokens": [score.to_dict() for score in scores],
            "summary": {
                "scored_tokens": len(scores),
                "avg_score": sum(s.adjusted_value for s in scores) / len(scores) if scores else 0.0,
                "negated_tokens": sum(1 for s in scores if s.negated),
            },
        }
