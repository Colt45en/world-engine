"""Contextual linguistic rules applied during token scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .parser import Token
from scales.seeds import (
    DIRECTION_LEMMAS,
    INTENSIFIERS,
    NEGATORS,
    REVERSERS_LIMITERS,
)

__all__ = ["ContextSignals", "compute_context_signals"]


@dataclass(frozen=True)
class ContextSignals:
    negated: bool
    intensity_multiplier: float
    direction_bias: float


def _accumulate_multiplier(window: Iterable[Token]) -> float:
    mult = 1.0
    for tok in window:
        lemma = tok.lemma.lower()
        if lemma in INTENSIFIERS:
            mult *= INTENSIFIERS[lemma]
        if lemma in REVERSERS_LIMITERS:
            mult *= REVERSERS_LIMITERS[lemma]
    return mult


def _detect_negation(token: Token, window: Iterable[Token]) -> bool:
    if token.dep == "neg" or token.lemma.lower() in NEGATORS:
        return True
    for tok in window:
        if tok.dep == "neg" or tok.lemma.lower() in NEGATORS:
            return True
    return False


def _direction_bias(window: Iterable[Token]) -> float:
    bias = 0.0
    for tok in window:
        bias += DIRECTION_LEMMAS.get(tok.lemma.lower(), 0.0)
    return bias


def compute_context_signals(token: Token, window: Sequence[Token]) -> ContextSignals:
    """Compute negation/intensity/direction signals for a token."""
    negated = _detect_negation(token, window)
    multiplier = _accumulate_multiplier(window)
    bias = _direction_bias(window)
    return ContextSignals(negated=negated, intensity_multiplier=multiplier, direction_bias=bias)
