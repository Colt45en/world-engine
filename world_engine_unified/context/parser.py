"""Compatibility shim exposing ``TextParser`` from the legacy module name."""

from .__init_praser__ import (  # noqa: F401
	ParsedSentence,
	ParserConfig,
	ParserWarning,
	TextParser,
	Token,
)

__all__ = [
	"TextParser",
	"ParserConfig",
	"ParserWarning",
	"Token",
	"ParsedSentence",
]
