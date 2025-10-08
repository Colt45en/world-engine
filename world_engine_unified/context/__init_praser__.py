"""
Text Parser - spaCy-based pipeline for tokenization, POS tagging, and dependency parsing.

Tier-4 IDE Upgrade:
- Strong typing & dataclasses with helpful reprs and .to_dict()
- Config object with GPU toggle, n_process, batch_size, cache size, etc.
- Structured logging + timing
- Graceful fallbacks if spaCy model/pipes are missing
- Batch parsing with nlp.pipe (parallel via n_process)
- Result caching (LRU-style) to avoid re-parsing identical texts
- Feature extraction (distributions + ratios + basic readability)
- Optional pandas DataFrame export (if pandas installed)
- CLI for quick ad-hoc parsing to JSON/CSV
"""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import spacy
except Exception:  # pragma: no cover
    spacy = None

if TYPE_CHECKING:
    try:
        from spacy.language import Language
        from spacy.tokens import Doc
    except Exception:  # pragma: no cover
        Language = Any  # type: ignore
        Doc = Any  # type: ignore
else:
    Language = Any  # type: ignore
    Doc = Any  # type: ignore

__all__ = [
    "ParserConfig",
    "Token",
    "ParsedSentence",
    "TextParser",
    "__version__",
]

__version__ = "1.4.0"

_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "not",
    "no",
    "never",
}


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("text_parser")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Token:
    """Represents a parsed token with linguistic features.

    Example
    -------
    >>> t = Token("Cats", "cat", "NOUN", "NNS", "nsubj", True, False, False, "purr", ["do"])
    >>> t.text, t.lemma
    ('Cats', 'cat')
    """
    text: str
    lemma: str
    pos: str
    tag: str
    dep: str
    is_alpha: bool
    is_stop: bool
    is_punct: bool
    head_text: str
    children: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedSentence:
    """Represents a parsed sentence with tokens and structure."""
    text: str
    tokens: List[Token]
    entities: List[Dict[str, Any]]
    noun_chunks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
            "entities": self.entities,
            "noun_chunks": self.noun_chunks,
        }


@dataclass
class ParserConfig:
    """Runtime configuration for TextParser."""
    model: str = "en_core_web_sm"
    use_gpu: bool = False                 # call spacy.require_gpu if True
    n_process: int = 1                    # parallel workers for nlp.pipe
    batch_size: int = 1000
    cache_size: int = 512                 # LRU cache entries
    log_timing: bool = True
    # If model isn't present, we build a lightweight blank('en') with sentencizer.
    # POS/DEP/NER will then be missing; we fall back gracefully.


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class ParserWarning(RuntimeWarning):
    """Non-fatal issues (missing model/components)."""


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _has_component(nlp: Language, name: str) -> bool:
    try:
        return nlp.has_pipe(name)
    except Exception:
        return False


def _iterable_len(x: Iterable[Any]) -> Optional[int]:
    try:
        return len(x)  # type: ignore
    except TypeError:
        return None


# -----------------------------------------------------------------------------
# Text Parser
# -----------------------------------------------------------------------------
class TextParser:
    """spaCy-based text parser for linguistic analysis with robust fallbacks.

    Notes
    -----
    - If `model` cannot be loaded, we fall back to `spacy.blank("en")` + sentencizer.
      In that mode, POS/DEP/NER/noun_chunks may be unavailable; methods degrade gracefully.
    - For speed/scalability, prefer `parse_batch()` which uses `nlp.pipe`.
    """

    _has_ner: bool
    _has_parser: bool
    _has_tagger: bool

    def __init__(self, config: ParserConfig | None = None):
        self.cfg = config or ParserConfig()
        self.nlp = None

        if spacy is None:
            logger.warning(
                "spaCy is not installed; falling back to lightweight whitespace tokenizer."
            )
            self._has_ner = False
            self._has_parser = False
            self._has_tagger = False
        else:
            if self.cfg.use_gpu:
                try:
                    require_gpu = getattr(spacy, "require_gpu", None)
                    if callable(require_gpu):
                        require_gpu()
                    logger.info("Using GPU for spaCy.")
                except Exception as e:
                    logger.warning(
                        "GPU requested but not available; falling back to CPU. (%s)", e
                    )

            try:
                self.nlp = spacy.load(self.cfg.model)
                logger.info("Loaded spaCy model: %s", self.cfg.model)
            except OSError:
                logger.warning(
                    "spaCy model '%s' not found. Using blank('en') with sentencizer only.",
                    self.cfg.model,
                    exc_info=False,
                )
                self.nlp = spacy.blank("en")
                if not _has_component(self.nlp, "sentencizer"):
                    self.nlp.add_pipe("sentencizer")

            self._has_ner = _has_component(self.nlp, "ner")
            self._has_parser = _has_component(self.nlp, "parser")
            self._has_tagger = _has_component(self.nlp, "tagger")

            if not self._has_parser:
                logger.debug(
                    "Dependency parser not available; noun_chunks/dep will be partial."
                )

        # Tiny LRU cache: text -> ParsedSentence
        self._cache = OrderedDict()

    # ------------------------- Core API -------------------------

    def parse(self, text: str) -> ParsedSentence:
        """Parse a single text into a structured `ParsedSentence` (cached)."""
        key = _hash_text(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        t0 = time.perf_counter()
        if self.nlp is None:
            parsed = self._fallback_parse(text)
        else:
            doc = self.nlp(text)
            parsed = self._doc_to_parsed(doc)
        t1 = time.perf_counter()

        if self.cfg.log_timing:
            logger.debug("Parsed 1 doc in %.2f ms", (t1 - t0) * 1000)

        # LRU insert
        self._cache[key] = parsed
        if len(self._cache) > self.cfg.cache_size:
            self._cache.popitem(last=False)

        return parsed

    def parse_batch(self, texts: Sequence[str] | Iterable[str]) -> List[ParsedSentence]:
        """Parse multiple texts efficiently with spaCy's `nlp.pipe`.

        Uses `n_process` workers if supported by the pipeline.
        """
        results: List[ParsedSentence] = []
        total = _iterable_len(texts) or 0

        t0 = time.perf_counter()
        if self.nlp is None:
            for text in texts:
                results.append(self._fallback_parse(text))
        else:
            for doc in self.nlp.pipe(
                texts,
                n_process=self.cfg.n_process,
                batch_size=self.cfg.batch_size,
            ):
                results.append(self._doc_to_parsed(doc))
        t1 = time.perf_counter()

        if self.cfg.log_timing:
            msg = f"Parsed {len(results)} docs"
            if total:
                msg += f"/{total}"
            msg += f" in {t1 - t0:.2f}s"
            logger.info(msg)

        return results

    def _fallback_parse(self, text: str) -> ParsedSentence:
        tokens: List[Token] = []
        for raw in text.split():
            word = raw.strip()
            if not word:
                continue
            lemma = word.lower()
            is_alpha = lemma.isalpha()
            tokens.append(
                Token(
                    text=word,
                    lemma=lemma,
                    pos="",
                    tag="",
                    dep="",
                    is_alpha=is_alpha,
                    is_stop=lemma in _FALLBACK_STOPWORDS,
                    is_punct=not is_alpha,
                    head_text="",
                    children=[],
                )
            )
        return ParsedSentence(text=text, tokens=tokens, entities=[], noun_chunks=[])

    # ----------------------- Feature helpers -----------------------

    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive linguistic features for one text."""
        parsed = self.parse(text)

        pos_counts: Dict[str, int] = {}
        dep_counts: Dict[str, int] = {}

        alpha_count = 0
        stop_count = 0
        punct_count = 0
        total_chars = len(text)

        for tok in parsed.tokens:
            pos_counts[tok.pos] = pos_counts.get(tok.pos, 0) + 1
            dep_counts[tok.dep] = dep_counts.get(tok.dep, 0) + 1
            alpha_count += int(tok.is_alpha)
            stop_count += int(tok.is_stop)
            punct_count += int(tok.is_punct)

        token_count = len(parsed.tokens)
        unique_tokens = len({t.text.lower() for t in parsed.tokens if t.is_alpha})
        avg_token_len = (
            sum(len(t.text) for t in parsed.tokens) / token_count if token_count else 0.0
        )

        # Basic readability proxy (very rough): letters per token & token per entity
        letters = sum(c.isalpha() for c in text)
        readability = (letters / max(1, token_count)) if token_count else 0.0

        return {
            "text": text,
            "token_count": token_count,
            "entity_count": len(parsed.entities),
            "noun_chunk_count": len(parsed.noun_chunks),
            "pos_distribution": pos_counts,
            "dependency_distribution": dep_counts,
            "entities": parsed.entities,
            "noun_chunks": parsed.noun_chunks,
            "alpha_ratio": (alpha_count / token_count) if token_count else 0.0,
            "stop_ratio": (stop_count / token_count) if token_count else 0.0,
            "punct_ratio": (punct_count / token_count) if token_count else 0.0,
            "avg_token_len": avg_token_len,
            "ttr": (unique_tokens / token_count) if token_count else 0.0,  # type-token ratio
            "char_count": total_chars,
            "readability_proxy": readability,
            "model": self.cfg.model,
        }

    def get_lemmas(self, text: str, unique: bool = False) -> List[str]:
        """Extract lemmatized tokens from text (letters only)."""
        parsed = self.parse(text)
        lemmas = [t.lemma for t in parsed.tokens if t.is_alpha]
        if unique:
            # preserve order of first occurrence
            seen, out = set(), []
            for l in lemmas:
                if l not in seen:
                    seen.add(l); out.append(l)
            return out
        return lemmas

    def get_dependencies(self, text: str) -> List[Dict[str, str]]:
        """Extract dependency relationships (if available)."""
        parsed = self.parse(text)
        deps = []
        for t in parsed.tokens:
            if t.dep and t.dep != "ROOT":
                deps.append({"token": t.text, "head": t.head_text, "relation": t.dep})
        return deps

    # ----------------------- Export helpers -----------------------

    def to_dataframe(self, parsed: ParsedSentence):
        """Return a pandas DataFrame (if pandas is installed) of token rows."""
        try:
            import pandas as pd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pandas is required for to_dataframe()") from e

        rows = [t.to_dict() for t in parsed.tokens]
        return pd.DataFrame(rows)

    # ----------------------- Internal conversion -----------------------

    def _doc_to_parsed(self, doc) -> ParsedSentence:
        """Convert spaCy Doc to our parsed structure with graceful fallbacks."""
        # Tokens
        tokens: List[Token] = []
        has_dep = self._has_parser
        has_lemma = True  # blank models still produce lemma_="", we'll handle defaulting

        for tok in doc:
            lemma = tok.lemma_ if tok.lemma_ else tok.text
            pos = tok.pos_ if tok.pos_ else ""
            tag = tok.tag_ if tok.tag_ else ""
            dep = tok.dep_ if has_dep else ""
            head_text = tok.head.text if has_dep else ""

            children = [c.text for c in tok.children] if has_dep else []

            tokens.append(
                Token(
                    text=tok.text,
                    lemma=lemma,
                    pos=pos,
                    tag=tag,
                    dep=dep,
                    is_alpha=bool(tok.is_alpha),
                    is_stop=bool(tok.is_stop),
                    is_punct=bool(tok.is_punct),
                    head_text=head_text,
                    children=children,
                )
            )

        # Entities
        entities: List[Dict[str, Any]] = []
        if self._has_ner:
            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": int(ent.start_char),
                        "end": int(ent.end_char),
                    }
                )

        # Noun chunks
        # If no parser, emulate naive chunks: sequences of alpha, non-stop tokens (very rough).
        noun_chunks: List[str] = []
        try:
            noun_chunks = [span.text for span in doc.noun_chunks]  # requires parser
        except Exception:
            noun_chunks = self._naive_noun_chunks(doc)

        return ParsedSentence(text=doc.text, tokens=tokens, entities=entities, noun_chunks=noun_chunks)

    @staticmethod
    def _naive_noun_chunks(doc) -> List[str]:
        """Very rough fallback chunker when parser is unavailable."""
        chunks: List[str] = []
        current: List[str] = []
        for tok in doc:
            if tok.is_alpha and not tok.is_stop and not tok.is_punct:
                current.append(tok.text)
            else:
                if current:
                    chunks.append(" ".join(current))
                    current = []
        if current:
            chunks.append(" ".join(current))
        # de-duplicate while preserving order
        seen, out = set(), []
        for ch in chunks:
            if ch and ch not in seen:
                seen.add(ch); out.append(ch)
        return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Text Parser (spaCy-based) CLI")
    p.add_argument("input", help="Raw text or a path to a UTF-8 text file")
    p.add_argument("--model", default="en_core_web_sm", help="spaCy model name")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available")
    p.add_argument("--n-process", type=int, default=1, help="Num processes for nlp.pipe")
    p.add_argument("--batch-size", type=int, default=1000, help="Batch size for nlp.pipe")
    p.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    p.add_argument("--features", action="store_true", help="Output feature summary instead of full parse")
    p.add_argument("--outfile", default="", help="Write output to file (default: stdout)")
    return p


def _read_input(text_or_path: str) -> str:
    try:
        # If it's a file path, read it; otherwise treat as raw input
        import os
        if os.path.exists(text_or_path):
            with open(text_or_path, "r", encoding="utf-8") as f:
                return f.read()
        return text_or_path
    except Exception as e:
        logger.error("Failed to read input: %s", e)
        raise


def _cli_main() -> int:
    args = _build_arg_parser().parse_args()

    cfg = ParserConfig(
        model=args.model,
        use_gpu=args.gpu,
        n_process=max(1, int(args.n_process)),
        batch_size=max(1, int(args.batch_size)),
        cache_size=512,
    )
    parser = TextParser(cfg)
    text = _read_input(args.input)

    if args.features:
        out = parser.extract_features(text)
        payload = json.dumps(out, ensure_ascii=False, indent=2)
    else:
        parsed = parser.parse(text)
        if args.format == "json":
            payload = json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2)
        else:
            # CSV token table (requires pandas)
            try:
                df = parser.to_dataframe(parsed)
                payload = df.to_csv(index=False)
            except Exception as e:
                logger.error("CSV export requires pandas: %s", e)
                return 2

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())
