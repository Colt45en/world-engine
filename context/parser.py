"""
Text Parser - spaCy-based pipeline for tokenization, POS tagging, and dependency parsing.

Provides linguistic analysis foundation for the World Engine context processing.
"""

import spacy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Token:
    """Represents a parsed token with linguistic features."""
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


@dataclass
class ParsedSentence:
    """Represents a parsed sentence with tokens and structure."""
    text: str
    tokens: List[Token]
    entities: List[Dict[str, Any]]
    noun_chunks: List[str]


class TextParser:
    """spaCy-based text parser for linguistic analysis."""

    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize with spaCy model."""
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Warning: spaCy model '{model}' not found. Using blank model.")
            self.nlp = spacy.blank("en")
            # Add basic components for blank model
            self.nlp.add_pipe("sentencizer")

        self.model_name = model

    def parse(self, text: str) -> ParsedSentence:
        """Parse text and return structured linguistic information."""
        doc = self.nlp(text)

        tokens = []
        for token in doc:
            parsed_token = Token(
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
                is_alpha=token.is_alpha,
                is_stop=token.is_stop,
                is_punct=token.is_punct,
                head_text=token.head.text,
                children=[child.text for child in token.children]
            )
            tokens.append(parsed_token)

        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        noun_chunks = [chunk.text for chunk in doc.noun_chunks]

        return ParsedSentence(
            text=text,
            tokens=tokens,
            entities=entities,
            noun_chunks=noun_chunks
        )

    def parse_batch(self, texts: List[str]) -> List[ParsedSentence]:
        """Parse multiple texts efficiently."""
        results = []
        for doc in self.nlp.pipe(texts):
            # Convert spaCy doc to ParsedSentence
            tokens = []
            for token in doc:
                parsed_token = Token(
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    tag=token.tag_,
                    dep=token.dep_,
                    is_alpha=token.is_alpha,
                    is_stop=token.is_stop,
                    is_punct=token.is_punct,
                    head_text=token.head.text,
                    children=[child.text for child in token.children]
                )
                tokens.append(parsed_token)

            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

            noun_chunks = [chunk.text for chunk in doc.noun_chunks]

            results.append(ParsedSentence(
                text=doc.text,
                tokens=tokens,
                entities=entities,
                noun_chunks=noun_chunks
            ))

        return results

    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive linguistic features."""
        parsed = self.parse(text)

        # Count features
        pos_counts = {}
        dep_counts = {}

        for token in parsed.tokens:
            pos_counts[token.pos] = pos_counts.get(token.pos, 0) + 1
            dep_counts[token.dep] = dep_counts.get(token.dep, 0) + 1

        return {
            'text': text,
            'token_count': len(parsed.tokens),
            'entity_count': len(parsed.entities),
            'noun_chunk_count': len(parsed.noun_chunks),
            'pos_distribution': pos_counts,
            'dependency_distribution': dep_counts,
            'entities': parsed.entities,
            'noun_chunks': parsed.noun_chunks,
            'alpha_ratio': sum(1 for t in parsed.tokens if t.is_alpha) / len(parsed.tokens) if parsed.tokens else 0,
            'stop_ratio': sum(1 for t in parsed.tokens if t.is_stop) / len(parsed.tokens) if parsed.tokens else 0
        }

    def get_lemmas(self, text: str) -> List[str]:
        """Extract lemmatized tokens from text."""
        parsed = self.parse(text)
        return [token.lemma for token in parsed.tokens if token.is_alpha]

    def get_dependencies(self, text: str) -> List[Dict[str, str]]:
        """Extract dependency relationships."""
        parsed = self.parse(text)
        deps = []

        for token in parsed.tokens:
            if token.dep != 'ROOT':
                deps.append({
                    'token': token.text,
                    'head': token.head_text,
                    'relation': token.dep
                })

        return deps
