"""
Meaning Analyzer

Advanced meaning analysis framework implementing the concepts from the problem statement.
Includes etymological, semantic, syntactic, and holistic analysis capabilities.
"""

import re
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from context.parser import TextParser


class AnalysisType(Enum):
    """Types of meaning analysis."""
    LEXICAL = "lexical"
    GRAMMATICAL = "grammatical"
    SYNTACTICAL = "syntactical"
    PROPOSITIONAL = "propositional"
    VERACITY = "veracity"
    SUBTEXT = "subtext"
    SYMBOLIC = "symbolic"
    IMAGERY = "imagery"
    HOLISTIC = "holistic"
    ETYMOLOGICAL = "etymological"
    INTENT_TONE = "intent_tone"
    ADJECTIVE = "adjective"


@dataclass
class AnalysisResult:
    """Result of a meaning analysis operation."""
    analysis_type: AnalysisType
    confidence: float
    findings: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class MeaningAnalyzer:
    """
    Advanced meaning analysis system implementing multiple analysis types
    as described in the problem statement.
    
    Provides:
    - Etymological Analysis
    - Holistic Synthesis  
    - Imagery Analysis
    - Symbolic Analysis
    - Subtext & Implication
    - Veracity Analysis
    - Propositional Analysis
    - Component Analysis
    - Syntactical Analysis
    - Grammatical Role
    - Intent & Tone Analysis
    - Lexical Analysis
    - Adjective Analysis
    """
    
    def __init__(self):
        self.text_parser = TextParser()
        self.analysis_cache = {}
        
        # Pre-built knowledge for etymological analysis
        self.etymology_db = self._build_etymology_db()
        
        # Symbolic mappings
        self.symbol_mappings = self._build_symbol_mappings()
        
        # Common patterns for various analyses
        self.patterns = self._build_analysis_patterns()
    
    async def analyze_meaning(self, text: str, analysis_types: Optional[List[AnalysisType]] = None) -> Dict[str, AnalysisResult]:
        """
        Perform comprehensive meaning analysis.
        
        Args:
            text: Text to analyze
            analysis_types: Specific analysis types to perform (default: all)
            
        Returns:
            Dictionary mapping analysis type to results
        """
        if analysis_types is None:
            analysis_types = list(AnalysisType)
        
        results = {}
        
        # Parse text first
        parsed = self.text_parser.parse(text)
        
        for analysis_type in analysis_types:
            start_time = time.time()
            
            try:
                result = await self._perform_analysis(analysis_type, text, parsed)
                result.processing_time = time.time() - start_time
                results[analysis_type.value] = result
            except Exception as e:
                results[analysis_type.value] = AnalysisResult(
                    analysis_type=analysis_type,
                    confidence=0.0,
                    findings={'error': str(e)},
                    processing_time=time.time() - start_time
                )
        
        return results
    
    async def _perform_analysis(self, analysis_type: AnalysisType, text: str, parsed) -> AnalysisResult:
        """Perform specific type of analysis."""
        
        if analysis_type == AnalysisType.LEXICAL:
            return await self._lexical_analysis(text, parsed)
        elif analysis_type == AnalysisType.GRAMMATICAL:
            return await self._grammatical_analysis(text, parsed)
        elif analysis_type == AnalysisType.SYNTACTICAL:
            return await self._syntactical_analysis(text, parsed)
        elif analysis_type == AnalysisType.PROPOSITIONAL:
            return await self._propositional_analysis(text, parsed)
        elif analysis_type == AnalysisType.VERACITY:
            return await self._veracity_analysis(text, parsed)
        elif analysis_type == AnalysisType.SUBTEXT:
            return await self._subtext_analysis(text, parsed)
        elif analysis_type == AnalysisType.SYMBOLIC:
            return await self._symbolic_analysis(text, parsed)
        elif analysis_type == AnalysisType.IMAGERY:
            return await self._imagery_analysis(text, parsed)
        elif analysis_type == AnalysisType.HOLISTIC:
            return await self._holistic_analysis(text, parsed)
        elif analysis_type == AnalysisType.ETYMOLOGICAL:
            return await self._etymological_analysis(text, parsed)
        elif analysis_type == AnalysisType.INTENT_TONE:
            return await self._intent_tone_analysis(text, parsed)
        elif analysis_type == AnalysisType.ADJECTIVE:
            return await self._adjective_analysis(text, parsed)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    async def _lexical_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze lexical features of the text."""
        findings = {
            'total_tokens': len(parsed.tokens),
            'unique_tokens': len(set(token.lemma for token in parsed.tokens if token.is_alpha)),
            'lexical_diversity': 0.0,
            'word_frequency': {},
            'pos_distribution': {},
            'complexity_score': 0.0
        }
        
        # Calculate lexical diversity
        alpha_tokens = [token for token in parsed.tokens if token.is_alpha]
        if alpha_tokens:
            unique_lemmas = set(token.lemma for token in alpha_tokens)
            findings['lexical_diversity'] = len(unique_lemmas) / len(alpha_tokens)
        
        # Word frequency
        lemma_counts = {}
        pos_counts = {}
        
        for token in parsed.tokens:
            if token.is_alpha:
                lemma_counts[token.lemma] = lemma_counts.get(token.lemma, 0) + 1
                pos_counts[token.pos] = pos_counts.get(token.pos, 0) + 1
        
        findings['word_frequency'] = dict(sorted(lemma_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        findings['pos_distribution'] = pos_counts
        
        # Complexity score based on lexical diversity and rare words
        complexity = findings['lexical_diversity'] * 0.6
        rare_words = sum(1 for count in lemma_counts.values() if count == 1)
        if alpha_tokens:
            complexity += (rare_words / len(alpha_tokens)) * 0.4
        findings['complexity_score'] = complexity
        
        return AnalysisResult(
            analysis_type=AnalysisType.LEXICAL,
            confidence=0.9,
            findings=findings
        )
    
    async def _grammatical_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze grammatical roles and structures."""
        findings = {
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'dependency_relations': {},
            'grammatical_roles': {},
            'clause_types': [],
            'complexity_indicators': {}
        }
        
        # Dependency relations
        dep_counts = {}
        role_counts = {}
        
        for token in parsed.tokens:
            if token.dep:
                dep_counts[token.dep] = dep_counts.get(token.dep, 0) + 1
            if token.pos:
                role_counts[token.pos] = role_counts.get(token.pos, 0) + 1
        
        findings['dependency_relations'] = dep_counts
        findings['grammatical_roles'] = role_counts
        
        # Complexity indicators
        findings['complexity_indicators'] = {
            'subordinate_clauses': dep_counts.get('advcl', 0) + dep_counts.get('acl', 0),
            'relative_clauses': dep_counts.get('acl:relcl', 0),
            'passive_indicators': sum(1 for token in parsed.tokens if token.dep == 'nsubjpass'),
            'coordination': dep_counts.get('conj', 0)
        }
        
        return AnalysisResult(
            analysis_type=AnalysisType.GRAMMATICAL,
            confidence=0.85,
            findings=findings
        )
    
    async def _syntactical_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze syntactic structures and patterns."""
        findings = {
            'sentence_types': [],
            'phrase_structures': [],
            'syntactic_patterns': {},
            'tree_depth': 0,
            'branching_factor': 0.0
        }
        
        # Sentence type detection
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        for sentence in sentences:
            if sentence.endswith('?'):
                findings['sentence_types'].append('interrogative')
            elif sentence.endswith('!'):
                findings['sentence_types'].append('exclamatory')
            elif any(word in sentence.lower() for word in ['please', 'do', 'go', 'come']):
                findings['sentence_types'].append('imperative')
            else:
                findings['sentence_types'].append('declarative')
        
        # Phrase structures (based on noun chunks and entities)
        findings['phrase_structures'] = [
            {'type': 'noun_phrase', 'text': chunk} for chunk in parsed.noun_chunks
        ]
        
        # Syntactic patterns
        patterns = {}
        for token in parsed.tokens:
            if token.head_text != token.text:  # Not root
                pattern = f"{token.pos}->{token.head_text}"
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        findings['syntactic_patterns'] = patterns
        
        return AnalysisResult(
            analysis_type=AnalysisType.SYNTACTICAL,
            confidence=0.8,
            findings=findings
        )
    
    async def _propositional_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze propositional content and logical structure."""
        findings = {
            'propositions': [],
            'logical_connectors': [],
            'truth_conditions': {},
            'modal_indicators': [],
            'quantifiers': []
        }
        
        # Extract potential propositions (simplified)
        verbs = [token for token in parsed.tokens if token.pos in ['VERB', 'AUX']]
        for verb in verbs:
            subjects = [child for child in verb.children if 'subj' in child.lower()]
            objects = [child for child in verb.children if 'obj' in child.lower()]
            
            proposition = {
                'predicate': verb.lemma,
                'subjects': subjects,
                'objects': objects,
                'modifiers': [child for child in verb.children if child not in subjects + objects]
            }
            findings['propositions'].append(proposition)
        
        # Logical connectors
        logical_words = ['and', 'or', 'but', 'if', 'then', 'because', 'since', 'although']
        for token in parsed.tokens:
            if token.lemma.lower() in logical_words:
                findings['logical_connectors'].append({
                    'connector': token.lemma,
                    'type': self._classify_connector(token.lemma)
                })
        
        # Modal indicators
        modal_words = ['can', 'could', 'may', 'might', 'must', 'should', 'would', 'will']
        for token in parsed.tokens:
            if token.lemma.lower() in modal_words:
                findings['modal_indicators'].append(token.lemma)
        
        return AnalysisResult(
            analysis_type=AnalysisType.PROPOSITIONAL,
            confidence=0.75,
            findings=findings
        )
    
    async def _veracity_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze truth value and veracity indicators."""
        findings = {
            'certainty_level': 0.5,
            'factual_markers': [],
            'uncertainty_markers': [],
            'hedge_words': [],
            'assertiveness_score': 0.5,
            'evidence_indicators': []
        }
        
        # Certainty indicators
        certain_words = ['definitely', 'certainly', 'absolutely', 'clearly', 'obviously']
        uncertain_words = ['maybe', 'perhaps', 'possibly', 'probably', 'might', 'could']
        hedge_words = ['sort of', 'kind of', 'somewhat', 'rather', 'quite', 'fairly']
        
        certainty_score = 0.5
        
        for token in parsed.tokens:
            lemma_lower = token.lemma.lower()
            if lemma_lower in certain_words:
                findings['factual_markers'].append(token.lemma)
                certainty_score += 0.1
            elif lemma_lower in uncertain_words:
                findings['uncertainty_markers'].append(token.lemma)
                certainty_score -= 0.1
            elif lemma_lower in hedge_words:
                findings['hedge_words'].append(token.lemma)
                certainty_score -= 0.05
        
        findings['certainty_level'] = max(0.0, min(1.0, certainty_score))
        
        # Assertiveness based on sentence structure
        assertive_patterns = ['!', 'must', 'will', 'is', 'are']
        assertiveness = 0.5
        for pattern in assertive_patterns:
            if pattern in text.lower():
                assertiveness += 0.1
        
        findings['assertiveness_score'] = max(0.0, min(1.0, assertiveness))
        
        return AnalysisResult(
            analysis_type=AnalysisType.VERACITY,
            confidence=0.7,
            findings=findings
        )
    
    async def _subtext_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze subtext and implicit meanings."""
        findings = {
            'implicit_meanings': [],
            'connotations': {},
            'cultural_references': [],
            'emotional_undertones': [],
            'irony_indicators': [],
            'metaphorical_language': []
        }
        
        # Look for potential metaphorical language
        metaphor_indicators = ['like', 'as', 'than', 'seems', 'appears']
        for token in parsed.tokens:
            if token.lemma.lower() in metaphor_indicators:
                findings['metaphorical_language'].append({
                    'indicator': token.lemma,
                    'context': token.text
                })
        
        # Emotional undertones based on adjectives
        emotional_adjectives = {
            'positive': ['good', 'great', 'excellent', 'wonderful', 'amazing'],
            'negative': ['bad', 'terrible', 'awful', 'horrible', 'disgusting'],
            'neutral': ['normal', 'average', 'typical', 'standard', 'regular']
        }
        
        for token in parsed.tokens:
            if token.pos == 'ADJ':
                for emotion, words in emotional_adjectives.items():
                    if token.lemma.lower() in words:
                        findings['emotional_undertones'].append({
                            'adjective': token.lemma,
                            'emotion': emotion
                        })
        
        # Irony indicators (simplified detection)
        irony_patterns = ['really', 'sure', 'right', 'yeah right', 'of course']
        for pattern in irony_patterns:
            if pattern in text.lower():
                findings['irony_indicators'].append(pattern)
        
        return AnalysisResult(
            analysis_type=AnalysisType.SUBTEXT,
            confidence=0.6,
            findings=findings
        )
    
    async def _symbolic_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze symbolic meanings and representations."""
        findings = {
            'symbols_detected': [],
            'archetypal_patterns': [],
            'color_symbolism': [],
            'number_symbolism': [],
            'cultural_symbols': []
        }
        
        # Symbol detection
        for symbol, meaning in self.symbol_mappings.items():
            if symbol in text.lower():
                findings['symbols_detected'].append({
                    'symbol': symbol,
                    'meaning': meaning
                })
        
        # Color symbolism
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange']
        color_meanings = {
            'red': 'passion, danger, energy',
            'blue': 'calm, trust, sadness',
            'green': 'nature, growth, money',
            'yellow': 'happiness, caution, intellect',
            'black': 'mystery, death, elegance',
            'white': 'purity, peace, emptiness',
            'purple': 'royalty, mystery, spirituality',
            'orange': 'enthusiasm, creativity, warmth'
        }
        
        for color in colors:
            if color in text.lower():
                findings['color_symbolism'].append({
                    'color': color,
                    'symbolic_meaning': color_meanings[color]
                })
        
        # Number symbolism (basic)
        number_meanings = {
            'one': 'unity, beginning',
            'two': 'duality, balance',
            'three': 'completion, trinity',
            'seven': 'perfection, spirituality',
            'thirteen': 'unlucky, transformation'
        }
        
        for number, meaning in number_meanings.items():
            if number in text.lower():
                findings['number_symbolism'].append({
                    'number': number,
                    'symbolic_meaning': meaning
                })
        
        return AnalysisResult(
            analysis_type=AnalysisType.SYMBOLIC,
            confidence=0.65,
            findings=findings
        )
    
    async def _imagery_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze imagery and sensory language."""
        findings = {
            'visual_imagery': [],
            'auditory_imagery': [],
            'tactile_imagery': [],
            'olfactory_imagery': [],
            'gustatory_imagery': [],
            'kinesthetic_imagery': [],
            'imagery_density': 0.0
        }
        
        # Sensory word categories
        sensory_words = {
            'visual': ['see', 'look', 'bright', 'dark', 'colorful', 'shiny', 'dim', 'glowing'],
            'auditory': ['hear', 'sound', 'loud', 'quiet', 'whisper', 'roar', 'music', 'noise'],
            'tactile': ['touch', 'feel', 'soft', 'rough', 'smooth', 'hard', 'warm', 'cold'],
            'olfactory': ['smell', 'fragrant', 'stink', 'aroma', 'scent', 'perfume'],
            'gustatory': ['taste', 'sweet', 'sour', 'bitter', 'salty', 'delicious', 'flavor'],
            'kinesthetic': ['move', 'run', 'jump', 'dance', 'flow', 'rush', 'crawl']
        }
        
        total_imagery = 0
        for sense, words in sensory_words.items():
            sense_imagery = []
            for token in parsed.tokens:
                if token.lemma.lower() in words:
                    sense_imagery.append(token.lemma)
                    total_imagery += 1
            findings[f'{sense}_imagery'] = sense_imagery
        
        # Calculate imagery density
        total_words = len([token for token in parsed.tokens if token.is_alpha])
        if total_words > 0:
            findings['imagery_density'] = total_imagery / total_words
        
        return AnalysisResult(
            analysis_type=AnalysisType.IMAGERY,
            confidence=0.8,
            findings=findings
        )
    
    async def _holistic_analysis(self, text: str, parsed) -> AnalysisResult:
        """Perform holistic synthesis of all analysis types."""
        findings = {
            'overall_complexity': 0.0,
            'semantic_coherence': 0.0,
            'emotional_tone': 'neutral',
            'communication_effectiveness': 0.0,
            'key_themes': [],
            'synthesis_score': 0.0
        }
        
        # Calculate overall complexity
        complexity_factors = [
            len(parsed.tokens) / 100,  # Length factor
            len(set(token.lemma for token in parsed.tokens if token.is_alpha)) / 50,  # Vocabulary diversity
            len(parsed.entities) / 10,  # Entity complexity
            len(parsed.noun_chunks) / 20  # Structural complexity
        ]
        findings['overall_complexity'] = min(sum(complexity_factors) / len(complexity_factors), 1.0)
        
        # Semantic coherence (simplified)
        coherence = 0.7  # Base coherence
        if parsed.entities:
            coherence += 0.1  # Entities add coherence
        if parsed.noun_chunks:
            coherence += 0.1  # Structure adds coherence
        findings['semantic_coherence'] = min(coherence, 1.0)
        
        # Extract key themes (most frequent meaningful words)
        meaningful_tokens = [token.lemma.lower() for token in parsed.tokens 
                           if token.is_alpha and not token.is_stop and len(token.lemma) > 3]
        
        if meaningful_tokens:
            from collections import Counter
            common_themes = Counter(meaningful_tokens).most_common(5)
            findings['key_themes'] = [theme for theme, count in common_themes if count > 1]
        
        # Communication effectiveness
        effectiveness = findings['semantic_coherence'] * 0.6 + (1.0 - findings['overall_complexity']) * 0.4
        findings['communication_effectiveness'] = effectiveness
        
        # Synthesis score
        findings['synthesis_score'] = (
            findings['semantic_coherence'] * 0.4 +
            findings['communication_effectiveness'] * 0.4 +
            (1.0 if findings['key_themes'] else 0.5) * 0.2
        )
        
        return AnalysisResult(
            analysis_type=AnalysisType.HOLISTIC,
            confidence=0.85,
            findings=findings
        )
    
    async def _etymological_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze word origins and etymology."""
        findings = {
            'word_origins': {},
            'language_families': {},
            'root_morphemes': [],
            'derived_words': [],
            'etymological_depth': 0.0
        }
        
        etymology_count = 0
        for token in parsed.tokens:
            if token.is_alpha and len(token.lemma) > 3:
                lemma_lower = token.lemma.lower()
                if lemma_lower in self.etymology_db:
                    etymology_info = self.etymology_db[lemma_lower]
                    findings['word_origins'][token.lemma] = etymology_info
                    etymology_count += 1
        
        # Calculate etymological depth
        total_meaningful_words = len([token for token in parsed.tokens 
                                    if token.is_alpha and not token.is_stop])
        if total_meaningful_words > 0:
            findings['etymological_depth'] = etymology_count / total_meaningful_words
        
        return AnalysisResult(
            analysis_type=AnalysisType.ETYMOLOGICAL,
            confidence=0.7,
            findings=findings
        )
    
    async def _intent_tone_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze intent and tone."""
        findings = {
            'primary_intent': 'informative',
            'tone_indicators': [],
            'formality_level': 'neutral',
            'urgency_level': 'normal',
            'emotional_charge': 0.0
        }
        
        # Intent detection
        if '?' in text:
            findings['primary_intent'] = 'interrogative'
        elif any(word in text.lower() for word in ['please', 'could you', 'would you']):
            findings['primary_intent'] = 'request'
        elif '!' in text or any(word in text.lower() for word in ['must', 'should', 'need to']):
            findings['primary_intent'] = 'directive'
        elif any(word in text.lower() for word in ['thank', 'sorry', 'congratulations']):
            findings['primary_intent'] = 'social'
        
        # Tone detection
        formal_indicators = ['therefore', 'furthermore', 'consequently', 'moreover']
        informal_indicators = ['gonna', 'wanna', 'yeah', 'ok', 'cool']
        
        formality_score = 0.5
        for token in parsed.tokens:
            if token.lemma.lower() in formal_indicators:
                formality_score += 0.1
            elif token.lemma.lower() in informal_indicators:
                formality_score -= 0.1
        
        if formality_score > 0.7:
            findings['formality_level'] = 'formal'
        elif formality_score < 0.3:
            findings['formality_level'] = 'informal'
        
        # Urgency detection
        urgent_words = ['urgent', 'immediate', 'asap', 'emergency', 'quick', 'fast']
        if any(word in text.lower() for word in urgent_words):
            findings['urgency_level'] = 'high'
        
        return AnalysisResult(
            analysis_type=AnalysisType.INTENT_TONE,
            confidence=0.75,
            findings=findings
        )
    
    async def _adjective_analysis(self, text: str, parsed) -> AnalysisResult:
        """Analyze adjectives and their semantic properties."""
        findings = {
            'adjectives_found': [],
            'semantic_categories': {},
            'intensity_levels': {},
            'evaluative_adjectives': [],
            'descriptive_adjectives': []
        }
        
        adjectives = [token for token in parsed.tokens if token.pos == 'ADJ']
        
        # Categorize adjectives
        evaluative_adj = ['good', 'bad', 'excellent', 'terrible', 'wonderful', 'awful']
        intensity_adj = ['very', 'extremely', 'quite', 'rather', 'somewhat', 'slightly']
        
        for adj in adjectives:
            adj_info = {
                'word': adj.lemma,
                'form': adj.text,
                'category': 'descriptive'
            }
            
            if adj.lemma.lower() in evaluative_adj:
                adj_info['category'] = 'evaluative'
                findings['evaluative_adjectives'].append(adj.lemma)
            else:
                findings['descriptive_adjectives'].append(adj.lemma)
            
            findings['adjectives_found'].append(adj_info)
        
        return AnalysisResult(
            analysis_type=AnalysisType.ADJECTIVE,
            confidence=0.9,
            findings=findings
        )
    
    def _build_etymology_db(self) -> Dict[str, Dict[str, str]]:
        """Build basic etymology database."""
        return {
            'condition': {
                'origin': 'Latin',
                'etymology': 'conditio (agreement; situation), from condicere (declare together)',
                'root': 'con- + dic-'
            },
            'status': {
                'origin': 'Latin', 
                'etymology': 'status (standing, position)',
                'root': 'stat-'
            },
            'state': {
                'origin': 'Old French/Latin',
                'etymology': 'estat; Latin status (standing), PIE *steh₂- (to stand)',
                'root': 'stat-'
            },
            'identity': {
                'origin': 'Latin',
                'etymology': 'identitas, from idem (same)',
                'root': 'id-'
            }
        }
    
    def _build_symbol_mappings(self) -> Dict[str, str]:
        """Build symbolic meaning mappings."""
        return {
            'light': 'knowledge, enlightenment, hope',
            'dark': 'mystery, unknown, fear',
            'water': 'life, purification, emotion',
            'fire': 'passion, destruction, transformation',
            'earth': 'stability, foundation, material',
            'air': 'freedom, thought, communication',
            'tree': 'growth, life, connection',
            'mountain': 'challenge, achievement, permanence',
            'circle': 'completeness, unity, cycle',
            'square': 'stability, structure, order'
        }
    
    def _build_analysis_patterns(self) -> Dict[str, List[str]]:
        """Build pattern recognition data."""
        return {
            'question_patterns': [r'\?', r'^(what|how|why|when|where|who)', r'(could|would|can|will) you'],
            'emphasis_patterns': [r'!', r'[A-Z]{2,}', r'very', r'extremely', r'absolutely'],
            'uncertainty_patterns': [r'maybe', r'perhaps', r'might', r'could', r'possibly']
        }
    
    def _classify_connector(self, connector: str) -> str:
        """Classify logical connectors."""
        conjunctive = ['and', 'also', 'furthermore', 'moreover']
        disjunctive = ['or', 'either', 'alternatively']
        adversative = ['but', 'however', 'although', 'yet']
        causal = ['because', 'since', 'therefore', 'thus']
        conditional = ['if', 'unless', 'provided']
        
        if connector in conjunctive:
            return 'conjunctive'
        elif connector in disjunctive:
            return 'disjunctive'
        elif connector in adversative:
            return 'adversative'
        elif connector in causal:
            return 'causal'
        elif connector in conditional:
            return 'conditional'
        else:
            return 'other'