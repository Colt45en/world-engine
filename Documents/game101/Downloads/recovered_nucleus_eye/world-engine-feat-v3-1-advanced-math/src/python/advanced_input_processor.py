# ADVANCED INPUT PROCESSING SYSTEM
# Enhanced ingest module with sophisticated segmentation, command detection, and language analysis

import re
import unicodedata
import json
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from enum import Enum
from datetime import datetime
import pytz

class AdvancedSegmentType(Enum):
    """Extended segment types for detailed content analysis"""
    TEXT = "text"
    CODE_BLOCK = "code_block"
    INLINE_CODE = "inline_code"
    MATH_BLOCK = "math_block"
    INLINE_MATH = "inline_math"
    COMMAND = "command"
    FILE_PATH = "file_path"
    URL = "url"
    EMAIL = "email"
    QUOTED_TEXT = "quoted_text"
    REGEX_PATTERN = "regex_pattern"
    DATE_TIME = "date_time"
    NUMBER = "number"
    VARIABLE_REF = "variable_ref"
    FUNCTION_CALL = "function_call"

class LanguageHint(NamedTuple):
    """Language detection result"""
    language: str
    confidence: float
    indicators: List[str]

class ContentSegment(NamedTuple):
    """Enhanced content segment with detailed metadata"""
    segment_type: AdvancedSegmentType
    content: str
    start_pos: int
    end_pos: int
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 1.0

class ProcessedTurn(NamedTuple):
    """Enhanced turn data structure"""
    turn_id: str
    timestamp: str
    raw_input: str
    normalized_input: str
    segments: List[ContentSegment]
    detected_language: LanguageHint
    profanity_flags: List[str]
    pii_detected: List[Dict[str, Any]]
    command_detected: bool
    urgency_score: float
    complexity_score: float

class AdvancedInputProcessor:
    """Sophisticated input processing with multi-layered analysis"""

    def __init__(self):
        self.timezone = pytz.timezone("America/Chicago")

        # Language detection patterns
        self.language_patterns = {
            "python": [
                r"\bdef\s+\w+\s*\(",
                r"\bimport\s+\w+",
                r"\bfrom\s+\w+\s+import",
                r"\bclass\s+\w+\s*\(",
                r"\bif\s+__name__\s*==\s*['\"]__main__['\"]",
                r"print\s*\("
            ],
            "typescript": [
                r"\binterface\s+\w+",
                r"\btype\s+\w+\s*=",
                r"\bconst\s+\w+\s*:\s*\w+",
                r"\bfunction\s+\w+\s*\(",
                r"\bexport\s+(default\s+)?(class|function|interface)",
                r"\.tsx?$"
            ],
            "javascript": [
                r"\bconst\s+\w+\s*=",
                r"\blet\s+\w+\s*=",
                r"\bfunction\s+\w+\s*\(",
                r"\bconsole\.log\s*\(",
                r"=>\s*{?",
                r"\.jsx?$"
            ],
            "csharp": [
                r"\bpublic\s+(class|interface|struct)",
                r"\busing\s+System",
                r"\bnamespace\s+\w+",
                r"\bConsole\.WriteLine\s*\(",
                r"\bpublic\s+(static\s+)?void\s+Main",
                r"\.cs$"
            ],
            "java": [
                r"\bpublic\s+class\s+\w+",
                r"\bimport\s+java\.",
                r"\bpublic\s+static\s+void\s+main",
                r"\bSystem\.out\.print",
                r"\.java$"
            ],
            "regex": [
                r"^[\[\](){}^$.*+?|\\-]+$",
                r"\\\w\+",
                r"\[.*\]",
                r"\{.*,.*\}",
                r"\(\?\:",
                r"\\[ntrfvd]"
            ],
            "sql": [
                r"\bSELECT\s+.*\bFROM\b",
                r"\bINSERT\s+INTO\b",
                r"\bUPDATE\s+.*\bSET\b",
                r"\bDELETE\s+FROM\b",
                r"\bCREATE\s+(TABLE|INDEX|VIEW)",
                r"\bJOIN\b.*\bON\b"
            ]
        }

        # Command detection patterns
        self.command_patterns = [
            r"^/(create|make|build|generate)\s+",
            r"^@\w+\s+",
            r"^#\w+\s+",
            r"^![\w-]+",
            r"^>>\s*",
            r"^\$\s*\w+"
        ]

        # PII detection patterns
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        }

        # Profanity detection (basic placeholder)
        self.profanity_words = set([
            # Add actual profanity list here
            "badword1", "badword2"  # Placeholder
        ])

        # Urgency indicators
        self.urgency_patterns = [
            (r"\b(urgent|asap|immediately|now|quick)\b", 0.8),
            (r"\b(soon|fast|hurry)\b", 0.6),
            (r"\b(when you can|eventually)\b", 0.2),
            (r"[!]{2,}", 0.9),  # Multiple exclamation marks
            (r"\b(emergency|critical|broken)\b", 1.0)
        ]

        # Complexity indicators
        self.complexity_patterns = [
            (r"\b(complex|complicated|sophisticated|advanced)\b", 0.8),
            (r"\b(simple|basic|easy|quick)\b", 0.2),
            (r"\b(algorithm|architecture|design pattern)\b", 0.9),
            (r"\b(integrate|combine|merge|connect)\b", 0.7),
            (r"\b(optimize|refactor|redesign)\b", 0.8)
        ]

    def process_input(self, raw_input: str) -> ProcessedTurn:
        """Main processing pipeline for input analysis"""

        # Generate turn metadata
        turn_id = f"t-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{hash(raw_input) % 1000:03d}"
        timestamp = datetime.now(self.timezone).isoformat()

        # Step 1: Normalize input
        normalized = self._normalize_unicode(raw_input)

        # Step 2: Detect language
        language_hint = self._detect_language(normalized)

        # Step 3: Segment content
        segments = self._segment_content(normalized)

        # Step 4: Detect profanity and PII
        profanity_flags = self._detect_profanity(normalized)
        pii_detected = self._detect_pii(normalized)

        # Step 5: Detect commands
        command_detected = self._detect_commands(normalized)

        # Step 6: Calculate urgency and complexity scores
        urgency_score = self._calculate_urgency(normalized)
        complexity_score = self._calculate_complexity(normalized)

        return ProcessedTurn(
            turn_id=turn_id,
            timestamp=timestamp,
            raw_input=raw_input,
            normalized_input=normalized,
            segments=segments,
            detected_language=language_hint,
            profanity_flags=profanity_flags,
            pii_detected=pii_detected,
            command_detected=command_detected,
            urgency_score=urgency_score,
            complexity_score=complexity_score
        )

    def _normalize_unicode(self, text: str) -> str:
        """Comprehensive Unicode normalization and cleaning"""

        # Normalize Unicode to NFC (Canonical Decomposition + Canonical Composition)
        normalized = unicodedata.normalize('NFC', text)

        # Remove zero-width characters
        zero_width_chars = ['\u200B', '\u200C', '\u200D', '\u2060', '\uFEFF']
        for char in zero_width_chars:
            normalized = normalized.replace(char, '')

        # Normalize various quotation marks to standard ASCII
        quote_mappings = {
            '\u2018': "'", '\u2019': "'",  # Smart single quotes
            '\u201C': '"', '\u201D': '"',  # Smart double quotes
            '\u00AB': '"', '\u00BB': '"',  # French quotes
            '\u2039': "'", '\u203A': "'"   # Single angle quotes
        }

        for unicode_char, ascii_char in quote_mappings.items():
            normalized = normalized.replace(unicode_char, ascii_char)

        # Normalize whitespace (but preserve intentional formatting)
        # Replace multiple spaces with single space, but keep line breaks
        normalized = re.sub(r'[ \t]+', ' ', normalized)
        normalized = re.sub(r'\n\s*\n', '\n\n', normalized)  # Normalize paragraph breaks

        # Trim leading/trailing whitespace but preserve internal structure
        normalized = normalized.strip()

        return normalized

    def _detect_language(self, text: str) -> LanguageHint:
        """Detect programming language and natural language"""

        language_scores = {}
        all_indicators = []

        # Check for programming language patterns
        for lang, patterns in self.language_patterns.items():
            score = 0
            indicators = []

            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    score += len(matches) * 0.2
                    indicators.extend(matches[:3])  # Limit indicators

            if score > 0:
                language_scores[lang] = min(score, 1.0)  # Cap at 1.0
                all_indicators.extend(indicators)

        # Determine best language match
        if language_scores:
            best_lang = max(language_scores.keys(), key=lambda k: language_scores[k])
            confidence = language_scores[best_lang]
        else:
            # Default to English natural language
            best_lang = "english"
            confidence = 0.7

        return LanguageHint(
            language=best_lang,
            confidence=confidence,
            indicators=all_indicators[:5]  # Top 5 indicators
        )

    def _segment_content(self, text: str) -> List[ContentSegment]:
        """Advanced content segmentation with multiple pattern types"""

        segments = []
        processed_ranges = []  # Track what's been processed

        # Pattern processors in order of specificity
        processors = [
            (self._extract_code_blocks, AdvancedSegmentType.CODE_BLOCK),
            (self._extract_math_blocks, AdvancedSegmentType.MATH_BLOCK),
            (self._extract_inline_code, AdvancedSegmentType.INLINE_CODE),
            (self._extract_inline_math, AdvancedSegmentType.INLINE_MATH),
            (self._extract_file_paths, AdvancedSegmentType.FILE_PATH),
            (self._extract_urls, AdvancedSegmentType.URL),
            (self._extract_emails, AdvancedSegmentType.EMAIL),
            (self._extract_quoted_text, AdvancedSegmentType.QUOTED_TEXT),
            (self._extract_regex_patterns, AdvancedSegmentType.REGEX_PATTERN),
            (self._extract_date_times, AdvancedSegmentType.DATE_TIME),
            (self._extract_numbers, AdvancedSegmentType.NUMBER),
            (self._extract_variable_refs, AdvancedSegmentType.VARIABLE_REF),
            (self._extract_function_calls, AdvancedSegmentType.FUNCTION_CALL)
        ]

        # Apply each processor
        for processor, segment_type in processors:
            new_segments = processor(text)
            for segment in new_segments:
                # Check if this range overlaps with already processed content
                if not self._ranges_overlap(segment.start_pos, segment.end_pos, processed_ranges):
                    segments.append(segment)
                    processed_ranges.append((segment.start_pos, segment.end_pos))

        # Fill in remaining text segments
        segments.extend(self._extract_remaining_text(text, processed_ranges))

        # Sort segments by position
        segments.sort(key=lambda s: s.start_pos)

        return segments

    def _extract_code_blocks(self, text: str) -> List[ContentSegment]:
        """Extract fenced code blocks"""
        segments = []

        # Match fenced code blocks with optional language specification
        pattern = r'```(\w+)?\n(.*?)\n```'

        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1)
            code_content = match.group(2)
            start_pos, end_pos = match.span()

            # Analyze code complexity
            complexity = self._analyze_code_complexity(code_content)

            metadata = {
                "language": language,
                "line_count": len(code_content.split('\n')),
                "complexity": complexity,
                "has_functions": bool(re.search(r'\b(def|function|func)\s+\w+', code_content)),
                "has_classes": bool(re.search(r'\b(class|interface|struct)\s+\w+', code_content))
            }

            segments.append(ContentSegment(
                segment_type=AdvancedSegmentType.CODE_BLOCK,
                content=code_content,
                start_pos=start_pos,
                end_pos=end_pos,
                language=language,
                metadata=metadata,
                confidence=0.95
            ))

        return segments

    def _extract_math_blocks(self, text: str) -> List[ContentSegment]:
        """Extract LaTeX-style math blocks"""
        segments = []

        patterns = [
            (r'\$\$(.*?)\$\$', "display_math"),
            (r'\\\[(.*?)\\\]', "display_math"),
            (r'\\begin\{equation\}(.*?)\\end\{equation\}', "equation")
        ]

        for pattern, math_type in patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                math_content = match.group(1).strip()
                start_pos, end_pos = match.span()

                metadata = {
                    "math_type": math_type,
                    "has_fractions": '\\frac' in math_content,
                    "has_integrals": '\\int' in math_content,
                    "has_summations": '\\sum' in math_content
                }

                segments.append(ContentSegment(
                    segment_type=AdvancedSegmentType.MATH_BLOCK,
                    content=math_content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata=metadata,
                    confidence=0.9
                ))

        return segments

    def _extract_inline_code(self, text: str) -> List[ContentSegment]:
        """Extract inline code snippets"""
        segments = []

        # Pattern for backtick-enclosed code
        pattern = r'`([^`\n]+)`'

        for match in re.finditer(pattern, text):
            code_content = match.group(1)
            start_pos, end_pos = match.span()

            # Detect if it looks like actual code vs just formatted text
            code_indicators = [
                r'\w+\(',  # Function calls
                r'\w+\.\w+',  # Method/property access
                r'[=<>!]=?',  # Operators
                r'[{}[\]()]',  # Brackets
                r'[A-Z][a-z]+[A-Z]'  # CamelCase
            ]

            code_score = sum(1 for pattern in code_indicators
                           if re.search(pattern, code_content))

            confidence = min(0.9, 0.5 + code_score * 0.1)

            metadata = {
                "likely_code": code_score >= 2,
                "code_score": code_score
            }

            segments.append(ContentSegment(
                segment_type=AdvancedSegmentType.INLINE_CODE,
                content=code_content,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata=metadata,
                confidence=confidence
            ))

        return segments

    def _extract_inline_math(self, text: str) -> List[ContentSegment]:
        """Extract inline mathematical expressions"""
        segments = []

        # Pattern for inline math
        pattern = r'\$([^$\n]+)\$'

        for match in re.finditer(pattern, text):
            math_content = match.group(1)
            start_pos, end_pos = match.span()

            # Check if it actually looks like math
            math_indicators = [
                r'[+\-*/=]',  # Math operators
                r'[αβγδεθλμπσφψω]',  # Greek letters
                r'\\[a-zA-Z]+',  # LaTeX commands
                r'\b\d+\.\d+\b',  # Decimals
                r'[₀₁₂₃₄₅₆₇₈₉]',  # Subscripts
                r'[⁰¹²³⁴⁵⁶⁷⁸⁹]'  # Superscripts
            ]

            math_score = sum(1 for pattern in math_indicators
                           if re.search(pattern, math_content))

            confidence = min(0.9, 0.4 + math_score * 0.15)

            if confidence > 0.5:  # Only include if likely math
                segments.append(ContentSegment(
                    segment_type=AdvancedSegmentType.INLINE_MATH,
                    content=math_content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=confidence
                ))

        return segments

    def _extract_file_paths(self, text: str) -> List[ContentSegment]:
        """Extract file paths and directory references"""
        segments = []

        patterns = [
            r'\b[a-zA-Z]:[\\\/][\w\\\/.-]+\.\w{2,5}',  # Windows absolute paths
            r'\/[\w\/.-]+\.\w{2,5}',  # Unix absolute paths
            r'\.\/[\w\/.-]+',  # Relative paths starting with ./
            r'\.\.\/[\w\/.-]+',  # Relative paths starting with ../
            r'\b[\w.-]+\/[\w\/.-]*\w',  # Directory-like patterns
            r'\b\w+\.\w{2,5}\b'  # Simple filenames
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                path_content = match.group(0)
                start_pos, end_pos = match.span()

                # Analyze path characteristics
                is_absolute = path_content.startswith(('/', '\\', 'C:', 'D:'))
                has_extension = '.' in path_content.split('/')[-1]

                metadata = {
                    "is_absolute": is_absolute,
                    "has_extension": has_extension,
                    "extension": path_content.split('.')[-1] if has_extension else None,
                    "depth": path_content.count('/') + path_content.count('\\')
                }

                segments.append(ContentSegment(
                    segment_type=AdvancedSegmentType.FILE_PATH,
                    content=path_content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata=metadata,
                    confidence=0.8
                ))

        return segments

    def _extract_urls(self, text: str) -> List[ContentSegment]:
        """Extract URLs and web addresses"""
        segments = []

        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+|www\.[^\s<>"{}|\\^`[\]]+'

        for match in re.finditer(url_pattern, text):
            url_content = match.group(0)
            start_pos, end_pos = match.span()

            metadata = {
                "protocol": "https" if url_content.startswith("https") else "http" if url_content.startswith("http") else "www",
                "domain": url_content.split('/')[2] if '://' in url_content else url_content.split('/')[0]
            }

            segments.append(ContentSegment(
                segment_type=AdvancedSegmentType.URL,
                content=url_content,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata=metadata,
                confidence=0.95
            ))

        return segments

    def _extract_emails(self, text: str) -> List[ContentSegment]:
        """Extract email addresses"""
        segments = []

        for match in re.finditer(self.pii_patterns["email"], text):
            email_content = match.group(0)
            start_pos, end_pos = match.span()

            metadata = {
                "domain": email_content.split('@')[1],
                "is_pii": True
            }

            segments.append(ContentSegment(
                segment_type=AdvancedSegmentType.EMAIL,
                content=email_content,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata=metadata,
                confidence=0.9
            ))

        return segments

    def _extract_quoted_text(self, text: str) -> List[ContentSegment]:
        """Extract quoted text sections"""
        segments = []

        patterns = [
            r'"([^"]*)"',  # Double quotes
            r"'([^']*)'",  # Single quotes
            r'`([^`]*)`'   # Backticks (if not already processed as code)
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                quoted_content = match.group(1)
                start_pos, end_pos = match.span()

                # Skip empty quotes
                if not quoted_content.strip():
                    continue

                metadata = {
                    "quote_style": pattern[0],
                    "word_count": len(quoted_content.split())
                }

                segments.append(ContentSegment(
                    segment_type=AdvancedSegmentType.QUOTED_TEXT,
                    content=quoted_content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata=metadata,
                    confidence=0.8
                ))

        return segments

    def _extract_regex_patterns(self, text: str) -> List[ContentSegment]:
        """Extract regex patterns"""
        segments = []

        # Look for regex-like patterns
        regex_indicators = [
            r'[\[\](){}^$.*+?|\\-]',
            r'\\\w\+?',
            r'\[.*\]',
            r'\{.*,.*\}'
        ]

        # Find potential regex patterns
        words = text.split()
        for i, word in enumerate(words):
            if len(word) > 3:  # Minimum length
                regex_score = sum(1 for pattern in regex_indicators
                                if re.search(pattern, word))

                if regex_score >= 2:  # Threshold for regex-like appearance
                    # Find position in original text
                    start_pos = text.find(word)
                    if start_pos != -1:
                        metadata = {
                            "regex_score": regex_score,
                            "likely_regex": regex_score >= 3
                        }

                        segments.append(ContentSegment(
                            segment_type=AdvancedSegmentType.REGEX_PATTERN,
                            content=word,
                            start_pos=start_pos,
                            end_pos=start_pos + len(word),
                            metadata=metadata,
                            confidence=min(0.9, 0.4 + regex_score * 0.15)
                        ))

        return segments

    def _extract_date_times(self, text: str) -> List[ContentSegment]:
        """Extract date and time expressions"""
        segments = []

        patterns = [
            (r'\b\d{4}-\d{2}-\d{2}\b', "iso_date"),
            (r'\b\d{1,2}\/\d{1,2}\/\d{4}\b', "us_date"),
            (r'\b\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\b', "time"),
            (r'\b(today|tomorrow|yesterday)\b', "relative_date"),
            (r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', "weekday")
        ]

        for pattern, date_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                datetime_content = match.group(0)
                start_pos, end_pos = match.span()

                metadata = {
                    "date_type": date_type,
                    "needs_parsing": date_type in ["relative_date", "us_date"]
                }

                segments.append(ContentSegment(
                    segment_type=AdvancedSegmentType.DATE_TIME,
                    content=datetime_content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata=metadata,
                    confidence=0.85
                ))

        return segments

    def _extract_numbers(self, text: str) -> List[ContentSegment]:
        """Extract numeric values"""
        segments = []

        patterns = [
            (r'\b\d+\.\d+\b', "decimal"),
            (r'\b\d+%\b', "percentage"),
            (r'\$\d+(\.\d{2})?\b', "currency"),
            (r'\b\d{1,3}(,\d{3})*\b', "large_number"),
            (r'\b\d+\b', "integer")
        ]

        for pattern, number_type in patterns:
            for match in re.finditer(pattern, text):
                number_content = match.group(0)
                start_pos, end_pos = match.span()

                metadata = {
                    "number_type": number_type,
                    "value": number_content
                }

                segments.append(ContentSegment(
                    segment_type=AdvancedSegmentType.NUMBER,
                    content=number_content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata=metadata,
                    confidence=0.9
                ))

        return segments

    def _extract_variable_refs(self, text: str) -> List[ContentSegment]:
        """Extract variable and identifier references"""
        segments = []

        patterns = [
            r'\b[a-z][a-zA-Z0-9_]*\b',  # camelCase/snake_case variables
            r'\b[A-Z][A-Z0-9_]*\b',  # CONSTANTS
            r'\$\w+\b',  # Shell/PHP variables
            r'@\w+\b'   # Decorator/annotation references
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                var_content = match.group(0)
                start_pos, end_pos = match.span()

                # Skip common English words
                common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                              'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day',
                              'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new',
                              'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'use'}

                if var_content.lower() not in common_words:
                    metadata = {
                        "naming_convention": self._detect_naming_convention(var_content),
                        "likely_variable": True
                    }

                    segments.append(ContentSegment(
                        segment_type=AdvancedSegmentType.VARIABLE_REF,
                        content=var_content,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        metadata=metadata,
                        confidence=0.7
                    ))

        return segments

    def _extract_function_calls(self, text: str) -> List[ContentSegment]:
        """Extract function call patterns"""
        segments = []

        pattern = r'\b\w+\s*\([^)]*\)'

        for match in re.finditer(pattern, text):
            func_content = match.group(0)
            start_pos, end_pos = match.span()

            func_name = func_content.split('(')[0].strip()

            metadata = {
                "function_name": func_name,
                "has_parameters": ',' in func_content or len(func_content.split('(')[1].strip()) > 1
            }

            segments.append(ContentSegment(
                segment_type=AdvancedSegmentType.FUNCTION_CALL,
                content=func_content,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata=metadata,
                confidence=0.8
            ))

        return segments

    def _extract_remaining_text(self, text: str, processed_ranges: List[Tuple[int, int]]) -> List[ContentSegment]:
        """Extract remaining text that wasn't captured by other processors"""
        segments = []

        # Sort processed ranges
        processed_ranges.sort(key=lambda x: x[0])

        current_pos = 0

        for start, end in processed_ranges:
            # Add text before this processed range
            if current_pos < start:
                text_content = text[current_pos:start].strip()
                if text_content:
                    segments.append(ContentSegment(
                        segment_type=AdvancedSegmentType.TEXT,
                        content=text_content,
                        start_pos=current_pos,
                        end_pos=start,
                        confidence=1.0
                    ))
            current_pos = max(current_pos, end)

        # Add any remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                segments.append(ContentSegment(
                    segment_type=AdvancedSegmentType.TEXT,
                    content=remaining_text,
                    start_pos=current_pos,
                    end_pos=len(text),
                    confidence=1.0
                ))

        return segments

    # Helper methods

    def _ranges_overlap(self, start: int, end: int, ranges: List[Tuple[int, int]]) -> bool:
        """Check if a range overlaps with any existing ranges"""
        for r_start, r_end in ranges:
            if not (end <= r_start or start >= r_end):
                return True
        return False

    def _analyze_code_complexity(self, code: str) -> float:
        """Analyze code complexity score"""
        complexity_score = 0.0

        # Count control structures
        control_patterns = [
            r'\b(if|else|elif|for|while|switch|case)\b',
            r'\btry\b.*\bcatch\b',
            r'\bfunction\b|\bdef\b|\bclass\b'
        ]

        for pattern in control_patterns:
            matches = len(re.findall(pattern, code, re.IGNORECASE))
            complexity_score += matches * 0.1

        # Count nesting level (approximation)
        nesting_level = max(code.count('{'), code.count('    ')) / 4
        complexity_score += nesting_level * 0.2

        return min(1.0, complexity_score)

    def _detect_naming_convention(self, identifier: str) -> str:
        """Detect the naming convention used"""
        if '_' in identifier:
            return "snake_case"
        elif identifier.isupper():
            return "UPPER_CASE"
        elif identifier[0].isupper():
            return "PascalCase"
        elif any(c.isupper() for c in identifier[1:]):
            return "camelCase"
        else:
            return "lowercase"

    def _detect_profanity(self, text: str) -> List[str]:
        """Detect profanity in text (placeholder implementation)"""
        flags = []
        words = text.lower().split()

        for word in words:
            if word in self.profanity_words:
                flags.append(word)

        return flags

    def _detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect personally identifiable information"""
        pii_detected = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                pii_detected.append({
                    "type": pii_type,
                    "value": match.group(0),
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "should_redact": True
                })

        return pii_detected

    def _detect_commands(self, text: str) -> bool:
        """Detect command-like input"""
        for pattern in self.command_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _calculate_urgency(self, text: str) -> float:
        """Calculate urgency score based on text indicators"""
        urgency_score = 0.0

        for pattern, weight in self.urgency_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            urgency_score += matches * weight

        return min(1.0, urgency_score)

    def _calculate_complexity(self, text: str) -> float:
        """Calculate complexity score based on text indicators"""
        complexity_score = 0.5  # Base complexity

        for pattern, weight in self.complexity_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            complexity_score += matches * weight

        # Adjust based on text length
        word_count = len(text.split())
        if word_count > 50:
            complexity_score += 0.2
        elif word_count < 10:
            complexity_score -= 0.2

        return min(1.0, max(0.0, complexity_score))

    def get_segment_summary(self, processed_turn: ProcessedTurn) -> Dict[str, Any]:
        """Generate a summary of the segmented content"""

        segment_counts = {}
        for segment in processed_turn.segments:
            segment_type = segment.segment_type.value
            segment_counts[segment_type] = segment_counts.get(segment_type, 0) + 1

        return {
            "turn_id": processed_turn.turn_id,
            "total_segments": len(processed_turn.segments),
            "segment_counts": segment_counts,
            "detected_language": {
                "language": processed_turn.detected_language.language,
                "confidence": processed_turn.detected_language.confidence
            },
            "flags": {
                "has_profanity": len(processed_turn.profanity_flags) > 0,
                "has_pii": len(processed_turn.pii_detected) > 0,
                "is_command": processed_turn.command_detected,
                "urgency_score": processed_turn.urgency_score,
                "complexity_score": processed_turn.complexity_score
            },
            "content_analysis": {
                "has_code": any(s.segment_type in [AdvancedSegmentType.CODE_BLOCK,
                                                  AdvancedSegmentType.INLINE_CODE]
                               for s in processed_turn.segments),
                "has_math": any(s.segment_type in [AdvancedSegmentType.MATH_BLOCK,
                                                  AdvancedSegmentType.INLINE_MATH]
                               for s in processed_turn.segments),
                "has_files": any(s.segment_type == AdvancedSegmentType.FILE_PATH
                                for s in processed_turn.segments),
                "has_urls": any(s.segment_type == AdvancedSegmentType.URL
                               for s in processed_turn.segments)
            }
        }

# Demo/Test function
def demo_advanced_processing():
    """Demonstrate the advanced input processing capabilities"""

    processor = AdvancedInputProcessor()

    test_inputs = [
        "Can you write a regex for dates like 2025-09-28?",

        '''Create a Python function to validate emails:
```python
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[A-Z|a-z]{2,}$'
    return bool(re.match(pattern, email))
```
''',

        "I need URGENT help with fixing the file ./src/components/UserProfile.tsx - it's broken!",

        "Analyze the function `calculateTotalPrice(items, taxRate)` and optimize performance",

        "The math formula is $\\int_0^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$ - can you explain it?",

        "/create a REST API endpoint for user authentication with JWT tokens"
    ]

    print("=" * 80)
    print("ADVANCED INPUT PROCESSING DEMO")
    print("=" * 80)

    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n🧠 Test Input {i}:")
        print(f"'{test_input[:60]}{'...' if len(test_input) > 60 else ''}'")

        # Process the input
        processed = processor.process_input(test_input)
        summary = processor.get_segment_summary(processed)

        print(f"\n📊 Processing Results:")
        print(f"   • Language: {summary['detected_language']['language']} "
              f"({summary['detected_language']['confidence']:.2f})")
        print(f"   • Segments: {summary['total_segments']} total")
        print(f"   • Types: {list(summary['segment_counts'].keys())}")
        print(f"   • Urgency: {summary['flags']['urgency_score']:.2f}")
        print(f"   • Complexity: {summary['flags']['complexity_score']:.2f}")

        if summary['flags']['is_command']:
            print("   • ⚡ Command detected")
        if summary['content_analysis']['has_code']:
            print("   • 💻 Contains code")
        if summary['content_analysis']['has_math']:
            print("   • 🧮 Contains math")
        if summary['flags']['has_pii']:
            print("   • 🔒 PII detected")

        print("   " + "-" * 50)

if __name__ == "__main__":
    demo_advanced_processing()
