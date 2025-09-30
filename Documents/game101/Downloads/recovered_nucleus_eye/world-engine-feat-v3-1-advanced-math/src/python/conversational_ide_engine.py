# CONVERSATIONAL IDE ENGINE
# INGEST → UNDERSTAND → PLAN → RESPOND with CONTEXT & POLICY guardrails
# Based on the blueprint for teaching IDEs to understand English and maintain coherent conversations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, NamedTuple
from enum import Enum
import unicodedata
import pytz

class DialogueAct(Enum):
    """Dialogue acts for conversation understanding"""
    QUESTION = "QUESTION"
    REQUEST = "REQUEST"
    INFORM = "INFORM"
    CRITIQUE = "CRITIQUE"
    META = "META"

class Intent(Enum):
    """User intents for action classification"""
    CODE_GENERATION = "CODE_GENERATION"
    EXPLAIN = "EXPLAIN"
    DEBUG = "DEBUG"
    SUMMARIZE = "SUMMARIZE"
    SEARCH = "SEARCH"
    REFACTOR = "REFACTOR"
    ANALYZE = "ANALYZE"
    PLAN = "PLAN"
    GENERAL = "GENERAL"

class PolicyAction(Enum):
    """Policy gate decisions"""
    ALLOW = "ALLOW"
    ALLOW_WITH_REDACTION = "ALLOW_WITH_REDACTION"
    REFUSE_WITH_REASON = "REFUSE_WITH_REASON"

class SegmentType(Enum):
    """Content segment types"""
    TEXT = "text"
    CODE = "code"
    MATH = "math"
    COMMAND = "command"

# Data structures for the conversation pipeline

class Segment(NamedTuple):
    """Content segment with type and metadata"""
    type: SegmentType
    content: str
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Turn(NamedTuple):
    """Normalized user turn"""
    turn_id: str
    timestamp: str
    raw: str
    segments: List[Segment]
    language: str = "en"
    profanity_flags: List[str] = []
    pii_redacted: bool = False

class Entity(NamedTuple):
    """Extracted entity with type and metadata"""
    name: str
    entity_type: str
    confidence: float = 1.0
    aliases: List[str] = []
    metadata: Optional[Dict[str, Any]] = None

class Understanding(NamedTuple):
    """Turn comprehension result"""
    act: DialogueAct
    intents: List[Intent]
    entities: List[Entity]
    constraints: Dict[str, Any]
    confidence: float
    ambiguity_score: float

class Evidence(NamedTuple):
    """Grounding evidence from retrieval"""
    source: str
    parts: List[str]
    snippet: str
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

class PolicyDecision(NamedTuple):
    """Policy gate decision"""
    action: PolicyAction
    reason: Optional[str] = None
    redactions: List[str] = []

class PlanSection(NamedTuple):
    """Response plan section"""
    section_type: str
    title: Optional[str]
    content: Any
    metadata: Optional[Dict[str, Any]] = None

class ResponsePlan(NamedTuple):
    """Complete response plan"""
    style: str
    sections: List[PlanSection]
    followup_question: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []

class ConversationState:
    """Persistent conversation state"""

    def __init__(self):
        # Working Memory - last N turns
        self.topic_stack: List[str] = []
        self.current_file: Optional[str] = None
        self.active_function: Optional[str] = None
        self.turn_history: List[Turn] = []

        # Long-Term Memory
        self.user_profile: Dict[str, Any] = {
            "tone": "factual-forward",
            "ask_rate": "minimal",
            "preferred_languages": ["python", "typescript", "csharp"],
            "style_preferences": []
        }

        # Entity Ledger
        self.entities: Dict[str, Entity] = {}
        self.entity_aliases: Dict[str, str] = {}  # alias -> canonical_name

        # Commitments & Timeline
        self.commitments: List[Dict[str, Any]] = []
        self.timebase = pytz.timezone("America/Chicago")

        # Session metadata
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now(self.timebase).isoformat()

class ConversationalIDEEngine:
    """Main engine implementing INGEST → UNDERSTAND → PLAN → RESPOND"""

    def __init__(self):
        self.state = ConversationState()

        # Classification rules for speech acts and intents
        self.act_rules = [
            (re.compile(r'^(make|build|write|generate|create)\b', re.I), DialogueAct.REQUEST, Intent.CODE_GENERATION),
            (re.compile(r'\b(why|how|what|when|where)\??$', re.I), DialogueAct.QUESTION, Intent.EXPLAIN),
            (re.compile(r'^(fix|debug|refactor)\b', re.I), DialogueAct.REQUEST, Intent.DEBUG),
            (re.compile(r'^(explain|describe|tell me about)\b', re.I), DialogueAct.QUESTION, Intent.EXPLAIN),
            (re.compile(r'^(search|find|look for)\b', re.I), DialogueAct.REQUEST, Intent.SEARCH),
            (re.compile(r'^(analyze|review|check)\b', re.I), DialogueAct.REQUEST, Intent.ANALYZE),
            (re.compile(r'^(plan|design|architect)\b', re.I), DialogueAct.REQUEST, Intent.PLAN),
            (re.compile(r'^(summarize|sum up)\b', re.I), DialogueAct.REQUEST, Intent.SUMMARIZE)
        ]

        # Entity extraction patterns
        self.entity_patterns = [
            (re.compile(r'\b([A-Z][a-zA-Z0-9]+)\b'), "class_or_type"),  # CamelCase
            (re.compile(r'\b([a-z_][a-z0-9_]+)\b'), "variable_or_function"),  # snake_case
            (re.compile(r'\b(\w+\.\w{1,5})\b'), "file"),  # file.ext
            (re.compile(r'`([^`]+)`'), "code_reference"),  # `code`
            (re.compile(r'"([^"]+)"'), "quoted_text")  # "text"
        ]

        # Teaching patterns for response generation
        self.teaching_patterns = {
            Intent.EXPLAIN: ["definition", "steps", "example", "pitfalls"],
            Intent.CODE_GENERATION: ["intro", "code", "usage", "guardrails"],
            Intent.DEBUG: ["repro_steps", "hypotheses", "fix", "verify"],
            Intent.SUMMARIZE: ["one_liner", "bullets", "source_evidence"],
            Intent.PLAN: ["goals", "options", "tradeoffs", "recommendation"]
        }

    # 1. INGEST - Normalize raw input
    def ingest(self, raw_text: str) -> Turn:
        """Normalize raw input into structured turn"""
        turn_id = f"t-{datetime.now().strftime('%Y-%m-%d')}-{len(self.state.turn_history):03d}"
        timestamp = datetime.now(self.state.timebase).isoformat()

        # Normalize Unicode and clean whitespace
        normalized = unicodedata.normalize('NFKC', raw_text)
        normalized = re.sub(r'\s+', ' ', normalized.strip())

        # Segment content
        segments = self._segment_content(normalized)

        # Basic language detection (assume English default)
        language = "en"

        # Basic profanity/PII detection (placeholder)
        profanity_flags = []
        pii_redacted = False

        turn = Turn(
            turn_id=turn_id,
            timestamp=timestamp,
            raw=normalized,
            segments=segments,
            language=language,
            profanity_flags=profanity_flags,
            pii_redacted=pii_redacted
        )

        return turn

    def _segment_content(self, text: str) -> List[Segment]:
        """Segment text into code blocks, math, commands, and prose"""
        segments = []

        # Extract code blocks
        code_pattern = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
        last_end = 0

        for match in code_pattern.finditer(text):
            start, end = match.span()

            # Add text before code block
            if start > last_end:
                text_content = text[last_end:start].strip()
                if text_content:
                    segments.append(Segment(SegmentType.TEXT, text_content))

            # Add code block
            language = match.group(1) or None
            code_content = match.group(2)
            segments.append(Segment(SegmentType.CODE, code_content, language))

            last_end = end

        # Add remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                segments.append(Segment(SegmentType.TEXT, remaining))

        # If no segments found, treat entire input as text
        if not segments:
            segments.append(Segment(SegmentType.TEXT, text))

        return segments

    # 2. UNDERSTAND - Parse turn into structured comprehension
    def understand(self, turn: Turn) -> Understanding:
        """Parse turn comprehension: act, intents, entities, constraints"""

        # Combine text segments for analysis
        text_content = " ".join(
            segment.content for segment in turn.segments
            if segment.type == SegmentType.TEXT
        )

        # Classify dialogue act and primary intent
        act, primary_intent = self._classify_act_and_intent(text_content)

        # Extract all intents
        intents = self._extract_intents(text_content, primary_intent)

        # Extract entities
        entities = self._extract_entities(turn)

        # Parse constraints
        constraints = self._parse_constraints(turn, entities)

        # Calculate confidence and ambiguity
        confidence = self._calculate_confidence(act, intents, entities)
        ambiguity_score = self._calculate_ambiguity(text_content, entities)

        return Understanding(
            act=act,
            intents=intents,
            entities=entities,
            constraints=constraints,
            confidence=confidence,
            ambiguity_score=ambiguity_score
        )

    def _classify_act_and_intent(self, text: str) -> Tuple[DialogueAct, Intent]:
        """Classify dialogue act and primary intent using rules + patterns"""

        # Apply classification rules
        for pattern, act, intent in self.act_rules:
            if pattern.search(text):
                return act, intent

        # Default classification
        if '?' in text:
            return DialogueAct.QUESTION, Intent.EXPLAIN
        else:
            return DialogueAct.INFORM, Intent.GENERAL

    def _extract_intents(self, text: str, primary: Intent) -> List[Intent]:
        """Extract all relevant intents from text"""
        intents = [primary]

        # Look for secondary intents
        intent_keywords = {
            Intent.CODE_GENERATION: ["code", "function", "class", "script", "program"],
            Intent.EXPLAIN: ["explain", "understand", "clarify", "describe"],
            Intent.DEBUG: ["error", "bug", "issue", "problem", "fix"],
            Intent.ANALYZE: ["analyze", "review", "examine", "inspect"],
            Intent.SEARCH: ["find", "search", "locate", "look for"]
        }

        for intent, keywords in intent_keywords.items():
            if intent != primary and any(kw in text.lower() for kw in keywords):
                intents.append(intent)

        return intents

    def _extract_entities(self, turn: Turn) -> List[Entity]:
        """Extract entities from turn content"""
        entities = []

        # Extract from all text segments
        for segment in turn.segments:
            if segment.type in [SegmentType.TEXT, SegmentType.CODE]:
                for pattern, entity_type in self.entity_patterns:
                    for match in pattern.finditer(segment.content):
                        name = match.group(1)

                        # Skip common words
                        if name.lower() in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all']:
                            continue

                        entity = Entity(
                            name=name,
                            entity_type=entity_type,
                            confidence=0.8
                        )
                        entities.append(entity)

        # Deduplicate entities
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.name not in seen:
                seen.add(entity.name)
                unique_entities.append(entity)

        return unique_entities

    def _parse_constraints(self, turn: Turn, entities: List[Entity]) -> Dict[str, Any]:
        """Parse constraints from turn and entities"""
        constraints = {}

        # Detect output format preferences
        has_code = any(seg.type == SegmentType.CODE for seg in turn.segments)
        if has_code:
            constraints["format"] = "code"

        # Detect language preferences
        code_segments = [seg for seg in turn.segments if seg.type == SegmentType.CODE]
        if code_segments and code_segments[0].language:
            constraints["language"] = code_segments[0].language

        # Detect style constraints
        text_lower = turn.raw.lower()
        if any(word in text_lower for word in ["simple", "basic", "quick"]):
            constraints["style"] = "concise"
        elif any(word in text_lower for word in ["detailed", "complete", "comprehensive"]):
            constraints["style"] = "detailed"

        return constraints

    def _calculate_confidence(self, act: DialogueAct, intents: List[Intent], entities: List[Entity]) -> float:
        """Calculate confidence in understanding"""
        base_confidence = 0.7

        # Boost confidence for clear patterns
        if len(intents) == 1:
            base_confidence += 0.1

        # Boost for recognized entities
        if entities:
            base_confidence += min(0.2, len(entities) * 0.05)

        return min(1.0, base_confidence)

    def _calculate_ambiguity(self, text: str, entities: List[Entity]) -> float:
        """Calculate ambiguity score"""
        base_ambiguity = 0.3

        # Reduce ambiguity with specific entities
        if entities:
            base_ambiguity -= min(0.2, len(entities) * 0.05)

        # Increase ambiguity for vague language
        vague_words = ["something", "anything", "stuff", "thing", "it"]
        vague_count = sum(1 for word in vague_words if word in text.lower())
        base_ambiguity += vague_count * 0.1

        return max(0.0, min(1.0, base_ambiguity))

    # 3. CONTEXT - Bind understanding to conversation context
    def bind_context(self, understanding: Understanding, turn: Turn) -> Understanding:
        """Bind understanding to conversation context and update state"""

        # Update entity ledger
        for entity in understanding.entities:
            self._update_entity_ledger(entity)

        # Update topic stack
        self._update_topic_stack(understanding, turn)

        # Resolve pronouns and references (basic implementation)
        resolved_entities = self._resolve_references(understanding.entities)

        # Update turn history
        self.state.turn_history.append(turn)
        if len(self.state.turn_history) > 20:  # Keep last 20 turns
            self.state.turn_history.pop(0)

        # Return understanding with resolved entities
        return understanding._replace(entities=resolved_entities)

    def _update_entity_ledger(self, entity: Entity):
        """Update the entity ledger with new entity"""
        canonical_name = entity.name

        # Check for existing entity
        if canonical_name in self.state.entities:
            # Update existing entity
            existing = self.state.entities[canonical_name]
            # Merge aliases, update confidence, etc.
            updated_aliases = list(set(existing.aliases + entity.aliases))
            self.state.entities[canonical_name] = existing._replace(
                aliases=updated_aliases,
                confidence=max(existing.confidence, entity.confidence)
            )
        else:
            # Add new entity
            self.state.entities[canonical_name] = entity

        # Update alias mapping
        for alias in entity.aliases:
            self.state.entity_aliases[alias] = canonical_name

    def _update_topic_stack(self, understanding: Understanding, turn: Turn):
        """Update conversation topic stack"""

        # Extract topics from entities and intents
        topics = []

        # Add entities as topics
        for entity in understanding.entities:
            if entity.entity_type in ["class_or_type", "file"]:
                topics.append(entity.name.lower())

        # Add intent-based topics
        intent_topics = {
            Intent.CODE_GENERATION: "coding",
            Intent.DEBUG: "debugging",
            Intent.EXPLAIN: "explanation",
            Intent.ANALYZE: "analysis"
        }

        for intent in understanding.intents:
            if intent in intent_topics:
                topics.append(intent_topics[intent])

        # Update stack (keep most recent topics at front)
        for topic in topics:
            if topic in self.state.topic_stack:
                self.state.topic_stack.remove(topic)
            self.state.topic_stack.insert(0, topic)

        # Keep stack size manageable
        self.state.topic_stack = self.state.topic_stack[:10]

    def _resolve_references(self, entities: List[Entity]) -> List[Entity]:
        """Resolve pronouns and aliases to canonical entities"""
        resolved = []

        for entity in entities:
            # Check if entity is an alias
            canonical_name = self.state.entity_aliases.get(entity.name, entity.name)

            if canonical_name in self.state.entities:
                # Use canonical entity
                canonical_entity = self.state.entities[canonical_name]
                resolved.append(canonical_entity)
            else:
                # Keep original entity
                resolved.append(entity)

        return resolved

    # Main conversation pipeline
    def plan_and_respond(self, user_input: str) -> Dict[str, Any]:
        """Main pipeline: INGEST → UNDERSTAND → PLAN → RESPOND"""

        try:
            # 1. INGEST
            turn = self.ingest(user_input)

            # 2. UNDERSTAND
            understanding = self.understand(turn)

            # 3. CONTEXT
            contextual_understanding = self.bind_context(understanding, turn)

            # 4. GROUNDING (placeholder - would integrate with LLEX/Upflow)
            evidence = self._retrieve_grounding(contextual_understanding)

            # 5. POLICY GATE
            policy_decision = self._policy_gate(contextual_understanding, evidence)

            if policy_decision.action == PolicyAction.REFUSE_WITH_REASON:
                return self._create_refusal_response(policy_decision.reason)

            # 6. PLAN
            response_plan = self._make_plan(contextual_understanding, evidence, policy_decision)

            # 7. RESPOND
            response_text = self._realize_response(response_plan, contextual_understanding)

            # 8. QUALITY CHECKS
            checked_response = self._quality_checks(response_text, contextual_understanding, evidence)

            # 9. UPDATE MEMORY
            self._update_memory(contextual_understanding, response_text)

            return {
                "success": True,
                "response": checked_response,
                "turn_id": turn.turn_id,
                "understanding": {
                    "act": understanding.act.value,
                    "intents": [intent.value for intent in understanding.intents],
                    "entities": [{"name": e.name, "type": e.entity_type} for e in understanding.entities],
                    "confidence": understanding.confidence
                },
                "plan": {
                    "style": response_plan.style,
                    "sections": len(response_plan.sections),
                    "followup": response_plan.followup_question
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error processing your request. Please try rephrasing your question."
            }

    # Placeholder implementations for remaining pipeline stages

    def _retrieve_grounding(self, understanding: Understanding) -> List[Evidence]:
        """Retrieve grounding evidence (placeholder for LLEX/Upflow integration)"""
        # This would integrate with your existing LLEX/Upflow system
        return []

    def _policy_gate(self, understanding: Understanding, evidence: List[Evidence]) -> PolicyDecision:
        """Apply policy checks"""
        # Basic safety checks
        text_lower = " ".join(
            entity.name.lower() for entity in understanding.entities
        )

        # Check for potentially harmful requests
        harmful_patterns = ["delete", "remove", "destroy", "hack", "exploit"]
        if any(pattern in text_lower for pattern in harmful_patterns):
            return PolicyDecision(
                action=PolicyAction.REFUSE_WITH_REASON,
                reason="I cannot assist with potentially harmful operations."
            )

        return PolicyDecision(action=PolicyAction.ALLOW)

    def _make_plan(self, understanding: Understanding, evidence: List[Evidence], policy: PolicyDecision) -> ResponsePlan:
        """Create response plan"""

        primary_intent = understanding.intents[0] if understanding.intents else Intent.GENERAL

        # Get teaching pattern for intent
        pattern = self.teaching_patterns.get(primary_intent, ["response"])

        # Create sections based on pattern
        sections = []

        if primary_intent == Intent.CODE_GENERATION:
            sections.extend([
                PlanSection("intro", "Brief Introduction", "I'll create code for your request."),
                PlanSection("code", "Code Block", {"language": understanding.constraints.get("language", "python")}),
                PlanSection("usage", "Usage Notes", "Here's how to use this code."),
                PlanSection("guardrails", "Important Notes", "Keep these limitations in mind.")
            ])
        elif primary_intent == Intent.EXPLAIN:
            sections.extend([
                PlanSection("definition", "Definition", "Let me explain what this means."),
                PlanSection("steps", "Step by Step", []),
                PlanSection("example", "Example", "Here's a practical example."),
                PlanSection("pitfalls", "Common Pitfalls", "Watch out for these issues.")
            ])
        else:
            # General response
            sections.append(PlanSection("response", "Response", "I'll help you with that."))

        # Add followup question if ambiguity is high
        followup_question = None
        if understanding.ambiguity_score > 0.6:
            followup_question = "Would you like me to provide more specific details or examples?"

        return ResponsePlan(
            style=self.state.user_profile.get("tone", "factual-forward"),
            sections=sections,
            followup_question=followup_question
        )

    def _realize_response(self, plan: ResponsePlan, understanding: Understanding) -> str:
        """Generate natural language response from plan"""

        response_parts = []

        for section in plan.sections:
            if section.section_type == "intro":
                response_parts.append(section.content)

            elif section.section_type == "code":
                # Generate code based on understanding
                code = self._generate_code(understanding, section.content)
                response_parts.append(f"```{section.content.get('language', 'python')}\n{code}\n```")

            elif section.section_type == "definition":
                # Generate definition
                entities = [e.name for e in understanding.entities]
                if entities:
                    response_parts.append(f"{entities[0]} is a programming concept that...")
                else:
                    response_parts.append("This concept refers to...")

            elif section.section_type == "steps":
                response_parts.append("Here are the key steps:\n1. First step\n2. Second step\n3. Final step")

            elif section.section_type == "example":
                response_parts.append("For example, if you have...")

            elif section.section_type == "response":
                response_parts.append(section.content)

        # Join parts with appropriate spacing
        response = "\n\n".join(response_parts)

        # Add followup question if present
        if plan.followup_question:
            response += f"\n\n{plan.followup_question}"

        return response

    def _generate_code(self, understanding: Understanding, code_metadata: Dict[str, Any]) -> str:
        """Generate code based on understanding"""

        language = code_metadata.get("language", "python")
        entities = understanding.entities

        if Intent.CODE_GENERATION in understanding.intents:
            if "regex" in str(understanding.entities).lower():
                if language == "python":
                    return """import re

# Regex pattern for YYYY-MM-DD date format
date_pattern = r'^\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])$'

def validate_date_format(date_string):
    return bool(re.match(date_pattern, date_string))

# Example usage
print(validate_date_format("2025-09-28"))  # True
print(validate_date_format("2025-13-28"))  # False"""
                else:
                    return "^\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])$"

            # Generic function template
            if language == "python":
                return """def example_function():
    \"\"\"Example function based on your request\"\"\"
    # Implementation here
    return "result" """
            elif language == "typescript":
                return """function exampleFunction(): string {
    // Implementation here
    return "result";
}"""
            elif language == "csharp":
                return """public class ExampleClass
{
    public string ExampleMethod()
    {
        // Implementation here
        return "result";
    }
}"""

        return "// Code implementation based on your request"

    def _quality_checks(self, response: str, understanding: Understanding, evidence: List[Evidence]) -> str:
        """Perform quality checks and repairs"""

        # Basic quality checks
        checked_response = response

        # Ensure code blocks are properly formatted
        if "```" in response:
            # Basic code block validation
            code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response, re.DOTALL)
            if code_blocks:
                # Validate syntax (placeholder)
                pass

        # Ensure all intents were addressed
        intent_keywords = {
            Intent.EXPLAIN: ["explain", "definition", "means"],
            Intent.CODE_GENERATION: ["```", "function", "code"],
            Intent.DEBUG: ["fix", "error", "issue"]
        }

        for intent in understanding.intents:
            if intent in intent_keywords:
                keywords = intent_keywords[intent]
                if not any(kw in checked_response.lower() for kw in keywords):
                    # Add missing coverage (basic repair)
                    if intent == Intent.EXPLAIN:
                        checked_response += "\n\nTo explain further: this addresses your question about the topic."

        return checked_response

    def _update_memory(self, understanding: Understanding, response: str):
        """Update conversation memory"""

        # Update user preferences based on interaction
        if Intent.CODE_GENERATION in understanding.intents:
            # User seems to like code generation
            prefs = self.state.user_profile.get("style_preferences", [])
            if "code_focused" not in prefs:
                prefs.append("code_focused")
                self.state.user_profile["style_preferences"] = prefs

        # Track successful interactions
        # This would be expanded for learning and adaptation

    def _create_refusal_response(self, reason: str) -> Dict[str, Any]:
        """Create response for refused requests"""
        return {
            "success": False,
            "response": f"I'm sorry, but I cannot help with that request. {reason}",
            "refused": True,
            "reason": reason
        }

# Interactive CLI for testing
def main():
    """Interactive CLI for testing the conversational engine"""

    print("=" * 60)
    print("CONVERSATIONAL IDE ENGINE - Interactive Test Mode")
    print("=" * 60)
    print("This engine implements INGEST → UNDERSTAND → PLAN → RESPOND")
    print("Type your requests to test the conversation system")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60)

    engine = ConversationalIDEEngine()

    while True:
        try:
            user_input = input("\n🧠 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Conversation engine shutting down...")
                break

            if not user_input:
                continue

            # Process input through the pipeline
            print("\n🤖 Processing through pipeline...")
            result = engine.plan_and_respond(user_input)

            if result["success"]:
                print(f"\n💬 Engine: {result['response']}")

                # Show pipeline info
                print(f"\n📊 Pipeline Info:")
                print(f"   • Turn ID: {result['turn_id']}")
                print(f"   • Dialogue Act: {result['understanding']['act']}")
                print(f"   • Intents: {', '.join(result['understanding']['intents'])}")
                print(f"   • Entities: {len(result['understanding']['entities'])} found")
                print(f"   • Confidence: {result['understanding']['confidence']:.2f}")
                print(f"   • Plan Sections: {result['plan']['sections']}")

                if result['plan']['followup']:
                    print(f"   • Followup: {result['plan']['followup']}")
            else:
                print(f"\n❌ Error: {result['response']}")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Conversation engine shutting down...")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
