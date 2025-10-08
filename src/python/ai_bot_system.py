"""
AI Bot Knowledge System - Scalable Learning Engine for World Engine

This system provides:
1. Persistent knowledge storage with semantic understanding
2. Self-learning capabilities from interactions
3. Integration with World Engine lexicon processing
4. Vector embeddings for semantic similarity
5. Adaptive response generation based on learned patterns
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib
import pickle
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class KnowledgeEntry:
    """Represents a piece of knowledge in the system."""
    id: str
    content: str
    category: str
    confidence: float
    semantic_score: Optional[float] = None
    embeddings: Optional[List[float]] = None
    created_at: str = None
    updated_at: str = None
    usage_count: int = 0
    success_rate: float = 1.0
    context_tags: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.context_tags is None:
            self.context_tags = []


@dataclass
class LearningEvent:
    """Represents a learning event for self-improvement."""
    interaction_id: str
    input_text: str
    bot_response: str
    user_feedback: Optional[str] = None
    world_engine_analysis: Optional[Dict] = None
    success: bool = True
    timestamp: str = None
    semantic_insights: Dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.semantic_insights is None:
            self.semantic_insights = {}


class AIBotKnowledgeSystem:
    """
    Scalable AI bot with built-in knowledge system and self-learning.
    Integrates with World Engine for enhanced semantic understanding.
    """

    def __init__(self, db_path: str = "ai_bot_knowledge.db", world_engine_api=None):
        self.db_path = Path(db_path)
        self.world_engine = world_engine_api
        self.knowledge_cache = {}
        self.learning_buffer = []
        self.response_patterns = defaultdict(list)
        self.semantic_associations = defaultdict(set)

        self._init_database()
        self._load_knowledge_cache()

    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Knowledge entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT,
                confidence REAL,
                semantic_score REAL,
                embeddings BLOB,
                created_at TEXT,
                updated_at TEXT,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0,
                context_tags TEXT
            )
        """)

        # Learning events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_events (
                interaction_id TEXT PRIMARY KEY,
                input_text TEXT,
                bot_response TEXT,
                user_feedback TEXT,
                world_engine_analysis TEXT,
                success INTEGER,
                timestamp TEXT,
                semantic_insights TEXT
            )
        """)

        # Semantic associations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_associations (
                word1 TEXT,
                word2 TEXT,
                strength REAL,
                context TEXT,
                updated_at TEXT,
                PRIMARY KEY (word1, word2)
            )
        """)

        # Response patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_patterns (
                pattern_id TEXT PRIMARY KEY,
                input_pattern TEXT,
                response_template TEXT,
                success_rate REAL,
                usage_count INTEGER,
                last_used TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_knowledge_cache(self):
        """Load frequently accessed knowledge into memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load high-usage knowledge entries
        cursor.execute("""
            SELECT * FROM knowledge
            WHERE usage_count > 5 OR confidence > 0.8
            ORDER BY usage_count DESC, confidence DESC
            LIMIT 1000
        """)

        for row in cursor.fetchall():
            entry = self._row_to_knowledge_entry(row)
            self.knowledge_cache[entry.id] = entry

        conn.close()

    def _row_to_knowledge_entry(self, row) -> KnowledgeEntry:
        """Convert database row to KnowledgeEntry."""
        embeddings = pickle.loads(row[5]) if row[5] else None
        context_tags = json.loads(row[10]) if row[10] else []

        return KnowledgeEntry(
            id=row[0],
            content=row[1],
            category=row[2],
            confidence=row[3],
            semantic_score=row[4],
            embeddings=embeddings,
            created_at=row[6],
            updated_at=row[7],
            usage_count=row[8],
            success_rate=row[9],
            context_tags=context_tags
        )

    def add_knowledge(self, content: str, category: str, confidence: float = 0.5,
                     context_tags: List[str] = None) -> str:
        """Add new knowledge to the system."""
        knowledge_id = hashlib.md5(content.encode()).hexdigest()

        # Get semantic analysis from World Engine if available
        semantic_score = None
        embeddings = None

        if self.world_engine:
            try:
                analysis = self.world_engine.analyze_text(content)
                semantic_score = analysis.get('overall_score', 0.0)
                # Generate simple embeddings (in real implementation, use proper embeddings)
                embeddings = self._generate_simple_embeddings(content)
            except Exception as e:
                print(f"World Engine analysis failed: {e}")

        entry = KnowledgeEntry(
            id=knowledge_id,
            content=content,
            category=category,
            confidence=confidence,
            semantic_score=semantic_score,
            embeddings=embeddings,
            context_tags=context_tags or []
        )

        # Store in database
        self._save_knowledge_entry(entry)

        # Update cache if high confidence
        if confidence > 0.7:
            self.knowledge_cache[knowledge_id] = entry

        return knowledge_id

    def _save_knowledge_entry(self, entry: KnowledgeEntry):
        """Save knowledge entry to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        embeddings_blob = pickle.dumps(entry.embeddings) if entry.embeddings else None
        context_tags_json = json.dumps(entry.context_tags)

        cursor.execute("""
            INSERT OR REPLACE INTO knowledge
            (id, content, category, confidence, semantic_score, embeddings,
             created_at, updated_at, usage_count, success_rate, context_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id, entry.content, entry.category, entry.confidence,
            entry.semantic_score, embeddings_blob, entry.created_at,
            entry.updated_at, entry.usage_count, entry.success_rate,
            context_tags_json
        ))

        conn.commit()
        conn.close()

    def _generate_simple_embeddings(self, text: str) -> List[float]:
        """Generate simple embeddings (placeholder for proper embedding model)."""
        # This is a placeholder - in production, use proper embeddings like sentence-transformers
        words = text.lower().split()
        embedding = [0.0] * 100  # 100-dimensional embedding

        for i, word in enumerate(words[:50]):  # Limit to first 50 words
            hash_val = hash(word) % 100
            embedding[hash_val] += 1.0 / len(words)

        return embedding

    def find_relevant_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeEntry]:
        """Find relevant knowledge entries for a query."""
        relevant = []

        # Simple keyword matching (can be enhanced with semantic search)
        query_words = set(query.lower().split())

        # Check cache first
        for entry in self.knowledge_cache.values():
            content_words = set(entry.content.lower().split())
            overlap = len(query_words.intersection(content_words))

            if overlap > 0:
                # Calculate relevance score
                relevance = (overlap / len(query_words)) * entry.confidence
                relevant.append((relevance, entry))

        # Sort by relevance and return top results
        relevant.sort(reverse=True)
        return [entry for _, entry in relevant[:limit]]

    def generate_response(self, user_input: str, context: Dict = None) -> str:
        """Generate AI bot response using knowledge and learning."""
        context = context or {}

        # Find relevant knowledge
        relevant_knowledge = self.find_relevant_knowledge(user_input)

        # Get World Engine analysis
        world_engine_insight = None
        if self.world_engine:
            try:
                analysis = self.world_engine.analyze_text(user_input)
                world_engine_insight = analysis
            except Exception as e:
                print(f"World Engine analysis failed: {e}")

        # Generate response based on knowledge and patterns
        response = self._synthesize_response(user_input, relevant_knowledge, world_engine_insight)

        # Record learning event
        learning_event = LearningEvent(
            interaction_id=hashlib.md5((user_input + str(datetime.now())).encode()).hexdigest(),
            input_text=user_input,
            bot_response=response,
            world_engine_analysis=world_engine_insight
        )

        self.learning_buffer.append(learning_event)

        # Process learning if buffer is full
        if len(self.learning_buffer) >= 10:
            self._process_learning_batch()

        return response

    def _synthesize_response(self, input_text: str, knowledge: List[KnowledgeEntry],
                           world_engine_analysis: Dict = None) -> str:
        """Synthesize response from available knowledge and analysis."""
        if not knowledge:
            return "I'm still learning about that topic. Can you tell me more?"

        # Use highest confidence knowledge as base
        primary_knowledge = knowledge[0]

        # Enhance with World Engine semantic insights
        sentiment_modifier = ""
        if world_engine_analysis:
            overall_score = world_engine_analysis.get('overall_score', 0.0)
            if overall_score > 0.3:
                sentiment_modifier = " This seems positive based on my analysis."
            elif overall_score < -0.3:
                sentiment_modifier = " This seems to have negative connotations."

        # Generate contextual response
        base_response = f"Based on what I know: {primary_knowledge.content}"

        if len(knowledge) > 1:
            base_response += f" I also found that {knowledge[1].content}"

        return base_response + sentiment_modifier

    def _process_learning_batch(self):
        """Process accumulated learning events to improve the system."""
        if not self.learning_buffer:
            return

        # Save learning events to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for event in self.learning_buffer:
            cursor.execute("""
                INSERT OR REPLACE INTO learning_events
                (interaction_id, input_text, bot_response, user_feedback,
                 world_engine_analysis, success, timestamp, semantic_insights)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.interaction_id,
                event.input_text,
                event.bot_response,
                event.user_feedback,
                json.dumps(event.world_engine_analysis) if event.world_engine_analysis else None,
                1 if event.success else 0,
                event.timestamp,
                json.dumps(event.semantic_insights)
            ))

        conn.commit()
        conn.close()

        # Clear buffer
        self.learning_buffer = []

        # Trigger knowledge refinement
        self._refine_knowledge()

    def _refine_knowledge(self):
        """Refine knowledge based on learning events."""
        # Analyze recent learning events to identify patterns
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find frequently asked questions
        cursor.execute("""
            SELECT input_text, COUNT(*) as frequency
            FROM learning_events
            WHERE timestamp > date('now', '-7 days')
            GROUP BY input_text
            HAVING frequency > 2
        """)

        frequent_queries = cursor.fetchall()

        for query, frequency in frequent_queries:
            # Check if we have good knowledge for this query
            relevant = self.find_relevant_knowledge(query)

            if not relevant or relevant[0].confidence < 0.7:
                # Create new knowledge entry for frequently asked questions
                self.add_knowledge(
                    content=f"Frequently asked: {query}",
                    category="faq",
                    confidence=min(0.9, 0.5 + (frequency * 0.1)),
                    context_tags=["frequent", "user_generated"]
                )

        conn.close()

    def learn_from_feedback(self, interaction_id: str, feedback: str, success: bool):
        """Learn from user feedback on responses."""
        # Update learning event
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE learning_events
            SET user_feedback = ?, success = ?
            WHERE interaction_id = ?
        """, (feedback, 1 if success else 0, interaction_id))

        conn.commit()
        conn.close()

        # If negative feedback, reduce confidence in related knowledge
        if not success:
            # Find related knowledge and reduce confidence
            cursor.execute("""
                SELECT input_text, bot_response FROM learning_events
                WHERE interaction_id = ?
            """, (interaction_id,))

            result = cursor.fetchone()
            if result:
                input_text, bot_response = result
                relevant_knowledge = self.find_relevant_knowledge(input_text)

                for knowledge in relevant_knowledge:
                    knowledge.confidence = max(0.1, knowledge.confidence - 0.1)
                    knowledge.success_rate *= 0.9
                    self._save_knowledge_entry(knowledge)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the AI bot's knowledge and learning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM knowledge")
        knowledge_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM learning_events")
        interactions_count = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence) FROM knowledge")
        avg_confidence = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT AVG(success) FROM learning_events")
        success_rate = cursor.fetchone()[0] or 1.0

        conn.close()

        return {
            "knowledge_entries": knowledge_count,
            "total_interactions": interactions_count,
            "average_confidence": round(avg_confidence, 3),
            "success_rate": round(success_rate, 3),
            "cache_size": len(self.knowledge_cache),
            "world_engine_connected": self.world_engine is not None
        }


# Example usage and integration
if __name__ == "__main__":
    # Initialize AI bot system
    ai_bot = AIBotKnowledgeSystem()

    # Add some initial knowledge
    ai_bot.add_knowledge("The World Engine processes lexical semantics", "technical", 0.9)
    ai_bot.add_knowledge("Semantic scaling uses hand-labeled seeds", "methodology", 0.85)

    # Generate response
    response = ai_bot.generate_response("How does the World Engine work?")
    print(f"Bot Response: {response}")

    # Show system stats
    stats = ai_bot.get_system_stats()
    print(f"System Stats: {stats}")
