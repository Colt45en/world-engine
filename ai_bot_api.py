"""
AI Bot API Service - REST endpoints for the AI Bot Knowledge System

Extends the World Engine API with AI bot capabilities:
- Chat with AI bot
- Knowledge management
- Learning feedback
- System statistics
"""

from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from ai_bot_system import AIBotKnowledgeSystem, KnowledgeEntry, LearningEvent


class ChatRequest(BaseModel):
    """Request model for chat with AI bot."""
    message: str
    context: Optional[Dict[str, Any]] = {}
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for AI bot chat."""
    response: str
    interaction_id: str
    confidence: float
    knowledge_sources: List[str]
    world_engine_analysis: Optional[Dict] = None


class KnowledgeRequest(BaseModel):
    """Request model for adding knowledge."""
    content: str
    category: str
    confidence: float = 0.5
    context_tags: Optional[List[str]] = []


class FeedbackRequest(BaseModel):
    """Request model for learning feedback."""
    interaction_id: str
    feedback: str
    success: bool


class AIBotAPIExtension:
    """Extension to World Engine API for AI Bot functionality."""

    def __init__(self, world_engine_api=None):
        self.ai_bot = AIBotKnowledgeSystem(world_engine_api=world_engine_api)
        self._init_default_knowledge()

    def _init_default_knowledge(self):
        """Initialize with default World Engine knowledge."""
        default_knowledge = [
            {
                "content": "World Engine is a lexicon processing and semantic analysis system with integrated recording, chat interface, and real-time analysis capabilities.",
                "category": "system_overview",
                "confidence": 0.95,
                "tags": ["worldengine", "overview", "system"]
            },
            {
                "content": "The system uses hand-labeled seeds with semantic values from -1.0 to 1.0 for sentiment analysis.",
                "category": "semantic_scaling",
                "confidence": 0.9,
                "tags": ["semantics", "seeds", "scaling"]
            },
            {
                "content": "The studio interface includes Chat Controller, Engine Controller, and Recorder Controller communicating via StudioBridge.",
                "category": "architecture",
                "confidence": 0.85,
                "tags": ["architecture", "controllers", "bridge"]
            },
            {
                "content": "Use /run <text> to analyze content, /rec start to begin recording, and /help for full command list.",
                "category": "commands",
                "confidence": 0.9,
                "tags": ["commands", "usage", "help"]
            }
        ]

        for knowledge in default_knowledge:
            self.ai_bot.add_knowledge(
                content=knowledge["content"],
                category=knowledge["category"],
                confidence=knowledge["confidence"],
                context_tags=knowledge["tags"]
            )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle chat request with AI bot."""
        try:
            # Generate response using AI bot
            response = self.ai_bot.generate_response(request.message, request.context)

            # Get the last interaction details
            if self.ai_bot.learning_buffer:
                last_event = self.ai_bot.learning_buffer[-1]
                interaction_id = last_event.interaction_id
                world_engine_analysis = last_event.world_engine_analysis
            else:
                interaction_id = "unknown"
                world_engine_analysis = None

            # Find knowledge sources used
            relevant_knowledge = self.ai_bot.find_relevant_knowledge(request.message, limit=3)
            knowledge_sources = [k.category for k in relevant_knowledge]

            # Calculate confidence based on knowledge quality
            confidence = sum(k.confidence for k in relevant_knowledge) / len(relevant_knowledge) if relevant_knowledge else 0.5

            return ChatResponse(
                response=response,
                interaction_id=interaction_id,
                confidence=round(confidence, 3),
                knowledge_sources=knowledge_sources,
                world_engine_analysis=world_engine_analysis
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI bot error: {str(e)}")

    def add_knowledge(self, request: KnowledgeRequest) -> Dict[str, str]:
        """Add new knowledge to the AI bot."""
        try:
            knowledge_id = self.ai_bot.add_knowledge(
                content=request.content,
                category=request.category,
                confidence=request.confidence,
                context_tags=request.context_tags
            )

            return {
                "status": "success",
                "knowledge_id": knowledge_id,
                "message": f"Added knowledge to category '{request.category}'"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to add knowledge: {str(e)}")

    def provide_feedback(self, request: FeedbackRequest) -> Dict[str, str]:
        """Provide feedback on AI bot response for learning."""
        try:
            self.ai_bot.learn_from_feedback(
                interaction_id=request.interaction_id,
                feedback=request.feedback,
                success=request.success
            )

            return {
                "status": "success",
                "message": "Feedback recorded for learning"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

    def get_knowledge(self, query: str = None, category: str = None, limit: int = 10) -> List[Dict]:
        """Get knowledge entries, optionally filtered."""
        try:
            if query:
                knowledge_entries = self.ai_bot.find_relevant_knowledge(query, limit)
            else:
                # Get from cache or database
                knowledge_entries = list(self.ai_bot.knowledge_cache.values())[:limit]

            # Filter by category if specified
            if category:
                knowledge_entries = [k for k in knowledge_entries if k.category == category]

            # Convert to dict format
            return [
                {
                    "id": k.id,
                    "content": k.content,
                    "category": k.category,
                    "confidence": k.confidence,
                    "usage_count": k.usage_count,
                    "success_rate": k.success_rate,
                    "context_tags": k.context_tags,
                    "created_at": k.created_at
                }
                for k in knowledge_entries
            ]

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get knowledge: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get AI bot system statistics."""
        try:
            stats = self.ai_bot.get_system_stats()

            # Add timestamp
            stats["last_updated"] = datetime.now().isoformat()

            return stats

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

    def get_learning_history(self, limit: int = 50) -> List[Dict]:
        """Get recent learning events."""
        try:
            import sqlite3

            conn = sqlite3.connect(self.ai_bot.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT interaction_id, input_text, bot_response, user_feedback, success, timestamp
                FROM learning_events
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            events = []
            for row in cursor.fetchall():
                events.append({
                    "interaction_id": row[0],
                    "input_text": row[1],
                    "bot_response": row[2],
                    "user_feedback": row[3],
                    "success": bool(row[4]),
                    "timestamp": row[5]
                })

            conn.close()
            return events

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get learning history: {str(e)}")


# Integration with existing World Engine API
def extend_world_engine_api(app, world_engine_api=None):
    """Add AI bot endpoints to existing FastAPI app."""

    ai_bot_api = AIBotAPIExtension(world_engine_api)

    @app.post("/api/ai-bot/chat", response_model=ChatResponse)
    async def chat_with_ai_bot(request: ChatRequest):
        """Chat with the AI bot."""
        return ai_bot_api.chat(request)

    @app.post("/api/ai-bot/knowledge")
    async def add_knowledge(request: KnowledgeRequest):
        """Add knowledge to the AI bot."""
        return ai_bot_api.add_knowledge(request)

    @app.post("/api/ai-bot/feedback")
    async def provide_feedback(request: FeedbackRequest):
        """Provide feedback for AI bot learning."""
        return ai_bot_api.provide_feedback(request)

    @app.get("/api/ai-bot/knowledge")
    async def get_knowledge(query: str = None, category: str = None, limit: int = 10):
        """Get knowledge entries."""
        return ai_bot_api.get_knowledge(query, category, limit)

    @app.get("/api/ai-bot/stats")
    async def get_ai_bot_stats():
        """Get AI bot statistics."""
        return ai_bot_api.get_stats()

    @app.get("/api/ai-bot/learning-history")
    async def get_learning_history(limit: int = 50):
        """Get AI bot learning history."""
        return ai_bot_api.get_learning_history(limit)

    return ai_bot_api


# Standalone testing
if __name__ == "__main__":
    # Test the AI bot API
    api = AIBotAPIExtension()

    # Test chat
    chat_req = ChatRequest(message="How do I use the World Engine?")
    response = api.chat(chat_req)
    print(f"Chat Response: {response.response}")

    # Test stats
    stats = api.get_stats()
    print(f"AI Bot Stats: {stats}")
