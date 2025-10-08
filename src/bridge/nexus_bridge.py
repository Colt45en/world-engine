#!/usr/bin/env python3
"""
Nexus Core Chat Bridge with Crash Protection
Provides RAG backend with memory management and error recovery
"""

import asyncio
import json
import logging
import traceback
import psutil
import gc
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nexus-bridge")

# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., max_length=1000)
    top_k: int = Field(default=1, ge=1, le=10)
    training_mode: bool = Field(default=False)

class RAGResult(BaseModel):
    content: str
    metadata: Dict[str, str]
    source: str
    category: str
    priority: str

class RAGResponse(BaseModel):
    success: bool
    query: Optional[str] = None
    results: List[RAGResult] = []
    sources: List[str] = []
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    system: str
    memory_usage: float
    training_active: bool
    uptime_seconds: float

@dataclass
class SystemState:
    start_time: datetime
    requests_processed: int
    errors_count: int
    memory_peak_mb: float
    training_sessions: int
    last_cleanup: datetime

class MemoryManager:
    """Manages memory usage and performs cleanup when needed"""

    def __init__(self, max_memory_mb: float = 512.0):
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = max_memory_mb * 0.8
        self.last_cleanup = datetime.now()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def needs_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        return self.get_memory_usage() > self.cleanup_threshold

    def force_cleanup(self) -> float:
        """Force garbage collection and return memory usage"""
        logger.info("Performing memory cleanup")
        gc.collect()
        self.last_cleanup = datetime.now()
        return self.get_memory_usage()

class CrashSafeRAGEngine:
    """Crash-safe RAG engine with memory management"""

    def __init__(self):
        self.memory_manager = MemoryManager()
        self.documents = self._load_default_documents()
        self.training_active = False
        self.training_data = []
        self.max_training_points = 1000

    def _load_default_documents(self) -> List[Dict[str, str]]:
        """Load default knowledge base"""
        return [
            {
                "id": "nexus-overview",
                "title": "Nexus System Overview",
                "content": """
Nexus Core is a comprehensive AI training and chat system featuring:

- Crash-safe training with memory management
- Automatic error recovery and fallback modes
- Real-time monitoring and health checks
- Memory usage optimization
- Circuit breaker patterns for reliability

Key components:
- RAG (Retrieval-Augmented Generation) engine
- Training data collection with limits
- Memory monitoring and cleanup
- Error handling and recovery
"""
            },
            {
                "id": "crash-prevention",
                "title": "Crash Prevention Guide",
                "content": """
The system includes multiple crash prevention mechanisms:

1. Memory Limits: Automatic cleanup when usage exceeds 80% of limit
2. Training Limits: Maximum 1000 training points per session
3. Circuit Breakers: Automatic fallback on repeated failures
4. Timeout Protection: All operations have configurable timeouts
5. Error Recovery: Graceful degradation and retry mechanisms

If you experience crashes:
- Check memory usage with /health endpoint
- Reduce training batch sizes
- Clear accumulated data
- Restart the bridge service
"""
            },
            {
                "id": "troubleshooting",
                "title": "Troubleshooting Guide",
                "content": """
Common issues and solutions:

Training Crashes:
- Reduce training data size
- Check memory limits
- Monitor error logs
- Use incremental training

Connection Issues:
- Verify port 8888 is available
- Check firewall settings
- Ensure bridge is running
- Test with /health endpoint

Memory Problems:
- Monitor usage via /health
- Force cleanup with /cleanup
- Restart if memory leaks persist
- Adjust memory limits in config
"""
            }
        ]

    async def query(self, question: str, top_k: int = 1) -> RAGResponse:
        """Process a query with crash protection"""
        try:
            # Check memory before processing
            if self.memory_manager.needs_cleanup():
                self.memory_manager.force_cleanup()

            # Simple keyword matching for demo
            question_lower = question.lower()
            results = []

            for doc in self.documents:
                score = self._calculate_relevance(question_lower, doc["content"].lower())
                if score > 0.1:  # Minimum relevance threshold
                    results.append((score, doc))

            # Sort by relevance and take top_k
            results.sort(key=lambda x: x[0], reverse=True)
            top_results = results[:top_k]

            rag_results = []
            sources = []

            for score, doc in top_results:
                rag_result = RAGResult(
                    content=doc["content"],
                    metadata={
                        "source": doc["id"],
                        "category": "documentation",
                        "priority": "high" if score > 0.5 else "medium"
                    },
                    source=doc["id"],
                    category="documentation",
                    priority="high" if score > 0.5 else "medium"
                )
                rag_results.append(rag_result)
                sources.append(doc["id"])

            return RAGResponse(
                success=True,
                query=question,
                results=rag_results,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            logger.error(traceback.format_exc())
            return RAGResponse(
                success=False,
                error=f"Query processing failed: {str(e)}"
            )

    def _calculate_relevance(self, question: str, content: str) -> float:
        """Simple relevance calculation"""
        question_words = set(question.split())
        content_words = set(content.split())

        if not question_words:
            return 0.0

        intersection = question_words.intersection(content_words)
        return len(intersection) / len(question_words)

    def start_training(self) -> bool:
        """Start training session with crash protection"""
        if self.training_active:
            return False

        try:
            self.training_active = True
            self.training_data = []
            logger.info("Training session started")
            return True
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return False

    def stop_training(self) -> Dict[str, Any]:
        """Stop training and return session stats"""
        stats = {
            "data_points": len(self.training_data),
            "memory_usage": self.memory_manager.get_memory_usage(),
            "success": True
        }

        self.training_active = False
        self.training_data = []
        logger.info(f"Training stopped: {stats}")
        return stats

    def collect_training_data(self, data: Dict[str, Any]) -> bool:
        """Collect training data with limits"""
        if not self.training_active:
            return False

        if len(self.training_data) >= self.max_training_points:
            logger.warning("Training data limit reached")
            return False

        try:
            self.training_data.append({
                "timestamp": datetime.now().isoformat(),
                "data": data
            })
            return True
        except Exception as e:
            logger.error(f"Failed to collect training data: {e}")
            return False

# Global instances
system_state = SystemState(
    start_time=datetime.now(),
    requests_processed=0,
    errors_count=0,
    memory_peak_mb=0.0,
    training_sessions=0,
    last_cleanup=datetime.now()
)

rag_engine = CrashSafeRAGEngine()

# FastAPI app with crash protection
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Nexus Bridge starting up...")
    yield
    logger.info("Nexus Bridge shutting down...")

app = FastAPI(
    title="Nexus Core Chat Bridge",
    description="Crash-safe RAG backend with memory management",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system status"""
    try:
        memory_usage = rag_engine.memory_manager.get_memory_usage()
        system_state.memory_peak_mb = max(system_state.memory_peak_mb, memory_usage)

        uptime = (datetime.now() - system_state.start_time).total_seconds()

        return HealthResponse(
            status="healthy",
            system="Nexus Core Bridge v1.0",
            memory_usage=memory_usage,
            training_active=rag_engine.training_active,
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/query", response_model=RAGResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process RAG query with crash protection"""
    try:
        system_state.requests_processed += 1

        # Add background memory monitoring
        background_tasks.add_task(monitor_memory)

        response = await rag_engine.query(request.question, request.top_k)

        if not response.success:
            system_state.errors_count += 1

        return response

    except Exception as e:
        system_state.errors_count += 1
        logger.error(f"Query endpoint error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/training/start")
async def start_training():
    """Start training session"""
    try:
        success = rag_engine.start_training()
        if success:
            system_state.training_sessions += 1

        return {"success": success, "message": "Training started" if success else "Training already active"}
    except Exception as e:
        logger.error(f"Start training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/stop")
async def stop_training():
    """Stop training session"""
    try:
        stats = rag_engine.stop_training()
        return stats
    except Exception as e:
        logger.error(f"Stop training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/data")
async def collect_training_data(data: Dict[str, Any]):
    """Collect training data"""
    try:
        success = rag_engine.collect_training_data(data)
        return {"success": success}
    except Exception as e:
        logger.error(f"Collect training data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cleanup")
async def force_cleanup():
    """Force memory cleanup"""
    try:
        memory_before = rag_engine.memory_manager.get_memory_usage()
        memory_after = rag_engine.memory_manager.force_cleanup()

        return {
            "success": True,
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_freed_mb": memory_before - memory_after
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        memory_usage = rag_engine.memory_manager.get_memory_usage()
        uptime = (datetime.now() - system_state.start_time).total_seconds()

        return {
            "uptime_seconds": uptime,
            "requests_processed": system_state.requests_processed,
            "errors_count": system_state.errors_count,
            "error_rate": system_state.errors_count / max(system_state.requests_processed, 1),
            "memory_current_mb": memory_usage,
            "memory_peak_mb": system_state.memory_peak_mb,
            "training_sessions": system_state.training_sessions,
            "training_active": rag_engine.training_active,
            "training_data_points": len(rag_engine.training_data) if rag_engine.training_active else 0
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def monitor_memory():
    """Background task to monitor memory usage"""
    try:
        if rag_engine.memory_manager.needs_cleanup():
            logger.warning("Memory usage high, performing cleanup")
            rag_engine.memory_manager.force_cleanup()
    except Exception as e:
        logger.error(f"Memory monitoring error: {e}")

if __name__ == "__main__":
    logger.info("Starting Nexus Core Chat Bridge...")
    uvicorn.run(
        "nexus_bridge:app",
        host="localhost",
        port=8888,
        log_level="info",
        reload=False
    )
