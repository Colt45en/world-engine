"""
Core Thought Pipeline Implementation

Orchestrates the 5-stage thought process for AI bots.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .stages import *
from .quantum_thought import QuantumThoughtProcessor
from .asset_manager import AssetManager
from .request_handler import RequestHandler


class ThoughtStage(Enum):
    """Enumeration of thought process stages."""
    PERCEPTION = "perception"
    PROCESSING = "processing"
    CONTEXTUALIZATION = "contextualization"
    DECISION = "decision"
    MEMORY = "memory"


@dataclass
class ThoughtContext:
    """Context object that flows through the thought pipeline."""
    request_id: str
    input_data: Any
    stage_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 1
    
    def add_stage_result(self, stage: str, result: Any):
        """Add result from a stage."""
        self.stage_results[stage] = result
        
    def get_stage_result(self, stage: str) -> Optional[Any]:
        """Get result from a specific stage."""
        return self.stage_results.get(stage)


class ThoughtPipeline:
    """
    Main thought pipeline orchestrating the 5-stage AI thought process.
    
    Stage 1: Perception & Ingestion
    Stage 2: Core Processing & Meaning-Making
    Stage 3: Connection & Contextualization
    Stage 4: Decision & Action
    Stage 5: Memory & Learning (The Feedback Loop)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the thought pipeline."""
        self.config = config or {}
        
        # Initialize components
        self.asset_manager = AssetManager()
        self.request_handler = RequestHandler()
        self.quantum_processor = QuantumThoughtProcessor()
        
        # Initialize stages
        self.stages = {
            ThoughtStage.PERCEPTION: PerceptionStage(self.asset_manager),
            ThoughtStage.PROCESSING: ProcessingStage(self.quantum_processor),
            ThoughtStage.CONTEXTUALIZATION: ContextualizationStage(),
            ThoughtStage.DECISION: DecisionStage(),
            ThoughtStage.MEMORY: MemoryStage(self.asset_manager)
        }
        
        # Pipeline state
        self.is_running = False
        self.active_contexts = {}
        
    async def process(self, input_data: Any, priority: int = 1, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process input through the complete thought pipeline.
        
        Args:
            input_data: The input to process
            priority: Processing priority (1-10, higher = more urgent)
            metadata: Additional metadata
            
        Returns:
            Complete thought process results
        """
        # Create thought context
        context = ThoughtContext(
            request_id=self.request_handler.generate_id(),
            input_data=input_data,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Track active context
        self.active_contexts[context.request_id] = context
        
        try:
            # Execute stages in sequence
            await self._execute_stage(ThoughtStage.PERCEPTION, context)
            await self._execute_stage(ThoughtStage.PROCESSING, context) 
            await self._execute_stage(ThoughtStage.CONTEXTUALIZATION, context)
            await self._execute_stage(ThoughtStage.DECISION, context)
            await self._execute_stage(ThoughtStage.MEMORY, context)
            
            # Compile final results
            return {
                'request_id': context.request_id,
                'input': input_data,
                'results': context.stage_results,
                'metadata': context.metadata,
                'processing_time': time.time() - context.timestamp,
                'status': 'completed'
            }
            
        except Exception as e:
            # Handle errors and still run memory stage for learning
            await self._execute_stage(ThoughtStage.MEMORY, context, error=str(e))
            
            return {
                'request_id': context.request_id,
                'input': input_data,
                'results': context.stage_results,
                'error': str(e),
                'status': 'error'
            }
            
        finally:
            # Clean up
            self.active_contexts.pop(context.request_id, None)
    
    async def _execute_stage(self, stage: ThoughtStage, context: ThoughtContext, error: Optional[str] = None):
        """Execute a specific stage of the thought process."""
        stage_processor = self.stages[stage]
        
        try:
            if error and stage != ThoughtStage.MEMORY:
                # Skip non-memory stages if there's an error
                return
                
            result = await stage_processor.process(context, error=error)
            context.add_stage_result(stage.value, result)
            
        except Exception as e:
            context.add_stage_result(stage.value, {'error': str(e)})
            if stage != ThoughtStage.MEMORY:
                raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'is_running': self.is_running,
            'active_contexts': len(self.active_contexts),
            'stages_configured': list(self.stages.keys()),
            'asset_manager_status': self.asset_manager.get_status(),
            'quantum_processor_status': self.quantum_processor.get_status()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline."""
        self.is_running = False
        
        # Wait for active contexts to complete
        while self.active_contexts:
            await asyncio.sleep(0.1)
            
        # Shutdown components
        await self.asset_manager.shutdown()
        await self.quantum_processor.shutdown()