"""
Request Handler

Manages request processing, prioritization, and lifecycle for the thought pipeline.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue
import threading


class RequestType(Enum):
    """Types of requests that can be processed."""
    QUERY = "query"
    CREATION = "creation"
    ANALYSIS = "analysis"
    MEMORY = "memory"
    SYSTEM = "system"


class RequestStatus(Enum):
    """Request processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Request:
    """Represents a processing request in the system."""
    request_id: str
    request_type: RequestType
    data: Any
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: RequestStatus = RequestStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    
    def __lt__(self, other):
        """Priority comparison for queue ordering."""
        # Higher priority first, then by creation time for same priority
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at
    
    def start_processing(self):
        """Mark request as started."""
        self.status = RequestStatus.PROCESSING
        self.started_at = time.time()
    
    def complete(self, result: Any):
        """Mark request as completed."""
        self.status = RequestStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result
    
    def fail(self, error: str):
        """Mark request as failed."""
        self.status = RequestStatus.FAILED
        self.completed_at = time.time()
        self.error = error
    
    def cancel(self):
        """Cancel the request."""
        self.status = RequestStatus.CANCELLED
        self.completed_at = time.time()
    
    def get_processing_time(self) -> Optional[float]:
        """Get total processing time."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class RequestHandler:
    """
    Handles request processing, prioritization, and lifecycle management.
    
    Implements the request processing concepts from the problem statement:
    - Request queuing with priority
    - Type-based routing
    - Status tracking
    - Function resolution and rejection
    """
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.request_queue = PriorityQueue()
        self.active_requests = {}
        self.completed_requests = {}
        self.request_history = []
        
        # Threading
        self.mtx = threading.Lock()
        self.processing_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'cancelled_requests': 0,
            'average_processing_time': 0.0
        }
        
        # Type-specific handlers
        self.type_handlers = {
            RequestType.QUERY: self._handle_query,
            RequestType.CREATION: self._handle_creation,
            RequestType.ANALYSIS: self._handle_analysis,
            RequestType.MEMORY: self._handle_memory,
            RequestType.SYSTEM: self._handle_system
        }
    
    def generate_id(self) -> str:
        """Generate unique request ID."""
        return str(uuid.uuid4())
    
    async def submit_request(self, request_type: RequestType, data: Any, 
                           priority: int = 1, metadata: Optional[Dict] = None) -> str:
        """
        Submit a new request for processing.
        
        Args:
            request_type: Type of request
            data: Request data
            priority: Processing priority (1-10, higher = more urgent)
            metadata: Additional metadata
            
        Returns:
            Request ID for tracking
        """
        request_id = self.generate_id()
        
        request = Request(
            request_id=request_id,
            request_type=request_type,
            data=data,
            priority=priority,
            metadata=metadata or {}
        )
        
        with self.mtx:
            self.request_queue.put(request)
            self.stats['total_requests'] += 1
        
        return request_id
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request."""
        with self.mtx:
            # Check active requests
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                return self._request_to_dict(request)
            
            # Check completed requests
            if request_id in self.completed_requests:
                request = self.completed_requests[request_id]
                return self._request_to_dict(request)
        
        return None
    
    def _request_to_dict(self, request: Request) -> Dict[str, Any]:
        """Convert request to dictionary representation."""
        return {
            'request_id': request.request_id,
            'type': request.request_type.value,
            'status': request.status.value,
            'priority': request.priority,
            'created_at': request.created_at,
            'started_at': request.started_at,
            'completed_at': request.completed_at,
            'processing_time': request.get_processing_time(),
            'error': request.error,
            'has_result': request.result is not None,
            'metadata': request.metadata
        }
    
    async def process_requests(self):
        """Main request processing loop."""
        while True:
            try:
                # Get next request from queue
                if self.request_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                
                request = self.request_queue.get()
                
                # Check if we can process (concurrency limit)
                async with self.processing_semaphore:
                    await self._process_single_request(request)
                
            except Exception as e:
                print(f"Error in request processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_single_request(self, request: Request):
        """Process a single request."""
        with self.mtx:
            self.active_requests[request.request_id] = request
        
        try:
            request.start_processing()
            
            # Route to appropriate handler
            handler = self.type_handlers.get(request.request_type)
            if handler:
                result = await handler(request)
                request.complete(result)
            else:
                request.fail(f"No handler for request type: {request.request_type}")
            
        except Exception as e:
            request.fail(str(e))
        
        finally:
            # Move to completed requests
            with self.mtx:
                self.active_requests.pop(request.request_id, None)
                self.completed_requests[request.request_id] = request
                
                # Update statistics
                if request.status == RequestStatus.COMPLETED:
                    self.stats['completed_requests'] += 1
                elif request.status == RequestStatus.FAILED:
                    self.stats['failed_requests'] += 1
                elif request.status == RequestStatus.CANCELLED:
                    self.stats['cancelled_requests'] += 1
                
                # Update average processing time
                processing_time = request.get_processing_time()
                if processing_time:
                    total_completed = self.stats['completed_requests']
                    if total_completed > 0:
                        current_avg = self.stats['average_processing_time']
                        self.stats['average_processing_time'] = (
                            (current_avg * (total_completed - 1) + processing_time) / total_completed
                        )
                
                # Add to history
                self.request_history.append({
                    'request_id': request.request_id,
                    'type': request.request_type.value,
                    'status': request.status.value,
                    'processing_time': processing_time,
                    'completed_at': request.completed_at
                })
                
                # Keep history manageable
                if len(self.request_history) > 1000:
                    self.request_history = self.request_history[-500:]
    
    async def _handle_query(self, request: Request) -> Dict[str, Any]:
        """Handle query-type requests."""
        return {
            'type': 'query_response',
            'query': request.data,
            'processed_at': time.time(),
            'resolution': 'query_processed'
        }
    
    async def _handle_creation(self, request: Request) -> Dict[str, Any]:
        """Handle creation-type requests."""
        return {
            'type': 'creation_response',
            'creation_request': request.data,
            'processed_at': time.time(),
            'resolution': 'creation_processed'
        }
    
    async def _handle_analysis(self, request: Request) -> Dict[str, Any]:
        """Handle analysis-type requests."""
        return {
            'type': 'analysis_response',
            'analysis_target': request.data,
            'processed_at': time.time(),
            'resolution': 'analysis_processed'
        }
    
    async def _handle_memory(self, request: Request) -> Dict[str, Any]:
        """Handle memory-type requests."""
        return {
            'type': 'memory_response',
            'memory_operation': request.data,
            'processed_at': time.time(),
            'resolution': 'memory_processed'
        }
    
    async def _handle_system(self, request: Request) -> Dict[str, Any]:
        """Handle system-type requests."""
        return {
            'type': 'system_response',
            'system_command': request.data,
            'processed_at': time.time(),
            'resolution': 'system_processed'
        }
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or active request."""
        with self.mtx:
            # Check if in active requests
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                request.cancel()
                return True
            
            # For queue cancellation, would need to implement queue scanning
            # This is a simplified implementation
            return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self.mtx:
            return {
                'queue_size': self.request_queue.qsize(),
                'active_requests': len(self.active_requests),
                'completed_requests': len(self.completed_requests),
                'max_concurrent': self.max_concurrent,
                'stats': self.stats.copy()
            }
    
    def get_request_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent request history."""
        with self.mtx:
            return self.request_history[-limit:]
    
    async def clear_completed_requests(self, older_than_seconds: int = 3600):
        """Clear completed requests older than specified time."""
        current_time = time.time()
        to_remove = []
        
        with self.mtx:
            for request_id, request in self.completed_requests.items():
                if request.completed_at and (current_time - request.completed_at) > older_than_seconds:
                    to_remove.append(request_id)
            
            for request_id in to_remove:
                del self.completed_requests[request_id]
        
        return len(to_remove)
    
    async def shutdown(self):
        """Shutdown request handler."""
        # Cancel all active requests
        with self.mtx:
            for request in self.active_requests.values():
                request.cancel()
            
            # Clear queues
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except:
                    break