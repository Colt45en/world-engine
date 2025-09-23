"""
Asset Manager

Handles memory management, asset loading, and resource allocation
for the thought pipeline system.
"""

import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
from queue import Queue, PriorityQueue
from collections import defaultdict


@dataclass
class Asset:
    """Represents a managed asset in the system."""
    asset_id: str
    asset_type: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    size: int = 0
    
    def access(self):
        """Record asset access."""
        self.last_access = time.time()
        self.access_count += 1


@dataclass 
class LoadRequest:
    """Represents an asset loading request."""
    request_id: str
    asset_type: str
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """Priority comparison for queue ordering."""
        return self.priority > other.priority  # Higher priority first


class AssetManager:
    """
    Manages assets, memory, and resource allocation for the thought pipeline.
    
    Implements the asset management concepts from the problem statement:
    - Asset loading and preloading
    - Memory limits and management
    - Request queuing and prioritization
    - Process queue management
    """
    
    def __init__(self, memory_limit: int = 1024 * 1024 * 100):  # 100MB default
        self.memory_limit = memory_limit
        self.total_memory = 0
        self.assets = {}
        self.loading_queue = PriorityQueue()
        self.loading_in_progress = set()
        self.mtx = threading.Lock()
        
        # Memory tracking
        self.memory_by_type = defaultdict(int)
        self.access_history = []
        
        # Asset metadata
        self.asset_meta = {}
        
        # Loading statistics
        self.load_stats = {
            'total_loads': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'cache_hits': 0
        }
        
    async def load_asset(self, asset_id: str, asset_type: str, data: Any, 
                        priority: int = 1, metadata: Optional[Dict] = None) -> bool:
        """
        Load an asset into memory.
        
        Args:
            asset_id: Unique identifier for the asset
            asset_type: Type of asset (memory, processing, etc.)
            data: The actual asset data
            priority: Loading priority (1-10)
            metadata: Additional metadata
            
        Returns:
            True if loaded successfully
        """
        with self.mtx:
            # Check if already loaded
            if asset_id in self.assets:
                self.assets[asset_id].access()
                self.load_stats['cache_hits'] += 1
                return True
            
            # Estimate size
            size = self._estimate_size(data)
            
            # Check memory limits
            if not await self._ensure_memory_available(size):
                return False
            
            # Create asset
            asset = Asset(
                asset_id=asset_id,
                asset_type=asset_type,
                data=data,
                metadata=metadata or {},
                size=size
            )
            
            # Store asset
            self.assets[asset_id] = asset
            self.total_memory += size
            self.memory_by_type[asset_type] += size
            
            # Update metadata
            self.asset_meta[asset_id] = {
                'type': asset_type,
                'size': size,
                'loaded_at': time.time(),
                'metadata': metadata or {}
            }
            
            self.load_stats['total_loads'] += 1
            self.load_stats['successful_loads'] += 1
            
            return True
    
    async def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Get an asset by ID."""
        with self.mtx:
            asset = self.assets.get(asset_id)
            if asset:
                asset.access()
                self.access_history.append({
                    'asset_id': asset_id,
                    'timestamp': time.time(),
                    'type': 'access'
                })
            return asset
    
    async def preload_assets(self, asset_specs: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Preload multiple assets based on specifications.
        
        Args:
            asset_specs: List of asset specifications with id, type, data, priority
            
        Returns:
            Dictionary mapping asset_id to load success status
        """
        results = {}
        
        # Sort by priority
        sorted_specs = sorted(asset_specs, key=lambda x: x.get('priority', 1), reverse=True)
        
        for spec in sorted_specs:
            asset_id = spec['id']
            asset_type = spec['type']
            data = spec['data']
            priority = spec.get('priority', 1)
            metadata = spec.get('metadata', {})
            
            success = await self.load_asset(asset_id, asset_type, data, priority, metadata)
            results[asset_id] = success
        
        return results
    
    async def store_memory(self, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """Store memory data as an asset."""
        return await self.load_asset(
            asset_id=f"memory_{memory_id}",
            asset_type="memory",
            data=memory_data,
            priority=3,  # Medium priority for memories
            metadata={'category': 'episodic_memory'}
        )
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory data."""
        asset = await self.get_asset(f"memory_{memory_id}")
        return asset.data if asset else None
    
    async def query_memories(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query memories based on parameters."""
        memories = []
        
        with self.mtx:
            for asset_id, asset in self.assets.items():
                if asset_id.startswith('memory_') and asset.asset_type == 'memory':
                    # Simple query matching - can be enhanced
                    if self._matches_query(asset.data, query_params):
                        memories.append({
                            'memory_id': asset_id,
                            'data': asset.data,
                            'metadata': asset.metadata,
                            'last_access': asset.last_access
                        })
        
        # Sort by relevance/recency
        memories.sort(key=lambda x: x['last_access'], reverse=True)
        return memories
    
    def _matches_query(self, memory_data: Dict[str, Any], query_params: Dict[str, Any]) -> bool:
        """Check if memory data matches query parameters."""
        # Simple matching logic - can be enhanced with more sophisticated filtering
        for key, value in query_params.items():
            if key in memory_data:
                if isinstance(value, str) and isinstance(memory_data[key], str):
                    if value.lower() not in memory_data[key].lower():
                        return False
                elif memory_data[key] != value:
                    return False
        return True
    
    async def _ensure_memory_available(self, required_size: int) -> bool:
        """Ensure sufficient memory is available for new asset."""
        if self.total_memory + required_size <= self.memory_limit:
            return True
        
        # Need to free memory - implement LRU eviction
        return await self._free_memory(required_size)
    
    async def _free_memory(self, required_size: int) -> bool:
        """Free memory using LRU eviction policy."""
        # Sort assets by last access time (LRU first)
        assets_by_access = sorted(
            self.assets.items(),
            key=lambda x: x[1].last_access
        )
        
        freed_memory = 0
        assets_to_remove = []
        
        for asset_id, asset in assets_by_access:
            # Don't evict critical assets
            if asset.metadata.get('critical', False):
                continue
            
            assets_to_remove.append(asset_id)
            freed_memory += asset.size
            
            if freed_memory >= required_size:
                break
        
        # Remove selected assets
        for asset_id in assets_to_remove:
            await self._evict_asset(asset_id)
        
        return freed_memory >= required_size
    
    async def _evict_asset(self, asset_id: str):
        """Evict an asset from memory."""
        with self.mtx:
            if asset_id in self.assets:
                asset = self.assets[asset_id]
                
                # Update memory tracking
                self.total_memory -= asset.size
                self.memory_by_type[asset.asset_type] -= asset.size
                
                # Remove asset
                del self.assets[asset_id]
                
                # Log eviction
                self.access_history.append({
                    'asset_id': asset_id,
                    'timestamp': time.time(),
                    'type': 'eviction'
                })
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        try:
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data, default=str).encode('utf-8'))
            elif hasattr(data, '__sizeof__'):
                return data.__sizeof__()
            else:
                return len(str(data).encode('utf-8'))
        except:
            return 1024  # Default estimate
    
    def get_status(self) -> Dict[str, Any]:
        """Get current asset manager status."""
        with self.mtx:
            return {
                'total_memory': self.total_memory,
                'memory_limit': self.memory_limit,
                'memory_usage_percent': (self.total_memory / self.memory_limit) * 100,
                'asset_count': len(self.assets),
                'assets_by_type': dict(self.memory_by_type),
                'load_stats': self.load_stats.copy(),
                'queue_size': self.loading_queue.qsize(),
                'loading_in_progress': len(self.loading_in_progress)
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        with self.mtx:
            assets_by_type = defaultdict(list)
            for asset_id, asset in self.assets.items():
                assets_by_type[asset.asset_type].append({
                    'id': asset_id,
                    'size': asset.size,
                    'access_count': asset.access_count,
                    'last_access': asset.last_access
                })
            
            return {
                'total_memory': self.total_memory,
                'available_memory': self.memory_limit - self.total_memory,
                'assets_by_type': dict(assets_by_type),
                'recent_access_history': self.access_history[-10:],  # Last 10 accesses
                'memory_pressure': self.total_memory / self.memory_limit
            }
    
    async def cleanup_expired_assets(self, max_age_seconds: int = 3600):
        """Clean up assets that haven't been accessed recently."""
        current_time = time.time()
        expired_assets = []
        
        with self.mtx:
            for asset_id, asset in self.assets.items():
                if current_time - asset.last_access > max_age_seconds:
                    if not asset.metadata.get('persistent', False):
                        expired_assets.append(asset_id)
        
        for asset_id in expired_assets:
            await self._evict_asset(asset_id)
        
        return len(expired_assets)
    
    async def rebuild_index(self):
        """Rebuild asset indices and optimize memory layout."""
        # Implement index rebuilding logic
        with self.mtx:
            # Recalculate memory usage
            total_memory = 0
            memory_by_type = defaultdict(int)
            
            for asset in self.assets.values():
                total_memory += asset.size
                memory_by_type[asset.asset_type] += asset.size
            
            self.total_memory = total_memory
            self.memory_by_type = memory_by_type
    
    async def shutdown(self):
        """Shutdown asset manager and clean up resources."""
        with self.mtx:
            self.assets.clear()
            self.asset_meta.clear()
            self.access_history.clear()
            
            # Clear queues
            while not self.loading_queue.empty():
                try:
                    self.loading_queue.get_nowait()
                except:
                    break
            
            self.loading_in_progress.clear()
            self.total_memory = 0
            self.memory_by_type.clear()