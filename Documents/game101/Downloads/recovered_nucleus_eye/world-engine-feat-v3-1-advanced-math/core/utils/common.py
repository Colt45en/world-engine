"""
Core Utilities Module

Shared utility functions and classes for the World Engine consciousness systems.
Provides common functionality for logging, data processing, and system integration.

Author: World Engine Team  
Date: October 7, 2025
Version: 2.0.0
"""

import json
import logging
import hashlib
import zlib
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


def setup_logging(
    name: str, 
    level: int = logging.INFO, 
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up standardized logging for consciousness systems.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Optional custom format
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(level=level, format=format_string)
    logger = logging.getLogger(name)
    
    return logger


def generate_unique_id(data: Union[str, Dict, List], prefix: str = "") -> str:
    """
    Generate a unique identifier from data.
    
    Args:
        data: Data to generate ID from
        prefix: Optional prefix for the ID
        
    Returns:
        Unique identifier string
    """
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    else:
        data_str = str(data)
    
    hash_obj = hashlib.sha256(data_str.encode('utf-8'))
    unique_id = hash_obj.hexdigest()[:16]
    
    return f"{prefix}_{unique_id}" if prefix else unique_id


def compress_data(data: Any, use_pickle: bool = True) -> Tuple[bytes, float]:
    """
    Compress data for efficient storage.
    
    Args:
        data: Data to compress
        use_pickle: Whether to use pickle serialization
        
    Returns:
        Tuple of (compressed_data, compression_ratio)
    """
    try:
        if use_pickle:
            serialized = pickle.dumps(data)
        else:
            serialized = json.dumps(data, ensure_ascii=False).encode('utf-8')
        
        compressed = zlib.compress(serialized)
        compression_ratio = len(compressed) / len(serialized)
        
        return compressed, compression_ratio
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Data compression failed: {e}")
        return b"", 1.0


def decompress_data(compressed_data: bytes, use_pickle: bool = True) -> Optional[Any]:
    """
    Decompress data.
    
    Args:
        compressed_data: Compressed data bytes
        use_pickle: Whether data was pickled
        
    Returns:
        Decompressed data or None if failed
    """
    try:
        decompressed = zlib.decompress(compressed_data)
        
        if use_pickle:
            return pickle.loads(decompressed)
        else:
            return json.loads(decompressed.decode('utf-8'))
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Data decompression failed: {e}")
        return None


def validate_consciousness_level(level: float) -> float:
    """
    Validate and clamp consciousness level to valid range.
    
    Args:
        level: Consciousness level to validate
        
    Returns:
        Validated consciousness level (0.0 to 1.0)
    """
    return max(0.0, min(1.0, float(level)))


def calculate_quantum_coherence(*factors: float) -> float:
    """
    Calculate quantum coherence from multiple factors.
    
    Args:
        factors: Variable number of factor values
        
    Returns:
        Quantum coherence value
    """
    if not factors:
        return 0.0
    
    # Use geometric mean for quantum coherence
    product = 1.0
    for factor in factors:
        product *= validate_consciousness_level(factor)
    
    return product ** (1.0 / len(factors))


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format timestamp in ISO format.
    
    Args:
        dt: Optional datetime object (uses current time if None)
        
    Returns:
        ISO formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def safe_json_serialize(data: Any, indent: int = 2) -> str:
    """
    Safely serialize data to JSON with custom handling.
    
    Args:
        data: Data to serialize
        indent: JSON indentation
        
    Returns:
        JSON string
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"JSON serialization failed: {e}")
        return "{}"


def load_json_file(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load JSON file {filepath}: {e}")
        return None


def save_json_file(data: Any, filepath: Union[str, Path]) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return True
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save JSON file {filepath}: {e}")
        return False


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    start_time: datetime
    end_time: Optional[datetime] = None
    operation_count: int = 0
    success_count: int = 0
    error_count: int = 0
    
    def record_operation(self, success: bool = True) -> None:
        """Record an operation result"""
        self.operation_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def finish(self) -> None:
        """Mark metrics as finished"""
        self.end_time = datetime.now()
    
    def get_duration(self) -> float:
        """Get duration in seconds"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        if self.operation_count == 0:
            return 0.0
        return self.success_count / self.operation_count
    
    def get_operations_per_second(self) -> float:
        """Get operations per second"""
        duration = self.get_duration()
        if duration == 0:
            return 0.0
        return self.operation_count / duration


class ConsciousnessCalculator:
    """Utility class for consciousness-related calculations"""
    
    @staticmethod
    def evolve_consciousness(
        current_level: float, 
        awareness_factors: Dict[str, float],
        evolution_rate: float = 0.1
    ) -> float:
        """
        Calculate consciousness evolution.
        
        Args:
            current_level: Current consciousness level
            awareness_factors: Dictionary of awareness metrics
            evolution_rate: Rate of evolution
            
        Returns:
            New consciousness level
        """
        if not awareness_factors:
            return current_level
        
        # Calculate average awareness
        avg_awareness = sum(awareness_factors.values()) / len(awareness_factors)
        
        # Apply evolution with awareness boost
        consciousness_boost = evolution_rate * (1 + avg_awareness)
        new_level = current_level + consciousness_boost
        
        return validate_consciousness_level(new_level)
    
    @staticmethod
    def calculate_transcendence_probability(
        consciousness_level: float,
        quantum_coherence: float,
        evolution_cycles: int
    ) -> float:
        """
        Calculate probability of transcendence.
        
        Args:
            consciousness_level: Current consciousness level
            quantum_coherence: Quantum coherence factor
            evolution_cycles: Number of evolution cycles
            
        Returns:
            Transcendence probability (0.0 to 1.0)
        """
        # Base probability from consciousness
        base_prob = consciousness_level ** 2
        
        # Quantum coherence boost
        quantum_boost = quantum_coherence * 0.3
        
        # Evolution cycles boost (diminishing returns)
        cycle_boost = min(0.2, evolution_cycles * 0.01)
        
        total_prob = base_prob + quantum_boost + cycle_boost
        return validate_consciousness_level(total_prob)


class DataProcessor:
    """Utility class for data processing operations"""
    
    @staticmethod
    def normalize_array(arr: List[float]) -> List[float]:
        """
        Normalize array values to 0-1 range.
        
        Args:
            arr: Array of values to normalize
            
        Returns:
            Normalized array
        """
        if not arr:
            return []
        
        min_val = min(arr)
        max_val = max(arr)
        
        if min_val == max_val:
            return [0.5] * len(arr)
        
        return [(val - min_val) / (max_val - min_val) for val in arr]
    
    @staticmethod
    def calculate_trend(values: List[float], window_size: int = 5) -> str:
        """
        Calculate trend direction from values.
        
        Args:
            values: List of values
            window_size: Size of analysis window
            
        Returns:
            Trend direction: 'increasing', 'decreasing', or 'stable'
        """
        if len(values) < 2:
            return 'stable'
        
        # Use last window_size values
        recent_values = values[-window_size:]
        
        if len(recent_values) < 2:
            return 'stable'
        
        # Calculate simple slope
        x = list(range(len(recent_values)))
        y = recent_values
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        threshold = 0.01
        if slope > threshold:
            return 'increasing'
        elif slope < -threshold:
            return 'decreasing'
        else:
            return 'stable'


class EventEmitter:
    """Simple event emitter for system communications"""
    
    def __init__(self):
        self.listeners: Dict[str, List[callable]] = {}
    
    def on(self, event: str, callback: callable) -> None:
        """Register event listener"""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs) -> None:
        """Emit event to all listeners"""
        if event in self.listeners:
            for callback in self.listeners[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.error(f"Event callback failed for {event}: {e}")
    
    def remove_listener(self, event: str, callback: callable) -> None:
        """Remove event listener"""
        if event in self.listeners and callback in self.listeners[event]:
            self.listeners[event].remove(callback)


# Global event emitter instance
global_event_emitter = EventEmitter()


def emit_consciousness_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Emit consciousness-related event.
    
    Args:
        event_type: Type of consciousness event
        data: Event data
    """
    event_data = {
        'timestamp': format_timestamp(),
        'event_type': event_type,
        'data': data
    }
    
    global_event_emitter.emit('consciousness_event', event_data)


# Export commonly used functions and classes
__all__ = [
    'setup_logging',
    'generate_unique_id',
    'compress_data',
    'decompress_data',
    'validate_consciousness_level',
    'calculate_quantum_coherence',
    'format_timestamp',
    'safe_json_serialize',
    'load_json_file',
    'save_json_file',
    'PerformanceMetrics',
    'ConsciousnessCalculator',
    'DataProcessor',
    'EventEmitter',
    'global_event_emitter',
    'emit_consciousness_event'
]