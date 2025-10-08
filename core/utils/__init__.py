"""
Core Utilities Package

Shared utilities and helper functions for the World Engine consciousness systems.

Author: World Engine Team  
Date: October 7, 2025
Version: 2.0.0
"""

from .common import (
    setup_logging,
    generate_unique_id,
    compress_data,
    decompress_data,
    validate_consciousness_level,
    calculate_quantum_coherence,
    format_timestamp,
    safe_json_serialize,
    load_json_file,
    save_json_file,
    PerformanceMetrics,
    ConsciousnessCalculator,
    DataProcessor,
    EventEmitter,
    global_event_emitter,
    emit_consciousness_event
)

__version__ = "2.0.0"
__author__ = "World Engine Team"

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