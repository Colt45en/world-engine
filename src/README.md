# Source Code Organization

This directory contains the organized source code for the World Engine project.

## Directory Structure

### `python/` - Python Engine Implementations
- `world_engine_v4.py` - Latest World Engine implementation
- `world_engine.py` - Core World Engine
- `world_engine_robust.py` - Robust variant implementation
- `train_step.py` - Training step logic

### `typescript/` - Tier-4 System Components
- `tier4_meta_system.ts` - Core Tier-4 meta system
- `tier4_meta_system_complete.ts` - Complete Tier-4 implementation
- `tier4_audio_bridge.ts` - Audio integration bridge
- `tier4_integration_enhancements.ts` - Integration enhancements
- `tier4_nlp.ts` - Natural language processing components
- `tier4_websocket_sync.ts` - WebSocket synchronization
- `world_engine_tier4_ide.ts` - IDE integration

### `graph_neural/` - Graph Neural Network Components
- `graph_encoder_tg.py` - Graph encoder implementation
- `rel_gatv2_encoder_tg.py` - Relational GAT v2 encoder
- `collate_tg.py` - Graph collation utilities

### `demos/` - Demonstration and Example Code
- `demo.py` - Main demonstration script
- `examples/` - Additional example implementations

### `optimization/` - Performance Optimization
- `opt_type_scale.py` - Type scaling optimizations

### `utils/` - Utility Functions and Tools
- `factor_naming.py` - Factor naming utilities
- `umap_zviz.py` - UMAP visualization tools
- `QuantumThoughtPipeline.hpp` - C++ pipeline header

### `scripts/` - Automation Scripts
- `GetItem.lua` - Item retrieval script
- `WeightedPicker.lua` - Weighted selection script

## Entry Points

- `main.py` - Primary entry point for the World Engine system

## Usage

Each subdirectory contains specialized components. Import from the appropriate subdirectory based on your needs:

```python
# Python engines
from src.python.world_engine_v4 import WorldEngine

# Graph neural components
from src.graph_neural.graph_encoder_tg import GraphEncoder

# Utilities
from src.utils.factor_naming import FactorNamer
```
