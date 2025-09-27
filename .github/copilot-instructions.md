# AI Agent Instructions - World Engine

This is a lexicon processing and semantic analysis system with integrated recording, chat interface, and real-time analysis capabilities.

## Core Architecture

**Multi-Component Bridge System**: Three main controllers communicate via `StudioBridge`:
- `ChatController` - Command router and workflow coordinator
- `EngineController` - Lexicon analysis engine wrapper
- `RecorderController` - Audio/video capture with timeline markers

**Data Flow**: `Chat Controller ↔ Recorder ↔ World Engine ↔ External Store`

## Key Components

### Semantic Scaling System (`scales/`)
- **Seed Manager** (`scales/seeds.py`): Hand-labeled words with semantic values (-1.0 to 1.0)
- **DEFAULT_SEEDS**: Predefined sentiment words (`terrible: -0.8`, `excellent: 0.8`, etc.)
- **Constraints**: Ordering relationships (`terrible < bad < neutral < good < excellent`)

### NLP Pipeline (`context/`)
- **TextParser** (`context/parser.py`): spaCy-powered linguistic processing
- **Token Analysis**: POS tagging, dependency parsing, entity recognition
- **Scoring Logic**: Seed lookup + contextual analysis + constraint validation

### Web Interface (`web/`)
- **Studio Interface**: `web/studio.html` - Complete interface with Chat, Engine, and Recorder
- **Engine Only**: `web/worldengine.html` - Just the lexicon analysis engine
- **Controllers**: Independent but coordinated via event messaging

## Entry Points

- **Web Interface**: `python launch_studio.py` → Open `web/studio.html`
- **API Server**: `python main.py` → http://localhost:8000/web/studio.html
- **Demo**: `python demo.py` → Interactive demo
- **CLI**: Direct Python scripts for testing

## Development Workflows

### Running the System
```bash
# Quick launch
python launch_studio.py

# Full API server
python main.py

# Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### File Structure
```
api/                     # FastAPI service endpoints
config/                  # Configuration files
context/                 # NLP processing modules
scales/                  # Semantic scaling system
web/                     # Web interfaces
├── studio.html         # Main studio interface
├── worldengine.html    # Engine-only view
├── chat-controller.js  # Chat controls
├── engine-controller.js # Engine controls
└── recorder-controller.js # Recording controls
```

## Project-Specific Conventions

### Semantic Scoring Approach
- **Hand-labeled seeds**: Core truth values, not ML-generated
- **Constraint-based scaling**: Explicit ordering relationships
- **Context-sensitive scoring**: Same word scored differently based on syntactic role

### API Usage Pattern
```python
from api.service import create_app, WorldEngineAPI
from scales.seeds import DEFAULT_SEEDS, DEFAULT_CONSTRAINTS

api = WorldEngineAPI()
for word, value in DEFAULT_SEEDS.items():
    api.seed_manager.add_seed(word, value)
```

## Critical Dependencies

**Python**: FastAPI, spaCy (en_core_web_sm), numpy, pandas, scikit-learn
**Web**: Vanilla JavaScript, no bundler required
**Architecture**: Component communication via message bridge system

Focus on the semantic scaling system and component communication patterns - these are unique to this project.
