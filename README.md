# 🌍 World Engine Studio

A unified lexicon processing and semantic analysis system with integrated recording, chat interface, and real-time analysis capabilities.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/world-engine-studio.git
cd world-engine-studio

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Launch the studio
python launch_studio.py
```

Open http://localhost:8000/web/studio.html

## ✨ Features

- **🎤 Recording Studio** - Audio/video capture with timeline markers
- **💬 AI Chat Interface** - Command-driven workflow control
- **🌍 World Engine** - Advanced lexicon analysis and semantic scaling
- **🔗 Event Bridge** - Real-time component communication
- **📊 Data Integration** - Automatic linking of recordings to analysis runs
- **🎯 Voice Commands** - Speech-to-command processing

## 🏗️ Architecture

```
📱 Chat Controller ← → 🎙️ Recorder
       ↕️
🌍 World Engine (Lexicon Analysis)
       ↕️
📊 External Store (Persistent Data)
```

## 🎮 Usage

### Web Interface (Recommended)
```bash
python -m world_engine_unified.main server
```
- Open http://localhost:8000/web/worldengine.html
- API docs at http://localhost:8000/docs

### Command Line Interface
```bash
python -m world_engine_unified.main cli
```

### Interactive Demo
```bash
python -m world_engine_unified.main demo
```

## Project Structure

```
world_engine_unified/
├── __init__.py              # Main package
├── main.py                  # Entry point launcher
├── demo.py                  # Demo script
├── requirements.txt         # Python dependencies
├── scales/                  # Semantic scaling
│   ├── __init__.py
│   ├── seeds.py            # Hand-labeled seed words
│   ├── graph.py            # Synonym/antonym graphs
│   ├── isotonic.py         # Order-preserving calibration
│   ├── embeddings.py       # Neighbor expansion
│   └── typescores.py       # Type-level vectors
├── context/                 # NLP processing
│   ├── __init__.py
│   ├── parser.py           # spaCy pipeline
│   ├── rules.py            # Linguistic rules
│   ├── senses.py           # Sense disambiguation
│   ├── sarcasm.py          # Sarcasm detection
│   ├── domains.py          # Domain adaptation
│   └── scorer.py           # Token scoring
├── api/                     # FastAPI service
│   ├── __init__.py
│   └── service.py          # REST endpoints
├── web/                     # Web interfaces
│   ├── worldengine.html    # Main lexicon interface
│   └── word_engine_init.js # Initialization
├── config/                  # Configuration files
│   ├── toggle_schema.json  # View definitions
│   └── versions.json       # Version tracking
└── data/                    # Data storage
    └── (CSV files, vectors)
```

## Key Features

- **Seed-based Scoring**: Hand-labeled words with semantic values
- **Contextual Analysis**: spaCy-powered linguistic processing
- **Web Interface**: Interactive lexicon exploration
- **REST API**: Programmatic access to all features
- **Configurable Views**: Toggle between lexicon and runes modes
- **Extensible**: Modular design for easy enhancement

## API Endpoints

- `POST /score_word` - Score individual words
- `POST /score_token` - Analyze text tokens
- `POST /scale_between` - Compare word scales
- `GET /seeds` - Get seed words and constraints
- `GET /health` - System health check

## Configuration

Edit `config/toggle_schema.json` to customize:
- Data source paths
- View definitions
- Column mappings
- Default settings
