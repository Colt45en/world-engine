# 🌍 World Engine Studio

A unified lexicon processing and semantic analysis system with integrated recording, chat interface, and real-time analysis capabilities.

## 🚀 Quick Start

```powershell
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/world-engine-studio.git
cd world-engine-studio

# 2. (Optional) create a virtual environment
py -3 -m venv .venv
.\.venv\Scripts\activate

# 3. Install Python runtime deps
python -m pip install -U pip wheel
python -m pip install "spacy>=3.8" "pybind11>=2.11" "scikit-build-core>=0.9"
python -m spacy download en_core_web_md

# 4. Build the native extensions (see sections below for details)
python -m pip install -v -e quantum-thought
# Optional: build the asset bridge if you need realtime asset routing
python -m pip install -v -e assets_bridge

# 5. Launch the studio backend/frontends
python -m world_engine_unified.main server
```

Open [http://localhost:8000/web/studio.html](http://localhost:8000/web/studio.html) (or the Tier-4 IDE listed below).

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

- Open [http://localhost:8000/web/worldengine.html](http://localhost:8000/web/worldengine.html)
- API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

### Command Line Interface

```bash
python -m world_engine_unified.main cli
```

### Interactive Demo

```bash
python -m world_engine_unified.main demo
```

## 🛠️ Native Components

The repository ships two native extensions that power Tier-4 analytics and asset routing.

### Quantum Thought Pipeline (`qtp`)

- Location: `quantum-thought/`
- Built with: C++17, pybind11, scikit-build-core + CMake
- What it provides: the `qtp` Python package (`QuantumThoughtPipeline`, zone/agent accessors, typed NumPy views)

Build/install it into your active environment:

```powershell
python -m pip install -v -e quantum-thought
```

Make sure you have a compiler toolchain (MSVC on Windows, clang/LLVM or GCC elsewhere) and CMake ≥ 3.20 available on PATH. The accompanying smoke test lives in `quantum-thought/tests/test_qtp_smoke.py`; once `pytest` is installed you can validate the binding with:

```powershell
python -m pip install pytest
python -m pytest quantum-thought\tests\test_qtp_smoke.py
```

### Tier-4 Asset Bridge (`assets` module)

- Location: `assets_bridge/`
- Exposes the `AssetBridge` class that wraps `AssetResourceBridge.hpp`
- Pairs with `assets_bridge/assets_daemon.py` to translate NDJSON requests into async asset loads

Install the module (same toolchain prerequisites as above):

```powershell
python -m pip install -v -e assets_bridge
```

Once the module builds successfully, you can run the daemon and stream requests over STDIN:

```powershell
python assets_bridge\assets_daemon.py
```

Send NDJSON messages such as

```json
{"type": "ASSET_REGISTER", "payload": {"type": "mesh", "basePath": "C:/assets/meshes"}}
```

and listen on STDOUT for `ASSET_EVENT` updates (`loaded` / `error`). The daemon is designed to be wired into the Tier-4 WebSocket relay so that IDE operators can request preview assets on-demand.

## 🛰️ Tier-4 Nova Plugin

`tier4_nova_plugin.py` provides clustering, EWMA forecasting, and health scoring for `NovaIntelligence` instances while streaming NDJSON telemetry to `run.ndjson`.

Key traits:

- NumPy is optional: install it for fast clustering, or the plugin will fall back to lightweight tracking.
- State persists in `nova_history.json`, so successive runs pick up previous compression ratios.
- Emits `T4_EVENT` records (`NOVA/CLUSTER`, `NOVA/FORECAST`, `NOVA/HEALTH`) that the Tier-4 IDE can visualize.

Minimal usage example:

```python
from nova.core import NovaIntelligence
from tier4_nova_plugin import Tier4Nova

nova = NovaIntelligence()
nova.original_data = [0.12, 0.34, 0.56, 0.78]

plugin = Tier4Nova(nova)
for step in range(5): plugin.step(step)

print("last latent health", plugin.history[-1].latent_health)
```

The plugin writes telemetry to `run.ndjson`; point your NDJSON tailer or Tier-4 relay at that file to feed the IDE. If you want to integrate with a `RecursiveAgent`, pass it via the constructor so `log_imprint` receives each step summary.

## 🧩 Tier-4 Unified Bundle

The repository now ships with a single-file bundle, `tier4_unified_bundle.js`, combining the browser EngineRoom bridge and the Tier-4 Node relay/CLI utilities.

### Browser

1. Copy `tier4_unified_bundle.js` to your web server’s static assets (for example, `/web/js/`).
2. Include it before your EngineRoom bootstrap:

```html
<script src="/web/js/tier4_unified_bundle.js"></script>
<script>
       const iframe = document.getElementById('engine-room');
       const bridge = window.WorldEngineTier4.createTier4RoomBridge(iframe, 'ws://localhost:9000');
       bridge.applyOperator('ST');
</script>
```

### Node / CLI

Run the bundled relay or setup helper directly:

```powershell
node tier4_unified_bundle.js --relay --port 9000
node tier4_unified_bundle.js --setup --port 9000 --demo web/tier4_collaborative_demo.html
```

- `--relay` starts the Tier-4 WebSocket relay (defaults to port 9000).
- `--setup` bootstraps the relay, ensures `ws` is installed, and optionally opens the demo page.
- Pipe NDJSON events to `stdin` to enrich and broadcast them through the relay.

### Square–Triangle–Circle IDE

- Drop `stc_ide_unified.html` into your static host (for example, the `web/` folder).
- Open it directly to use the IDE with the inlined Upflow sharded IndexedDB store.
- When `window.StudioBridge` is present, subscribe to `lex.ingest { tokens, patterns }`, `lex.scale { updated, scales }`, and `lex.seed { scale, anchors }` events.

### Tier-4 Trainer

```python
from tier4_trainer import Tier4Trainer, Tier4Config
import torch

model = ...  # your WorldEngine
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10_000)

cfg = Tier4Config(
              lr=1e-4,
              weight_decay=0.01,
              max_epochs=50,
              accumulation_steps=4,
              amp=True,
              compile_model=False,
              use_ema=True,
              use_swa=False,
              log_dir="./runs/worldengine_v4",
              ckpt_dir="./checkpoints/worldengine_v4",
              w_rec=1.0,
              w_roles=1.0,
              w_contrastive=0.1,
)

trainer = Tier4Trainer(model, optimizer, scheduler, device="cuda", cfg=cfg)
trainer.fit(train_loader, val_loader)
trainer.operator("ST")
```

Notes:

- Automatically wraps in DistributedDataParallel when `torchrun` initializes process group.
- EMA, SWA, mixed precision, accumulation, and checkpoints are all configurable via `Tier4Config`.
- Operators (`ST`, `UP`, `PR`, `CV`, `RB`, `RS`) allow mid-training adjustments (stabilize, scheduler step, report, converge, rollback, reset).

## Project Structure

```text
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
