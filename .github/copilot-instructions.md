# World Engine - GitHub Copilot Instructions

**ALWAYS follow these instructions first. Only fallback to additional search and context gathering if the information in these instructions is incomplete or found to be in error.**

World Engine is a unified lexicon processing and semantic analysis system built with Python (FastAPI, spaCy) and JavaScript/TypeScript. It provides web interfaces for interactive lexicon exploration, CLI tools, and REST API endpoints for programmatic access.

## Quick Setup & Validation

**Bootstrap the repository:**
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install spaCy English model 
python -m spacy download en_core_web_sm

# 3. Install Node.js dependencies (for linting/TypeScript)
npm install
```

**Validate installation:**
```bash
# Test CLI (takes ~1 second)
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✓ spaCy working')"

# Test FastAPI import
python -c "import fastapi, uvicorn; print('✓ FastAPI working')"
```

## Running the Application

### Web Interface (Recommended)
```bash
# Start web server - NEVER CANCEL: Allow 15+ seconds for startup
# Timeout: Set to 180+ seconds minimum
python -c "
import sys
sys.path.insert(0, '.')
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path

app = FastAPI(title='World Engine API', version='0.1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
web_dir = Path('./web')
if web_dir.exists():
    app.mount('/web', StaticFiles(directory=str(web_dir)), name='web')

@app.get('/')
async def root():
    return {'message': 'World Engine API', 'status': 'running'}

@app.get('/health') 
async def health():
    return {'status': 'healthy', 'version': '0.1.0'}

print('Starting World Engine at http://localhost:8000')
print('Web interface: http://localhost:8000/web/worldengine.html')
print('API docs: http://localhost:8000/docs')
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
"
```

- **Access:** http://localhost:8000/web/worldengine.html
- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

### Command Line Interface
```bash
# Quick CLI test (starts in ~1 second)
python -c "
import spacy
nlp = spacy.load('en_core_web_sm')
text = 'This is an excellent example of natural language processing'
doc = nlp(text)
print(f'Text: {text}')
print('Tokens:')
for token in doc:
    if token.is_alpha and not token.is_stop:
        print(f'  - {token.text} ({token.lemma_}) [{token.pos_}]')
"
```

## Build and Linting

### JavaScript/TypeScript
```bash
# Lint JavaScript (has known formatting issues - non-blocking)
npm run lint
# Fix most issues automatically
npm run lint:fix

# TypeScript compilation (has minor syntax error in word_engine_init.js - non-blocking)
npm run build
```

### Python
```bash
# No specific Python linting configured, but you can run:
python -m py_compile *.py api/*.py context/*.py scales/*.py
```

## Manual Validation Scenarios

**ALWAYS run these validation steps after making changes:**

### 1. Web Interface Validation
1. Start the web server using the command above
2. Navigate to http://localhost:8000/web/worldengine.html
3. Verify the interface loads with default text in the input box
4. Click one of the test buttons (e.g., "happiness") 
5. Verify it loads the word into the input and shows analysis results
6. Click "Actions" to expand, then click "Analyze" to process custom text
7. Enter custom text like "excellent amazing terrible bad"
8. Verify JSON output shows lexical analysis with prefix/root/suffix detection
9. Verify the Visual panel shows "1 parsed • animated"

### 2. CLI Validation
1. Run the CLI test command above
2. Verify spaCy processes the text and shows tokenized output
3. Verify POS tagging works (should show ADJ, NOUN, etc.)

### 3. API Validation
1. Start the web server
2. Test health endpoint: `curl http://localhost:8000/health`
3. Verify response: `{"status":"healthy","version":"0.1.0"}`
4. Test API docs: `curl http://localhost:8000/docs | head -5`
5. Verify swagger UI loads

## Timing and Timeout Requirements

**CRITICAL: NEVER CANCEL these operations. Set appropriate timeouts:**

- **CLI startup:** ~1 second (timeout: 30 seconds)
- **Web server startup:** ~10-15 seconds (timeout: 180+ seconds minimum)
- **spaCy model loading:** ~2-3 seconds first time (timeout: 60 seconds)
- **NPM install:** ~30-60 seconds (timeout: 300 seconds)
- **pip install:** ~60-120 seconds (timeout: 300 seconds)

## Project Structure

```
/
├── README.md              # Project documentation
├── main.py               # Entry point (has import path issues)
├── requirements.txt      # Python dependencies  
├── package.json         # Node.js dependencies & scripts
├── .eslintrc.json       # ESLint configuration
├── tsconfig.json        # TypeScript configuration
├── api/                 # FastAPI service layer
│   ├── __init__.py
│   └── service.py       # REST endpoints, create_app()
├── context/             # NLP processing with spaCy
│   ├── __init__.py
│   └── parser.py        # TextParser class, Token/ParsedSentence
├── scales/              # Semantic scaling system
│   ├── __init__.py
│   └── seeds.py         # SeedManager, DEFAULT_SEEDS
├── web/                 # Web interfaces
│   ├── worldengine.html # Main lexicon interface ✓ WORKING
│   ├── studio.html      # Studio interface
│   ├── engine-controller.js
│   ├── chat-controller.js
│   ├── recorder-controller.js
│   ├── studio-bridge.js
│   ├── word_engine_init.js  # Has syntax error (non-blocking)
│   └── types.d.ts
└── config/              # Configuration files
    ├── toggle_schema.json
    └── versions.json
```

## Known Issues and Workarounds

### Import Path Issues
The main.py file uses `world_engine_unified` imports but the directory is not named that way. **Workaround:** Use the minimal server launcher provided above instead of main.py.

### Linting Issues  
ESLint reports 98 formatting errors (mostly indentation). **Impact:** Non-blocking, application works correctly.

### TypeScript Compilation
Build fails on syntax error in web/word_engine_init.js line 17. **Impact:** Non-blocking, web interface works correctly.

### No Tests
Repository has no test files or test runner configured. **Recommendation:** Validate manually using the scenarios above.

## Key Components and Entry Points

- **API Entry:** `api/service.py` - Contains `create_app()` and `WorldEngineAPI` class
- **NLP Processing:** `context/parser.py` - Contains `TextParser` with spaCy integration  
- **Semantic Seeds:** `scales/seeds.py` - Contains `SeedManager` and `DEFAULT_SEEDS`
- **Web Interface:** `web/worldengine.html` - Fully functional lexicon analysis UI
- **Main Interface:** Use the web interface at `/web/worldengine.html` for interactive work

## Example Commands Reference

```bash
# Quick health check
curl http://localhost:8000/health

# Start development server (NEVER CANCEL - wait for completion)
[Use Python launcher script above]

# Test spaCy functionality  
python -c "import spacy; nlp=spacy.load('en_core_web_sm'); print([(t.text, t.pos_) for t in nlp('test text')])"

# Check dependencies
python -c "import fastapi, uvicorn, spacy, numpy, pandas; print('All deps OK')"

# JavaScript linting (non-blocking if it fails)
npm run lint || echo "Linting issues present but non-blocking"
```

Remember: The web interface at http://localhost:8000/web/worldengine.html is the primary working interface for interacting with World Engine's lexicon processing capabilities.