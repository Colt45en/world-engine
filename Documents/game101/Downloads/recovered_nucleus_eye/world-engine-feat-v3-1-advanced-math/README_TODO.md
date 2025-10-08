# Project TODO - Quick README Task List

This file is a small, actionable TODO list for the World-engine workspace. Use it as a quick checklist while debugging and running services.

## High-level status
- [x] Dependencies verified (numpy, pandas, websockets, psutil, aiohttp, jsonschema, etc.)
- [x] `consciousness_websocket_server.py` import fixes applied (safe fallbacks)
- [x] `core/utils/performance_metrics.py` cleaned and `get_duration()` added
- [x] `core/utils/timestamp_utils.py` cleaned

## Priority TODOs (short-term)
- [ ] Start and smoke-test `consciousness_websocket_server.py` on a safe port (e.g. 9201)
  - Command: `python consciousness_websocket_server.py`
  - Expected: `✅ WebSocket server started successfully` in logs
- [ ] Fix `simple_consciousness_server.py` imports and run smoke test
- [ ] Resolve `vector_node_network.py` port conflicts and run analytics mode on an assigned port (e.g. 9200)
  - Command: `python vector_node_network.py --analytics-mode --port 9200`
- [ ] Run `analyze_consciousness_patterns.py` and handle any timestamp/parse errors
- [ ] Test `master_nexus_codepad_simple.py` end-to-end with Nexus client (start CODEPAD then send a `status` command)

## Secondary TODOs (medium-term)
- [ ] Replace stub fallbacks with real `core` implementations where available
- [ ] Add unit tests for key modules: `consciousness_websocket_server`, `vector_node_network`, `pain_opportunity` system
- [ ] Add a small script to allocate non-conflicting ports and write `port_config.json` for services
- [ ] Document each service's ports and health-check endpoints in `SERVICES.md`

## Quick run & debug commands (PowerShell)
- Change to project root:

  Set-Location "C:\Users\colte\World-engine\Documents\game101\Downloads\recovered_nucleus_eye\world-engine-feat-v3-1-advanced-math"

- Run consciousness WebSocket server (safe port selection recommended):

  python consciousness_websocket_server.py

- Run vector network analytics on port 9200:

  python vector_node_network.py --analytics-mode --port 9200

- Run pain/opportunity system:

  python implement_pain_opportunity_system.py

- Quick module import test (no servers started):

  python -c "import importlib.util,sys; spec=importlib.util.spec_from_file_location('m','consciousness_websocket_server.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('Import OK')"

## How to mark progress
- Check a box above when a task is verified manually.
- If you make code changes, run the appropriate smoke test immediately and add a brief log line to `nexus_codepad.log` or the relevant service log.

## Notes & context
- I inserted small stub fallbacks in `consciousness_websocket_server.py` to avoid application-wide ImportError when some `core` modules are absent; these stubs are temporary and should be replaced with real implementations when available.
- Many lint/type warnings from static analysis (pyright) are benign for now; prioritize runtime errors first.

---
If you want, I can:
- Start the consciousness server on port 9201 and stream logs here.
- Create `SERVICES.md` documenting ports and run commands.
- Auto-generate `port_config.json` and a small `run_all.ps1` script to start services on non-conflicting ports.

Tell me which of those to do next.