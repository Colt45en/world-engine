# world-engine

## Fractal Intelligence Dashboard v6.0

An upgraded control surface now streams live data from the Fractal Intelligence Engine through a lightweight FastAPI service.

### Run the API bridge

Use the Python 3.11 virtual environment and install dependencies if needed:

```powershell
pip install -r requirements.txt
python Documents/game101/Downloads/recovered_nucleus_eye/world-engine-feat-v3-1-advanced-math/services/fractal_dashboard_service.py
```

The service listens on `http://localhost:8600` and exposes engine controls, pain metrics, and insight logs for the dashboard.

### Launch the dashboard

Serve the Layer5 assets (or open the HTML file directly) and visit `fractal-intelligence-dashboard.html`. The page now:

- Polls `http://localhost:8600/engine/state` for live iteration metrics
- Streams knowledge insights into a timeline with export button
- Reflects Nick's compression stats and pain summaries in real time
- Lets you trigger pain injections, force optimization, and run compression diagnostics from the UI

If the pain-detection API is offline, the dashboard still tracks the engine while marking the integration as disconnected.

### HSV shader color demo

For quick color explorations or AI-facing demos, open `apps/shader-demo/index.html` in any modern browser. The page bundles:

- An HSV wheel with brightness ramp, contrast guardrails, and swatch holders
- A six-swatch timing harness for collecting turn-to-completion metrics
- A WebGL fragment-shader playground wired to the currently selected color (`u_color`)

No build step is required—double-click the file or host it from a static server to keep localStorage scoped per origin.
