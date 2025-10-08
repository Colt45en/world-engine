import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from pathlib import Path as _P

def get_ws_port():
    try:
        cfg = _P(__file__).resolve().parents[2] / 'ports.json'
        if cfg.exists():
            import json as _json
            data = _json.loads(cfg.read_text())
            return int(data.get('ws_port', 9100))
    except Exception:
        pass
    return 9100

SERVER_PATH = Path('Documents/game101/Downloads/recovered_nucleus_eye/world-engine-feat-v3-1-advanced-math/working_python_server.py')
VENV_PY = Path('.venv311') / 'Scripts' / 'python.exe'
READY_FILE = Path('server_ready.tmp')

async def ws_ping_test(uri: str):
    import websockets

    # Connect and send a ping message in JSON
    async with websockets.connect(uri) as ws:
        # The server sends a welcome message on connect; consume it if present
        try:
            first = await asyncio.wait_for(ws.recv(), timeout=1.0)
            try:
                first_data = json.loads(first)
                if first_data.get('type') == 'welcome':
                    print('Received welcome message, continuing to ping test')
                else:
                    # If first message isn't welcome, it may be a reply to our future ping
                    print('Received initial message:', first_data.get('type'))
            except Exception:
                # Non-JSON or unexpected; ignore and continue
                pass
        except asyncio.TimeoutError:
            # No welcome message arrived quickly; that's fine
            pass

        await asyncio.sleep(0.05)
        await ws.send(json.dumps({'type': 'ping'}))
        reply = await ws.recv()
        data = json.loads(reply)
        assert data.get('type') in ('pong', 'status_response', 'math_result', 'echo')
        print('Smoke test: received:', data.get('type'))

def run_server_subprocess():
    # Choose interpreter: prefer .venv311 python if available
    interpreter = str(VENV_PY) if VENV_PY.exists() else sys.executable

    # Start the server as a subprocess
    p = subprocess.Popen([interpreter, str(SERVER_PATH)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Wait for readiness file that contains the started PID, or a server stdout line
    start_time = time.time()
    detected_port = None
    while True:
        # Check readiness file; ensure it points to our started PID
        try:
            if READY_FILE.exists():
                try:
                    pid_text = READY_FILE.read_text().strip()
                    if pid_text and int(pid_text) == p.pid:
                        print(f'Detected readiness file for PID {p.pid}')
                        return p
                    else:
                        # readiness file exists but for a different PID; keep waiting
                        print(f'Readiness file exists for PID {pid_text}; waiting for PID {p.pid}')
                except Exception:
                    pass
        except Exception:
            pass

        # Otherwise read stdout lines for a short while
        if p.stdout:
            line = p.stdout.readline()
        else:
            line = None

        if line:
            print(line.strip())
            # Try to parse the printed URL/Port
            if 'URL:' in line and 'ws://' in line:
                # Example line: "🌐 URL: ws://localhost:9100"
                try:
                    parts = line.split('ws://')[-1].strip()
                    hostport = parts.split()[0]
                    detected_port = int(hostport.split(':')[-1])
                    print(f'Parsed server port: {detected_port}')
                except Exception:
                    pass
            if 'Server running on' in line or 'server listening on' in line:
                # Prefer parsed port if available
                return p, detected_port

        if (time.time() - start_time) > 20:
            print('Timeout waiting for server start')
            # If readiness file never matched, kill the started process and return
            try:
                p.kill()
            except Exception:
                pass
            return p, detected_port

def main():
    p, detected_port = run_server_subprocess()
    uri = f'ws://localhost:{detected_port or get_ws_port()}'
    try:
        asyncio.run(ws_ping_test(uri))
        print('Smoke test: PASS')
    except Exception as e:
        print('Smoke test: FAIL', e)
        raise
    finally:
        try:
            p.kill()
        except Exception:
            pass

if __name__ == '__main__':
    main()
