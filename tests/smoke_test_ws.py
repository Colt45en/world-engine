import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

SERVER_PATH = Path('Documents/game101/Downloads/recovered_nucleus_eye/world-engine-feat-v3-1-advanced-math/working_python_server.py')

async def ws_ping_test():
    import websockets

    uri = 'ws://localhost:9100'
    # Connect and send a ping message in JSON
    async with websockets.connect(uri) as ws:
        await asyncio.sleep(0.1)
        await ws.send(json.dumps({'type': 'ping'}))
        reply = await ws.recv()
        data = json.loads(reply)
        assert data.get('type') in ('pong', 'status_response', 'math_result', 'echo')
        print('Smoke test: received:', data.get('type'))

def run_server_subprocess():
    # Start the server as a subprocess
    p = subprocess.Popen([sys.executable, str(SERVER_PATH)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Wait until server starts listening
    start_time = time.time()
    while True:
        line = p.stdout.readline()
        if not line and (time.time() - start_time) > 10:
            break
        if line:
            print(line.strip())
            if 'Server running on' in line or 'server listening on' in line:
                return p
    return p

def main():
    p = run_server_subprocess()
    try:
        asyncio.run(ws_ping_test())
        print('Smoke test: PASS')
    except Exception as e:
        print('Smoke test: FAIL', e)
        raise
    finally:
        p.kill()

if __name__ == '__main__':
    main()
