import asyncio
import json
import websockets
from pathlib import Path

def get_ws_port():
    try:
        cfg = Path(__file__).resolve().parents[2] / 'ports.json'
        if cfg.exists():
            import json as _json
            data = _json.loads(cfg.read_text())
            return int(data.get('ws_port', 9100))
    except Exception:
        pass
    return 9100


async def main():
    port = get_ws_port()
    uri = f'ws://localhost:{port}'
    try:
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({'type': 'ping'}))
            reply = await ws.recv()
            print('Client received:', reply)
    except Exception as e:
        print('Client error:', e)

if __name__ == '__main__':
    asyncio.run(main())
