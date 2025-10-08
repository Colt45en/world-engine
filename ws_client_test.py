import asyncio
import websockets
import json
from pathlib import Path

def get_ws_port():
    try:
        cfg = Path(__file__).resolve().parents[1] / 'ports.json'
        if cfg.exists():
            import json as _json
            data = _json.loads(cfg.read_text())
            return int(data.get('ws_port', 8701))
    except Exception:
        pass
    return 8701


async def test():
    port = get_ws_port()
    uri = f'ws://localhost:{port}'
    try:
        async with websockets.connect(uri, ping_interval=10, ping_timeout=5) as ws:
            print('CLIENT: connected to', uri)
            initial = await ws.recv()
            print('CLIENT: initial:', initial)
            await ws.send(json.dumps({'type':'ping'}))
            pong = await ws.recv()
            print('CLIENT: pong:', pong)
    except Exception as e:
        print('CLIENT_ERROR', e)

if __name__ == '__main__':
    asyncio.run(test())
