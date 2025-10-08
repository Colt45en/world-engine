import asyncio
import importlib.util
import os

MODULE_PATH = os.path.abspath(r'C:\Users\colte\World-engine\Documents\game101\Downloads\recovered_nucleus_eye\world-engine-feat-v3-1-advanced-math\simple_consciousness_server.py')

spec = importlib.util.spec_from_file_location('simple_consciousness_server', MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load spec for module at {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

server = module.SimpleConsciousnessServer(host='localhost', port=8701)

async def run():
    print('SERVER_RUNNER: starting server on localhost:8701')
    await server.start_server()

if __name__ == '__main__':
    try:
        asyncio.run(run())
    except Exception as e:
        print('SERVER_RUNNER_ERROR', e)
