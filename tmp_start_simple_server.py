import asyncio
import importlib.util

spec = importlib.util.spec_from_file_location('mod', r'C:\Users\colte\World-engine\Documents\game101\Downloads\recovered_nucleus_eye\world-engine-feat-v3-1-advanced-math\simple_consciousness_server.py')
if spec is None or spec.loader is None:
    raise ImportError("Could not load module spec or loader.")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

server = module.SimpleConsciousnessServer(host='localhost', port=9201)

async def run():
    await server.start_server()

if __name__ == '__main__':
    try:
        asyncio.run(run())
    except Exception as e:
        print('SERVER_ERROR', e)
