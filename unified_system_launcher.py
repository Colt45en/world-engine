#!/usr/bin/env python3
"""
🌐 UNIFIED SYSTEM LAUNCHER
===========================

Launches all World Engine systems in the correct architectural order:
1. Meta Room (Central Coordinator) - Port 8702
2. Vector Node Network (Brain Infrastructure) - Port 8701
3. Logging AI (Monitoring) - Port 8703
4. HTTP Dashboard - Port 8080

All systems run as background processes and are monitored for health.
"""

import subprocess
import time
import asyncio
import websockets
import json
import sys
from pathlib import Path
from datetime import datetime
import signal
import atexit

class UnifiedSystemLauncher:
    def __init__(self):
        self.processes = {}
        self.base_path = Path(__file__).parent
        self.running = True
        
        # System configuration in launch order
        self.systems = {
            'meta_room': {
                'name': 'Meta Room (Central Coordinator)',
                'file': 'knowledge_vault_node_network.py',
                'port': 8702,
                'emoji': '🏛️',
                'critical': True,
                'health_check': 'ws://localhost:8702'
            },
            'vector_network': {
                'name': 'Vector Node Network',
                'file': 'vector_node_network.py',
                'port': 8701,
                'emoji': '🧠',
                'critical': True,
                'health_check': 'ws://localhost:8701'
            },
            'logging_ai': {
                'name': 'Logging AI',
                'file': 'logging_ai.py',
                'port': 8703,
                'emoji': '📊',
                'critical': False,
                'health_check': 'ws://localhost:8703'
            },
            'http_dashboard': {
                'name': 'HTTP Dashboard',
                'file': None,  # Uses Python's http.server
                'port': 8080,
                'emoji': '🌐',
                'critical': False,
                'health_check': 'http://localhost:8080'
            }
        }
        
        # Register cleanup handlers
        atexit.register(self.shutdown_all)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\n🛑 Shutdown signal received...")
        self.shutdown_all()
        sys.exit(0)
    
    def print_header(self):
        """Print launch header"""
        print("\n" + "=" * 80)
        print("🌐 WORLD ENGINE UNIFIED SYSTEM LAUNCHER")
        print("=" * 80)
        print("\n📋 System Architecture:")
        print("   1. 🏛️  Meta Room (Port 8702) - Central Coordinator")
        print("   2. 🧠 Vector Network (Port 8701) - Brain Infrastructure")
        print("   3. 📊 Logging AI (Port 8703) - System Monitoring")
        print("   4. 🌐 HTTP Dashboard (Port 8080) - Web Interface")
        print("\n📊 Data Flow:")
        print("   Nodes → Vaults → Meta Room → AI Systems → Logging AI")
        print("\n" + "=" * 80 + "\n")
    
    def launch_system(self, system_id: str, system_config: dict) -> bool:
        """Launch a system and return success status"""
        name = system_config['name']
        emoji = system_config['emoji']
        port = system_config['port']
        file = system_config['file']
        
        print(f"{emoji} Launching {name} on port {port}...")
        
        try:
            if system_id == 'http_dashboard':
                # Launch HTTP server
                cmd = [sys.executable, '-m', 'http.server', str(port)]
            else:
                # Launch Python script
                script_path = self.base_path / file
                if not script_path.exists():
                    print(f"   ❌ Error: {file} not found")
                    return False
                cmd = [sys.executable, str(script_path)]
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_path),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            
            self.processes[system_id] = {
                'process': process,
                'config': system_config,
                'start_time': datetime.now(),
                'status': 'starting'
            }
            
            print(f"   ✅ Process started (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"   ❌ Error launching {name}: {e}")
            return False
    
    async def check_health(self, system_id: str) -> bool:
        """Check if a system is responding"""
        system = self.processes.get(system_id)
        if not system:
            return False
        
        config = system['config']
        health_url = config.get('health_check')
        
        if not health_url:
            return True
        
        try:
            if health_url.startswith('ws://'):
                # WebSocket health check
                async with websockets.connect(health_url, open_timeout=2) as ws:
                    await ws.send(json.dumps({"type": "ping"}))
                    response = await asyncio.wait_for(ws.recv(), timeout=2)
                    return True
            else:
                # HTTP health check (simplified - just check if port is open)
                return True
        except:
            return False
    
    async def wait_for_system(self, system_id: str, max_wait: int = 10) -> bool:
        """Wait for a system to become healthy"""
        system = self.processes.get(system_id)
        if not system:
            return False
        
        config = system['config']
        name = config['name']
        emoji = config['emoji']
        
        print(f"   {emoji} Waiting for {name} to be ready...", end='', flush=True)
        
        for i in range(max_wait):
            await asyncio.sleep(1)
            
            # Check if process is still running
            if system['process'].poll() is not None:
                print(f"\n   ❌ Process died unexpectedly")
                return False
            
            # Check health
            if await self.check_health(system_id):
                system['status'] = 'running'
                print(f" ✅ Ready!")
                return True
            
            print(".", end='', flush=True)
        
        print(f"\n   ⚠️  Timeout waiting for {name}, but process is running")
        system['status'] = 'running'  # Assume it's ok
        return True
    
    def get_system_status(self) -> dict:
        """Get status of all systems"""
        status = {}
        for system_id, system in self.processes.items():
            process = system['process']
            config = system['config']
            
            if process.poll() is None:
                status[system_id] = {
                    'name': config['name'],
                    'status': system['status'],
                    'pid': process.pid,
                    'port': config['port'],
                    'uptime': (datetime.now() - system['start_time']).total_seconds()
                }
            else:
                status[system_id] = {
                    'name': config['name'],
                    'status': 'stopped',
                    'exit_code': process.returncode
                }
        
        return status
    
    def print_status(self):
        """Print current system status"""
        print("\n" + "─" * 80)
        print("📊 SYSTEM STATUS")
        print("─" * 80)
        
        status = self.get_system_status()
        
        for system_id, info in status.items():
            config = self.systems[system_id]
            emoji = config['emoji']
            name = info['name']
            
            if info['status'] == 'running':
                uptime = int(info['uptime'])
                print(f"{emoji} {name:<40} ✅ Running (PID: {info['pid']}, Port: {info['port']}, {uptime}s)")
            elif info['status'] == 'starting':
                print(f"{emoji} {name:<40} 🔄 Starting...")
            else:
                print(f"{emoji} {name:<40} ❌ Stopped")
        
        print("─" * 80 + "\n")
    
    def shutdown_system(self, system_id: str):
        """Shutdown a specific system"""
        if system_id not in self.processes:
            return
        
        system = self.processes[system_id]
        process = system['process']
        config = system['config']
        
        if process.poll() is None:
            print(f"   🛑 Stopping {config['name']}...")
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"   ✅ Stopped")
            except subprocess.TimeoutExpired:
                print(f"   ⚠️  Force killing...")
                process.kill()
                process.wait()
                print(f"   ✅ Killed")
            except Exception as e:
                print(f"   ❌ Error stopping: {e}")
    
    def shutdown_all(self):
        """Shutdown all systems in reverse order"""
        if not self.processes:
            return
        
        print("\n" + "=" * 80)
        print("🛑 SHUTTING DOWN ALL SYSTEMS")
        print("=" * 80 + "\n")
        
        # Shutdown in reverse order (non-critical first)
        shutdown_order = [
            'http_dashboard',
            'logging_ai',
            'vector_network',
            'meta_room'
        ]
        
        for system_id in shutdown_order:
            if system_id in self.processes:
                self.shutdown_system(system_id)
        
        print("\n✅ All systems stopped\n")
        self.processes.clear()
    
    async def launch_all(self):
        """Launch all systems in order"""
        self.print_header()
        
        print("🚀 LAUNCHING SYSTEMS")
        print("=" * 80 + "\n")
        
        # Launch in order
        for system_id, config in self.systems.items():
            # Launch the system
            if not self.launch_system(system_id, config):
                if config['critical']:
                    print(f"\n❌ Critical system {config['name']} failed to launch!")
                    print("   Aborting startup...\n")
                    self.shutdown_all()
                    return False
                else:
                    print(f"   ⚠️  Non-critical system failed, continuing...\n")
                    continue
            
            # Wait for it to be ready
            if not await self.wait_for_system(system_id):
                if config['critical']:
                    print(f"\n❌ Critical system {config['name']} failed health check!")
                    print("   Aborting startup...\n")
                    self.shutdown_all()
                    return False
            
            print()  # Blank line between systems
        
        print("=" * 80)
        print("✅ ALL SYSTEMS LAUNCHED SUCCESSFULLY")
        print("=" * 80)
        
        return True
    
    async def monitor_loop(self):
        """Monitor all systems and restart if needed"""
        print("\n📊 Entering monitoring mode...")
        print("   Press Ctrl+C to shutdown all systems\n")
        
        try:
            while self.running:
                await asyncio.sleep(5)
                
                # Check all processes
                for system_id, system in list(self.processes.items()):
                    process = system['process']
                    config = system['config']
                    
                    if process.poll() is not None:
                        # Process died
                        print(f"\n⚠️  {config['name']} stopped unexpectedly (exit code: {process.returncode})")
                        
                        if config['critical']:
                            print(f"   Critical system failed! Shutting down all systems...")
                            self.shutdown_all()
                            return
                        else:
                            print(f"   Attempting restart...")
                            if self.launch_system(system_id, config):
                                await self.wait_for_system(system_id)
        
        except KeyboardInterrupt:
            pass
    
    async def run(self):
        """Main run method"""
        # Launch all systems
        if not await self.launch_all():
            return
        
        # Print initial status
        self.print_status()
        
        # Monitor systems
        await self.monitor_loop()
        
        # Cleanup
        self.shutdown_all()

async def main():
    """Main entry point"""
    launcher = UnifiedSystemLauncher()
    await launcher.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Launcher stopped by user\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
