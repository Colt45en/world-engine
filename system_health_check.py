#!/usr/bin/env python3
"""
SYSTEM HEALTH CHECK
===================
Comprehensive diagnostic tool to verify all AI systems, connections, and services.
"""

import json
import socket
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

class SystemHealthChecker:
    """Comprehensive system health checker"""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.root_dir = Path(__file__).parent
        
    def check_port(self, port: int, name: str) -> bool:
        """Check if a port is in use"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def check_file_exists(self, filepath: str) -> bool:
        """Check if a file exists"""
        return (self.root_dir / filepath).exists()
    
    def load_json_config(self, filepath: str) -> Dict[str, Any]:
        """Load and validate JSON configuration"""
        try:
            with open(self.root_dir / filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    
    def check_python_imports(self) -> Dict[str, bool]:
        """Check if critical Python packages are available"""
        packages = {
            'numpy': False,
            'pandas': False,
            'websockets': False,
            'sympy': False,
            'scipy': False,
            'nltk': False
        }
        
        for package in packages.keys():
            try:
                __import__(package)
                packages[package] = True
            except ImportError:
                packages[package] = False
        
        return packages
    
    async def test_websocket_connection(self, port: int) -> Dict[str, Any]:
        """Test WebSocket connection"""
        try:
            import websockets
            uri = f"ws://localhost:{port}"
            async with websockets.connect(uri, open_timeout=2) as ws:
                test_msg = json.dumps({"type": "health_check", "timestamp": "2025-10-07"})
                await ws.send(test_msg)
                response = await asyncio.wait_for(ws.recv(), timeout=2)
                return {"connected": True, "response": response[:100]}
        except Exception as e:
            return {"connected": False, "error": str(e)}
    
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}{title:^70}{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}\n")
    
    def print_status(self, item: str, status: bool, details: str = ""):
        """Print status line"""
        icon = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
        status_text = f"{GREEN}ONLINE{RESET}" if status else f"{RED}OFFLINE{RESET}"
        print(f"{icon} {item:.<50} {status_text}")
        if details:
            print(f"  └─ {YELLOW}{details}{RESET}")
    
    def run_health_check(self):
        """Run comprehensive health check"""
        print(f"\n{BLUE}{'*' * 70}")
        print(f"{'🔍 WORLD ENGINE - SYSTEM HEALTH CHECK':^70}")
        print(f"{'*' * 70}{RESET}\n")
        
        # 1. Configuration Files Check
        self.print_header("📋 CONFIGURATION FILES")
        config_files = {
            "port_config.json": "Port configuration",
            "repo_compile_check.py": "Compile checker",
            "smoke_test_websocket.py": "WebSocket tester",
            "consciousness_websocket_server.py": "Consciousness server",
            "vector_node_network.py": "Vector network",
            "analyze_consciousness_patterns.py": "Pattern analyzer"
        }
        
        for file, desc in config_files.items():
            exists = self.check_file_exists(file)
            self.print_status(f"{desc} ({file})", exists)
            self.results[f"file_{file}"] = exists
        
        # 2. Port Configuration
        self.print_header("🔌 PORT CONFIGURATION")
        port_config = self.load_json_config("port_config.json")
        
        if "error" not in port_config:
            ws_port = port_config.get("ws_port", "N/A")
            print(f"  WebSocket Port: {GREEN}{ws_port}{RESET}")
            self.results["ws_port"] = ws_port
        else:
            print(f"  {RED}Error loading config: {port_config['error']}{RESET}")
            self.results["ws_port_error"] = port_config["error"]
        
        # 3. Python Dependencies
        self.print_header("📦 PYTHON DEPENDENCIES")
        packages = self.check_python_imports()
        for package, installed in packages.items():
            self.print_status(f"{package}", installed)
            self.results[f"package_{package}"] = installed
        
        # 4. Service Status
        self.print_header("🌐 SERVICE STATUS")
        services = {
            8701: "WebSocket Server (consciousness/vector)",
            8080: "HTTP Server",
            3000: "Pain Service",
        }
        
        for port, name in services.items():
            is_active = self.check_port(port, name)
            self.print_status(f"{name} (port {port})", is_active)
            self.results[f"service_port_{port}"] = is_active
        
        # 5. WebSocket Connection Test
        self.print_header("🔗 WEBSOCKET CONNECTION TEST")
        if self.check_port(8701, "WebSocket"):
            try:
                loop = asyncio.get_event_loop()
                ws_result = loop.run_until_complete(self.test_websocket_connection(8701))
                
                if ws_result.get("connected"):
                    self.print_status("WebSocket connectivity", True, "Connection successful")
                    print(f"  └─ Response: {ws_result.get('response', 'N/A')[:80]}...")
                else:
                    self.print_status("WebSocket connectivity", False, ws_result.get("error", "Unknown error"))
                
                self.results["websocket_test"] = ws_result
            except Exception as e:
                self.print_status("WebSocket connectivity", False, str(e))
                self.results["websocket_test"] = {"error": str(e)}
        else:
            self.print_status("WebSocket connectivity", False, "No server running on port 8701")
            self.results["websocket_test"] = {"error": "No server running"}
        
        # 6. AI System Status
        self.print_header("🤖 AI SYSTEM STATUS")
        
        # Check for running Python processes
        try:
            import psutil
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and len(cmdline) > 1:
                        script = cmdline[-1]
                        if any(name in script for name in ['consciousness', 'vector', 'nexus', 'brain', 'intelligence']):
                            python_processes.append({
                                'pid': proc.info['pid'],
                                'script': Path(script).name if script else 'unknown'
                            })
            
            if python_processes:
                print(f"{GREEN}✓{RESET} Active AI processes detected:")
                for proc in python_processes:
                    print(f"  └─ PID {proc['pid']}: {proc['script']}")
                self.results["ai_processes"] = python_processes
            else:
                print(f"{YELLOW}⚠{RESET} No AI processes detected (servers may be stopped)")
                self.results["ai_processes"] = []
        except ImportError:
            print(f"{YELLOW}⚠{RESET} psutil not installed - cannot check running processes")
            self.results["ai_processes"] = "psutil_not_available"
        
        # 7. Summary
        self.print_header("📊 SYSTEM SUMMARY")
        
        total_checks = len([k for k in self.results.keys() if not k.endswith('_error')])
        passed_checks = len([v for k, v in self.results.items() if v is True])
        
        health_percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        print(f"  Total Checks: {total_checks}")
        print(f"  Passed: {GREEN}{passed_checks}{RESET}")
        print(f"  Failed: {RED}{total_checks - passed_checks}{RESET}")
        print(f"  Health Score: {GREEN if health_percentage >= 80 else YELLOW if health_percentage >= 50 else RED}{health_percentage:.1f}%{RESET}")
        
        # Status determination
        print(f"\n{BLUE}{'─' * 70}{RESET}")
        if health_percentage >= 80:
            print(f"\n{GREEN}✅ SYSTEM STATUS: OPERATIONAL{RESET}")
            print(f"{GREEN}All critical systems are functioning normally.{RESET}\n")
        elif health_percentage >= 50:
            print(f"\n{YELLOW}⚠️  SYSTEM STATUS: DEGRADED{RESET}")
            print(f"{YELLOW}Some systems are offline. Review failures above.{RESET}\n")
        else:
            print(f"\n{RED}❌ SYSTEM STATUS: CRITICAL{RESET}")
            print(f"{RED}Multiple system failures detected. Immediate attention required.{RESET}\n")
        
        # Save results
        results_file = self.root_dir / "system_health_report.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved to: {results_file}\n")
        
        return self.results

def main():
    """Main entry point"""
    checker = SystemHealthChecker()
    results = checker.run_health_check()
    
    # Exit with appropriate code
    total_checks = len([k for k in results.keys() if not k.endswith('_error')])
    passed_checks = len([v for k, v in results.items() if v is True])
    health_percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    sys.exit(0 if health_percentage >= 80 else 1)

if __name__ == "__main__":
    main()
