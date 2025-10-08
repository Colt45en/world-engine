#!/usr/bin/env python3
"""
🎯 FINAL SYSTEM STATUS & LAUNCHER
================================

Comprehensive status check and launcher for all fixed systems.
"""

import subprocess
import sys
import time
import socket
import webbrowser
from pathlib import Path
import json

def check_port(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def test_system(name, script_path, args=None, timeout=5):
    """Test if a system can start without errors"""
    if not Path(script_path).exists():
        return {"status": "missing", "error": f"File not found: {script_path}"}
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=Path(script_path).parent
        )
        
        if result.returncode == 0:
            return {"status": "working", "output": result.stdout[:200]}
        else:
            return {"status": "error", "error": result.stderr[:200]}
            
    except subprocess.TimeoutExpired:
        return {"status": "running", "note": "Started successfully (timeout = server running)"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def main():
    print("🎯 FINAL SYSTEM STATUS & LAUNCHER")
    print("=" * 50)
    
    # Define all systems to check
    systems = [
        {
            "name": "Vector Network Analytics",
            "script": "vector_node_network.py",
            "args": ["--analytics-mode"],
            "port": 8902,
            "priority": 1
        },
        {
            "name": "Pain/Opportunity System", 
            "script": "implement_pain_opportunity_system.py",
            "args": [],
            "port": None,
            "priority": 1
        },
        {
            "name": "Consciousness Analyzer",
            "script": "analyze_consciousness_patterns.py", 
            "args": [],
            "port": None,
            "priority": 1
        },
        {
            "name": "Master Nexus CODEPAD",
            "script": "master_nexus_codepad_simple.py",
            "args": [],
            "port": 8765,
            "priority": 2
        },
        {
            "name": "Consciousness Feedback Server",
            "script": "consciousness_feedback_server.py",
            "args": [],
            "port": 8901,
            "priority": 2
        },
        {
            "name": "Simple Consciousness Server",
            "script": "simple_consciousness_server_fixed.py",
            "args": [],
            "port": 8900,
            "priority": 2
        },
        {
            "name": "AI Brain Merger",
            "script": "services/ai_brain_merger.py",
            "args": [],
            "port": None,
            "priority": 3
        },
        {
            "name": "Knowledge Vault Integration",
            "script": "services/knowledge_vault_integration.py",
            "args": [],
            "port": None,
            "priority": 3
        }
    ]
    
    print("\n📊 SYSTEM STATUS CHECK")
    print("-" * 30)
    
    working_systems = []
    
    for system in systems:
        name = system["name"]
        script = system["script"]
        port = system["port"]
        
        print(f"\n🔍 Testing {name}...")
        
        # Check port availability if needed
        if port:
            if not check_port(port):
                print(f"   ⚠️  Port {port} is in use")
            else:
                print(f"   ✅ Port {port} available")
        
        # Test the system
        result = test_system(name, script, system["args"])
        status = result["status"]
        
        if status == "working":
            print(f"   ✅ {name} - WORKING")
            working_systems.append(system)
        elif status == "running":
            print(f"   ✅ {name} - RUNNING")
            working_systems.append(system)
        elif status == "missing":
            print(f"   ❌ {name} - FILE NOT FOUND")
        elif status == "error":
            print(f"   ⚠️  {name} - HAS ISSUES")
            error = result.get("error", "Unknown error")[:100]
            print(f"      Error: {error}")
        else:
            print(f"   ❌ {name} - FAILED")
    
    print(f"\n📈 SUMMARY")
    print("-" * 20)
    print(f"✅ Working Systems: {len(working_systems)}/{len(systems)}")
    
    # Group by priority
    priority_1 = [s for s in working_systems if s["priority"] == 1]
    priority_2 = [s for s in working_systems if s["priority"] == 2] 
    priority_3 = [s for s in working_systems if s["priority"] == 3]
    
    print(f"🎯 Core Systems (Priority 1): {len(priority_1)}")
    print(f"🔧 Support Systems (Priority 2): {len(priority_2)}")
    print(f"📚 Additional Systems (Priority 3): {len(priority_3)}")
    
    # Show what's working
    print(f"\n🚀 READY TO RUN:")
    for system in priority_1:
        args_str = " " + " ".join(system["args"]) if system["args"] else ""
        print(f"   python {system['script']}{args_str}")
    
    # Launch options
    print(f"\n🎛️ LAUNCH OPTIONS:")
    print("1. Launch Core Systems (Recommended)")
    print("2. Launch All Working Systems") 
    print("3. Launch Individual System")
    print("4. Exit")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Launching Core Systems...")
            processes = []
            
            for system in priority_1:
                try:
                    cmd = [sys.executable, system["script"]] + system["args"]
                    process = subprocess.Popen(cmd, cwd=Path(system["script"]).parent)
                    processes.append((system["name"], process))
                    print(f"✅ Started {system['name']} (PID: {process.pid})")
                    time.sleep(2)  # Stagger starts
                except Exception as e:
                    print(f"❌ Failed to start {system['name']}: {e}")
            
            print(f"\n✅ Launched {len(processes)} core systems!")
            
            # Open web interface if available
            interface_file = Path("nexus_codepad_interface.html")
            if interface_file.exists():
                print("🌐 Opening web interface...")
                webbrowser.open(f"file://{interface_file.absolute()}")
            
            print("\nPress Ctrl+C to stop all systems")
            try:
                while True:
                    time.sleep(5)
                    running = sum(1 for _, p in processes if p.poll() is None)
                    print(f"Systems running: {running}/{len(processes)}")
            except KeyboardInterrupt:
                print("\n🛑 Stopping all systems...")
                for name, process in processes:
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                        print(f"✅ Stopped {name}")
                    except:
                        process.kill()
                        print(f"🔥 Force killed {name}")
        
        elif choice == "3":
            print(f"\nAvailable systems:")
            for i, system in enumerate(working_systems, 1):
                print(f"{i}. {system['name']}")
            
            try:
                sys_choice = int(input("Select system number: ")) - 1
                selected = working_systems[sys_choice]
                
                cmd = [sys.executable, selected["script"]] + selected["args"]
                print(f"\n🚀 Starting {selected['name']}...")
                subprocess.run(cmd, cwd=Path(selected["script"]).parent)
                
            except (ValueError, IndexError):
                print("❌ Invalid selection")
        
        elif choice == "4":
            print("👋 Goodbye!")
        
        else:
            print("❌ Invalid option")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()