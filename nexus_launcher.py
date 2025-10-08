#!/usr/bin/env python3
"""
NEXUS CODEPAD LAUNCHER
=====================

Quick launcher for the Master Nexus CODEPAD system.
"""

import subprocess
import sys
import time
import socket
import webbrowser
from pathlib import Path

# Ensure UTF-8 output on Windows consoles so emoji printing doesn't raise
# UnicodeEncodeError (cp1252 can't encode many emoji characters).
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    # Fall back silently if reconfigure isn't available
    pass


def find_available_port(start_port: int = 8766) -> int | None:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    return None


def main() -> None:
    print("NEXUS CODEPAD LAUNCHER")
    print("=====================")

    # Find available port
    port = find_available_port()
    if port is None:
        print("ERROR: Could not find available port")
        return

    print(f"Using port: {port}")

    # Get current directory (absolute)
    current_dir = Path(__file__).parent.resolve()

    print("\nTesting core systems...")

    systems_to_test = [
        ("Vector Network", "vector_node_network.py", ["--analytics-mode"]),
        ("Pain/Opportunity System", "implement_pain_opportunity_system.py", []),
        ("Consciousness Analyzer", "analyze_consciousness_patterns.py", []),
    ]

    working_systems: list[tuple[str, str]] = []

    for name, filename, args in systems_to_test:
        filepath = current_dir / filename
        if filepath.exists():
            try:
                print(f"Testing {name}...")
                cmd = [sys.executable, str(filepath)] + args
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=str(current_dir))

                if result.returncode == 0 or "NetworkVisualizationServer started" in result.stdout:
                    print(f"✅ {name} - WORKING")
                    working_systems.append((name.lower().replace(" ", "_").replace("/", "_"), str(filepath)))
                else:
                    print(f"⚠️ {name} - ISSUES DETECTED")
                    if result.stderr:
                        print(f"   Error: {result.stderr[:200]}...")

            except subprocess.TimeoutExpired:
                # If the process didn't exit within the timeout it may be a long-running server
                print(f"✅ {name} - RUNNING (timeout indicates server started)")
                working_systems.append((name.lower().replace(" ", "_").replace("/", "_"), str(filepath)))
            except Exception as e:
                print(f"❌ {name} - FAILED: {e}")
        else:
            print(f"❌ {name} - FILE NOT FOUND")

    print(f"\n✅ {len(working_systems)} systems are ready")

    # Launch systems directly since the main CODEPAD isn't starting
    print("\nLaunching systems directly...")

    processes: list[tuple[str, subprocess.Popen]] = []

    for name, filepath in working_systems:
        try:
            if "vector_network" in name:
                print(f"🚀 Starting {name} with analytics...")
                process = subprocess.Popen([sys.executable, filepath, "--analytics-mode"], cwd=str(current_dir))
                processes.append((name, process))
                time.sleep(2)

            elif "pain_opportunity" in name:
                print(f"🚀 Starting {name}...")
                process = subprocess.Popen([sys.executable, filepath], cwd=str(current_dir))
                processes.append((name, process))
                time.sleep(2)

            elif "consciousness" in name:
                print(f"🚀 Starting {name}...")
                process = subprocess.Popen([sys.executable, filepath], cwd=str(current_dir))
                processes.append((name, process))
                time.sleep(2)

        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")

    print(f"\n✅ Launched {len(processes)} systems!")

    # Open web interface if present
    interface_file = current_dir / "nexus_codepad_interface.html"
    if interface_file.exists():
        print("\n🌐 Opening web interface...")
        try:
            webbrowser.open(interface_file.as_uri())
        except Exception as e:
            print(f"Could not open web interface in browser: {e}")
    else:
        # Show friendly message when interface file is missing
        new_func()

    print("\n" + "=" * 50)
    print("NEXUS SYSTEMS STATUS")
    print("=" * 50)

    for name, process in processes:
        status = "RUNNING" if process.poll() is None else "STOPPED"
        pid_display = process.pid if process.poll() is None else 'N/A'
        print(f"{name}: {status} (PID: {pid_display})")

    print("\nPress Ctrl+C to stop all systems")

    try:
        while True:
            time.sleep(5)
            running_count = sum(1 for _, p in processes if p.poll() is None)
            print(f"Systems running: {running_count}/{len(processes)}")
    except KeyboardInterrupt:
        print("\n🛑 Stopping all systems...")
        from subprocess import Popen
        for name, process in processes:
            try:
                process: Popen
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Stopped {name}")
            except Exception:
                pid_str = str(int(process.pid)) if getattr(process, 'pid', None) is not None else "N/A"
                print(f"⚠️ Force killing {pid_str}")
                try:
                    process.kill()
                except Exception:
                    pass
        print("All systems stopped. Goodbye!")


def new_func() -> None:
    print("\n⚠️ Web interface file not found")


if __name__ == "__main__":
    main()