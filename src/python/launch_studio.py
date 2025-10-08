#!/usr/bin/env python3
"""
World Engine Launcher - Start the integrated studio interface

This launcher helps you access the World Engine Studio with all controls:
- Chat Controller
- Engine Interface
- Recording Studio
"""

import os
import sys
import webbrowser
import time
from pathlib import Path

def main():
    """Launch World Engine Studio"""
    print("🌍 World Engine Studio Launcher")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("web/studio.html").exists():
        print("❌ Error: studio.html not found in web/ directory")
        print("Make sure you're running this from the World Engine root directory")
        sys.exit(1)

    print("📁 Found World Engine web interface")
    print("\nSelect an option:")
    print("1. Open Studio Interface directly (File)")
    print("2. Start local HTTP server")
    print("3. Start full FastAPI server (if available)")

    choice = input("\nChoice (1-3): ").strip()

    if choice == "1":
        # Open studio.html directly
        studio_path = Path("web/studio.html").resolve()
        print(f"🌍 Opening: {studio_path}")
        webbrowser.open(f"file://{studio_path}")

    elif choice == "2":
        # Start simple HTTP server
        print("🚀 Starting HTTP server on port 8080...")
        os.chdir("web")
        os.system("python -m http.server 8080")

    elif choice == "3":
        # Try to start FastAPI server
        try:
            print("🚀 Starting FastAPI server...")
            from api.service import create_app
            import uvicorn

            app = create_app()

            # Open browser after delay
            def open_browser():
                time.sleep(2)
                webbrowser.open("http://localhost:8000/web/studio.html")

            import threading
            threading.Thread(target=open_browser, daemon=True).start()

            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

        except ImportError:
            print("❌ FastAPI not available. Install with:")
            print("   pip install fastapi uvicorn")
            print("   pip install -r requirements.txt")
        except Exception as e:
            print(f"❌ Error starting server: {e}")

    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
