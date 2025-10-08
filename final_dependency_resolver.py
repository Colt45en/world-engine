#!/usr/bin/env python3
"""
Minimal Final Dependency & Port Resolver
Creates a safe port_config.json with a single ws_port and provides utilities
to create a minimal test server for later smoke testing.
"""
import os
import json
import socket
from pathlib import Path


def ensure_port_config(port=8701):
    cfg = {"ws_port": port}
    with open("port_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"Created port_config.json with ws_port={port}")
    return cfg


def find_available_port(start=8701):
    for p in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", p))
                return p
            except OSError:
                continue
    return start


def main():
    cfg_path = Path("port_config.json")
    if not cfg_path.exists():
        port = find_available_port(8701)
        ensure_port_config(port)
    else:
        print("port_config.json already exists")


if __name__ == "__main__":
    main()
    # end of minimal resolver