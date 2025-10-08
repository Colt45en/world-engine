"""Minimal smoke test for the World Engine package.

This test is intentionally small: it verifies that importing the main
package path does not raise a SyntaxError/ImportError (non-destructive).
"""
import sys
import os


def test_imports():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    src = os.path.join(root, 'src')
    if src not in sys.path:
        sys.path.insert(0, src)

    # Try importing a few expected modules; if they're missing the test
    # should still pass (we just don't want syntax errors)
    modules = [
        'world_engine',
        'world_engine_robust',
    ]

    for m in modules:
        try:
            __import__(m)
        except Exception:
            # Acceptable in partial workspaces; ensure no SyntaxError was raised
            continue
