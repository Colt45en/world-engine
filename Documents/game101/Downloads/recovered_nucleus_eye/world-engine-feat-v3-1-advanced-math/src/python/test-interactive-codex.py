#!/usr/bin/env python3
"""
Test the interactive codex functionality
"""

from scene_glyph_generator import WorldHistoryEngine, interactive_terminal_codex

# Create a world engine with sample data
engine = WorldHistoryEngine()

# Add some agents
engine.register_agent("Test Agent", "Novice", {"test": True})
engine.agent_advance_stage("Test Agent", "Apprentice", "Completed first trial")

print("🧪 TESTING INTERACTIVE CODEX")
print("Type some test queries:")
print("  - 'convergence'")
print("  - 'void'")
print("  - 'agent Test Agent'")
print("  - 'status'")
print("  - 'exit'")
print()

# Start interactive session
interactive_terminal_codex(engine)
