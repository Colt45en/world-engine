
"""Timestamp Utilities Module"""
from datetime import datetime, timezone

def format_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_current_timestamp() -> float:
    return datetime.now(timezone.utc).timestamp()
