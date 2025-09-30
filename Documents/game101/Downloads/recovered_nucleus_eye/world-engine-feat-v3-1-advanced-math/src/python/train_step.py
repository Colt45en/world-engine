"""
Backwards compatibility shim for Tier-4 training APIs.

"""

try:
    from .tier4_trainer import *  # type: ignore F401,F403
except ImportError:  # pragma: no cover
    from tier4_trainer import *  # type: ignore F401,F403

__all__ = [
    "Tier4Config",
    "Tier4Trainer",
    "is_dist",
    "rank0",
    "dist_avg",
    "move_batch_to_device",
]
