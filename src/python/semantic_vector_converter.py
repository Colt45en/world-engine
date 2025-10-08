"""
Semantic Vector Converter - Advanced transformation system for World Engine

This system extends the World Engine with vector-based semantic transformations:
1. State vectors representing [polarity, intensity, granularity, confidence]
2. Button-based operators that transform the semantic space
3. Composition of transformations for complex semantic processing
4. Integration with AI bot learning and lexicon analysis

Each word/command becomes a transformation operator that evolves the semantic state.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime
import copy


@dataclass
class TransformationResult:
    """Result of applying a transformation to the state vector."""
    old_state: np.ndarray
    new_state: np.ndarray
    operator: str
    timestamp: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticVectorConverter:
    """
    Vector-based semantic transformation system.

    Maintains a 4D state vector [polarity, intensity, granularity, confidence]
    and applies affine transformations based on semantic operators (buttons).
    """

    def __init__(self, initial_state: Optional[np.ndarray] = None):
        # Initialize state vector [polarity, intensity, granularity, confidence]
        self.state = initial_state if initial_state is not None else np.array([0.0, 0.5, 0.3, 0.6])
        self.snapshots = []
        self.transformation_history = []
        self.operator_library = self._build_operator_library()

    def _build_operator_library(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Build the library of transformation operators.
        Each operator is defined as (D, R, b) for transformation: s' = D @ R @ s + b

        Where:
        - D: Damping/scaling matrix
        - R: Rotation/mixing matrix
        - b: Bias/offset vector
        """

        # Identity matrices for base transformations
        I = np.eye(4)

        library = {
            # === CORE SEMANTIC OPERATORS ===

            "REBUILD": (
                # Damping: moderate scaling
                np.diag([1.0, 1.2, 1.3, 0.8]),
                # Rotation: slight mixing between intensity and granularity
                np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.9, 0.1, 0.0],
                         [0.0, 0.1, 0.9, 0.0],
                         [0.0, 0.0, 0.0, 1.0]]),
                # Bias: increase intensity and granularity, decrease confidence
                np.array([0.0, 0.2, 0.2, -0.1])
            ),

            "UPDATE": (
                # Moderate damping with confidence boost
                np.diag([1.0, 1.1, 1.0, 1.2]),
                # Minimal rotation
                I + 0.05 * (np.random.random((4, 4)) - 0.5),
                # Small positive adjustments
                np.array([0.0, 0.1, 0.0, 0.15])
            ),

            "REFACTOR": (
                # Strong transformation with granularity focus
                np.diag([0.9, 0.8, 1.4, 0.9]),
                # Rotation matrix that mixes all components
                np.array([[0.8, 0.1, 0.1, 0.0],
                         [0.1, 0.8, 0.1, 0.0],
                         [0.1, 0.1, 0.8, 0.0],
                         [0.0, 0.0, 0.0, 1.0]]),
                # Increase granularity, moderate confidence adjustment
                np.array([0.0, 0.0, 0.3, 0.0])
            ),

            "OPTIMIZE": (
                # Smooth scaling with confidence emphasis
                np.diag([1.0, 0.95, 1.1, 1.3]),
                # Minimal rotation for stability
                I + 0.02 * np.random.random((4, 4)),
                # Conservative bias toward confidence
                np.array([0.0, 0.0, 0.05, 0.2])
            ),

            "DEBUG": (
                # Increase granularity and intensity for detailed analysis
                np.diag([1.0, 1.2, 1.4, 0.9]),
                # Identity rotation (no mixing)
                I,
                # Strong granularity boost
                np.array([0.0, 0.15, 0.25, -0.05])
            ),

            "VALIDATE": (
                # Conservative scaling with confidence focus
                np.diag([1.0, 0.9, 1.0, 1.4]),
                # Stability-focused rotation
                I,
                # Strong confidence boost
                np.array([0.0, 0.0, 0.0, 0.3])
            ),

            "ENHANCE": (
                # Boost all positive aspects
                np.diag([1.1, 1.2, 1.1, 1.1]),
                # Gentle mixing for enhancement
                I + 0.03 * np.ones((4, 4)) - 0.03 * I,
                # Positive bias across the board
                np.array([0.05, 0.1, 0.05, 0.1])
            ),

            "SIMPLIFY": (
                # Reduce complexity, increase clarity
                np.diag([1.0, 0.8, 0.7, 1.2]),
                # Simplification rotation
                np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.9, -0.1, 0.0],
                         [0.0, -0.1, 0.8, 0.0],
                         [0.0, 0.0, 0.0, 1.0]]),
                # Reduce intensity and granularity
                np.array([0.0, -0.1, -0.2, 0.1])
            ),

            "AMPLIFY": (
                # Increase all signal components
                np.diag([1.3, 1.4, 1.2, 1.0]),
                # Amplification mixing
                I + 0.1 * np.ones((4, 4)),
                # Strong positive bias
                np.array([0.1, 0.2, 0.1, 0.0])
            ),

            "STABILIZE": (
                # Damping toward stable state
                np.diag([0.9, 0.9, 0.9, 1.1]),
                # Smoothing rotation
                0.7 * I + 0.3 * np.ones((4, 4)) / 4,
                # Gentle bias toward center
                np.array([0.0, 0.0, 0.0, 0.1])
            ),

            # === CONTROL OPERATORS ===

            "STATUS": (
                # Identity transformation (no change)
                I, I, np.zeros(4)
            ),

            "RESTORE": (
                # Identity (actual restoration handled separately)
                I, I, np.zeros(4)
            ),

            "RESET": (
                # Strong damping toward origin
                np.diag([0.5, 0.5, 0.5, 0.5]),
                I,
                # Bias toward initial state
                np.array([0.0, 0.25, 0.15, 0.3])
            ),

            "PREVENT": (
                # Cap intensity, boost confidence
                np.diag([1.0, 0.7, 1.0, 1.3]),
                I,
                # Preventive bias
                np.array([0.0, -0.1, 0.0, 0.2])
            ),

            # === SEMANTIC POLARITY OPERATORS ===

            "POSITIVE": (
                # Boost positive polarity
                np.diag([1.2, 1.0, 1.0, 1.0]),
                I,
                np.array([0.3, 0.0, 0.0, 0.0])
            ),

            "NEGATIVE": (
                # Boost negative polarity
                np.diag([1.2, 1.0, 1.0, 1.0]),
                I,
                np.array([-0.3, 0.0, 0.0, 0.0])
            ),

            "NEUTRAL": (
                # Drive polarity toward zero
                np.diag([0.5, 1.0, 1.0, 1.0]),
                I,
                np.array([0.0, 0.0, 0.0, 0.0])
            ),

            # === LEARNING OPERATORS ===

            "LEARN": (
                # Increase intensity and confidence over time
                np.diag([1.0, 1.1, 1.0, 1.2]),
                # Learning rotation that mixes experience
                I + 0.05 * np.random.random((4, 4)),
                np.array([0.0, 0.1, 0.0, 0.15])
            ),

            "FORGET": (
                # Reduce intensity and confidence
                np.diag([1.0, 0.8, 1.0, 0.9]),
                I,
                np.array([0.0, -0.1, 0.0, -0.1])
            ),

            "REMEMBER": (
                # Boost confidence and intensity
                np.diag([1.0, 1.15, 1.0, 1.25]),
                I,
                np.array([0.0, 0.1, 0.0, 0.2])
            )
        }

        return library

    def apply(self, button: str, metadata: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """
        Apply a transformation operator to the current state.

        Args:
            button: The operator name (e.g., "REBUILD", "UPDATE")
            metadata: Optional metadata to store with the transformation

        Returns:
            TransformationResult containing old state, new state, and metadata
        """
        old_state = self.state.copy()

        # Handle special control operators
        if button == "STATUS":
            self.snapshots.append(self.state.copy())
            result = TransformationResult(
                old_state=old_state,
                new_state=self.state.copy(),
                operator=button,
                timestamp=datetime.now().isoformat(),
                metadata={"action": "snapshot_saved", "snapshot_count": len(self.snapshots)}
            )
        elif button == "RESTORE" and self.snapshots:
            self.state = self.snapshots[-1].copy()
            result = TransformationResult(
                old_state=old_state,
                new_state=self.state.copy(),
                operator=button,
                timestamp=datetime.now().isoformat(),
                metadata={"action": "restored_from_snapshot", "snapshot_index": len(self.snapshots) - 1}
            )
        elif button in self.operator_library:
            # Apply standard transformation
            D, R, b = self.operator_library[button]
            self.state = D @ R @ self.state + b

            # Clamp values to reasonable bounds
            self.state = np.clip(self.state, -2.0, 2.0)

            result = TransformationResult(
                old_state=old_state,
                new_state=self.state.copy(),
                operator=button,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {}
            )
        else:
            # Unknown operator - no change
            result = TransformationResult(
                old_state=old_state,
                new_state=self.state.copy(),
                operator=button,
                timestamp=datetime.now().isoformat(),
                metadata={"error": f"Unknown operator: {button}"}
            )

        # Store transformation history
        self.transformation_history.append(result)

        # Limit history size
        if len(self.transformation_history) > 1000:
            self.transformation_history = self.transformation_history[-500:]

        return result

    def run(self, sequence: List[str], metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Run a sequence of transformations.

        Args:
            sequence: List of operator names to apply in order
            metadata: Optional metadata to associate with the sequence

        Returns:
            Final state vector after all transformations
        """
        sequence_metadata = metadata or {}
        sequence_metadata.update({
            "sequence_length": len(sequence),
            "sequence": sequence,
            "start_state": self.state.copy().tolist()
        })

        for i, button in enumerate(sequence):
            step_metadata = {
                "sequence_step": i,
                "sequence_total": len(sequence),
                "sequence_id": sequence_metadata.get("id", "unknown")
            }
            self.apply(button, step_metadata)

        return self.state

    def get_state_description(self) -> Dict[str, Any]:
        """Get human-readable description of current state."""
        p, i, g, c = self.state

        return {
            "polarity": {
                "value": float(p),
                "description": "positive" if p > 0.2 else "negative" if p < -0.2 else "neutral"
            },
            "intensity": {
                "value": float(i),
                "description": "high" if i > 0.7 else "low" if i < 0.3 else "moderate"
            },
            "granularity": {
                "value": float(g),
                "description": "detailed" if g > 0.7 else "coarse" if g < 0.3 else "balanced"
            },
            "confidence": {
                "value": float(c),
                "description": "confident" if c > 0.7 else "uncertain" if c < 0.3 else "moderate"
            },
            "vector": self.state.tolist()
        }

    def analyze_transformation(self, result: TransformationResult) -> Dict[str, Any]:
        """Analyze the effect of a transformation."""
        delta = result.new_state - result.old_state
        magnitude = np.linalg.norm(delta)

        return {
            "operator": result.operator,
            "delta_vector": delta.tolist(),
            "magnitude": float(magnitude),
            "primary_change": ["polarity", "intensity", "granularity", "confidence"][np.argmax(np.abs(delta))],
            "direction": "increase" if np.sum(delta) > 0 else "decrease" if np.sum(delta) < 0 else "mixed",
            "timestamp": result.timestamp
        }

    def get_operator_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about operator usage and effects."""
        stats = {}

        for result in self.transformation_history:
            op = result.operator
            if op not in stats:
                stats[op] = {
                    "usage_count": 0,
                    "total_magnitude": 0.0,
                    "average_magnitude": 0.0,
                    "effects": {
                        "polarity": [],
                        "intensity": [],
                        "granularity": [],
                        "confidence": []
                    }
                }

            stats[op]["usage_count"] += 1

            delta = result.new_state - result.old_state
            magnitude = np.linalg.norm(delta)
            stats[op]["total_magnitude"] += magnitude
            stats[op]["average_magnitude"] = stats[op]["total_magnitude"] / stats[op]["usage_count"]

            # Track effects on each dimension
            for i, dim in enumerate(["polarity", "intensity", "granularity", "confidence"]):
                stats[op]["effects"][dim].append(float(delta[i]))

        return stats

    def suggest_next_operators(self, target_state: Optional[np.ndarray] = None,
                             max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Suggest operators that would move the state toward a target.

        Args:
            target_state: Desired state vector (if None, suggests general improvements)
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of (operator_name, effectiveness_score) tuples
        """
        if target_state is None:
            # Default target: balanced state with high confidence
            target_state = np.array([0.0, 0.6, 0.5, 0.8])

        suggestions = []
        current_state = self.state.copy()

        for op_name, (D, R, b) in self.operator_library.items():
            if op_name in ["STATUS", "RESTORE"]:
                continue  # Skip control operators

            # Simulate applying this operator
            predicted_state = D @ R @ current_state + b
            predicted_state = np.clip(predicted_state, -2.0, 2.0)

            # Calculate how much closer this gets us to the target
            current_distance = np.linalg.norm(current_state - target_state)
            predicted_distance = np.linalg.norm(predicted_state - target_state)

            effectiveness = max(0, current_distance - predicted_distance)
            suggestions.append((op_name, effectiveness))

        # Sort by effectiveness and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]

    def export_state(self) -> Dict[str, Any]:
        """Export complete state for serialization."""
        return {
            "current_state": self.state.tolist(),
            "snapshots": [s.tolist() for s in self.snapshots],
            "transformation_count": len(self.transformation_history),
            "available_operators": list(self.operator_library.keys()),
            "state_description": self.get_state_description(),
            "timestamp": datetime.now().isoformat()
        }

    def import_state(self, state_data: Dict[str, Any]):
        """Import state from serialized data."""
        self.state = np.array(state_data["current_state"])
        self.snapshots = [np.array(s) for s in state_data.get("snapshots", [])]
        # Note: transformation_history is not restored to avoid conflicts


# Integration utilities for World Engine
def create_word_engine_converter(world_engine_api=None) -> SemanticVectorConverter:
    """Create a converter specifically tuned for World Engine integration."""

    # Initialize with World Engine-specific state if available
    if world_engine_api and hasattr(world_engine_api, 'get_global_sentiment'):
        try:
            sentiment = world_engine_api.get_global_sentiment()
            initial_state = np.array([sentiment, 0.5, 0.4, 0.7])
        except:
            initial_state = None
    else:
        initial_state = None

    converter = SemanticVectorConverter(initial_state)

    # Add World Engine specific operators
    world_engine_ops = {
        "ANALYZE": (
            np.diag([1.0, 1.3, 1.2, 1.1]),
            np.eye(4),
            np.array([0.0, 0.2, 0.15, 0.1])
        ),

        "SEED": (
            np.diag([1.4, 1.0, 1.0, 1.2]),
            np.eye(4),
            np.array([0.0, 0.0, 0.0, 0.15])
        ),

        "CONSTRAIN": (
            np.diag([1.0, 0.9, 1.1, 1.3]),
            np.eye(4),
            np.array([0.0, 0.0, 0.1, 0.2])
        )
    }

    converter.operator_library.update(world_engine_ops)
    return converter


if __name__ == "__main__":
    # Example usage
    converter = SemanticVectorConverter()

    print("Initial state:", converter.get_state_description())

    # Run a sequence of operations
    sequence = ["REBUILD", "STATUS", "OPTIMIZE", "VALIDATE", "ENHANCE"]
    final_state = converter.run(sequence, {"id": "demo_sequence"})

    print("Final state:", converter.get_state_description())

    # Analyze transformations
    for result in converter.transformation_history[-len(sequence):]:
        analysis = converter.analyze_transformation(result)
        print(f"{analysis['operator']}: {analysis['direction']} (magnitude: {analysis['magnitude']:.3f})")

    # Get suggestions
    suggestions = converter.suggest_next_operators()
    print("Suggested next operators:", [(op, f"{score:.3f}") for op, score in suggestions])
