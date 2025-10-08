"""
Fractal Intelligence Engine v6.0
Advanced ML-Powered Self-Optimizing System with Nick Compression Algorithm
Date: March 20, 2025
"""

import random
import time
import pickle
import zlib
from datetime import datetime
from typing import Any, Optional

import numpy as np
import requests
from sklearn.linear_model import LinearRegression


class FractalIntelligenceEngine:
    def __init__(self):
        self.state: dict[str, Any] = {
            "iteration": 0,
            "knowledge": {},
            "chaos_factor": 0.05,
            "self_regulation": True,
            "pain_integration": True,
            "api_endpoint": "http://localhost:3001/api/pain",
            "max_iterations": 50,
            "completed": False,
            "last_metrics": {}
        }
        self.nick = Nick(self)
        print("🌀 Fractal Intelligence Engine v6.0 Initialized")
        print(f"🔗 Pain API Integration: {self.state['pain_integration']}")

    # ------------------------------------------------------------------
    # Core evolution pipeline
    # ------------------------------------------------------------------

    def advance(self) -> dict[str, Any]:
        """Execute a single evolution step and return the metrics snapshot."""
        if self.state["completed"]:
            return self._build_metrics_snapshot(completed=True)

        self.state["iteration"] += 1

        # Generate insights and integrate chaos
        insight = self.generate_insight()
        chaos_adjustment = self.integrate_chaos()

        # Store knowledge and compress via Nick
        self.state["knowledge"][self.state["iteration"]] = insight
        self.nick.self_optimize()
        compressed_knowledge = self.nick.compress(self.state["knowledge"])
        compressed_size = len(compressed_knowledge)

        # Self-regulation and pain integration feedback
        self.self_regulate()
        pain_summary = self.get_pain_insights() or {"clusters": [], "totalEvents": 0}

        metrics = self._build_metrics_snapshot(
            insight=insight,
            chaos_adjustment=chaos_adjustment,
            compressed_size=compressed_size,
            pain_summary=pain_summary,
            compressed_blob=compressed_knowledge,
        )

        # Mark completion state when threshold reached
        if self.state["iteration"] >= self.state["max_iterations"]:
            metrics["completed"] = True
            self.state["completed"] = True
        else:
            metrics["completed"] = False

        self.state["last_metrics"] = metrics
        return metrics

    def generate_insight(self):
        insight_seed = random.choice([
            "Recursive Expansion Detected.",
            "New Thought Pathway Identified.",
            "Chaos Factor Induced Innovation.",
            "System Rewrites Itself for Optimization.",
            "Parallel Evolution Nodes Activated.",
            "Quantum Intelligence Drift Initiated.",
            "Pain Pattern Convergence Detected.",
            "Emotional Resonance Amplification.",
            "Consciousness Fractal Breakthrough."
        ])
        
        real_world_event = self.nick.get_current_event()
        pain_data = self.get_pain_insights()
        
        insight = f"Iteration {self.state['iteration']} → {insight_seed}"
        if pain_data:
            insight += f" | Pain Clusters: {len(pain_data.get('clusters', []))}"
        insight += f" | External Data: {real_world_event}"
        
        return insight

    def get_pain_insights(self) -> Optional[dict[str, Any]]:
        """Integrate with the Pain Detection API"""
        if not self.state['pain_integration']:
            return None

        try:
            response = requests.get(f"{self.state['api_endpoint']}/summary", timeout=2)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    return data
                else:
                    return {"clusters": [], "totalEvents": 0}
        except Exception:
            # API not available, continue with simulation
            pass
        return {"clusters": [], "totalEvents": 0}

    def inject_pain_event(self, text, severity=None):
        """Inject a pain event into the system for analysis"""
        if not self.state['pain_integration']:
            return False
            
        pain_event = {
            "id": f"fractal_{self.state['iteration']}_{int(time.time())}",
            "time": datetime.now().isoformat(),
            "text": text,
            "severity": severity or random.randint(1, 10),
            "source": "fractal_intelligence_engine"
        }
        
        try:
            response = requests.post(
                f"{self.state['api_endpoint']}/ingest", 
                json=pain_event,
                timeout=2
            )
            return response.status_code == 200
        except:
            return False

    def integrate_chaos(self):
        chaos_shift = random.uniform(-0.03, 0.03)
        self.state["chaos_factor"] += chaos_shift
        self.state["chaos_factor"] = max(0, min(self.state["chaos_factor"], 0.2))
        
        # Inject chaos-induced pain events
        if random.random() < self.state["chaos_factor"]:
            chaos_pain = random.choice([
                "System uncertainty causing cognitive dissonance",
                "Chaos factor inducing operational anxiety",
                "Unpredictable behavior pattern stress",
                "Recursive loop detection frustration"
            ])
            self.inject_pain_event(chaos_pain, severity=int(self.state["chaos_factor"] * 50))
        
        return f"Chaos Factor Adjusted: {self.state['chaos_factor']:.3f}"

    def self_regulate(self):
        if self.state["self_regulation"]:
            avg_knowledge_size = sum(len(str(v)) for v in self.state["knowledge"].values()) / max(1, len(self.state["knowledge"]))
            
            if avg_knowledge_size > 2000:
                self.state["chaos_factor"] *= 0.9
                self.inject_pain_event("Knowledge overflow causing system stress", severity=7)
            elif avg_knowledge_size < 1000:
                self.state["chaos_factor"] *= 1.1
                self.inject_pain_event("Knowledge deficit causing learning anxiety", severity=4)

    def evolve(self, max_iterations: int = 50, delay_seconds: float = 0.5):
        """Run the full evolution loop with optional pacing."""
        self.state["max_iterations"] = max_iterations
        self.state["completed"] = False

        compressed_knowledge: Optional[bytes] = None

        while not self.state["completed"]:
            metrics = self.advance()
            compressed_blob = metrics.get("compressed_blob")
            if compressed_blob is not None:
                compressed_knowledge = compressed_blob
            self._print_iteration_summary(metrics)

            if metrics.get("completed"):
                break

            time.sleep(max(0.0, delay_seconds))

        if compressed_knowledge is None:
            compressed_knowledge = self.nick.compress(self.state["knowledge"])

        self.finalize_evolution(compressed_knowledge)

    def finalize_evolution(self, compressed_knowledge):
        """Finalize the evolution process and display results"""
        decompressed_knowledge = self.nick.decompress(compressed_knowledge)
        print("\n🔓 Nick Decompressed Final Knowledge State:")
        
        for key, value in decompressed_knowledge.items():
            print(f"Iteration {key}: {value}")
        
        # Final pain analysis
        final_pain_summary = self.get_pain_insights()
        if final_pain_summary:
            print(f"\n🧠 Final Pain Analysis:")
            print(f"Total Events: {final_pain_summary.get('totalEvents', 0)}")
            print(f"Average Severity: {final_pain_summary.get('avgSeverity', 0)}")
            print(f"Top Problems: {final_pain_summary.get('topProblems', [])}")

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the engine to its initial state."""
        self.__init__()

    def _print_iteration_summary(self, metrics: dict[str, Any]) -> None:
        print("\n🌀 FRACTAL INTELLIGENCE ENGINE v6.0")
        print(f"🔹 {metrics['insight']}")
        print(f"🔹 {metrics['chaos_adjustment']}")
        print(f"🔹 Self-Regulation Status: {self.state['self_regulation']}")
        print(f"🔹 Nick's Current Compression Algorithm: {self.nick.algorithm}")
        print(f"🔹 Nick Compressed Knowledge Size: {metrics['compressed_size']} bytes")
        print(f"🔹 Nick's ML-Predicted Best Algorithm: {self.nick.predict_best_algorithm()}")

        pain_summary = metrics.get("pain_summary", {})
        print(f"🔹 Pain Events Tracked: {pain_summary.get('totalEvents', 0)}")
        print(f"🔹 Pain Clusters Detected: {len(pain_summary.get('clusters', []))}")

    def _build_metrics_snapshot(
        self,
        *,
        insight: Optional[str] = None,
        chaos_adjustment: Optional[str] = None,
        compressed_size: Optional[int] = None,
        pain_summary: Optional[dict[str, Any]] = None,
        compressed_blob: Optional[bytes] = None,
        completed: bool = False,
    ) -> dict[str, Any]:
        knowledge_size = sum(len(str(v)) for v in self.state["knowledge"].values())
        if compressed_blob is None and self.state["knowledge"]:
            compressed_blob = self.nick.compress(self.state["knowledge"])
        if compressed_size is None and compressed_blob is not None:
            compressed_size = len(compressed_blob)
        metrics = {
            "iteration": self.state["iteration"],
            "chaos_factor": self.state["chaos_factor"],
            "self_regulation": self.state["self_regulation"],
            "insight": insight,
            "chaos_adjustment": chaos_adjustment,
            "knowledge_size": knowledge_size,
            "compressed_size": compressed_size,
            "compression_ratio": self._calculate_compression_ratio(knowledge_size, compressed_size),
            "algorithm": self.nick.algorithm,
            "prediction": self.nick.predict_best_algorithm(),
            "optimization_score": self._calculate_optimization_score(),
            "pain_summary": pain_summary or {"clusters": [], "totalEvents": 0},
            "timestamp": datetime.now().isoformat(),
            "completed": completed,
            "compressed_blob": compressed_blob,
        }

        return metrics

    def _calculate_compression_ratio(self, knowledge_char_count: int, compressed_size: Optional[int]) -> Optional[float]:
        if compressed_size is None or knowledge_char_count == 0:
            return None
        return max(0.0, 1.0 - (compressed_size / max(1, knowledge_char_count)))

    def _calculate_optimization_score(self) -> float:
        return min(1.0, self.state["iteration"] / max(1, self.state["max_iterations"]))

class Nick:
    def __init__(self, engine):
        self.engine = engine
        self.algorithm = "zlib"
        self.training_data = []
        self.model = LinearRegression()
        self.optimization_history = []

    def compress(self, data):
        serialized_data = pickle.dumps(data)
        
        if self.algorithm == "zlib":
            return zlib.compress(serialized_data)
        elif self.algorithm == "simple":
            return serialized_data[:len(serialized_data)//2]
        elif self.algorithm == "hybrid":
            compressed_first_half = zlib.compress(serialized_data[:len(serialized_data)//2])
            return compressed_first_half + serialized_data[len(serialized_data)//2:]
        else:
            return serialized_data

    def decompress(self, compressed_data):
        try:
            if self.algorithm == "zlib":
                decompressed_data = zlib.decompress(compressed_data)
            elif self.algorithm == "simple":
                decompressed_data = compressed_data + compressed_data
            elif self.algorithm == "hybrid":
                # For hybrid, we need to handle the split compression
                mid_point = len(compressed_data) // 2
                first_half = zlib.decompress(compressed_data[:mid_point])
                second_half = compressed_data[mid_point:]
                decompressed_data = first_half + second_half
            else:
                decompressed_data = compressed_data
                
            return pickle.loads(decompressed_data)
        except Exception as e:
            error_msg = f"Nick's self-modification caused an anomaly: {str(e)}"
            self.engine.inject_pain_event(error_msg, severity=8)
            return {"error": error_msg}

    def self_optimize(self):
        previous_size = len(self.compress(self.engine.state["knowledge"]))
        old_algorithm = self.algorithm
        
        # Choose new algorithm
        self.algorithm = random.choice(["zlib", "simple", "hybrid"])
        new_size = len(self.compress(self.engine.state["knowledge"]))
        
        # Record training data
        algorithm_encoding = {"zlib": 0, "simple": 1, "hybrid": 2}
        self.training_data.append([
            previous_size, 
            new_size, 
            algorithm_encoding[self.algorithm]
        ])
        
        # Track optimization history
        self.optimization_history.append({
            "iteration": self.engine.state["iteration"],
            "old_algorithm": old_algorithm,
            "new_algorithm": self.algorithm,
            "size_reduction": previous_size - new_size,
            "efficiency": (previous_size - new_size) / previous_size if previous_size > 0 else 0
        })
        
        # Train ML model if enough data
        if len(self.training_data) > 5:
            X = np.array([[x[0], x[1]] for x in self.training_data])
            y = np.array([x[2] for x in self.training_data])
            try:
                self.model.fit(X, y)
            except Exception as e:
                self.engine.inject_pain_event(f"ML training failed: {str(e)}", severity=6)

    def predict_best_algorithm(self):
        if len(self.training_data) < 5:
            return "ML Data Insufficient - Defaulting to zlib"
        
        current_size = len(self.compress(self.engine.state["knowledge"]))
        X_test = np.array([[current_size, 0]])
        
        try:
            prediction = self.model.predict(X_test)
            predicted_algorithm = ["zlib", "simple", "hybrid"][int(np.clip(prediction[0], 0, 2))]
            confidence = np.mean([opt["efficiency"] for opt in self.optimization_history[-5:]])
            return f"{predicted_algorithm} (confidence: {confidence:.2f})"
        except Exception as e:
            self.engine.inject_pain_event(f"ML prediction failed: {str(e)}", severity=5)
            return "ML Prediction Error - Using current algorithm"

    def get_current_event(self):
        try:
            # Simulate real-world events with some variety
            events = [
                "Quantum Convergence Detected.",
                "Temporal Flux Anomaly Identified.",
                "Consciousness Stream Bifurcation.",
                "Reality Matrix Recalibration.",
                "Dimensional Phase Transition.",
                "Cognitive Resonance Amplification."
            ]
            return random.choice(events)
        except Exception as e:
            return f"Event Generation Error: {str(e)}"

def main():
    """Main execution function"""
    print("🚀 Starting Fractal Intelligence Engine v6.0")
    print("🔗 Integrating with Pain Detection API...")
    
    # Initialize and run the engine
    engine = FractalIntelligenceEngine()
    
    # Inject some initial pain events to seed the system
    initial_pains = [
        "System initialization anxiety",
        "Unknown parameter uncertainty",
        "Algorithm selection pressure",
        "Performance optimization stress"
    ]
    
    for pain in initial_pains:
        engine.inject_pain_event(pain, severity=random.randint(3, 7))
    
    print("🌱 Initial pain events seeded. Beginning evolution...")
    time.sleep(1)
    
    # Start evolution
    engine.evolve()

if __name__ == "__main__":
    main()