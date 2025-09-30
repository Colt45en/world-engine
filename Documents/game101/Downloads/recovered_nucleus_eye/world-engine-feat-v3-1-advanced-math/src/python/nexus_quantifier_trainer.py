"""
Nexus Training Data: Mathematical Logic - Nested Quantifiers
============================================================

This module processes and structures the nested quantifiers mathematical content
for training the Nexus intelligence system.
"""

import json
import requests
from typing import Dict, List, Any
from datetime import datetime

class NexusQuantifierTrainer:
    """Training class for nested quantifiers mathematical content"""

    def __init__(self, bridge_url: str = "http://localhost:8888"):
        self.bridge_url = bridge_url
        self.training_data = self._prepare_quantifier_training_data()

    def _prepare_quantifier_training_data(self) -> Dict[str, Any]:
        """Structure the mathematical content for Nexus training"""

        return {
            "domain": "mathematical_logic",
            "subdomain": "nested_quantifiers",
            "content_type": "theoretical_knowledge",
            "timestamp": datetime.now().isoformat(),

            "fundamental_concepts": {
                "quantifiers": {
                    "universal": {
                        "symbol": "∀",
                        "meaning": "for all",
                        "description": "The predicate is true for all values of x in the domain"
                    },
                    "existential": {
                        "symbol": "∃",
                        "meaning": "there exists",
                        "description": "The predicate is true for at least one x in the domain"
                    }
                },
                "scope": "The range in the formula that the quantifier engages in",
                "nesting": "Two quantifiers are nested if one is within the scope of the other"
            },

            "core_theorems": [
                {
                    "theorem_id": 1,
                    "statement": "The order of nested existential quantifiers can be changed without changing the meaning of the statement",
                    "formal": "∃x ∃y P(x, y) ≡ ∃y ∃x P(x, y)",
                    "explanation": "Existential quantifiers are commutative when nested",
                    "mathematical_importance": "high"
                },
                {
                    "theorem_id": 2,
                    "statement": "The order of nested universal quantifiers can be changed without changing the meaning of the statement",
                    "formal": "∀x ∀y P(x, y) ≡ ∀y ∀x P(x, y)",
                    "explanation": "Universal quantifiers are commutative when nested",
                    "mathematical_importance": "high"
                },
                {
                    "theorem_id": 3,
                    "statement": "To negate a sequence of nested quantifiers, you change each quantifier in the sequence to the other type and then negate the predicate",
                    "formal": "¬(∀x ∃y P(x, y)) ≡ ∃x ∀y ¬P(x, y)",
                    "explanation": "Negation distributes over quantifiers by flipping their types",
                    "mathematical_importance": "critical"
                }
            ],

            "worked_examples": [
                {
                    "example_id": 1,
                    "statement": "∀x ∃y (x+y=5)",
                    "interpretation": "For every x, there exists a y such that x+y=5",
                    "solution": "Choosing y=5-x will satisfy the equation for any x",
                    "domain": "real_numbers",
                    "truth_value": True,
                    "key_insight": "Universal-existential pattern allows dependent choice"
                },
                {
                    "example_id": 2,
                    "statement": "∀x ∃y (x+y=10)",
                    "interpretation": "For every x, there exists a y such that x+y=10",
                    "solution": "Choosing y=10-x will satisfy the equation for any x",
                    "domain": "real_numbers",
                    "truth_value": True,
                    "key_insight": "Linear equations always have solutions in this pattern"
                },
                {
                    "example_id": 3,
                    "statement": "∃y ∀x (x+y>x)",
                    "interpretation": "There exists a y such that for every x, x+y>x",
                    "solution": "Choosing any y>0 will satisfy this condition",
                    "domain": "real_numbers",
                    "truth_value": True,
                    "key_insight": "Existential-universal pattern requires fixed choice"
                },
                {
                    "example_id": 4,
                    "statement": "∀x ∃y (x⋅y=1)",
                    "interpretation": "For every x (where x≠0), there exists a y such that x⋅y=1",
                    "solution": "Choosing y=1/x will satisfy the equation",
                    "domain": "non_zero_real_numbers",
                    "truth_value": True,
                    "key_insight": "Multiplicative inverse relationship"
                }
            ],

            "practice_problems": [
                {
                    "problem_id": 1,
                    "statement": "∀x ∃y (x²+y²=1)",
                    "analysis": "For any x with |x|≤1, choose y=±√(1-x²)",
                    "complexity": "medium"
                },
                {
                    "problem_id": 2,
                    "statement": "∃y ∀x (x+y≥0)",
                    "analysis": "No such y exists for all real x",
                    "complexity": "medium"
                },
                {
                    "problem_id": 3,
                    "statement": "∀x ∃y (xy=x+y)",
                    "analysis": "For x≠1, choose y=x/(x-1)",
                    "complexity": "high"
                }
            ],

            "logical_patterns": {
                "quantifier_order_sensitivity": {
                    "same_type": "Order doesn't matter (commutative)",
                    "mixed_type": "Order critically affects meaning",
                    "example_difference": "∀x ∃y P(x,y) vs ∃y ∀x P(x,y)"
                },
                "negation_rules": {
                    "demorgan_extension": "Negation flips quantifier types",
                    "double_negation": "¬¬P(x) ≡ P(x)",
                    "distribution": "Negation must be carefully distributed"
                },
                "domain_dependency": {
                    "finite_domains": "All quantifiers reduce to finite operations",
                    "infinite_domains": "Require careful mathematical analysis",
                    "empty_domains": "Universal statements are vacuously true"
                }
            },

            "cognitive_connections": [
                "Links to predicate logic and first-order logic",
                "Foundation for mathematical proof techniques",
                "Essential for understanding mathematical definitions",
                "Connects to set theory and topology",
                "Fundamental to algorithm correctness proofs"
            ],

            "training_objectives": [
                "Recognize quantifier patterns in mathematical statements",
                "Apply commutativity rules correctly",
                "Perform logical negations systematically",
                "Distinguish between different quantifier orderings",
                "Generate examples and counterexamples appropriately"
            ]
        }

    def start_training_session(self) -> bool:
        """Initialize a new training session"""
        try:
            response = requests.post(f"{self.bridge_url}/training/start")
            return response.json().get("success", False)
        except Exception as e:
            print(f"Failed to start training session: {e}")
            return False

    def send_training_data(self, data_chunk: Dict[str, Any]) -> bool:
        """Send a chunk of training data to Nexus"""
        try:
            response = requests.post(
                f"{self.bridge_url}/training/data",
                json=data_chunk
            )
            return response.json().get("success", False)
        except Exception as e:
            print(f"Failed to send training data: {e}")
            return False

    def train_nexus_with_quantifiers(self) -> Dict[str, Any]:
        """Execute the complete training process"""

        print("🧠 Starting Nexus Quantifier Training Session...")

        # Start training session
        if not self.start_training_session():
            return {"success": False, "error": "Failed to start training session"}

        print("✅ Training session started")

        # Send data in chunks for better processing
        chunks = [
            {"type": "concepts", "data": self.training_data["fundamental_concepts"]},
            {"type": "theorems", "data": self.training_data["core_theorems"]},
            {"type": "examples", "data": self.training_data["worked_examples"]},
            {"type": "problems", "data": self.training_data["practice_problems"]},
            {"type": "patterns", "data": self.training_data["logical_patterns"]},
            {"type": "connections", "data": self.training_data["cognitive_connections"]},
            {"type": "objectives", "data": self.training_data["training_objectives"]}
        ]

        successful_chunks = 0
        for i, chunk in enumerate(chunks):
            print(f"📤 Sending chunk {i+1}/{len(chunks)}: {chunk['type']}")
            if self.send_training_data(chunk):
                successful_chunks += 1
                print(f"✅ Chunk {chunk['type']} sent successfully")
            else:
                print(f"❌ Failed to send chunk {chunk['type']}")

        # Get training statistics
        try:
            stats_response = requests.post(f"{self.bridge_url}/training/stop")
            stats = stats_response.json()
        except Exception as e:
            stats = {"error": str(e)}

        return {
            "success": True,
            "chunks_sent": successful_chunks,
            "total_chunks": len(chunks),
            "completion_rate": successful_chunks / len(chunks),
            "training_stats": stats
        }

if __name__ == "__main__":
    trainer = NexusQuantifierTrainer()
    results = trainer.train_nexus_with_quantifiers()

    print("\n🎓 Training Results:")
    print(f"Success: {results['success']}")
    print(f"Chunks sent: {results['chunks_sent']}/{results['total_chunks']}")
    print(f"Completion rate: {results['completion_rate']:.1%}")
    print(f"Training stats: {results.get('training_stats', 'N/A')}")
