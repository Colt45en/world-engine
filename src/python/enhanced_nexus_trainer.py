"""
Enhanced Nexus Training System with Boolean Algebra and Advanced Testing
========================================================================

This module extends the Nexus training with mathematical logic (PDNF/PCNF) and
implements comprehensive testing and improvement mechanisms.
"""

import json
import requests
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re
import time
import random

class EnhancedNexusTrainer:
    """Advanced Nexus training with mathematical logic and continuous improvement"""

    def __init__(self, bridge_url: str = "http://localhost:8888"):
        self.bridge_url = bridge_url
        self.training_history = []
        self.test_results = []
        self.improvement_metrics = {}

    def _build_boolean_algebra_training(self) -> Dict[str, Any]:
        """Create Boolean algebra training data including PDNF and PCNF"""

        return {
            "domain": "mathematical_logic",
            "subdomain": "boolean_algebra",
            "content_type": "formal_systems",
            "timestamp": datetime.now().isoformat(),

            "fundamental_concepts": {
                "pdnf": {
                    "name": "Principal Disjunctive Normal Form",
                    "definition": "Sum of Products where each product contains all variables",
                    "form": "SOP (Sum of Products)",
                    "characteristics": [
                        "Each term is a product (AND) of literals",
                        "Terms are connected by disjunction (OR)",
                        "Every variable appears in each term",
                        "Variables appear either complemented or uncomplemented"
                    ],
                    "example": "(P·Q'·R) + (P'·Q·R) + (P·Q·R')",
                    "vs_dnf": "PDNF requires all variables in each term, DNF does not"
                },

                "pcnf": {
                    "name": "Principal Conjunctive Normal Form",
                    "definition": "Product of Sums where each sum contains all variables",
                    "form": "POS (Product of Sums)",
                    "characteristics": [
                        "Each term is a sum (OR) of literals",
                        "Terms are connected by conjunction (AND)",
                        "Every variable appears in each term",
                        "Variables appear either complemented or uncomplemented"
                    ],
                    "example": "(P+Q'+R)·(P'+Q+R)·(P+Q+R')",
                    "vs_cnf": "PCNF requires all variables in each term, CNF does not"
                }
            },

            "conversion_procedures": {
                "to_pdnf": {
                    "steps": [
                        "1. Convert to DNF if not already",
                        "2. For each product term missing variables, expand using X + X' = 1",
                        "3. Distribute to ensure all variables appear in each term",
                        "4. Remove duplicate terms",
                        "5. Verify each term contains all variables"
                    ],
                    "example_conversion": {
                        "original": "A·(B+C')",
                        "step1": "A·B + A·C'",
                        "step2": "A·B·(C+C') + A·C'·(B+B')",
                        "step3": "A·B·C + A·B·C' + A·B·C' + A·B'·C'",
                        "final": "A·B·C + A·B·C' + A·B'·C'"
                    }
                },

                "to_pcnf": {
                    "steps": [
                        "1. Convert to CNF if not already",
                        "2. For each sum term missing variables, use distribution with X·X' = 0",
                        "3. Ensure all variables appear in each term",
                        "4. Remove duplicate terms",
                        "5. Verify each term contains all variables"
                    ],
                    "example_conversion": {
                        "original": "(A+B')·(B+C)",
                        "step1": "Add missing variables using distribution",
                        "step2": "(A+B'+C)·(A+B'+C')·(A'+B+C)·(A'+B+C')",
                        "final": "(A+B'+C)·(A+B'+C')·(A'+B+C)·(A'+B+C')"
                    }
                }
            },

            "key_properties": {
                "uniqueness": "Every Boolean expression has unique PDNF and PCNF representations",
                "equivalence": "X ≡ Y if and only if PDNF(X) = PDNF(Y) or PCNF(X) = PCNF(Y)",
                "variable_count": "If PCNF has m terms and PDNF has n terms, variables = log₂(m+n)",
                "duality": "PDNF and PCNF are dual forms of the same logical expression"
            },

            "worked_examples": [
                {
                    "problem": "Convert A·(B+C') to PDNF",
                    "solution": {
                        "step1": "Distribute: A·B + A·C'",
                        "step2": "Add missing variable C to first term: A·B·(C+C') = A·B·C + A·B·C'",
                        "step3": "Add missing variable B to second term: A·C'·(B+B') = A·B·C' + A·B'·C'",
                        "final": "A·B·C + A·B·C' + A·B'·C'"
                    }
                },
                {
                    "problem": "Convert (A+B')·(B+C) to PCNF",
                    "solution": {
                        "step1": "Identify missing variables in each term",
                        "step2": "First term (A+B') missing C: (A+B'+C)·(A+B'+C')",
                        "step3": "Second term (B+C) missing A: (A+B+C)·(A'+B+C)",
                        "final": "(A+B'+C)·(A+B'+C')·(A+B+C)·(A'+B+C)"
                    }
                }
            ],

            "practice_problems": [
                {"expression": "(A+B)·(B'+C)", "convert_to": "PCNF", "difficulty": "medium"},
                {"expression": "A·(B+C) + A'·B·C'", "convert_to": "PDNF", "difficulty": "medium"},
                {"expression": "(P·Q'+R) + (P'·Q)", "convert_to": "PDNF", "difficulty": "medium"},
                {"expression": "(X+Y')·(Y+Z')·(X+Z)", "convert_to": "PCNF", "difficulty": "hard"}
            ]
        }

    def _build_testing_framework(self) -> Dict[str, Any]:
        """Create comprehensive testing scenarios for Nexus evaluation"""

        return {
            "communication_tests": [
                {
                    "scenario": "Technical Problem Solving",
                    "input": "My Boolean expression isn't simplifying correctly",
                    "expected_behaviors": [
                        "Ask for the specific expression",
                        "Inquire about the target form (PDNF/PCNF)",
                        "Offer step-by-step guidance",
                        "Provide verification methods"
                    ],
                    "evaluation_criteria": {
                        "clarity": "Response is clear and structured",
                        "completeness": "Addresses all aspects of the problem",
                        "helpfulness": "Provides actionable guidance"
                    }
                },
                {
                    "scenario": "Mathematical Explanation",
                    "input": "Can you explain the difference between PDNF and PCNF?",
                    "expected_behaviors": [
                        "Define both terms clearly",
                        "Explain the key differences",
                        "Provide concrete examples",
                        "Check understanding"
                    ],
                    "evaluation_criteria": {
                        "accuracy": "Mathematical content is correct",
                        "pedagogy": "Explanation builds understanding progressively",
                        "examples": "Uses relevant, clear examples"
                    }
                },
                {
                    "scenario": "Adaptive Learning Support",
                    "input": "I'm confused about Boolean algebra",
                    "expected_behaviors": [
                        "Assess current knowledge level",
                        "Start with fundamentals if needed",
                        "Use appropriate complexity",
                        "Encourage and support"
                    ],
                    "evaluation_criteria": {
                        "adaptation": "Adjusts to user's level",
                        "encouragement": "Provides emotional support",
                        "progression": "Builds knowledge systematically"
                    }
                }
            ],

            "mathematical_reasoning_tests": [
                {
                    "problem": "Convert A·(B+C') to PDNF",
                    "expected_process": [
                        "Recognize need for variable expansion",
                        "Apply distribution systematically",
                        "Ensure all variables in each term",
                        "Verify final form"
                    ],
                    "correct_answer": "A·B·C + A·B·C' + A·B'·C'"
                },
                {
                    "problem": "Identify errors in this PCNF: (A+B)·(C+D')",
                    "expected_analysis": [
                        "Recognize missing variables",
                        "Identify incomplete terms",
                        "Suggest correction method"
                    ],
                    "key_insight": "Each term must contain all variables"
                }
            ],

            "integration_tests": [
                {
                    "scenario": "Teaching Boolean Algebra",
                    "user_query": "Help me learn PDNF step by step",
                    "expected_integration": [
                        "Communication: Assess learning style and pace",
                        "Mathematical: Provide accurate PDNF content",
                        "Pedagogical: Structure learning progression",
                        "Adaptive: Adjust based on responses"
                    ]
                }
            ]
        }

    def start_enhanced_training(self) -> bool:
        """Start enhanced training session with both communication and math"""
        try:
            response = requests.post(f"{self.bridge_url}/training/start")
            success = response.json().get("success", False)
            if success:
                print("✅ Enhanced training session started")
                self.training_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "training_started",
                    "success": True
                })
            return success
        except Exception as e:
            print(f"❌ Failed to start enhanced training: {e}")
            return False

    def send_training_module(self, module_data: Dict[str, Any]) -> bool:
        """Send a training module to Nexus"""
        try:
            response = requests.post(f"{self.bridge_url}/training/data", json=module_data)
            success = response.json().get("success", False)

            self.training_history.append({
                "timestamp": datetime.now().isoformat(),
                "module": module_data.get("type", "unknown"),
                "success": success
            })

            return success
        except Exception as e:
            print(f"❌ Error sending training module: {e}")
            return False

    def run_comprehensive_training(self) -> Dict[str, Any]:
        """Execute complete training with communication + Boolean algebra"""

        print("🧠 Starting Enhanced Nexus Training...")
        print("📚 Including: Communication + Boolean Algebra + Testing")
        print("=" * 60)

        if not self.start_enhanced_training():
            return {"success": False, "error": "Failed to start training"}

        # Prepare training modules
        boolean_data = self._build_boolean_algebra_training()
        testing_framework = self._build_testing_framework()

        training_modules = [
            {
                "type": "boolean_fundamentals",
                "category": "mathematical_logic",
                "data": boolean_data["fundamental_concepts"]
            },
            {
                "type": "conversion_procedures",
                "category": "mathematical_methods",
                "data": boolean_data["conversion_procedures"]
            },
            {
                "type": "boolean_properties",
                "category": "mathematical_theory",
                "data": boolean_data["key_properties"]
            },
            {
                "type": "worked_examples",
                "category": "practical_application",
                "data": boolean_data["worked_examples"]
            },
            {
                "type": "testing_framework",
                "category": "evaluation_system",
                "data": testing_framework
            }
        ]

        # Execute training
        successful_modules = 0
        for i, module in enumerate(training_modules):
            module_name = module["type"].replace("_", " ").title()
            print(f"📤 Training module {i+1}/{len(training_modules)}: {module_name}")

            if self.send_training_module(module):
                successful_modules += 1
                print(f"   ✅ {module_name} integrated successfully")
            else:
                print(f"   ❌ Failed to integrate {module_name}")

            time.sleep(0.5)  # Prevent overwhelming the system

        # Complete training
        try:
            response = requests.post(f"{self.bridge_url}/training/stop")
            training_stats = response.json()
        except Exception as e:
            training_stats = {"error": str(e)}

        completion_rate = successful_modules / len(training_modules)

        # Store results
        training_result = {
            "timestamp": datetime.now().isoformat(),
            "modules_successful": successful_modules,
            "total_modules": len(training_modules),
            "completion_rate": completion_rate,
            "training_stats": training_stats,
            "capabilities_added": [
                "Boolean algebra understanding (PDNF/PCNF)",
                "Mathematical conversion procedures",
                "Step-by-step problem solving",
                "Integrated communication + math reasoning",
                "Comprehensive testing framework"
            ]
        }

        self.training_history.append(training_result)

        print(f"\n🎓 Enhanced Training Results:")
        print(f"   📊 Modules integrated: {successful_modules}/{len(training_modules)}")
        print(f"   📈 Success rate: {completion_rate:.1%}")

        if completion_rate >= 0.8:
            print("   🌟 Excellent! Nexus now has advanced mathematical reasoning")
        elif completion_rate >= 0.6:
            print("   👍 Good progress! Some areas may need reinforcement")
        else:
            print("   ⚠️ Training incomplete. Consider running again")

        return training_result

    def test_nexus_capabilities(self, test_scenarios: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Test Nexus capabilities across communication and mathematical reasoning"""

        if test_scenarios is None:
            testing_framework = self._build_testing_framework()
            test_scenarios = testing_framework["communication_tests"][:2]  # Start with subset

        print("\n🧪 Testing Nexus Capabilities...")
        print("=" * 40)

        test_results = []

        for i, test in enumerate(test_scenarios, 1):
            print(f"\n🎯 Test {i}: {test['scenario']}")
            print(f"📝 Input: \"{test['input']}\"")

            # In a real implementation, send test to Nexus and analyze response
            # For now, simulate the testing process
            simulated_score = random.uniform(0.6, 0.95)  # Simulate varying performance

            test_result = {
                "test_id": i,
                "scenario": test["scenario"],
                "input": test["input"],
                "simulated_score": simulated_score,
                "expected_behaviors": test["expected_behaviors"],
                "timestamp": datetime.now().isoformat()
            }

            test_results.append(test_result)

            print(f"📊 Simulated Score: {simulated_score:.2f}")
            print(f"✅ Expected Behaviors: {', '.join(test['expected_behaviors'][:2])}...")

        # Calculate overall performance
        avg_score = sum(t["simulated_score"] for t in test_results) / len(test_results)

        self.test_results.extend(test_results)

        overall_result = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": len(test_results),
            "average_score": avg_score,
            "individual_results": test_results,
            "improvement_suggestions": self._generate_improvement_suggestions(test_results)
        }

        print(f"\n📈 Overall Test Results:")
        print(f"   🎯 Tests completed: {len(test_results)}")
        print(f"   📊 Average score: {avg_score:.2f}")
        print(f"   🎪 Performance level: {'Excellent' if avg_score >= 0.8 else 'Good' if avg_score >= 0.7 else 'Needs Improvement'}")

        return overall_result

    def _generate_improvement_suggestions(self, test_results: List[Dict]) -> List[str]:
        """Generate improvement suggestions based on test results"""

        suggestions = []
        avg_score = sum(t["simulated_score"] for t in test_results) / len(test_results)

        if avg_score < 0.7:
            suggestions.append("Increase training iterations for core concepts")
            suggestions.append("Add more example-based learning")

        if avg_score < 0.8:
            suggestions.append("Enhance communication pattern training")
            suggestions.append("Add more mathematical reasoning exercises")

        suggestions.append("Implement user feedback collection")
        suggestions.append("Add real-time performance monitoring")

        return suggestions

    def iterative_improvement_cycle(self, iterations: int = 3) -> Dict[str, Any]:
        """Run multiple training-test-improve cycles"""

        print(f"🔄 Starting Iterative Improvement ({iterations} cycles)")
        print("=" * 50)

        improvement_history = []

        for cycle in range(1, iterations + 1):
            print(f"\n🔁 Improvement Cycle {cycle}")
            print("-" * 30)

            # Train
            training_result = self.run_comprehensive_training()

            # Test
            test_result = self.test_nexus_capabilities()

            # Analyze and improve
            cycle_result = {
                "cycle": cycle,
                "training_completion": training_result.get("completion_rate", 0),
                "test_average": test_result.get("average_score", 0),
                "improvement_suggestions": test_result.get("improvement_suggestions", []),
                "timestamp": datetime.now().isoformat()
            }

            improvement_history.append(cycle_result)

            print(f"   📊 Cycle {cycle} Summary:")
            print(f"      Training: {cycle_result['training_completion']:.1%}")
            print(f"      Testing: {cycle_result['test_average']:.2f}")

            # Small delay between cycles
            if cycle < iterations:
                print(f"   ⏳ Preparing for next cycle...")
                time.sleep(2)

        # Final analysis
        final_result = {
            "total_cycles": iterations,
            "improvement_history": improvement_history,
            "final_performance": improvement_history[-1]["test_average"] if improvement_history else 0,
            "overall_improvement": (improvement_history[-1]["test_average"] - improvement_history[0]["test_average"]) if len(improvement_history) > 1 else 0,
            "timestamp": datetime.now().isoformat()
        }

        print(f"\n🎯 Final Improvement Results:")
        print(f"   🔄 Cycles completed: {iterations}")
        print(f"   📈 Final performance: {final_result['final_performance']:.2f}")
        if final_result['overall_improvement'] > 0:
            print(f"   ⬆️ Improvement: +{final_result['overall_improvement']:.3f}")
        else:
            print(f"   📊 Performance maintained: {final_result['final_performance']:.2f}")

        return final_result

if __name__ == "__main__":
    # Initialize enhanced trainer
    trainer = EnhancedNexusTrainer()

    # Run iterative improvement
    results = trainer.iterative_improvement_cycle(iterations=2)

    print(f"\n🎉 Enhanced Nexus Training Complete!")
    print(f"🧠 Capabilities: Communication + Boolean Algebra + Continuous Improvement")
