#!/usr/bin/env python3
"""
Quick demo of the enhanced 4D system with limb integration
"""

import importlib.util
import re

# Import the 4D engine module
spec = importlib.util.spec_from_file_location("engine", "4d_shape_integration_engine.py")
engine_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_module)
Engine = engine_module.FourDimensionalShapeEngine

def demo_limb_commands():
    """Demo limb commands with the 4D system"""
    print("🚀 Enhanced 4D System - Limb Integration Demo")
    print("=" * 50)

    engine = Engine()

    # Test limb spawn commands
    limb_commands = [
        "spawn limbs on front wall",
        "control limbs speed 200",
        "set limb angle 90 degrees",
        "make limbs longer 250px"
    ]

    print("\n🔬 Testing Limb Commands:")
    for i, command in enumerate(limb_commands, 1):
        print(f"\n{i}. Testing: '{command}'")

        try:
            result = engine.process_4d_request(command)
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Type: {result.get('command_type', 'unknown')}")

            if "analysis" in result:
                analysis = result["analysis"]
                print(f"   Commands: {analysis.get('detected_commands', [])}")
                if "limb_parameters" in analysis and analysis["limb_parameters"]:
                    print(f"   Limb Params: {analysis['limb_parameters']}")

            if "javascript" in result:
                js = result["javascript"][:200] + "..." if len(result["javascript"]) > 200 else result["javascript"]
                print(f"   JavaScript: {js}")

        except Exception as e:
            print(f"   Error: {str(e)}")

    print("\n✅ Limb integration demo complete!")
    print("The enhanced 4D system can now understand limb-related commands")
    print("and generate appropriate JavaScript for oscillating limb control.")

def test_parameter_extraction():
    """Test the limb parameter extraction directly"""
    print("\n🔍 Testing Parameter Extraction:")
    print("=" * 40)

    engine = Engine()

    test_strings = [
        "speed 150",
        "angle 45 degrees",
        "length 200px",
        "fast oscillation",
        "wide angle movement"
    ]

    for test_str in test_strings:
        try:
            params = engine._extract_limb_parameters(test_str)
            print(f"'{test_str}' → {params}")
        except Exception as e:
            print(f"'{test_str}' → Error: {str(e)}")

if __name__ == "__main__":
    demo_limb_commands()
    test_parameter_extraction()
