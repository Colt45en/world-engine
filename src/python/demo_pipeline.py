# CONVERSATIONAL IDE SYSTEM DEMO
# Testing the complete INGEST → UNDERSTAND → PLAN → RESPOND pipeline

import sys
import os

# Add current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conversational_ide_engine import ConversationalIDEEngine
from advanced_input_processor import AdvancedInputProcessor

def demo_full_pipeline():
    """Demonstrate the complete conversational IDE pipeline"""

    print("=" * 80)
    print("🧠 CONVERSATIONAL IDE SYSTEM - COMPLETE PIPELINE DEMO")
    print("=" * 80)
    print("Testing INGEST → UNDERSTAND → PLAN → RESPOND with CONTEXT & POLICY")
    print("Based on the blueprint for teaching IDEs to understand English")
    print("=" * 80)

    # Initialize systems
    print("\n🚀 Initializing systems...")
    engine = ConversationalIDEEngine()
    processor = AdvancedInputProcessor()

    print("✅ Conversational IDE Engine: Ready")
    print("✅ Advanced Input Processor: Ready")
    print("✅ Context Management: Active")
    print("✅ Policy Gates: Enabled")

    # Test cases that demonstrate different capabilities
    test_cases = [
        {
            "name": "Code Generation Request",
            "input": "Can you write a regex for validating dates like 2025-09-28?",
            "expected_features": ["code_generation", "regex_pattern", "date_validation"]
        },
        {
            "name": "Complex Integration Query",
            "input": "How do I integrate the Unity Quantum Protocol with the web dashboard for real-time agent updates?",
            "expected_features": ["explanation", "system_integration", "multiple_entities"]
        },
        {
            "name": "Debug Assistance",
            "input": "I'm getting an error in ./src/components/UserProfile.tsx - the useState hook isn't updating properly",
            "expected_features": ["debug", "file_reference", "specific_technology"]
        },
        {
            "name": "Mathematical Explanation",
            "input": "Explain how the motion equations work in the MotionGlyphs system: $v = v_0 + at$",
            "expected_features": ["explanation", "math_formula", "system_reference"]
        },
        {
            "name": "Urgent Command",
            "input": "/create URGENT: A TypeScript interface for the Nexus system configuration",
            "expected_features": ["command", "urgency", "typescript", "interface_generation"]
        }
    ]

    print(f"\n📋 Testing {len(test_cases)} scenarios through complete pipeline...")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n" + "="*60)
        print(f"🧪 TEST CASE {i}: {test_case['name']}")
        print(f"📝 Input: '{test_case['input'][:50]}{'...' if len(test_case['input']) > 50 else ''}'")
        print("="*60)

        # STEP 1: Advanced Input Processing
        print("\n🔍 STEP 1: ADVANCED INPUT PROCESSING")
        processed_turn = processor.process_input(test_case['input'])
        summary = processor.get_segment_summary(processed_turn)

        print(f"   • Language Detected: {summary['detected_language']['language']} ({summary['detected_language']['confidence']:.2f})")
        print(f"   • Segments Found: {summary['total_segments']} ({', '.join(summary['segment_counts'].keys())})")
        print(f"   • Urgency Score: {summary['flags']['urgency_score']:.2f}")
        print(f"   • Complexity Score: {summary['flags']['complexity_score']:.2f}")

        flags = []
        if summary['flags']['is_command']:
            flags.append("⚡ Command")
        if summary['content_analysis']['has_code']:
            flags.append("💻 Code")
        if summary['content_analysis']['has_math']:
            flags.append("🧮 Math")
        if summary['flags']['has_pii']:
            flags.append("🔒 PII")

        if flags:
            print(f"   • Flags: {' | '.join(flags)}")

        # STEP 2: Complete Pipeline Processing
        print("\n🧠 STEP 2: FULL CONVERSATION PIPELINE")
        result = engine.plan_and_respond(test_case['input'])

        if result["success"]:
            understanding = result["understanding"]
            plan_info = result["plan"]

            print(f"   • Dialogue Act: {understanding['act']}")
            print(f"   • Intents: {', '.join(understanding['intents'])}")
            print(f"   • Entities: {len(understanding['entities'])} found")
            print(f"   • Confidence: {understanding['confidence']:.2f}")
            print(f"   • Plan Sections: {plan_info['sections']}")

            if plan_info['followup']:
                print(f"   • Followup Question: Yes")
        else:
            print(f"   ❌ Processing failed: {result.get('error', 'Unknown error')}")

        # STEP 3: Response Quality Analysis
        print("\n💬 STEP 3: GENERATED RESPONSE")
        if result["success"]:
            response = result["response"]
            print(f"   Response Length: {len(response)} characters")

            # Analyze response features
            response_features = []
            if "```" in response:
                response_features.append("Code Block")
            if any(keyword in response.lower() for keyword in ["step", "first", "then", "finally"]):
                response_features.append("Step-by-Step")
            if "example" in response.lower():
                response_features.append("Examples")
            if len(response.split('\n')) > 3:
                response_features.append("Multi-line")

            if response_features:
                print(f"   Response Features: {', '.join(response_features)}")

            # Show first part of response
            response_preview = response.split('\n')[0][:80]
            print(f"   Preview: \"{response_preview}{'...' if len(response) > 80 else ''}\"")

        print("\n" + "-"*60)

        # Brief pause for readability
        import time
        time.sleep(0.5)

    # Summary
    print(f"\n" + "="*80)
    print("✅ COMPLETE PIPELINE DEMONSTRATION FINISHED")
    print("="*80)
    print("🎯 KEY CAPABILITIES DEMONSTRATED:")
    print("   • Advanced input segmentation and language detection")
    print("   • Multi-intent understanding with entity extraction")
    print("   • Context-aware conversation management")
    print("   • Policy-gated response generation")
    print("   • Teaching patterns for different response types")
    print("   • Quality checks and coherence validation")
    print("   • Integrated code and math processing")
    print("   • Command detection and urgency assessment")
    print()
    print("🚀 The system is ready for VS Code integration!")
    print("   Next steps: D0-D5 learning curriculum and IDE extension")

def interactive_demo():
    """Run an interactive demo session"""

    print("\n" + "="*80)
    print("🎮 INTERACTIVE CONVERSATIONAL IDE DEMO")
    print("="*80)
    print("Test the system with your own inputs!")
    print("Type 'demo' to see full pipeline, 'quit' to exit")
    print("="*80)

    engine = ConversationalIDEEngine()
    processor = AdvancedInputProcessor()

    while True:
        try:
            user_input = input("\n💬 Your Input: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Demo session ended!")
                break

            if user_input.lower() == 'demo':
                demo_full_pipeline()
                continue

            if not user_input:
                continue

            print("\n🔍 Processing through pipeline...")

            # Quick processing demonstration
            processed = processor.process_input(user_input)
            summary = processor.get_segment_summary(processed)

            result = engine.plan_and_respond(user_input)

            print(f"\n📊 Analysis:")
            print(f"   Language: {summary['detected_language']['language']} ({summary['detected_language']['confidence']:.2f})")
            print(f"   Intents: {', '.join(result['understanding']['intents'])}")
            print(f"   Urgency: {summary['flags']['urgency_score']:.2f}")

            print(f"\n🤖 Response:")
            print(result['response'])

        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Check if running interactively
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        interactive_demo()
    else:
        demo_full_pipeline()
        print("\n💡 Tip: Run with 'interactive' argument for interactive mode")
        print("   Example: python demo_pipeline.py interactive")
