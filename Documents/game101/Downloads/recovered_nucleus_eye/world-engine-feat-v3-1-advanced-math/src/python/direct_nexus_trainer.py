#!/usr/bin/env python3
"""
Interactive Direct Nexus Training System
========================================
Direct conversational training for Nexus with:
- If-then conditional logic patterns
- Natural conversation flows (greetings, responses)
- Real-time interactive training
- Pattern recognition and response generation

Author: Nexus Training Team
Date: 2024-12-29
"""

import requests
import time
import json
from typing import Dict, List, Any, Optional

class DirectNexusTrainer:
    """Interactive trainer for direct conversation and logic patterns"""

    def __init__(self, bridge_url: str = "http://localhost:8888"):
        self.bridge_url = bridge_url
        self.conversation_patterns = []
        self.conditional_patterns = []
        self.training_session_active = False

    def check_bridge_connection(self) -> bool:
        """Check if Nexus Bridge is available"""
        try:
            response = requests.get(f"{self.bridge_url}/health", timeout=3)
            return response.status_code == 200
        except:
            return False

    def teach_conversation_pattern(self, user_input: str, expected_response: str, context: str = "") -> bool:
        """Teach Nexus a conversation pattern"""
        pattern = {
            "type": "conversation",
            "input": user_input.lower().strip(),
            "response": expected_response,
            "context": context,
            "timestamp": time.time()
        }

        self.conversation_patterns.append(pattern)

        # Try to send to bridge if available
        if self.check_bridge_connection():
            try:
                response = requests.post(
                    f"{self.bridge_url}/training/conversation",
                    json=pattern,
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"✅ Pattern sent to Nexus Bridge")
                    return True
                else:
                    print(f"⚠️ Bridge responded with status {response.status_code}")
            except Exception as e:
                print(f"⚠️ Could not send to bridge: {e}")

        print(f"📝 Pattern stored locally (Bridge offline)")
        return True

    def teach_conditional_logic(self, condition: str, result: str, logic_type: str = "if-then") -> bool:
        """Teach Nexus conditional logic patterns"""
        pattern = {
            "type": "conditional",
            "logic_type": logic_type,
            "condition": condition,
            "result": result,
            "timestamp": time.time()
        }

        self.conditional_patterns.append(pattern)

        # Try to send to bridge if available
        if self.check_bridge_connection():
            try:
                response = requests.post(
                    f"{self.bridge_url}/training/logic",
                    json=pattern,
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"✅ Logic pattern sent to Nexus Bridge")
                    return True
            except Exception as e:
                print(f"⚠️ Could not send logic to bridge: {e}")

        print(f"📝 Logic pattern stored locally (Bridge offline)")
        return True

    def test_nexus_response(self, test_input: str) -> str:
        """Test how Nexus responds to input"""
        if self.check_bridge_connection():
            try:
                response = requests.post(
                    f"{self.bridge_url}/chat/test",
                    json={"message": test_input},
                    timeout=8
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "No response received")
                else:
                    return f"Bridge error: HTTP {response.status_code}"
            except Exception as e:
                return f"Connection error: {e}"
        else:
            # Simulate response based on local patterns
            test_lower = test_input.lower().strip()

            # Check conversation patterns
            for pattern in self.conversation_patterns:
                if pattern["input"] in test_lower or test_lower in pattern["input"]:
                    return f"[Simulated] {pattern['response']}"

            # Check conditional patterns
            for pattern in self.conditional_patterns:
                if pattern["condition"].lower() in test_lower:
                    return f"[Simulated] {pattern['result']}"

            return "[Simulated] I'm learning! Please teach me more patterns."

    def start_interactive_training(self):
        """Start interactive training session"""
        print("\n🎓 INTERACTIVE NEXUS TRAINING SESSION")
        print("=" * 60)
        print("🎯 Goal: Teach Nexus conversation patterns and if-then logic")
        print("💡 Commands:")
        print("   - 'teach: input -> response' (conversation)")
        print("   - 'logic: if X then Y' (conditional logic)")
        print("   - 'test: your message' (test Nexus)")
        print("   - 'quit' (end session)")
        print("=" * 60)

        if self.check_bridge_connection():
            print("✅ Nexus Bridge connected - live training active!")
        else:
            print("⚠️ Nexus Bridge offline - using simulation mode")

        self.training_session_active = True

        while self.training_session_active:
            try:
                user_input = input("\n🎓 Training Command: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    break

                self.process_training_command(user_input)

            except KeyboardInterrupt:
                print("\n\n🛑 Training session interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

        self.end_training_session()

    def process_training_command(self, command: str):
        """Process a training command from user"""
        command = command.strip()

        if command.startswith('teach:'):
            # Conversation pattern: teach: hello -> hey how are you
            pattern = command[6:].strip()
            if '->' in pattern:
                parts = pattern.split('->', 1)
                if len(parts) == 2:
                    input_text = parts[0].strip()
                    response_text = parts[1].strip()

                    print(f"\n📚 Teaching conversation pattern:")
                    print(f"   👤 Input: \"{input_text}\"")
                    print(f"   🤖 Response: \"{response_text}\"")

                    success = self.teach_conversation_pattern(input_text, response_text)
                    if success:
                        print(f"✅ Pattern learned!")

                        # Test immediately
                        print(f"\n🧪 Testing pattern...")
                        test_response = self.test_nexus_response(input_text)
                        print(f"🤖 Nexus responds: \"{test_response}\"")
                    else:
                        print(f"❌ Failed to teach pattern")
                else:
                    print("❌ Invalid format. Use: teach: input -> response")
            else:
                print("❌ Invalid format. Use: teach: input -> response")

        elif command.startswith('logic:'):
            # Conditional logic: logic: if 1 then 2
            logic_text = command[6:].strip()

            if 'if ' in logic_text.lower() and ' then ' in logic_text.lower():
                # Parse if-then structure
                parts = logic_text.lower().split(' then ', 1)
                if len(parts) == 2:
                    condition = parts[0].replace('if ', '').strip()
                    result = parts[1].strip()

                    print(f"\n🧠 Teaching conditional logic:")
                    print(f"   🔍 If: \"{condition}\"")
                    print(f"   ➡️ Then: \"{result}\"")

                    success = self.teach_conditional_logic(condition, result)
                    if success:
                        print(f"✅ Logic pattern learned!")

                        # Test the logic
                        print(f"\n🧪 Testing logic...")
                        test_response = self.test_nexus_response(f"what if {condition}?")
                        print(f"🤖 Nexus responds: \"{test_response}\"")
                    else:
                        print(f"❌ Failed to teach logic")
                else:
                    print("❌ Invalid format. Use: logic: if X then Y")
            else:
                print("❌ Invalid format. Use: logic: if X then Y")

        elif command.startswith('test:'):
            # Test Nexus response: test: hello
            test_input = command[5:].strip()

            print(f"\n🧪 Testing Nexus with: \"{test_input}\"")
            response = self.test_nexus_response(test_input)
            print(f"🤖 Nexus responds: \"{response}\"")

        elif command.startswith('show'):
            # Show learned patterns
            self.show_learned_patterns()

        else:
            print("❌ Unknown command. Use: teach:, logic:, test:, show, or quit")

    def show_learned_patterns(self):
        """Show all learned patterns"""
        print(f"\n📊 LEARNED PATTERNS SUMMARY")
        print("=" * 40)

        print(f"\n💬 Conversation Patterns ({len(self.conversation_patterns)}):")
        for i, pattern in enumerate(self.conversation_patterns[-5:], 1):  # Show last 5
            print(f"   {i}. \"{pattern['input']}\" -> \"{pattern['response']}\"")

        print(f"\n🧠 Logic Patterns ({len(self.conditional_patterns)}):")
        for i, pattern in enumerate(self.conditional_patterns[-5:], 1):  # Show last 5
            print(f"   {i}. If \"{pattern['condition']}\" then \"{pattern['result']}\"")

        if len(self.conversation_patterns) > 5 or len(self.conditional_patterns) > 5:
            print(f"\n   (Showing last 5 of each type)")

    def end_training_session(self):
        """End the training session"""
        print(f"\n🎉 TRAINING SESSION COMPLETE!")
        print("=" * 50)
        print(f"📊 Statistics:")
        print(f"   💬 Conversation patterns taught: {len(self.conversation_patterns)}")
        print(f"   🧠 Logic patterns taught: {len(self.conditional_patterns)}")
        print(f"   📚 Total patterns: {len(self.conversation_patterns) + len(self.conditional_patterns)}")

        if self.conversation_patterns or self.conditional_patterns:
            print(f"\n💾 Saving training data...")
            self.save_training_data()

        print(f"\n✨ Nexus is now enhanced with your direct training! ✨")
        self.training_session_active = False

    def save_training_data(self):
        """Save training data to file"""
        training_data = {
            "conversation_patterns": self.conversation_patterns,
            "conditional_patterns": self.conditional_patterns,
            "session_timestamp": time.time()
        }

        filename = f"nexus_direct_training_{int(time.time())}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(training_data, f, indent=2)
            print(f"✅ Training data saved to {filename}")
        except Exception as e:
            print(f"⚠️ Could not save training data: {e}")

    def load_predefined_examples(self):
        """Load some predefined examples to get started"""
        print(f"\n📚 Loading predefined training examples...")

        # Conversation examples
        conversation_examples = [
            ("hey", "hey how are you?"),
            ("hello", "hello! nice to see you"),
            ("hi", "hi there! how's it going?"),
            ("good morning", "good morning! have a great day"),
            ("how are you", "I'm doing well, thanks for asking!"),
            ("bye", "goodbye! take care"),
            ("thanks", "you're welcome!"),
            ("what's up", "not much, just here to help!")
        ]

        # Logic examples
        logic_examples = [
            ("1", "2"),
            ("input is hello", "output is greeting"),
            ("user is confused", "provide clear explanation"),
            ("question about math", "give step-by-step solution"),
            ("user says goodbye", "respond politely"),
            ("error occurs", "handle gracefully"),
            ("user is frustrated", "show empathy"),
            ("task is complete", "confirm and offer more help")
        ]

        print(f"💬 Adding {len(conversation_examples)} conversation patterns...")
        for inp, resp in conversation_examples:
            self.teach_conversation_pattern(inp, resp, "predefined")

        print(f"🧠 Adding {len(logic_examples)} logic patterns...")
        for cond, result in logic_examples:
            self.teach_conditional_logic(cond, result, "if-then")

        print(f"✅ Predefined examples loaded!")

def main():
    """Main function to start direct training"""
    print("🌟 DIRECT NEXUS TRAINING SYSTEM")
    print("=" * 50)
    print("🎯 Teach Nexus conversation patterns and if-then logic")
    print("💡 Interactive training with real-time feedback")

    trainer = DirectNexusTrainer()

    # Load predefined examples first
    trainer.load_predefined_examples()

    # Start interactive training
    trainer.start_interactive_training()

if __name__ == "__main__":
    main()
