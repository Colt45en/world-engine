# Real Nexus Engine Training
# Connecting directly to the World Engine RAG system

import json
import time
from datetime import datetime
from world_engine_rag import WorldEngineRAG, RAGConfig

# Function to teach Nexus using real engine
def teach_nexus_real(prompt, nexus):
    print(f"Nexus, listen: {prompt}")
    try:
        response = nexus.query(prompt)
        print(f"Nexus understands: {response}")
        return response
    except Exception as e:
        print(f"RAG error: {e}")
        return "I had trouble understanding."

# Load previous training results
def load_previous_training():
    try:
        with open('nexus_advanced_communication_training_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Interactive mode with real engine
def interactive_mode(nexus):
    print("\n--- Real Nexus Interactive Mode ---")
    print("Now talking to the real Nexus RAG engine! Ask questions or type 'exit' to stop.")

    previous_results = load_previous_training()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye! Nexus learned a lot today.")
            break

        # Get response from real Nexus
        nexus_response = teach_nexus_real(user_input, nexus)

        # Save interaction
        interaction = {
            'user_input': user_input,
            'nexus_response': nexus_response,
            'timestamp': datetime.now().isoformat()
        }
        previous_results.append(interaction)

    # Save updated results
    with open('nexus_real_interactions.json', 'w') as f:
        json.dump(previous_results, f, indent=2)
    print("Interactions saved!")

if __name__ == "__main__":
    print("Initializing real Nexus RAG engine...")
    try:
        config = RAGConfig()
        nexus = WorldEngineRAG(config)
        print("Nexus is ready! Let's continue learning together.")

        # Quick review of previous lessons
        print("Reviewing what we've learned...")
        review_questions = [
            "What is a greeting?",
            "How do we build words?",
            "What is logic?"
        ]

        for q in review_questions:
            teach_nexus_real(q, nexus)
            time.sleep(1)  # Brief pause

        # Enter interactive mode
        interactive_mode(nexus)

    except Exception as e:
        print(f"Error initializing Nexus RAG: {e}")
        print("Falling back to simulation mode...")

        # Fallback to simulation
        def teach_nexus(prompt, expected_response):
            print(f"Nexus, listen: {prompt}")
            print(f"Good job! The answer is: {expected_response}")
            return expected_response

        print("Hi Nexus! Let's practice some more.")
        practice = [
            ("Hello Nexus!", "Hello friend!"),
            ("What's fun?", "Learning is fun!"),
            ("Good job!", "Thank you!")
        ]

        for p, e in practice:
            teach_nexus(p, e)

        print("Practice complete!")
