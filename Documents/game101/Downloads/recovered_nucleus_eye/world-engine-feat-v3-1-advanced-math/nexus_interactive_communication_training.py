# Interactive Communication Training for Nexus
# Now you can talk to Nexus directly! Type prompts and see how it learns.

import json
import random

# Simulate Nexus learning (replace with real engine)
def teach_nexus(prompt, expected_response):
    print(f"Nexus, listen: {prompt}")
    print(f"Good job! The answer is: {expected_response}")
    # In real: nexus.learn(prompt, expected_response)
    return expected_response

# Lesson data (same as before)
greetings = [
    ("Hello! How are you?", "I'm fine, thank you!"),
    ("Hi there! What's your name?", "My name is Nexus!"),
    ("Good morning! How's the weather?", "It's sunny today!"),
]

questions = [
    ("What is your favorite color?", "My favorite color is blue!"),
    ("Do you like to play?", "Yes, I love to play!"),
    ("Where do you live?", "I live in the computer!"),
]

descriptions = [
    ("Tell me about a cat.", "A cat is soft and furry. It says meow!"),
    ("What is a ball?", "A ball is round and bounces. It's fun to play with!"),
    ("Describe your friend.", "My friend is kind and smart. We play together!"),
]

logic = [
    ("If hello is a greeting, then a greeting is what?", "hello"),
    ("If cat is an animal, then an animal is what?", "cat"),
    ("If blue is a color, then a color is what?", "blue"),
]

lessons = [
    ("Greetings", greetings),
    ("Questions", questions),
    ("Descriptions", descriptions),
    ("Simple Logic", logic),
]

print("Hi Nexus! I'm your teacher. Let's learn to talk like friends!")
print("We'll go slow and have fun. Ready? Let's start!\n")

results = []
for lesson_name, examples in lessons:
    print(f"--- Lesson: {lesson_name} ---")
    for prompt, expected in examples:
        response = teach_nexus(prompt, expected)
        results.append({
            'lesson': lesson_name,
            'prompt': prompt,
            'expected': expected,
            'response': response,
            'correct': (response == expected)
        })
        print("Great work, Nexus! You did it!\n")

# Summary
correct = sum(r['correct'] for r in results)
total = len(results)
print(f"Nexus, you got {correct} out of {total} right! You're learning fast!")
print("Keep practicing, and soon you'll talk like a pro!")

with open('nexus_communication_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Your progress is saved. Good job today!")

# Now, let's make it interactive!
print("\n--- Interactive Mode ---")
print("Now you can ask Nexus questions! Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bye Nexus! See you next time!")
        break
    # Simulate Nexus response (simple echo for now)
    nexus_response = f"Nexus says: {user_input} is fun!"
    print(nexus_response)
    # In real: nexus_response = nexus.process(user_input)
    # Then save to results if needed
print("Interactive session ended.")
