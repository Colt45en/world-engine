# Advanced Communication Training for Nexus
# Now with phonics and more fun lessons!

import json
import random

# Simulate Nexus learning (replace with real engine)
def teach_nexus(prompt, expected_response):
    print(f"Nexus, listen: {prompt}")
    print(f"Good job! The answer is: {expected_response}")
    # In real: nexus.learn(prompt, expected_response)
    return expected_response

# Lesson 1: Greetings (same as before)
greetings = [
    ("Hello! How are you?", "I'm fine, thank you!"),
    ("Hi there! What's your name?", "My name is Nexus!"),
    ("Good morning! How's the weather?", "It's sunny today!"),
]

# Lesson 2: Questions (same as before)
questions = [
    ("What is your favorite color?", "My favorite color is blue!"),
    ("Do you like to play?", "Yes, I love to play!"),
    ("Where do you live?", "I live in the computer!"),
]

# Lesson 3: Describing Things (same as before)
descriptions = [
    ("Tell me about a cat.", "A cat is soft and furry. It says meow!"),
    ("What is a ball?", "A ball is round and bounces. It's fun to play with!"),
    ("Describe your friend.", "My friend is kind and smart. We play together!"),
]

# Lesson 4: Simple Logic (same as before)
logic = [
    ("If hello is a greeting, then a greeting is what?", "hello"),
    ("If cat is an animal, then an animal is what?", "cat"),
    ("If blue is a color, then a color is what?", "blue"),
]

# New Lesson 5: Phonics Fun!
phonics = [
    ("What sounds in 'cat'?", "[k][a][t]"),
    ("What sounds in 'dog'?", "[d][o][g]"),
    ("What sounds in 'sun'?", "[s][u][n]"),
]

# New Lesson 6: Word Building
word_building = [
    ("Put 're' and 'build' together. What word?", "rebuild"),
    ("Put 'run' and 'ing' together. What word?", "running"),
    ("Put 'un' and 'happy' together. What word?", "unhappy"),
]

# New Lesson 7: Math Logic (simple)
math_logic = [
    ("If 1 + 1 = 2, then 2 is what?", "1 + 1"),
    ("If 2 + 2 = 4, then 4 is what?", "2 + 2"),
    ("If 3 + 3 = 6, then 6 is what?", "3 + 3"),
]

lessons = [
    ("Greetings", greetings),
    ("Questions", questions),
    ("Descriptions", descriptions),
    ("Simple Logic", logic),
    ("Phonics Fun", phonics),
    ("Word Building", word_building),
    ("Math Logic", math_logic),
]

print("Hi Nexus! I'm your teacher. Let's learn more to talk like friends!")
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

with open('nexus_advanced_communication_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Your progress is saved. Good job today!")

# Interactive mode (same as before)
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
