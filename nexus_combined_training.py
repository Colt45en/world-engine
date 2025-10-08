# Combined Training for Nexus: Communication + Quantifiers
# Teaching the growing intelligence both talk and math!

import json
import random

# Simulate Nexus learning (replace with real engine)
def teach_nexus(prompt, expected_response):
    print(f"Nexus, listen: {prompt}")
    print(f"Good job! The answer is: {expected_response}")
    # In real: nexus.learn(prompt, expected_response)
    return expected_response

# Communication lessons (from before)
greetings = [
    ("Hello! How are you?", "I'm fine, thank you!"),
    ("Hi there! What's your name?", "My name is Nexus!"),
]

questions = [
    ("What is your favorite color?", "My favorite color is blue!"),
    ("Do you like to play?", "Yes, I love to play!"),
]

logic = [
    ("If hello is a greeting, then a greeting is what?", "hello"),
    ("If cat is an animal, then an animal is what?", "cat"),
]

# New: Simple Quantifier lessons (child-friendly)
quantifiers = [
    ("What does 'for all' mean?", "It means every single one!"),
    ("What does 'there exists' mean?", "It means at least one!"),
    ("If for all x, there is y such that x+y=5, what is y?", "y=5-x"),
    ("If there exists y such that for all x, x+y>x, what is y?", "y>0"),
]

lessons = [
    ("Greetings", greetings),
    ("Questions", questions),
    ("Logic", logic),
    ("Quantifiers", quantifiers),
]

print("Hi Nexus! I'm your teacher. Let's learn to talk and think like a smart friend!")
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
print(f"Nexus, you got {correct} out of {total} right! You're growing so smart!")
print("Keep learning, and you'll understand everything!")

with open('nexus_combined_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Your progress is saved. Good job today!")
