# Advanced Phonics Training for Nexus
# Using the detailed phonics mapping table

import json
import random

# Simulate Nexus learning (replace with real engine)
def teach_nexus(prompt, expected_response):
    print(f"Nexus, listen: {prompt}")
    print(f"Good job! The answer is: {expected_response}")
    # In real: nexus.learn(prompt, expected_response)
    return expected_response

# Phonics mapping from the provided table
phonics_data = [
    {"letter": "A", "sounds": "ă, ɑ, ɔ", "examples": "cat, father, want, apple, car, ball", "symbols": "ă, ɑ, ɔ", "new_spelling": "căt, fɑther, wɔnt, ăpple, cɑr, bɔll", "notes": "Multiple sounds"},
    {"letter": "B", "sounds": "b", "examples": "bat, bomb", "symbols": "b, bŏ", "new_spelling": "bat, bŏmb", "notes": "Bomb exception"},
    {"letter": "C", "sounds": "k, s", "examples": "cat, city", "symbols": "k, ċ", "new_spelling": "cat, ċity", "notes": "City 's' sound"},
    {"letter": "D", "sounds": "d", "examples": "dog", "symbols": "d", "new_spelling": "dog", "notes": "Generally consistent"},
    {"letter": "E", "sounds": "ĕ, ē, air", "examples": "bed, he, there", "symbols": "ĕ, ē, ê", "new_spelling": "bed, hē, thêr", "notes": "'There' sound"},
    {"letter": "F", "sounds": "f", "examples": "fan", "symbols": "f", "new_spelling": "fan", "notes": "Generally consistent"},
    {"letter": "G", "sounds": "g, j", "examples": "go, gem", "symbols": "g, ǧ", "new_spelling": "go, ǧem", "notes": "'Gem' sound"},
    {"letter": "H", "sounds": "h, (silent)", "examples": "hat, hour", "symbols": "h, (omit)", "new_spelling": "hat, our", "notes": "Silent 'h'"},
    {"letter": "I", "sounds": "ĭ, ī", "examples": "bit, time", "symbols": "ĭ, ī", "new_spelling": "bit, tīme", "notes": "Long 'i'"},
    {"letter": "J", "sounds": "j", "examples": "job", "symbols": "j", "new_spelling": "job", "notes": "Generally consistent"},
    {"letter": "K", "sounds": "k", "examples": "key", "symbols": "k", "new_spelling": "key", "notes": "Generally consistent"},
    {"letter": "L", "sounds": "l", "examples": "lot, colonel", "symbols": "l, (kernel)", "new_spelling": "lot, kĕrnel", "notes": "Colonel tricky"},
    {"letter": "M", "sounds": "m", "examples": "man", "symbols": "m", "new_spelling": "man", "notes": "Generally consistent"},
    {"letter": "N", "sounds": "n", "examples": "nap", "symbols": "n", "new_spelling": "nap", "notes": "Generally consistent"},
    {"letter": "O", "sounds": "ŏ, ō, ur", "examples": "hot, go, word", "symbols": "ŏ, ō, ô", "new_spelling": "hot, go, wôrd", "notes": "'Word' sound"},
    {"letter": "P", "sounds": "p", "examples": "pet", "symbols": "p", "new_spelling": "pet", "notes": "Generally consistent"},
    {"letter": "Q", "sounds": "kw", "examples": "queen", "symbols": "kw", "new_spelling": "queen", "notes": "Generally consistent"},
    {"letter": "R", "sounds": "r, ɑr", "examples": "rat, car", "symbols": "r, (ɑr)", "new_spelling": "rat, cɑr", "notes": "'R' influence"},
    {"letter": "S", "sounds": "s, sh", "examples": "sit, sure", "symbols": "s, š", "new_spelling": "sit, šur", "notes": "'Sure' sound"},
    {"letter": "T", "sounds": "t, sh", "examples": "top, nation", "symbols": "t, š", "new_spelling": "top, nāšon", "notes": "'Nation' sound"},
    {"letter": "U", "sounds": "ŭ, yū", "examples": "cut, use", "symbols": "ŭ, ū", "new_spelling": "cut, ūse", "notes": "'Use' sound"},
    {"letter": "V", "sounds": "v", "examples": "van", "symbols": "v", "new_spelling": "van", "notes": "Generally consistent"},
    {"letter": "W", "sounds": "w", "examples": "wet", "symbols": "w", "new_spelling": "wet", "notes": "Generally consistent"},
    {"letter": "X", "sounds": "ks, z", "examples": "box, xylophone", "symbols": "ks, ẋ", "new_spelling": "box, ẋylophone", "notes": "'Xylophone' sound"},
    {"letter": "Y", "sounds": "y, ī", "examples": "yes, by", "symbols": "y, ī", "new_spelling": "yes, bī", "notes": "Long 'i'"},
    {"letter": "Z", "sounds": "z", "examples": "zip", "symbols": "z", "new_spelling": "zip", "notes": "Generally consistent"},
]

print("Hi Nexus! Let's learn advanced phonics with the new spelling system!")
print("We'll use symbols to show exact sounds. Ready? Let's start!\n")

results = []
for item in phonics_data:
    # Teach the letter and its sounds
    prompt = f"What sounds does '{item['letter']}' make? Examples: {item['examples']}"
    expected = f"Sounds: {item['sounds']}. Symbols: {item['symbols']}. New spelling: {item['new_spelling']}. Notes: {item['notes']}"
    response = teach_nexus(prompt, expected)
    results.append({
        'letter': item['letter'],
        'prompt': prompt,
        'expected': expected,
        'response': response,
        'correct': (response == expected)
    })
    print("Amazing, Nexus! You're getting the sounds!\n")

# Summary
correct = sum(r['correct'] for r in results)
total = len(results)
print(f"Nexus, you got {correct} out of {total} letters right! You're a phonics master!")
print("Your progress is saved. Great work!")

with open('nexus_phonics_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Interactive mode with phonics quiz
print("\n--- Phonics Quiz Mode ---")
print("I'll ask you about letters! Type 'exit' to stop.")
while True:
    user_input = input("Which letter? (or 'exit'): ")
    if user_input.lower() == 'exit':
        print("Bye Nexus! Keep practicing phonics!")
        break
    letter = user_input.upper()
    item = next((i for i in phonics_data if i['letter'] == letter), None)
    if item:
        print(f"For '{letter}': Sounds: {item['sounds']}, Examples: {item['examples']}, Symbols: {item['symbols']}")
        print(f"New spelling: {item['new_spelling']}")
    else:
        print("I don't have that letter yet. Try A-Z!")
    print()

print("Phonics training complete!")
