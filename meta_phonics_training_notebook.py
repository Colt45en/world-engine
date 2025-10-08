# Phonics and Meta-Understanding Training Notebook
# For MetaFractalEngine: 'if that is that then this is that' logic

import json

def atomic_breakdown(word):
    # Simple phonics breakdown (replace with real phonics if available)
    phonics = list(word)
    return phonics

def morpheme_breakdown(word):
    # Simple morpheme split (replace with real rules)
    if word.startswith('re'):
        return ['re', word[2:]]
    elif word.endswith('ing'):
        return [word[:-3], 'ing']
    else:
        return [word]

# Training pairs: 'if X is Y then Y is Z' style
training_pairs = [
    # Phonics
    ("if cat is [k][a][t] then [k][a][t] is cat", "cat"),
    ("if phone is [f][o][n] then [f][o][n] is phone", "phone"),
    # Morphemes
    ("if rebuild is re + build then re + build is rebuild", "rebuild"),
    ("if running is run + ing then run + ing is running", "running"),
    # Meta-understanding
    ("if cat is animal then animal is what?", "cat"),
    ("if run is verb then verb is what?", "run"),
]

# Display and prepare for feeding to engine
for prompt, expected in training_pairs:
    print(f"Prompt: {prompt}")
    print(f"Expected: {expected}")
    print("Breakdown:")
    if '[' in prompt:
        # Phonics
        word = expected
        print(f"  Phonics: {atomic_breakdown(word)}")
    elif '+' in prompt:
        # Morphemes
        word = expected
        print(f"  Morphemes: {morpheme_breakdown(word)}")
    else:
        print("  Meta: Logic mapping")
    print()

# Save as JSONL for engine training
with open('meta_phonics_training.jsonl', 'w') as f:
    for prompt, expected in training_pairs:
        entry = {'prompt': prompt, 'expected': expected}
        f.write(json.dumps(entry) + '\n')
print("Training data saved to meta_phonics_training.jsonl")
