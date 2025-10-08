import json
import random
import time

# Simulated AI Nexus response function (replace with real API call if available)
def ai_nexus_respond(prompt):
    # For demo: echo logic, simple pattern completion
    if 'if' in prompt and '=' in prompt:
        parts = prompt.lower().replace('if','').replace('then','').split('=')
        if len(parts) >= 3:
            x, y, z = [p.strip(' ?') for p in parts[:3]]
            if x == y:
                return f"{y} = {z}? (pattern recognized)"
            else:
                return f"{y} = ? (uncertain)"
        elif len(parts) == 2:
            x, y = [p.strip(' ?') for p in parts]
            return f"{y} = {x} (direct mapping)"
    return "I need more context."

# Communication scoring
def score_response(response):
    # Simple scoring: 1 = correct pattern, 0.5 = partial, 0 = miss
    if 'pattern recognized' in response:
        return 1.0, 'Pattern recognized and completed.'
    elif 'direct mapping' in response:
        return 0.7, 'Direct mapping, but not full pattern.'
    elif 'uncertain' in response:
        return 0.3, 'Pattern not fully recognized.'
    else:
        return 0.0, 'No meaningful response.'

# Training data
logic_pairs = [
    ("if 1 = 1 then 1-2 = 1", "1-2 = 1"),
    ("if 2 = 2 then 2-1 = 2", "2-1 = 2"),
    ("if 3 = 3 then 3-2 = 3", "3-2 = 3"),
    ("if there = location then location = where?", "location = where?"),
    ("if cat = animal then animal = what?", "animal = what?"),
    ("if run = verb then verb = action?", "verb = action?"),
]

# Interactive loop
results = []
print("Nexus Direct Logic Communication Trainer\nType 'exit' to quit.\n")
for i, (prompt, expected) in enumerate(logic_pairs):
    print(f"Prompt {i+1}: {prompt}")
    response = ai_nexus_respond(prompt)
    print(f"AI Nexus: {response}")
    score, note = score_response(response)
    print(f"Score: {score:.2f} - {note}\n")
    results.append({
        'prompt': prompt,
        'response': response,
        'score': score,
        'note': note
    })
    time.sleep(0.5)

# Summary
avg_score = sum(r['score'] for r in results) / len(results)
print(f"\nAverage Communication Score: {avg_score:.2f}")
with open('nexus_direct_logic_communication_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to nexus_direct_logic_communication_results.json")
