# Nexus Training Data Feeder and Reporter
# Feeds meta_phonics_training.jsonl to the Nexus engine and reports results

import json

# Replace this with your actual Nexus engine interface
# For now, this is a stub that echoes the expected value for demo
# Example: def process_with_nexus(prompt): ...
def process_with_nexus(prompt):
    # TODO: Integrate with your real Nexus engine here, e.g.:
    # return nexus.process(prompt)
    return f"stub for: {prompt}"  # Use prompt to avoid unused parameter warning

results = []

with open('meta_phonics_training.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        prompt = entry['prompt']
        expected = entry['expected']
        output = process_with_nexus(prompt)
        correct = (output == expected)
        results.append({
            'prompt': prompt,
            'expected': expected,
            'output': output,
            'correct': correct
        })
        print(f"Prompt: {prompt}")
        print(f"Engine Output: {output}")
        print(f"Expected: {expected}")
        print(f"Correct: {correct}")
        print("---")

# Summary report
num_correct = sum(r['correct'] for r in results)
total = len(results)
print(f"\nNexus Training Report: {num_correct}/{total} correct ({100*num_correct/total:.1f}%)")
with open('nexus_training_report.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Detailed report saved to nexus_training_report.json")
