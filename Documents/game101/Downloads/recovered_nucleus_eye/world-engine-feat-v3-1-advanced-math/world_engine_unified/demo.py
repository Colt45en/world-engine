"""Demo script for World Engine."""

from .api.service import WorldEngineAPI
from .scales.seeds import DEFAULT_SEEDS, DEFAULT_CONSTRAINTS
import sys


def main():
    """Run the interactive demo."""
    print("🌍 World Engine Demo")
    print("==================")

    # Initialize API
    api = WorldEngineAPI()

    # Load default seeds
    for word, value in DEFAULT_SEEDS.items():
        api.seed_manager.add_seed(word, value)

    for word1, word2 in DEFAULT_CONSTRAINTS:
        api.seed_manager.add_constraint(word1, word2)

    print(f"Loaded {len(api.seed_manager.seeds)} seed words")
    print("\nTry some examples:")

    # Demo examples
    examples = [
        "This movie is excellent!",
        "The weather is terrible today",
        "I feel great about this project"
    ]

    for text in examples:
        print(f"\nText: '{text}'")
        result = api.score_token(text)
        for token in result['tokens']:
            if token['is_seed']:
                print(f"  - {token['text']}: {token['seed_value']}")

    print("\nDemo complete!")
    return True


if __name__ == "__main__":
    main()
