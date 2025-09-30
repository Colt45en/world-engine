"""
World Engine Demo - Quick demonstration of lexicon processing capabilities.

This script shows how to use the World Engine for analyzing text and scoring words.
"""

import asyncio
from pathlib import Path
from world_engine_unified.api.service import create_app, WordEngineAPI
from world_engine_unified.scales.seeds import SeedManager, DEFAULT_SEEDS, DEFAULT_CONSTRAINTS


def demo_basic_scoring():
    """Demonstrate basic word scoring functionality."""
    print("=== World Engine Demo ===\n")

    # Initialize API
    api = WorldEngineAPI()

    # Add some default seeds
    for word, value in DEFAULT_SEEDS.items():
        api.seed_manager.add_seed(word, value)

    for word1, word2 in DEFAULT_CONSTRAINTS:
        api.seed_manager.add_constraint(word1, word2)

    print("1. Seed Words and Values:")
    for word, value in api.seed_manager.seeds.items():
        print(f"   {word}: {value}")
    print()

    # Test sentences
    test_sentences = [
        "The movie was absolutely amazing and great!",
        "This is a terrible and bad experience.",
        "The weather is neutral today.",
        "She gave an excellent performance that was truly outstanding."
    ]

    print("2. Text Analysis:")
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n   Sentence {i}: '{sentence}'")

        result = api.score_token(sentence)
        scored_tokens = result['tokens']

        print(f"   Scored tokens:")
        for token in scored_tokens:
            if token['is_seed']:
                print(f"     - {token['text']} (lemma: {token['lemma']}, value: {token['seed_value']}, pos: {token['pos']})")

        print(f"   Summary: {result['summary']['scored_tokens']} tokens scored out of {result['summary']['total_tokens']} total")

    print("\n3. Word Comparisons:")
    comparisons = [
        ("terrible", "excellent"),
        ("bad", "good"),
        ("neutral", "amazing")
    ]

    for word1, word2 in comparisons:
        result = api.scale_between(word1, word2)
        if result['word1_value'] is not None and result['word2_value'] is not None:
            print(f"   {word1} ({result['word1_value']}) vs {word2} ({result['word2_value']}) -> {result['comparison']}")


def demo_contextual_analysis():
    """Demonstrate contextual word analysis."""
    print("\n4. Contextual Analysis:")

    api = WorldEngineAPI()

    # Add seeds
    for word, value in DEFAULT_SEEDS.items():
        api.seed_manager.add_seed(word, value)

    context_examples = [
        ("good", "The good doctor helped many patients."),
        ("good", "This is a really good pizza."),
        ("state", "The state of California has many laws."),
        ("state", "The machine is in a broken state.")
    ]

    for word, context in context_examples:
        result = api.score_word(word, context)
        print(f"\n   Word: '{word}' in context: '{context}'")
        if result.get('pos'):
            print(f"   Part of speech: {result['pos']}")
            print(f"   Dependency: {result['dependency']}")
            print(f"   Head word: {result['head']}")


async def demo_api_server():
    """Demonstrate running the API server."""
    print("\n=== Starting API Server Demo ===")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("Web interface at: http://localhost:8000/web/worldengine.html")
    print("\nPress Ctrl+C to stop the server\n")

    import uvicorn
    app = create_app()

    # Configure uvicorn to be less verbose
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except KeyboardInterrupt:
        print("\nShutting down server...")


def main():
    """Main demo function."""
    print("World Engine - Unified Lexicon Processing System")
    print("================================================")

    # Run basic demos
    demo_basic_scoring()
    demo_contextual_analysis()

    print("\n" + "="*50)
    print("Demo completed! Key features demonstrated:")
    print("- Seed-based word scoring")
    print("- Text tokenization and analysis")
    print("- Contextual word processing")
    print("- Word comparison and scaling")
    print("\nTo start the API server, run:")
    print("python -m world_engine_unified.demo server")
    print("\nOr run individual components:")
    print("python -m world_engine_unified.api.service")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run the API server
        asyncio.run(demo_api_server())
    else:
        # Run the basic demo
        main()
