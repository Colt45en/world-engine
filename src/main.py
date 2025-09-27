"""
World Engine - Main Launcher

Unified entry point for the World Engine system.
Provides multiple ways to interact with the lexicon processing capabilities.
"""

import sys
import argparse
import asyncio
from pathlib import Path


def launch_web_server():
    """Launch the FastAPI web server with integrated web interface."""
    print("Starting World Engine Web Server...")
    print("Web interface will be available at: http://localhost:8000/web/worldengine.html")
    print("API documentation at: http://localhost:8000/docs")

    try:
        import uvicorn
        from world_engine_unified.api.service import create_app
    except ImportError as e:
        print(f"Error: {e}. Please ensure all dependencies and local modules are installed.")
        print("Try running: pip install -r world_engine_unified/requirements.txt")
        return
def launch_demo():
    """Run the interactive demo."""
    try:
        from world_engine_unified.demo import main
    except ImportError as e:
        print(f"Error: {e}. Please ensure all dependencies and local modules are installed.")
def launch_cli():
    """Launch command-line interface for text processing."""
    try:
        from world_engine_unified.api.service import WorldEngineAPI
        from world_engine_unified.scales.seeds import DEFAULT_SEEDS, DEFAULT_CONSTRAINTS
    except ImportError as e:
        print(f"Error: {e}. Please ensure all dependencies and local modules are installed.")
        print("Try running: pip install -r world_engine_unified/requirements.txt")
        return

    print("World Engine CLI")
    print("================")

    # Initialize API
    api = WorldEngineAPI()

    # Load default seeds
    for word, value in DEFAULT_SEEDS.items():
        api.seed_manager.add_seed(word, value)

    for word1, word2 in DEFAULT_CONSTRAINTS:
        api.seed_manager.add_constraint(word1, word2)

    print(f"Loaded {len(api.seed_manager.seeds)} seed words")
    print("Commands: 'score <text>', 'word <word>', 'compare <word1> <word2>', 'quit'\n")

    def handle_score(api, text):
        result = api.score_token(text)
        print(f"Text: {text}")
        for token in result['tokens']:
            if token['is_seed']:
                print(f"  - {token['text']}: {token['seed_value']} ({token['pos']})")
        print(f"Summary: {result['summary']['scored_tokens']} tokens scored")

    def handle_word(api, word):
        result = api.score_word(word)
        print(f"Word: {word}")
        if result['is_seed']:
            print(f"  Seed value: {result['seed_value']}")
        else:
            print("  Not a seed word")

    def handle_compare(api, word1, word2):
        result = api.scale_between(word1, word2)
        print(f"Comparison: {word1} vs {word2}")
        if result['word1_value'] is not None and result['word2_value'] is not None:
            print(f"  {word1}: {result['word1_value']}")
            print(f"  {word2}: {result['word2_value']}")
            print(f"  Result: {result['comparison']}")
        else:
            print("  One or both words are not seed words")

    def parse_command(command):
        if command.lower() in ['quit', 'exit', 'q']:
            return 'quit', None
        elif command.startswith('score '):
            return 'score', command[6:]
        elif command.startswith('word '):
            return 'word', command[5:]
        elif command.startswith('compare '):
            parts = command[8:].split()
            if len(parts) >= 2:
                return 'compare', (parts[0], parts[1])
            else:
                return 'usage', None
        elif command.strip() == '':
            return 'empty', None
        else:
            return 'unknown', None

    while True:
        try:
            command = input("> ").strip()
            cmd_type, arg = parse_command(command)

            if cmd_type == 'quit':
                break
            elif cmd_type == 'score':
                handle_score(api, arg)
            elif cmd_type == 'word':
                handle_word(api, arg)
            elif cmd_type == 'compare':
                handle_compare(api, arg[0], arg[1])
            elif cmd_type == 'usage':
                print("Usage: compare <word1> <word2>")
            elif cmd_type == 'empty':
                continue
            else:
                print("Unknown command. Try: score, word, compare, or quit")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")

            else:
                print("Unknown command. Try: score, word, compare, or quit")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


def setup_project():
    """Run initial project setup."""
    print("Setting up World Engine...")

    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "world_engine_unified").exists():
        print("Error: Please run this from the WORLD ENGINE directory")
        return False

    print("✓ Project structure found")

    # Check Python dependencies
    try:
        import fastapi
        import spacy
        print("✓ Core dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: pip install -r world_engine_unified/requirements.txt")
        return False

    # Check spaCy model
    try:
        import spacy # type: ignore
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy English model available")
    except OSError:
        print("✗ spaCy English model not found")
        print("Run: python -m spacy download en_core_web_sm")
        return False

    print("\n✅ Setup complete! You can now run:")
    print("  python -m world_engine_unified.main server    # Web interface")
    print("  python -m world_engine_unified.main demo      # Interactive demo")
    print("  python -m world_engine_unified.main cli       # Command line")

    return True


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="World Engine - Unified Lexicon Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m world_engine_unified.main server    # Start web server
  python -m world_engine_unified.main demo      # Run demo
  python -m world_engine_unified.main cli       # Command line interface
  python -m world_engine_unified.main setup     # Initial setup
        """
    )

    parser.add_argument(
        'mode',
        choices=['server', 'demo', 'cli', 'setup'],
        help='Execution mode'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for web server (default: 8000)'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.mode == 'setup':
        setup_project()
    elif args.mode == 'server':
        launch_web_server()
    elif args.mode == 'demo':
        launch_demo()
    elif args.mode == 'cli':
        launch_cli()


if __name__ == "__main__":
    main()
