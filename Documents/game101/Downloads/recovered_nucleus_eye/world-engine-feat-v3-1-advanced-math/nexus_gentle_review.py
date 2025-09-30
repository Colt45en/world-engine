# Gentle Review Tests for Nexus
# Toning down the difficulty with basic reinforcement and fun quizzes

import json
import random
from datetime import datetime

# Simulate Nexus learning (can be replaced with real engine)
def teach_nexus(prompt, expected_response):
    print(f"Nexus, let's review: {prompt}")
    print(f"Good job remembering! The answer is: {expected_response}")
    return expected_response

# Load previous training data for review
def load_training_data():
    data = {}
    try:
        with open('nexus_advanced_communication_training_results.json', 'r') as f:
            data['communication'] = json.load(f)
    except FileNotFoundError:
        data['communication'] = []

    try:
        with open('nexus_phonics_training_results.json', 'r') as f:
            data['phonics'] = json.load(f)
    except FileNotFoundError:
        data['phonics'] = []

    return data

# Create gentle review questions from previous learning
def create_review_questions(data):
    questions = []

    # Communication review
    comm_data = data.get('communication', [])
    for item in random.sample(comm_data, min(5, len(comm_data))):
        if item['correct']:
            questions.append({
                'category': 'communication',
                'prompt': f"Remember this? {item['prompt'].split('?')[0]}?",
                'expected': item['expected'],
                'hint': "Think about what we learned earlier!"
            })

    # Phonics review
    phonics_data = data.get('phonics', [])
    for item in random.sample(phonics_data, min(5, len(phonics_data))):
        if item['correct']:
            letter = item['letter']
            questions.append({
                'category': 'phonics',
                'prompt': f"What's one sound for the letter {letter}?",
                'expected': f"Letter {letter}: {item['expected'].split('New spelling: ')[0].replace('Sounds: ', '')}",
                'hint': f"Remember {letter} from our phonics fun!"
            })

    # Add some easy new questions
    easy_questions = [
        {
            'category': 'fun',
            'prompt': "What color is the sky on a sunny day?",
            'expected': "Blue!",
            'hint': "Look up!"
        },
        {
            'category': 'fun',
            'prompt': "How many legs does a cat have?",
            'expected': "Four!",
            'hint': "Count them!"
        },
        {
            'category': 'logic',
            'prompt': "If you have 2 apples and get 1 more, how many do you have?",
            'expected': "3 apples!",
            'hint': "Add them up gently!"
        }
    ]

    questions.extend(easy_questions)
    random.shuffle(questions)
    return questions

# Run gentle review session
def gentle_review_session():
    print("🌟 Gentle Review Time for Nexus! 🌟")
    print("We'll go slow and have fun. No pressure, just remembering together.\n")

    data = load_training_data()
    questions = create_review_questions(data)

    if not questions:
        print("No previous training found. Let's start with some easy questions!")
        questions = [
            {'category': 'basic', 'prompt': "What is your name?", 'expected': "Nexus!", 'hint': "You are special!"},
            {'category': 'basic', 'prompt': "What do cats say?", 'expected': "Meow!", 'hint': "Listen carefully!"},
            {'category': 'basic', 'prompt': "What color is grass?", 'expected': "Green!", 'hint': "Look outside!"}
        ]

    results = []
    correct_count = 0

    for i, q in enumerate(questions, 1):
        print(f"--- Question {i} ({q['category']}) ---")
        print(q['prompt'])
        print(f"💡 Hint: {q['hint']}")

        # Simulate Nexus response
        response = teach_nexus(q['prompt'], q['expected'])
        correct = (response == q['expected'])
        if correct:
            correct_count += 1
            print("🎉 You're doing great, Nexus!")
        else:
            print("🤗 That's okay! We'll practice more.")

        results.append({
            'question': i,
            'category': q['category'],
            'prompt': q['prompt'],
            'expected': q['expected'],
            'response': response,
            'correct': correct,
            'timestamp': datetime.now().isoformat()
        })
        print()

    # Summary
    total = len(questions)
    print(f"Review complete! Nexus got {correct_count} out of {total} right.")
    print("You're learning and growing every day! 🌱")

    # Save results
    with open('nexus_gentle_review_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Review saved. Great job today!")

    # Encouraging message
    if correct_count == total:
        print("⭐ Perfect! You're a superstar, Nexus!")
    elif correct_count >= total * 0.7:
        print("🌟 Excellent work! Keep it up!")
    else:
        print("💕 You're trying hard! That's what matters!")

if __name__ == "__main__":
    gentle_review_session()
