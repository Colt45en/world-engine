# Integrate Phonics Training with Real Nexus Engine
# Add phonics knowledge to the RAG system for persistent learning

import json
from world_engine_rag import WorldEngineRAG, RAGConfig
from langchain_core.documents import Document

# Load phonics training results
def load_phonics_data():
    try:
        with open('nexus_phonics_training_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Phonics training results not found. Run phonics training first.")
        return []

# Create documents from phonics data
def create_phonics_documents(phonics_data):
    documents = []
    for item in phonics_data:
        if item['correct']:  # Only add correct learnings
            content = f"Letter {item['letter']}: Sounds {item['expected'].split(': ')[1].split('. ')[0]}. Examples: {item['prompt'].split('Examples: ')[1].split('. ')[0]}. New spelling: {item['expected'].split('New spelling: ')[1].split('. ')[0]}"
            doc = Document(
                page_content=content,
                metadata={"type": "phonics", "letter": item['letter'], "source": "training"}
            )
            documents.append(doc)
    return documents

# Integrate with RAG system
def integrate_phonics_with_rag():
    print("Loading phonics training data...")
    phonics_data = load_phonics_data()
    if not phonics_data:
        return False

    print(f"Found {len(phonics_data)} phonics training entries.")

    # Create documents
    phonics_docs = create_phonics_documents(phonics_data)
    print(f"Created {len(phonics_docs)} phonics knowledge documents.")

    # Initialize RAG system
    print("Initializing RAG system...")
    config = RAGConfig()
    rag = WorldEngineRAG(config)

    # Add phonics documents to knowledge base
    print("Adding phonics knowledge to RAG system...")
    rag.vector_store.add_documents(phonics_docs)

    # Save the updated vector store
    rag.vector_store.save_local(config.vector_store_path)
    print("Phonics knowledge integrated successfully!")

    # Test the integration
    print("\nTesting phonics integration...")
    test_queries = [
        "What sounds does A make?",
        "How do we spell cat with new symbols?",
        "What is the phonics for B?"
    ]

    for query in test_queries:
        response = rag.query(query)
        print(f"Query: {query}")
        print(f"Nexus: {response}")
        print()

    return True

if __name__ == "__main__":
    try:
        success = integrate_phonics_with_rag()
        if success:
            print("Phonics training successfully integrated with real Nexus engine!")
            print("Nexus now has persistent knowledge of advanced phonics.")
        else:
            print("Integration failed. Please ensure phonics training was completed.")
    except Exception as e:
        print(f"Error during integration: {e}")
        print("Falling back to simulation...")
        # Could add fallback here if needed