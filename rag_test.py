#!/usr/bin/env python3
"""
RAG System Test - Modern LangChain Implementation
Testing modern LangChain namespaces and FAISS integration
"""

import sys
import traceback

def test_imports():
    """Test all required imports"""
    print("Testing imports...")

    try:
        from langchain_core.documents import Document
        from langchain_core.vectorstores import VectorStore
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ All LangChain imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_embeddings():
    """Test HuggingFace embeddings"""
    print("\nTesting embeddings...")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        # Use a lightweight model for testing
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Test embedding a simple text
        test_text = "This is a test document for the World Engine RAG system."
        embedding = embeddings.embed_query(test_text)

        print(f"✅ Embedding successful - dimension: {len(embedding)}")
        return embeddings
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        traceback.print_exc()
        return None

def test_vector_store(embeddings):
    """Test FAISS vector store"""
    print("\nTesting vector store...")

    try:
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS

        # Create test documents
        docs = [
            Document(
                page_content="World Engine is a spatial computing platform with advanced sensory framework.",
                metadata={"source": "world_engine_docs", "type": "technical"}
            ),
            Document(
                page_content="NEXUS 3.0 provides recursive intelligence capabilities for advanced reasoning.",
                metadata={"source": "nexus_docs", "type": "ai_capability"}
            ),
            Document(
                page_content="The quantum-thought component handles complex mathematical operations.",
                metadata={"source": "quantum_docs", "type": "component"}
            )
        ]

        # Create vector store
        vector_store = FAISS.from_documents(docs, embeddings)

        # Test similarity search
        query = "What is World Engine?"
        results = vector_store.similarity_search(query, k=2)

        print(f"✅ Vector store created with {len(docs)} documents")
        print(f"✅ Similarity search returned {len(results)} results")

        for i, result in enumerate(results):
            print(f"  Result {i+1}: {result.page_content[:50]}...")

        return vector_store
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        traceback.print_exc()
        return None

def test_rag_retrieval(vector_store):
    """Test RAG retrieval functionality"""
    print("\nTesting RAG retrieval...")

    try:
        # Test different queries
        test_queries = [
            "How does World Engine work?",
            "What are NEXUS capabilities?",
            "Tell me about quantum components"
        ]

        for query in test_queries:
            results = vector_store.similarity_search(query, k=1)
            if results:
                print(f"✅ Query: '{query}'")
                print(f"   Best match: {results[0].page_content}")
                print(f"   Metadata: {results[0].metadata}")
            else:
                print(f"❌ No results for query: '{query}'")

        return True
    except Exception as e:
        print(f"❌ RAG retrieval error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== World Engine RAG System Test ===")
    print(f"Python version: {sys.version}")

    # Test imports
    if not test_imports():
        print("❌ Import test failed - cannot continue")
        return False

    # Test embeddings
    embeddings = test_embeddings()
    if not embeddings:
        print("❌ Embeddings test failed - cannot continue")
        return False

    # Test vector store
    vector_store = test_vector_store(embeddings)
    if not vector_store:
        print("❌ Vector store test failed - cannot continue")
        return False

    # Test RAG retrieval
    if not test_rag_retrieval(vector_store):
        print("❌ RAG retrieval test failed")
        return False

    print("\n🎉 All RAG tests passed! System is ready for integration.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
