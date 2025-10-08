#!/usr/bin/env python3
"""
Quick RAG System Test
Simple test to verify the system is working
"""

import sys
from world_engine_rag import WorldEngineRAG, RAGConfig

def main():
    print("=== Quick RAG Test ===")

    # Create RAG system with lightweight config
    config = RAGConfig(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        top_k=2
    )

    rag = WorldEngineRAG(config)

    print("Building vector store...")
    if not rag.build_vector_store():
        print("❌ Failed to build vector store")
        return False

    print("Testing query...")
    response = rag.query("What is World Engine?", top_k=1)

    if response["success"]:
        result = response["results"][0]
        print(f"✅ Query successful!")
        print(f"📄 Source: {result['source']}")
        print(f"📝 Content: {result['content'][:200]}...")
    else:
        print(f"❌ Query failed: {response['error']}")
        return False

    print("\n🎉 RAG system is working correctly!")
    return True

if __name__ == "__main__":
    main()
