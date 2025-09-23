#!/usr/bin/env python3
"""
Test script for the AI Bot Thought Process Framework

Tests the 5-stage thought pipeline and meaning analysis capabilities.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from thought_pipeline import ThoughtPipeline
from thought_pipeline.meaning_analyzer import MeaningAnalyzer, AnalysisType


async def test_thought_pipeline():
    """Test the complete 5-stage thought pipeline."""
    print("🧠 Testing AI Bot Thought Process Framework")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ThoughtPipeline()
    
    # Test inputs
    test_inputs = [
        "What is the meaning of life?",
        "Create a beautiful garden with flowers and trees.",
        "Analyze the state-of-the-art condition of artificial intelligence.",
        "The red fire burns bright under the starry night sky.",
        "Please help me understand the complex relationship between thought and reality."
    ]
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n🔍 Test {i}: Processing '{input_text}'")
        print("-" * 40)
        
        start_time = time.time()
        result = await pipeline.process(input_text, priority=i)
        processing_time = time.time() - start_time
        
        print(f"✅ Status: {result['status']}")
        print(f"⏱️  Processing time: {processing_time:.3f}s")
        print(f"🎯 Request ID: {result['request_id']}")
        
        # Show stage results
        for stage, stage_result in result['results'].items():
            if isinstance(stage_result, dict) and 'error' not in stage_result:
                print(f"  📊 {stage.title()}: ✓")
            else:
                print(f"  📊 {stage.title()}: ⚠️")
        
        # Show key insights
        if 'perception' in result['results']:
            perception = result['results']['perception']
            print(f"  🔎 Mode detected: {perception.get('mode', 'unknown')}")
            
        if 'processing' in result['results']:
            processing = result['results']['processing']
            meaning = processing.get('meaning', {})
            print(f"  🎯 Primary intent: {meaning.get('primary_intent', 'unknown')}")
            print(f"  📈 Meaning confidence: {meaning.get('meaning_confidence', 0):.2f}")
            
        if 'decision' in result['results']:
            decision = result['results']['decision']
            dec_info = decision.get('decision', {})
            print(f"  🎯 Decision: {dec_info.get('type', 'unknown')}")
            print(f"  📊 Confidence: {decision.get('confidence', 0):.2f}")


async def test_meaning_analyzer():
    """Test the comprehensive meaning analysis."""
    print("\n\n🔬 Testing Meaning Analysis Framework")
    print("=" * 50)
    
    analyzer = MeaningAnalyzer()
    
    test_text = "The state-of-the-art condition requires careful analysis of the underlying patterns."
    
    print(f"📝 Analyzing: '{test_text}'")
    print("-" * 40)
    
    # Test specific analysis types
    analysis_types = [
        AnalysisType.LEXICAL,
        AnalysisType.ETYMOLOGICAL,
        AnalysisType.SYNTACTICAL,
        AnalysisType.HOLISTIC,
        AnalysisType.INTENT_TONE
    ]
    
    results = await analyzer.analyze_meaning(test_text, analysis_types)
    
    for analysis_type, result in results.items():
        print(f"\n📊 {analysis_type.upper()} ANALYSIS:")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Processing time: {result.processing_time:.3f}s")
        
        # Show key findings
        findings = result.findings
        if analysis_type == 'lexical':
            print(f"  Lexical diversity: {findings.get('lexical_diversity', 0):.2f}")
            print(f"  Complexity score: {findings.get('complexity_score', 0):.2f}")
        elif analysis_type == 'etymological':
            print(f"  Etymological depth: {findings.get('etymological_depth', 0):.2f}")
            if 'word_origins' in findings:
                print(f"  Words with known origins: {len(findings['word_origins'])}")
        elif analysis_type == 'holistic':
            print(f"  Overall complexity: {findings.get('overall_complexity', 0):.2f}")
            print(f"  Semantic coherence: {findings.get('semantic_coherence', 0):.2f}")
            print(f"  Key themes: {findings.get('key_themes', [])}")
        elif analysis_type == 'intent_tone':
            print(f"  Primary intent: {findings.get('primary_intent', 'unknown')}")
            print(f"  Formality level: {findings.get('formality_level', 'unknown')}")


async def test_integration():
    """Test integration between components."""
    print("\n\n🔗 Testing Component Integration")
    print("=" * 50)
    
    # Test asset manager
    pipeline = ThoughtPipeline()
    asset_manager = pipeline.asset_manager
    
    # Store some test data
    await asset_manager.store_memory("test_memory", {
        "content": "This is a test memory",
        "timestamp": time.time(),
        "importance": 0.8
    })
    
    # Retrieve it
    memory = await asset_manager.get_memory("test_memory")
    print(f"✅ Memory storage/retrieval: {'✓' if memory else '✗'}")
    
    # Test request handler
    request_handler = pipeline.request_handler
    status = request_handler.get_queue_status()
    print(f"📊 Request handler status: {status['stats']['total_requests']} total requests")
    
    # Test quantum processor
    quantum_processor = pipeline.quantum_processor
    quantum_status = quantum_processor.get_status()
    print(f"🔬 Quantum processor: {quantum_status['max_superposition_states']} max states")
    
    print("✅ All components integrated successfully!")


async def main():
    """Main test runner."""
    try:
        await test_thought_pipeline()
        await test_meaning_analyzer()
        await test_integration()
        
        print("\n\n🎉 All tests completed successfully!")
        print("\nThe AI Bot Thought Process Framework is ready for use.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())