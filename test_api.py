#!/usr/bin/env python3
"""
Test API endpoints for the thought pipeline
"""

import asyncio
import aiohttp
import json


async def test_api_endpoints():
    """Test the new API endpoints."""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # Test process_thought endpoint
        print("Testing /process_thought endpoint...")
        thought_data = {
            "text": "What is the meaning of life and the universe?",
            "priority": 5,
            "metadata": {"test": True}
        }
        
        async with session.post(f"{base_url}/process_thought", json=thought_data) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ process_thought: {result['status']}")
                print(f"   Request ID: {result['request_id']}")
                print(f"   Stages: {len(result['results'])}")
            else:
                print(f"❌ process_thought failed: {response.status}")
        
        # Test analyze_meaning endpoint
        print("\nTesting /analyze_meaning endpoint...")
        meaning_data = {
            "text": "The red fire burns bright under the starry night sky.",
            "analysis_types": ["lexical", "holistic", "imagery", "symbolic"]
        }
        
        async with session.post(f"{base_url}/analyze_meaning", json=meaning_data) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ analyze_meaning: {result['analysis_count']} analyses")
                for analysis_type, analysis_result in result['analysis_results'].items():
                    print(f"   {analysis_type}: confidence {analysis_result['confidence']:.2f}")
            else:
                print(f"❌ analyze_meaning failed: {response.status}")


if __name__ == "__main__":
    print("Testing API endpoints...")
    print("Make sure the server is running with: python main.py server")
    print("=" * 50)
    
    try:
        asyncio.run(test_api_endpoints())
    except Exception as e:
        print(f"Error testing API: {e}")
        print("Make sure the server is running!")