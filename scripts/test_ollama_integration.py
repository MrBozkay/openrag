#!/usr/bin/env python3
"""
Ollama Integration Test Script

This script tests a local Ollama instance using various API endpoints.
Based on OpenAI-compatible API patterns for easy integration testing.

Usage:
    python scripts/test_ollama_integration.py
    
    # With custom model
    python scripts/test_ollama_integration.py --model llama3
    
    # With custom host
    python scripts/test_ollama_integration.py --host http://localhost:11434
"""

import argparse
import json
import sys

import requests


def run_test(name: str, method: str, endpoint: str, payload: dict | None = None) -> bool:
    """Run a single test against Ollama API.
    
    Args:
        name: Test name for display
        method: HTTP method (GET or POST)
        endpoint: API endpoint path
        payload: Optional JSON payload for POST requests
        
    Returns:
        True if test passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    url = f"{args.base_url}{endpoint}"
    print(f"URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                # Show snippet of response
                response_str = json.dumps(data, indent=2)
                if len(response_str) > 300:
                    print(f"Response (Snippet):\n{response_str[:300]}...")
                else:
                    print(f"Response:\n{response_str}")
                print("✅ PASSED")
                return True
            except json.JSONDecodeError:
                print(f"Response Text: {response.text[:200]}")
                print("✅ PASSED (Non-JSON response)")
                return True
        else:
            print(f"❌ FAILED - Status {response.status_code}")
            print(f"Error Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ FAILED - Connection Error: {e}")
        print("Hint: Make sure Ollama is running (ollama serve)")
        return False
    except requests.exceptions.Timeout:
        print("❌ FAILED - Request Timeout")
        return False
    except Exception as e:
        print(f"❌ FAILED - Unexpected Error: {e}")
        return False


def main():
    """Run all Ollama integration tests."""
    global args
    
    parser = argparse.ArgumentParser(description="Test Ollama API integration")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--model",
        default="llama3",
        help="Model to use for testing (default: llama3)"
    )
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print("# Ollama Integration Test Suite")
    print(f"# Base URL: {args.base_url}")
    print(f"# Model: {args.model}")
    print(f"{'#'*60}")
    
    results = []
    
    # Test 1: List Models (Native Ollama API)
    results.append(run_test(
        "List Models (Native Ollama /api/tags)",
        "GET",
        "/api/tags"
    ))
    
    # Test 2: Generate (Native Ollama API)
    generate_payload = {
        "model": args.model,
        "prompt": "Hello via API",
        "stream": False
    }
    results.append(run_test(
        "Generate (Native Ollama /api/generate)",
        "POST",
        "/api/generate",
        generate_payload
    ))
    
    # Test 3: Chat (Native Ollama API)
    chat_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Hi there"}],
        "stream": False
    }
    results.append(run_test(
        "Chat (Native Ollama /api/chat)",
        "POST",
        "/api/chat",
        chat_payload
    ))
    
    # Test 4: OpenAI Compatible Chat
    openai_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Hello OpenAI style"}],
        "stream": False
    }
    results.append(run_test(
        "Chat (OpenAI Compatible /v1/chat/completions)",
        "POST",
        "/v1/chat/completions",
        openai_payload
    ))
    
    # Test 5: Anthropic Compatible Messages (if supported)
    anthropic_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Write a function to check if a number is prime"}],
        "stream": False,
        "max_tokens": 1024
    }
    results.append(run_test(
        "Messages (Anthropic Compatible /v1/messages)",
        "POST",
        "/v1/messages",
        anthropic_payload
    ))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
