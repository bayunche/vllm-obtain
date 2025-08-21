#!/usr/bin/env python3
"""
验证修复后的推理功能
"""

import requests
import json
import time


def test_chat_completion():
    """测试聊天补全"""
    url = "http://127.0.0.1:8001/v1/chat/completions"
    
    payload = {
        "model": "qwen-0.5b",
        "messages": [
            {"role": "user", "content": "Hello, can you introduce yourself briefly?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    print("Testing chat completion...")
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    response_time = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Response time: {response_time:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Extract AI reply
        content = data['choices'][0]['message']['content']
        print(f"\nAI Reply: {content}")
        
        # Check token usage
        usage = data.get('usage', {})
        print(f"\nToken Usage:")
        print(f"  Input: {usage.get('prompt_tokens', 0)}")
        print(f"  Output: {usage.get('completion_tokens', 0)}")
        print(f"  Total: {usage.get('total_tokens', 0)}")
        
        return True
    else:
        print(f"Request failed: {response.text}")
        return False


def test_text_completion():
    """测试文本补全"""
    url = "http://127.0.0.1:8001/v1/completions"
    
    payload = {
        "model": "qwen-0.5b",
        "prompt": "The future of artificial intelligence is",
        "max_tokens": 50,
        "temperature": 0.5
    }
    
    print("\nTesting text completion...")
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    response_time = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Response time: {response_time:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract completion
        text = data['choices'][0]['text']
        print(f"\nCompletion: {text}")
        return True
    else:
        print(f"Request failed: {response.text}")
        return False


def main():
    """Main test function"""
    print("=== Inference Fix Verification ===")
    
    # Check service status
    health_response = requests.get("http://127.0.0.1:8001/health")
    health_data = health_response.json()
    print(f"Service status: {health_data.get('status')}")
    print(f"Loaded models: {health_data.get('models', 0)}")
    
    # Run tests
    tests = [
        ("Chat Completion", test_chat_completion),
        ("Text Completion", test_text_completion),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Test: {test_name}")
        print('='*40)
        
        try:
            if test_func():
                print(f"PASS: {test_name}")
                passed += 1
            else:
                print(f"FAIL: {test_name}")
        except Exception as e:
            print(f"ERROR: {test_name} - {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{len(tests)}")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)