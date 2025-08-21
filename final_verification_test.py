#!/usr/bin/env python3
"""
全功能验证测试 - Bug修复后的综合测试
"""

import requests
import json
import time
import sys


def test_system_health():
    """测试系统健康状态"""
    print("=== 系统健康检查 ===")
    
    try:
        response = requests.get("http://127.0.0.1:8001/health")
        health_data = response.json()
        
        print(f"服务状态: {health_data.get('status')}")
        print(f"已加载模型: {health_data.get('models', 0)}")
        print(f"响应时间: {time.time() - time.time():.3f}s")
        
        return health_data.get('status') == 'healthy'
        
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False


def test_model_management():
    """测试模型管理功能"""
    print("\n=== 模型管理测试 ===")
    
    model_config = {
        "model_name": "qwen-0.5b",
        "model_path": "./models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf"
    }
    
    # 1. 尝试加载模型
    print("1. 加载模型...")
    try:
        response = requests.post(
            "http://127.0.0.1:8001/v1/models/load",
            json=model_config,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✓ 模型加载成功")
            load_success = True
        else:
            print(f"✗ 模型加载失败: {response.text}")
            load_success = False
            
    except Exception as e:
        print(f"✗ 模型加载异常: {e}")
        load_success = False
    
    if not load_success:
        return False
    
    # 2. 检查模型状态
    print("2. 检查模型状态...")
    try:
        response = requests.get("http://127.0.0.1:8001/v1/models/qwen-0.5b/status")
        if response.status_code == 200:
            print("✓ 模型状态查询成功")
        else:
            print(f"✗ 模型状态查询失败: {response.status_code}")
            
    except Exception as e:
        print(f"✗ 模型状态查询异常: {e}")
    
    return True


def test_openai_compatibility():
    """测试OpenAI API兼容性"""
    print("\n=== OpenAI API 兼容性测试 ===")
    
    # 1. 模型列表
    print("1. 测试模型列表...")
    try:
        response = requests.get("http://127.0.0.1:8001/v1/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 模型列表获取成功，共 {len(data['data'])} 个模型")
        else:
            print(f"✗ 模型列表获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 模型列表获取异常: {e}")
        return False
    
    # 2. 聊天补全 (修复后的核心功能)
    print("2. 测试聊天补全...")
    try:
        payload = {
            "model": "qwen-0.5b",
            "messages": [
                {"role": "user", "content": "Hello! Say hi and introduce yourself in one sentence."}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        start_time = time.time()
        response = requests.post(
            "http://127.0.0.1:8001/v1/chat/completions",
            json=payload,
            timeout=30
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"✓ 聊天补全成功 ({response_time:.2f}s)")
            print(f"  AI回复: {content[:80]}...")
            return True
        else:
            print(f"✗ 聊天补全失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 聊天补全异常: {e}")
        return False


def test_model_unload():
    """测试模型卸载功能"""
    print("\n=== 模型卸载测试 ===")
    
    # 测试修复后的卸载API
    print("测试 POST /v1/models/unload...")
    try:
        response = requests.post(
            "http://127.0.0.1:8001/v1/models/unload",
            json={"model_name": "qwen-0.5b"},
            timeout=10
        )
        
        if response.status_code == 200:
            print("✓ 模型卸载成功")
            return True
        else:
            print(f"✗ 模型卸载失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 模型卸载异常: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("VLLM Inference Framework - Bug Fix Verification")
    print("=" * 60)
    
    # 测试项目
    tests = [
        ("系统健康检查", test_system_health),
        ("模型管理", test_model_management),
        ("OpenAI API兼容性", test_openai_compatibility),
        ("模型卸载", test_model_unload),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"PASS: {test_name}")
                passed += 1
            else:
                print(f"FAIL: {test_name}")
        except Exception as e:
            print(f"ERROR: {test_name} - {e}")
    
    # 结果总结
    print(f"\n{'='*60}")
    print("Final Test Results")
    print(f"{'='*60}")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("All tests passed! Bug fixes successful!")
        grade = "A+"
    elif passed >= total * 0.8:
        print("Most functions working well, good fix results!")
        grade = "A"
    elif passed >= total * 0.6:
        print("Basic functions working, room for improvement.")
        grade = "B"
    else:
        print("Some core functions still have issues.")
        grade = "C"
    
    print(f"Fix Quality Grade: {grade}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)