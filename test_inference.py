#!/usr/bin/env python3
"""
测试推理功能
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
            {"role": "user", "content": "你好！请简单介绍一下自己。"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("发送聊天请求...")
    print(f"请求: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    start_time = time.time()
    response = requests.post(url, json=payload, headers=headers)
    response_time = time.time() - start_time
    
    print(f"响应状态码: {response.status_code}")
    print(f"响应时间: {response_time:.2f}秒")
    
    if response.status_code == 200:
        data = response.json()
        print(f"响应: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        # 提取回复内容
        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0]['message']['content']
            print(f"\n🤖 AI回复: {content}")
            
            # 检查token使用
            usage = data.get('usage', {})
            print(f"\n📊 Token使用:")
            print(f"  输入: {usage.get('prompt_tokens', 0)}")
            print(f"  输出: {usage.get('completion_tokens', 0)}")
            print(f"  总计: {usage.get('total_tokens', 0)}")
            
            return True
    else:
        print(f"请求失败: {response.text}")
        return False


def test_text_completion():
    """测试文本补全"""
    url = "http://127.0.0.1:8001/v1/completions"
    
    payload = {
        "model": "qwen-0.5b",
        "prompt": "人工智能的未来发展趋势是",
        "max_tokens": 50,
        "temperature": 0.5
    }
    
    print("\n发送文本补全请求...")
    
    start_time = time.time()
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    response_time = time.time() - start_time
    
    print(f"响应状态码: {response.status_code}")
    print(f"响应时间: {response_time:.2f}秒")
    
    if response.status_code == 200:
        data = response.json()
        
        # 提取补全内容
        if 'choices' in data and len(data['choices']) > 0:
            text = data['choices'][0]['text']
            print(f"\n📝 补全结果: {text}")
            return True
    else:
        print(f"请求失败: {response.text}")
        return False


def test_models_list():
    """测试模型列表"""
    url = "http://127.0.0.1:8001/v1/models"
    
    response = requests.get(url)
    print(f"\n模型列表状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"可用模型数量: {len(data['data'])}")
        for model in data['data']:
            print(f"  - {model['id']}")
        return True
    else:
        print(f"获取模型列表失败: {response.text}")
        return False


def main():
    """主测试函数"""
    print("=== 推理功能测试 ===")
    
    # 检查服务状态
    health_response = requests.get("http://127.0.0.1:8001/health")
    if health_response.status_code not in [200, 503]:
        print("服务不可用")
        return False
    
    health_data = health_response.json()
    print(f"服务状态: {health_data.get('status')}")
    print(f"已加载模型: {health_data.get('models', 0)}")
    
    # 测试项目
    tests = [
        ("模型列表", test_models_list),
        ("聊天补全", test_chat_completion),
        ("文本补全", test_text_completion),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        print(f"测试: {test_name}")
        print('='*20)
        
        try:
            if test_func():
                print(f"✓ {test_name} 测试通过")
                passed += 1
            else:
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
    
    print(f"\n=== 测试完成 ===")
    print(f"通过率: {passed}/{len(tests)}")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)