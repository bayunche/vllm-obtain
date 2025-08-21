#!/usr/bin/env python3
"""
简化推理测试
"""

import requests
import json
import time


def test_inference():
    """测试推理功能"""
    
    # 聊天补全测试
    print("测试聊天补全...")
    
    url = "http://127.0.0.1:8001/v1/chat/completions"
    payload = {
        "model": "qwen-0.5b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        response_time = time.time() - start_time
        
        print(f"状态码: {response.status_code}")
        print(f"响应时间: {response_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            
            print(f"AI回复: {content}")
            print(f"Token使用: {usage.get('total_tokens', 0)}")
            return True
        else:
            print(f"错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"异常: {e}")
        return False


def main():
    # 检查健康状态
    print("检查服务状态...")
    health = requests.get("http://127.0.0.1:8001/health")
    health_data = health.json()
    print(f"服务状态: {health_data.get('status')}")
    print(f"模型数量: {health_data.get('models', 0)}")
    
    # 测试推理
    if test_inference():
        print("推理测试成功!")
        return True
    else:
        print("推理测试失败!")
        return False


if __name__ == "__main__":
    main()