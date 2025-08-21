#!/usr/bin/env python3
"""
OpenAI API 兼容性测试脚本
验证服务是否完全兼容 OpenAI API 格式
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# 服务配置
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/v1"


def test_health_check():
    """测试健康检查"""
    print("🏥 测试健康检查...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        print(f"✅ 健康检查通过: {data.get('status', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False


def test_list_models():
    """测试模型列表接口"""
    print("\n📋 测试模型列表...")
    
    try:
        response = requests.get(f"{API_BASE}/models")
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        
        models = data["data"]
        print(f"✅ 发现 {len(models)} 个模型:")
        
        for model in models:
            print(f"   - {model['id']} (创建时间: {model.get('created', 'unknown')})")
        
        return models
        
    except Exception as e:
        print(f"❌ 模型列表测试失败: {e}")
        return []


def test_chat_completion(models):
    """测试聊天补全接口"""
    print("\n💬 测试聊天补全...")
    
    if not models:
        print("⚠️ 没有可用模型，跳过聊天测试")
        return False
    
    model_name = models[0]["id"]
    
    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": "请简单介绍一下人工智能，不超过50字。"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        print(f"使用模型: {model_name}")
        print("发送请求...")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response_time = time.time() - start_time
        
        print(f"响应时间: {response_time:.2f}秒")
        
        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
            return False
        
        data = response.json()
        
        # 验证响应格式
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            assert field in data, f"缺少字段: {field}"
        
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        
        choice = data["choices"][0]
        assert "message" in choice
        assert "role" in choice["message"]
        assert "content" in choice["message"]
        assert choice["message"]["role"] == "assistant"
        
        # 验证使用统计
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        
        content = choice["message"]["content"]
        print(f"✅ 聊天补全成功:")
        print(f"   回复: {content[:100]}...")
        print(f"   Token使用: {usage['prompt_tokens']}+{usage['completion_tokens']}={usage['total_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 聊天补全测试失败: {e}")
        return False


def test_stream_chat(models):
    """测试流式聊天"""
    print("\n🌊 测试流式聊天...")
    
    if not models:
        print("⚠️ 没有可用模型，跳过流式测试")
        return False
    
    model_name = models[0]["id"]
    
    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "请数从1到5"}
            ],
            "max_tokens": 50,
            "stream": True
        }
        
        print(f"使用模型: {model_name}")
        print("开始流式接收...")
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if response.status_code != 200:
            print(f"❌ 流式请求失败: {response.status_code}")
            return False
        
        chunks_received = 0
        content_received = ""
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # 移除 'data: ' 前缀
                    
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        chunks_received += 1
                        
                        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                            choice = chunk_data["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                content_received += content
                                print(content, end="", flush=True)
                    
                    except json.JSONDecodeError:
                        continue
        
        print(f"\n✅ 流式聊天成功:")
        print(f"   接收到 {chunks_received} 个数据块")
        print(f"   完整内容: {content_received}")
        
        return True
        
    except Exception as e:
        print(f"❌ 流式聊天测试失败: {e}")
        return False


def test_text_completion(models):
    """测试文本补全接口"""
    print("\n📝 测试文本补全...")
    
    if not models:
        print("⚠️ 没有可用模型，跳过文本补全测试")
        return False
    
    model_name = models[0]["id"]
    
    try:
        payload = {
            "model": model_name,
            "prompt": "人工智能技术的发展历程包括",
            "max_tokens": 80,
            "temperature": 0.5
        }
        
        response = requests.post(
            f"{API_BASE}/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
            return False
        
        data = response.json()
        
        # 验证响应格式
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            assert field in data, f"缺少字段: {field}"
        
        assert data["object"] == "text_completion"
        
        choice = data["choices"][0]
        assert "text" in choice
        
        text = choice["text"]
        usage = data["usage"]
        
        print(f"✅ 文本补全成功:")
        print(f"   补全文本: {text[:100]}...")
        print(f"   Token使用: {usage['total_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文本补全测试失败: {e}")
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n🚫 测试错误处理...")
    
    try:
        # 测试不存在的模型
        payload = {
            "model": "non-existent-model",
            "messages": [{"role": "user", "content": "test"}]
        }
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            json=payload
        )
        
        # 应该返回错误
        assert response.status_code >= 400
        
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        
        print(f"✅ 错误处理正确: {data['error']['message']}")
        
        # 测试缺少参数
        payload = {"model": "test"}  # 缺少 messages
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            json=payload
        )
        
        assert response.status_code >= 400
        data = response.json()
        assert "error" in data
        
        print(f"✅ 参数验证正确: {data['error']['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False


def test_n8n_compatibility():
    """测试 n8n 兼容性"""
    print("\n🔗 测试 n8n 兼容性...")
    
    try:
        # 模拟 n8n 的典型请求
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "n8n",
            "Authorization": "Bearer test-key"  # n8n 通常会发送这个
        }
        
        payload = {
            "model": "gpt-3.5-turbo",  # n8n 可能使用标准模型名
            "messages": [
                {"role": "user", "content": "Hello from n8n!"}
            ],
            "max_tokens": 50
        }
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            json=payload,
            headers=headers
        )
        
        # 检查是否正确处理（即使模型名不存在也应该有合理的错误）
        if response.status_code == 200 or (response.status_code >= 400 and "error" in response.json()):
            print("✅ n8n 请求格式兼容")
            return True
        else:
            print(f"❌ n8n 兼容性问题: {response.status_code}")
            return False
        
    except Exception as e:
        print(f"❌ n8n 兼容性测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 OpenAI API 兼容性测试")
    print("=" * 50)
    
    # 检查服务是否运行
    try:
        response = requests.get(BASE_URL, timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务: {BASE_URL}")
        print("请确保服务正在运行: python run.py --mode dev")
        sys.exit(1)
    
    # 运行测试
    tests = [
        test_health_check,
        test_list_models,
        test_chat_completion,
        test_stream_chat,
        test_text_completion,
        test_error_handling,
        test_n8n_compatibility
    ]
    
    passed = 0
    total = len(tests)
    models = []
    
    for test_func in tests:
        try:
            if test_func.__name__ == 'test_list_models':
                result = test_func()
                if isinstance(result, list):
                    models = result
                    if result:
                        passed += 1
                elif result:
                    passed += 1
            elif test_func.__name__ in ['test_chat_completion', 'test_stream_chat', 'test_text_completion']:
                if test_func(models):
                    passed += 1
            else:
                if test_func():
                    passed += 1
                    
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！服务完全兼容 OpenAI API")
        print("\n✅ 可以在 n8n 中直接使用:")
        print(f"   Base URL: {API_BASE}")
        print(f"   API Key: 可以留空或任意填写")
        print(f"   模型列表: {[m['id'] for m in models]}")
    else:
        print(f"⚠️ {total - passed} 个测试失败，请检查服务配置")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)