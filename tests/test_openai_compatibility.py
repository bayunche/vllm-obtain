#!/usr/bin/env python3
"""
测试OpenAI API完全兼容性
使用真实的OpenAI Python客户端库进行测试
"""

import asyncio
import json
import base64
import sys
import os
import time
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import requests


# 配置测试客户端（指向本地服务）
TEST_BASE_URL = "http://localhost:8001/v1"
TEST_API_KEY = "test-key"  # 本地服务可能不需要真实的API密钥


class OpenAICompatibilityTester:
    """OpenAI兼容性测试器"""
    
    def __init__(self, base_url: str = TEST_BASE_URL, api_key: str = TEST_API_KEY):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def test_models_endpoint(self):
        """测试模型列表接口"""
        print("\n=== 测试 GET /v1/models ===")
        
        try:
            # 使用requests直接测试
            response = requests.get(f"{self.base_url}/models", headers=self.headers)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"返回格式正确: {data.get('object') == 'list'}")
                print(f"模型数量: {len(data.get('data', []))}")
                
                # 检查是否包含OpenAI标准模型
                model_ids = [model['id'] for model in data.get('data', [])]
                openai_models = ['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003']
                has_openai_models = any(model in model_ids for model in openai_models)
                print(f"包含OpenAI模型: {has_openai_models}")
                
                print("✅ 模型接口测试通过")
                return True
            else:
                print(f"❌ 模型接口测试失败: {response.status_code}")
                print(response.text)
                return False
                
        except Exception as e:
            print(f"❌ 模型接口测试异常: {e}")
            return False
    
    def test_chat_completions(self):
        """测试聊天补全接口"""
        print("\n=== 测试 POST /v1/chat/completions ===")
        
        test_cases = [
            # 基础聊天测试
            {
                "name": "基础聊天",
                "data": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "Hello!"}
                    ]
                }
            },
            
            # 系统消息测试
            {
                "name": "系统消息",
                "data": {
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is 2+2?"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            },
            
            # 多轮对话测试
            {
                "name": "多轮对话",
                "data": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello! How can I help you?"},
                        {"role": "user", "content": "Tell me a joke"}
                    ],
                    "temperature": 0.9
                }
            },
            
            # Base64内容测试
            {
                "name": "Base64内容",
                "data": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": base64.b64encode("Hello, world!".encode()).decode()}
                    ]
                }
            }
        ]
        
        success_count = 0
        
        for case in test_cases:
            print(f"\n  测试: {case['name']}")
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={**self.headers, "Content-Type": "application/json"},
                    json=case['data']
                )
                
                print(f"    状态码: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 验证响应格式
                    required_fields = ['id', 'object', 'created', 'model', 'choices', 'usage']
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        print(f"    ✅ 响应格式正确")
                        print(f"    响应内容: {data['choices'][0]['message']['content'][:50]}...")
                        success_count += 1
                    else:
                        print(f"    ❌ 缺少字段: {missing_fields}")
                else:
                    print(f"    ❌ 请求失败: {response.text}")
                    
            except Exception as e:
                print(f"    ❌ 异常: {e}")
        
        print(f"\n聊天补全测试: {success_count}/{len(test_cases)} 通过")
        return success_count == len(test_cases)
    
    def test_completions(self):
        """测试文本补全接口"""
        print("\n=== 测试 POST /v1/completions ===")
        
        test_cases = [
            # 基础补全测试
            {
                "name": "基础补全",
                "data": {
                    "model": "text-davinci-003",
                    "prompt": "The quick brown fox",
                    "max_tokens": 50
                }
            },
            
            # 多个prompt测试
            {
                "name": "多个prompt",
                "data": {
                    "model": "text-davinci-002",
                    "prompt": ["Hello", "World"],
                    "max_tokens": 30
                }
            },
            
            # 参数测试
            {
                "name": "详细参数",
                "data": {
                    "model": "text-davinci-003",
                    "prompt": "Explain quantum computing in simple terms:",
                    "max_tokens": 100,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.1,
                    "stop": ["\n\n"]
                }
            }
        ]
        
        success_count = 0
        
        for case in test_cases:
            print(f"\n  测试: {case['name']}")
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers={**self.headers, "Content-Type": "application/json"},
                    json=case['data']
                )
                
                print(f"    状态码: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 验证响应格式
                    required_fields = ['id', 'object', 'created', 'model', 'choices', 'usage']
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        print(f"    ✅ 响应格式正确")
                        print(f"    响应内容: {data['choices'][0]['text'][:50]}...")
                        success_count += 1
                    else:
                        print(f"    ❌ 缺少字段: {missing_fields}")
                else:
                    print(f"    ❌ 请求失败: {response.text}")
                    
            except Exception as e:
                print(f"    ❌ 异常: {e}")
        
        print(f"\n文本补全测试: {success_count}/{len(test_cases)} 通过")
        return success_count == len(test_cases)
    
    def test_streaming(self):
        """测试流式响应"""
        print("\n=== 测试流式响应 ===")
        
        try:
            # 测试流式聊天
            print("  测试流式聊天补全...")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={**self.headers, "Content-Type": "application/json"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Count to 5"}],
                    "stream": True
                },
                stream=True
            )
            
            print(f"    状态码: {response.status_code}")
            
            if response.status_code == 200:
                chunks_received = 0
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data_part = line_text[6:]  # 移除 'data: ' 前缀
                            
                            if data_part == '[DONE]':
                                break
                                
                            try:
                                chunk_data = json.loads(data_part)
                                if 'choices' in chunk_data:
                                    chunks_received += 1
                                    if chunks_received <= 3:  # 只显示前3个chunk
                                        content = chunk_data['choices'][0].get('delta', {}).get('content', '')
                                        print(f"    Chunk {chunks_received}: {content}")
                            except json.JSONDecodeError:
                                pass
                
                print(f"    ✅ 收到 {chunks_received} 个数据块")
                return True
            else:
                print(f"    ❌ 流式请求失败: {response.text}")
                return False
                
        except Exception as e:
            print(f"    ❌ 流式测试异常: {e}")
            return False
    
    def test_multimodal_content(self):
        """测试多模态内容"""
        print("\n=== 测试多模态内容 ===")
        
        # 创建一个简单的测试图片（1x1像素的红色PNG）
        test_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x00\xad\xad\xacI\x00\x00\x00\x00IEND\xaeB`\x82'
        test_image_b64 = base64.b64encode(test_image_bytes).decode()
        
        try:
            # 测试OpenAI Vision API格式
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={**self.headers, "Content-Type": "application/json"},
                json={
                    "model": "gpt-4-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What's in this image?"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{test_image_b64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ]
                }
            )
            
            print(f"  状态码: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ 多模态内容处理成功")
                print(f"  响应: {data['choices'][0]['message']['content'][:100]}...")
                return True
            else:
                print(f"  ❌ 多模态请求失败: {response.text}")
                return False
                
        except Exception as e:
            print(f"  ❌ 多模态测试异常: {e}")
            return False
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n=== 测试错误处理 ===")
        
        error_cases = [
            # 缺少model参数
            {
                "name": "缺少model参数",
                "data": {"messages": [{"role": "user", "content": "test"}]},
                "expected_status": 400
            },
            
            # 缺少messages参数
            {
                "name": "缺少messages参数",
                "data": {"model": "gpt-3.5-turbo"},
                "expected_status": 400
            },
            
            # 无效的temperature
            {
                "name": "无效的temperature",
                "data": {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "test"}],
                    "temperature": 3.0  # 超出范围
                },
                "expected_status": 400
            },
            
            # 不存在的模型
            {
                "name": "不存在的模型",
                "data": {
                    "model": "nonexistent-model",
                    "messages": [{"role": "user", "content": "test"}]
                },
                "expected_status": 404
            }
        ]
        
        success_count = 0
        
        for case in error_cases:
            print(f"\n  测试: {case['name']}")
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={**self.headers, "Content-Type": "application/json"},
                    json=case['data']
                )
                
                print(f"    状态码: {response.status_code}")
                expected = case.get('expected_status')
                
                if expected and response.status_code == expected:
                    # 检查错误响应格式
                    if response.status_code >= 400:
                        try:
                            error_data = response.json()
                            if 'error' in error_data:
                                print(f"    ✅ 错误格式正确: {error_data['error']['message']}")
                                success_count += 1
                            else:
                                print(f"    ❌ 错误格式不正确")
                        except json.JSONDecodeError:
                            print(f"    ❌ 错误响应不是JSON格式")
                    else:
                        success_count += 1
                elif not expected:  # 没有指定期望状态码，只要不是500就算成功
                    if response.status_code < 500:
                        success_count += 1
                        print(f"    ✅ 错误处理正确")
                    else:
                        print(f"    ❌ 服务器内部错误")
                else:
                    print(f"    ❌ 状态码不匹配，期望: {expected}")
                    
            except Exception as e:
                print(f"    ❌ 异常: {e}")
        
        print(f"\n错误处理测试: {success_count}/{len(error_cases)} 通过")
        return success_count == len(error_cases)
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("开始OpenAI API兼容性测试")
        print(f"测试目标: {self.base_url}")
        print("=" * 60)
        
        tests = [
            ("模型接口", self.test_models_endpoint),
            ("聊天补全", self.test_chat_completions),
            ("文本补全", self.test_completions),
            ("流式响应", self.test_streaming),
            ("多模态内容", self.test_multimodal_content),
            ("错误处理", self.test_error_handling)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"\n❌ {test_name}测试出现异常: {e}")
                results.append((test_name, False))
        
        # 汇总结果
        print("\n" + "=" * 60)
        print("测试结果汇总:")
        print("=" * 60)
        
        passed = 0
        for test_name, result in results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name:15} {status}")
            if result:
                passed += 1
        
        print("\n" + "=" * 60)
        print(f"总计: {passed}/{len(results)} 项测试通过")
        
        if passed == len(results):
            print("🎉 所有测试通过！API完全兼容OpenAI规范")
        else:
            print(f"⚠️  {len(results) - passed} 项测试失败，需要修复")
        
        print("=" * 60)
        
        return passed == len(results)


def check_service_status(base_url: str = TEST_BASE_URL):
    """检查服务状态"""
    try:
        response = requests.get(f"{base_url.replace('/v1', '')}/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False


def main():
    """主函数"""
    # 检查服务是否运行
    if not check_service_status():
        print(f"❌ 服务未运行或无法访问: {TEST_BASE_URL}")
        print("请确保服务已启动并监听在 http://localhost:8001")
        sys.exit(1)
    
    # 运行测试
    tester = OpenAICompatibilityTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()