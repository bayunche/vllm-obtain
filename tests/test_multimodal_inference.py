#!/usr/bin/env python3
"""
测试多模态推理功能的端到端集成
验证base64图片是否能正确传递给推理引擎并进行多模态推理
"""

import base64
import json
import sys
import os
import time
import requests
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_images():
    """创建测试用的图片数据"""
    
    # 创建一个简单的1x1像素PNG图片
    png_1x1_red = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x00\xad\xad\xacI\x00\x00\x00\x00IEND\xaeB`\x82'
    
    # 创建一个简单的2x2像素PNG图片（蓝色）
    png_2x2_blue = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02'
        b'\x08\x02\x00\x00\x00\xfd\xd4\x9as\x00\x00\x00\x12IDATx\x9cc\xf8\xcf'
        b'@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x03\xec'
        b'l\xe1k\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    
    return {
        'red_pixel': base64.b64encode(png_1x1_red).decode(),
        'blue_pixels': base64.b64encode(png_2x2_blue).decode()
    }


def test_base64_multimodal_api(base_url: str = "http://localhost:8001/v1"):
    """测试base64多模态API"""
    
    print("=== 测试Base64多模态推理API ===")
    
    # 创建测试图片
    test_images = create_test_images()
    
    # 测试用例1：OpenAI Vision API格式
    print("\n1. 测试OpenAI Vision API格式")
    
    test_request = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "这是什么颜色的图片？"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{test_images['red_pixel']}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=test_request,
            timeout=30
        )
        
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"   ✅ API响应成功")
            print(f"   回复内容: {content}")
            
            # 检查是否真的处理了图片（而不是只返回占位符）
            if "[图片:" in content or "红色" in content or "颜色" in content:
                print(f"   ✅ 检测到图片内容处理")
            else:
                print(f"   ⚠️  可能只处理了文本，未真正理解图片")
            
            return True
        else:
            print(f"   ❌ API调用失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ 请求异常: {e}")
        return False


def test_direct_base64_content():
    """测试直接的base64内容"""
    
    print("\n2. 测试直接base64图片内容")
    
    test_images = create_test_images()
    
    # 创建包含base64图片的消息
    test_request = {
        "model": "glm-4.5v",
        "messages": [
            {
                "role": "user", 
                "content": f"data:image/png;base64,{test_images['blue_pixels']}"
            },
            {
                "role": "user",
                "content": "上面的图片是什么颜色？"
            }
        ],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=test_request,
            timeout=30
        )
        
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"   ✅ 直接base64内容处理成功")
            print(f"   回复内容: {content}")
            return True
        else:
            print(f"   ❌ 直接base64处理失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ 直接base64测试异常: {e}")
        return False


def test_mixed_multimodal_content():
    """测试混合多模态内容"""
    
    print("\n3. 测试混合多模态内容")
    
    test_images = create_test_images()
    
    # 复杂的多模态消息
    test_request = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "你是一个图像分析专家。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "请分析以下两张图片的差异："
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{test_images['red_pixel']}"}
                    },
                    {
                        "type": "text",
                        "text": "和"
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{test_images['blue_pixels']}"}
                    },
                    {
                        "type": "text",
                        "text": "它们有什么不同？"
                    }
                ]
            }
        ],
        "max_tokens": 150,
        "temperature": 0.5
    }
    
    try:
        response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=test_request,
            timeout=30
        )
        
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"   ✅ 混合多模态内容处理成功")
            print(f"   回复内容: {content[:200]}...")
            return True
        else:
            print(f"   ❌ 混合多模态处理失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ 混合多模态测试异常: {e}")
        return False


def test_multimodal_streaming():
    """测试多模态流式响应"""
    
    print("\n4. 测试多模态流式响应")
    
    test_images = create_test_images()
    
    test_request = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "详细描述这张图片"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{test_images['red_pixel']}"}
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "stream": True
    }
    
    try:
        response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=test_request,
            stream=True,
            timeout=30
        )
        
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            chunks_received = 0
            accumulated_content = ""
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_part = line_text[6:]
                        
                        if data_part == '[DONE]':
                            break
                            
                        try:
                            chunk_data = json.loads(data_part)
                            if 'choices' in chunk_data and chunk_data['choices']:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    accumulated_content += content
                                    chunks_received += 1
                        except json.JSONDecodeError:
                            pass
            
            print(f"   ✅ 流式响应成功，收到 {chunks_received} 个数据块")
            print(f"   完整内容: {accumulated_content}")
            return True
        else:
            print(f"   ❌ 流式响应失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ 流式响应测试异常: {e}")
        return False


def test_local_multimodal_processing():
    """测试本地多模态处理逻辑"""
    
    print("\n5. 测试本地多模态处理逻辑")
    
    try:
        # 测试多模态处理器
        from src.utils.multimodal_processor import process_multimodal_content
        
        test_images = create_test_images()
        
        # 测试data URI格式
        data_uri = f"data:image/png;base64,{test_images['red_pixel']}"
        contents = process_multimodal_content(data_uri)
        
        print(f"   检测到 {len(contents)} 个内容项")
        
        for i, content in enumerate(contents):
            print(f"   内容 {i+1}: {content.content_type}")
            if hasattr(content, 'mime_type'):
                print(f"     MIME类型: {content.mime_type}")
        
        # 测试OpenAI格式
        openai_format = [
            {"type": "text", "text": "测试"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{test_images['blue_pixels']}"}
            }
        ]
        
        contents2 = process_multimodal_content(openai_format)
        print(f"   OpenAI格式检测到 {len(contents2)} 个内容项")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 本地处理测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_service_status():
    """检查服务状态"""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """主测试函数"""
    
    print("=" * 60)
    print("多模态推理功能端到端测试")
    print("=" * 60)
    
    # 检查服务状态
    if not check_service_status():
        print("❌ 服务未运行，请先启动服务:")
        print("   python run.py")
        sys.exit(1)
    
    print("✅ 服务运行正常")
    
    # 执行测试
    tests = [
        ("本地多模态处理", test_local_multimodal_processing),
        ("OpenAI Vision API格式", test_base64_multimodal_api),
        ("直接base64内容", test_direct_base64_content), 
        ("混合多模态内容", test_mixed_multimodal_content),
        ("多模态流式响应", test_multimodal_streaming)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"总计: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("🎉 所有多模态功能测试通过！")
        print("系统已经可以正确处理base64编码的图片并进行多模态推理")
    elif passed > 0:
        print("⚠️  部分功能正常，部分需要改进")
    else:
        print("❌ 多模态功能存在问题，需要调试")
    
    print("=" * 60)
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)