#!/usr/bin/env python3
"""
快速测试多模态API功能
"""

import requests
import base64
import json

def test_multimodal_api():
    """测试多模态API"""
    
    # 创建一个简单的测试图片（1x1像素红色PNG）
    test_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x00\xad\xad\xacI\x00\x00\x00\x00IEND\xaeB`\x82'
    image_b64 = base64.b64encode(test_image).decode()
    
    # API请求
    url = "http://localhost:8001/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # OpenAI Vision API格式的请求
    data = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "描述这张图片"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    print("=== 测试多模态API ===")
    print(f"发送请求到: {url}")
    print(f"包含base64图片数据（长度: {len(image_b64)} 字符）")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"\n状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ 成功！")
            print(f"回复: {content}")
        else:
            print(f"❌ 失败！")
            print(f"错误: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        print("请确保服务已启动: python run.py")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    test_multimodal_api()