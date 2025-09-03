#!/usr/bin/env python3
"""
测试多模态内容处理功能
"""

import base64
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.multimodal_processor import (
    detect_base64_media_type,
    process_multimodal_content,
    format_multimodal_for_inference,
    MultiModalContent
)
from src.utils.message_processor import process_message_content


def create_test_image_base64():
    """创建一个测试用的base64编码的图片（1x1像素的红色PNG）"""
    # 1x1 红色PNG图片的字节数据
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x00\xad\xad\xacI\x00\x00\x00\x00IEND\xaeB`\x82'
    return base64.b64encode(png_bytes).decode()


def create_test_audio_base64():
    """创建一个测试用的base64编码的音频（简单的WAV头）"""
    # 最小的WAV文件头
    wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    return base64.b64encode(wav_header).decode()


def test_media_type_detection():
    """测试媒体类型检测"""
    print("\n=== 测试媒体类型检测 ===")
    
    # 测试图片
    image_b64 = create_test_image_base64()
    is_media, mime_type, data = detect_base64_media_type(image_b64)
    assert is_media, "图片检测失败"
    assert mime_type == 'image/png', f"MIME类型错误: {mime_type}"
    print(f"✓ PNG图片检测: {mime_type}")
    
    # 测试Data URI格式
    data_uri = f"data:image/png;base64,{image_b64}"
    is_media, mime_type, data = detect_base64_media_type(data_uri)
    assert is_media, "Data URI检测失败"
    assert mime_type == 'image/png', f"Data URI MIME类型错误: {mime_type}"
    print(f"✓ Data URI检测: {mime_type}")
    
    # 测试音频
    audio_b64 = create_test_audio_base64()
    is_media, mime_type, data = detect_base64_media_type(audio_b64)
    assert is_media, "音频检测失败"
    assert mime_type == 'audio/wav', f"音频MIME类型错误: {mime_type}"
    print(f"✓ WAV音频检测: {mime_type}")
    
    # 测试普通文本的base64
    text_b64 = base64.b64encode("Hello World".encode()).decode()
    is_media, mime_type, data = detect_base64_media_type(text_b64)
    assert is_media, "Base64文本检测失败"
    assert data.decode('utf-8') == "Hello World", "文本解码失败"
    print(f"✓ Base64文本检测和解码")
    
    print("✅ 媒体类型检测测试通过")


def test_multimodal_content_processing():
    """测试多模态内容处理"""
    print("\n=== 测试多模态内容处理 ===")
    
    image_b64 = create_test_image_base64()
    audio_b64 = create_test_audio_base64()
    
    # 测试OpenAI格式的多模态消息
    test_cases = [
        # 纯文本
        ("Hello World", 1, ['text']),
        
        # Base64编码的文本
        (base64.b64encode("你好世界".encode()).decode(), 1, ['text']),
        
        # Data URI格式的图片
        (f"data:image/png;base64,{image_b64}", 1, ['image']),
        
        # OpenAI的图片URL格式
        ([{
            "type": "text",
            "text": "这是一张图片："
        }, {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}",
                "detail": "high"
            }
        }], 2, ['text', 'image']),
        
        # 混合内容：文本+图片+音频
        ([{
            "type": "text",
            "text": "多模态测试"
        }, {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        }, {
            "type": "audio",
            "audio": {"data": audio_b64}
        }], 3, ['text', 'image', 'audio']),
    ]
    
    for i, (content, expected_count, expected_types) in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        contents = process_multimodal_content(content)
        
        assert len(contents) == expected_count, f"内容数量不匹配: {len(contents)} != {expected_count}"
        
        actual_types = [c.content_type for c in contents]
        assert actual_types == expected_types, f"内容类型不匹配: {actual_types} != {expected_types}"
        
        formatted = format_multimodal_for_inference(contents)
        print(f"  输入类型: {type(content).__name__}")
        print(f"  检测到的内容: {actual_types}")
        print(f"  格式化输出: {formatted[:100]}...")
        
        print(f"  ✓ 测试通过")
    
    print("\n✅ 多模态内容处理测试通过")


def test_message_processing_with_multimodal():
    """测试消息处理器的多模态支持"""
    print("\n=== 测试消息处理器多模态支持 ===")
    
    image_b64 = create_test_image_base64()
    
    # 测试案例1：带图片的消息
    content1 = [{
        "type": "text",
        "text": "请分析这张图片："
    }, {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_b64}"
        }
    }]
    
    result1 = process_message_content(content1)
    assert "请分析这张图片" in result1, "文本内容丢失"
    assert "[图片:" in result1, "图片占位符丢失"
    print(f"✓ 图片消息处理: {result1}")
    
    # 测试案例2：Base64编码的视频
    fake_video_b64 = base64.b64encode(b'\x00\x00\x00\x18ftypmp4\x00' + b'\x00' * 100).decode()
    video_uri = f"data:video/mp4;base64,{fake_video_b64}"
    
    result2 = process_message_content(video_uri)
    assert "[视频:" in result2, "视频占位符丢失"
    print(f"✓ 视频消息处理: {result2}")
    
    # 测试案例3：混合内容with token限制
    long_text = "这是一段很长的文本。" * 100
    content3 = [{
        "type": "text",
        "text": long_text
    }, {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
    }]
    
    result3 = process_message_content(content3, max_tokens=50)
    assert len(result3) < len(long_text), "文本未被截断"
    print(f"✓ Token限制处理: 原始{len(long_text)}字符 -> {len(result3)}字符")
    
    print("\n✅ 消息处理器多模态支持测试通过")


def test_openai_api_compatibility():
    """测试OpenAI API兼容性"""
    print("\n=== 测试OpenAI API兼容性 ===")
    
    image_b64 = create_test_image_base64()
    
    # 模拟OpenAI的Vision API请求格式
    messages = [
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
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]
    
    # 处理消息
    for message in messages:
        content = message.get("content")
        if content:
            processed = process_message_content(content)
            print(f"角色: {message['role']}")
            print(f"处理后内容: {processed}")
            assert "What's in this image?" in processed
            assert "[图片:" in processed
    
    print("✅ OpenAI API兼容性测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试多模态内容处理功能")
    print("=" * 60)
    
    try:
        test_media_type_detection()
        test_multimodal_content_processing()
        test_message_processing_with_multimodal()
        test_openai_api_compatibility()
        
        print("\n" + "=" * 60)
        print("🎉 所有多模态测试通过！")
        print("系统现在支持：")
        print("  ✓ Base64编码的图片解析")
        print("  ✓ Base64编码的音频解析")
        print("  ✓ Base64编码的视频解析")
        print("  ✓ OpenAI Vision API格式兼容")
        print("  ✓ Data URI格式支持")
        print("  ✓ 多模态内容的Token管理")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()