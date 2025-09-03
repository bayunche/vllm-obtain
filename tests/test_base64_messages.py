#!/usr/bin/env python3
"""
测试base64消息处理和token限制功能
"""

import base64
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.message_processor import (
    is_base64,
    decode_base64_content,
    estimate_tokens,
    truncate_to_token_limit,
    process_message_content,
    process_messages,
    validate_and_process_messages
)


def test_base64_detection():
    """测试base64检测功能"""
    print("\n=== 测试Base64检测 ===")
    
    # 测试用例
    test_cases = [
        ("SGVsbG8gV29ybGQh", True, "Hello World!"),  # 有效的base64
        ("5L2g5aW977yM5LiW55WM", True, "你好，世界"),  # 中文base64
        ("Hello World!", False, "Hello World!"),  # 普通文本
        ("SGVsbG8gV29ybGQ", True, "Hello World"),  # 没有padding的base64
        ("!!!invalid!!!", False, "!!!invalid!!!"),  # 无效字符
        ("", False, ""),  # 空字符串
    ]
    
    for encoded, expected_is_base64, expected_decoded in test_cases:
        is_b64 = is_base64(encoded)
        decoded = decode_base64_content(encoded)
        
        print(f"输入: '{encoded[:20]}{'...' if len(encoded) > 20 else ''}'")
        print(f"  是否base64: {is_b64} (期望: {expected_is_base64})")
        print(f"  解码结果: '{decoded}'")
        
        assert is_b64 == expected_is_base64, f"Base64检测失败: {encoded}"
        if expected_is_base64:
            assert decoded == expected_decoded, f"解码结果不匹配: {decoded} != {expected_decoded}"
    
    print("✅ Base64检测测试通过")


def test_token_estimation():
    """测试token估算功能"""
    print("\n=== 测试Token估算 ===")
    
    test_cases = [
        ("Hello World", 3),  # 英文
        ("你好世界", 2),  # 中文
        ("Hello 你好", 3),  # 混合
        ("", 0),  # 空文本
        ("a" * 100, 25),  # 长英文文本
        ("中" * 100, 50),  # 长中文文本
    ]
    
    for text, expected_min in test_cases:
        tokens = estimate_tokens(text)
        print(f"文本: '{text[:30]}{'...' if len(text) > 30 else ''}' ({len(text)}字符)")
        print(f"  估算tokens: {tokens} (期望最少: {expected_min})")
        assert tokens >= expected_min, f"Token估算过低: {tokens} < {expected_min}"
    
    print("✅ Token估算测试通过")


def test_text_truncation():
    """测试文本截断功能"""
    print("\n=== 测试文本截断 ===")
    
    # 生成长文本
    long_text_en = "Hello World. " * 1000  # 约3000 tokens
    long_text_cn = "你好世界。" * 1000  # 约2000 tokens
    
    # 测试英文截断
    truncated_en = truncate_to_token_limit(long_text_en, 100)
    tokens_en = estimate_tokens(truncated_en)
    print(f"英文文本截断: {len(long_text_en)}字符 -> {len(truncated_en)}字符")
    print(f"  Token数: 约{estimate_tokens(long_text_en)} -> {tokens_en}")
    assert tokens_en <= 100, f"截断后token超限: {tokens_en} > 100"
    
    # 测试中文截断
    truncated_cn = truncate_to_token_limit(long_text_cn, 100)
    tokens_cn = estimate_tokens(truncated_cn)
    print(f"中文文本截断: {len(long_text_cn)}字符 -> {len(truncated_cn)}字符")
    print(f"  Token数: 约{estimate_tokens(long_text_cn)} -> {tokens_cn}")
    assert tokens_cn <= 100, f"截断后token超限: {tokens_cn} > 100"
    
    print("✅ 文本截断测试通过")


def test_message_content_processing():
    """测试消息内容处理"""
    print("\n=== 测试消息内容处理 ===")
    
    # 测试字符串内容
    text = "Hello World!"
    text_b64 = base64.b64encode(text.encode()).decode()
    
    result1 = process_message_content(text)
    assert result1 == text, "普通文本处理失败"
    print(f"✓ 普通文本: '{text}' -> '{result1}'")
    
    result2 = process_message_content(text_b64)
    assert result2 == text, "Base64文本处理失败"
    print(f"✓ Base64文本: '{text_b64}' -> '{result2}'")
    
    # 测试数组格式（OpenAI多模态格式）
    content_array = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": base64.b64encode("World".encode()).decode()},
        {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
    ]
    result3 = process_message_content(content_array)
    assert "Hello" in result3 and "World" in result3 and "[图像内容]" in result3
    print(f"✓ 数组格式处理成功: {result3}")
    
    # 测试字典格式
    content_dict = {"text": text_b64}
    result4 = process_message_content(content_dict)
    assert result4 == text, "字典格式处理失败"
    print(f"✓ 字典格式: {content_dict} -> '{result4}'")
    
    # 测试token限制
    long_text = "Hello " * 1000
    result5 = process_message_content(long_text, max_tokens=50)
    tokens = estimate_tokens(result5)
    assert tokens <= 50, f"Token限制失败: {tokens} > 50"
    print(f"✓ Token限制: {len(long_text)}字符 -> {len(result5)}字符 ({tokens} tokens)")
    
    print("✅ 消息内容处理测试通过")


def test_message_list_processing():
    """测试消息列表处理"""
    print("\n=== 测试消息列表处理 ===")
    
    # 创建测试消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": base64.b64encode("Hello!".encode()).decode()},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    
    # 测试无限制处理
    result1 = process_messages(messages)
    assert len(result1) == len(messages), "消息数量不匹配"
    assert result1[1]["content"] == "Hello!", "Base64解码失败"
    print(f"✓ 无限制处理: {len(messages)}条消息全部保留")
    
    # 测试token限制（设置很小的限制）
    result2 = process_messages(messages, max_context_tokens=50)
    assert len(result2) <= len(messages), "消息未被截断"
    print(f"✓ Token限制处理: {len(messages)}条消息 -> {len(result2)}条消息")
    
    # 测试超长消息列表
    long_messages = []
    for i in range(100):
        long_messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}: " + "x" * 100
        })
    
    result3 = process_messages(long_messages, max_context_tokens=1000)
    assert len(result3) < len(long_messages), "长消息列表未被截断"
    print(f"✓ 长消息列表处理: {len(long_messages)}条消息 -> {len(result3)}条消息")
    
    print("✅ 消息列表处理测试通过")


def test_integration():
    """集成测试"""
    print("\n=== 集成测试 ===")
    
    # 模拟配置对象
    class MockConfig:
        max_sequence_length = 1000
    
    config = MockConfig()
    
    # 测试正常消息
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": base64.b64encode("What is 2+2?".encode()).decode()},
        {"role": "assistant", "content": "2+2 equals 4"},
        {"role": "user", "content": "Thanks!"}
    ]
    
    try:
        result = validate_and_process_messages(messages, config)
        assert len(result) > 0, "处理结果为空"
        assert result[1]["content"] == "What is 2+2?", "Base64解码失败"
        print(f"✓ 正常消息处理成功: {len(result)}条消息")
    except Exception as e:
        print(f"✗ 正常消息处理失败: {e}")
        raise
    
    # 测试空消息
    try:
        result = validate_and_process_messages([], config)
        assert False, "空消息应该抛出异常"
    except ValueError as e:
        print(f"✓ 空消息正确抛出异常: {e}")
    
    # 测试超长消息
    huge_message = [{"role": "user", "content": "x" * 1000000}]
    try:
        result = validate_and_process_messages(huge_message, config)
        content_length = len(result[0]["content"])
        assert content_length < 1000000, "消息未被截断"
        print(f"✓ 超长消息被正确截断: 1000000字符 -> {content_length}字符")
    except Exception as e:
        print(f"✗ 超长消息处理失败: {e}")
        raise
    
    print("✅ 集成测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试Base64消息处理和Token限制功能")
    print("=" * 60)
    
    try:
        test_base64_detection()
        test_token_estimation()
        test_text_truncation()
        test_message_content_processing()
        test_message_list_processing()
        test_integration()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
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