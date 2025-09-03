"""
消息处理工具
处理base64编码的消息内容和token计算
支持多模态内容（文本、图片、音频、视频）
"""

import base64
import json
import re
from typing import Dict, Any, List, Union, Optional
from loguru import logger
from .multimodal_processor import (
    process_multimodal_content,
    format_multimodal_for_inference,
    extract_text_from_multimodal,
    MultiModalContent
)


def is_base64(s: str) -> bool:
    """
    检查字符串是否为有效的base64编码
    
    Args:
        s: 要检查的字符串
        
    Returns:
        是否为base64编码
    """
    if not s or len(s) < 4:
        return False
    
    # Base64字符集正则
    base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
    
    # 长度必须是4的倍数
    if len(s) % 4 != 0:
        return False
    
    # 检查是否符合base64格式
    if not base64_pattern.match(s):
        return False
    
    # 尝试解码验证
    try:
        decoded = base64.b64decode(s, validate=True)
        # 检查是否为有效的UTF-8文本
        decoded.decode('utf-8')
        return True
    except Exception:
        return False


def decode_base64_content(content: str) -> str:
    """
    解码base64内容
    
    Args:
        content: 可能是base64编码的内容
        
    Returns:
        解码后的内容
    """
    if not content:
        return content
    
    # 检查是否为base64
    if is_base64(content):
        try:
            decoded = base64.b64decode(content).decode('utf-8')
            logger.debug(f"成功解码base64内容，原长度: {len(content)}, 解码后长度: {len(decoded)}")
            return decoded
        except Exception as e:
            logger.warning(f"Base64解码失败: {e}")
            return content
    
    return content


def estimate_tokens(text: str, model_name: str = None) -> int:
    """
    估算文本的token数量
    使用简单的估算方法：
    - 英文：约4个字符=1个token
    - 中文：约2个字符=1个token
    
    Args:
        text: 要估算的文本
        model_name: 模型名称（用于更精确的估算）
        
    Returns:
        估算的token数量
    """
    if not text:
        return 0
    
    # 简单估算：混合文本平均3个字符=1个token
    # 这是一个保守的估算，实际可能更少
    estimated = len(text) // 3
    
    # 对中文字符进行更准确的估算
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars > 0:
        # 中文字符通常2个字符=1个token
        non_chinese = len(text) - chinese_chars
        estimated = (chinese_chars // 2) + (non_chinese // 4)
    
    return max(1, estimated)  # 至少返回1个token


def truncate_to_token_limit(text: str, max_tokens: int, model_name: str = None) -> str:
    """
    将文本截断到指定的token限制
    
    Args:
        text: 要截断的文本
        max_tokens: 最大token数
        model_name: 模型名称
        
    Returns:
        截断后的文本
    """
    current_tokens = estimate_tokens(text, model_name)
    
    if current_tokens <= max_tokens:
        return text
    
    # 计算需要保留的字符比例
    ratio = max_tokens / current_tokens
    target_length = int(len(text) * ratio * 0.9)  # 留10%的余量
    
    # 截断文本
    truncated = text[:target_length]
    
    # 尝试在句子边界截断
    last_period = truncated.rfind('。')
    last_period_en = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    # 找到最后一个合适的截断点
    cut_point = max(last_period, last_period_en, last_newline)
    if cut_point > target_length * 0.8:  # 如果找到的截断点不会损失太多内容
        truncated = truncated[:cut_point + 1]
    
    logger.info(f"文本已截断: 原始{len(text)}字符/{current_tokens}tokens -> {len(truncated)}字符/{estimate_tokens(truncated, model_name)}tokens")
    
    return truncated


def process_message_content(content: Union[str, Dict, List], max_tokens: Optional[int] = None, model_name: Optional[str] = None) -> str:
    """
    处理消息内容，支持多种格式和多模态内容
    兼容OpenAI的content格式：
    1. 字符串（可能是base64编码的文本或媒体）
    2. base64编码的字符串
    3. 对象数组格式 [{type: "text", text: "..."}, {type: "image_url", image_url: {...}}]
    4. 支持音频、视频等多媒体内容
    
    Args:
        content: 消息内容
        max_tokens: 最大token限制
        model_name: 模型名称（用于特定格式化）
        
    Returns:
        处理后的文本内容（包含多模态占位符）
    """
    # 使用多模态处理器处理内容
    multimodal_contents = process_multimodal_content(content)
    
    # 格式化为推理引擎可理解的格式
    result = format_multimodal_for_inference(multimodal_contents, model_name)
    
    # 如果结果为空，尝试传统处理方式
    if not result:
        # 处理字符串内容
        if isinstance(content, str):
            result = decode_base64_content(content)
        # 处理其他格式
        else:
            result = json.dumps(content, ensure_ascii=False)
    
    # 如果指定了最大token数，进行截断
    if max_tokens and result:
        result = truncate_to_token_limit(result, max_tokens, model_name)
    
    # 记录多模态内容信息
    media_types = [c.content_type for c in multimodal_contents if c.content_type != 'text']
    if media_types:
        logger.info(f"检测到多模态内容: {set(media_types)}")
    
    return result


def process_messages(messages: List[Dict[str, Any]], max_context_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    处理消息列表，解码base64内容并控制总token数
    
    Args:
        messages: 消息列表
        max_context_tokens: 最大上下文token数
        
    Returns:
        处理后的消息列表
    """
    if not messages:
        return messages
    
    processed_messages = []
    total_tokens = 0
    
    # 预留一些token给输出
    if max_context_tokens:
        context_limit = int(max_context_tokens * 0.8)  # 使用80%作为输入限制
    else:
        context_limit = None
    
    # 从最新的消息开始处理（保留最近的对话）
    for message in reversed(messages):
        role = message.get("role", "user")
        content = message.get("content", "")
        
        # 处理content（可能是base64）
        processed_content = process_message_content(content)
        
        # 估算token数
        message_tokens = estimate_tokens(processed_content) + 10  # 加10个token作为角色等元数据
        
        # 检查是否超过限制
        if context_limit and total_tokens + message_tokens > context_limit:
            # 如果这是第一条消息，至少要包含它（但可能需要截断）
            if not processed_messages:
                remaining_tokens = context_limit - total_tokens
                if remaining_tokens > 100:  # 至少保留100个token的内容
                    processed_content = truncate_to_token_limit(
                        processed_content, 
                        remaining_tokens - 10
                    )
                    processed_messages.append({
                        "role": role,
                        "content": processed_content
                    })
            logger.info(f"达到上下文限制，截断对话历史。保留{len(processed_messages)}条消息")
            break
        
        processed_messages.append({
            "role": role,
            "content": processed_content
        })
        total_tokens += message_tokens
    
    # 反转回原始顺序
    processed_messages.reverse()
    
    logger.debug(f"处理了{len(processed_messages)}条消息，估算总tokens: {total_tokens}")
    
    return processed_messages


def validate_and_process_messages(messages: List[Dict[str, Any]], config: Any) -> List[Dict[str, Any]]:
    """
    验证和处理消息，包括base64解码和token限制
    
    Args:
        messages: 原始消息列表
        config: 配置对象
        
    Returns:
        处理后的消息列表
    """
    if not messages:
        raise ValueError("消息列表不能为空")
    
    # 获取最大上下文长度
    max_context = getattr(config, 'max_sequence_length', 128000)
    
    # 处理消息
    processed = process_messages(messages, max_context)
    
    if not processed:
        raise ValueError("处理后的消息列表为空，可能是因为消息过长")
    
    return processed