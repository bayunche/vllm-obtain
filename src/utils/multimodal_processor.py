"""
多模态内容处理器
处理base64编码的图片、音频、视频等多媒体内容
完全兼容OpenAI的多模态API格式
"""

import base64
import json
import mimetypes
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
from loguru import logger


class MultiModalContent:
    """多模态内容封装"""
    
    def __init__(self, content_type: str, data: Union[str, bytes], metadata: Dict[str, Any] = None):
        self.content_type = content_type  # text, image, audio, video
        self.data = data
        self.metadata = metadata or {}
        self.is_base64 = False
        self.mime_type = None
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.content_type,
            "data": self.data if not isinstance(self.data, bytes) else base64.b64encode(self.data).decode(),
            "metadata": self.metadata,
            "mime_type": self.mime_type
        }


def detect_base64_media_type(data: str) -> Tuple[bool, Optional[str], Optional[bytes]]:
    """
    检测base64数据的媒体类型
    
    Args:
        data: base64编码的数据或数据URI
        
    Returns:
        (是否为有效的媒体数据, MIME类型, 解码后的二进制数据)
    """
    # 检查是否为数据URI格式 (data:image/png;base64,...)
    data_uri_pattern = r'^data:([^;]+);base64,(.+)$'
    match = re.match(data_uri_pattern, data)
    
    if match:
        mime_type = match.group(1)
        base64_data = match.group(2)
    else:
        # 假设是纯base64数据
        base64_data = data
        mime_type = None
    
    # 尝试解码base64
    try:
        decoded_data = base64.b64decode(base64_data, validate=True)
        
        # 如果没有从URI获取到MIME类型，尝试从内容推断
        if not mime_type:
            mime_type = guess_mime_type_from_bytes(decoded_data)
        
        return True, mime_type, decoded_data
    except Exception:
        return False, None, None


def guess_mime_type_from_bytes(data: bytes) -> Optional[str]:
    """
    从字节数据推断MIME类型
    
    Args:
        data: 二进制数据
        
    Returns:
        推断的MIME类型
    """
    if not data or len(data) < 4:
        return None
    
    # 文件签名（魔术数字）
    signatures = {
        b'\xFF\xD8\xFF': 'image/jpeg',
        b'\x89PNG\r\n\x1a\n': 'image/png',
        b'GIF87a': 'image/gif',
        b'GIF89a': 'image/gif',
        b'RIFF': 'video/webm',  # 可能是WebM或WAV
        b'\x00\x00\x00\x18ftypmp4': 'video/mp4',
        b'\x00\x00\x00\x14ftypM4A': 'audio/mp4',
        b'ID3': 'audio/mpeg',
        b'\xFF\xFB': 'audio/mpeg',
        b'OggS': 'audio/ogg',
        b'fLaC': 'audio/flac',
    }
    
    # 检查文件签名
    for signature, mime_type in signatures.items():
        if data.startswith(signature):
            # 特殊处理RIFF格式
            if signature == b'RIFF' and len(data) > 12:
                if data[8:12] == b'WAVE':
                    return 'audio/wav'
                elif data[8:12] == b'WEBP':
                    return 'image/webp'
            return mime_type
    
    # 检查是否为文本
    try:
        data.decode('utf-8')
        return 'text/plain'
    except UnicodeDecodeError:
        pass
    
    return 'application/octet-stream'


def process_multimodal_content(content: Union[str, Dict, List]) -> List[MultiModalContent]:
    """
    处理多模态内容，支持OpenAI的各种格式
    
    支持的格式：
    1. 纯文本字符串
    2. Base64编码的文本
    3. Base64编码的媒体数据（data URI或纯base64）
    4. 对象格式：{type: "text", text: "..."} 
    5. 对象格式：{type: "image_url", image_url: {url: "data:image/png;base64,..."}}
    6. 对象格式：{type: "audio", audio: {data: "base64..."}}
    7. 对象格式：{type: "video", video: {data: "base64..."}}
    8. 数组格式：[{type: "text", ...}, {type: "image_url", ...}, ...]
    
    Args:
        content: 原始内容
        
    Returns:
        多模态内容列表
    """
    contents = []
    
    # 处理字符串
    if isinstance(content, str):
        # 检查是否为媒体数据
        is_media, mime_type, media_data = detect_base64_media_type(content)
        
        if is_media and mime_type:
            # 根据MIME类型确定内容类型
            if mime_type.startswith('image/'):
                content_obj = MultiModalContent('image', media_data, {
                    'mime_type': mime_type,
                    'format': 'base64'
                })
                content_obj.mime_type = mime_type
                contents.append(content_obj)
            elif mime_type.startswith('audio/'):
                content_obj = MultiModalContent('audio', media_data, {
                    'mime_type': mime_type,
                    'format': 'base64'
                })
                content_obj.mime_type = mime_type
                contents.append(content_obj)
            elif mime_type.startswith('video/'):
                content_obj = MultiModalContent('video', media_data, {
                    'mime_type': mime_type,
                    'format': 'base64'
                })
                content_obj.mime_type = mime_type
                contents.append(content_obj)
            else:
                # 尝试作为base64文本解码
                try:
                    text = media_data.decode('utf-8')
                    contents.append(MultiModalContent('text', text))
                except:
                    # 如果无法解码为文本，作为二进制数据
                    contents.append(MultiModalContent('binary', media_data, {
                        'mime_type': mime_type
                    }))
        else:
            # 普通文本
            contents.append(MultiModalContent('text', content))
    
    # 处理数组
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                content_type = item.get('type', 'text')
                
                if content_type == 'text':
                    text = item.get('text', '')
                    # 递归处理文本（可能是base64）
                    sub_contents = process_multimodal_content(text)
                    contents.extend(sub_contents)
                    
                elif content_type == 'image_url':
                    image_data = item.get('image_url', {})
                    url = image_data.get('url', '')
                    
                    # 处理data URI或base64
                    if url.startswith('data:') or is_valid_base64(url):
                        is_media, mime_type, media_data = detect_base64_media_type(url)
                        if is_media:
                            content_obj = MultiModalContent('image', media_data, {
                                'mime_type': mime_type,
                                'format': 'base64',
                                'detail': image_data.get('detail', 'auto')
                            })
                            content_obj.mime_type = mime_type
                            contents.append(content_obj)
                    else:
                        # URL引用
                        contents.append(MultiModalContent('image', url, {
                            'format': 'url',
                            'detail': image_data.get('detail', 'auto')
                        }))
                
                elif content_type == 'audio':
                    audio_data = item.get('audio', {})
                    data = audio_data.get('data', '')
                    is_media, mime_type, media_data = detect_base64_media_type(data)
                    if is_media:
                        content_obj = MultiModalContent('audio', media_data, {
                            'mime_type': mime_type,
                            'format': 'base64'
                        })
                        content_obj.mime_type = mime_type
                        contents.append(content_obj)
                
                elif content_type == 'video':
                    video_data = item.get('video', {})
                    data = video_data.get('data', '')
                    is_media, mime_type, media_data = detect_base64_media_type(data)
                    if is_media:
                        content_obj = MultiModalContent('video', media_data, {
                            'mime_type': mime_type,
                            'format': 'base64'
                        })
                        content_obj.mime_type = mime_type
                        contents.append(content_obj)
                        
            elif isinstance(item, str):
                # 递归处理字符串项
                sub_contents = process_multimodal_content(item)
                contents.extend(sub_contents)
    
    # 处理字典
    elif isinstance(content, dict):
        # 将字典作为单个项处理
        contents = process_multimodal_content([content])
    
    return contents


def is_valid_base64(s: str) -> bool:
    """检查是否为有效的base64字符串"""
    if not s or len(s) < 4:
        return False
    
    # 移除数据URI前缀（如果有）
    if s.startswith('data:'):
        match = re.match(r'^data:[^;]+;base64,(.+)$', s)
        if match:
            s = match.group(1)
        else:
            return False
    
    # Base64字符集正则
    base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
    
    # 长度必须是4的倍数（加上padding）
    if len(s) % 4 != 0:
        # 尝试添加padding
        s = s + '=' * (4 - len(s) % 4)
    
    return bool(base64_pattern.match(s))


def format_multimodal_for_inference(contents: List[MultiModalContent], model_name: str = None) -> str:
    """
    将多模态内容格式化为推理引擎可以理解的格式
    
    Args:
        contents: 多模态内容列表
        model_name: 模型名称（用于特定模型的格式化）
        
    Returns:
        格式化后的文本提示
    """
    parts = []
    
    for content in contents:
        if content.content_type == 'text':
            parts.append(content.data)
        elif content.content_type == 'image':
            # 为图片添加占位符或描述
            if content.metadata.get('format') == 'url':
                parts.append(f"[图片: {content.data}]")
            else:
                parts.append(f"[图片: {content.mime_type or '未知格式'}]")
        elif content.content_type == 'audio':
            parts.append(f"[音频: {content.mime_type or '未知格式'}]")
        elif content.content_type == 'video':
            parts.append(f"[视频: {content.mime_type or '未知格式'}]")
        else:
            parts.append(f"[{content.content_type}内容]")
    
    return "\n".join(parts)


def extract_text_from_multimodal(contents: List[MultiModalContent]) -> str:
    """
    从多模态内容中提取纯文本
    
    Args:
        contents: 多模态内容列表
        
    Returns:
        提取的文本
    """
    text_parts = []
    
    for content in contents:
        if content.content_type == 'text':
            text_parts.append(content.data)
    
    return "\n".join(text_parts)


def save_multimodal_content(content: MultiModalContent, output_dir: str = "./multimodal_cache") -> str:
    """
    保存多模态内容到文件
    
    Args:
        content: 多模态内容
        output_dir: 输出目录
        
    Returns:
        保存的文件路径
    """
    import os
    import time
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    timestamp = int(time.time() * 1000)
    
    # 根据内容类型确定扩展名
    ext_map = {
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp',
        'audio/mpeg': '.mp3',
        'audio/wav': '.wav',
        'audio/ogg': '.ogg',
        'video/mp4': '.mp4',
        'video/webm': '.webm',
    }
    
    ext = ext_map.get(content.mime_type, f'.{content.content_type}')
    filename = f"{content.content_type}_{timestamp}{ext}"
    filepath = os.path.join(output_dir, filename)
    
    # 保存文件
    if isinstance(content.data, bytes):
        with open(filepath, 'wb') as f:
            f.write(content.data)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.data)
    
    logger.info(f"保存{content.content_type}内容到: {filepath}")
    return filepath