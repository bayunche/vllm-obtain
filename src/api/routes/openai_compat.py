"""
OpenAI API 完全兼容接口
支持 n8n 等工具直接调用，完全复刻 OpenAI API 格式
"""

import time
import uuid
import asyncio
from flask import Blueprint, request, jsonify, current_app, g, Response
from typing import Dict, Any, List

from ...core.inference_engine import InferenceRequest
from ...core.exceptions import ModelNotFoundError, InferenceError
from ...utils import api_monitor
from ..app import run_async_in_thread

openai_bp = Blueprint('openai', __name__)


def get_model_manager():
    """获取模型管理器"""
    return current_app.config.get('MODEL_MANAGER')


def get_event_loop():
    """获取事件循环"""
    return current_app.config.get('EVENT_LOOP')


def create_error_response(error_message: str, error_type: str = "invalid_request_error", 
                         error_code: str = None, status_code: int = 400) -> tuple:
    """创建标准的 OpenAI 错误响应"""
    return jsonify({
        "error": {
            "message": error_message,
            "type": error_type,
            "param": None,
            "code": error_code
        }
    }), status_code


@openai_bp.route('/models', methods=['GET'])
@api_monitor
def list_models():
    """
    列出可用模型
    GET /v1/models
    
    完全兼容 OpenAI API 格式
    """
    try:
        manager = get_model_manager()
        if not manager:
            return create_error_response("服务未就绪", "server_error", status_code=503)
        
        # 获取已加载的模型
        loop = get_event_loop()
        loaded_models = run_async_in_thread(
            manager.get_system_status(), loop
        )
        
        # 转换为 OpenAI 格式
        models_data = []
        for model in loaded_models.get("models", []):
            models_data.append({
                "id": model["name"],
                "object": "model",
                "created": int(model.get("loaded_at", time.time())),
                "owned_by": "local",
                "permission": [],
                "root": model["name"],
                "parent": None
            })
        
        # 添加注册但未加载的模型
        for model_config in manager.list_registered_models():
            if not any(m["id"] == model_config.name for m in models_data):
                models_data.append({
                    "id": model_config.name,
                    "object": "model", 
                    "created": int(time.time()),
                    "owned_by": "local",
                    "permission": [],
                    "root": model_config.name,
                    "parent": None
                })
        
        return jsonify({
            "object": "list",
            "data": models_data
        })
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"列出模型失败: {e}")
        return create_error_response(str(e), "server_error", status_code=500)


@openai_bp.route('/models/<model_id>', methods=['GET'])
@api_monitor
def retrieve_model(model_id):
    """
    获取特定模型信息
    GET /v1/models/{model_id}
    
    完全兼容 OpenAI API 格式
    """
    try:
        manager = get_model_manager()
        if not manager:
            return create_error_response("服务未就绪", "server_error", status_code=503)
        
        # 检查模型是否存在
        model_info = manager.get_model_info(model_id)
        if not model_info:
            return create_error_response(f"模型不存在: {model_id}", "not_found_error", status_code=404)
        
        return jsonify({
            "id": model_id,
            "object": "model",
            "created": int(model_info.get("loaded_at", time.time())),
            "owned_by": "local",
            "permission": [],
            "root": model_id,
            "parent": None
        })
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"获取模型信息失败: {e}")
        return create_error_response(str(e), "server_error", status_code=500)


@openai_bp.route('/chat/completions', methods=['POST'])
@api_monitor
def chat_completions():
    """
    聊天补全接口
    POST /v1/chat/completions
    
    完全兼容 OpenAI ChatGPT API 格式，支持 n8n 调用
    """
    try:
        manager = get_model_manager()
        if not manager:
            return create_error_response("服务未就绪", "server_error", status_code=503)
        
        # 解析请求数据
        data = request.get_json()
        if not data:
            return create_error_response("请求体不能为空", "invalid_request_error")
        
        # 验证必需参数
        if 'model' not in data:
            return create_error_response("缺少必需参数: model", "invalid_request_error")
        
        if 'messages' not in data:
            return create_error_response("缺少必需参数: messages", "invalid_request_error")
        
        model_name = data['model']
        messages = data['messages']
        
        # 验证消息格式
        if not isinstance(messages, list) or len(messages) == 0:
            return create_error_response("messages 必须是非空数组", "invalid_request_error")
        
        # 将messages转换为prompt格式
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        
        # 构建推理请求
        request_obj = InferenceRequest(
            model_name=model_name,
            prompt=prompt,
            messages=messages,
            max_tokens=data.get('max_tokens', 100),
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            repetition_penalty=data.get('repetition_penalty', 1.0),
            stop_sequences=data.get('stop'),
            stream=data.get('stream', False),
            request_id=g.request_id
        )
        
        # 处理流式响应
        if request_obj.stream:
            return handle_stream_chat_completion(manager, request_obj, data)
        
        # 执行推理
        loop = get_event_loop()
        response = run_async_in_thread(
            manager.inference(request_obj), loop
        )
        
        # 转换为 OpenAI 格式
        chat_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.text
                },
                "finish_reason": response.finish_reason
            }],
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens
            },
            "system_fingerprint": f"fp_{int(time.time())}"
        }
        
        return jsonify(chat_response)
        
    except ModelNotFoundError as e:
        return create_error_response(str(e), "not_found_error", status_code=404)
    except InferenceError as e:
        return create_error_response(str(e), "server_error", status_code=500)
    except Exception as e:
        current_app.config['LOGGER'].error(f"聊天补全失败: {e}")
        return create_error_response(str(e), "server_error", status_code=500)


def handle_stream_chat_completion(manager, request_obj, data):
    """处理流式聊天补全"""
    def generate():
        try:
            loop = get_event_loop()
            
            # 创建流式生成器
            async def stream_generator():
                async for chunk in manager.inference_stream(request_obj):
                    if chunk:
                        yield chunk
            
            # 执行流式生成
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            accumulated_content = ""
            
            # 发送开始事件
            chunk_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request_obj.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            }
            yield f"data: {jsonify(chunk_data).get_data(as_text=True)}\n\n"
            
            # 执行异步流式生成
            async def async_stream():
                nonlocal accumulated_content
                async for chunk in manager.inference_stream(request_obj):
                    if chunk:
                        accumulated_content += chunk
                        chunk_data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request_obj.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {jsonify(chunk_data).get_data(as_text=True)}\n\n"
                
                # 发送结束事件
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request_obj.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {jsonify(final_chunk).get_data(as_text=True)}\n\n"
                yield "data: [DONE]\n\n"
            
            # 在事件循环中运行
            if loop and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(async_stream(), loop)
                for chunk in future.result():
                    yield chunk
            
        except Exception as e:
            current_app.config['LOGGER'].error(f"流式生成失败: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {jsonify(error_chunk).get_data(as_text=True)}\n\n"
    
    return Response(
        generate(),
        mimetype='text/plain',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@openai_bp.route('/completions', methods=['POST'])
@api_monitor
def completions():
    """
    文本补全接口
    POST /v1/completions
    
    完全兼容 OpenAI API 格式
    """
    try:
        manager = get_model_manager()
        if not manager:
            return create_error_response("服务未就绪", "server_error", status_code=503)
        
        # 解析请求数据
        data = request.get_json()
        if not data:
            return create_error_response("请求体不能为空", "invalid_request_error")
        
        # 验证必需参数
        if 'model' not in data:
            return create_error_response("缺少必需参数: model", "invalid_request_error")
        
        if 'prompt' not in data:
            return create_error_response("缺少必需参数: prompt", "invalid_request_error")
        
        model_name = data['model']
        prompt = data['prompt']
        
        # 构建推理请求
        request_obj = InferenceRequest(
            model_name=model_name,
            prompt=prompt,
            max_tokens=data.get('max_tokens', 100),
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            repetition_penalty=data.get('repetition_penalty', 1.0),
            stop_sequences=data.get('stop'),
            stream=data.get('stream', False),
            request_id=g.request_id
        )
        
        # 执行推理
        loop = get_event_loop()
        response = run_async_in_thread(
            manager.inference(request_obj), loop
        )
        
        # 转换为 OpenAI 格式
        completion_response = {
            "id": f"cmpl-{uuid.uuid4().hex[:29]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "text": response.text,
                "index": 0,
                "logprobs": None,
                "finish_reason": response.finish_reason
            }],
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens
            }
        }
        
        return jsonify(completion_response)
        
    except ModelNotFoundError as e:
        return create_error_response(str(e), "not_found_error", status_code=404)
    except InferenceError as e:
        return create_error_response(str(e), "server_error", status_code=500)
    except Exception as e:
        current_app.config['LOGGER'].error(f"文本补全失败: {e}")
        return create_error_response(str(e), "server_error", status_code=500)


@openai_bp.route('/embeddings', methods=['POST'])
@api_monitor
def embeddings():
    """
    文本嵌入接口 (占位符实现)
    POST /v1/embeddings
    
    为了完整的 OpenAI API 兼容性
    """
    return create_error_response(
        "嵌入功能暂未实现", 
        "feature_not_implemented", 
        status_code=501
    )


@openai_bp.route('/audio/transcriptions', methods=['POST'])
@api_monitor
def audio_transcriptions():
    """
    音频转录接口 (占位符实现)
    POST /v1/audio/transcriptions
    """
    return create_error_response(
        "音频转录功能暂未实现", 
        "feature_not_implemented", 
        status_code=501
    )


@openai_bp.route('/audio/translations', methods=['POST'])
@api_monitor
def audio_translations():
    """
    音频翻译接口 (占位符实现)
    POST /v1/audio/translations
    """
    return create_error_response(
        "音频翻译功能暂未实现", 
        "feature_not_implemented", 
        status_code=501
    )


@openai_bp.route('/images/generations', methods=['POST'])
@api_monitor
def image_generations():
    """
    图像生成接口 (占位符实现)
    POST /v1/images/generations
    """
    return create_error_response(
        "图像生成功能暂未实现", 
        "feature_not_implemented", 
        status_code=501
    )


# 错误处理
@openai_bp.errorhandler(400)
def bad_request(error):
    """400 错误处理"""
    return create_error_response("请求格式错误", "bad_request_error", status_code=400)


@openai_bp.errorhandler(401)
def unauthorized(error):
    """401 错误处理"""
    return create_error_response("未授权访问", "authentication_error", status_code=401)


@openai_bp.errorhandler(429)
def rate_limit(error):
    """429 错误处理"""
    return create_error_response("请求频率过高", "rate_limit_error", status_code=429)


@openai_bp.errorhandler(500)
def internal_error(error):
    """500 错误处理"""
    return create_error_response("内部服务器错误", "server_error", status_code=500)


def register_openai_routes(app):
    """注册OpenAI兼容路由"""
    app.register_blueprint(openai_bp, url_prefix='/v1')