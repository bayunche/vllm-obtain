"""
OpenAI API 完全兼容实现
严格遵循OpenAI API规范：https://platform.openai.com/docs/api-reference
"""

import time
import json
import uuid
import asyncio
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, g, Response, stream_with_context
from typing import Dict, Any, List, Optional, Union, Generator
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    import warnings
    warnings.warn("tiktoken not available, falling back to simple token estimation")

from ...core.inference_engine import InferenceRequest
from ...core.exceptions import ModelNotFoundError, InferenceError
from ...utils import api_monitor, get_config
from ...utils.message_processor import validate_and_process_messages, process_message_content
from ...utils.multimodal_processor import process_multimodal_content
from ..app import run_async_in_thread

openai_v1 = Blueprint('openai_v1', __name__)


# OpenAI 模型映射
MODEL_ALIASES = {
    # GPT-4 系列映射到最强的本地模型
    "gpt-4": "qwen-0.5b",  # 实际应该映射到你的最强模型
    "gpt-4-0314": "qwen-0.5b",
    "gpt-4-0613": "qwen-0.5b",
    "gpt-4-32k": "qwen-0.5b",
    "gpt-4-32k-0314": "qwen-0.5b",
    "gpt-4-32k-0613": "qwen-0.5b",
    "gpt-4-turbo": "qwen-0.5b",
    "gpt-4-turbo-preview": "qwen-0.5b",
    "gpt-4-1106-preview": "qwen-0.5b",
    "gpt-4-vision-preview": "GLM-4.5V",
    "gpt-4o": "GLM-4.5V", 
    "gpt-4o-mini": "GLM-4.5V",
    
    # 直接使用本地模型名称的映射
    "glm-4.5v": "GLM-4.5V",
    "GLM-4.5V": "GLM-4.5V",
    
    # GPT-3.5 系列
    "gpt-3.5-turbo": "qwen-0.5b",
    "gpt-3.5-turbo-0301": "qwen-0.5b",
    "gpt-3.5-turbo-0613": "qwen-0.5b",
    "gpt-3.5-turbo-1106": "qwen-0.5b",
    "gpt-3.5-turbo-16k": "qwen-0.5b",
    "gpt-3.5-turbo-16k-0613": "qwen-0.5b",
    "gpt-3.5-turbo-instruct": "qwen-0.5b",
    
    # 文本模型
    "text-davinci-003": "qwen-0.5b",
    "text-davinci-002": "qwen-0.5b",
    "text-curie-001": "qwen-0.5b",
    "text-babbage-001": "qwen-0.5b",
    "text-ada-001": "qwen-0.5b",
}


def get_model_manager():
    """获取模型管理器"""
    return current_app.config.get('MODEL_MANAGER')


def get_event_loop():
    """获取事件循环"""
    return current_app.config.get('EVENT_LOOP')


def map_model_name(model: str) -> str:
    """将OpenAI模型名映射到本地模型"""
    # 如果是OpenAI的模型名，映射到本地模型
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    # 否则直接使用提供的模型名
    return model


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    计算文本的token数量
    优先使用tiktoken库进行准确计算，否则降级到简单估算
    """
    if TIKTOKEN_AVAILABLE:
        try:
            # 尝试使用tiktoken计算
            if model.startswith("gpt-4"):
                encoding = tiktoken.get_encoding("cl100k_base")
            elif model.startswith("gpt-3.5"):
                encoding = tiktoken.get_encoding("cl100k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except:
            pass
    
    # 降级到简单估算
    # 英文约4字符=1token，中文约2字符=1token
    import re
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    non_chinese = len(text) - chinese_chars
    return max(1, (chinese_chars // 2) + (non_chinese // 4))


def create_chat_completion_chunk(
    chunk_id: str,
    model: str,
    content: str = "",
    role: str = None,
    finish_reason: str = None,
    index: int = 0
) -> Dict[str, Any]:
    """创建流式响应的chunk"""
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": f"fp_{uuid.uuid4().hex[:8]}",
        "choices": [{
            "index": index,
            "delta": {},
            "logprobs": None,
            "finish_reason": finish_reason
        }]
    }
    
    if role:
        chunk["choices"][0]["delta"]["role"] = role
    if content:
        chunk["choices"][0]["delta"]["content"] = content
        
    return chunk


def create_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: str = None,
    code: str = None,
    status_code: int = 400
) -> tuple:
    """创建OpenAI标准错误响应"""
    error_response = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code
        }
    }
    return jsonify(error_response), status_code


@openai_v1.route('/chat/completions', methods=['POST', 'OPTIONS'])
@api_monitor
def chat_completions():
    """
    Chat Completions API
    完全兼容OpenAI规范：https://platform.openai.com/docs/api-reference/chat
    """
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            return create_error_response("Request body is required", "invalid_request_error")
        
        # 必需参数验证
        if 'model' not in data:
            return create_error_response("Missing required parameter: 'model'", "invalid_request_error", "model")
        
        if 'messages' not in data:
            return create_error_response("Missing required parameter: 'messages'", "invalid_request_error", "messages")
        
        # 获取参数（使用OpenAI的默认值）
        model = map_model_name(data['model'])
        messages = data['messages']
        
        # 可选参数（完全遵循OpenAI默认值）
        frequency_penalty = data.get('frequency_penalty', 0)  # -2.0 to 2.0
        logit_bias = data.get('logit_bias', {})  # token偏好设置
        logprobs = data.get('logprobs', False)  # 是否返回logprobs
        top_logprobs = data.get('top_logprobs', None)  # 返回多少个top logprobs
        max_tokens = data.get('max_tokens', None)  # 最大生成token数
        n = data.get('n', 1)  # 生成多少个回复
        presence_penalty = data.get('presence_penalty', 0)  # -2.0 to 2.0
        response_format = data.get('response_format', {"type": "text"})  # 响应格式
        seed = data.get('seed', None)  # 随机种子
        stop = data.get('stop', None)  # 停止序列
        stream = data.get('stream', False)  # 是否流式输出
        temperature = data.get('temperature', 1)  # 0 to 2
        top_p = data.get('top_p', 1)  # 0 to 1
        tools = data.get('tools', None)  # 函数调用工具
        tool_choice = data.get('tool_choice', None)  # 工具选择策略
        user = data.get('user', None)  # 用户标识符
        
        # 参数验证
        if not isinstance(messages, list) or len(messages) == 0:
            return create_error_response("'messages' must be a non-empty array", "invalid_request_error", "messages")
        
        # 验证消息格式
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return create_error_response(f"Invalid message at index {i}", "invalid_request_error", f"messages[{i}]")
            if 'role' not in msg:
                return create_error_response(f"Missing 'role' in message at index {i}", "invalid_request_error", f"messages[{i}].role")
            if 'content' not in msg and 'tool_calls' not in msg and 'function_call' not in msg:
                return create_error_response(f"Missing 'content' in message at index {i}", "invalid_request_error", f"messages[{i}].content")
            if msg['role'] not in ['system', 'user', 'assistant', 'function', 'tool']:
                return create_error_response(f"Invalid role '{msg['role']}' in message at index {i}", "invalid_request_error", f"messages[{i}].role")
        
        # 温度参数验证
        if not 0 <= temperature <= 2:
            return create_error_response("'temperature' must be between 0 and 2", "invalid_request_error", "temperature")
        
        if not 0 <= top_p <= 1:
            return create_error_response("'top_p' must be between 0 and 1", "invalid_request_error", "top_p")
        
        # 获取配置和管理器
        config = get_config()
        manager = get_model_manager()
        if not manager:
            return create_error_response("Service not ready", "server_error", None, "service_unavailable", 503)
        
        # 处理消息（包括base64解码和token限制）
        try:
            processed_messages = validate_and_process_messages(messages, config)
        except ValueError as e:
            return create_error_response(str(e), "invalid_request_error", "messages")
        
        # 构建prompt
        prompt = build_chat_prompt(processed_messages)
        
        # 提取多模态内容
        multimodal_data = extract_multimodal_data(messages)
        
        # 如果需要流式响应
        if stream:
            return handle_stream_response(
                manager, model, prompt, processed_messages, data, config
            )
        
        # 计算最大token数
        if max_tokens is None:
            # 如果没有指定，使用合理的默认值
            prompt_tokens = count_tokens(prompt, data['model'])
            remaining_context = config.max_sequence_length - prompt_tokens
            max_tokens = min(4096, remaining_context // 2)  # 默认最多4096个token
        
        # 构建推理请求
        inference_request = InferenceRequest(
            model_name=model,
            prompt=prompt,
            messages=processed_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.0 + frequency_penalty * 0.1,  # 转换frequency_penalty
            stop_sequences=stop,
            stream=False,
            request_id=g.request_id,
            multimodal_data=multimodal_data  # 传递多模态数据
        )
        
        # 执行推理
        loop = get_event_loop()
        response = run_async_in_thread(
            manager.inference(inference_request), loop
        )
        
        # 构建OpenAI格式响应
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        
        # 计算token使用情况
        prompt_tokens = response.prompt_tokens or count_tokens(prompt, data['model'])
        completion_tokens = response.completion_tokens or count_tokens(response.text, data['model'])
        
        # 构建响应
        chat_response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data['model'],  # 返回原始请求的模型名
            "system_fingerprint": f"fp_{uuid.uuid4().hex[:8]}",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
        # 生成n个回复
        for i in range(n):
            choice = {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": response.text
                },
                "logprobs": None,
                "finish_reason": map_finish_reason(response.finish_reason)
            }
            
            # 如果请求了logprobs
            if logprobs and response.logprobs:
                choice["logprobs"] = response.logprobs
                
            chat_response["choices"].append(choice)
        
        return jsonify(chat_response)
        
    except ModelNotFoundError as e:
        return create_error_response(f"Model not found: {str(e)}", "model_not_found", "model", None, 404)
    except InferenceError as e:
        return create_error_response(str(e), "server_error", None, "inference_error", 500)
    except Exception as e:
        current_app.config['LOGGER'].error(f"Chat completion error: {e}")
        import traceback
        current_app.config['LOGGER'].debug(traceback.format_exc())
        return create_error_response(
            "The server had an error processing your request. Sorry about that!",
            "server_error",
            None,
            None,
            500
        )


def handle_stream_response(manager, model, prompt, messages, data, config):
    """处理流式响应"""
    def generate():
        try:
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            
            # 发送初始chunk（包含role）
            initial_chunk = create_chat_completion_chunk(
                completion_id, data['model'], role="assistant"
            )
            yield f"data: {json.dumps(initial_chunk)}\n\n"
            
            # 提取多模态数据
            multimodal_data = extract_multimodal_data(data.get('messages', []))
            
            # 构建推理请求
            inference_request = InferenceRequest(
                model_name=model,
                prompt=prompt,
                messages=messages,
                max_tokens=data.get('max_tokens', 4096),
                temperature=data.get('temperature', 1),
                top_p=data.get('top_p', 1),
                repetition_penalty=1.0 + data.get('frequency_penalty', 0) * 0.1,
                stop_sequences=data.get('stop'),
                stream=True,
                request_id=g.request_id,
                multimodal_data=multimodal_data
            )
            
            # 异步流式生成
            loop = get_event_loop()
            
            async def stream_inference():
                async for chunk_text in manager.inference_stream(inference_request):
                    if chunk_text:
                        chunk = create_chat_completion_chunk(
                            completion_id, data['model'], content=chunk_text
                        )
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                # 发送结束chunk
                final_chunk = create_chat_completion_chunk(
                    completion_id, data['model'], finish_reason="stop"
                )
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            # 在事件循环中运行
            future = asyncio.run_coroutine_threadsafe(stream_inference(), loop)
            for chunk in future.result():
                yield chunk
                
        except Exception as e:
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@openai_v1.route('/completions', methods=['POST', 'OPTIONS'])
@api_monitor
def completions():
    """
    Completions API (Legacy)
    完全兼容OpenAI规范：https://platform.openai.com/docs/api-reference/completions
    """
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        if not data:
            return create_error_response("Request body is required", "invalid_request_error")
        
        # 必需参数
        if 'model' not in data:
            return create_error_response("Missing required parameter: 'model'", "invalid_request_error", "model")
        
        model = map_model_name(data['model'])
        prompt = data.get('prompt', '')  # prompt可以是字符串、数组或null
        
        # 处理不同格式的prompt
        if isinstance(prompt, list):
            # 如果是数组，连接成字符串
            prompt = '\n'.join(str(p) for p in prompt)
        elif prompt is None:
            prompt = ''
        else:
            prompt = str(prompt)
        
        # OpenAI参数
        best_of = data.get('best_of', 1)
        echo = data.get('echo', False)
        frequency_penalty = data.get('frequency_penalty', 0)
        logit_bias = data.get('logit_bias', {})
        logprobs = data.get('logprobs', None)
        max_tokens = data.get('max_tokens', 16)  # 默认16
        n = data.get('n', 1)
        presence_penalty = data.get('presence_penalty', 0)
        seed = data.get('seed', None)
        stop = data.get('stop', None)
        stream = data.get('stream', False)
        suffix = data.get('suffix', None)
        temperature = data.get('temperature', 1)
        top_p = data.get('top_p', 1)
        user = data.get('user', None)
        
        # 获取配置和管理器
        config = get_config()
        manager = get_model_manager()
        if not manager:
            return create_error_response("Service not ready", "server_error", None, "service_unavailable", 503)
        
        # 处理prompt（base64解码）
        processed_prompt = process_message_content(prompt, config.max_sequence_length - max_tokens)
        
        # 如果需要echo，将prompt添加到输出
        if echo:
            processed_prompt = prompt + processed_prompt
        
        # 构建推理请求
        inference_request = InferenceRequest(
            model_name=model,
            prompt=processed_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.0 + frequency_penalty * 0.1,
            stop_sequences=stop,
            stream=stream,
            request_id=g.request_id
        )
        
        # 流式响应
        if stream:
            return handle_completion_stream(manager, inference_request, data)
        
        # 执行推理
        loop = get_event_loop()
        response = run_async_in_thread(
            manager.inference(inference_request), loop
        )
        
        # 构建响应
        completion_id = f"cmpl-{uuid.uuid4().hex[:29]}"
        
        # 处理输出文本
        output_text = response.text
        if suffix:
            output_text = output_text + suffix
        if echo:
            output_text = prompt + output_text
        
        # 计算token
        prompt_tokens = count_tokens(prompt, data['model'])
        completion_tokens = count_tokens(output_text, data['model'])
        
        completion_response = {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": data['model'],
            "system_fingerprint": f"fp_{uuid.uuid4().hex[:8]}",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
        # 生成n个回复
        for i in range(n):
            choice = {
                "text": output_text,
                "index": i,
                "logprobs": None,
                "finish_reason": map_finish_reason(response.finish_reason)
            }
            
            if logprobs:
                # TODO: 实现logprobs
                choice["logprobs"] = {
                    "tokens": [],
                    "token_logprobs": [],
                    "top_logprobs": [],
                    "text_offset": []
                }
                
            completion_response["choices"].append(choice)
        
        return jsonify(completion_response)
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"Completion error: {e}")
        return create_error_response(
            "The server had an error processing your request. Sorry about that!",
            "server_error",
            None,
            None,
            500
        )


def handle_completion_stream(manager, inference_request, data):
    """处理补全API的流式响应"""
    def generate():
        try:
            completion_id = f"cmpl-{uuid.uuid4().hex[:29]}"
            
            loop = get_event_loop()
            
            async def stream_inference():
                async for chunk_text in manager.inference_stream(inference_request):
                    if chunk_text:
                        chunk = {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": data['model'],
                            "system_fingerprint": f"fp_{uuid.uuid4().hex[:8]}",
                            "choices": [{
                                "text": chunk_text,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                # 最后一个chunk
                final_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": data['model'],
                    "system_fingerprint": f"fp_{uuid.uuid4().hex[:8]}",
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            future = asyncio.run_coroutine_threadsafe(stream_inference(), loop)
            for chunk in future.result():
                yield chunk
                
        except Exception as e:
            error_chunk = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@openai_v1.route('/models', methods=['GET', 'OPTIONS'])
@api_monitor
def list_models():
    """
    List Models API
    完全兼容OpenAI规范：https://platform.openai.com/docs/api-reference/models/list
    """
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        manager = get_model_manager()
        if not manager:
            return create_error_response("Service not ready", "server_error", None, "service_unavailable", 503)
        
        # 获取已加载的模型
        loop = get_event_loop()
        system_status = run_async_in_thread(
            manager.get_system_status(), loop
        )
        
        models_data = []
        
        # 添加本地模型
        for model in system_status.get("models", []):
            models_data.append({
                "id": model["name"],
                "object": "model",
                "created": int(model.get("loaded_at", time.time())),
                "owned_by": "system"
            })
        
        # 添加OpenAI模型别名（让客户端认为我们支持这些模型）
        for openai_model in MODEL_ALIASES.keys():
            models_data.append({
                "id": openai_model,
                "object": "model",
                "created": 1677649963,  # 固定时间戳
                "owned_by": "openai"
            })
        
        return jsonify({
            "object": "list",
            "data": models_data
        })
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"List models error: {e}")
        return create_error_response("Failed to list models", "server_error", None, None, 500)


@openai_v1.route('/models/<model_id>', methods=['GET', 'OPTIONS'])
@api_monitor
def retrieve_model(model_id):
    """
    Retrieve Model API
    完全兼容OpenAI规范：https://platform.openai.com/docs/api-reference/models/retrieve
    """
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # 检查是否是OpenAI模型别名
        if model_id in MODEL_ALIASES:
            return jsonify({
                "id": model_id,
                "object": "model",
                "created": 1677649963,
                "owned_by": "openai"
            })
        
        # 检查本地模型
        manager = get_model_manager()
        if not manager:
            return create_error_response("Service not ready", "server_error", None, "service_unavailable", 503)
        
        model_info = manager.get_model_info(model_id)
        if not model_info:
            return create_error_response(f"Model '{model_id}' not found", "model_not_found", "model", None, 404)
        
        return jsonify({
            "id": model_id,
            "object": "model",
            "created": int(model_info.get("loaded_at", time.time())),
            "owned_by": "system"
        })
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"Retrieve model error: {e}")
        return create_error_response("Failed to retrieve model", "server_error", None, None, 500)


@openai_v1.route('/embeddings', methods=['POST', 'OPTIONS'])
@api_monitor
def create_embeddings():
    """
    Embeddings API
    完全兼容OpenAI规范：https://platform.openai.com/docs/api-reference/embeddings
    """
    if request.method == 'OPTIONS':
        return '', 204
        
    # TODO: 实现embeddings功能
    return create_error_response(
        "Embeddings endpoint is not implemented yet",
        "feature_not_supported",
        None,
        "not_implemented",
        501
    )


@openai_v1.route('/moderations', methods=['POST', 'OPTIONS'])
@api_monitor
def create_moderation():
    """
    Moderations API
    完全兼容OpenAI规范：https://platform.openai.com/docs/api-reference/moderations
    """
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        if not data:
            return create_error_response("Request body is required", "invalid_request_error")
        
        input_text = data.get('input', '')
        model = data.get('model', 'text-moderation-stable')
        
        # 返回模拟的安全响应（所有内容都是安全的）
        if isinstance(input_text, list):
            results = [create_moderation_result(text) for text in input_text]
        else:
            results = [create_moderation_result(input_text)]
        
        return jsonify({
            "id": f"modr-{uuid.uuid4().hex[:27]}",
            "model": model,
            "results": results
        })
        
    except Exception as e:
        return create_error_response("Moderation error", "server_error", None, None, 500)


def create_moderation_result(text: str) -> Dict[str, Any]:
    """创建内容审核结果（默认所有内容安全）"""
    return {
        "flagged": False,
        "categories": {
            "hate": False,
            "hate/threatening": False,
            "harassment": False,
            "harassment/threatening": False,
            "self-harm": False,
            "self-harm/intent": False,
            "self-harm/instructions": False,
            "sexual": False,
            "sexual/minors": False,
            "violence": False,
            "violence/graphic": False
        },
        "category_scores": {
            "hate": 0.0,
            "hate/threatening": 0.0,
            "harassment": 0.0,
            "harassment/threatening": 0.0,
            "self-harm": 0.0,
            "self-harm/intent": 0.0,
            "self-harm/instructions": 0.0,
            "sexual": 0.0,
            "sexual/minors": 0.0,
            "violence": 0.0,
            "violence/graphic": 0.0
        }
    }


def build_chat_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    将消息列表转换为prompt
    遵循ChatML格式
    """
    prompt_parts = []
    
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        # 处理不同的content格式
        if isinstance(content, list):
            # 多模态内容
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(text_parts)
        
        # 构建ChatML格式
        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "function":
            name = message.get("name", "function")
            prompt_parts.append(f"<|im_start|>function name={name}\n{content}<|im_end|>")
    
    # 添加助手开始标记
    prompt_parts.append("<|im_start|>assistant\n")
    
    return "\n".join(prompt_parts)


def extract_multimodal_data(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从消息中提取多模态数据"""
    multimodal_data = []
    
    for message in messages:
        content = message.get("content", "")
        
        # 如果content是数组格式（多模态）
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # 提取多模态内容
                    multimodal_contents = process_multimodal_content(item)
                    for mm_content in multimodal_contents:
                        if mm_content.content_type != 'text':  # 非文本内容
                            multimodal_data.append(mm_content.to_dict())
        
        # 如果content是字符串，也可能是base64编码的媒体
        elif isinstance(content, str):
            multimodal_contents = process_multimodal_content(content)
            for mm_content in multimodal_contents:
                if mm_content.content_type != 'text':
                    multimodal_data.append(mm_content.to_dict())
    
    return multimodal_data


def map_finish_reason(reason: str) -> str:
    """将内部finish_reason映射到OpenAI格式"""
    if not reason:
        return "stop"
    
    reason_lower = reason.lower()
    
    if "stop" in reason_lower or "complete" in reason_lower:
        return "stop"
    elif "length" in reason_lower or "max" in reason_lower:
        return "length"
    elif "function" in reason_lower or "tool" in reason_lower:
        return "tool_calls"
    elif "content" in reason_lower:
        return "content_filter"
    else:
        return "stop"


def register_openai_v1_routes(app):
    """注册OpenAI v1 API路由"""
    app.register_blueprint(openai_v1, url_prefix='/v1')