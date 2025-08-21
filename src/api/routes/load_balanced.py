#!/usr/bin/env python3
"""
负载均衡路由
通过负载均衡器分发 OpenAI API 请求到后端实例
"""

import asyncio
import time
from flask import request, jsonify, current_app, stream_template
from typing import Dict, Any, Optional

from ...core.exceptions import InferenceError
from ...utils.logger import get_logger

logger = get_logger(__name__)


def register_load_balanced_routes(app):
    """注册负载均衡路由"""
    
    # OpenAI 兼容接口
    @app.route('/v1/models', methods=['GET'])
    async def list_models_lb():
        """列出可用模型（负载均衡版本）"""
        try:
            load_balancer = current_app.config.get('LOAD_BALANCER')
            if not load_balancer:
                return jsonify({"error": {"message": "负载均衡器未初始化"}}), 503
            
            # 从任意健康实例获取模型列表
            instance = load_balancer.get_instance()
            if not instance:
                return jsonify({"error": {"message": "没有可用的实例"}}), 503
            
            # 直接使用实例的模型信息或发送请求获取
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{instance.url}/v1/models") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return jsonify({"error": {"message": "获取模型列表失败"}}), 503
        
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return jsonify({"error": {"message": str(e)}}), 500
    
    @app.route('/v1/models/<model_id>', methods=['GET'])
    async def get_model_lb(model_id: str):
        """获取特定模型信息（负载均衡版本）"""
        try:
            load_balancer = current_app.config.get('LOAD_BALANCER')
            if not load_balancer:
                return jsonify({"error": {"message": "负载均衡器未初始化"}}), 503
            
            # 获取支持该模型的实例
            instance = load_balancer.get_instance(model_id)
            if not instance:
                return jsonify({"error": {"message": f"没有支持模型 {model_id} 的实例"}}), 404
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{instance.url}/v1/models/{model_id}") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return jsonify({"error": {"message": "获取模型信息失败"}}), response.status
        
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return jsonify({"error": {"message": str(e)}}), 500
    
    @app.route('/v1/chat/completions', methods=['POST'])
    async def chat_completions_lb():
        """聊天补全（负载均衡版本）"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": {"message": "请求体不能为空"}}), 400
            
            model_name = data.get('model')
            if not model_name:
                return jsonify({"error": {"message": "缺少 model 参数"}}), 400
            
            # 获取负载均衡器
            load_balancer = current_app.config.get('LOAD_BALANCER')
            if not load_balancer:
                return jsonify({"error": {"message": "负载均衡器未初始化"}}), 503
            
            # 选择实例
            instance = load_balancer.get_instance(model_name)
            if not instance:
                return jsonify({
                    "error": {
                        "message": f"没有支持模型 {model_name} 的可用实例",
                        "type": "model_not_found",
                        "code": "model_not_found"
                    }
                }), 404
            
            # 标记请求开始
            load_balancer.mark_request_start(instance.instance_id)
            
            start_time = time.time()
            success = False
            
            try:
                # 转发请求到选定的实例
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{instance.url}/v1/chat/completions",
                        json=data,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            success = True
                            
                            # 处理流式响应
                            if data.get('stream', False):
                                # 流式响应
                                def generate():
                                    while True:
                                        chunk = yield from response.content.iter_chunked(1024)
                                        if not chunk:
                                            break
                                        yield chunk
                                
                                return current_app.response_class(
                                    generate(),
                                    mimetype='text/plain',
                                    headers={'Content-Type': 'text/plain; charset=utf-8'}
                                )
                            else:
                                # 非流式响应
                                result = await response.json()
                                return jsonify(result)
                        else:
                            error_text = await response.text()
                            logger.error(f"实例 {instance.instance_id} 返回错误: {response.status} - {error_text}")
                            return jsonify({
                                "error": {
                                    "message": "后端实例处理失败",
                                    "details": error_text
                                }
                            }), response.status
            
            except Exception as e:
                response_time = time.time() - start_time
                logger.error(f"转发请求到实例 {instance.instance_id} 失败: {e}")
                return jsonify({
                    "error": {
                        "message": "请求转发失败",
                        "details": str(e)
                    }
                }), 500
            
            finally:
                # 标记请求结束
                response_time = time.time() - start_time
                load_balancer.mark_request_end(instance.instance_id, response_time, success)
        
        except Exception as e:
            logger.error(f"聊天补全处理失败: {e}")
            return jsonify({"error": {"message": str(e)}}), 500
    
    @app.route('/v1/completions', methods=['POST'])
    async def completions_lb():
        """文本补全（负载均衡版本）"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": {"message": "请求体不能为空"}}), 400
            
            model_name = data.get('model')
            if not model_name:
                return jsonify({"error": {"message": "缺少 model 参数"}}), 400
            
            # 获取负载均衡器
            load_balancer = current_app.config.get('LOAD_BALANCER')
            if not load_balancer:
                return jsonify({"error": {"message": "负载均衡器未初始化"}}), 503
            
            # 选择实例
            instance = load_balancer.get_instance(model_name)
            if not instance:
                return jsonify({
                    "error": {
                        "message": f"没有支持模型 {model_name} 的可用实例",
                        "type": "model_not_found",
                        "code": "model_not_found"
                    }
                }), 404
            
            # 标记请求开始
            load_balancer.mark_request_start(instance.instance_id)
            
            start_time = time.time()
            success = False
            
            try:
                # 转发请求到选定的实例
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{instance.url}/v1/completions",
                        json=data,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            success = True
                            result = await response.json()
                            return jsonify(result)
                        else:
                            error_text = await response.text()
                            logger.error(f"实例 {instance.instance_id} 返回错误: {response.status} - {error_text}")
                            return jsonify({
                                "error": {
                                    "message": "后端实例处理失败",
                                    "details": error_text
                                }
                            }), response.status
            
            except Exception as e:
                logger.error(f"转发请求到实例 {instance.instance_id} 失败: {e}")
                return jsonify({
                    "error": {
                        "message": "请求转发失败",
                        "details": str(e)
                    }
                }), 500
            
            finally:
                # 标记请求结束
                response_time = time.time() - start_time
                load_balancer.mark_request_end(instance.instance_id, response_time, success)
        
        except Exception as e:
            logger.error(f"文本补全处理失败: {e}")
            return jsonify({"error": {"message": str(e)}}), 500
    
    # 系统监控接口
    @app.route('/v1/system/status', methods=['GET'])
    def system_status_lb():
        """系统状态（负载均衡版本）"""
        try:
            cluster_manager = current_app.config.get('CLUSTER_MANAGER')
            load_balancer = current_app.config.get('LOAD_BALANCER')
            
            if not cluster_manager or not load_balancer:
                return jsonify({"error": "服务未初始化"}), 503
            
            cluster_status = cluster_manager.get_cluster_status()
            lb_stats = load_balancer.get_instance_stats()
            
            return jsonify({
                "status": "healthy" if cluster_status['running_instances'] > 0 else "unhealthy",
                "cluster": cluster_status,
                "load_balancer": {
                    "strategy": load_balancer.strategy.value,
                    "instance_stats": lb_stats
                }
            })
        
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return jsonify({"error": {"message": str(e)}}), 500
    
    @app.route('/v1/system/metrics', methods=['GET'])
    def system_metrics_lb():
        """系统指标（负载均衡版本）"""
        try:
            load_balancer = current_app.config.get('LOAD_BALANCER')
            if not load_balancer:
                return jsonify({"error": "负载均衡器未初始化"}), 503
            
            stats = load_balancer.get_instance_stats()
            
            # 聚合指标
            total_requests = sum(inst.get('total_requests', 0) for inst in stats.values())
            total_failed = sum(inst.get('failed_requests', 0) for inst in stats.values())
            active_connections = sum(inst.get('active_connections', 0) for inst in stats.values())
            avg_response_times = [inst.get('avg_response_time', 0) for inst in stats.values()]
            
            overall_avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0
            success_rate = (total_requests - total_failed) / total_requests if total_requests > 0 else 1.0
            
            return jsonify({
                "overview": {
                    "total_instances": len(stats),
                    "healthy_instances": sum(1 for inst in stats.values() if inst.get('is_healthy')),
                    "total_requests": total_requests,
                    "failed_requests": total_failed,
                    "success_rate": success_rate,
                    "active_connections": active_connections,
                    "avg_response_time": overall_avg_response_time
                },
                "instances": stats
            })
        
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return jsonify({"error": {"message": str(e)}}), 500