"""
管理接口
提供系统管理、模型管理等功能
"""

import time
from flask import Blueprint, request, jsonify, current_app

from ...core.exceptions import ModelNotFoundError, ModelLoadError
from ...utils import api_monitor
from ..app import run_async_in_thread

management_bp = Blueprint('management', __name__)


def get_model_manager():
    """获取模型管理器"""
    return current_app.config.get('MODEL_MANAGER')


def get_event_loop():
    """获取事件循环"""
    return current_app.config.get('EVENT_LOOP')


@management_bp.route('/system/status', methods=['GET'])
@api_monitor
def system_status():
    """
    获取系统状态
    GET /v1/system/status
    """
    try:
        manager = get_model_manager()
        if not manager:
            return jsonify({
                "error": "模型管理器未初始化"
            }), 503
        
        # 获取系统状态
        loop = get_event_loop()
        status = run_async_in_thread(
            manager.get_system_status(), loop
        )
        
        return jsonify(status)
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"获取系统状态失败: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@management_bp.route('/system/health', methods=['GET'])
@api_monitor
def system_health():
    """
    系统健康检查
    GET /v1/system/health
    """
    try:
        manager = get_model_manager()
        if not manager:
            return jsonify({
                "status": "unhealthy",
                "message": "模型管理器未初始化",
                "timestamp": time.time()
            }), 503
        
        # 健康检查
        loop = get_event_loop()
        health_result = run_async_in_thread(
            manager.health_check(), loop
        )
        
        status_code = 200 if health_result["status"] == "healthy" else 503
        return jsonify(health_result), status_code
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"健康检查失败: {e}")
        return jsonify({
            "status": "unhealthy",
            "message": str(e),
            "timestamp": time.time()
        }), 503


@management_bp.route('/system/metrics', methods=['GET'])
@api_monitor
def system_metrics():
    """
    获取系统性能指标
    GET /v1/system/metrics
    """
    try:
        # 记录系统指标
        logger = current_app.config['LOGGER']
        logger.log_system_metrics()
        
        # 获取最新指标
        metrics = logger.metrics.get_system_metrics()
        
        return jsonify(metrics)
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"获取系统指标失败: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@management_bp.route('/models/load', methods=['POST'])
@api_monitor
def load_model():
    """
    加载模型
    POST /v1/models/load
    """
    try:
        manager = get_model_manager()
        if not manager:
            return jsonify({
                "error": "模型管理器未初始化"
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "请求体不能为空"
            }), 400
        
        model_name = data.get('model_name')
        model_path = data.get('model_path')
        
        if not model_name:
            return jsonify({
                "error": "缺少必需参数: model_name"
            }), 400
        
        if not model_path:
            return jsonify({
                "error": "缺少必需参数: model_path"
            }), 400
        
        # 注册并加载模型
        from ...core.model_manager import ModelConfig
        
        # 确定引擎类型：优先级为 用户指定 > 现有配置 > 自动检测
        engine_type = data.get('engine_type')
        if not engine_type:
            # 检查是否已有注册的模型配置
            existing_config = manager.model_configs.get(model_name)
            if existing_config and existing_config.engine_type:
                # 保留现有的引擎类型
                engine_type = existing_config.engine_type
                print(f"保留现有模型配置的引擎类型: {engine_type}")
            else:
                # 自动检测引擎类型
                from pathlib import Path
                engine_type = manager._detect_model_engine_type(Path(model_path))
                print(f"自动检测到引擎类型: {engine_type}")
        
        model_config = ModelConfig(
            name=model_name,
            path=model_path,
            engine_type=engine_type,
            auto_load=True,
            priority=data.get('priority', 1),
            max_context_length=data.get('max_context_length'),
            extra_params=data.get('extra_params', {})
        )
        
        manager.register_model(model_config)
        
        # 执行加载
        loop = get_event_loop()
        success = run_async_in_thread(
            manager.load_model(model_name, data.get('force_reload', False)), 
            loop
        )
        
        if success:
            return jsonify({
                "message": f"模型加载成功: {model_name}",
                "model_name": model_name,
                "status": "loaded"
            })
        else:
            return jsonify({
                "error": f"模型加载失败: {model_name}"
            }), 500
        
    except ModelLoadError as e:
        return jsonify({
            "error": str(e)
        }), 400
    except Exception as e:
        current_app.config['LOGGER'].error(f"模型加载失败: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@management_bp.route('/models/unload', methods=['POST'])
@api_monitor
def unload_model_by_name():
    """
    卸载模型 (通过请求体指定模型名)
    POST /v1/models/unload
    """
    try:
        manager = get_model_manager()
        if not manager:
            return jsonify({
                "error": "模型管理器未初始化"
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "请求体不能为空"
            }), 400
        
        model_name = data.get('model_name')
        if not model_name:
            return jsonify({
                "error": "缺少必需参数: model_name"
            }), 400
        
        # 执行卸载
        loop = get_event_loop()
        success = run_async_in_thread(
            manager.unload_model(model_name), loop
        )
        
        if success:
            return jsonify({
                "message": f"模型卸载成功: {model_name}",
                "model_name": model_name,
                "status": "unloaded"
            })
        else:
            return jsonify({
                "error": f"模型卸载失败: {model_name}"
            }), 500
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"模型卸载失败: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@management_bp.route('/models/<model_name>/unload', methods=['DELETE', 'POST'])
@api_monitor
def unload_model(model_name):
    """
    卸载模型 (通过URL路径指定模型名)
    DELETE /v1/models/{model_name}/unload
    POST /v1/models/{model_name}/unload
    """
    try:
        manager = get_model_manager()
        if not manager:
            return jsonify({
                "error": "模型管理器未初始化"
            }), 503
        
        # 执行卸载
        loop = get_event_loop()
        success = run_async_in_thread(
            manager.unload_model(model_name), loop
        )
        
        if success:
            return jsonify({
                "message": f"模型卸载成功: {model_name}",
                "model_name": model_name,
                "status": "unloaded"
            })
        else:
            return jsonify({
                "error": f"模型卸载失败: {model_name}"
            }), 500
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"模型卸载失败: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@management_bp.route('/models/<model_name>/status', methods=['GET'])
@api_monitor
def model_status(model_name):
    """
    获取模型状态
    GET /v1/models/{model_name}/status
    """
    try:
        manager = get_model_manager()
        if not manager:
            return jsonify({
                "error": "模型管理器未初始化"
            }), 503
        
        # 获取模型信息
        model_info = manager.get_model_info(model_name)
        
        if not model_info:
            return jsonify({
                "error": f"模型未找到: {model_name}"
            }), 404
        
        return jsonify(model_info)
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"获取模型状态失败: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@management_bp.route('/models/list', methods=['GET'])
@api_monitor
def list_all_models():
    """
    列出所有模型 (包括已注册和已加载)
    GET /v1/models/list
    """
    try:
        manager = get_model_manager()
        if not manager:
            return jsonify({
                "error": "模型管理器未初始化"
            }), 503
        
        # 获取注册的模型
        registered_models = []
        for model_config in manager.list_registered_models():
            registered_models.append({
                "name": model_config.name,
                "path": model_config.path,
                "engine_type": model_config.engine_type,
                "auto_load": model_config.auto_load,
                "priority": model_config.priority,
                "status": "registered"
            })
        
        # 获取已加载的模型
        loaded_models = manager.list_loaded_models()
        
        return jsonify({
            "registered_models": registered_models,
            "loaded_models": loaded_models,
            "total_registered": len(registered_models),
            "total_loaded": len(loaded_models)
        })
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"列出模型失败: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@management_bp.route('/config', methods=['GET'])
@api_monitor
def get_config():
    """
    获取当前配置
    GET /v1/config
    """
    try:
        config = current_app.config['APP_CONFIG']
        
        # 过滤敏感信息
        safe_config = {
            "host": config.host,
            "port": config.port,
            "inference_engine": config.inference_engine,
            "inference_mode": config.inference_mode,
            "max_concurrent_models": config.max_concurrent_models,
            "default_model": config.default_model,
            "device_type": config.device_type,
            "max_gpu_memory": config.max_gpu_memory,
            "max_cpu_threads": config.max_cpu_threads,
            "log_level": config.log_level,
            "enable_caching": config.enable_caching,
            "max_concurrent_requests": config.max_concurrent_requests
        }
        
        return jsonify(safe_config)
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"获取配置失败: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@management_bp.route('/logs/recent', methods=['GET'])
@api_monitor
def get_recent_logs():
    """
    获取最近的日志
    GET /v1/logs/recent
    """
    try:
        # 获取日志文件路径
        config = current_app.config['APP_CONFIG']
        log_file = config.log_file
        
        # 读取最近的日志行数
        limit = request.args.get('limit', 100, type=int)
        limit = min(limit, 1000)  # 最多1000行
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent_lines = lines[-limit:] if len(lines) > limit else lines
                
            return jsonify({
                "logs": [line.strip() for line in recent_lines],
                "total_lines": len(recent_lines),
                "log_file": log_file
            })
            
        except FileNotFoundError:
            return jsonify({
                "logs": [],
                "total_lines": 0,
                "message": "日志文件不存在"
            })
        
    except Exception as e:
        current_app.config['LOGGER'].error(f"获取日志失败: {e}")
        return jsonify({
            "error": str(e)
        }), 500


def register_management_routes(app):
    """注册管理路由"""
    app.register_blueprint(management_bp, url_prefix='/v1')