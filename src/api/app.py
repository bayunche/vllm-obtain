"""
Flask 应用主体
提供 Web API 服务入口
"""

import asyncio
import time
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import threading

from ..utils import get_config, setup_logger, get_logger, api_monitor
from ..core.model_manager import get_model_manager


def create_app(config=None, **kwargs):
    """创建 Flask 应用"""
    app = Flask(__name__)
    
    # 加载配置
    app_config = config or get_config()
    
    # 设置 Flask 配置
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['JSON_AS_ASCII'] = False
    app.config['JSON_SORT_KEYS'] = False
    
    # 启用 CORS
    CORS(app, origins="*", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    
    # 设置日志
    logger = setup_logger(app_config)
    
    # 存储配置和服务
    app.config['APP_CONFIG'] = app_config
    app.config['LOGGER'] = logger
    app.config['MODEL_MANAGER'] = None
    
    # 创建事件循环
    app.config['EVENT_LOOP'] = None
    app.config['LOOP_THREAD'] = None
    
    # 用于追踪是否已初始化
    _initialized = [False]
    
    def initialize_services():
        """初始化服务"""
        if _initialized[0]:
            return
        _initialized[0] = True
        
        try:
            logger.info("初始化 Flask 应用服务")
            
            # 创建新的事件循环在后台线程中运行
            def run_event_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                app.config['EVENT_LOOP'] = loop
                
                # 初始化模型管理器
                async def init_manager():
                    manager = await get_model_manager()
                    app.config['MODEL_MANAGER'] = manager
                    logger.info("模型管理器初始化完成")
                
                loop.run_until_complete(init_manager())
                loop.run_forever()
            
            # 启动后台线程
            thread = threading.Thread(target=run_event_loop, daemon=True)
            thread.start()
            app.config['LOOP_THREAD'] = thread
            
            # 等待初始化完成
            time.sleep(2)
            
            logger.info("Flask 应用服务初始化完成")
            
        except Exception as e:
            logger.error(f"服务初始化失败: {e}")
            raise
    
    @app.before_request
    def before_request():
        """请求前处理"""
        # 确保服务已初始化（只在第一次请求时执行）
        initialize_services()
        
        g.start_time = time.time()
        g.request_id = f"req_{int(time.time() * 1000)}_{id(request)}"
        
        # 记录请求日志
        logger = app.config['LOGGER']
        logger.debug(f"收到请求: {request.method} {request.path} | ID: {g.request_id}")
    
    @app.after_request
    def after_request(response):
        """请求后处理"""
        logger = app.config['LOGGER']
        response_time = time.time() - g.start_time
        
        # 记录响应日志
        logger.log_api_request(
            endpoint=request.endpoint or request.path,
            method=request.method,
            status_code=response.status_code,
            response_time=response_time,
            request_id=g.request_id
        )
        
        return response
    
    @app.errorhandler(404)
    def not_found(error):
        """404 错误处理"""
        return jsonify({
            "error": {
                "message": f"未找到端点: {request.path}",
                "type": "not_found_error",
                "code": "not_found"
            }
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """500 错误处理"""
        logger = app.config['LOGGER']
        logger.error(f"内部服务器错误: {error}")
        
        return jsonify({
            "error": {
                "message": "内部服务器错误",
                "type": "server_error", 
                "code": "internal_error"
            }
        }), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """通用异常处理"""
        logger = app.config['LOGGER']
        logger.error(f"未处理的异常: {error}")
        
        return jsonify({
            "error": {
                "message": str(error),
                "type": "server_error",
                "code": "unhandled_exception"
            }
        }), 500
    
    # 注册路由
    from .routes.openai_complete import register_openai_v1_routes
    from .routes.management import management_bp
    
    register_openai_v1_routes(app)
    app.register_blueprint(management_bp, url_prefix='/v1')
    
    # 健康检查端点
    @app.route('/health')
    @api_monitor
    def health_check():
        """健康检查"""
        try:
            manager = app.config.get('MODEL_MANAGER')
            if not manager:
                return jsonify({
                    "status": "unhealthy",
                    "message": "模型管理器未初始化",
                    "timestamp": time.time()
                }), 503
            
            # 异步调用健康检查
            loop = app.config['EVENT_LOOP']
            if loop and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    manager.health_check(), loop
                )
                health_result = future.result(timeout=5)
            else:
                health_result = {"status": "unhealthy", "message": "事件循环未运行"}
            
            status_code = 200 if health_result["status"] == "healthy" else 503
            return jsonify(health_result), status_code
            
        except Exception as e:
            logger = app.config['LOGGER']
            logger.error(f"健康检查失败: {e}")
            return jsonify({
                "status": "unhealthy",
                "message": str(e),
                "timestamp": time.time()
            }), 503
    
    # 根路径信息
    @app.route('/')
    def index():
        """根路径信息"""
        return jsonify({
            "name": "VLLM 跨平台推理服务",
            "version": "1.0.0",
            "description": "OpenAI API 兼容的大语言模型推理服务",
            "docs": {
                "openai_compatible": "/v1/models",
                "management": "/v1/system/status", 
                "health": "/health"
            },
            "supported_engines": ["vllm", "mlx", "llama_cpp"],
            "timestamp": time.time()
        })
    
    return app


def run_async_in_thread(coro, loop):
    """在事件循环线程中运行协程"""
    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        # 使用配置的请求超时时间，而不是硬编码的30秒
        from ..utils import get_config
        config = get_config()
        timeout = config.request_timeout if hasattr(config, 'request_timeout') else 300
        return future.result(timeout=timeout)
    else:
        raise RuntimeError("事件循环未运行")


# 全局应用实例
app = None


def get_app(*args, **kwargs):
    """获取应用实例"""
    global app
    if app is None:
        # 如果只有一个参数且是配置对象，使用它
        config = args[0] if len(args) == 1 and not isinstance(args[0], str) else None
        app = create_app(config, **kwargs)
    return app

# 创建WSGI应用实例
application = get_app()


if __name__ == '__main__':
    # 开发模式启动
    import os
    
    app = create_app()
    config = app.config['APP_CONFIG']
    
    print("🚀 启动 VLLM 推理服务")
    print(f"📍 服务地址: http://{config.host}:{config.port}")
    print(f"🔧 推理引擎: {config.inference_engine}")
    print(f"📚 OpenAI 兼容接口: http://{config.host}:{config.port}/v1/")
    print(f"🏥 健康检查: http://{config.host}:{config.port}/health")
    
    # 开发环境启动
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
        threaded=True
    )