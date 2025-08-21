#!/usr/bin/env python3
"""
负载均衡应用
支持多实例部署和负载均衡的 Flask 应用
"""

import asyncio
import time
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import threading

from ..utils import get_config, setup_logger, get_logger, api_monitor
from ..utils.cluster_manager import ClusterManager, create_default_cluster
from ..core.load_balancer import LoadBalancer, BalanceStrategy


def create_load_balanced_app(config=None):
    """创建负载均衡应用"""
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
    app.config['CLUSTER_MANAGER'] = None
    app.config['LOAD_BALANCER'] = None
    
    @app.before_first_request
    def initialize_services():
        """初始化负载均衡服务"""
        try:
            logger.info("初始化负载均衡应用服务")
            
            # 创建集群管理器
            cluster_manager = create_default_cluster(app_config)
            app.config['CLUSTER_MANAGER'] = cluster_manager
            
            # 获取负载均衡器
            load_balancer = cluster_manager.load_balancer
            app.config['LOAD_BALANCER'] = load_balancer
            
            # 设置负载均衡策略
            try:
                strategy = BalanceStrategy(app_config.load_balance_strategy)
                load_balancer.set_strategy(strategy)
            except ValueError:
                logger.warning(f"不支持的负载均衡策略: {app_config.load_balance_strategy}, 使用默认策略")
            
            # 启动集群
            if cluster_manager.start_cluster():
                logger.info("负载均衡集群启动成功")
            else:
                logger.error("负载均衡集群启动失败")
                raise RuntimeError("集群启动失败")
            
            logger.info("负载均衡应用服务初始化完成")
            
        except Exception as e:
            logger.error(f"负载均衡服务初始化失败: {e}")
            raise
    
    @app.before_request
    def before_request():
        """请求前处理"""
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
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        """全局异常处理"""
        logger = app.config['LOGGER']
        logger.error(f"请求处理异常: {e}", exc_info=True)
        
        return jsonify({
            "error": {
                "type": "internal_error",
                "message": "服务内部错误"
            }
        }), 500
    
    # 健康检查端点
    @app.route('/health', methods=['GET'])
    def health_check():
        """健康检查"""
        cluster_manager = app.config.get('CLUSTER_MANAGER')
        
        if not cluster_manager:
            return jsonify({"status": "error", "message": "集群管理器未初始化"}), 503
        
        cluster_status = cluster_manager.get_cluster_status()
        
        if cluster_status['running_instances'] > 0:
            return jsonify({
                "status": "healthy",
                "cluster": {
                    "running_instances": cluster_status['running_instances'],
                    "total_instances": cluster_status['total_instances'],
                    "strategy": cluster_status['load_balancer_strategy']
                }
            })
        else:
            return jsonify({
                "status": "unhealthy",
                "message": "没有运行中的实例"
            }), 503
    
    # 注册负载均衡路由
    from ..api.routes import register_load_balanced_routes
    register_load_balanced_routes(app)
    
    # 集群管理接口
    @app.route('/v1/cluster/status', methods=['GET'])
    def cluster_status():
        """获取集群状态"""
        cluster_manager = app.config.get('CLUSTER_MANAGER')
        if not cluster_manager:
            return jsonify({"error": "集群管理器未初始化"}), 503
        
        return jsonify(cluster_manager.get_cluster_status())
    
    @app.route('/v1/cluster/scale', methods=['POST'])
    def cluster_scale():
        """集群扩缩容"""
        cluster_manager = app.config.get('CLUSTER_MANAGER')
        if not cluster_manager:
            return jsonify({"error": "集群管理器未初始化"}), 503
        
        data = request.get_json()
        action = data.get('action')  # scale_up or scale_down
        count = data.get('count', 1)
        
        if action == 'scale_up':
            success = cluster_manager.scale_up(count)
        elif action == 'scale_down':
            success = cluster_manager.scale_down(count)
        else:
            return jsonify({"error": "无效的操作，支持 scale_up 或 scale_down"}), 400
        
        if success:
            return jsonify({"message": f"{action} 操作成功", "count": count})
        else:
            return jsonify({"error": f"{action} 操作失败"}), 500
    
    @app.route('/v1/cluster/strategy', methods=['POST'])
    def set_load_balance_strategy():
        """设置负载均衡策略"""
        cluster_manager = app.config.get('CLUSTER_MANAGER')
        if not cluster_manager:
            return jsonify({"error": "集群管理器未初始化"}), 503
        
        data = request.get_json()
        strategy = data.get('strategy')
        
        if cluster_manager.set_load_balance_strategy(strategy):
            return jsonify({"message": f"负载均衡策略已设置为: {strategy}"})
        else:
            return jsonify({"error": f"设置负载均衡策略失败: {strategy}"}), 400
    
    @app.route('/v1/cluster/restart', methods=['POST'])
    def restart_instance():
        """重启实例"""
        cluster_manager = app.config.get('CLUSTER_MANAGER')
        if not cluster_manager:
            return jsonify({"error": "集群管理器未初始化"}), 503
        
        data = request.get_json()
        instance_id = data.get('instance_id')
        
        if not instance_id:
            return jsonify({"error": "需要提供 instance_id"}), 400
        
        if cluster_manager.restart_instance(instance_id):
            return jsonify({"message": f"实例 {instance_id} 重启成功"})
        else:
            return jsonify({"error": f"实例 {instance_id} 重启失败"}), 500
    
    return app


def get_load_balanced_app():
    """获取负载均衡应用实例"""
    config = get_config()
    return create_load_balanced_app(config)


if __name__ == '__main__':
    # 测试模式
    app = get_load_balanced_app()
    app.run(host='0.0.0.0', port=8000, debug=True)