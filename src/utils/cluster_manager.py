#!/usr/bin/env python3
"""
集群管理器
管理多个推理服务实例的启动、停止和配置
"""

import os
import sys
import time
import signal
import subprocess
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import psutil

from src.utils.logger import get_logger
from src.utils.config import Config
from src.core.load_balancer import LoadBalancer, BalanceStrategy

logger = get_logger(__name__)


@dataclass
class InstanceConfig:
    """实例配置"""
    instance_id: str
    port: int
    weight: int = 1
    models: List[str] = None
    env_vars: Dict[str, str] = None
    process: subprocess.Popen = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = []
        if self.env_vars is None:
            self.env_vars = {}


class ClusterManager:
    """集群管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.instances: Dict[str, InstanceConfig] = {}
        self.load_balancer = LoadBalancer(BalanceStrategy.ROUND_ROBIN)
        self.base_port = config.port
        self.is_running = False
        self.monitor_thread = None
        
        # 支持的负载均衡策略
        self.available_strategies = list(BalanceStrategy)
        
        logger.info("集群管理器初始化完成")
    
    def add_instance(self, instance_id: str, port: int = None, weight: int = 1, 
                    models: List[str] = None, env_vars: Dict[str, str] = None):
        """添加实例配置"""
        if port is None:
            port = self.base_port + len(self.instances)
        
        if models is None:
            models = []
        
        if env_vars is None:
            env_vars = {}
        
        instance_config = InstanceConfig(
            instance_id=instance_id,
            port=port,
            weight=weight,
            models=models,
            env_vars=env_vars
        )
        
        self.instances[instance_id] = instance_config
        logger.info(f"添加实例配置: {instance_id} (端口: {port}, 权重: {weight})")
    
    def remove_instance(self, instance_id: str):
        """移除实例配置"""
        if instance_id in self.instances:
            # 先停止实例
            self.stop_instance(instance_id)
            del self.instances[instance_id]
            self.load_balancer.remove_instance(instance_id)
            logger.info(f"移除实例: {instance_id}")
    
    def start_instance(self, instance_id: str) -> bool:
        """启动单个实例"""
        if instance_id not in self.instances:
            logger.error(f"实例配置不存在: {instance_id}")
            return False
        
        instance_config = self.instances[instance_id]
        
        if instance_config.process and instance_config.process.poll() is None:
            logger.warning(f"实例 {instance_id} 已在运行")
            return True
        
        try:
            # 准备环境变量
            env = os.environ.copy()
            env.update({
                'PORT': str(instance_config.port),
                'HOST': '127.0.0.1',
                'INSTANCE_ID': instance_id,
                **instance_config.env_vars
            })
            
            # 启动实例进程
            cmd = [sys.executable, 'run.py', '--mode', 'prod', '--port', str(instance_config.port)]
            
            logger.info(f"启动实例 {instance_id}: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            instance_config.process = process
            
            # 等待实例启动
            if self._wait_for_instance_ready(instance_id, timeout=60):
                # 添加到负载均衡器
                self.load_balancer.add_instance(
                    instance_id,
                    '127.0.0.1',
                    instance_config.port,
                    instance_config.weight,
                    instance_config.models
                )
                logger.info(f"实例 {instance_id} 启动成功")
                return True
            else:
                logger.error(f"实例 {instance_id} 启动超时")
                self.stop_instance(instance_id)
                return False
                
        except Exception as e:
            logger.error(f"启动实例 {instance_id} 失败: {e}")
            return False
    
    def stop_instance(self, instance_id: str) -> bool:
        """停止单个实例"""
        if instance_id not in self.instances:
            logger.error(f"实例配置不存在: {instance_id}")
            return False
        
        instance_config = self.instances[instance_id]
        
        if not instance_config.process:
            logger.warning(f"实例 {instance_id} 未运行")
            return True
        
        try:
            process = instance_config.process
            
            # 尝试优雅关闭
            if process.poll() is None:
                logger.info(f"正在停止实例 {instance_id}...")
                process.terminate()
                
                # 等待进程结束
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"实例 {instance_id} 未能优雅关闭，强制终止")
                    process.kill()
                    process.wait()
            
            instance_config.process = None
            self.load_balancer.remove_instance(instance_id)
            logger.info(f"实例 {instance_id} 已停止")
            return True
            
        except Exception as e:
            logger.error(f"停止实例 {instance_id} 失败: {e}")
            return False
    
    def start_cluster(self) -> bool:
        """启动整个集群"""
        if self.is_running:
            logger.warning("集群已在运行")
            return True
        
        logger.info("启动集群...")
        
        success_count = 0
        for instance_id in self.instances:
            if self.start_instance(instance_id):
                success_count += 1
        
        if success_count > 0:
            self.is_running = True
            
            # 启动监控线程
            self.monitor_thread = threading.Thread(target=self._monitor_instances, daemon=True)
            self.monitor_thread.start()
            
            # 启动负载均衡器健康检查
            import asyncio
            asyncio.create_task(self.load_balancer.start_health_check())
            
            logger.info(f"集群启动完成，成功启动 {success_count}/{len(self.instances)} 个实例")
            return True
        else:
            logger.error("集群启动失败，没有实例成功启动")
            return False
    
    def stop_cluster(self):
        """停止整个集群"""
        if not self.is_running:
            logger.warning("集群未运行")
            return
        
        logger.info("停止集群...")
        
        self.is_running = False
        
        # 停止负载均衡器健康检查
        import asyncio
        asyncio.create_task(self.load_balancer.stop_health_check())
        
        # 停止所有实例
        for instance_id in list(self.instances.keys()):
            self.stop_instance(instance_id)
        
        logger.info("集群已停止")
    
    def restart_instance(self, instance_id: str) -> bool:
        """重启实例"""
        logger.info(f"重启实例 {instance_id}")
        self.stop_instance(instance_id)
        time.sleep(2)  # 等待资源释放
        return self.start_instance(instance_id)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        status = {
            "is_running": self.is_running,
            "total_instances": len(self.instances),
            "running_instances": 0,
            "load_balancer_strategy": self.load_balancer.strategy.value,
            "instances": {},
            "load_balancer_stats": self.load_balancer.get_instance_stats()
        }
        
        for instance_id, config in self.instances.items():
            is_running = config.process and config.process.poll() is None
            if is_running:
                status["running_instances"] += 1
            
            instance_status = {
                "port": config.port,
                "weight": config.weight,
                "models": config.models,
                "is_running": is_running,
                "pid": config.process.pid if config.process else None
            }
            
            # 添加进程资源使用情况
            if is_running and config.process:
                try:
                    proc = psutil.Process(config.process.pid)
                    instance_status.update({
                        "cpu_percent": proc.cpu_percent(),
                        "memory_mb": proc.memory_info().rss / 1024 / 1024,
                        "memory_percent": proc.memory_percent()
                    })
                except psutil.NoSuchProcess:
                    instance_status["is_running"] = False
            
            status["instances"][instance_id] = instance_status
        
        return status
    
    def set_load_balance_strategy(self, strategy: str) -> bool:
        """设置负载均衡策略"""
        try:
            balance_strategy = BalanceStrategy(strategy)
            self.load_balancer.set_strategy(balance_strategy)
            logger.info(f"负载均衡策略已切换为: {strategy}")
            return True
        except ValueError:
            logger.error(f"不支持的负载均衡策略: {strategy}")
            return False
    
    def _wait_for_instance_ready(self, instance_id: str, timeout: int = 60) -> bool:
        """等待实例就绪"""
        import requests
        
        instance_config = self.instances[instance_id]
        url = f"http://127.0.0.1:{instance_config.port}/health"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        return False
    
    def _monitor_instances(self):
        """监控实例状态"""
        logger.info("启动实例监控线程")
        
        while self.is_running:
            try:
                for instance_id, config in self.instances.items():
                    if config.process and config.process.poll() is not None:
                        # 进程已退出
                        logger.warning(f"检测到实例 {instance_id} 意外退出，尝试重启")
                        self.restart_instance(instance_id)
                
                time.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                logger.error(f"实例监控异常: {e}")
                time.sleep(5)
        
        logger.info("实例监控线程退出")
    
    def scale_up(self, count: int = 1) -> bool:
        """扩容：增加实例"""
        logger.info(f"扩容集群，增加 {count} 个实例")
        
        success_count = 0
        for i in range(count):
            instance_id = f"auto_instance_{int(time.time())}_{i}"
            port = self.base_port + len(self.instances)
            
            self.add_instance(instance_id, port)
            
            if self.is_running and self.start_instance(instance_id):
                success_count += 1
        
        logger.info(f"扩容完成，成功增加 {success_count}/{count} 个实例")
        return success_count > 0
    
    def scale_down(self, count: int = 1) -> bool:
        """缩容：减少实例"""
        if len(self.instances) <= count:
            logger.warning("无法缩容，实例数量不足")
            return False
        
        logger.info(f"缩容集群，减少 {count} 个实例")
        
        # 选择最后添加的实例进行删除
        instances_to_remove = list(self.instances.keys())[-count:]
        
        for instance_id in instances_to_remove:
            self.remove_instance(instance_id)
        
        logger.info(f"缩容完成，减少了 {len(instances_to_remove)} 个实例")
        return True


def create_default_cluster(config: Config) -> ClusterManager:
    """创建默认集群配置"""
    cluster_manager = ClusterManager(config)
    
    # 根据配置决定实例数量
    instance_count = getattr(config, 'cluster_instances', 1)
    
    if instance_count > 1:
        logger.info(f"创建 {instance_count} 个实例的集群")
        
        for i in range(instance_count):
            instance_id = f"instance_{i}"
            port = config.port + i
            weight = 1
            
            # 可以根据需要为不同实例配置不同的模型
            models = []  # 默认支持所有模型
            env_vars = {}
            
            cluster_manager.add_instance(instance_id, port, weight, models, env_vars)
    else:
        # 单实例模式
        cluster_manager.add_instance("main_instance", config.port)
    
    return cluster_manager