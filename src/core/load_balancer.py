#!/usr/bin/env python3
"""
负载均衡器
支持多种负载均衡策略：轮询、最少连接、加权分配
"""

import time
import asyncio
import threading
import aiohttp
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import random

from src.utils.logger import get_logger
from src.core.exceptions import InferenceError

logger = get_logger()


class BalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"        # 轮询
    LEAST_CONNECTIONS = "least_connections"  # 最少连接
    WEIGHTED = "weighted"              # 加权
    RANDOM = "random"                  # 随机
    RESPONSE_TIME = "response_time"    # 响应时间优先


@dataclass
class InstanceInfo:
    """实例信息"""
    instance_id: str
    host: str
    port: int
    weight: int = 1
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = field(default_factory=time.time)
    is_healthy: bool = True
    models: List[str] = field(default_factory=list)
    
    @property
    def url(self) -> str:
        """获取实例 URL"""
        return f"http://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    def update_stats(self, response_time: float, success: bool = True):
        """更新统计信息"""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        
        # 更新平均响应时间（使用移动平均）
        alpha = 0.1  # 平滑因子
        self.avg_response_time = (1 - alpha) * self.avg_response_time + alpha * response_time


class LoadBalanceStrategy(ABC):
    """负载均衡策略抽象基类"""
    
    @abstractmethod
    def select_instance(self, instances: List[InstanceInfo], **kwargs) -> Optional[InstanceInfo]:
        """选择实例"""
        pass


class RoundRobinStrategy(LoadBalanceStrategy):
    """轮询策略"""
    
    def __init__(self):
        self.current_index = 0
        self.lock = threading.Lock()
    
    def select_instance(self, instances: List[InstanceInfo], **kwargs) -> Optional[InstanceInfo]:
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            return None
        
        with self.lock:
            instance = healthy_instances[self.current_index % len(healthy_instances)]
            self.current_index += 1
            return instance


class LeastConnectionsStrategy(LoadBalanceStrategy):
    """最少连接策略"""
    
    def select_instance(self, instances: List[InstanceInfo], **kwargs) -> Optional[InstanceInfo]:
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            return None
        
        return min(healthy_instances, key=lambda x: x.active_connections)


class WeightedStrategy(LoadBalanceStrategy):
    """加权策略"""
    
    def select_instance(self, instances: List[InstanceInfo], **kwargs) -> Optional[InstanceInfo]:
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            return None
        
        # 根据权重和当前负载计算得分
        best_instance = None
        best_score = float('inf')
        
        for instance in healthy_instances:
            # 得分越低越好：连接数/权重
            score = instance.active_connections / max(instance.weight, 1)
            if score < best_score:
                best_score = score
                best_instance = instance
        
        return best_instance


class RandomStrategy(LoadBalanceStrategy):
    """随机策略"""
    
    def select_instance(self, instances: List[InstanceInfo], **kwargs) -> Optional[InstanceInfo]:
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            return None
        
        return random.choice(healthy_instances)


class ResponseTimeStrategy(LoadBalanceStrategy):
    """响应时间优先策略"""
    
    def select_instance(self, instances: List[InstanceInfo], **kwargs) -> Optional[InstanceInfo]:
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            return None
        
        # 选择平均响应时间最短的实例
        return min(healthy_instances, key=lambda x: x.avg_response_time)


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, strategy: BalanceStrategy = BalanceStrategy.ROUND_ROBIN):
        self.instances: Dict[str, InstanceInfo] = {}
        self.strategy = strategy
        self._strategy_impl = self._create_strategy(strategy)
        self.health_check_interval = 30  # 健康检查间隔（秒）
        self.health_check_timeout = 5    # 健康检查超时（秒）
        self._health_check_task = None
        self._running = False
        
        logger.info(f"负载均衡器初始化，策略: {strategy.value}")
    
    def _create_strategy(self, strategy: BalanceStrategy) -> LoadBalanceStrategy:
        """创建策略实现"""
        strategy_map = {
            BalanceStrategy.ROUND_ROBIN: RoundRobinStrategy,
            BalanceStrategy.LEAST_CONNECTIONS: LeastConnectionsStrategy,
            BalanceStrategy.WEIGHTED: WeightedStrategy,
            BalanceStrategy.RANDOM: RandomStrategy,
            BalanceStrategy.RESPONSE_TIME: ResponseTimeStrategy,
        }
        
        strategy_class = strategy_map.get(strategy)
        if not strategy_class:
            raise ValueError(f"不支持的负载均衡策略: {strategy}")
        
        return strategy_class()
    
    def add_instance(self, instance_id: str, host: str, port: int, weight: int = 1, models: List[str] = None):
        """添加实例"""
        if models is None:
            models = []
        
        instance = InstanceInfo(
            instance_id=instance_id,
            host=host,
            port=port,
            weight=weight,
            models=models
        )
        
        self.instances[instance_id] = instance
        logger.info(f"添加实例: {instance_id} ({host}:{port}), 权重: {weight}")
    
    def remove_instance(self, instance_id: str):
        """移除实例"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            logger.info(f"移除实例: {instance_id}")
    
    def get_instance(self, model_name: str = None) -> Optional[InstanceInfo]:
        """获取可用实例"""
        available_instances = list(self.instances.values())
        
        # 如果指定了模型，过滤支持该模型的实例
        if model_name:
            available_instances = [
                inst for inst in available_instances 
                if not inst.models or model_name in inst.models
            ]
        
        if not available_instances:
            logger.warning(f"没有可用实例" + (f"支持模型 {model_name}" if model_name else ""))
            return None
        
        # 使用策略选择实例
        selected = self._strategy_impl.select_instance(available_instances)
        
        if selected:
            logger.debug(f"选择实例: {selected.instance_id} ({selected.host}:{selected.port})")
        
        return selected
    
    def mark_request_start(self, instance_id: str):
        """标记请求开始"""
        if instance_id in self.instances:
            self.instances[instance_id].active_connections += 1
    
    def mark_request_end(self, instance_id: str, response_time: float, success: bool = True):
        """标记请求结束"""
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            instance.active_connections = max(0, instance.active_connections - 1)
            instance.update_stats(response_time, success)
    
    def get_instance_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取实例统计信息"""
        stats = {}
        for instance_id, instance in self.instances.items():
            stats[instance_id] = {
                "url": instance.url,
                "weight": instance.weight,
                "active_connections": instance.active_connections,
                "total_requests": instance.total_requests,
                "failed_requests": instance.failed_requests,
                "success_rate": instance.success_rate,
                "avg_response_time": instance.avg_response_time,
                "is_healthy": instance.is_healthy,
                "models": instance.models,
                "last_health_check": instance.last_health_check
            }
        return stats
    
    async def start_health_check(self):
        """启动健康检查"""
        if self._running:
            return
        
        self._running = True
        logger.info("启动负载均衡器健康检查")
        
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"健康检查异常: {e}")
                await asyncio.sleep(5)  # 出错时短暂等待
    
    async def stop_health_check(self):
        """停止健康检查"""
        self._running = False
        logger.info("停止负载均衡器健康检查")
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        import aiohttp
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)) as session:
            tasks = []
            for instance in self.instances.values():
                task = self._check_instance_health(session, instance)
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_instance_health(self, session: aiohttp.ClientSession, instance: InstanceInfo):
        """检查单个实例健康状态"""
        try:
            start_time = time.time()
            async with session.get(f"{instance.url}/health") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    instance.is_healthy = True
                    instance.last_health_check = time.time()
                    # 更新响应时间（健康检查的响应时间）
                    alpha = 0.05  # 健康检查用更小的权重
                    instance.avg_response_time = (1 - alpha) * instance.avg_response_time + alpha * response_time
                    logger.debug(f"实例 {instance.instance_id} 健康检查通过")
                else:
                    instance.is_healthy = False
                    logger.warning(f"实例 {instance.instance_id} 健康检查失败: HTTP {response.status}")
        
        except Exception as e:
            instance.is_healthy = False
            logger.warning(f"实例 {instance.instance_id} 健康检查异常: {e}")
    
    def set_strategy(self, strategy: BalanceStrategy):
        """切换负载均衡策略"""
        self.strategy = strategy
        self._strategy_impl = self._create_strategy(strategy)
        logger.info(f"切换负载均衡策略为: {strategy.value}")


class LoadBalanceMiddleware:
    """负载均衡中间件"""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.logger = get_logger(__name__)
    
    async def dispatch_request(self, request_data: Dict[str, Any], model_name: str = None) -> Dict[str, Any]:
        """分发请求到负载均衡的实例"""
        instance = self.load_balancer.get_instance(model_name)
        
        if not instance:
            raise InferenceError("没有可用的推理实例")
        
        # 标记请求开始
        self.load_balancer.mark_request_start(instance.instance_id)
        
        start_time = time.time()
        success = False
        
        try:
            # 这里应该实际发送请求到选定的实例
            # 由于我们在同一个进程中，这里只是模拟
            # 在实际实现中，这里会使用 HTTP 客户端发送请求
            
            response = await self._send_request_to_instance(instance, request_data)
            success = True
            return response
            
        except Exception as e:
            self.logger.error(f"请求实例 {instance.instance_id} 失败: {e}")
            raise
        
        finally:
            # 标记请求结束
            response_time = time.time() - start_time
            self.load_balancer.mark_request_end(instance.instance_id, response_time, success)
    
    async def _send_request_to_instance(self, instance: InstanceInfo, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求到指定实例"""
        # 这是一个示例实现，实际中需要根据具体的 API 格式来实现
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # 根据请求类型确定端点
            endpoint = "/v1/chat/completions"  # 默认聊天补全
            if "prompt" in request_data:
                endpoint = "/v1/completions"
            
            url = f"{instance.url}{endpoint}"
            
            async with session.post(url, json=request_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise InferenceError(f"实例返回错误: {response.status} - {error_text}")
                
                return await response.json()