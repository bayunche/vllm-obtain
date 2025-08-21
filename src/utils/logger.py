"""
日志系统
支持结构化日志、性能监控和本地文件存储
"""

import os
import sys
import time
import json
import functools
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from loguru import logger as loguru_logger
import psutil


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def record_inference(self, model_name: str, prompt_tokens: int, 
                        completion_tokens: int, inference_time: float):
        """记录推理性能指标"""
        total_tokens = prompt_tokens + completion_tokens
        tokens_per_second = completion_tokens / inference_time if inference_time > 0 else 0
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "inference_time": round(inference_time, 3),
            "tokens_per_second": round(tokens_per_second, 2),
            "memory_usage": self._get_memory_usage(),
            "gpu_memory_usage": self._get_gpu_memory_usage()
        }
        
        self.metrics[f"inference_{int(time.time())}"] = metric
        return metric
    
    def record_model_operation(self, operation: str, model_name: str, 
                             duration: float, success: bool, **kwargs):
        """记录模型操作指标"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "model_name": model_name,
            "duration": round(duration, 3),
            "success": success,
            "memory_usage": self._get_memory_usage(),
            **kwargs
        }
        
        self.metrics[f"model_op_{int(time.time())}"] = metric
        return metric
    
    def record_api_request(self, endpoint: str, method: str, status_code: int,
                          response_time: float, **kwargs):
        """记录API请求指标"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": round(response_time, 3),
            "memory_usage": self._get_memory_usage(),
            **kwargs
        }
        
        self.metrics[f"api_{int(time.time())}"] = metric
        return metric
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统性能指标"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime": round(time.time() - self.start_time, 2),
            "cpu_usage": cpu_percent,
            "memory_total": memory.total,
            "memory_used": memory.used,
            "memory_percent": memory.percent,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_percent": round((disk.used / disk.total) * 100, 2)
        }
        
        # 添加GPU指标 (如果可用)
        gpu_metrics = self._get_gpu_metrics()
        if gpu_metrics:
            metrics.update(gpu_metrics)
        
        return metrics
    
    def _get_memory_usage(self) -> str:
        """获取当前进程内存使用"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f}MB"
    
    def _get_gpu_memory_usage(self) -> Optional[str]:
        """获取GPU内存使用 (CUDA)"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                cached = torch.cuda.memory_reserved() / 1024**3  # GB
                return f"{allocated:.1f}GB/{cached:.1f}GB"
        except ImportError:
            pass
        return None
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """获取详细GPU指标"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                gpu_metrics = {
                    "gpu_count": device_count,
                    "gpu_devices": []
                }
                
                for i in range(device_count):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    
                    gpu_metrics["gpu_devices"].append({
                        "device_id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated": round(allocated, 2),
                        "memory_cached": round(cached, 2),
                        "memory_total": round(total, 2),
                        "memory_percent": round((allocated / total) * 100, 1)
                    })
                
                return gpu_metrics
        except ImportError:
            pass
        return None


class InferenceLogger:
    """推理服务专用日志器"""
    
    def __init__(self, config=None):
        self.config = config
        self.metrics = PerformanceMetrics()
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志器"""
        # 移除默认handler
        loguru_logger.remove()
        
        # 获取配置
        if self.config:
            log_level = self.config.log_level
            log_file = self.config.log_file
            rotation = self.config.log_rotation
            retention = self.config.log_retention
        else:
            log_level = os.getenv("LOG_LEVEL", "INFO")
            log_file = os.getenv("LOG_FILE", "./logs/inference.log")
            rotation = os.getenv("LOG_ROTATION", "100MB")
            retention = os.getenv("LOG_RETENTION", "7 days")
        
        # 确保日志目录存在
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 控制台输出 (彩色格式)
        loguru_logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # 文件输出 (JSON格式，便于解析)
        loguru_logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            serialize=True  # JSON格式
        )
        
        # 性能指标专用日志
        metrics_file = str(Path(log_file).parent / "metrics.log")
        loguru_logger.add(
            metrics_file,
            level="INFO",
            format="{message}",
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            filter=lambda record: "METRICS" in record["extra"]
        )
        
        self.logger = loguru_logger
        self.logger.info("日志系统初始化完成")
    
    def log_inference(self, model_name: str, prompt_tokens: int,
                     completion_tokens: int, inference_time: float,
                     request_id: Optional[str] = None, **kwargs):
        """记录推理日志"""
        metric = self.metrics.record_inference(
            model_name, prompt_tokens, completion_tokens, inference_time
        )
        
        log_data = {
            "event": "model_inference",
            "request_id": request_id,
            **metric,
            **kwargs
        }
        
        self.logger.bind(METRICS=True).info(json.dumps(log_data, ensure_ascii=False))
        
        self.logger.info(
            f"推理完成 | 模型: {model_name} | "
            f"Token: {prompt_tokens}+{completion_tokens}={prompt_tokens + completion_tokens} | "
            f"耗时: {inference_time:.2f}s | "
            f"速度: {metric['tokens_per_second']:.1f} tokens/s"
        )
    
    def log_model_operation(self, operation: str, model_name: str,
                           duration: float, success: bool, **kwargs):
        """记录模型操作日志"""
        metric = self.metrics.record_model_operation(
            operation, model_name, duration, success, **kwargs
        )
        
        log_data = {
            "event": "model_operation",
            **metric,
            **kwargs
        }
        
        self.logger.bind(METRICS=True).info(json.dumps(log_data, ensure_ascii=False))
        
        status = "成功" if success else "失败"
        self.logger.info(
            f"模型操作 | {operation} | 模型: {model_name} | "
            f"耗时: {duration:.2f}s | 状态: {status}"
        )
    
    def log_api_request(self, endpoint: str, method: str, status_code: int,
                       response_time: float, **kwargs):
        """记录API请求日志"""
        metric = self.metrics.record_api_request(
            endpoint, method, status_code, response_time, **kwargs
        )
        
        log_data = {
            "event": "api_request",
            **metric,
            **kwargs
        }
        
        self.logger.bind(METRICS=True).info(json.dumps(log_data, ensure_ascii=False))
        
        self.logger.info(
            f"API请求 | {method} {endpoint} | "
            f"状态: {status_code} | 耗时: {response_time:.3f}s"
        )
    
    def log_system_metrics(self):
        """记录系统指标"""
        metrics = self.metrics.get_system_metrics()
        
        log_data = {
            "event": "system_metrics",
            **metrics
        }
        
        self.logger.bind(METRICS=True).info(json.dumps(log_data, ensure_ascii=False))
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, **kwargs)


def performance_monitor(operation_name: str = None):
    """性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 记录成功的操作
                get_logger().log_model_operation(
                    operation=operation,
                    model_name=kwargs.get('model_name', 'unknown'),
                    duration=duration,
                    success=True
                )
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                
                # 记录失败的操作
                get_logger().log_model_operation(
                    operation=operation,
                    model_name=kwargs.get('model_name', 'unknown'),
                    duration=duration,
                    success=False,
                    error=str(e)
                )
                
                get_logger().error(f"操作失败: {operation} | 错误: {e}")
                raise
        
        return wrapper
    return decorator


def api_monitor(func: Callable) -> Callable:
    """API请求监控装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            # 尝试从Flask request获取信息
            try:
                from flask import request
                endpoint = request.endpoint or func.__name__
                method = request.method
            except:
                endpoint = func.__name__
                method = 'UNKNOWN'
            
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            
            # 假设成功返回的状态码为200
            status_code = getattr(result, 'status_code', 200)
            
            get_logger().log_api_request(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time=response_time
            )
            
            return result
        
        except Exception as e:
            response_time = time.time() - start_time
            
            get_logger().log_api_request(
                endpoint=endpoint if 'endpoint' in locals() else func.__name__,
                method=method if 'method' in locals() else 'UNKNOWN',
                status_code=500,
                response_time=response_time,
                error=str(e)
            )
            
            raise
    
    return wrapper


# 全局日志实例
_global_logger: Optional[InferenceLogger] = None


def setup_logger(config=None) -> InferenceLogger:
    """设置全局日志器"""
    global _global_logger
    _global_logger = InferenceLogger(config)
    return _global_logger


def get_logger() -> InferenceLogger:
    """获取全局日志器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = InferenceLogger()
    return _global_logger


if __name__ == "__main__":
    # 测试日志系统
    print("=== 日志系统测试 ===")
    
    logger_instance = setup_logger()
    
    # 测试基本日志
    logger_instance.info("测试信息日志")
    logger_instance.warning("测试警告日志")
    logger_instance.error("测试错误日志")
    
    # 测试推理日志
    logger_instance.log_inference(
        model_name="test-model",
        prompt_tokens=100,
        completion_tokens=50,
        inference_time=1.5,
        request_id="test-123"
    )
    
    # 测试模型操作日志
    logger_instance.log_model_operation(
        operation="load_model",
        model_name="test-model",
        duration=30.5,
        success=True
    )
    
    # 测试API请求日志
    logger_instance.log_api_request(
        endpoint="/v1/chat/completions",
        method="POST",
        status_code=200,
        response_time=2.1
    )
    
    # 测试系统指标
    logger_instance.log_system_metrics()
    
    # 测试装饰器
    @performance_monitor("test_function")
    def test_function(model_name="test"):
        time.sleep(0.1)
        return "success"
    
    test_function()
    
    print("✅ 日志系统测试完成，请检查 logs/ 目录下的日志文件")