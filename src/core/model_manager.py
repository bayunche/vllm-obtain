"""
模型管理器
负责模型生命周期管理、资源监控和并发控制
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

from .inference_engine import InferenceEngine, EngineConfig, ModelInfo, ModelStatus
from .inference_engine import InferenceRequest, InferenceResponse
from .exceptions import ModelLoadError, ModelNotFoundError, ResourceLimitError
from ..engines import create_engine
from ..utils import get_logger, get_config, get_optimal_engine, performance_monitor


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    path: str
    engine_type: Optional[str] = None
    auto_load: bool = False
    priority: int = 1  # 优先级，数字越小优先级越高
    max_context_length: Optional[int] = None
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class ModelManager:
    """模型管理器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger()
        
        # 推理引擎实例
        self.engines: Dict[str, InferenceEngine] = {}
        
        # 模型配置注册表
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # 当前加载的模型统计
        self.loaded_models: Dict[str, str] = {}  # model_name -> engine_type
        
        # 并发控制
        self._model_locks: Dict[str, asyncio.Lock] = {}
        self._operation_lock = asyncio.Lock()
        
        # 资源监控
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5分钟
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """初始化模型管理器"""
        try:
            self.logger.info("初始化模型管理器")
            
            # 确定推理引擎类型
            engine_type = self.config.inference_engine
            if engine_type == "auto":
                engine_type = get_optimal_engine()
            
            self.logger.info(f"尝试使用推理引擎: {engine_type}")
            
            # 创建引擎配置
            engine_config = EngineConfig(
                engine_type=engine_type,
                device_type=self.config.device_type,
                max_gpu_memory=self.config.max_gpu_memory,
                max_cpu_threads=self.config.max_cpu_threads,
                max_sequence_length=self.config.max_concurrent_models,
                enable_streaming=True
            )
            
            # 创建并初始化主引擎，支持自动回退
            main_engine = None
            engines_to_try = [engine_type]
            
            # 添加回退选项
            if engine_type != 'llama_cpp':
                engines_to_try.append('llama_cpp')  # 总是回退到 llama.cpp
            
            for try_engine in engines_to_try:
                try:
                    self.logger.info(f"尝试初始化引擎: {try_engine}")
                    engine_config.engine_type = try_engine
                    main_engine = create_engine(try_engine, engine_config)
                    success = await main_engine.initialize()
                    
                    if success:
                        self.logger.info(f"成功初始化引擎: {try_engine}")
                        engine_type = try_engine
                        break
                    else:
                        self.logger.warning(f"引擎初始化失败: {try_engine}")
                except Exception as e:
                    self.logger.warning(f"创建引擎失败 {try_engine}: {e}")
                    continue
            
            if main_engine is None or not success:
                self.logger.error("所有引擎都初始化失败")
                return False
            
            self.engines[engine_type] = main_engine
            
            # 注册默认模型
            if self.config.default_model:
                await self._register_default_model()
            
            # 启动后台任务
            asyncio.create_task(self._background_monitor())
            
            self._initialized = True
            self.logger.info("模型管理器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"模型管理器初始化失败: {e}")
            return False
    
    async def _register_default_model(self):
        """注册默认模型"""
        default_model = self.config.default_model
        
        # 优先使用 model_dir 如果它指向具体的模型路径
        if hasattr(self.config, 'model_dir') and self.config.model_dir:
            model_path = Path(self.config.model_dir)
            # 检查是否是具体的模型目录（包含config.json）
            if (model_path / 'config.json').exists():
                self.register_model(
                    ModelConfig(
                        name=default_model,
                        path=str(model_path),
                        auto_load=True,
                        priority=1
                    )
                )
                self.logger.info(f"已注册默认模型: {default_model} (路径: {model_path})")
                return
        
        # 否则使用 model_base_path / default_model
        model_path = Path(self.config.model_base_path) / default_model
        
        if model_path.exists():
            self.register_model(
                ModelConfig(
                    name=default_model,
                    path=str(model_path),
                    auto_load=True,
                    priority=1
                )
            )
            self.logger.info(f"已注册默认模型: {default_model}")
        else:
            self.logger.warning(f"默认模型路径不存在: {model_path}")
    
    def register_model(self, model_config: ModelConfig):
        """注册模型配置"""
        self.model_configs[model_config.name] = model_config
        self._model_locks[model_config.name] = asyncio.Lock()
        
        self.logger.info(f"已注册模型: {model_config.name} -> {model_config.path}")
    
    def unregister_model(self, model_name: str):
        """取消注册模型"""
        if model_name in self.model_configs:
            del self.model_configs[model_name]
        if model_name in self._model_locks:
            del self._model_locks[model_name]
        
        self.logger.info(f"已取消注册模型: {model_name}")
    
    @performance_monitor("load_model")
    async def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            force_reload: 是否强制重新加载
            
        Returns:
            是否加载成功
        """
        if not self._initialized:
            raise RuntimeError("模型管理器未初始化")
        
        # 检查模型是否已注册
        if model_name not in self.model_configs:
            raise ModelNotFoundError(model_name)
        
        # 并发控制
        async with self._model_locks[model_name]:
            # 检查是否已加载
            if not force_reload and self.is_model_loaded(model_name):
                self.logger.info(f"模型已加载: {model_name}")
                return True
            
            # 检查并发数量限制
            if not await self._check_concurrent_limit():
                raise ResourceLimitError(
                    "并发模型数量", 
                    len(self.loaded_models), 
                    self.config.max_concurrent_models
                )
            
            model_config = self.model_configs[model_name]
            
            # 确定使用的引擎
            engine_type = model_config.engine_type or list(self.engines.keys())[0]
            
            if engine_type not in self.engines:
                # 创建新引擎
                await self._create_engine_if_needed(engine_type)
            
            engine = self.engines[engine_type]
            
            try:
                # 执行加载
                success = await engine.load_model(
                    model_name,
                    model_config.path,
                    **model_config.extra_params
                )
                
                if success:
                    self.loaded_models[model_name] = engine_type
                    self.logger.info(f"模型加载成功: {model_name} (引擎: {engine_type})")
                    return True
                else:
                    self.logger.error(f"模型加载失败: {model_name}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"模型加载异常: {model_name} - {e}")
                raise ModelLoadError(model_name, str(e))
    
    @performance_monitor("unload_model")
    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if not self.is_model_loaded(model_name):
            self.logger.warning(f"模型未加载: {model_name}")
            return False
        
        async with self._model_locks[model_name]:
            engine_type = self.loaded_models[model_name]
            engine = self.engines[engine_type]
            
            try:
                success = await engine.unload_model(model_name)
                
                if success:
                    del self.loaded_models[model_name]
                    self.logger.info(f"模型卸载成功: {model_name}")
                    return True
                else:
                    self.logger.error(f"模型卸载失败: {model_name}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"模型卸载异常: {model_name} - {e}")
                return False
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理"""
        model_name = request.model_name
        
        # 检查模型是否已加载
        if not self.is_model_loaded(model_name):
            # 尝试自动加载
            if model_name in self.model_configs:
                self.logger.info(f"自动加载模型: {model_name}")
                await self.load_model(model_name)
            else:
                raise ModelNotFoundError(model_name)
        
        # 获取对应的引擎
        engine_type = self.loaded_models[model_name]
        engine = self.engines[engine_type]
        
        # 执行推理
        return await engine.generate(request)
    
    async def inference_stream(self, request: InferenceRequest):
        """流式推理"""
        model_name = request.model_name
        
        # 检查模型是否已加载
        if not self.is_model_loaded(model_name):
            # 尝试自动加载
            if model_name in self.model_configs:
                self.logger.info(f"自动加载模型: {model_name}")
                await self.load_model(model_name)
            else:
                raise ModelNotFoundError(model_name)
        
        # 获取对应的引擎
        engine_type = self.loaded_models[model_name]
        engine = self.engines[engine_type]
        
        # 执行流式推理
        async for chunk in engine.generate_stream(request):
            yield chunk
    
    def is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        return model_name in self.loaded_models
    
    def list_registered_models(self) -> List[ModelConfig]:
        """列出所有注册的模型"""
        return list(self.model_configs.values())
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """列出所有已加载的模型"""
        loaded_info = []
        
        for model_name, engine_type in self.loaded_models.items():
            engine = self.engines[engine_type]
            model_info = engine.get_model_info(model_name)
            
            if model_info:
                loaded_info.append({
                    "name": model_name,
                    "engine_type": engine_type,
                    "status": model_info.status.value,
                    "loaded_at": model_info.loaded_at,
                    "memory_usage": model_info.memory_usage,
                    "context_length": model_info.context_length
                })
        
        return loaded_info
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型详细信息"""
        if model_name not in self.loaded_models:
            return None
        
        engine_type = self.loaded_models[model_name]
        engine = self.engines[engine_type]
        model_info = engine.get_model_info(model_name)
        
        if model_info:
            return {
                "name": model_name,
                "engine_type": engine_type,
                "path": model_info.path,
                "status": model_info.status.value,
                "size_mb": model_info.size_mb,
                "parameters": model_info.parameters,
                "context_length": model_info.context_length,
                "loaded_at": model_info.loaded_at,
                "memory_usage": model_info.memory_usage,
                "error_message": model_info.error_message
            }
        
        return None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        loaded_models = self.list_loaded_models()
        
        status = {
            "initialized": self._initialized,
            "total_engines": len(self.engines),
            "registered_models": len(self.model_configs),
            "loaded_models": len(self.loaded_models),
            "max_concurrent_models": self.config.max_concurrent_models,
            "models": loaded_models,
            "engines": {}
        }
        
        # 获取各引擎状态
        for engine_type, engine in self.engines.items():
            status["engines"][engine_type] = engine.get_engine_status()
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "manager_initialized": self._initialized,
            "engines": {},
            "models": len(self.loaded_models),
            "issues": []
        }
        
        # 检查各引擎健康状态
        for engine_type, engine in self.engines.items():
            engine_health = await engine.health_check()
            health_status["engines"][engine_type] = engine_health
            
            if engine_health["status"] != "healthy":
                health_status["issues"].append(f"引擎 {engine_type} 状态异常")
        
        # 检查是否有加载失败的模型
        for model_name, engine_type in self.loaded_models.items():
            engine = self.engines[engine_type]
            model_info = engine.get_model_info(model_name)
            if model_info and model_info.status == ModelStatus.ERROR:
                health_status["issues"].append(f"模型 {model_name} 状态错误")
        
        # 设置总体状态
        if health_status["issues"]:
            health_status["status"] = "degraded"
        
        return health_status
    
    async def _check_concurrent_limit(self) -> bool:
        """检查并发数量限制"""
        current_count = len(self.loaded_models)
        max_count = self.config.max_concurrent_models
        
        if current_count >= max_count:
            # 尝试清理未使用的模型
            await self._cleanup_unused_models()
            current_count = len(self.loaded_models)
        
        return current_count < max_count
    
    async def _cleanup_unused_models(self):
        """清理未使用的模型"""
        self.logger.info("开始清理未使用的模型")
        
        # 简单策略：卸载最早加载的模型
        # 实际应用中可以基于使用频率、优先级等更复杂的策略
        
        models_to_unload = []
        for model_name, engine_type in self.loaded_models.items():
            engine = self.engines[engine_type]
            model_info = engine.get_model_info(model_name)
            
            if model_info and model_info.loaded_at:
                models_to_unload.append((model_name, model_info.loaded_at))
        
        # 按加载时间排序，优先卸载最早的
        models_to_unload.sort(key=lambda x: x[1])
        
        # 卸载最早的一个模型
        if models_to_unload:
            model_to_unload = models_to_unload[0][0]
            await self.unload_model(model_to_unload)
            self.logger.info(f"已清理模型: {model_to_unload}")
    
    async def _create_engine_if_needed(self, engine_type: str):
        """按需创建引擎"""
        if engine_type not in self.engines:
            self.logger.info(f"创建新引擎: {engine_type}")
            
            engine_config = EngineConfig(
                engine_type=engine_type,
                device_type=self.config.device_type,
                max_gpu_memory=self.config.max_gpu_memory,
                max_cpu_threads=self.config.max_cpu_threads
            )
            
            engine = create_engine(engine_type, engine_config)
            success = await engine.initialize()
            
            if success:
                self.engines[engine_type] = engine
                self.logger.info(f"引擎创建成功: {engine_type}")
            else:
                raise RuntimeError(f"引擎创建失败: {engine_type}")
    
    async def _background_monitor(self):
        """后台监控任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                # 定期清理
                current_time = time.time()
                if current_time - self._last_cleanup > self._cleanup_interval:
                    await self._periodic_cleanup()
                    self._last_cleanup = current_time
                
                # 记录系统指标
                self.logger.log_system_metrics()
                
            except Exception as e:
                self.logger.error(f"后台监控任务异常: {e}")
    
    async def _periodic_cleanup(self):
        """定期清理"""
        self.logger.debug("执行定期清理")
        
        # 可以在这里添加更多清理逻辑
        # 例如：清理缓存、检查模型健康状态等
        pass
    
    async def cleanup(self):
        """清理所有资源"""
        self.logger.info("开始清理模型管理器资源")
        
        # 卸载所有模型
        model_names = list(self.loaded_models.keys())
        for model_name in model_names:
            try:
                await self.unload_model(model_name)
            except Exception as e:
                self.logger.error(f"清理模型失败 {model_name}: {e}")
        
        # 清理所有引擎
        for engine_type, engine in self.engines.items():
            try:
                await engine.cleanup()
                self.logger.info(f"引擎清理完成: {engine_type}")
            except Exception as e:
                self.logger.error(f"引擎清理失败 {engine_type}: {e}")
        
        self.engines.clear()
        self.loaded_models.clear()
        self._initialized = False
        
        self.logger.info("模型管理器资源清理完成")


# 全局模型管理器实例
_global_model_manager: Optional[ModelManager] = None


async def get_model_manager() -> ModelManager:
    """获取全局模型管理器"""
    global _global_model_manager
    
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
        await _global_model_manager.initialize()
    
    return _global_model_manager


if __name__ == "__main__":
    # 测试模型管理器
    import asyncio
    
    async def test_model_manager():
        print("=== 模型管理器测试 ===")
        
        manager = ModelManager()
        
        # 初始化
        success = await manager.initialize()
        if not success:
            print("❌ 模型管理器初始化失败")
            return
        
        print("✅ 模型管理器初始化成功")
        
        # 注册测试模型
        test_config = ModelConfig(
            name="test-model",
            path="/path/to/test/model",
            auto_load=True
        )
        manager.register_model(test_config)
        print("✅ 模型注册完成")
        
        # 获取系统状态
        status = await manager.get_system_status()
        print(f"✅ 系统状态: {status}")
        
        # 健康检查
        health = await manager.health_check()
        print(f"✅ 健康检查: {health}")
        
        await manager.cleanup()
        print("✅ 资源清理完成")
    
    asyncio.run(test_model_manager())